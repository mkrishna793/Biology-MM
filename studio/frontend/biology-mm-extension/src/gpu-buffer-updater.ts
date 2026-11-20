/**
 * Zero-allocation GPU buffer & Mol* ParticleList updater for TriumStream v1.
 *
 * Pre-allocates persistent TypedArrays for up to `capacity` (default 1,000,000) atoms
 * so live simulation frames update coordinates, radii (GCMC tombstoning), and scalar
 * physics attributes in-place without per-frame heap array allocations.
 */

import { Column } from '@molstar/core/data/db';
import { Cell } from '@molstar/core/math/geometry/spacegroup/cell';
import { Vec3 } from '@molstar/core/math/linear-algebra';
import type { ModelFormat } from '@molstar/model/formats/format';
import { CustomProperties } from '@molstar/model/model/custom-property';
import {
  Particle,
  ParticleValues,
  type ParticleAttribute,
  type ParticleList,
  type ParticleTargetInfo,
} from '@molstar/model/model/particles/particle-list';
import { GcmcSlotPool } from './gcmc-slot-pool.js';
import {
  decodeInt16Fractional,
  FrameFlags,
  HEADER_BYTES,
  PHYSICS_CHANNELS,
  type FrameFullMessage,
  type FrameTilePatchMessage,
  type GcmcSlotDeltaMessage,
  type PhysicsChannelName,
  type TopologyInitMessage,
  type TriumHeader,
} from './protocol.js';

export interface ScalarRange {
  min: number;
  max: number;
}

function cellFromBoxH(boxH: Float32Array): Cell | undefined {
  const ax = boxH[0], ay = boxH[3], az = boxH[6];
  const bx = boxH[1], by = boxH[4], bz = boxH[7];
  const cx = boxH[2], cy = boxH[5], cz = boxH[8];

  const la = Math.hypot(ax, ay, az);
  const lb = Math.hypot(bx, by, bz);
  const lc = Math.hypot(cx, cy, cz);
  if (la <= 1e-6 || lb <= 1e-6 || lc <= 1e-6) return undefined;

  const clampCos = (v: number) => Math.max(-0.999999, Math.min(0.999999, v));
  const alpha = Math.acos(clampCos((bx * cx + by * cy + bz * cz) / (lb * lc)));
  const beta = Math.acos(clampCos((ax * cx + ay * cy + az * cz) / (la * lc)));
  const gamma = Math.acos(clampCos((ax * bx + ay * by + az * bz) / (la * lb)));

  return Cell.create(Vec3.create(la, lb, lc), Vec3.create(alpha, beta, gamma));
}

export class GpuParticleStreamUpdater {
  readonly capacity: number;
  readonly coords: Float32Array;
  readonly boxH = new Float32Array(9);
  readonly slotPool: GcmcSlotPool;
  readonly scalarChannels = new Map<PhysicsChannelName, Float32Array>();
  readonly scalarRanges = new Map<PhysicsChannelName, ScalarRange>();

  private readonly attributes = new Map<string, ParticleAttribute>();
  private readonly targetOffsets = new Int32Array(2);
  private readonly targetValues = Int32Array.of(0);
  private readonly targetInfo: ReadonlyMap<number, ParticleTargetInfo> = new Map([[0, { name: 'simulation' }]]);
  private readonly sourceData: ModelFormat = { kind: 'biology-mm-stream', name: 'TriumStream', data: {} };

  private _count = 0;
  private _frameId = 0;
  private _simTimePs = 0;
  private _reversed = false;
  private _activeChannel: PhysicsChannelName = 'force-magnitude';
  private _topologyDirty = true;
  private _currentList: ParticleList | undefined;
  private _bonds: Uint32Array = new Uint32Array(0);
  private _residueIds: Uint32Array = new Uint32Array(0);
  private _tileIds: Uint32Array = new Uint32Array(0);
  private _hasCell = false;

  constructor(capacity = 1_000_000) {
    if (!Number.isInteger(capacity) || capacity <= 0) {
      throw new RangeError(`GpuParticleStreamUpdater capacity must be a positive integer, got ${capacity}`);
    }
    this.capacity = capacity;
    this.coords = new Float32Array(capacity * 3);
    this.slotPool = new GcmcSlotPool(capacity);
  }

  get count(): number {
    return this._count;
  }

  get activeCount(): number {
    return this.slotPool.activeCount;
  }

  get radii(): Float32Array {
    return this.slotPool.radii;
  }

  get frameId(): number {
    return this._frameId;
  }

  get simTimePs(): number {
    return this._simTimePs;
  }

  get reversed(): boolean {
    return this._reversed;
  }

  get activeChannel(): PhysicsChannelName {
    return this._activeChannel;
  }

  get bonds(): Uint32Array {
    return this._bonds;
  }

  get residueIds(): Uint32Array {
    return this._residueIds;
  }

  get tileIds(): Uint32Array {
    return this._tileIds;
  }

  getScalarRange(channel: PhysicsChannelName = this._activeChannel): ScalarRange {
    return this.scalarRanges.get(channel) ?? { min: 0, max: 1 };
  }

  getOrCreateChannelBuffer(channel: PhysicsChannelName): Float32Array {
    let buf = this.scalarChannels.get(channel);
    if (!buf) {
      buf = new Float32Array(this.capacity);
      this.scalarChannels.set(channel, buf);
    }
    return buf;
  }

  getAttribute(channel: PhysicsChannelName): ParticleAttribute | undefined {
    return this.attributes.get(channel);
  }

  setActiveChannel(channel: PhysicsChannelName): void {
    this._activeChannel = channel;
    this.ensureAttribute(channel);
  }

  private ensureAttribute(channel: PhysicsChannelName): void {
    const buf = this.getOrCreateChannelBuffer(channel);
    const range = this.scalarRanges.get(channel) ?? { min: 0, max: 1 };
    const view = buf.subarray(0, this._count);
    const existing = this.attributes.get(channel);
    if (existing && existing.column.rowCount === this._count) {
      this.attributes.set(channel, {
        column: existing.column,
        label: channel,
        min: range.min,
        max: range.max,
      });
    } else {
      this.attributes.set(channel, {
        column: Column.ofFloatArray(view),
        label: channel,
        min: range.min,
        max: range.max,
      });
    }
  }

  applyTopologyInit(msg: TopologyInitMessage): void {
    const n = Math.min(msg.header.nAtoms, this.capacity);
    this._count = n;
    this._frameId = msg.header.frameId;
    this._simTimePs = msg.header.simTimePs;
    this._reversed = false;

    this.boxH.set(msg.boxH.subarray(0, 9), 0);
    this._hasCell = true;

    this.slotPool.initFromTopology(n, msg.atomicNumbers, msg.radii);
    if (msg.coords) {
      const len = Math.min(n * 3, msg.coords.length);
      this.coords.set(msg.coords.subarray(0, len), 0);
    }

    this._bonds = msg.bonds;
    this._residueIds = msg.residueIds;
    this._tileIds = msg.tileIds;

    this.targetOffsets[0] = 0;
    this.targetOffsets[1] = n;
    this.attributes.clear();
    this.ensureAttribute(this._activeChannel);

    this._topologyDirty = true;
    this._currentList = undefined;
  }

  applyFrameFull(msg: FrameFullMessage): void {
    const n = Math.min(msg.header.nAtoms, this.capacity);
    if (n !== this._count) {
      this._count = n;
      this.targetOffsets[1] = n;
      this._topologyDirty = true;
    }

    this._frameId = msg.header.frameId;
    this._simTimePs = msg.header.simTimePs;
    this._reversed = msg.reversed;

    if (msg.boxH) {
      this.boxH.set(msg.boxH.subarray(0, 9), 0);
      this._hasCell = true;
    }

    const coordLen = Math.min(n * 3, msg.coords.length);
    this.coords.set(msg.coords.subarray(0, coordLen), 0);

    if (msg.scalars) {
      const channel = PHYSICS_CHANNELS[msg.scalarChannel] ?? this._activeChannel;
      this._activeChannel = channel;
      const buf = this.getOrCreateChannelBuffer(channel);
      const scalarLen = Math.min(n, msg.scalars.length);
      buf.set(msg.scalars.subarray(0, scalarLen), 0);
      this.scalarRanges.set(channel, { min: msg.header.scalarMin, max: msg.header.scalarMax });
      this.ensureAttribute(channel);
    }
  }

  /**
   * Zero-allocation binary decode for `MsgType.FRAME_FULL` directly into persistent buffers.
   * Uses `decodeInt16Fractional(..., this.coords)` when `INT16_FRACTIONAL` flag is set.
   */
  applyFrameFullBinary(buffer: ArrayBuffer, header: TriumHeader): void {
    const n = Math.min(header.nAtoms, this.capacity);
    if (n !== this._count) {
      this._count = n;
      this.targetOffsets[1] = n;
      this._topologyDirty = true;
    }

    this._frameId = header.frameId;
    this._simTimePs = header.simTimePs;
    this._reversed = (header.flags & FrameFlags.REVERSED) !== 0;

    let off = HEADER_BYTES;
    const dv = new DataView(buffer);

    if (header.flags & FrameFlags.HAS_BOX) {
      if (off % 4 === 0) {
        this.boxH.set(new Float32Array(buffer, off, 9), 0);
      } else {
        for (let i = 0; i < 9; i++) this.boxH[i] = dv.getFloat32(off + i * 4, true);
      }
      this._hasCell = true;
      off += 36;
    }

    if (header.flags & FrameFlags.INT16_FRACTIONAL) {
      if (off % 2 === 0) {
        const q16 = new Int16Array(buffer, off, n * 3);
        decodeInt16Fractional(q16, this.boxH, this.coords);
      } else {
        // Unaligned fallback without allocating n*3 array
        const scale = 1.0 / (2.0 * 32767.0);
        const h00 = this.boxH[0], h01 = this.boxH[1], h02 = this.boxH[2];
        const h10 = this.boxH[3], h11 = this.boxH[4], h12 = this.boxH[5];
        const h20 = this.boxH[6], h21 = this.boxH[7], h22 = this.boxH[8];
        for (let i = 0; i < n; i++) {
          const j = i * 3;
          const bOff = off + j * 2;
          const fx = dv.getInt16(bOff, true) * scale + 0.5;
          const fy = dv.getInt16(bOff + 2, true) * scale + 0.5;
          const fz = dv.getInt16(bOff + 4, true) * scale + 0.5;
          this.coords[j] = h00 * fx + h01 * fy + h02 * fz;
          this.coords[j + 1] = h10 * fx + h11 * fy + h12 * fz;
          this.coords[j + 2] = h20 * fx + h21 * fy + h22 * fz;
        }
      }
      off += n * 6;
    } else {
      if (off % 4 === 0) {
        this.coords.set(new Float32Array(buffer, off, n * 3), 0);
      } else {
        for (let i = 0; i < n * 3; i++) this.coords[i] = dv.getFloat32(off + i * 4, true);
      }
      off += n * 12;
    }

    if (header.flags & FrameFlags.HAS_SCALAR) {
      const channel = PHYSICS_CHANNELS[header.reserved] ?? this._activeChannel;
      this._activeChannel = channel;
      const buf = this.getOrCreateChannelBuffer(channel);
      if (off % 4 === 0) {
        buf.set(new Float32Array(buffer, off, n), 0);
      } else {
        for (let i = 0; i < n; i++) buf[i] = dv.getFloat32(off + i * 4, true);
      }
      this.scalarRanges.set(channel, { min: header.scalarMin, max: header.scalarMax });
      this.ensureAttribute(channel);
    }
  }

  applyTilePatch(msg: FrameTilePatchMessage): void {
    this._frameId = msg.header.frameId;
    this._simTimePs = msg.header.simTimePs;
    this.boxH.set(msg.boxH.subarray(0, 9), 0);
    this._hasCell = true;

    const indices = msg.atomIndices;
    const srcCoords = msg.coords;
    const dstCoords = this.coords;
    const count = indices.length;

    for (let i = 0; i < count; i++) {
      const slot = indices[i];
      if (slot >= this.capacity) continue;
      const s3 = i * 3;
      const d3 = slot * 3;
      dstCoords[d3] = srcCoords[s3];
      dstCoords[d3 + 1] = srcCoords[s3 + 1];
      dstCoords[d3 + 2] = srcCoords[s3 + 2];
    }

    if (msg.scalars) {
      const channel = PHYSICS_CHANNELS[msg.scalarChannel] ?? this._activeChannel;
      this._activeChannel = channel;
      const buf = this.getOrCreateChannelBuffer(channel);
      const srcScalars = msg.scalars;
      for (let i = 0; i < count; i++) {
        const slot = indices[i];
        if (slot < this.capacity) {
          buf[slot] = srcScalars[i];
        }
      }
      this.scalarRanges.set(channel, { min: msg.header.scalarMin, max: msg.header.scalarMax });
      this.ensureAttribute(channel);
    }
  }

  applyGcmcDelta(msg: GcmcSlotDeltaMessage): void {
    this._frameId = msg.header.frameId;
    this._simTimePs = msg.header.simTimePs;

    const res = this.slotPool.applyDelta(msg.insertedSlots, msg.deletedSlots, {
      atomicNumbers: msg.insertedAtomicNumbers,
      coords: this.coords,
      insertedCoords: msg.insertedCoords,
    });

    if (res.maxActiveSlot > this._count) {
      this._count = Math.min(res.maxActiveSlot, this.capacity);
      this.targetOffsets[1] = this._count;
    }
    this._topologyDirty = true;
  }

  /**
   * Create or update the Mol* `ParticleList` backed by our persistent typed-array views.
   * Uses `Particle.withPoses` when topology/count did not change so Mol* reuses GPU buffers.
   */
  getParticleList(forceNew = false): ParticleList {
    const coordView = this.coords.subarray(0, this._count * 3);

    if (!forceNew && !this._topologyDirty && this._currentList && this._currentList.count === this._count) {
      this._currentList = Particle.withPoses(this._currentList, coordView);
      return this._currentList;
    }

    const radiiView = this.slotPool.radii.subarray(0, this._count);
    const cell = this._hasCell ? cellFromBoxH(this.boxH) : undefined;
    this.ensureAttribute(this._activeChannel);

    const list: ParticleList = {
      entryId: 'biology-mm-live',
      label: 'Biology-MM Live Stream',
      count: this._count,
      cell,
      targets: ParticleValues.ofRuns(this.targetOffsets, this.targetValues),
      targetInfo: this.targetInfo,
      coordinates: coordView,
      radii: radiiView,
      attributes: this.attributes,
      getParticleLabel: (index: number) => {
        const z = index >= 0 && index < this._count ? this.slotPool.atomicNumbers[index] : 0;
        const active = this.slotPool.isActive(index);
        const val = this.scalarChannels.get(this._activeChannel)?.[index];
        const valStr = val !== undefined ? ` · ${this._activeChannel} ${val.toFixed(3)}` : '';
        return `Atom ${index} (Z=${z}${active ? '' : ', inactive'})${valStr}`;
      },
      sourceData: this.sourceData,
      customProperties: new CustomProperties(),
      _propertyData: Object.create(null),
    };

    this._currentList = list;
    this._topologyDirty = false;
    return list;
  }
}
