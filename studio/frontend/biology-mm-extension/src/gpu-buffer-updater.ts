/**
 * Zero-allocation GPU buffer & Mol* ParticleList updater for TriumStream v1.
 *
 * Pre-allocates persistent TypedArrays for up to `capacity` (default 1,000,000) atoms
 * so live simulation frames (both world-space non-PBC Float32 frames and sparse
 * `FRAME_TILE_PATCH` deltas) update coordinates, radii (GCMC tombstoning), color
 * staging, and scalar physics attributes in-place with dirty-range tracking
 * (`minDirtyIdx`..`maxDirtyIdx`) and zero per-frame heap buffer reallocations.
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

export interface DirtyIndexRange {
  readonly minDirtyIdx: number;
  readonly maxDirtyIdx: number;
  readonly count: number;
  readonly isDirty: boolean;
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
  private readonly scalarViews = new Map<PhysicsChannelName, { count: number; view: Float32Array }>();
  private readonly targetOffsets = new Int32Array(2);
  private readonly targetValues = Int32Array.of(0);
  private readonly targetInfo: ReadonlyMap<number, ParticleTargetInfo> = new Map([[0, { name: 'simulation' }]]);
  private readonly sourceData: ModelFormat = { kind: 'biology-mm-stream', name: 'TriumStream', data: {} };

  private _colorStaging: Float32Array | undefined;
  private _coordView: Float32Array | undefined;
  private _coordViewCount = -1;
  private _radiiView: Float32Array | undefined;
  private _radiiViewCount = -1;

  private _count = 0;
  private _frameId = 0;
  private _simTimePs = 0;
  private _reversed = false;
  private _pbcWrapped = false;
  private _activeChannel: PhysicsChannelName = 'force-magnitude';
  private _topologyDirty = true;
  private _scalarsDirty = false;
  private _currentList: ParticleList | undefined;
  private _bonds: Uint32Array = new Uint32Array(0);
  private _residueIds: Uint32Array = new Uint32Array(0);
  private _tileIds: Uint32Array = new Uint32Array(0);
  private _hasCell = false;
  private _lastDirtyAtomCount = 0;
  private _lastPacketBytes = 0;

  private _minDirtyIdx = Infinity;
  private _maxDirtyIdx = -1;
  private _dirtyVersion = 0;

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

  /** Lazily allocated, persistent RGB Float32 staging buffer reused across frames without GC churn. */
  get colorStaging(): Float32Array {
    if (!this._colorStaging) {
      this._colorStaging = new Float32Array(this.capacity * 3);
    }
    return this._colorStaging;
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

  get pbcWrapped(): boolean {
    return this._pbcWrapped;
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

  get lastDirtyAtomCount(): number {
    return this._lastDirtyAtomCount;
  }

  get lastPacketBytes(): number {
    return this._lastPacketBytes;
  }

  get minDirtyIdx(): number {
    return this._maxDirtyIdx >= this._minDirtyIdx ? this._minDirtyIdx : -1;
  }

  get maxDirtyIdx(): number {
    return this._maxDirtyIdx;
  }

  get hasDirtyRange(): boolean {
    return this._maxDirtyIdx >= this._minDirtyIdx && this._minDirtyIdx >= 0;
  }

  get dirtyVersion(): number {
    return this._dirtyVersion;
  }

  get scalarsDirty(): boolean {
    return this._scalarsDirty;
  }

  getDirtyRange(): DirtyIndexRange {
    if (!this.hasDirtyRange) {
      return { minDirtyIdx: -1, maxDirtyIdx: -1, count: 0, isDirty: false };
    }
    return {
      minDirtyIdx: this._minDirtyIdx,
      maxDirtyIdx: this._maxDirtyIdx,
      count: this._maxDirtyIdx - this._minDirtyIdx + 1,
      isDirty: true,
    };
  }

  /**
   * Consume and clear the accumulated dirty index range (`minDirtyIdx`..`maxDirtyIdx`)
   * once a coalesced `requestAnimationFrame` WebGL upload commits.
   */
  consumeDirtyRange(): DirtyIndexRange {
    const range = this.getDirtyRange();
    this._minDirtyIdx = Infinity;
    this._maxDirtyIdx = -1;
    this._scalarsDirty = false;
    return range;
  }

  private markDirtyIndex(slot: number): void {
    if (slot < this._minDirtyIdx) this._minDirtyIdx = slot;
    if (slot > this._maxDirtyIdx) this._maxDirtyIdx = slot;
  }

  private markDirtyRange(minIdx: number, maxIdx: number): void {
    if (maxIdx < minIdx || minIdx < 0) return;
    if (minIdx < this._minDirtyIdx) this._minDirtyIdx = minIdx;
    if (maxIdx > this._maxDirtyIdx) this._maxDirtyIdx = maxIdx;
  }

  private setScalarRangeInPlace(channel: PhysicsChannelName, min: number, max: number): void {
    const existing = this.scalarRanges.get(channel);
    if (existing) {
      existing.min = min;
      existing.max = max;
    } else {
      this.scalarRanges.set(channel, { min, max });
    }
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

  getScalarStagingView(channel: PhysicsChannelName = this._activeChannel): Float32Array {
    const buf = this.getOrCreateChannelBuffer(channel);
    const cached = this.scalarViews.get(channel);
    if (cached && cached.count === this._count) {
      return cached.view;
    }
    const view = buf.subarray(0, this._count);
    this.scalarViews.set(channel, { count: this._count, view });
    return view;
  }

  getPositionStagingView(): Float32Array {
    if (!this._coordView || this._coordViewCount !== this._count) {
      this._coordView = this.coords.subarray(0, this._count * 3);
      this._coordViewCount = this._count;
    }
    return this._coordView;
  }

  getAttribute(channel: PhysicsChannelName): ParticleAttribute | undefined {
    return this.attributes.get(channel);
  }

  setActiveChannel(channel: PhysicsChannelName): void {
    this._activeChannel = channel;
    this.ensureAttribute(channel);
  }

  private ensureAttribute(channel: PhysicsChannelName): void {
    const range = this.scalarRanges.get(channel) ?? { min: 0, max: 1 };
    const existing = this.attributes.get(channel);
    if (existing && existing.column.rowCount === this._count) {
      if (existing.min === range.min && existing.max === range.max) {
        return;
      }
      this.attributes.set(channel, {
        column: existing.column,
        label: channel,
        min: range.min,
        max: range.max,
      });
    } else {
      const view = this.getScalarStagingView(channel);
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
    this._pbcWrapped = (msg.header.flags & FrameFlags.PBC_WRAPPED) !== 0;

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
    this._coordViewCount = -1;
    this._radiiViewCount = -1;
    this.scalarViews.clear();
    this.attributes.clear();
    this.ensureAttribute(this._activeChannel);

    this._lastDirtyAtomCount = n;
    this._topologyDirty = true;
    this._currentList = undefined;
    if (n > 0) {
      this.markDirtyRange(0, n - 1);
      this._dirtyVersion++;
    }
  }

  applyFrameFull(msg: FrameFullMessage): void {
    const n = Math.min(msg.header.nAtoms, this.capacity);
    if (n !== this._count) {
      this._count = n;
      this.targetOffsets[1] = n;
      this._coordViewCount = -1;
      this._radiiViewCount = -1;
      this._topologyDirty = true;
    }

    this._frameId = msg.header.frameId;
    this._simTimePs = msg.header.simTimePs;
    this._reversed = msg.reversed;
    this._pbcWrapped = (msg.header.flags & FrameFlags.PBC_WRAPPED) !== 0;

    if (msg.boxH && msg.boxH.length >= 9) {
      this.boxH.set(msg.boxH.subarray(0, 9), 0);
      this._hasCell = true;
    }

    const coordLen = Math.min(n * 3, msg.coords.length);
    if (coordLen > 0) {
      this.coords.set(msg.coords.subarray(0, coordLen), 0);
    }
    this._lastDirtyAtomCount = n;
    if (n > 0) {
      this.markDirtyRange(0, n - 1);
      this._dirtyVersion++;
    }

    if (msg.scalars) {
      const channel = PHYSICS_CHANNELS[msg.scalarChannel] ?? this._activeChannel;
      this._activeChannel = channel;
      const buf = this.getOrCreateChannelBuffer(channel);
      const scalarLen = Math.min(n, msg.scalars.length);
      if (scalarLen > 0) {
        buf.set(msg.scalars.subarray(0, scalarLen), 0);
      }
      this.setScalarRangeInPlace(channel, msg.header.scalarMin, msg.header.scalarMax);
      this.ensureAttribute(channel);
      this._scalarsDirty = true;
    }
  }

  /**
   * Zero-allocation binary decode for `MsgType.FRAME_FULL` directly into persistent buffers.
   * Supports both world-space non-PBC-wrapped Float32 frames and `INT16_FRACTIONAL` frames
   * without intermediate heap allocations.
   */
  applyFrameFullBinary(buffer: ArrayBuffer, header: TriumHeader): void {
    const n = Math.min(header.nAtoms, this.capacity);
    if (n !== this._count) {
      this._count = n;
      this.targetOffsets[1] = n;
      this._coordViewCount = -1;
      this._radiiViewCount = -1;
      this._topologyDirty = true;
    }

    this._frameId = header.frameId;
    this._simTimePs = header.simTimePs;
    this._reversed = (header.flags & FrameFlags.REVERSED) !== 0;
    this._pbcWrapped = (header.flags & FrameFlags.PBC_WRAPPED) !== 0;
    this._lastDirtyAtomCount = n;
    this._lastPacketBytes = buffer.byteLength;

    let off = HEADER_BYTES;
    const byteLen = buffer.byteLength;
    const dv = new DataView(buffer);

    if ((header.flags & FrameFlags.HAS_BOX) && off + 36 <= byteLen) {
      if (off % 4 === 0) {
        this.boxH.set(new Float32Array(buffer, off, 9), 0);
      } else {
        for (let i = 0; i < 9; i++) this.boxH[i] = dv.getFloat32(off + i * 4, true);
      }
      this._hasCell = true;
      off += 36;
    }

    if (header.flags & FrameFlags.INT16_FRACTIONAL) {
      const coordAtoms = Math.min(n, Math.floor(Math.max(0, byteLen - off) / 6));
      if (coordAtoms > 0) {
        if (off % 2 === 0) {
          const q16 = new Int16Array(buffer, off, coordAtoms * 3);
          decodeInt16Fractional(q16, this.boxH, this.coords);
        } else {
          const scale = 1.0 / (2.0 * 32767.0);
          const h00 = this.boxH[0], h01 = this.boxH[1], h02 = this.boxH[2];
          const h10 = this.boxH[3], h11 = this.boxH[4], h12 = this.boxH[5];
          const h20 = this.boxH[6], h21 = this.boxH[7], h22 = this.boxH[8];
          for (let i = 0; i < coordAtoms; i++) {
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
        this.markDirtyRange(0, coordAtoms - 1);
        this._dirtyVersion++;
      }
      off += header.nAtoms * 6;
    } else {
      const coordAtoms = Math.min(n, Math.floor(Math.max(0, byteLen - off) / 12));
      if (coordAtoms > 0) {
        if (off % 4 === 0) {
          this.coords.set(new Float32Array(buffer, off, coordAtoms * 3), 0);
        } else {
          const totalFloats = coordAtoms * 3;
          for (let i = 0; i < totalFloats; i++) this.coords[i] = dv.getFloat32(off + i * 4, true);
        }
        this.markDirtyRange(0, coordAtoms - 1);
        this._dirtyVersion++;
      }
      off += header.nAtoms * 12;
    }

    if ((header.flags & FrameFlags.HAS_SCALAR) && off < byteLen) {
      const scalarAtoms = Math.min(n, Math.floor((byteLen - off) / 4));
      const channel = PHYSICS_CHANNELS[header.reserved] ?? this._activeChannel;
      this._activeChannel = channel;
      const buf = this.getOrCreateChannelBuffer(channel);
      if (scalarAtoms > 0) {
        if (off % 4 === 0) {
          buf.set(new Float32Array(buffer, off, scalarAtoms), 0);
        } else {
          for (let i = 0; i < scalarAtoms; i++) buf[i] = dv.getFloat32(off + i * 4, true);
        }
      }
      this.setScalarRangeInPlace(channel, header.scalarMin, header.scalarMax);
      this.ensureAttribute(channel);
      this._scalarsDirty = true;
    }
  }

  /**
   * Zero-allocation binary decode and sparse in-place scatter for `MsgType.FRAME_TILE_PATCH`.
   * Updates ONLY the dirty `atomIndices` slots, marks strictly `minDirtyIdx`..`maxDirtyIdx`,
   * and supports both world-space non-PBC Float32 and Int16 fractional payloads.
   */
  applyTilePatchBinary(buffer: ArrayBuffer, header: TriumHeader): void {
    this._frameId = header.frameId;
    this._simTimePs = header.simTimePs;
    this._reversed = (header.flags & FrameFlags.REVERSED) !== 0;
    this._pbcWrapped = (header.flags & FrameFlags.PBC_WRAPPED) !== 0;
    this._lastPacketBytes = buffer.byteLength;

    const payloadEnd = Math.min(buffer.byteLength, HEADER_BYTES + header.payloadBytes);
    const dv = new DataView(buffer);
    let off = HEADER_BYTES;

    const hasBox =
      (header.flags & FrameFlags.HAS_BOX) !== 0 ||
      (header.flags & FrameFlags.INT16_FRACTIONAL) !== 0;
    if (hasBox && off + 36 <= payloadEnd) {
      if (off % 4 === 0) {
        this.boxH.set(new Float32Array(buffer, off, 9), 0);
      } else {
        for (let i = 0; i < 9; i++) this.boxH[i] = dv.getFloat32(off + i * 4, true);
      }
      this._hasCell = true;
      off += 36;
    }

    const k = header.nAtoms;
    if (k <= 0 || off + k * 4 > payloadEnd) {
      this._lastDirtyAtomCount = 0;
      return;
    }

    const indicesAligned = off % 4 === 0;
    const indicesView = indicesAligned ? new Uint32Array(buffer, off, k) : undefined;
    const indicesOff = off;
    off += k * 4;

    const dstCoords = this.coords;
    const cap = this.capacity;
    let updatedCount = 0;

    if (header.flags & FrameFlags.INT16_FRACTIONAL) {
      const avail = Math.min(k, Math.floor(Math.max(0, payloadEnd - off) / 6));
      const scale = 1.0 / (2.0 * 32767.0);
      const h00 = this.boxH[0], h01 = this.boxH[1], h02 = this.boxH[2];
      const h10 = this.boxH[3], h11 = this.boxH[4], h12 = this.boxH[5];
      const h20 = this.boxH[6], h21 = this.boxH[7], h22 = this.boxH[8];

      if (off % 2 === 0 && avail > 0) {
        const q16 = new Int16Array(buffer, off, avail * 3);
        for (let i = 0; i < avail; i++) {
          const slot = indicesView ? indicesView[i] : dv.getUint32(indicesOff + i * 4, true);
          if (slot >= cap) continue;
          const s3 = i * 3;
          const fx = q16[s3] * scale + 0.5;
          const fy = q16[s3 + 1] * scale + 0.5;
          const fz = q16[s3 + 2] * scale + 0.5;
          const d3 = slot * 3;
          dstCoords[d3] = h00 * fx + h01 * fy + h02 * fz;
          dstCoords[d3 + 1] = h10 * fx + h11 * fy + h12 * fz;
          dstCoords[d3 + 2] = h20 * fx + h21 * fy + h22 * fz;
          this.markDirtyIndex(slot);
          updatedCount++;
        }
      } else {
        for (let i = 0; i < avail; i++) {
          const slot = indicesView ? indicesView[i] : dv.getUint32(indicesOff + i * 4, true);
          if (slot >= cap) continue;
          const bOff = off + i * 6;
          const fx = dv.getInt16(bOff, true) * scale + 0.5;
          const fy = dv.getInt16(bOff + 2, true) * scale + 0.5;
          const fz = dv.getInt16(bOff + 4, true) * scale + 0.5;
          const d3 = slot * 3;
          dstCoords[d3] = h00 * fx + h01 * fy + h02 * fz;
          dstCoords[d3 + 1] = h10 * fx + h11 * fy + h12 * fz;
          dstCoords[d3 + 2] = h20 * fx + h21 * fy + h22 * fz;
          this.markDirtyIndex(slot);
          updatedCount++;
        }
      }
      off += k * 6;
      const pad = (4 - (off % 4)) % 4;
      if (pad > 0 && off + pad <= payloadEnd) {
        off += pad;
      }
    } else {
      const avail = Math.min(k, Math.floor(Math.max(0, payloadEnd - off) / 12));
      if (off % 4 === 0 && avail > 0) {
        const srcCoords = new Float32Array(buffer, off, avail * 3);
        for (let i = 0; i < avail; i++) {
          const slot = indicesView ? indicesView[i] : dv.getUint32(indicesOff + i * 4, true);
          if (slot >= cap) continue;
          const s3 = i * 3;
          const d3 = slot * 3;
          dstCoords[d3] = srcCoords[s3];
          dstCoords[d3 + 1] = srcCoords[s3 + 1];
          dstCoords[d3 + 2] = srcCoords[s3 + 2];
          this.markDirtyIndex(slot);
          updatedCount++;
        }
      } else {
        for (let i = 0; i < avail; i++) {
          const slot = indicesView ? indicesView[i] : dv.getUint32(indicesOff + i * 4, true);
          if (slot >= cap) continue;
          const bOff = off + i * 12;
          const d3 = slot * 3;
          dstCoords[d3] = dv.getFloat32(bOff, true);
          dstCoords[d3 + 1] = dv.getFloat32(bOff + 4, true);
          dstCoords[d3 + 2] = dv.getFloat32(bOff + 8, true);
          this.markDirtyIndex(slot);
          updatedCount++;
        }
      }
      off += k * 12;
    }

    if ((header.flags & FrameFlags.HAS_SCALAR) && off < payloadEnd) {
      const scalarChannelIdx = header.reserved & 0xff;
      const channel = PHYSICS_CHANNELS[scalarChannelIdx] ?? this._activeChannel;
      this._activeChannel = channel;
      const buf = this.getOrCreateChannelBuffer(channel);
      const availScalars = Math.min(k, Math.floor((payloadEnd - off) / 4));
      if (off % 4 === 0 && availScalars > 0) {
        const srcScalars = new Float32Array(buffer, off, availScalars);
        for (let i = 0; i < availScalars; i++) {
          const slot = indicesView ? indicesView[i] : dv.getUint32(indicesOff + i * 4, true);
          if (slot < cap) {
            buf[slot] = srcScalars[i];
            this.markDirtyIndex(slot);
          }
        }
      } else {
        for (let i = 0; i < availScalars; i++) {
          const slot = indicesView ? indicesView[i] : dv.getUint32(indicesOff + i * 4, true);
          if (slot < cap) {
            buf[slot] = dv.getFloat32(off + i * 4, true);
            this.markDirtyIndex(slot);
          }
        }
      }
      this.setScalarRangeInPlace(channel, header.scalarMin, header.scalarMax);
      this.ensureAttribute(channel);
      this._scalarsDirty = true;
    }

    if (updatedCount > 0) {
      this._dirtyVersion++;
    }
    this._lastDirtyAtomCount = updatedCount;
  }

  applyTilePatch(msg: FrameTilePatchMessage, packetBytes?: number): void {
    // Sparse in-place scatter: update ONLY the dirty atomIndices slots without
    // overwriting from offset 0 or mutating the total atom count.
    this._frameId = msg.header.frameId;
    this._simTimePs = msg.header.simTimePs;
    this._reversed = (msg.header.flags & FrameFlags.REVERSED) !== 0;
    this._pbcWrapped = (msg.header.flags & FrameFlags.PBC_WRAPPED) !== 0;
    if (msg.boxH && msg.boxH.length >= 9) {
      this.boxH.set(msg.boxH.subarray(0, 9), 0);
      this._hasCell = true;
    }

    const indices = msg.atomIndices;
    const srcCoords = msg.coords;
    const dstCoords = this.coords;
    const count = indices.length;
    const maxSrcCoords = srcCoords.length;
    let updatedCount = 0;

    for (let i = 0; i < count; i++) {
      const slot = indices[i];
      const s3 = i * 3;
      if (slot >= this.capacity || s3 + 2 >= maxSrcCoords) continue;
      const d3 = slot * 3;
      dstCoords[d3] = srcCoords[s3];
      dstCoords[d3 + 1] = srcCoords[s3 + 1];
      dstCoords[d3 + 2] = srcCoords[s3 + 2];
      this.markDirtyIndex(slot);
      updatedCount++;
    }

    if (msg.scalars) {
      const channel = PHYSICS_CHANNELS[msg.scalarChannel] ?? this._activeChannel;
      this._activeChannel = channel;
      const buf = this.getOrCreateChannelBuffer(channel);
      const srcScalars = msg.scalars;
      const maxSrcScalars = srcScalars.length;
      for (let i = 0; i < count && i < maxSrcScalars; i++) {
        const slot = indices[i];
        if (slot < this.capacity) {
          buf[slot] = srcScalars[i];
          this.markDirtyIndex(slot);
        }
      }
      this.setScalarRangeInPlace(channel, msg.header.scalarMin, msg.header.scalarMax);
      this.ensureAttribute(channel);
      this._scalarsDirty = true;
    }

    if (updatedCount > 0) {
      this._dirtyVersion++;
    }
    this._lastDirtyAtomCount = updatedCount;
    if (packetBytes !== undefined) {
      this._lastPacketBytes = packetBytes;
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

    for (let i = 0; i < msg.insertedSlots.length; i++) {
      const s = msg.insertedSlots[i];
      if (s < this.capacity) this.markDirtyIndex(s);
    }
    for (let i = 0; i < msg.deletedSlots.length; i++) {
      const s = msg.deletedSlots[i];
      if (s < this.capacity) this.markDirtyIndex(s);
    }

    if (res.maxActiveSlot > this._count) {
      this._count = Math.min(res.maxActiveSlot, this.capacity);
      this.targetOffsets[1] = this._count;
      this._coordViewCount = -1;
      this._radiiViewCount = -1;
    }
    this._topologyDirty = true;
    this._dirtyVersion++;
  }

  /**
   * Create or update the Mol* `ParticleList` backed by our persistent typed-array views.
   * Uses `Particle.withPoses` when topology/count did not change so Mol* reuses GPU buffers.
   */
  getParticleList(forceNew = false): ParticleList {
    const coordView = this.getPositionStagingView();

    if (!forceNew && !this._topologyDirty && this._currentList && this._currentList.count === this._count) {
      this._currentList = Particle.withPoses(this._currentList, coordView);
      return this._currentList;
    }

    if (!this._radiiView || this._radiiViewCount !== this._count) {
      this._radiiView = this.slotPool.radii.subarray(0, this._count);
      this._radiiViewCount = this._count;
    }
    const radiiView = this._radiiView;
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
