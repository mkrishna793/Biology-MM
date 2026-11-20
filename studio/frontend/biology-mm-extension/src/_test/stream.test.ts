import { describe, expect, it, vi } from 'vitest';
import { Subject } from 'rxjs';
import { Particle } from '@molstar/model/model/particles/particle-list';
import { GcmcSlotPool } from '../gcmc-slot-pool.js';
import { GpuParticleStreamUpdater } from '../gpu-buffer-updater.js';
import { BiologyMmStreamController, type AtomPickedEvent, type StreamFrameStats } from '../live-stream-behavior.js';
import {
  decodeTriumMessage,
  FrameFlags,
  HEADER_BYTES,
  MsgType,
  TRIUM_MAGIC,
  type FrameTilePatchMessage,
  type GcmcSlotDeltaMessage,
  type TopologyInitMessage,
} from '../protocol.js';

function writeHeader(
  dv: DataView,
  msgType: number,
  flags: number,
  frameId: number,
  nAtoms: number,
  scalarMin: number,
  scalarMax: number,
  simTimePs: number,
  payloadBytes: number,
  reserved = 0,
): void {
  dv.setUint32(0, TRIUM_MAGIC, true);
  dv.setUint16(4, msgType, true);
  dv.setUint16(6, flags, true);
  dv.setUint32(8, frameId, true);
  dv.setUint32(12, nAtoms, true);
  dv.setFloat32(16, scalarMin, true);
  dv.setFloat32(20, scalarMax, true);
  dv.setFloat64(24, simTimePs, true);
  dv.setUint32(32, payloadBytes, true);
  dv.setUint32(36, reserved, true);
}

function buildTopologyPacket(
  nAtoms: number,
  boxDiag: [number, number, number],
  coords?: Float32Array,
): ArrayBuffer {
  const nBonds = 0;
  const hasCoords = coords !== undefined;
  const payloadBytes = 4 + 36 + nAtoms + 8 * nBonds + 4 * nAtoms + 4 * nAtoms + 4 * nAtoms + (hasCoords ? 12 * nAtoms : 0);
  const buf = new ArrayBuffer(HEADER_BYTES + payloadBytes);
  const dv = new DataView(buf);
  writeHeader(dv, MsgType.TOPOLOGY_INIT, 0, 0, nAtoms, 0, 1, 0.0, payloadBytes, hasCoords ? 1 : 0);

  let off = HEADER_BYTES;
  dv.setUint32(off, nBonds, true);
  off += 4;

  const box = [boxDiag[0], 0, 0, 0, boxDiag[1], 0, 0, 0, boxDiag[2]];
  for (let i = 0; i < 9; i++) dv.setFloat32(off + i * 4, box[i], true);
  off += 36;

  for (let i = 0; i < nAtoms; i++) dv.setUint8(off + i, 6);
  off += nAtoms;

  // residueIds, tileIds
  for (let i = 0; i < nAtoms; i++) dv.setUint32(off + i * 4, i, true);
  off += 4 * nAtoms;
  for (let i = 0; i < nAtoms; i++) dv.setUint32(off + i * 4, 0, true);
  off += 4 * nAtoms;

  // radii
  for (let i = 0; i < nAtoms; i++) dv.setFloat32(off + i * 4, 1.7, true);
  off += 4 * nAtoms;

  if (coords) {
    for (let i = 0; i < nAtoms * 3; i++) dv.setFloat32(off + i * 4, coords[i], true);
  }
  return buf;
}

function buildFrameFullInt16Packet(
  frameId: number,
  simTimePs: number,
  boxDiag: [number, number, number],
  q16: Int16Array,
  scalars?: Float32Array,
  scalarMin = 0,
  scalarMax = 1,
  channelId = 0,
  reversed = false,
): ArrayBuffer {
  const nAtoms = Math.floor(q16.length / 3);
  let flags = FrameFlags.HAS_BOX | FrameFlags.INT16_FRACTIONAL;
  if (scalars) flags |= FrameFlags.HAS_SCALAR;
  if (reversed) flags |= FrameFlags.REVERSED;

  const payloadBytes = 36 + 6 * nAtoms + (scalars ? 4 * nAtoms : 0);
  const buf = new ArrayBuffer(HEADER_BYTES + payloadBytes);
  const dv = new DataView(buf);
  writeHeader(dv, MsgType.FRAME_FULL, flags, frameId, nAtoms, scalarMin, scalarMax, simTimePs, payloadBytes, channelId);

  let off = HEADER_BYTES;
  const box = [boxDiag[0], 0, 0, 0, boxDiag[1], 0, 0, 0, boxDiag[2]];
  for (let i = 0; i < 9; i++) dv.setFloat32(off + i * 4, box[i], true);
  off += 36;

  new Int16Array(buf, off, q16.length).set(q16);
  off += 6 * nAtoms;

  if (scalars) {
    for (let i = 0; i < nAtoms; i++) dv.setFloat32(off + i * 4, scalars[i], true);
  }
  return buf;
}

function buildFrameFullFloat32Packet(
  frameId: number,
  simTimePs: number,
  coords: Float32Array,
  scalars?: Float32Array,
  scalarMin = 0,
  scalarMax = 10,
): ArrayBuffer {
  const nAtoms = Math.floor(coords.length / 3);
  const flags = scalars ? FrameFlags.HAS_SCALAR : 0;
  const payloadBytes = 12 * nAtoms + (scalars ? 4 * nAtoms : 0);
  const buf = new ArrayBuffer(HEADER_BYTES + payloadBytes);
  const dv = new DataView(buf);
  writeHeader(dv, MsgType.FRAME_FULL, flags, frameId, nAtoms, scalarMin, scalarMax, simTimePs, payloadBytes, 0);

  let off = HEADER_BYTES;
  new Float32Array(buf, off, coords.length).set(coords);
  off += 12 * nAtoms;
  if (scalars) {
    for (let i = 0; i < nAtoms; i++) dv.setFloat32(off + i * 4, scalars[i], true);
  }
  return buf;
}

function createMockPlugin() {
  const click$ = new Subject<any>();
  const colorReg = new Map<string, any>();
  const sizeReg = new Map<string, any>();
  const reprReg = new Map<string, any>();

  const makeRegistry = (map: Map<string, any>) => ({
    get default() {
      return map.values().next().value;
    },
    has: (p: any) => map.has(p.name),
    add: (p: any) => map.set(p.name, p),
    remove: (p: any) => map.delete(p.name),
  });

  const plugin: any = {
    behaviors: {
      interaction: {
        click: click$.asObservable(),
      },
    },
    representation: {
      particles: {
        registry: makeRegistry(reprReg),
        themes: {
          colorThemeRegistry: makeRegistry(colorReg),
          sizeThemeRegistry: makeRegistry(sizeReg),
        },
      },
    },
    canvas3d: {
      requestCameraReset: vi.fn(),
    },
    // Omit state.data so controller runs cleanly in headless unit tests without full StateBuilder
  };

  return { plugin, click$, colorReg };
}

describe('GcmcSlotPool', () => {
  it('manages fixed capacity, active mask, and zero-radius tombstoning on delete', () => {
    const pool = new GcmcSlotPool(16);
    expect(pool.capacity).toBe(16);
    expect(pool.activeCount).toBe(0);

    pool.initFromTopology(4, Uint8Array.of(6, 6, 8, 1), Float32Array.of(1.7, 1.7, 1.52, 1.2));
    expect(pool.activeCount).toBe(4);
    expect(pool.isActive(2)).toBe(true);
    expect(pool.radii[2]).toBeCloseTo(1.52, 5);

    const coords = new Float32Array(16 * 3);
    const res = pool.applyDelta([4, 5], [1, 2], {
      atomicNumbers: Uint8Array.of(11, 17),
      coords,
      insertedCoords: Float32Array.of(10, 20, 30, 40, 50, 60),
    });

    expect(res.deletedCount).toBe(2);
    expect(res.insertedCount).toBe(2);
    expect(pool.activeCount).toBe(4);
    expect(pool.isActive(1)).toBe(false);
    expect(pool.isActive(2)).toBe(false);
    expect(pool.radii[1]).toBe(0);
    expect(pool.radii[2]).toBe(0);
    expect(pool.isActive(4)).toBe(true);
    expect(pool.isActive(5)).toBe(true);
    expect(pool.radii[4]).toBeGreaterThan(0);
    expect(coords[4 * 3]).toBe(10);
    expect(coords[5 * 3 + 2]).toBe(60);
  });
});

describe('GpuParticleStreamUpdater', () => {
  it('initializes topology and updates float32 & Int16 fractional frames in-place', () => {
    const updater = new GpuParticleStreamUpdater(64);
    const topoBuf = buildTopologyPacket(3, [20, 20, 20], Float32Array.of(1, 2, 3, 4, 5, 6, 7, 8, 9));
    const topoMsg = decodeTriumMessage(topoBuf) as TopologyInitMessage;
    updater.applyTopologyInit(topoMsg);

    const list1 = updater.getParticleList();
    expect(Particle.is(list1)).toBe(true);
    expect(list1.count).toBe(3);
    expect(updater.coords[0]).toBe(1);
    expect(updater.coords[8]).toBe(9);

    // Int16 fractional frame: q16 = [0, -32767, 32767, ...] => f = [0.5, 0.0, 1.0] * 20 = [10, 0, 20]
    const q16 = Int16Array.of(0, -32767, 32767, 0, 0, 0, -32767, 0, 0);
    const scalars = Float32Array.of(1.5, 2.5, 3.5);
    const frameBuf = buildFrameFullInt16Packet(5, 1.25, [20, 20, 20], q16, scalars, 1.5, 3.5, 0, true);
    const header = {
      msgType: MsgType.FRAME_FULL,
      flags: FrameFlags.HAS_BOX | FrameFlags.INT16_FRACTIONAL | FrameFlags.HAS_SCALAR | FrameFlags.REVERSED,
      frameId: 5,
      nAtoms: 3,
      scalarMin: 1.5,
      scalarMax: 3.5,
      simTimePs: 1.25,
      payloadBytes: 0,
      reserved: 0,
    };

    const coordsRef = updater.coords;
    updater.applyFrameFullBinary(frameBuf, header);

    expect(updater.coords).toBe(coordsRef);
    expect(updater.coords[0]).toBeCloseTo(10, 2);
    expect(updater.coords[1]).toBeCloseTo(0, 2);
    expect(updater.coords[2]).toBeCloseTo(20, 2);
    expect(updater.reversed).toBe(true);

    const attr = updater.getAttribute('force-magnitude');
    expect(attr).toBeDefined();
    expect(attr!.min).toBe(1.5);
    expect(attr!.max).toBe(3.5);
    expect(attr!.column.value(1)).toBeCloseTo(2.5, 5);

    const list2 = updater.getParticleList();
    expect(Particle.isSamePoses(list1, list2)).toBe(true);
  });

  it('applies Morton tile patches and GCMC tombstone deltas without reallocating', () => {
    const updater = new GpuParticleStreamUpdater(32);
    const topoMsg = decodeTriumMessage(
      buildTopologyPacket(4, [10, 10, 10], Float32Array.of(0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3)),
    ) as TopologyInitMessage;
    updater.applyTopologyInit(topoMsg);

    const patch: FrameTilePatchMessage = {
      kind: 'tile-patch',
      header: {
        msgType: MsgType.FRAME_TILE_PATCH,
        flags: FrameFlags.HAS_SCALAR,
        frameId: 2,
        nAtoms: 2,
        scalarMin: 0,
        scalarMax: 10,
        simTimePs: 0.5,
        payloadBytes: 0,
        reserved: 0,
      },
      boxH: Float32Array.of(10, 0, 0, 0, 10, 0, 0, 0, 10),
      atomIndices: Uint32Array.of(1, 3),
      coords: Float32Array.of(9, 9, 9, 8, 8, 8),
      scalars: Float32Array.of(4.2, 7.7),
      tileId: 12,
      scalarChannel: 0,
    };
    updater.applyTilePatch(patch);

    expect(updater.coords[0]).toBe(0);
    expect(updater.coords[3]).toBe(9);
    expect(updater.coords[6]).toBe(2);
    expect(updater.coords[9]).toBe(8);
    expect(updater.getAttribute('force-magnitude')?.column.value(3)).toBeCloseTo(7.7, 5);

    const delta: GcmcSlotDeltaMessage = {
      kind: 'gcmc-delta',
      header: {
        msgType: MsgType.GCMC_SLOT_DELTA,
        flags: 0,
        frameId: 3,
        nAtoms: 4,
        scalarMin: 0,
        scalarMax: 1,
        simTimePs: 0.75,
        payloadBytes: 0,
        reserved: 0,
      },
      insertedSlots: Uint32Array.of(4),
      insertedAtomicNumbers: Uint8Array.of(18),
      insertedCoords: Float32Array.of(5, 5, 5),
      deletedSlots: Uint32Array.of(0),
    };
    updater.applyGcmcDelta(delta);

    expect(updater.radii[0]).toBe(0);
    expect(updater.radii[4]).toBeGreaterThan(0);
    expect(updater.count).toBe(5);
    expect(updater.activeCount).toBe(4);
  });

  it('supports 1,000,000 atoms with zero per-frame buffer reallocation', () => {
    const N = 1_000_000;
    const updater = new GpuParticleStreamUpdater(N);
    expect(updater.coords.length).toBe(3_000_000);
    expect(updater.radii.length).toBe(1_000_000);

    const backingBuffer = updater.coords.buffer;
    const coords = new Float32Array(N * 3);
    coords[0] = 1.25;
    coords[N * 3 - 1] = 99.5;
    const pkt = buildFrameFullFloat32Packet(1, 0.1, coords);
    const header = {
      msgType: MsgType.FRAME_FULL,
      flags: 0,
      frameId: 1,
      nAtoms: N,
      scalarMin: 0,
      scalarMax: 1,
      simTimePs: 0.1,
      payloadBytes: N * 12,
      reserved: 0,
    };

    updater.applyFrameFullBinary(pkt, header);
    expect(updater.coords.buffer).toBe(backingBuffer);
    expect(updater.count).toBe(N);
    expect(updater.coords[0]).toBeCloseTo(1.25, 5);
    expect(updater.coords[N * 3 - 1]).toBeCloseTo(99.5, 5);

    const list = updater.getParticleList();
    expect(list.count).toBe(N);
  });
});

describe('BiologyMmStreamController', () => {
  it('throttles rendering via rAF, drops stale frames, emits stats & atomPicked$', async () => {
    const { plugin, click$ } = createMockPlugin();
    let scheduledCb: (() => void) | undefined;
    let nextId = 1;

    const controller = new BiologyMmStreamController(plugin, {
      capacity: 16,
      scheduleFrame: (cb) => {
        scheduledCb = cb;
        return nextId++;
      },
      cancelFrame: () => {
        scheduledCb = undefined;
      },
    });

    const statsHistory: StreamFrameStats[] = [];
    const sub1 = controller.frameStats$.subscribe((s) => statsHistory.push(s));
    const picked: AtomPickedEvent[] = [];
    const sub2 = controller.atomPicked$.subscribe((e) => picked.push(e));

    // Ingest topology + 3 rapid frames before rAF fires -> 2 stale frames should be dropped
    controller.ingestPacket(buildTopologyPacket(2, [10, 10, 10], Float32Array.of(1, 2, 3, 4, 5, 6)));
    controller.ingestPacket(buildFrameFullFloat32Packet(1, 0.1, Float32Array.of(1, 1, 1, 2, 2, 2)));
    controller.ingestPacket(buildFrameFullFloat32Packet(2, 0.2, Float32Array.of(3, 3, 3, 4, 4, 4)));
    controller.ingestPacket(buildFrameFullFloat32Packet(3, 0.3, Float32Array.of(5, 5, 5, 6, 6, 6), Float32Array.of(1.0, 9.0), 1.0, 9.0));

    expect(scheduledCb).toBeDefined();
    scheduledCb!();
    await new Promise((r) => setTimeout(r, 0));

    const latest = statsHistory[statsHistory.length - 1];
    expect(latest.frameId).toBe(3);
    expect(latest.nAtoms).toBe(2);
    expect(latest.droppedFrames).toBe(3); // topology + 3 frames -> 3 overwrites while pendingRender
    expect(latest.scalarMin).toBe(1.0);
    expect(latest.scalarMax).toBe(9.0);

    // Channel & palette updates
    controller.setChannel('atomic-stress');
    controller.setPalette('stress-inferno');
    expect(controller.updater.activeChannel).toBe('atomic-stress');

    // Atom click picking
    const list = controller.updater.getParticleList();
    click$.next({ current: { loci: Particle.Location(list, 1) } });
    expect(picked.length).toBe(1);
    expect(picked[0].index).toBe(1);
    expect(picked[0].coords).toEqual([6, 6, 6]);

    expect(plugin.canvas3d.requestCameraReset).toHaveBeenCalledTimes(1);
    controller.resetCamera();
    expect(plugin.canvas3d.requestCameraReset).toHaveBeenCalledTimes(2);

    sub1.unsubscribe();
    sub2.unsubscribe();
    controller.dispose();
  });
});
