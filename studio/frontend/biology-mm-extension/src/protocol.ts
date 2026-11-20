/**
 * TriumStream v1 binary WebSocket protocol decoder (byte-for-byte mirror of studio-backend/trium_studio/protocol.py).
 */

export const TRIUM_MAGIC = 0x314d5254; // 'TRM1' little-endian
export const HEADER_BYTES = 40;

export const MsgType = {
  TOPOLOGY_INIT: 0x01,
  FRAME_FULL: 0x02,
  FRAME_TILE_PATCH: 0x03,
  OVERLAY_VECTORS: 0x04,
  VOLUME_GRID3D: 0x05,
  GCMC_SLOT_DELTA: 0x06,
} as const;

export const FrameFlags = {
  INT16_FRACTIONAL: 0x01,
  HAS_SCALAR: 0x02,
  HAS_BOX: 0x04,
  PBC_WRAPPED: 0x08,
  REVERSED: 0x10,
} as const;

export const PHYSICS_CHANNELS = [
  'force-magnitude',
  'kinetic-temp',
  'charge-potential',
  'bond-order',
  'atomic-stress',
  'ternary-state',
  'multiscale-zone',
] as const;

export type PhysicsChannelName = (typeof PHYSICS_CHANNELS)[number];

export interface TriumHeader {
  msgType: number;
  flags: number;
  frameId: number;
  nAtoms: number;
  scalarMin: number;
  scalarMax: number;
  simTimePs: number;
  payloadBytes: number;
  reserved: number;
}

export interface TopologyInitMessage {
  kind: 'topology';
  header: TriumHeader;
  boxH: Float32Array;
  atomicNumbers: Uint8Array;
  bonds: Uint32Array;
  residueIds: Uint32Array;
  tileIds: Uint32Array;
  radii: Float32Array;
  coords?: Float32Array;
}

export interface FrameFullMessage {
  kind: 'frame-full';
  header: TriumHeader;
  boxH?: Float32Array;
  coords: Float32Array;
  scalars?: Float32Array;
  scalarChannel: number;
  reversed: boolean;
}

export interface FrameTilePatchMessage {
  kind: 'tile-patch';
  header: TriumHeader;
  boxH: Float32Array;
  atomIndices: Uint32Array;
  coords: Float32Array;
  scalars?: Float32Array;
  tileId: number;
  scalarChannel: number;
}

export interface OverlayVectorsMessage {
  kind: 'overlays';
  header: TriumHeader;
  nVectors: number;
  nBonds: number;
  vectorStarts: Float32Array;
  vectorDirs: Float32Array;
  vectorMags: Float32Array;
  bondPairs: Uint32Array;
  bondOrders: Float32Array;
}

export interface VolumeGrid3DMessage {
  kind: 'volume-grid3d';
  header: TriumHeader;
  dims: [number, number, number];
  origin: Float32Array;
  boxH: Float32Array;
  grid: Float32Array;
  channelNameId: number;
}

export interface GcmcSlotDeltaMessage {
  kind: 'gcmc-delta';
  header: TriumHeader;
  insertedSlots: Uint32Array;
  insertedAtomicNumbers: Uint8Array;
  insertedCoords: Float32Array;
  deletedSlots: Uint32Array;
}

export type TriumMessage =
  | TopologyInitMessage
  | FrameFullMessage
  | FrameTilePatchMessage
  | OverlayVectorsMessage
  | VolumeGrid3DMessage
  | GcmcSlotDeltaMessage;

export function parseHeader(view: DataView, byteOffset = 0): TriumHeader {
  if (view.byteLength - byteOffset < HEADER_BYTES) {
    throw new Error(`TriumStream header truncated: need ${HEADER_BYTES} bytes, got ${view.byteLength - byteOffset}`);
  }
  const magic = view.getUint32(byteOffset, true);
  if (magic !== TRIUM_MAGIC) {
    throw new Error(`Invalid TriumStream magic: 0x${magic.toString(16)}`);
  }
  return {
    msgType: view.getUint16(byteOffset + 4, true),
    flags: view.getUint16(byteOffset + 6, true),
    frameId: view.getUint32(byteOffset + 8, true),
    nAtoms: view.getUint32(byteOffset + 12, true),
    scalarMin: view.getFloat32(byteOffset + 16, true),
    scalarMax: view.getFloat32(byteOffset + 20, true),
    simTimePs: view.getFloat64(byteOffset + 24, true),
    payloadBytes: view.getUint32(byteOffset + 32, true),
    reserved: view.getUint32(byteOffset + 36, true),
  };
}

export function decodeInt16Fractional(q16: Int16Array, boxH: Float32Array, out?: Float32Array): Float32Array {
  const n = Math.floor(q16.length / 3);
  const dst = out && out.length >= n * 3 ? out : new Float32Array(n * 3);
  const scale = 1.0 / (2.0 * 32767.0);
  const h00 = boxH[0], h01 = boxH[1], h02 = boxH[2];
  const h10 = boxH[3], h11 = boxH[4], h12 = boxH[5];
  const h20 = boxH[6], h21 = boxH[7], h22 = boxH[8];

  for (let i = 0; i < n; i++) {
    const j = i * 3;
    const fx = q16[j] * scale + 0.5;
    const fy = q16[j + 1] * scale + 0.5;
    const fz = q16[j + 2] * scale + 0.5;
    dst[j] = h00 * fx + h01 * fy + h02 * fz;
    dst[j + 1] = h10 * fx + h11 * fy + h12 * fz;
    dst[j + 2] = h20 * fx + h21 * fy + h22 * fz;
  }
  return dst;
}

function copyFloat32Slice(buf: ArrayBuffer, offset: number, count: number): Float32Array {
  const bytes = count * 4;
  if (offset % 4 === 0) {
    return new Float32Array(buf.slice(offset, offset + bytes));
  }
  const out = new Float32Array(count);
  const dv = new DataView(buf, offset, bytes);
  for (let i = 0; i < count; i++) out[i] = dv.getFloat32(i * 4, true);
  return out;
}

function copyUint32Slice(buf: ArrayBuffer, offset: number, count: number): Uint32Array {
  const bytes = count * 4;
  if (offset % 4 === 0) {
    return new Uint32Array(buf.slice(offset, offset + bytes));
  }
  const out = new Uint32Array(count);
  const dv = new DataView(buf, offset, bytes);
  for (let i = 0; i < count; i++) out[i] = dv.getUint32(i * 4, true);
  return out;
}

function copyInt16Slice(buf: ArrayBuffer, offset: number, count: number): Int16Array {
  const bytes = count * 2;
  if (offset % 2 === 0) {
    return new Int16Array(buf, offset, count);
  }
  const out = new Int16Array(count);
  const dv = new DataView(buf, offset, bytes);
  for (let i = 0; i < count; i++) out[i] = dv.getInt16(i * 2, true);
  return out;
}

export function decodeTriumMessage(buffer: ArrayBuffer): TriumMessage {
  const view = new DataView(buffer);
  const header = parseHeader(view, 0);
  if (view.byteLength < HEADER_BYTES + header.payloadBytes) {
    throw new Error(`TriumStream payload truncated: need ${HEADER_BYTES + header.payloadBytes}, got ${view.byteLength}`);
  }
  let off = HEADER_BYTES;

  switch (header.msgType) {
    case MsgType.TOPOLOGY_INIT: {
      const nBonds = view.getUint32(off, true); off += 4;
      const boxH = copyFloat32Slice(buffer, off, 9); off += 36;
      const atomicNumbers = new Uint8Array(buffer.slice(off, off + header.nAtoms)); off += header.nAtoms;
      const bonds = copyUint32Slice(buffer, off, 2 * nBonds); off += 8 * nBonds;
      const residueIds = copyUint32Slice(buffer, off, header.nAtoms); off += 4 * header.nAtoms;
      const tileIds = copyUint32Slice(buffer, off, header.nAtoms); off += 4 * header.nAtoms;
      const radii = copyFloat32Slice(buffer, off, header.nAtoms); off += 4 * header.nAtoms;
      let coords: Float32Array | undefined;
      if (header.reserved === 1 && off + 12 * header.nAtoms <= buffer.byteLength) {
        coords = copyFloat32Slice(buffer, off, 3 * header.nAtoms);
      }
      return { kind: 'topology', header, boxH, atomicNumbers, bonds, residueIds, tileIds, radii, coords };
    }

    case MsgType.FRAME_FULL: {
      let boxH: Float32Array | undefined;
      if (header.flags & FrameFlags.HAS_BOX) {
        boxH = copyFloat32Slice(buffer, off, 9); off += 36;
      }
      let coords: Float32Array;
      if (header.flags & FrameFlags.INT16_FRACTIONAL) {
        if (!boxH) throw new Error('INT16_FRACTIONAL frame requires HAS_BOX');
        const q16 = copyInt16Slice(buffer, off, 3 * header.nAtoms); off += 6 * header.nAtoms;
        coords = decodeInt16Fractional(q16, boxH);
      } else {
        coords = copyFloat32Slice(buffer, off, 3 * header.nAtoms); off += 12 * header.nAtoms;
      }
      let scalars: Float32Array | undefined;
      if (header.flags & FrameFlags.HAS_SCALAR) {
        scalars = copyFloat32Slice(buffer, off, header.nAtoms);
      }
      return {
        kind: 'frame-full',
        header,
        boxH,
        coords,
        scalars,
        scalarChannel: header.reserved,
        reversed: (header.flags & FrameFlags.REVERSED) !== 0,
      };
    }

    case MsgType.FRAME_TILE_PATCH: {
      const boxH = copyFloat32Slice(buffer, off, 9); off += 36;
      const atomIndices = copyUint32Slice(buffer, off, header.nAtoms); off += 4 * header.nAtoms;
      let coords: Float32Array;
      if (header.flags & FrameFlags.INT16_FRACTIONAL) {
        const q16 = copyInt16Slice(buffer, off, 3 * header.nAtoms); off += 6 * header.nAtoms;
        const pad = (4 - (off % 4)) % 4;
        if (pad > 0 && off + pad <= HEADER_BYTES + header.payloadBytes) {
          off += pad;
        }
        coords = decodeInt16Fractional(q16, boxH);
      } else {
        coords = copyFloat32Slice(buffer, off, 3 * header.nAtoms); off += 12 * header.nAtoms;
      }
      let scalars: Float32Array | undefined;
      if (header.flags & FrameFlags.HAS_SCALAR) {
        scalars = copyFloat32Slice(buffer, off, header.nAtoms);
      }
      return {
        kind: 'tile-patch',
        header,
        boxH,
        atomIndices,
        coords,
        scalars,
        tileId: header.reserved >>> 8,
        scalarChannel: header.reserved & 0xff,
      };
    }

    case MsgType.OVERLAY_VECTORS: {
      const nVectors = view.getUint32(off, true); off += 4;
      const nBonds = view.getUint32(off, true); off += 4;
      const vectorStarts = copyFloat32Slice(buffer, off, 3 * nVectors); off += 12 * nVectors;
      const vectorDirs = copyFloat32Slice(buffer, off, 3 * nVectors); off += 12 * nVectors;
      const vectorMags = copyFloat32Slice(buffer, off, nVectors); off += 4 * nVectors;
      const bondPairs = copyUint32Slice(buffer, off, 2 * nBonds); off += 8 * nBonds;
      const bondOrders = copyFloat32Slice(buffer, off, nBonds);
      return { kind: 'overlays', header, nVectors, nBonds, vectorStarts, vectorDirs, vectorMags, bondPairs, bondOrders };
    }

    case MsgType.VOLUME_GRID3D: {
      const nx = view.getUint32(off, true); off += 4;
      const ny = view.getUint32(off, true); off += 4;
      const nz = view.getUint32(off, true); off += 4;
      const origin = copyFloat32Slice(buffer, off, 3); off += 12;
      const boxH = copyFloat32Slice(buffer, off, 9); off += 36;
      const grid = copyFloat32Slice(buffer, off, nx * ny * nz);
      return {
        kind: 'volume-grid3d',
        header,
        dims: [nx, ny, nz],
        origin,
        boxH,
        grid,
        channelNameId: header.reserved,
      };
    }

    case MsgType.GCMC_SLOT_DELTA: {
      const nInsert = view.getUint32(off, true); off += 4;
      const nDelete = view.getUint32(off, true); off += 4;
      const insertedSlots = copyUint32Slice(buffer, off, nInsert); off += 4 * nInsert;
      const insertedAtomicNumbers = new Uint8Array(buffer.slice(off, off + nInsert)); off += nInsert;
      const insertedCoords = copyFloat32Slice(buffer, off, 3 * nInsert); off += 12 * nInsert;
      const deletedSlots = copyUint32Slice(buffer, off, nDelete);
      return {
        kind: 'gcmc-delta',
        header,
        insertedSlots,
        insertedAtomicNumbers,
        insertedCoords,
        deletedSlots,
      };
    }

    default:
      throw new Error(`Unknown TriumStream msgType: 0x${header.msgType.toString(16)}`);
  }
}
