"""TriumStream v1 binary WebSocket protocol (exact 40-byte header mirrored in TypeScript `protocol.ts`)."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

MAGIC = b"TRM1"

# Little-endian 40-byte header:
#   magic (4s), msg_type (H), flags (H), frame_id (I), n_atoms (I),
#   scalar_min (f32), scalar_max (f32), sim_time_ps (f64),
#   payload_bytes (I), reserved (I)
HEADER_STRUCT = struct.Struct("<4sHHIIffdII")
HEADER_SIZE = HEADER_STRUCT.size  # 40 bytes

MSG_TOPOLOGY_INIT = 0x01
MSG_FRAME_FULL = 0x02
MSG_FRAME_TILE_PATCH = 0x03
MSG_OVERLAY_VECTORS = 0x04
MSG_VOLUME_GRID3D = 0x05
MSG_GCMC_SLOT_DELTA = 0x06

FLAG_INT16_FRACTIONAL = 0x01
FLAG_HAS_SCALAR = 0x02
FLAG_HAS_BOX = 0x04
FLAG_PBC_WRAPPED = 0x08
FLAG_REVERSED = 0x10

_INT16_MAX = 32767.0


@dataclass(frozen=True)
class FrameHeader:
    msg_type: int
    flags: int
    frame_id: int
    n_atoms: int
    scalar_min: float
    scalar_max: float
    sim_time_ps: float
    payload_bytes: int
    reserved: int = 0


def _ensure_box_3x3(box_H: Sequence[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(box_H, dtype=np.float32)
    if arr.shape == (3,):
        out = np.zeros((3, 3), dtype=np.float32)
        out[0, 0], out[1, 1], out[2, 2] = arr[0], arr[1], arr[2]
        return out
    if arr.shape == (9,):
        return arr.reshape((3, 3)).astype(np.float32, copy=False)
    if arr.shape == (3, 3):
        return np.ascontiguousarray(arr, dtype=np.float32)
    raise ValueError(f"Expected box shape (3,), (9,), or (3, 3); got {arr.shape}")


def encode_int16_fractional(coords: np.ndarray, box_H: Sequence[float] | np.ndarray) -> np.ndarray:
    """
    Encode `(N, 3)` Cartesian viewer coordinates using periodic cell matrix `H` (columns = lattice vectors)
    into compact wrapped `int16` fractional coordinates `[-32767, 32767]` (6 bytes/atom).
    """
    pts = np.asarray(coords, dtype=np.float64)
    H = _ensure_box_3x3(box_H).astype(np.float64)
    Hinv = np.linalg.inv(H)
    frac = pts @ Hinv.T
    frac = frac - np.floor(frac)  # [0, 1)
    q = np.rint((frac - 0.5) * (2.0 * _INT16_MAX)).clip(-_INT16_MAX, _INT16_MAX)
    return np.ascontiguousarray(q, dtype=np.int16)


def decode_int16_fractional(q16: np.ndarray, box_H: Sequence[float] | np.ndarray) -> np.ndarray:
    """Decode `int16` fractional coordinates back into `(N, 3)` `float32` Cartesian coordinates."""
    q = np.asarray(q16, dtype=np.float64)
    H = _ensure_box_3x3(box_H).astype(np.float64)
    frac = (q / (2.0 * _INT16_MAX)) + 0.5
    return np.ascontiguousarray(frac @ H.T, dtype=np.float32)


def pack_header(
    msg_type: int,
    flags: int,
    frame_id: int,
    n_atoms: int,
    scalar_min: float,
    scalar_max: float,
    sim_time_ps: float,
    payload_bytes: int,
    reserved: int = 0,
) -> bytes:
    return HEADER_STRUCT.pack(
        MAGIC,
        int(msg_type),
        int(flags),
        int(frame_id),
        int(n_atoms),
        float(scalar_min),
        float(scalar_max),
        float(sim_time_ps),
        int(payload_bytes),
        int(reserved),
    )


def unpack_header(data: bytes | memoryview) -> FrameHeader:
    if len(data) < HEADER_SIZE:
        raise ValueError(f"Expected >= {HEADER_SIZE} header bytes, got {len(data)}")
    magic, msg_type, flags, frame_id, n_atoms, s_min, s_max, sim_t, pbytes, reserved = HEADER_STRUCT.unpack_from(data, 0)
    if magic != MAGIC:
        raise ValueError(f"Invalid TriumStream magic: {magic!r}")
    return FrameHeader(
        msg_type=msg_type,
        flags=flags,
        frame_id=frame_id,
        n_atoms=n_atoms,
        scalar_min=s_min,
        scalar_max=s_max,
        sim_time_ps=sim_t,
        payload_bytes=pbytes,
        reserved=reserved,
    )


def pack_topology_init(
    atomic_numbers: np.ndarray,
    bonds: Optional[np.ndarray],
    residue_ids: Optional[np.ndarray],
    tile_ids: Optional[np.ndarray],
    box_H: Sequence[float] | np.ndarray,
    coords: Optional[np.ndarray] = None,
    radii: Optional[np.ndarray] = None,
    frame_id: int = 0,
    sim_time_ps: float = 0.0,
) -> bytes:
    """
    Payload layout (little-endian):
      uint32 n_bonds
      float32[9] box_H (row-major 3x3 in viewer Angstrom)
      uint8[n_atoms] atomic_numbers
      uint32[2*n_bonds] bonds (pair indices)
      uint32[n_atoms] residue_ids
      uint32[n_atoms] tile_ids
      float32[n_atoms] radii (Angstrom; defaults to element vdW)
      float32[3*n_atoms] initial_coords (optional if coords passed, flagged via FLAG_HAS_BOX|FLAG_HAS_SCALAR? no: use reserved=1 when coords attached!)
    """
    z = np.ascontiguousarray(atomic_numbers, dtype=np.uint8)
    n_atoms = int(z.shape[0])
    b_arr = np.zeros((0, 2), dtype=np.uint32) if bonds is None else np.ascontiguousarray(bonds, dtype=np.uint32).reshape((-1, 2))
    n_bonds = int(b_arr.shape[0])
    res_arr = np.zeros(n_atoms, dtype=np.uint32) if residue_ids is None else np.ascontiguousarray(residue_ids, dtype=np.uint32)
    tile_arr = np.zeros(n_atoms, dtype=np.uint32) if tile_ids is None else np.ascontiguousarray(tile_ids, dtype=np.uint32)
    rad_arr = np.full(n_atoms, 1.5, dtype=np.float32) if radii is None else np.ascontiguousarray(radii, dtype=np.float32)
    H = _ensure_box_3x3(box_H)

    chunks = [
        struct.pack("<I", n_bonds),
        H.tobytes(order="C"),
        z.tobytes(),
        b_arr.tobytes(order="C"),
        res_arr.tobytes(),
        tile_arr.tobytes(),
        rad_arr.tobytes(),
    ]
    has_coords = 0
    if coords is not None:
        c = np.ascontiguousarray(coords, dtype=np.float32).reshape((n_atoms, 3))
        chunks.append(c.tobytes(order="C"))
        has_coords = 1
    payload = b"".join(chunks)
    return pack_header(MSG_TOPOLOGY_INIT, FLAG_HAS_BOX, frame_id, n_atoms, 0.0, 0.0, sim_time_ps, len(payload), reserved=has_coords) + payload


def pack_frame_full(
    coords: np.ndarray,
    frame_id: int,
    sim_time_ps: float,
    box_H: Optional[Sequence[float] | np.ndarray] = None,
    scalars: Optional[np.ndarray] = None,
    use_int16: bool = False,
    pbc_wrapped: bool = True,
    reversed_step: bool = False,
    scalar_channel: int = 0,
) -> bytes:
    """
    Pack a full frame:
      [optional float32[9] box_H]
      coords: int16[3*N] (if use_int16) OR float32[3*N]
      [optional float32[N] scalars]
    """
    pts = np.ascontiguousarray(coords, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"Expected coords (N, 3), got {pts.shape}")
    n_atoms = int(pts.shape[0])

    flags = 0
    if pbc_wrapped:
        flags |= FLAG_PBC_WRAPPED
    if reversed_step:
        flags |= FLAG_REVERSED

    parts = []
    if box_H is not None:
        H = _ensure_box_3x3(box_H)
        flags |= FLAG_HAS_BOX
        parts.append(H.tobytes(order="C"))
    else:
        H = None

    if use_int16:
        if H is None:
            raise ValueError("use_int16=True requires box_H")
        flags |= FLAG_INT16_FRACTIONAL
        q16 = encode_int16_fractional(pts, H)
        parts.append(q16.tobytes(order="C"))
    else:
        parts.append(pts.tobytes(order="C"))

    s_min, s_max = 0.0, 0.0
    if scalars is not None:
        sc = np.ascontiguousarray(scalars, dtype=np.float32).reshape((n_atoms,))
        finite_mask = np.isfinite(sc)
        if np.any(finite_mask):
            s_min = float(np.min(sc[finite_mask]))
            s_max = float(np.max(sc[finite_mask]))
        flags |= FLAG_HAS_SCALAR
        parts.append(sc.tobytes())

    payload = b"".join(parts)
    return pack_header(
        MSG_FRAME_FULL,
        flags,
        frame_id,
        n_atoms,
        s_min,
        s_max,
        sim_time_ps,
        len(payload),
        reserved=int(scalar_channel),
    ) + payload


def pack_frame_tile_patch(
    atom_indices: np.ndarray,
    coords: np.ndarray,
    scalars: Optional[np.ndarray],
    box_H: Sequence[float] | np.ndarray,
    frame_id: int,
    sim_time_ps: float,
    use_int16: bool = True,
    scalar_channel: int = 0,
    tile_id: int = 0,
    pbc_wrapped: bool = True,
) -> bytes:
    """
    Pack a sparse ROI / Morton tile patch updating `K` specific atom slots in place:
      uint32[K] slot_indices
      int16[3*K] OR float32[3*K] coords
      [optional float32[K] scalars]
    (`header.n_atoms` = K, `header.reserved` encodes `(tile_id << 8) | (scalar_channel & 0xFF)`).
    """
    idx = np.ascontiguousarray(atom_indices, dtype=np.uint32).reshape((-1,))
    k = int(idx.shape[0])
    pts = np.ascontiguousarray(coords, dtype=np.float32).reshape((k, 3))
    H = _ensure_box_3x3(box_H)

    flags = FLAG_HAS_BOX | (FLAG_PBC_WRAPPED if pbc_wrapped else 0)
    parts = [H.tobytes(order="C"), idx.tobytes()]
    if use_int16:
        flags |= FLAG_INT16_FRACTIONAL
        q16_bytes = encode_int16_fractional(pts, H).tobytes(order="C") if k > 0 else b""
        parts.append(q16_bytes)
        # Strict 4-byte alignment padding before float32 scalars / packet end
        pad_len = (-len(q16_bytes)) % 4
        if pad_len:
            parts.append(b"\x00" * pad_len)
    else:
        parts.append(pts.tobytes(order="C") if k > 0 else b"")

    s_min, s_max = 0.0, 0.0
    if scalars is not None:
        sc = np.ascontiguousarray(scalars, dtype=np.float32).reshape((k,))
        finite = np.isfinite(sc)
        if np.any(finite):
            s_min = float(np.min(sc[finite]))
            s_max = float(np.max(sc[finite]))
        flags |= FLAG_HAS_SCALAR
        parts.append(sc.tobytes())

    payload = b"".join(parts)
    meta = ((int(tile_id) & 0xFFFFFF) << 8) | (int(scalar_channel) & 0xFF)
    return pack_header(MSG_FRAME_TILE_PATCH, flags, frame_id, k, s_min, s_max, sim_time_ps, len(payload), reserved=meta) + payload


def pack_dirty_tile_delta(
    dirty_indices: np.ndarray,
    coords_ang: np.ndarray,
    scalars: Optional[np.ndarray],
    box_H: Sequence[float] | np.ndarray,
    frame_id: int,
    sim_time_ps: float,
    n_dirty_tiles: int = 0,
    use_int16: bool = True,
    scalar_channel: int = 0,
    pbc_wrapped: bool = True,
) -> bytes:
    """
    Pack a multi-tile sparse delta patch for active displaced atoms.
    Accepts either full `(N, 3)` `coords_ang` (automatically sliced at `dirty_indices`)
    or pre-sliced `(K, 3)` coordinates, enforcing strict 4-byte alignment.
    """
    idx = np.ascontiguousarray(dirty_indices, dtype=np.uint32).reshape((-1,))
    k = int(idx.shape[0])
    pts = np.asarray(coords_ang, dtype=np.float32)
    if k == 0:
        sub_coords = np.zeros((0, 3), dtype=np.float32)
    elif pts.ndim == 2 and pts.shape[1] == 3 and pts.shape[0] > int(np.max(idx)):
        sub_coords = np.ascontiguousarray(pts[idx], dtype=np.float32)
    elif pts.ndim == 2 and pts.shape == (k, 3):
        sub_coords = np.ascontiguousarray(pts, dtype=np.float32)
    else:
        raise ValueError(f"Incompatible coords_ang shape {pts.shape} for {k} dirty indices")

    sub_scalars: Optional[np.ndarray] = None
    if scalars is not None:
        sc = np.asarray(scalars, dtype=np.float32).reshape((-1,))
        if k == 0:
            sub_scalars = np.zeros(0, dtype=np.float32)
        elif sc.shape[0] > int(np.max(idx)):
            sub_scalars = np.ascontiguousarray(sc[idx], dtype=np.float32)
        elif sc.shape[0] == k:
            sub_scalars = np.ascontiguousarray(sc, dtype=np.float32)
        else:
            raise ValueError(f"Incompatible scalars shape {sc.shape} for {k} dirty indices")

    return pack_frame_tile_patch(
        atom_indices=idx,
        coords=sub_coords,
        scalars=sub_scalars,
        box_H=box_H,
        frame_id=frame_id,
        sim_time_ps=sim_time_ps,
        use_int16=use_int16,
        scalar_channel=scalar_channel,
        tile_id=int(n_dirty_tiles),
        pbc_wrapped=pbc_wrapped,
    )


def pack_overlay_vectors(
    vector_starts: Optional[np.ndarray],
    vector_dirs: Optional[np.ndarray],
    vector_mags: Optional[np.ndarray],
    bond_pairs: Optional[np.ndarray],
    bond_orders: Optional[np.ndarray],
    frame_id: int,
    sim_time_ps: float,
) -> bytes:
    """
    Pack GPU-instanced overlay primitives:
      uint32 n_vectors
      uint32 n_bonds
      float32[3*n_vectors] starts (Angstrom)
      float32[3*n_vectors] dirs   (Angstrom)
      float32[n_vectors]   mags
      uint32[2*n_bonds]    bond_pairs
      float32[n_bonds]     bond_orders (0..2)
    """
    vs = np.zeros((0, 3), dtype=np.float32) if vector_starts is None else np.ascontiguousarray(vector_starts, dtype=np.float32).reshape((-1, 3))
    nv = int(vs.shape[0])
    vd = np.zeros((nv, 3), dtype=np.float32) if vector_dirs is None else np.ascontiguousarray(vector_dirs, dtype=np.float32).reshape((nv, 3))
    vm = np.ones(nv, dtype=np.float32) if vector_mags is None else np.ascontiguousarray(vector_mags, dtype=np.float32).reshape((nv,))

    bp = np.zeros((0, 2), dtype=np.uint32) if bond_pairs is None else np.ascontiguousarray(bond_pairs, dtype=np.uint32).reshape((-1, 2))
    nb = int(bp.shape[0])
    bo = np.ones(nb, dtype=np.float32) if bond_orders is None else np.ascontiguousarray(bond_orders, dtype=np.float32).reshape((nb,))

    payload = b"".join([
        struct.pack("<II", nv, nb),
        vs.tobytes(order="C"),
        vd.tobytes(order="C"),
        vm.tobytes(),
        bp.tobytes(order="C"),
        bo.tobytes(),
    ])
    v_max = float(np.max(vm)) if nv > 0 else 0.0
    return pack_header(MSG_OVERLAY_VECTORS, 0, frame_id, nv + nb, 0.0, v_max, sim_time_ps, len(payload)) + payload


def pack_volume_grid3d(
    grid: np.ndarray,
    box_H: Sequence[float] | np.ndarray,
    origin: Sequence[float] = (0.0, 0.0, 0.0),
    frame_id: int = 0,
    sim_time_ps: float = 0.0,
    channel_name_id: int = 0,
) -> bytes:
    """
    Pack a 3D scalar volumetric grid `(nx, ny, nz)` (e.g., TernaryMSM Phi or GCMC density) in C order:
      uint32[3] dims (nx, ny, nz)
      float32[3] origin (Angstrom)
      float32[9] cell_H (row-major, Angstrom)
      float32[nx*ny*nz] values
    """
    arr = np.ascontiguousarray(grid, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D volume grid, got shape {arr.shape}")
    nx, ny, nz = arr.shape
    H = _ensure_box_3x3(box_H)
    orig = np.ascontiguousarray(origin, dtype=np.float32).reshape((3,))
    v_min = float(np.min(arr)) if arr.size > 0 else 0.0
    v_max = float(np.max(arr)) if arr.size > 0 else 0.0
    payload = b"".join([
        struct.pack("<III", nx, ny, nz),
        orig.tobytes(),
        H.tobytes(order="C"),
        arr.tobytes(order="C"),
    ])
    return pack_header(
        MSG_VOLUME_GRID3D,
        FLAG_HAS_BOX,
        frame_id,
        nx * ny * nz,
        v_min,
        v_max,
        sim_time_ps,
        len(payload),
        reserved=int(channel_name_id),
    ) + payload


def pack_gcmc_slot_delta(
    inserted_slots: np.ndarray,
    inserted_atomic_numbers: np.ndarray,
    inserted_coords: np.ndarray,
    deleted_slots: np.ndarray,
    frame_id: int,
    sim_time_ps: float,
) -> bytes:
    """
    Pack variable-N GCMC slot pool updates without reallocating GPU textures:
      uint32 n_insert
      uint32 n_delete
      uint32[n_insert] inserted_slots
      uint8[n_insert]  inserted_atomic_numbers
      float32[3*n_insert] inserted_coords (Angstrom)
      uint32[n_delete] deleted_slots
    """
    ins_s = np.ascontiguousarray(inserted_slots, dtype=np.uint32).reshape((-1,))
    ni = int(ins_s.shape[0])
    ins_z = np.ascontiguousarray(inserted_atomic_numbers, dtype=np.uint8).reshape((ni,))
    ins_q = np.ascontiguousarray(inserted_coords, dtype=np.float32).reshape((ni, 3))
    del_s = np.ascontiguousarray(deleted_slots, dtype=np.uint32).reshape((-1,))
    nd = int(del_s.shape[0])
    payload = b"".join([
        struct.pack("<II", ni, nd),
        ins_s.tobytes(),
        ins_z.tobytes(),
        ins_q.tobytes(order="C"),
        del_s.tobytes(),
    ])
    return pack_header(MSG_GCMC_SLOT_DELTA, 0, frame_id, ni + nd, 0.0, 0.0, sim_time_ps, len(payload)) + payload


def unpack_frame(data: bytes) -> Dict[str, Any]:
    """Decode any TriumStream v1 binary packet (primarily for Python round-trip & QA verification)."""
    hdr = unpack_header(data)
    payload = memoryview(data)[HEADER_SIZE : HEADER_SIZE + hdr.payload_bytes]
    if len(payload) != hdr.payload_bytes:
        raise ValueError(f"Truncated payload: expected {hdr.payload_bytes}, got {len(payload)}")

    out: Dict[str, Any] = {"header": hdr}
    off = 0

    if hdr.msg_type == MSG_TOPOLOGY_INIT:
        (n_bonds,) = struct.unpack_from("<I", payload, off)
        off += 4
        H = np.frombuffer(payload, dtype=np.float32, count=9, offset=off).reshape((3, 3)).copy()
        off += 36
        z = np.frombuffer(payload, dtype=np.uint8, count=hdr.n_atoms, offset=off).copy()
        off += hdr.n_atoms
        bonds = np.frombuffer(payload, dtype=np.uint32, count=2 * n_bonds, offset=off).reshape((n_bonds, 2)).copy()
        off += 8 * n_bonds
        res_ids = np.frombuffer(payload, dtype=np.uint32, count=hdr.n_atoms, offset=off).copy()
        off += 4 * hdr.n_atoms
        tile_ids = np.frombuffer(payload, dtype=np.uint32, count=hdr.n_atoms, offset=off).copy()
        off += 4 * hdr.n_atoms
        radii = np.frombuffer(payload, dtype=np.float32, count=hdr.n_atoms, offset=off).copy()
        off += 4 * hdr.n_atoms
        coords = None
        if hdr.reserved == 1 and off + 12 * hdr.n_atoms <= len(payload):
            coords = np.frombuffer(payload, dtype=np.float32, count=3 * hdr.n_atoms, offset=off).reshape((hdr.n_atoms, 3)).copy()
        out.update({"box_H": H, "atomic_numbers": z, "bonds": bonds, "residue_ids": res_ids, "tile_ids": tile_ids, "radii": radii, "coords": coords})
        return out

    if hdr.msg_type == MSG_FRAME_FULL:
        H = None
        if hdr.flags & FLAG_HAS_BOX:
            H = np.frombuffer(payload, dtype=np.float32, count=9, offset=off).reshape((3, 3)).copy()
            off += 36
        if hdr.flags & FLAG_INT16_FRACTIONAL:
            q16 = np.frombuffer(payload, dtype=np.int16, count=3 * hdr.n_atoms, offset=off).reshape((hdr.n_atoms, 3))
            off += 6 * hdr.n_atoms
            coords = decode_int16_fractional(q16, H)
        else:
            coords = np.frombuffer(payload, dtype=np.float32, count=3 * hdr.n_atoms, offset=off).reshape((hdr.n_atoms, 3)).copy()
            off += 12 * hdr.n_atoms
        scalars = None
        if hdr.flags & FLAG_HAS_SCALAR:
            scalars = np.frombuffer(payload, dtype=np.float32, count=hdr.n_atoms, offset=off).copy()
        out.update({"box_H": H, "coords": coords, "scalars": scalars, "scalar_channel": hdr.reserved})
        return out

    if hdr.msg_type == MSG_FRAME_TILE_PATCH:
        H = np.frombuffer(payload, dtype=np.float32, count=9, offset=off).reshape((3, 3)).copy()
        off += 36
        idx = np.frombuffer(payload, dtype=np.uint32, count=hdr.n_atoms, offset=off).copy()
        off += 4 * hdr.n_atoms
        if hdr.flags & FLAG_INT16_FRACTIONAL:
            q16 = np.frombuffer(payload, dtype=np.int16, count=3 * hdr.n_atoms, offset=off).reshape((hdr.n_atoms, 3))
            off += 6 * hdr.n_atoms
            pad_len = (-6 * hdr.n_atoms) % 4
            if pad_len and off + pad_len <= len(payload):
                off += pad_len
            coords = decode_int16_fractional(q16, H) if hdr.n_atoms > 0 else np.zeros((0, 3), dtype=np.float32)
        else:
            coords = np.frombuffer(payload, dtype=np.float32, count=3 * hdr.n_atoms, offset=off).reshape((hdr.n_atoms, 3)).copy()
            off += 12 * hdr.n_atoms
        scalars = None
        if hdr.flags & FLAG_HAS_SCALAR:
            scalars = np.frombuffer(payload, dtype=np.float32, count=hdr.n_atoms, offset=off).copy()
        out.update({
            "box_H": H,
            "atom_indices": idx,
            "coords": coords,
            "scalars": scalars,
            "tile_id": hdr.reserved >> 8,
            "scalar_channel": hdr.reserved & 0xFF,
        })
        return out

    if hdr.msg_type == MSG_OVERLAY_VECTORS:
        nv, nb = struct.unpack_from("<II", payload, off)
        off += 8
        vs = np.frombuffer(payload, dtype=np.float32, count=3 * nv, offset=off).reshape((nv, 3)).copy()
        off += 12 * nv
        vd = np.frombuffer(payload, dtype=np.float32, count=3 * nv, offset=off).reshape((nv, 3)).copy()
        off += 12 * nv
        vm = np.frombuffer(payload, dtype=np.float32, count=nv, offset=off).copy()
        off += 4 * nv
        bp = np.frombuffer(payload, dtype=np.uint32, count=2 * nb, offset=off).reshape((nb, 2)).copy()
        off += 8 * nb
        bo = np.frombuffer(payload, dtype=np.float32, count=nb, offset=off).copy()
        out.update({"vector_starts": vs, "vector_dirs": vd, "vector_mags": vm, "bond_pairs": bp, "bond_orders": bo})
        return out

    if hdr.msg_type == MSG_VOLUME_GRID3D:
        nx, ny, nz = struct.unpack_from("<III", payload, off)
        off += 12
        origin = np.frombuffer(payload, dtype=np.float32, count=3, offset=off).copy()
        off += 12
        H = np.frombuffer(payload, dtype=np.float32, count=9, offset=off).reshape((3, 3)).copy()
        off += 36
        grid = np.frombuffer(payload, dtype=np.float32, count=nx * ny * nz, offset=off).reshape((nx, ny, nz)).copy()
        out.update({"dims": (nx, ny, nz), "origin": origin, "box_H": H, "grid": grid, "channel_name_id": hdr.reserved})
        return out

    if hdr.msg_type == MSG_GCMC_SLOT_DELTA:
        ni, nd = struct.unpack_from("<II", payload, off)
        off += 8
        ins_s = np.frombuffer(payload, dtype=np.uint32, count=ni, offset=off).copy()
        off += 4 * ni
        ins_z = np.frombuffer(payload, dtype=np.uint8, count=ni, offset=off).copy()
        off += ni
        ins_q = np.frombuffer(payload, dtype=np.float32, count=3 * ni, offset=off).reshape((ni, 3)).copy()
        off += 12 * ni
        del_s = np.frombuffer(payload, dtype=np.uint32, count=nd, offset=off).copy()
        out.update({"inserted_slots": ins_s, "inserted_atomic_numbers": ins_z, "inserted_coords": ins_q, "deleted_slots": del_s})
        return out

    raise ValueError(f"Unsupported TriumStream msg_type: {hdr.msg_type:#x}")
