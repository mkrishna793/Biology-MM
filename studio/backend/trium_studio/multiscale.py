"""Multiscale assembly expansion, soft-core clash relaxation, zone partitioning, and dirty-tile tracking."""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

TSYM_MAGIC = b"TSYM"

# Strictly 48-byte little-endian header:
#   magic (4s), n_asu_atoms (I), n_transforms (I), title_len (I),
#   temperature (f), dt (f), mode_len (I), flags (I),
#   bx (f), by (f), bz (f), reserved (I)
TSYM_HEADER_STRUCT = struct.Struct("<4sIIIffIIfffI")
TSYM_HEADER_SIZE = TSYM_HEADER_STRUCT.size  # 48 bytes (8-byte & 4-byte aligned)

TSYM_FLAG_HAS_ELEMENTS = 0x01

_Z_TO_ELEMENT: Dict[int, str] = {
    1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 10: "Ne",
    11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P", 16: "S", 17: "Cl", 18: "Ar", 19: "K", 20: "Ca",
    21: "Sc", 22: "Ti", 23: "V", 24: "Cr", 25: "Mn", 26: "Fe", 27: "Co", 28: "Ni", 29: "Cu", 30: "Zn",
    31: "Ga", 32: "Ge", 33: "As", 34: "Se", 35: "Br", 36: "Kr", 37: "Rb", 38: "Sr", 39: "Y", 40: "Zr",
    41: "Nb", 42: "Mo", 44: "Ru", 45: "Rh", 46: "Pd", 47: "Ag", 48: "Cd", 49: "In", 50: "Sn", 51: "Sb",
    52: "Te", 53: "I", 54: "Xe", 55: "Cs", 56: "Ba", 57: "La", 58: "Ce", 60: "Nd", 63: "Eu", 64: "Gd",
    72: "Hf", 73: "Ta", 74: "W", 75: "Re", 76: "Os", 77: "Ir", 78: "Pt", 79: "Au", 80: "Hg", 81: "Tl",
    82: "Pb", 83: "Bi", 90: "Th", 92: "U",
}

_ELEMENT_TO_Z: Dict[str, int] = {sym.upper(): z for z, sym in _Z_TO_ELEMENT.items()}

_HALF_SHELL_OFFSETS: Tuple[Tuple[int, int, int], ...] = (
    (1, 0, 0),
    (1, 1, 0),
    (0, 1, 0),
    (-1, 1, 0),
    (-1, -1, 1),
    (0, -1, 1),
    (1, -1, 1),
    (-1, 0, 1),
    (0, 0, 1),
    (1, 0, 1),
    (-1, 1, 1),
    (0, 1, 1),
    (1, 1, 1),
)


@dataclass
class SymmetryAssemblyPayload:
    """Decoded `TSYM` biological symmetry assembly payload."""

    n_asu_atoms: int
    n_transforms: int
    asu_coords_ang: np.ndarray  # (M, 3) float32
    transforms_4x4: np.ndarray  # (K, 4, 4) float32 row-major affine transforms
    asu_elements: List[str]
    temperature: float
    dt: float
    flags: int
    box_ang: Tuple[float, float, float]
    title: str
    mode: str


def _decode_element_byte(b: int) -> str:
    """Decode a uint8 element byte supporting ASCII letters, atomic numbers, and high-bit atomic numbers (b >= 128)."""
    b_int = int(b) & 0xFF
    if b_int >= 128:
        return _Z_TO_ELEMENT.get(b_int - 128, "C")
    if (65 <= b_int <= 90) or (97 <= b_int <= 122):
        return chr(b_int).upper()
    return _Z_TO_ELEMENT.get(b_int, "C")


def _encode_element_byte(elem: Union[str, int]) -> int:
    """Encode an element symbol or atomic number into a uint8 byte without colliding with ASCII letters."""
    if isinstance(elem, (int, np.integer)):
        z_val = int(elem) & 0xFF
        return (z_val | 0x80) if 65 <= z_val <= 122 else z_val
    cleaned = "".join(ch for ch in str(elem).strip() if ch.isalpha())
    if not cleaned:
        return ord("C")
    if len(cleaned) == 1:
        return ord(cleaned.upper())
    z = _ELEMENT_TO_Z.get(cleaned.upper())
    if z is None:
        return ord("C")
    return (z | 0x80) if z >= 65 else z


def pack_tsym_binary(
    asu_coords_ang: np.ndarray,
    transforms_4x4: np.ndarray,
    asu_elements: Optional[Sequence[Union[str, int]]] = None,
    title: str = "Symmetry Assembly",
    mode: str = "metal_verlet",
    temperature: float = 300.0,
    dt: float = 0.002,
    box_ang: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    flags: int = 0,
    reserved: int = 0,
) -> bytes:
    """
    Pack a strictly 4-byte-aligned `TSYM` binary packet:
      [0..48)                        48-byte header (<4sIIIffIIfffI)
      [48 .. 48+12*M)                float32[3 * M] asu_coords_ang
      [48+12*M .. 48+12*M+64*K)      float32[16 * K] transforms_4x4 (row-major)
      [optional]                     uint8[M] asu_elements (if flags & 0x01)
      [trailing]                     UTF-8 title, UTF-8 mode
    """
    coords = np.ascontiguousarray(asu_coords_ang, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"Expected asu_coords_ang shape (M, 3), got {coords.shape}")
    m = int(coords.shape[0])

    tfs = np.ascontiguousarray(transforms_4x4, dtype=np.float32)
    if tfs.ndim == 2 and tfs.shape == (4, 4):
        tfs = tfs.reshape((1, 4, 4))
    elif tfs.ndim != 3 or tfs.shape[1:] != (4, 4):
        raise ValueError(f"Expected transforms_4x4 shape (K, 4, 4), got {tfs.shape}")
    k = int(tfs.shape[0])

    hdr_flags = int(flags)
    elem_bytes = b""
    if asu_elements is not None:
        if len(asu_elements) != m:
            raise ValueError(f"asu_elements length ({len(asu_elements)}) must match n_asu_atoms ({m})")
        hdr_flags |= TSYM_FLAG_HAS_ELEMENTS
        elem_bytes = bytes(_encode_element_byte(e) for e in asu_elements)

    title_bytes = title.encode("utf-8") if title else b""
    mode_bytes = mode.encode("utf-8") if mode else b""
    bx, by, bz = (float(box_ang[0]), float(box_ang[1]), float(box_ang[2]))

    header = TSYM_HEADER_STRUCT.pack(
        TSYM_MAGIC,
        m,
        k,
        len(title_bytes),
        float(temperature),
        float(dt),
        len(mode_bytes),
        hdr_flags,
        bx,
        by,
        bz,
        int(reserved),
    )
    return b"".join([header, coords.tobytes(order="C"), tfs.tobytes(order="C"), elem_bytes, title_bytes, mode_bytes])


def unpack_tsym_binary(body: bytes | memoryview) -> SymmetryAssemblyPayload:
    """
    Unpack a strictly 4-byte-aligned `TSYM` binary packet.

    All float32 payloads (`asu_coords_ang` and `transforms_4x4`) begin at byte offset 48
    (8-byte aligned) and precede all variable-length uint8 / UTF-8 sections, guaranteeing
    4-byte alignment regardless of odd atom counts or odd-length title/mode strings.
    """
    if len(body) < TSYM_HEADER_SIZE:
        raise ValueError(f"TSYM payload too short: expected >= {TSYM_HEADER_SIZE} bytes, got {len(body)}")

    (
        magic,
        n_asu_atoms,
        n_transforms,
        title_len,
        temperature,
        dt,
        mode_len,
        flags,
        bx,
        by,
        bz,
        _reserved,
    ) = TSYM_HEADER_STRUCT.unpack_from(body, 0)

    if magic != TSYM_MAGIC:
        raise ValueError(f"Invalid TSYM magic: {magic!r} (expected {TSYM_MAGIC!r})")

    offset = TSYM_HEADER_SIZE
    coords_bytes = int(n_asu_atoms) * 3 * 4
    transforms_bytes = int(n_transforms) * 16 * 4
    has_elements = bool(flags & TSYM_FLAG_HAS_ELEMENTS)
    elem_bytes_len = int(n_asu_atoms) if has_elements else 0
    expected_min = offset + coords_bytes + transforms_bytes + elem_bytes_len + int(title_len) + int(mode_len)
    if len(body) < expected_min:
        raise ValueError(f"Truncated TSYM payload: expected >= {expected_min} bytes, got {len(body)}")

    # Strictly 4-byte aligned reads (offset == 48, coords_bytes % 4 == 0, transforms_bytes % 4 == 0)
    asu_coords = np.frombuffer(body, dtype=np.float32, count=int(n_asu_atoms) * 3, offset=offset).reshape(
        (int(n_asu_atoms), 3)
    ).copy()
    offset += coords_bytes

    transforms = np.frombuffer(body, dtype=np.float32, count=int(n_transforms) * 16, offset=offset).reshape(
        (int(n_transforms), 4, 4)
    ).copy()
    offset += transforms_bytes

    if has_elements:
        raw_elems = bytes(body[offset : offset + elem_bytes_len])
        asu_elements = [_decode_element_byte(b) for b in raw_elems]
        offset += elem_bytes_len
    else:
        asu_elements = ["C"] * int(n_asu_atoms)

    raw_title = bytes(body[offset : offset + int(title_len)])
    title = raw_title.decode("utf-8", errors="ignore") if title_len > 0 else "Symmetry Assembly"
    offset += int(title_len)

    raw_mode = bytes(body[offset : offset + int(mode_len)])
    mode = raw_mode.decode("utf-8", errors="ignore") if mode_len > 0 else "metal_verlet"

    return SymmetryAssemblyPayload(
        n_asu_atoms=int(n_asu_atoms),
        n_transforms=int(n_transforms),
        asu_coords_ang=asu_coords,
        transforms_4x4=transforms,
        asu_elements=asu_elements,
        temperature=float(temperature),
        dt=float(dt),
        flags=int(flags),
        box_ang=(float(bx), float(by), float(bz)),
        title=title,
        mode=mode,
    )


def expand_symmetry_assembly(
    asu_coords_ang: np.ndarray,
    asu_elements: Optional[Sequence[Any]],
    transforms_4x4: np.ndarray,
) -> Tuple[np.ndarray, List[str]]:
    """
    Vectorized NumPy expansion of an asymmetric unit `(M, 3)` by `K` 4x4 row-major
    affine transforms `(K, 4, 4)` into `(K*M, 3)` coordinates and `K*M` elements.
    """
    coords = np.asarray(asu_coords_ang)
    out_dtype = np.float32 if coords.dtype == np.float32 else np.float64
    coords_f64 = np.asarray(coords, dtype=np.float64)
    if coords_f64.ndim != 2 or coords_f64.shape[1] != 3:
        raise ValueError(f"Expected asu_coords_ang shape (M, 3), got {coords_f64.shape}")
    m = int(coords_f64.shape[0])

    tfs = np.asarray(transforms_4x4, dtype=np.float64)
    if tfs.ndim == 2 and tfs.shape == (4, 4):
        tfs = tfs.reshape((1, 4, 4))
    elif tfs.ndim != 3 or tfs.shape[1:] != (4, 4):
        raise ValueError(f"Expected transforms_4x4 shape (K, 4, 4), got {tfs.shape}")
    k = int(tfs.shape[0])

    if m == 0 or k == 0:
        return np.zeros((0, 3), dtype=out_dtype), []

    rot = tfs[:, :3, :3]  # (K, 3, 3)
    trans = tfs[:, :3, 3]  # (K, 3)
    # Row-major affine transform: x' = R @ x + t
    expanded = np.einsum("kij,mj->kmi", rot, coords_f64) + trans[:, None, :]
    expanded_flat = np.ascontiguousarray(expanded.reshape((k * m, 3)), dtype=out_dtype)

    if asu_elements is None:
        base_elems = ["C"] * m
    else:
        if len(asu_elements) != m:
            raise ValueError(f"asu_elements length ({len(asu_elements)}) must match n_asu_atoms ({m})")
        base_elems = [str(e) for e in asu_elements]

    expanded_elements = base_elems * k
    return expanded_flat, expanded_elements


def _as_box_lengths(box_nm: Union[Sequence[float], np.ndarray]) -> np.ndarray:
    arr = np.asarray(box_nm, dtype=np.float64)
    if arr.shape == (3,):
        return arr.copy()
    if arr.shape == (3, 3):
        return np.diag(arr).astype(np.float64).copy()
    raise ValueError(f"Expected box_nm shape (3,) or (3, 3), got {arr.shape}")


def _find_neighbor_pairs_mic(
    positions_nm: np.ndarray,
    box_L: np.ndarray,
    rcut: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Find all unique periodic minimum-image neighbor pairs `(i < j)` with `r < rcut`
    using O(N) vectorized cell-linked lists when the box accommodates >= 3 cells per axis.
    """
    n = int(positions_nm.shape[0])
    if n < 2:
        empty_i = np.zeros(0, dtype=np.int32)
        return empty_i, empty_i, np.zeros((0, 3), dtype=np.float64), np.zeros(0, dtype=np.float64)

    inv_L = 1.0 / np.maximum(box_L, 1e-9)
    if n > 128:
        try:
            from scipy.spatial import cKDTree

            pts_w = np.mod(positions_nm, box_L[None, :])
            pts_w = np.where((pts_w >= box_L[None, :]) | (pts_w < 0.0), 0.0, pts_w)
            tree = cKDTree(pts_w, boxsize=box_L)
            pairs = tree.query_pairs(r=float(rcut), output_type="ndarray")
            if pairs.size == 0:
                empty_i = np.zeros(0, dtype=np.int32)
                return empty_i, empty_i, np.zeros((0, 3), dtype=np.float64), np.zeros(0, dtype=np.float64)
            ii = pairs[:, 0].astype(np.int32, copy=False)
            jj = pairs[:, 1].astype(np.int32, copy=False)
            d_sel = positions_nm[ii] - positions_nm[jj]
            d_sel -= box_L[None, :] * np.rint(d_sel * inv_L[None, :])
            r_sel = np.sqrt(np.maximum(np.sum(d_sel * d_sel, axis=1), 0.0))
            return ii, jj, d_sel, r_sel
        except Exception:
            pass

    nx = int(math.floor(float(box_L[0]) / max(rcut, 1e-6)))
    ny = int(math.floor(float(box_L[1]) / max(rcut, 1e-6)))
    nz = int(math.floor(float(box_L[2]) / max(rcut, 1e-6)))

    if n > 256 and min(nx, ny, nz) >= 3:
        nx = min(nx, 64)
        ny = min(ny, 64)
        nz = min(nz, 64)
        frac = positions_nm * inv_L[None, :]
        frac -= np.floor(frac)
        cx = np.clip((frac[:, 0] * nx).astype(np.int32), 0, nx - 1)
        cy = np.clip((frac[:, 1] * ny).astype(np.int32), 0, ny - 1)
        cz = np.clip((frac[:, 2] * nz).astype(np.int32), 0, nz - 1)
        cid = cx + nx * (cy + ny * cz)

        order = np.argsort(cid, kind="mergesort")
        cid_sorted = cid[order]
        unique_cids, starts, counts = np.unique(cid_sorted, return_index=True, return_counts=True)

        n_cells = nx * ny * nz
        cell_start = np.zeros(n_cells, dtype=np.int32)
        cell_count = np.zeros(n_cells, dtype=np.int32)
        cell_start[unique_cids] = starts.astype(np.int32)
        cell_count[unique_cids] = counts.astype(np.int32)

        ii_parts: List[np.ndarray] = []
        jj_parts: List[np.ndarray] = []

        # Intra-cell pairs
        multi_mask = counts > 1
        for u_c, s0, cnt in zip(unique_cids[multi_mask], starts[multi_mask], counts[multi_mask]):
            atoms = order[s0 : s0 + cnt]
            a_i, a_j = np.triu_indices(int(cnt), k=1)
            ii_parts.append(atoms[a_i].astype(np.int32))
            jj_parts.append(atoms[a_j].astype(np.int32))

        # Cross-cell half-shell 13-neighbor pairs
        ucx = unique_cids % nx
        ucy = (unique_cids // nx) % ny
        ucz = unique_cids // (nx * ny)
        for dx, dy, dz in _HALF_SHELL_OFFSETS:
            ncx = (ucx + dx) % nx
            ncy = (ucy + dy) % ny
            ncz = (ucz + dz) % nz
            nb_cid = ncx + nx * (ncy + ny * ncz)
            nb_cnt = cell_count[nb_cid]
            valid = nb_cnt > 0
            if not np.any(valid):
                continue
            for s0, c0, nb_s0, nb_c0 in zip(
                starts[valid], counts[valid], cell_start[nb_cid[valid]], nb_cnt[valid]
            ):
                a0 = order[s0 : s0 + c0]
                a1 = order[nb_s0 : nb_s0 + nb_c0]
                ii_parts.append(np.repeat(a0, int(nb_c0)).astype(np.int32))
                jj_parts.append(np.tile(a1, int(c0)).astype(np.int32))

        if not ii_parts:
            empty_i = np.zeros(0, dtype=np.int32)
            return empty_i, empty_i, np.zeros((0, 3), dtype=np.float64), np.zeros(0, dtype=np.float64)

        cand_i = np.concatenate(ii_parts)
        cand_j = np.concatenate(jj_parts)
        lo = np.minimum(cand_i, cand_j)
        hi = np.maximum(cand_i, cand_j)
        cand_i, cand_j = lo, hi
    else:
        cand_i, cand_j = np.triu_indices(n, k=1)
        cand_i = cand_i.astype(np.int32)
        cand_j = cand_j.astype(np.int32)

    d = positions_nm[cand_i] - positions_nm[cand_j]
    d -= box_L[None, :] * np.rint(d * inv_L[None, :])
    r2 = np.sum(d * d, axis=1)
    mask = r2 < (rcut * rcut)
    ii = cand_i[mask]
    jj = cand_j[mask]
    d_sel = d[mask]
    r_sel = np.sqrt(np.maximum(r2[mask], 0.0))
    return ii, jj, d_sel, r_sel


def _eval_softcore_lj(
    positions_nm: np.ndarray,
    box_L: np.ndarray,
    sigma_arr: np.ndarray,
    epsilon_arr: np.ndarray,
    rcut: float,
    alpha_sc: float,
    lam: float,
    exclusions: Optional[Sequence[Tuple[int, int]]] = None,
) -> Tuple[float, np.ndarray, int]:
    """
    Evaluate Beutler soft-core Lennard-Jones energy & forces with a smooth core push
    for severe steric overlaps (`r < 0.7 * sigma_ij`) so even `r ~ 0` clashes relax rapidly
    while remaining completely finite.
    """
    n = int(positions_nm.shape[0])
    forces = np.zeros((n, 3), dtype=np.float64)
    if n < 2:
        return 0.0, forces, 0

    ii, jj, d_ij, r = _find_neighbor_pairs_mic(positions_nm, box_L, rcut)
    if ii.size == 0:
        return 0.0, forces, 0

    if exclusions:
        excl_arr = np.asarray(exclusions, dtype=np.int64)
        lo = np.minimum(excl_arr[:, 0], excl_arr[:, 1])
        hi = np.maximum(excl_arr[:, 0], excl_arr[:, 1])
        excl_keys = lo * n + hi
        pair_keys = ii.astype(np.int64) * n + jj.astype(np.int64)
        keep = ~np.isin(pair_keys, excl_keys)
        ii, jj, d_ij, r = ii[keep], jj[keep], d_ij[keep], r[keep]
        if ii.size == 0:
            return 0.0, forces, 0

    sig_ij = 0.5 * (sigma_arr[ii] + sigma_arr[jj])
    eps_ij = np.sqrt(np.maximum(epsilon_arr[ii] * epsilon_arr[jj], 0.0))
    n_clashes = int(np.sum(r < 0.7 * sig_ij))

    # Symmetry-breaking unit direction for exact r ~ 0 overlaps
    tiny = r < 1e-6
    if np.any(tiny):
        idx_tiny = np.flatnonzero(tiny)
        angles = (ii[idx_tiny].astype(np.float64) * 1.6180339887 + jj[idx_tiny].astype(np.float64) * 2.718281828)
        unit_nudge = np.stack([np.cos(angles), np.sin(angles), np.cos(2.0 * angles)], axis=1)
        unit_nudge /= np.maximum(np.linalg.norm(unit_nudge, axis=1, keepdims=True), 1e-12)
        d_ij = d_ij.copy()
        r = r.copy()
        d_ij[idx_tiny] = unit_nudge * 1e-5
        r[idx_tiny] = 1e-5

    sig6 = sig_ij ** 6
    a_sc = float(alpha_sc) * ((1.0 - float(lam)) ** 2) * sig6
    r2 = r * r
    r4 = r2 * r2
    r6 = r4 * r2
    u_den = a_sc + r6
    inv_u = 1.0 / np.maximum(u_den, 1e-30)
    s6_over_u = sig6 * inv_u

    # Beutler soft-core LJ energy & shifted cutoff
    u_rc_den = a_sc + (rcut ** 6)
    s6_rc = sig6 / np.maximum(u_rc_den, 1e-30)
    u_sc = 4.0 * float(lam) * eps_ij * ((s6_over_u * s6_over_u - s6_over_u) - (s6_rc * s6_rc - s6_rc))
    # Force along d_ij: F_ij = -(dV/dr) * (d_ij / r)
    coeff_sc = 24.0 * float(lam) * eps_ij * r4 * sig6 * (2.0 * sig6 - u_den) * (inv_u ** 3)

    # Smooth quadratic core repulsion active only inside r < 0.7 * sigma_ij
    # (prevents vanishing Beutler gradient near r -> 0 while staying strictly finite)
    r_core = 0.7 * sig_ij
    in_core = r < r_core
    u_core = np.zeros_like(u_sc)
    coeff_core = np.zeros_like(coeff_sc)
    if np.any(in_core):
        dr_core = r_core[in_core] - r[in_core]
        k_core = 5000.0
        u_core[in_core] = 0.5 * k_core * dr_core * dr_core
        coeff_core[in_core] = (k_core * dr_core) / np.maximum(r[in_core], 1e-6)

    total_energy = float(np.sum(u_sc + u_core))
    f_ij = d_ij * (coeff_sc + coeff_core)[:, None]
    for dim in range(3):
        forces[:, dim] = np.bincount(ii, weights=f_ij[:, dim], minlength=n) - np.bincount(
            jj, weights=f_ij[:, dim], minlength=n
        )
    return total_energy, forces, n_clashes


def softcore_relax_clashes(
    positions_nm: np.ndarray,
    box_nm: Union[Sequence[float], np.ndarray],
    sigma_nm: Union[float, np.ndarray],
    epsilon_kj: Union[float, np.ndarray],
    exclusions: Optional[Sequence[Tuple[int, int]]] = None,
    max_steps: int = 40,
    max_force_cap: float = 5000.0,
    alpha_sc: float = 0.5,
    lam: float = 0.5,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Vectorized cell-linked-list Beutler soft-core LJ clash minimizer that safely relaxes
    severe Cryo-EM/PDB overlaps via capped steepest-descent displacements.
    """
    pos = np.array(positions_nm, dtype=np.float64, copy=True)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"Expected positions_nm shape (N, 3), got {pos.shape}")
    n = int(pos.shape[0])
    box_L = _as_box_lengths(box_nm)

    if n < 2:
        return pos, {
            "initial_max_force": 0.0,
            "final_max_force": 0.0,
            "initial_energy": 0.0,
            "final_energy": 0.0,
            "initial_clashes": 0,
            "final_clashes": 0,
            "clashes_resolved": 0,
            "steps_run": 0,
            "status": "clean",
        }

    sigma_arr = (
        np.full(n, float(sigma_nm), dtype=np.float64)
        if np.ndim(sigma_nm) == 0
        else np.asarray(sigma_nm, dtype=np.float64).reshape((n,))
    )
    eps_arr = (
        np.full(n, float(epsilon_kj), dtype=np.float64)
        if np.ndim(epsilon_kj) == 0
        else np.asarray(epsilon_kj, dtype=np.float64).reshape((n,))
    )

    max_sig = float(np.max(sigma_arr)) if n > 0 else 0.34
    min_box = float(np.min(box_L))
    rcut = min(max(2.5 * max_sig, 0.8), 0.45 * min_box)

    # Wrap into [0, L)
    inv_L = 1.0 / np.maximum(box_L, 1e-9)
    pos = (pos * inv_L[None, :] - np.floor(pos * inv_L[None, :])) * box_L[None, :]

    # Build 1-2 bond spring pairs and 1-2/1-3 exclusion list so covalent bonds are never blown apart
    bond_i = np.zeros(0, dtype=np.int32)
    bond_j = np.zeros(0, dtype=np.int32)
    bond_r0 = np.zeros(0, dtype=np.float64)
    full_exclusions: List[Tuple[int, int]] = list(exclusions) if exclusions else []

    if not full_exclusions and n <= 8000:
        # Auto-detect initial covalent/coordination contacts (0.04 nm < r <= 0.245 nm)
        ii0, jj0, _, r0_init = _find_neighbor_pairs_mic(pos, box_L, min(0.245, rcut))
        cov_mask = (r0_init > 0.04) & (r0_init <= 0.245)
        if np.any(cov_mask):
            bond_i = ii0[cov_mask].astype(np.int32)
            bond_j = jj0[cov_mask].astype(np.int32)
            bond_r0 = r0_init[cov_mask].astype(np.float64)
            full_exclusions = list(zip(bond_i.tolist(), bond_j.tolist()))
    elif full_exclusions:
        ex_arr = np.asarray(full_exclusions, dtype=np.int32).reshape((-1, 2))
        d_ex = pos[ex_arr[:, 0]] - pos[ex_arr[:, 1]]
        d_ex -= box_L[None, :] * np.rint(d_ex * inv_L[None, :])
        r_ex = np.sqrt(np.maximum(np.sum(d_ex * d_ex, axis=1), 0.0))
        cov_mask = (r_ex > 0.04) & (r_ex <= 0.33)
        if np.any(cov_mask):
            bond_i = ex_arr[cov_mask, 0].astype(np.int32)
            bond_j = ex_arr[cov_mask, 1].astype(np.int32)
            bond_r0 = r_ex[cov_mask].astype(np.float64)

    # Expand exclusions to include 1-3 angle pairs and 1-4 ring pairs of detected covalent bonds
    if bond_i.size > 0 and n <= 25000:
        from collections import defaultdict
        adj_b: Dict[int, List[int]] = defaultdict(list)
        excl_set = set((min(int(u), int(v)), max(int(u), int(v))) for u, v in full_exclusions if int(u) != int(v))
        for u, v in zip(bond_i.tolist(), bond_j.tolist()):
            adj_b[int(u)].append(int(v))
            adj_b[int(v)].append(int(u))
            excl_set.add((min(int(u), int(v)), max(int(u), int(v))))
        for center, nbrs in adj_b.items():
            ln = len(nbrs)
            for a1 in range(ln):
                u1 = nbrs[a1]
                for a2 in range(a1 + 1, ln):
                    u2 = nbrs[a2]
                    if u1 != u2:
                        excl_set.add((min(u1, u2), max(u1, u2)))
                # 1-4 pairs across aromatic rings / small clusters
                if n <= 4000:
                    for w in adj_b.get(u1, ()):
                        if w != center:
                            for a2 in range(ln):
                                u2 = nbrs[a2]
                                if u2 != u1 and u2 != w:
                                    excl_set.add((min(u2, w), max(u2, w)))
        full_exclusions = sorted(list(excl_set))

    def _eval_with_bonds(q_pos: np.ndarray) -> Tuple[float, np.ndarray, int]:
        e_lj, f_lj, n_cl = _eval_softcore_lj(
            q_pos, box_L, sigma_arr, eps_arr, rcut, alpha_sc, lam, exclusions=full_exclusions or None
        )
        if bond_i.size > 0:
            db = q_pos[bond_i] - q_pos[bond_j]
            db -= box_L[None, :] * np.rint(db * inv_L[None, :])
            rb = np.sqrt(np.maximum(np.sum(db * db, axis=1), 1e-18))
            dr_b = rb - bond_r0
            k_b = 12000.0
            e_lj += 0.5 * k_b * float(np.sum(dr_b * dr_b))
            fb = db * (-(k_b * dr_b) / rb)[:, None]
            for dim in range(3):
                f_lj[:, dim] += np.bincount(bond_i, weights=fb[:, dim], minlength=n) - np.bincount(
                    bond_j, weights=fb[:, dim], minlength=n
                )
        return e_lj, f_lj, n_cl

    e_init, f_init, initial_clashes = _eval_with_bonds(pos)
    fmax_init = float(np.max(np.linalg.norm(f_init, axis=1)))

    if initial_clashes == 0 and fmax_init < 100.0:
        return pos, {
            "initial_max_force": fmax_init,
            "final_max_force": fmax_init,
            "initial_energy": e_init,
            "final_energy": e_init,
            "initial_clashes": 0,
            "final_clashes": 0,
            "clashes_resolved": 0,
            "steps_run": 0,
            "status": "clean",
        }

    e_curr = e_init
    f_curr = f_init
    clashes_curr = initial_clashes
    step_size = 0.025  # nm initial max displacement per step
    max_step_disp = 0.06
    steps_run = 0

    for step in range(1, max(1, int(max_steps)) + 1):
        f_mag = np.linalg.norm(f_curr, axis=1)
        f_max = float(np.max(f_mag))
        if clashes_curr == 0 and f_max < 25.0:
            break

        # Cap per-atom forces cleanly before computing steepest-descent move
        cap_scale = np.minimum(1.0, float(max_force_cap) / np.maximum(f_mag, 1e-12))
        f_capped = f_curr * cap_scale[:, None]
        fc_max = float(np.max(np.linalg.norm(f_capped, axis=1)))
        if fc_max < 1e-12:
            break

        disp = f_capped * (step_size / fc_max)
        cand = pos + disp
        cand = (cand * inv_L[None, :] - np.floor(cand * inv_L[None, :])) * box_L[None, :]
        e_new, f_new, clashes_new = _eval_with_bonds(cand)
        steps_run = step

        if np.isfinite(e_new) and e_new <= e_curr:
            pos = cand
            e_curr = e_new
            f_curr = f_new
            clashes_curr = clashes_new
            step_size = min(max_step_disp, step_size * 1.2)
        else:
            step_size *= 0.5
            if step_size < 1e-5:
                break

    fmax_final = float(np.max(np.linalg.norm(f_curr, axis=1)))
    clashes_resolved = max(0, int(initial_clashes - clashes_curr))
    status = "relaxed" if (steps_run > 0 and (e_curr < e_init or clashes_resolved > 0)) else "clean"

    return pos, {
        "initial_max_force": fmax_init,
        "final_max_force": fmax_final,
        "initial_energy": e_init,
        "final_energy": e_curr,
        "initial_clashes": int(initial_clashes),
        "final_clashes": int(clashes_curr),
        "clashes_resolved": clashes_resolved,
        "steps_run": int(steps_run),
        "status": status,
    }


@dataclass
class MultiscaleZonePartition:
    """3-zone spatial multiscale partition: Zone 0 (All-Atom), Zone 1 (CG), Zone 2 (Rigid)."""

    zone_ids: np.ndarray  # uint8[N]: 0=all_atom, 1=coarse_grained, 2=rigid_scaffold
    center_ang: Tuple[float, float, float]
    aa_radius_ang: float
    cg_radius_ang: float
    n_all_atom: int
    n_coarse_grained: int
    n_rigid_scaffold: int
    estimated_memory_mb: float

    def build_multiscale_summary(self) -> Dict[str, Any]:
        return {
            "center_ang": [float(c) for c in self.center_ang],
            "aa_radius_ang": float(self.aa_radius_ang),
            "cg_radius_ang": float(self.cg_radius_ang),
            "n_all_atom": int(self.n_all_atom),
            "n_coarse_grained": int(self.n_coarse_grained),
            "n_rigid_scaffold": int(self.n_rigid_scaffold),
            "all_atom": int(self.n_all_atom),
            "coarse_grained": int(self.n_coarse_grained),
            "rigid_scaffold": int(self.n_rigid_scaffold),
            "n_total": int(self.n_all_atom + self.n_coarse_grained + self.n_rigid_scaffold),
            "estimated_memory_mb": round(float(self.estimated_memory_mb), 4),
        }


def assign_multiscale_zones(
    positions_ang: np.ndarray,
    center_ang: Sequence[float],
    aa_radius_ang: float,
    cg_radius_ang: float,
) -> MultiscaleZonePartition:
    """
    Partition atoms into 3 concentric multiscale zones around `center_ang`:
      - Zone 0 (`all_atom`):        r <= aa_radius_ang
      - Zone 1 (`coarse_grained`):  aa_radius_ang < r <= cg_radius_ang
      - Zone 2 (`rigid_scaffold`):  r > cg_radius_ang
    """
    pts = np.asarray(positions_ang, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"Expected positions_ang shape (N, 3), got {pts.shape}")
    ctr = np.asarray(center_ang, dtype=np.float64).reshape((3,))
    r_aa = max(0.0, float(aa_radius_ang))
    r_cg = max(r_aa, float(cg_radius_ang))

    n = int(pts.shape[0])
    if n == 0:
        return MultiscaleZonePartition(
            zone_ids=np.zeros(0, dtype=np.uint8),
            center_ang=(float(ctr[0]), float(ctr[1]), float(ctr[2])),
            aa_radius_ang=r_aa,
            cg_radius_ang=r_cg,
            n_all_atom=0,
            n_coarse_grained=0,
            n_rigid_scaffold=0,
            estimated_memory_mb=0.0,
        )

    d2 = np.sum((pts - ctr[None, :]) ** 2, axis=1)
    zones = np.full(n, 2, dtype=np.uint8)
    zones[d2 <= (r_cg * r_cg)] = 1
    zones[d2 <= (r_aa * r_aa)] = 0

    n_aa = int(np.sum(zones == 0))
    n_cg = int(np.sum(zones == 1))
    n_rigid = int(np.sum(zones == 2))

    # Estimated per-atom runtime footprint: AA=256 B, CG=64 B, Rigid=16 B
    est_bytes = (n_aa * 256.0) + (n_cg * 64.0) + (n_rigid * 16.0)
    est_mb = est_bytes / (1024.0 * 1024.0)

    return MultiscaleZonePartition(
        zone_ids=zones,
        center_ang=(float(ctr[0]), float(ctr[1]), float(ctr[2])),
        aa_radius_ang=r_aa,
        cg_radius_ang=r_cg,
        n_all_atom=n_aa,
        n_coarse_grained=n_cg,
        n_rigid_scaffold=n_rigid,
        estimated_memory_mb=est_mb,
    )


class DirtyTileTracker:
    """
    Tracks per-atom transmitted coordinates (`last_sent_coords_ang`) to emit sparse
    Morton-tile delta patches containing only atoms displaced by `>= threshold_ang`.
    """

    def __init__(
        self,
        n_atoms: int,
        tile_ids: np.ndarray,
        threshold_ang: float = 0.05,
    ) -> None:
        self.n_atoms = int(n_atoms)
        self.tile_ids = np.ascontiguousarray(tile_ids, dtype=np.uint32).reshape((self.n_atoms,))
        self.threshold_ang = max(0.0, float(threshold_ang))
        self.last_sent_coords_ang: Optional[np.ndarray] = None

    def set_threshold(self, threshold_ang: float) -> None:
        self.threshold_ang = max(0.0, float(threshold_ang))

    def reset(self, coords_ang: Optional[np.ndarray] = None) -> None:
        if coords_ang is None:
            self.last_sent_coords_ang = None
        else:
            pts = np.asarray(coords_ang, dtype=np.float32).reshape((self.n_atoms, 3))
            self.last_sent_coords_ang = pts.copy()

    def compute_dirty(
        self,
        coords_ang: np.ndarray,
        commit: bool = True,
    ) -> Tuple[np.ndarray, float, int]:
        """
        Identify atoms with `|r(t) - r_sent| >= threshold_ang`, group them by 32-atom
        Morton tile IDs, update `last_sent_coords_ang` only for transmitted atoms,
        and return `(dirty_indices, dirty_ratio, n_dirty_tiles)`.
        """
        pts = np.asarray(coords_ang, dtype=np.float32).reshape((self.n_atoms, 3))
        if self.n_atoms == 0:
            return np.zeros(0, dtype=np.uint32), 0.0, 0

        if self.last_sent_coords_ang is None:
            order = np.argsort(self.tile_ids, kind="mergesort").astype(np.uint32)
            n_tiles = int(np.unique(self.tile_ids).size)
            if commit:
                self.last_sent_coords_ang = pts.copy()
            return order, 1.0, n_tiles

        disp = np.linalg.norm(pts - self.last_sent_coords_ang, axis=1)
        dirty_idx = np.flatnonzero(disp >= self.threshold_ang).astype(np.uint32)
        if dirty_idx.size == 0:
            return dirty_idx, 0.0, 0

        dirty_tiles = self.tile_ids[dirty_idx]
        tile_order = np.argsort(dirty_tiles, kind="mergesort")
        dirty_idx = dirty_idx[tile_order]
        n_dirty_tiles = int(np.unique(dirty_tiles).size)

        if commit:
            self.last_sent_coords_ang[dirty_idx] = pts[dirty_idx]

        dirty_ratio = float(dirty_idx.size) / float(self.n_atoms)
        return dirty_idx, dirty_ratio, n_dirty_tiles
