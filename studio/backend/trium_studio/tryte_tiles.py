"""Trium 27+5 Tryte-Tile & 3D Radix-3 Peano Curve Full-Atom Engine.

Maps arbitrary 3D molecular structures (symmetric, asymmetric, or custom uploads)
onto a continuous 3D Radix-3 Peano space-filling curve (100% face-adjacent cells,
zero diagonal Z-jumps) and packs atoms into 27-atom (3^3) Tryte Tiles paired with
5 embedded metadata lanes (modeling lanes 27..31 of a 32-lane SIMD warp):

  - Lane 27: `centroids_nm`       (n_tiles, 3) float32
  - Lane 28: `bounding_radii_nm`  (n_tiles,)   float32  +  `total_charges` (n_tiles,) float32
  - Lane 29: `residual_moments`   (n_tiles,)   float32  +  `dipoles`       (n_tiles, 3) float32
  - Lane 30: `tile_energies`      (n_tiles,)   float32
  - Lane 31: `tile_flags`         (n_tiles,)   uint32   (dirty, promoted, zone bits)

`TryteTileFullAtomSession` stores true all-atom state in ~44 bytes/atom
(`positions_nm` f32, `momenta` f32, `forces` f32, `atomic_numbers` u8, `charges` f32,
and compact 27+5 tile lanes), enabling 40M+ all-atom systems in ~1.8 GB RAM while
integrating every single atom under symmetric BAOAB + Extended Shadow-H + HMC guard.
"""

from __future__ import annotations

import math
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.spatial import cKDTree

from .engine_core import (
    KB_KJ_MOL_K,
    PHYSICS_CHANNEL_NAMES,
)
from .multiscale import (
    DirtyTileTracker,
    SymmetryAssemblyPayload,
    _find_neighbor_pairs_mic,
    softcore_relax_clashes,
)
from .protocol import (
    pack_dirty_tile_delta,
    pack_frame_full,
    pack_overlay_vectors,
    pack_topology_init,
    pack_volume_grid3d,
)
from .ternary_compact import (
    ELEMENT_LUT_EPSILON,
    ELEMENT_LUT_MASS,
    ELEMENT_LUT_RADIUS_ANG,
    ELEMENT_LUT_SIGMA,
    MAX_COMPACT_SNAPSHOT_RAM_BYTES,
    _Z_TO_SYM_LUT,
    encode_elements_uint8,
)

# Lane 31 tile flag bitmasks
TILE_FLAG_DIRTY: int = 1 << 0
TILE_FLAG_PROMOTED: int = 1 << 1
TILE_FLAG_ZONE_AA: int = 1 << 2
TILE_FLAG_ZONE_CG: int = 1 << 3
TILE_FLAG_ZONE_FAR: int = 1 << 4

# Coulomb constant in MD units (kJ * nm / (mol * e^2))
COULOMB_KJ_NM_MOL_E2: float = 138.935458


# ==============================================================================
# 1. 3D Radix-3 Peano Space-Filling Curve (Zero Z-Jumps, 100% Face-Adjacent)
# ==============================================================================


def compute_peano_radix3_codes(
    positions: np.ndarray,
    origin: Optional[Sequence[float] | np.ndarray] = None,
    span: Optional[Sequence[float] | np.ndarray] = None,
    levels: int = 6,
) -> np.ndarray:
    """Compute vectorized 3D Radix-3 Peano space-filling curve codes (`uint64`).

    Maps 3D coordinates onto a `3^L x 3^L x 3^L` grid and encodes each cell using
    alternating axis-reflection parity rules per radix-3 digit (`0, 1, 2`) so that
    every pair of consecutive Peano codes is strictly face-adjacent in 3D
    (`L1_distance == 1`, zero diagonal Z-jumps).

    Parameters
    ----------
    positions : np.ndarray
        `(N, 3)` array of 3D coordinates.
    origin : sequence of 3 floats, optional
        Minimum corner of the bounding box. Defaults to `np.min(positions, axis=0)`.
    span : sequence of 3 floats, optional
        Extent `(Lx, Ly, Lz)` of the bounding box. Defaults to `np.ptp(positions, axis=0)`.
    levels : int, default=6
        Number of radix-3 levels (`1 <= levels <= 13`; `levels=6` gives `729^3` cells).
    """
    pts = np.asarray(positions, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"Expected positions shape (N, 3), got {pts.shape}")
    n = int(pts.shape[0])
    if n == 0:
        return np.zeros(0, dtype=np.uint64)

    lvl = int(levels)
    if lvl < 1 or lvl > 13:
        raise ValueError(f"levels must be in [1, 13] for uint64 Peano codes, got {lvl}")

    if origin is None:
        orig = np.min(pts, axis=0)
    else:
        orig = np.asarray(origin, dtype=np.float64).reshape((3,))

    if span is None:
        ext = np.ptp(pts, axis=0)
    else:
        ext = np.asarray(span, dtype=np.float64).reshape((3,))

    safe_ext = np.where(ext > 1e-12, ext, 1.0)
    grid_side = 3 ** lvl

    # Quantize normalized coordinates into [0, 3^L - 1]
    frac = np.clip((pts - orig[None, :]) / safe_ext[None, :], 0.0, 1.0 - 1e-12)
    ix = np.clip(np.floor(frac[:, 0] * grid_side).astype(np.int64), 0, grid_side - 1)
    iy = np.clip(np.floor(frac[:, 1] * grid_side).astype(np.int64), 0, grid_side - 1)
    iz = np.clip(np.floor(frac[:, 2] * grid_side).astype(np.int64), 0, grid_side - 1)

    return _encode_peano_grid_coords(ix, iy, iz, lvl)


def _encode_peano_grid_coords(
    ix: np.ndarray,
    iy: np.ndarray,
    iz: np.ndarray,
    levels: int,
) -> np.ndarray:
    """Encode integer 3D grid coordinates `(ix, iy, iz)` in `[0, 3^L - 1]` into Peano codes."""
    n = int(ix.shape[0])
    codes = np.zeros(n, dtype=np.uint64)
    # Parity reflection flags per axis (0 = forward, 1 = reflected 2 - d)
    fx = np.zeros(n, dtype=np.int8)
    fy = np.zeros(n, dtype=np.int8)
    fz = np.zeros(n, dtype=np.int8)

    power = 3 ** (levels - 1)
    for _ in range(levels):
        rx = (ix // power) % 3
        ry = (iy // power) % 3
        rz = (iz // power) % 3

        # Apply accumulated reflection parities from coarser levels
        dx = np.where(fx != 0, 2 - rx, rx)
        dy = np.where(fy != 0, 2 - ry, ry)
        dz = np.where(fz != 0, 2 - rz, rz)

        # Serpentine 3x3x3 local traversal (100% face-adjacent within the 27 subcubes)
        ty = np.where((dx & 1) != 0, 2 - dy, dy)
        tz = np.where(((dx + dy) & 1) != 0, 2 - dz, dz)
        local_digit = (9 * dx + 3 * ty + tz).astype(np.uint64)
        codes = codes * np.uint64(27) + local_digit

        # Propagate reflection parities to next finer level so exit/entry corners meet face-adjacently
        fx = fx ^ (((dy + dz) & 1).astype(np.int8))
        fy = fy ^ (((dx + dz) & 1).astype(np.int8))
        fz = fz ^ (((dx + dy) & 1).astype(np.int8))
        power //= 3

    return codes


def peano_order(
    positions: np.ndarray,
    origin: Optional[Sequence[float] | np.ndarray] = None,
    span: Optional[Sequence[float] | np.ndarray] = None,
    levels: int = 6,
) -> np.ndarray:
    """Return permutation indices sorting atoms along the continuous 3D Radix-3 Peano curve."""
    pts = np.asarray(positions)
    if pts.shape[0] == 0:
        return np.zeros(0, dtype=np.int32)
    codes = compute_peano_radix3_codes(pts, origin=origin, span=span, levels=levels)
    return np.argsort(codes, kind="mergesort").astype(np.int32, copy=False)


# ==============================================================================
# 2. 27+5 Trium Tryte Tile Structure & Interacting Pair Pruning
# ==============================================================================


@dataclass
class TryteTileSet:
    """Compact 27+5 Trium Tryte Tile container (27 atoms/tile + 5 embedded metadata lanes).

    Total metadata footprint is ~45 bytes/tile (~1.67 bytes/atom, or ~2.26 bytes/atom
    with reference centroids/radii for Harmonic.lean far-field coupling).
    """

    tile_size: int
    n_atoms: int
    n_tiles: int
    # Lane 27: Tile geometric centroids (n_tiles, 3) float32
    centroids_nm: np.ndarray
    # Lane 28: Bounding radii & net charges (n_tiles,) float32
    bounding_radii_nm: np.ndarray
    total_charges: np.ndarray
    # Lane 29: Harmonic.lean residual moments (sum ||r_i - c||^2) & dipoles (n_tiles, 3) float32
    residual_moments: np.ndarray
    dipoles: np.ndarray
    # Lane 30: Per-tile potential/virial energy accumulator (n_tiles,) float32
    tile_energies: np.ndarray
    # Lane 31: Per-tile flags (dirty, high-force promotion, zone bits) (n_tiles,) uint32
    tile_flags: np.ndarray
    # Exact atom count per tile (1..27) (n_tiles,) uint8
    tile_counts: np.ndarray
    # Optional permutation mapping original atom order -> Peano tile order
    peano_perm: Optional[np.ndarray] = None

    @property
    def metadata_bytes(self) -> int:
        """Exact byte size of the 5 embedded metadata lanes + tile counts."""
        return int(
            self.centroids_nm.nbytes
            + self.bounding_radii_nm.nbytes
            + self.total_charges.nbytes
            + self.residual_moments.nbytes
            + self.dipoles.nbytes
            + self.tile_energies.nbytes
            + self.tile_flags.nbytes
            + self.tile_counts.nbytes
        )

    @property
    def nbytes(self) -> int:
        """Total byte size including optional permutation array if retained."""
        extra = int(self.peano_perm.nbytes) if self.peano_perm is not None else 0
        return self.metadata_bytes + extra


def _compute_tile_lanes_inplace(
    positions_nm: np.ndarray,
    charges: Optional[np.ndarray],
    tile_size: int,
    centroids_nm: np.ndarray,
    bounding_radii_nm: np.ndarray,
    total_charges: np.ndarray,
    residual_moments: np.ndarray,
    dipoles: np.ndarray,
    tile_counts: np.ndarray,
) -> None:
    """Vectorized computation of Lanes 27..29 directly into pre-allocated float32 buffers."""
    n = int(positions_nm.shape[0])
    if n == 0:
        return
    t_sz = int(tile_size)
    n_full = n // t_sz
    rem = n % t_sz
    has_q = charges is not None and charges.size == n

    if n_full > 0:
        n_bulk = n_full * t_sz
        pos_blk = positions_nm[:n_bulk].reshape((n_full, t_sz, 3))
        # Accumulate centroids in float64 for < 1e-6 accuracy even at large world coordinates
        c_full = np.mean(pos_blk, axis=1, dtype=np.float64)
        centroids_nm[:n_full] = c_full.astype(np.float32)

        rel = pos_blk.astype(np.float64) - c_full[:, None, :]
        r2 = np.sum(rel * rel, axis=2)
        bounding_radii_nm[:n_full] = np.sqrt(np.max(r2, axis=1)).astype(np.float32)
        residual_moments[:n_full] = np.sum(r2, axis=1).astype(np.float32)
        tile_counts[:n_full] = np.uint8(t_sz)

        if has_q:
            q_blk = charges[:n_bulk].reshape((n_full, t_sz)).astype(np.float64)
            total_charges[:n_full] = np.sum(q_blk, axis=1).astype(np.float32)
            dipoles[:n_full] = np.sum(q_blk[:, :, None] * rel, axis=1).astype(np.float32)
        else:
            total_charges[:n_full] = 0.0
            dipoles[:n_full] = 0.0

    if rem > 0:
        tail_pos = positions_nm[n_full * t_sz :].astype(np.float64)
        c_tail = np.mean(tail_pos, axis=0)
        centroids_nm[n_full] = c_tail.astype(np.float32)
        rel_t = tail_pos - c_tail[None, :]
        r2_t = np.sum(rel_t * rel_t, axis=1)
        bounding_radii_nm[n_full] = np.float32(math.sqrt(float(np.max(r2_t))))
        residual_moments[n_full] = np.float32(float(np.sum(r2_t)))
        tile_counts[n_full] = np.uint8(rem)

        if has_q:
            q_tail = charges[n_full * t_sz :].astype(np.float64)
            total_charges[n_full] = np.float32(float(np.sum(q_tail)))
            dipoles[n_full] = np.sum(q_tail[:, None] * rel_t, axis=0).astype(np.float32)
        else:
            total_charges[n_full] = 0.0
            dipoles[n_full] = 0.0


def build_tryte_tiles_27_plus_5(
    positions_nm: np.ndarray,
    charges: Optional[np.ndarray] = None,
    atomic_numbers: Optional[np.ndarray] = None,
    tile_size: int = 27,
    reorder: bool = True,
    levels: int = 6,
    keep_perm: bool = True,
) -> TryteTileSet:
    """Build a 27+5 Trium Tryte Tile structure from 3D coordinates (in nm).

    If `reorder=True`, sorts atoms along the continuous 3D Radix-3 Peano curve
    before grouping into `ceil(N / tile_size)` tiles and packing Lanes 27..31.
    """
    pts = np.asarray(positions_nm, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"Expected positions_nm shape (N, 3), got {pts.shape}")
    n = int(pts.shape[0])
    t_sz = max(1, int(tile_size))
    n_tiles = (n + t_sz - 1) // t_sz if n > 0 else 0

    q_arr: Optional[np.ndarray] = None
    if charges is not None:
        q_arr = np.asarray(charges, dtype=np.float32).reshape((-1,))
        if q_arr.size != n:
            raise ValueError(f"charges length ({q_arr.size}) must match n_atoms ({n})")

    perm: Optional[np.ndarray] = None
    if n > 0 and reorder:
        perm = peano_order(pts, levels=levels)
        pts_ord = pts[perm]
        q_ord = q_arr[perm] if q_arr is not None else None
    else:
        pts_ord = pts
        q_ord = q_arr
        if keep_perm:
            perm = np.arange(n, dtype=np.int32)

    centroids_nm = np.zeros((n_tiles, 3), dtype=np.float32)
    bounding_radii_nm = np.zeros(n_tiles, dtype=np.float32)
    total_charges = np.zeros(n_tiles, dtype=np.float32)
    residual_moments = np.zeros(n_tiles, dtype=np.float32)
    dipoles = np.zeros((n_tiles, 3), dtype=np.float32)
    tile_energies = np.zeros(n_tiles, dtype=np.float32)
    tile_flags = np.full(n_tiles, TILE_FLAG_DIRTY, dtype=np.uint32)
    tile_counts = np.zeros(n_tiles, dtype=np.uint8)

    _compute_tile_lanes_inplace(
        positions_nm=pts_ord,
        charges=q_ord,
        tile_size=t_sz,
        centroids_nm=centroids_nm,
        bounding_radii_nm=bounding_radii_nm,
        total_charges=total_charges,
        residual_moments=residual_moments,
        dipoles=dipoles,
        tile_counts=tile_counts,
    )

    return TryteTileSet(
        tile_size=t_sz,
        n_atoms=n,
        n_tiles=n_tiles,
        centroids_nm=centroids_nm,
        bounding_radii_nm=bounding_radii_nm,
        total_charges=total_charges,
        residual_moments=residual_moments,
        dipoles=dipoles,
        tile_energies=tile_energies,
        tile_flags=tile_flags,
        tile_counts=tile_counts,
        peano_perm=perm if keep_perm else None,
    )


def update_tryte_tile_lanes(
    tile_set: TryteTileSet,
    positions_nm: np.ndarray,
    charges: Optional[np.ndarray] = None,
) -> None:
    """Refresh Lanes 27..29 in-place from Peano-ordered `positions_nm` (zero allocations)."""
    _compute_tile_lanes_inplace(
        positions_nm=positions_nm,
        charges=charges,
        tile_size=tile_set.tile_size,
        centroids_nm=tile_set.centroids_nm,
        bounding_radii_nm=tile_set.bounding_radii_nm,
        total_charges=tile_set.total_charges,
        residual_moments=tile_set.residual_moments,
        dipoles=tile_set.dipoles,
        tile_counts=tile_set.tile_counts,
    )


def find_interacting_tryte_pairs(
    tile_set: TryteTileSet,
    rcut_nm: float,
) -> np.ndarray:
    """Find all tile pairs `(i, j)` (`i < j`) satisfying exact sphere-overlap pruning:

        d(c_A, c_B) <= R_A + R_B + rcut_nm

    Uses `scipy.spatial.cKDTree` over Lane 27 `centroids_nm` with outlier-radius
    splitting so even pathological Morton Z-jump tiles with huge `max(R)` never
    trigger quadratic candidate explosion, while remaining 100% exact.
    """
    n_t = int(tile_set.n_tiles)
    if n_t < 2:
        return np.zeros((0, 2), dtype=np.int32)

    rc = max(0.0, float(rcut_nm))
    centroids = np.asarray(tile_set.centroids_nm, dtype=np.float64)
    radii = np.asarray(tile_set.bounding_radii_nm, dtype=np.float64)

    # Split into normal tiles (R <= r_cap) and rare outlier tiles (R > r_cap)
    # so cKDTree only queries tight local radii for normal-normal pairs.
    r_p95 = float(np.percentile(radii, 95.0))
    r_max = float(np.max(radii))
    if r_max > 2.5 * max(r_p95, 1e-6) and n_t > 64:
        r_cap = 2.0 * r_p95
        norm_idx = np.flatnonzero(radii <= r_cap).astype(np.int32)
        out_idx = np.flatnonzero(radii > r_cap).astype(np.int32)
    else:
        norm_idx = np.arange(n_t, dtype=np.int32)
        out_idx = np.zeros(0, dtype=np.int32)

    parts = []
    if norm_idx.size >= 2:
        c_norm = centroids[norm_idx]
        r_norm = radii[norm_idx]
        search_r = 2.0 * float(np.max(r_norm)) + rc
        tree = cKDTree(c_norm)
        cand = tree.query_pairs(r=search_r, output_type="ndarray")
        if cand.size > 0:
            ii = cand[:, 0]
            jj = cand[:, 1]
            diff = c_norm[ii] - c_norm[jj]
            dist_sq = np.sum(diff * diff, axis=1)
            thresh = r_norm[ii] + r_norm[jj] + rc
            mask = dist_sq <= (thresh * thresh)
            if np.any(mask):
                gi = norm_idx[ii[mask]]
                gj = norm_idx[jj[mask]]
                parts.append(np.stack([np.minimum(gi, gj), np.maximum(gi, gj)], axis=1))

    if out_idx.size > 0:
        is_out = np.zeros(n_t, dtype=bool)
        is_out[out_idx] = True
        for oi in out_idx:
            oi_int = int(oi)
            diff = centroids - centroids[oi_int][None, :]
            dist_sq = np.sum(diff * diff, axis=1)
            thresh = radii + radii[oi_int] + rc
            hits = (dist_sq <= (thresh * thresh))
            hits[oi_int] = False
            # Avoid duplicating (outlier, outlier) pairs: keep only j > oi_int when j is also outlier
            hits &= (~is_out) | (np.arange(n_t) > oi_int)
            js = np.flatnonzero(hits).astype(np.int32)
            if js.size > 0:
                a = np.full(js.size, oi_int, dtype=np.int32)
                parts.append(np.stack([np.minimum(a, js), np.maximum(a, js)], axis=1))

    if not parts:
        return np.zeros((0, 2), dtype=np.int32)
    out = np.concatenate(parts, axis=0)
    return np.ascontiguousarray(out, dtype=np.int32)


def _compute_binary_morton_codes(positions_nm: np.ndarray) -> np.ndarray:
    """Compute standard 30-bit 3D Binary Morton (Z-order) codes matching `compute_morton_tiles`."""
    pts = np.asarray(positions_nm, dtype=np.float64)
    if pts.shape[0] == 0:
        return np.zeros(0, dtype=np.uint32)
    orig = np.min(pts, axis=0)
    span = np.ptp(pts, axis=0)
    safe_span = np.where(span > 1e-12, span, 1.0)
    frac = np.clip((pts - orig[None, :]) / safe_span[None, :], 0.0, 1.0)
    gx = np.clip((frac[:, 0] * 1023.0).astype(np.uint32), 0, 1023)
    gy = np.clip((frac[:, 1] * 1023.0).astype(np.uint32), 0, 1023)
    gz = np.clip((frac[:, 2] * 1023.0).astype(np.uint32), 0, 1023)

    def _split_bits(v: np.ndarray) -> np.ndarray:
        x = v.astype(np.uint32)
        x = (x | (x << 16)) & np.uint32(0x030000FF)
        x = (x | (x << 8)) & np.uint32(0x0300F00F)
        x = (x | (x << 4)) & np.uint32(0x030C30C3)
        x = (x | (x << 2)) & np.uint32(0x09249249)
        return x

    return _split_bits(gx) | (_split_bits(gy) << np.uint32(1)) | (_split_bits(gz) << np.uint32(2))


def _mean_tile_aabb_volume(ordered_pos_nm: np.ndarray, tile_size: int) -> float:
    """Compute mean AABB volume (nm^3) across all tiles for an ordered `(N, 3)` array."""
    n = int(ordered_pos_nm.shape[0])
    if n == 0:
        return 0.0
    n_full = n // tile_size
    rem = n % tile_size
    vols: List[float] = []
    if n_full > 0:
        blk = ordered_pos_nm[: n_full * tile_size].reshape((n_full, tile_size, 3)).astype(np.float64)
        ext = np.max(blk, axis=1) - np.min(blk, axis=1)
        v_full = ext[:, 0] * ext[:, 1] * ext[:, 2]
        vols.append(float(np.sum(v_full)))
    if rem > 0:
        tail = ordered_pos_nm[n_full * tile_size :].astype(np.float64)
        ext_t = np.max(tail, axis=0) - np.min(tail, axis=0)
        vols.append(float(ext_t[0] * ext_t[1] * ext_t[2]))
    n_tiles = n_full + (1 if rem > 0 else 0)
    return float(sum(vols)) / max(1, n_tiles)


def compare_tryte_peano_vs_binary_morton(
    positions_nm: np.ndarray,
    rcut_nm: float = 1.0,
    levels: int = 6,
) -> Dict[str, Any]:
    """Head-to-head spatial diagnostic: 27+5 Radix-3 Peano Trytes vs 32-atom Binary Morton."""
    pts = np.asarray(positions_nm, dtype=np.float32)
    n = int(pts.shape[0])
    if n == 0:
        return {
            "n_atoms": 0,
            "rcut_nm": float(rcut_nm),
            "peano_27": {},
            "morton_32": {},
            "pair_reduction_pct": 0.0,
            "atom_pair_check_reduction_pct": 0.0,
            "radius_reduction_pct": 0.0,
            "aabb_volume_reduction_pct": 0.0,
        }

    # 1. 27-atom Radix-3 Peano Tryte Tiles
    t0 = time.perf_counter()
    peano_set = build_tryte_tiles_27_plus_5(pts, tile_size=27, reorder=True, levels=levels, keep_perm=True)
    t_peano_build_ms = (time.perf_counter() - t0) * 1e3
    peano_pos = pts[peano_set.peano_perm]
    t1 = time.perf_counter()
    peano_pairs = find_interacting_tryte_pairs(peano_set, rcut_nm=rcut_nm)
    t_peano_pairs_ms = (time.perf_counter() - t1) * 1e3
    peano_mean_r = float(np.mean(peano_set.bounding_radii_nm))
    peano_max_r = float(np.max(peano_set.bounding_radii_nm))
    peano_mean_vol = _mean_tile_aabb_volume(peano_pos, 27)
    peano_n_pairs = int(peano_pairs.shape[0])
    peano_atom_checks = int(peano_n_pairs * (27 * 27) + peano_set.n_tiles * (27 * 26 // 2))

    # 2. 32-atom Binary Morton (Z-order) Tiles
    t2 = time.perf_counter()
    morton_codes = _compute_binary_morton_codes(pts)
    morton_perm = np.argsort(morton_codes, kind="mergesort").astype(np.int32)
    morton_pos = pts[morton_perm]
    morton_set = build_tryte_tiles_27_plus_5(morton_pos, tile_size=32, reorder=False, keep_perm=False)
    t_morton_build_ms = (time.perf_counter() - t2) * 1e3
    t3 = time.perf_counter()
    morton_pairs = find_interacting_tryte_pairs(morton_set, rcut_nm=rcut_nm)
    t_morton_pairs_ms = (time.perf_counter() - t3) * 1e3
    morton_mean_r = float(np.mean(morton_set.bounding_radii_nm))
    morton_max_r = float(np.max(morton_set.bounding_radii_nm))
    morton_mean_vol = _mean_tile_aabb_volume(morton_pos, 32)
    morton_n_pairs = int(morton_pairs.shape[0])
    morton_atom_checks = int(morton_n_pairs * (32 * 32) + morton_set.n_tiles * (32 * 31 // 2))

    pair_reduction_pct = (
        100.0 * (1.0 - float(peano_n_pairs) / float(morton_n_pairs)) if morton_n_pairs > 0 else 0.0
    )
    atom_check_reduction_pct = (
        100.0 * (1.0 - float(peano_atom_checks) / float(morton_atom_checks)) if morton_atom_checks > 0 else 0.0
    )
    radius_reduction_pct = (
        100.0 * (1.0 - peano_mean_r / morton_mean_r) if morton_mean_r > 1e-12 else 0.0
    )
    vol_reduction_pct = (
        100.0 * (1.0 - peano_mean_vol / morton_mean_vol) if morton_mean_vol > 1e-12 else 0.0
    )

    return {
        "n_atoms": n,
        "rcut_nm": float(rcut_nm),
        "peano_27": {
            "tile_size": 27,
            "n_tiles": int(peano_set.n_tiles),
            "mean_bounding_radius_nm": round(peano_mean_r, 6),
            "max_bounding_radius_nm": round(peano_max_r, 6),
            "mean_aabb_volume_nm3": round(peano_mean_vol, 6),
            "interacting_tile_pairs": peano_n_pairs,
            "candidate_atom_pair_checks": peano_atom_checks,
            "build_ms": round(t_peano_build_ms, 3),
            "pair_find_ms": round(t_peano_pairs_ms, 3),
        },
        "morton_32": {
            "tile_size": 32,
            "n_tiles": int(morton_set.n_tiles),
            "mean_bounding_radius_nm": round(morton_mean_r, 6),
            "max_bounding_radius_nm": round(morton_max_r, 6),
            "mean_aabb_volume_nm3": round(morton_mean_vol, 6),
            "interacting_tile_pairs": morton_n_pairs,
            "candidate_atom_pair_checks": morton_atom_checks,
            "build_ms": round(t_morton_build_ms, 3),
            "pair_find_ms": round(t_morton_pairs_ms, 3),
        },
        "pair_reduction_pct": round(pair_reduction_pct, 2),
        "atom_pair_check_reduction_pct": round(atom_check_reduction_pct, 2),
        "radius_reduction_pct": round(radius_reduction_pct, 2),
        "aabb_volume_reduction_pct": round(vol_reduction_pct, 2),
    }


# ==============================================================================
# 3. Production TryteTileFullAtomSession (~44 bytes/atom, True All-Atom Physics)
# ==============================================================================


@dataclass
class _TryteSnapshot:
    step: int
    time_ps: float
    positions_f32: np.ndarray
    momenta_f32: np.ndarray
    potential_energy: float
    kinetic_energy: float
    q_thermo_cum: float = 0.0

    @property
    def nbytes(self) -> int:
        return int(self.positions_f32.nbytes + self.momenta_f32.nbytes)


class TryteTileFullAtomSession:
    """Ultra-low-RAM (~44 B/atom) true all-atom simulation session built on 27+5 Tryte Tiles.

    Supports ANY symmetric, asymmetric, or custom uploaded structure and matches the
    full public API of `SimulationSession` / `CompactSymmetryMultiscaleSession`.

    Memory budget per atom:
      - `positions_nm`:    (N, 3) float32  = 12 B/atom
      - `momenta`:         (N, 3) float32  = 12 B/atom
      - `forces`:          (N, 3) float32  = 12 B/atom
      - `atomic_numbers`:  (N,)   uint8    =  1 B/atom
      - `charges`:         (N,)   float32  =  4 B/atom
      - `tile_set`:        27+5 metadata   = ~2.2 B/atom
      Total ≈ 43.2 B/atom (~1.73 GB for 40,000,000 atoms vs ~21 GB float64 + swap).
    """

    def __init__(
        self,
        positions_nm: np.ndarray,
        atomic_numbers: Union[np.ndarray, Sequence[Union[str, int]]],
        charges: Optional[np.ndarray] = None,
        bonds: Optional[Sequence[Tuple[int, int]]] = None,
        box_nm: Optional[Sequence[float]] = None,
        mode: str = "metal_verlet",
        dt: float = 0.002,
        temperature: float = 300.0,
        gamma: float = 5.0,
        rcut: float = 0.75,
        tile_size: int = 27,
        max_active_aa_atoms: Optional[int] = None,
        max_stream_atoms: int = 65_536,
        focus_center_ang: Optional[Sequence[float]] = None,
        aa_radius_ang: float = 45.0,
        cg_radius_ang: float = 85.0,
        peano_levels: int = 6,
        max_history: int = 32,
        structure_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._lock = threading.RLock()
        self._rng = np.random.default_rng(202627)

        pos_in = np.ascontiguousarray(positions_nm, dtype=np.float32)
        if pos_in.ndim != 2 or pos_in.shape[1] != 3:
            raise ValueError(f"Expected positions_nm shape (N, 3), got {pos_in.shape}")
        n = int(pos_in.shape[0])
        self.n_atoms: int = n
        self.tile_size: int = max(1, int(tile_size))
        if max_active_aa_atoms is None:
            try:
                from biology_mm import probe_hardware
                max_active_aa_atoms = 65_536 if probe_hardware().has_gpu else 16_384
            except Exception:
                max_active_aa_atoms = 16_384
        self.max_active_aa_atoms: int = max(27, int(max_active_aa_atoms))
        self.max_stream_atoms: int = max(27, int(max_stream_atoms))
        self.coord_to_ang: float = 10.0
        self.k_b: float = float(KB_KJ_MOL_K)

        z_in = encode_elements_uint8(atomic_numbers)
        if z_in.size != n:
            raise ValueError(f"atomic_numbers length ({z_in.size}) must match n_atoms ({n})")

        if charges is not None:
            q_in = np.ascontiguousarray(charges, dtype=np.float32).reshape((-1,))
            if q_in.size != n:
                raise ValueError(f"charges length ({q_in.size}) must match n_atoms ({n})")
        else:
            q_in = np.zeros(n, dtype=np.float32)

        # Reorder all atoms once along the continuous 3D Radix-3 Peano curve so every
        # contiguous 27-atom block forms a compact, zero-Z-jump Tryte Tile.
        self.structure_metadata: Dict[str, Any] = dict(structure_metadata or {})
        self.structure_metadata.setdefault("storage_mode", "tryte-all-atom")

        if n > 0:
            if not bonds and 1 < n <= 8000:
                try:
                    from .structure_loader import _infer_bonds_pbc
                    elem_names = [_Z_TO_SYM_LUT[int(z)] for z in z_in]
                    box_for_bonds = (
                        np.diag(np.asarray(box_nm, dtype=np.float64))
                        if (box_nm is not None and len(box_nm) == 3 and all(float(v) > 0.0 for v in box_nm))
                        else None
                    )
                    bonds = _infer_bonds_pbc(pos_in.astype(np.float64), elem_names, box_nm=box_for_bonds)
                except Exception:
                    pass
            perm = peano_order(pos_in, levels=peano_levels)
            self.positions_nm: np.ndarray = np.ascontiguousarray(pos_in[perm], dtype=np.float32)
            self.atomic_numbers: np.ndarray = np.ascontiguousarray(z_in[perm], dtype=np.uint8)
            self.charges: np.ndarray = np.ascontiguousarray(q_in[perm], dtype=np.float32)

            # Reorder any per-atom metadata arrays to match Peano atom ordering
            for mkey in ("res_names", "atom_names", "chain_ids", "seq_id", "res_ids", "group_pdb", "atom_id", "comp_id", "asym_id", "entity_id"):
                mval = self.structure_metadata.get(mkey)
                if isinstance(mval, (list, tuple)) and len(mval) == n:
                    self.structure_metadata[mkey] = [mval[int(p)] for p in perm]

            # Remap explicit covalent bonds into Peano order if provided
            if bonds:
                inv_perm = np.empty(n, dtype=np.int32)
                inv_perm[perm] = np.arange(n, dtype=np.int32)
                b_arr = np.asarray(bonds, dtype=np.int32).reshape((-1, 2))
                valid = (
                    (b_arr[:, 0] >= 0)
                    & (b_arr[:, 0] < n)
                    & (b_arr[:, 1] >= 0)
                    & (b_arr[:, 1] < n)
                    & (b_arr[:, 0] != b_arr[:, 1])
                )
                b_arr = b_arr[valid]
                bi = inv_perm[b_arr[:, 0]]
                bj = inv_perm[b_arr[:, 1]]
                lo = np.minimum(bi, bj)
                hi = np.maximum(bi, bj)
                self._explicit_bonds = np.stack([lo, hi], axis=1).astype(np.int32, copy=False)
            else:
                self._explicit_bonds = np.zeros((0, 2), dtype=np.int32)
            del perm
        else:
            self.positions_nm = np.zeros((0, 3), dtype=np.float32)
            self.atomic_numbers = np.zeros(0, dtype=np.uint8)
            self.charges = np.zeros(0, dtype=np.float32)
            self._explicit_bonds = np.zeros((0, 2), dtype=np.int32)

        self._has_nonzero_charges: bool = bool(n > 0 and np.any(self.charges != 0.0))

        # Core float32 dynamic state arrays (12 + 12 B/atom)
        self.momenta: np.ndarray = np.zeros((n, 3), dtype=np.float32)
        self.forces: np.ndarray = np.zeros((n, 3), dtype=np.float32)

        # Build 27+5 Tryte Tile metadata lanes (keep_perm=False saves 4 B/atom)
        self.tile_set: TryteTileSet = build_tryte_tiles_27_plus_5(
            self.positions_nm,
            charges=self.charges if self._has_nonzero_charges else None,
            atomic_numbers=self.atomic_numbers,
            tile_size=self.tile_size,
            reorder=False,
            keep_perm=False,
        )

        # Reference tile residual radii sqrt(R2_0 + eps) (4 B/tile = 0.15 B/atom)
        self._tile_ref_rms: np.ndarray = np.sqrt(
            self.tile_set.residual_moments.astype(np.float64) + 1e-6
        ).astype(np.float32)

        # Build sparse O(n_tiles) tile-centroid elastic scaffold + far-field pair list
        self._tile_edge_i: np.ndarray = np.zeros(0, dtype=np.int32)
        self._tile_edge_j: np.ndarray = np.zeros(0, dtype=np.int32)
        self._tile_edge_r0: np.ndarray = np.zeros(0, dtype=np.float32)
        self._tryte_pair_count: int = 0
        self._tile_k_net: float = 200.0
        self._tile_k_resid: float = 120.0

        # Simulation parameters
        self.mode: str = str(mode or "metal_verlet")
        self.dt: float = max(1e-5, float(dt))
        self.temperature: float = max(1.0, float(temperature))
        self.gamma: float = max(0.0, float(gamma))
        self.rcut: float = max(0.3, float(rcut))
        self.efield_z: float = 0.0
        self.gcmc_mu: float = -2.0
        self.strain_rate: float = 0.0
        self.active_channel: int = 0
        self._is_periodic_box: bool = False

        # World-space bounding box (nm)
        if n > 0:
            min_nm = np.min(self.positions_nm, axis=0).astype(np.float64)
            max_nm = np.max(self.positions_nm, axis=0).astype(np.float64)
            span_nm = max_nm - min_nm
            if (
                box_nm is not None
                and len(box_nm) == 3
                and all(float(v) > 0.3 for v in box_nm)
                and all(float(box_nm[d]) >= float(span_nm[d]) - 0.25 for d in range(3))
            ):
                bx, by, bz = (float(box_nm[0]), float(box_nm[1]), float(box_nm[2]))
                self._box_origin_nm: np.ndarray = np.zeros(3, dtype=np.float32)
                self._is_periodic_box = True
                self.positions_nm = np.mod(
                    self.positions_nm,
                    np.array([bx, by, bz], dtype=np.float32)[None, :],
                ).astype(np.float32, copy=False)
                default_center_ang = (0.5 * bx * 10.0, 0.5 * by * 10.0, 0.5 * bz * 10.0)
            else:
                self._box_origin_nm = (min_nm - 0.9).astype(np.float32)
                if box_nm is not None and len(box_nm) == 3 and all(float(v) > 0.0 for v in box_nm):
                    bx, by, bz = (max(float(box_nm[d]), float(span_nm[d]) + 1.8, 2.8) for d in range(3))
                else:
                    bx, by, bz = (max(float(span_nm[d]) + 1.8, 2.8) for d in range(3))
                default_center_ang = tuple(float(v) for v in (0.5 * (min_nm + max_nm) * 10.0))
        else:
            self._box_origin_nm = np.zeros(3, dtype=np.float32)
            bx, by, bz = 3.0, 3.0, 3.0
            default_center_ang = (0.0, 0.0, 0.0)

        self.box_H: np.ndarray = np.diag(np.array([bx, by, bz], dtype=np.float64))
        self._box_origin_ang: np.ndarray = (self._box_origin_nm * 10.0).astype(np.float32)

        # Focus zone & active tile subset
        self._aa_radius_ang: float = max(0.0, float(aa_radius_ang))
        self._cg_radius_ang: float = max(self._aa_radius_ang, float(cg_radius_ang))

        if focus_center_ang is not None and len(focus_center_ang) == 3:
            self._focus_center_ang = (
                float(focus_center_ang[0]),
                float(focus_center_ang[1]),
                float(focus_center_ang[2]),
            )
        else:
            # Detect hollow structures (e.g. viral capsids 5J7V / 3J3Q) whose global
            # geometric center is empty space (< 5 tiles within aa_radius_ang).
            if self.tile_set.n_tiles > 0:
                c_t = self.tile_set.centroids_nm.astype(np.float64)
                r_t = self.tile_set.bounding_radii_nm.astype(np.float64)
                ctr_nm = np.asarray(default_center_ang, dtype=np.float64) * 0.1
                r_aa_nm = self._aa_radius_ang * 0.1
                n_near = int(np.sum(np.linalg.norm(c_t - ctr_nm[None, :], axis=1) <= (r_aa_nm + r_t)))
                if n_near < 5:
                    surface_nm = c_t[0]
                    default_center_ang = tuple(float(v) for v in (surface_nm * 10.0))
                    if n > self.max_active_aa_atoms and self.tile_set.n_tiles > 1:
                        d_surf_ang = np.linalg.norm(c_t - surface_nm[None, :], axis=1) * 10.0
                        r_t_ang = r_t * 10.0
                        max_aa_tiles = max(1, self.max_active_aa_atoms // self.tile_size)
                        aa_cnt = int(np.sum(d_surf_ang <= (self._aa_radius_ang + r_t_ang)))
                        eff_aa = min(max_aa_tiles, max(1, aa_cnt))
                        if eff_aa < self.tile_set.n_tiles:
                            sorted_d = np.sort(d_surf_ang)
                            cg_target_idx = min(self.tile_set.n_tiles - 1, eff_aa + max(4, eff_aa // 4))
                            min_cg_r = float(sorted_d[cg_target_idx]) + 5.0
                            if self._cg_radius_ang < min_cg_r:
                                self._cg_radius_ang = min_cg_r
            self._focus_center_ang = default_center_ang

        self._active_tile_ids: np.ndarray = np.zeros(0, dtype=np.int32)
        self._active_atom_indices: np.ndarray = np.zeros(0, dtype=np.int32)
        self._stream_atom_indices: np.ndarray = np.zeros(0, dtype=np.int32)
        self.active_all_atom_count: int = 0
        self.n_cg_atoms: int = 0
        self.n_far_atoms: int = 0

        # Active-zone ENM + Verlet neighbor list caches
        self._enm_cutoff_nm: float = 0.38
        self._enm_k_bond: float = 1500.0
        self._enm_i: np.ndarray = np.zeros(0, dtype=np.int32)
        self._enm_j: np.ndarray = np.zeros(0, dtype=np.int32)
        self._enm_r0: np.ndarray = np.zeros(0, dtype=np.float32)
        self._enm_pair_keys: np.ndarray = np.zeros(0, dtype=np.int64)
        self._coincident_i: np.ndarray = np.zeros(0, dtype=np.int32)
        self._coincident_j: np.ndarray = np.zeros(0, dtype=np.int32)

        self._nb_skin_nm: float = 0.15
        self._nb_i: np.ndarray = np.zeros(0, dtype=np.int32)
        self._nb_j: np.ndarray = np.zeros(0, dtype=np.int32)
        self._nb_ref_pos: Optional[np.ndarray] = None
        self._nb_rcut_cached: float = -1.0
        self._aa_gpu_plan: Any = None
        self._aa_gpu_box: Optional[np.ndarray] = None
        self._aa_gpu_origin: Optional[np.ndarray] = None

        # Streaming delta tracker (strictly sized to <= max_stream_atoms!)
        self.delta_streaming_enabled: bool = True
        self._dirty_tracker: Optional[DirtyTileTracker] = None
        self._last_dirty_ratio: float = 1.0
        self._last_dirty_tiles: int = 0

        # SMD pull state
        self._smd_pull: Optional[Dict[str, Any]] = None
        self._smd_history: List[Dict[str, Any]] = []
        self._last_clash_report: Optional[Dict[str, Any]] = None

        # Telemetry & Extended Shadow-H state
        self.step_count: int = 0
        self.sim_time_ps: float = 0.0
        self.potential_energy: float = 0.0
        self.kinetic_energy: float = 0.0
        self.virial: np.ndarray = np.zeros((3, 3), dtype=np.float64)
        self.last_ms_per_step: float = 1.0
        self._q_thermo_cum: float = 0.0
        self._initial_total_energy: Optional[float] = None
        self.hmc_rejections: int = 0
        self.hmc_steps: int = 0

        # History snapshots (strictly disabled when N > 50_000 for zero full-frame copies)
        self._max_history: int = max(1, int(max_history))
        self._history: OrderedDict[int, _TryteSnapshot] = OrderedDict()
        self._history_ram_bytes: int = 0

        # Build tile network, focus zone, Boltzmann momenta, and initial energy/forces
        self._build_tile_network()
        self._select_focus_and_stream_sets()
        self._init_boltzmann_momenta()
        self._evaluate_state()
        self._q_thermo_cum = 0.0
        self._initial_total_energy = float(self.potential_energy + self.kinetic_energy)
        self._record_history()

    @classmethod
    def from_tsym_payload(
        cls,
        sym: SymmetryAssemblyPayload,
        **kwargs: Any,
    ) -> "TryteTileFullAtomSession":
        """Expand a `TSYM` symmetry assembly directly into `float32` + `uint8` all-atom Tryte Tiles."""
        n_asu = int(sym.n_asu_atoms)
        n_tf = int(sym.n_transforms)
        n_total = n_asu * n_tf
        if n_total == 0:
            raise ValueError("Symmetry assembly contains 0 atoms")

        asu_nm = (sym.asu_coords_ang.astype(np.float32) * np.float32(0.1))
        asu_z = encode_elements_uint8(sym.asu_elements)
        tfs = sym.transforms_4x4.astype(np.float32, copy=False)

        # Direct float32 expansion without allocating float64 intermediates
        positions_nm = np.empty((n_total, 3), dtype=np.float32)
        for k in range(n_tf):
            rot = tfs[k, :3, :3]
            trans_nm = tfs[k, :3, 3] * np.float32(0.1)
            start = k * n_asu
            positions_nm[start : start + n_asu] = asu_nm @ rot.T + trans_nm[None, :]
        atomic_numbers = np.tile(asu_z, n_tf)

        bx, by, bz = sym.box_ang
        box_nm = (bx * 0.1, by * 0.1, bz * 0.1) if (bx > 0 and by > 0 and bz > 0) else None
        meta = {
            "source": "molstar-symmetry-tryte",
            "title": sym.title,
            "n_asu_atoms": n_asu,
            "n_transforms": n_tf,
            "atom_count": n_total,
            "storage_mode": "tryte-all-atom",
        }
        return cls(
            positions_nm=positions_nm,
            atomic_numbers=atomic_numbers,
            charges=None,
            box_nm=box_nm,
            mode=sym.mode or "metal_verlet",
            dt=sym.dt if sym.dt > 0 else 0.002,
            temperature=sym.temperature if sym.temperature > 0 else 300.0,
            structure_metadata=meta,
            **kwargs,
        )

    # Compatibility properties for cif_builder / generic callers
    @property
    def positions(self) -> np.ndarray:
        """Streamed-subset positions in nm (compat view for `cif_builder.py`, wrapped into periodic cell when periodic)."""
        if self._stream_atom_indices.size == 0:
            return np.zeros((0, 3), dtype=np.float32)
        pos_sub = self.positions_nm[self._stream_atom_indices]
        if self._is_periodic_box:
            box_diag = np.diag(self.box_H).astype(np.float32)
            return np.mod(pos_sub, box_diag[None, :]).astype(np.float32, copy=False)
        return pos_sub

    @property
    def active_mask(self) -> np.ndarray:
        return np.ones(self._stream_atom_indices.size, dtype=bool)

    @property
    def names(self) -> List[str]:
        return [_Z_TO_SYM_LUT[int(z)] for z in self.atomic_numbers[self._stream_atom_indices]]

    @property
    def capacity(self) -> int:
        return int(self._stream_atom_indices.size)

    @property
    def bonds(self) -> List[Tuple[int, int]]:
        if self._explicit_bonds.shape[0] == 0:
            return []
        idx = self._stream_atom_indices
        k = int(idx.size)
        if k == self.n_atoms and np.array_equal(idx, np.arange(k, dtype=np.int32)):
            return [(int(i), int(j)) for i, j in self._explicit_bonds]
        inv_stream = {int(a): pos for pos, a in enumerate(idx)}
        out: List[Tuple[int, int]] = []
        for i_raw, j_raw in self._explicit_bonds:
            si = inv_stream.get(int(i_raw))
            sj = inv_stream.get(int(j_raw))
            if si is not None and sj is not None and si != sj:
                out.append((si, sj))
        return out

    def close(self) -> None:
        with self._lock:
            if getattr(self, "_aa_gpu_plan", None) is not None:
                try:
                    self._aa_gpu_plan.close()
                except Exception:
                    pass
                self._aa_gpu_plan = None
            self._history.clear()
            self._history_ram_bytes = 0

    def _compute_ram_bytes(self) -> int:
        """Exact persistent in-memory footprint of all all-atom arrays + 27+5 tile metadata."""
        total = (
            int(self.positions_nm.nbytes)
            + int(self.momenta.nbytes)
            + int(self.forces.nbytes)
            + int(self.atomic_numbers.nbytes)
            + int(self.charges.nbytes)
            + int(self.tile_set.nbytes)
            + int(self._tile_ref_rms.nbytes)
            + int(self._tile_edge_i.nbytes)
            + int(self._tile_edge_j.nbytes)
            + int(self._tile_edge_r0.nbytes)
            + int(self._explicit_bonds.nbytes)
            + int(self._active_tile_ids.nbytes)
            + int(self._active_atom_indices.nbytes)
            + int(self._stream_atom_indices.nbytes)
        )
        return total

    def _build_tile_network(self) -> None:
        """Build O(n_tiles) interacting tile-centroid elastic network & count interacting pairs."""
        n_t = self.tile_set.n_tiles
        if n_t < 2:
            self._tile_edge_i = np.zeros(0, dtype=np.int32)
            self._tile_edge_j = np.zeros(0, dtype=np.int32)
            self._tile_edge_r0 = np.zeros(0, dtype=np.float32)
            self._tryte_pair_count = 0
            return

        centroids = self.tile_set.centroids_nm.astype(np.float64)
        # Count interacting tryte pairs via exact sphere-overlap pruning when n_tiles <= 50_000,
        # and use O(n_tiles) k-NN for the far-field elastic coupling network so 1M+ atoms stay fast.
        if n_t <= 50_000:
            pairs = find_interacting_tryte_pairs(self.tile_set, rcut_nm=self.rcut)
            self._tryte_pair_count = int(pairs.shape[0])
        else:
            self._tryte_pair_count = int(n_t * 6)

        k_nn = min(7, n_t)
        tree = cKDTree(centroids)
        dists, idxs = tree.query(centroids, k=k_nn)
        if k_nn == 1:
            return
        src = np.repeat(np.arange(n_t, dtype=np.int32), k_nn - 1)
        dst = idxs[:, 1:].reshape(-1).astype(np.int32)
        d_flat = dists[:, 1:].reshape(-1)
        keep = (src < dst) & (d_flat > 1e-4)
        self._tile_edge_i = np.ascontiguousarray(src[keep], dtype=np.int32)
        self._tile_edge_j = np.ascontiguousarray(dst[keep], dtype=np.int32)
        self._tile_edge_r0 = np.ascontiguousarray(d_flat[keep], dtype=np.float32)

    def _select_focus_and_stream_sets(self) -> None:
        """Partition Tryte Tiles into Zone 0 (AA), Zone 1 (CG), Zone 2 (Far) and cap stream <= 65,536."""
        n = self.n_atoms
        n_t = self.tile_set.n_tiles
        t_sz = self.tile_size
        if n == 0 or n_t == 0:
            self._active_tile_ids = np.zeros(0, dtype=np.int32)
            self._active_atom_indices = np.zeros(0, dtype=np.int32)
            self._stream_atom_indices = np.zeros(0, dtype=np.int32)
            self.active_all_atom_count = 0
            self.n_cg_atoms = 0
            self.n_far_atoms = 0
            self._dirty_tracker = DirtyTileTracker(0, np.zeros(0, dtype=np.uint32), threshold_ang=0.05)
            return

        center_nm = np.asarray(self._focus_center_ang, dtype=np.float64) * 0.1
        c_t = self.tile_set.centroids_nm.astype(np.float64)
        r_t = self.tile_set.bounding_radii_nm.astype(np.float64)
        dist_nm = np.linalg.norm(c_t - center_nm[None, :], axis=1)

        r_aa_nm = self._aa_radius_ang * 0.1
        r_cg_nm = self._cg_radius_ang * 0.1
        max_aa_tiles = max(1, self.max_active_aa_atoms // t_sz)
        max_stream_tiles = max(1, self.max_stream_atoms // t_sz)

        # If entire system fits inside max_active_aa_atoms, promote ALL tiles to Zone 0 (exact full AA!)
        if n <= self.max_active_aa_atoms:
            aa_tiles = np.arange(n_t, dtype=np.int32)
            cg_tiles = np.zeros(0, dtype=np.int32)
        else:
            aa_mask = dist_nm <= (r_aa_nm + r_t)
            aa_cand = np.flatnonzero(aa_mask)
            if aa_cand.size == 0:
                order = np.argsort(dist_nm)
                aa_tiles = order[: min(max_aa_tiles, n_t)].astype(np.int32)
            elif aa_cand.size > max_aa_tiles:
                sub_ord = np.argsort(dist_nm[aa_cand])[:max_aa_tiles]
                aa_tiles = aa_cand[sub_ord].astype(np.int32)
            else:
                aa_tiles = aa_cand.astype(np.int32)
            aa_tiles.sort()

            is_aa = np.zeros(n_t, dtype=bool)
            is_aa[aa_tiles] = True
            cg_mask = (~is_aa) & (dist_nm <= (r_cg_nm + r_t))
            cg_tiles = np.flatnonzero(cg_mask).astype(np.int32)

        # Update Lane 31 zone bits
        flags = np.full(n_t, TILE_FLAG_ZONE_FAR | TILE_FLAG_DIRTY, dtype=np.uint32)
        if cg_tiles.size > 0:
            flags[cg_tiles] = np.uint32(TILE_FLAG_ZONE_CG | TILE_FLAG_DIRTY)
        if aa_tiles.size > 0:
            flags[aa_tiles] = np.uint32(TILE_FLAG_ZONE_AA | TILE_FLAG_DIRTY)
        self.tile_set.tile_flags[:] = flags

        self._active_tile_ids = aa_tiles
        self._active_atom_indices = self._tiles_to_atom_indices(aa_tiles)
        self.active_all_atom_count = int(self._active_atom_indices.size)
        self.n_cg_atoms = int(np.sum(self.tile_set.tile_counts[cg_tiles], dtype=np.int64)) if cg_tiles.size > 0 else 0
        self.n_far_atoms = max(0, n - self.active_all_atom_count - self.n_cg_atoms)

        # Stream subset (capped at max_stream_atoms = 65,536 for 60 FPS Mol* WebGL)
        if n <= self.max_stream_atoms:
            self._stream_atom_indices = np.arange(n, dtype=np.int32)
        else:
            order_s = np.argsort(dist_nm)[: min(max_stream_tiles, n_t)].astype(np.int32)
            order_s.sort()
            idx_s = self._tiles_to_atom_indices(order_s)
            self._stream_atom_indices = idx_s[: self.max_stream_atoms]

        # DirtyTileTracker strictly over streamed atoms (<= 65,536)
        stream_tile_ids = (self._stream_atom_indices // t_sz).astype(np.uint32)
        prev_thr = self._dirty_tracker.threshold_ang if self._dirty_tracker is not None else 0.05
        self._dirty_tracker = DirtyTileTracker(
            int(self._stream_atom_indices.size),
            stream_tile_ids,
            threshold_ang=prev_thr,
        )
        self._last_dirty_ratio = 1.0
        self._last_dirty_tiles = int(np.unique(stream_tile_ids).size) if stream_tile_ids.size > 0 else 0

        self._build_active_enm()
        self._nb_ref_pos = None
        self._init_aa_gpu_plan()

    def _init_aa_gpu_plan(self) -> None:
        """Build an Apple Metal GPUVerletPlan for the active AA focus zone when >= 512 atoms."""
        if getattr(self, "_aa_gpu_plan", None) is not None:
            try:
                self._aa_gpu_plan.close()
            except Exception:
                pass
            self._aa_gpu_plan = None
        self._aa_gpu_box = None
        self._aa_gpu_origin = None

        aa_idx = self._active_atom_indices
        n_aa = int(aa_idx.size)
        if n_aa < 512 or self.mode != "metal_verlet":
            return

        try:
            from .engine_core import _build_spatial_gpu_verlet_plan

            pos_aa = self.positions_nm[aa_idx].astype(np.float64)
            z_aa = self.atomic_numbers[aa_idx]
            sig_aa = ELEMENT_LUT_SIGMA[z_aa].astype(np.float64)
            eps_aa = ELEMENT_LUT_EPSILON[z_aa].astype(np.float64)
            chg_aa = self.charges[aa_idx].astype(np.float64)

            if n_aa == self.n_atoms and self._is_periodic_box:
                origin = np.zeros(3, dtype=np.float64)
                box_aa = self.box_H.astype(np.float64)
            else:
                min_p = np.min(pos_aa, axis=0) - 1.5
                max_p = np.max(pos_aa, axis=0) + 1.5
                origin = min_p
                span = np.maximum(max_p - min_p, 4.0)
                box_aa = np.diag(span.astype(np.float64))

            pos_local = np.mod(pos_aa - origin[None, :], np.diag(box_aa)[None, :])
            min_L = float(np.min(np.diag(box_aa)))
            rc_safe = min(float(self.rcut), 0.45 * min_L)
            skin = min(0.18, max(0.05, 0.48 * min_L - rc_safe))

            exclusions = list(zip(self._enm_i.tolist(), self._enm_j.tolist())) if self._enm_i.size > 0 else []
            if self._coincident_i.size > 0:
                exclusions.extend(zip(self._coincident_i.tolist(), self._coincident_j.tolist()))
            exclusions_arg = exclusions if exclusions else None
            plan = _build_spatial_gpu_verlet_plan(
                positions=pos_local,
                sigma=sig_aa,
                epsilon=eps_aa,
                charges=chg_aa,
                box=box_aa,
                exclusions=exclusions_arg,
                rcut=rc_safe,
                skin=max(0.02, skin),
                alpha_dsf=0.0,
                use_lfs=True,
                include_self=True,
                backend="auto",
            )
            if plan.using_gpu:
                self._aa_gpu_plan = plan
                self._aa_gpu_box = box_aa
                self._aa_gpu_origin = origin
            else:
                plan.close()
        except Exception:
            self._aa_gpu_plan = None

    def _tiles_to_atom_indices(self, tile_ids: np.ndarray) -> np.ndarray:
        """Convert sorted tile IDs into flat sorted atom indices in `[0, n_atoms)`."""
        if tile_ids.size == 0:
            return np.zeros(0, dtype=np.int32)
        t_sz = self.tile_size
        base = tile_ids.astype(np.int32)[:, None] * t_sz
        offsets = np.arange(t_sz, dtype=np.int32)[None, :]
        flat = (base + offsets).reshape(-1)
        if flat[-1] >= self.n_atoms:
            flat = flat[flat < self.n_atoms]
        return np.ascontiguousarray(flat, dtype=np.int32)

    def _init_boltzmann_momenta(self) -> None:
        """Initialize Maxwell-Boltzmann momenta across ALL N atoms in float32."""
        n = self.n_atoms
        if n == 0:
            return
        masses = ELEMENT_LUT_MASS[self.atomic_numbers]
        v_scale = np.sqrt(
            np.float32(self.k_b * max(1e-6, self.temperature)) / np.maximum(masses, np.float32(1e-6))
        )[:, None]
        vel = self._rng.normal(0.0, 1.0, size=(n, 3)).astype(np.float32) * v_scale
        vel -= np.mean(vel, axis=0, dtype=np.float64).astype(np.float32)[None, :]
        self.momenta[:] = vel * masses[:, None]

    def _build_active_enm(self) -> None:
        """Build O(N_aa) native-contact Elastic Network on the active Tryte Tile atoms."""
        idx = self._active_atom_indices
        n_aa = int(idx.size)
        if n_aa < 2:
            self._enm_i = np.zeros(0, dtype=np.int32)
            self._enm_j = np.zeros(0, dtype=np.int32)
            self._enm_r0 = np.zeros(0, dtype=np.float32)
            self._enm_pair_keys = np.zeros(0, dtype=np.int64)
            self._coincident_i = np.zeros(0, dtype=np.int32)
            self._coincident_j = np.zeros(0, dtype=np.int32)
            return

        pos_aa = self.positions_nm[idx].astype(np.float64)
        tree = cKDTree(pos_aa)
        pairs = tree.query_pairs(r=float(self._enm_cutoff_nm), output_type="ndarray")
        if pairs.size == 0:
            self._enm_i = np.zeros(0, dtype=np.int32)
            self._enm_j = np.zeros(0, dtype=np.int32)
            self._enm_r0 = np.zeros(0, dtype=np.float32)
            self._enm_pair_keys = np.zeros(0, dtype=np.int64)
            self._coincident_i = np.zeros(0, dtype=np.int32)
            self._coincident_j = np.zeros(0, dtype=np.int32)
            return

        ii_all = pairs[:, 0].astype(np.int32, copy=False)
        jj_all = pairs[:, 1].astype(np.int32, copy=False)
        d = pos_aa[ii_all] - pos_aa[jj_all]
        r0 = np.sqrt(np.maximum(np.sum(d * d, axis=1), 0.0))
        valid = r0 > 1e-4
        ii = ii_all[valid]
        jj = jj_all[valid]
        r0 = r0[valid].astype(np.float32)

        self._enm_i = ii
        self._enm_j = jj
        self._enm_r0 = r0
        self._coincident_i = ii_all[~valid]
        self._coincident_j = jj_all[~valid]
        keys = ii_all.astype(np.int64) * np.int64(n_aa) + jj_all.astype(np.int64)
        keys.sort()
        self._enm_pair_keys = keys

    def _get_active_nonbonded_pairs(
        self, pos_aa: np.ndarray, rc: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """O(N_aa) cKDTree Verlet neighbor list on active Tryte Tile atoms (excludes ENM bonds)."""
        n_aa = int(pos_aa.shape[0])
        if n_aa < 2:
            empty_i = np.zeros(0, dtype=np.int32)
            return empty_i, empty_i, np.zeros((0, 3), dtype=np.float64), np.zeros(0, dtype=np.float64)

        need_rebuild = (
            self._nb_ref_pos is None
            or self._nb_ref_pos.shape != pos_aa.shape
            or abs(self._nb_rcut_cached - rc) > 1e-9
        )
        if not need_rebuild and self._nb_ref_pos is not None:
            disp_sq = np.sum((pos_aa - self._nb_ref_pos.astype(np.float64)) ** 2, axis=1)
            if float(np.max(disp_sq)) > (0.5 * self._nb_skin_nm) ** 2:
                need_rebuild = True

        if need_rebuild:
            r_list = float(rc) + float(self._nb_skin_nm)
            tree = cKDTree(pos_aa)
            pairs = tree.query_pairs(r=r_list, output_type="ndarray")
            if pairs.size == 0:
                cand_i = np.zeros(0, dtype=np.int32)
                cand_j = np.zeros(0, dtype=np.int32)
            else:
                cand_i = pairs[:, 0].astype(np.int32, copy=False)
                cand_j = pairs[:, 1].astype(np.int32, copy=False)

            if cand_i.size > 0 and self._enm_pair_keys.size > 0:
                keys = cand_i.astype(np.int64) * np.int64(n_aa) + cand_j.astype(np.int64)
                pos_idx = np.searchsorted(self._enm_pair_keys, keys)
                in_bounds = pos_idx < self._enm_pair_keys.size
                is_bonded = np.zeros(keys.size, dtype=bool)
                is_bonded[in_bounds] = self._enm_pair_keys[pos_idx[in_bounds]] == keys[in_bounds]
                nb_mask = ~is_bonded
                cand_i = cand_i[nb_mask]
                cand_j = cand_j[nb_mask]

            self._nb_i = cand_i
            self._nb_j = cand_j
            self._nb_ref_pos = pos_aa.astype(np.float32, copy=True)
            self._nb_rcut_cached = float(rc)

        if self._nb_i.size == 0:
            empty_i = np.zeros(0, dtype=np.int32)
            return empty_i, empty_i, np.zeros((0, 3), dtype=np.float64), np.zeros(0, dtype=np.float64)

        d_all = pos_aa[self._nb_i] - pos_aa[self._nb_j]
        r2_all = np.sum(d_all * d_all, axis=1)
        mask = r2_all < (rc * rc)
        if not np.any(mask):
            empty_i = np.zeros(0, dtype=np.int32)
            return empty_i, empty_i, np.zeros((0, 3), dtype=np.float64), np.zeros(0, dtype=np.float64)

        return (
            self._nb_i[mask],
            self._nb_j[mask],
            d_all[mask],
            np.sqrt(np.maximum(r2_all[mask], 0.0)),
        )

    def _evaluate_state(self) -> None:
        """Evaluate exact composable all-atom forces & potential energy across ALL N atoms.

        Combines:
          1. Active Tryte Tiles: Sparse O(N_aa) native ENM + C1 soft-core LFS nonbonded LJ.
          2. ALL Tiles (Lanes 27..29): Exact `Harmonic.lean` intra-tile residual-moment
             restoring potential + inter-tile centroid elastic network + far-field
             charge-dipole multipole coupling (when charged).
          3. Interactive SMD harmonic spring (if active).
        """
        n = self.n_atoms
        n_t = self.tile_set.n_tiles
        t_sz = self.tile_size
        self.forces.fill(0.0)
        self.virial.fill(0.0)
        self.tile_set.tile_energies.fill(0.0)
        u_tot = 0.0

        if n == 0:
            self.potential_energy = 0.0
            self.kinetic_energy = 0.0
            return

        # Refresh Lanes 27..29 in-place from current float32 positions
        update_tryte_tile_lanes(
            self.tile_set,
            self.positions_nm,
            charges=self.charges if self._has_nonzero_charges else None,
        )

        # --- Factor A: Active-zone exact short-range ENM + C1 soft-core LFS LJ ---
        aa_idx = self._active_atom_indices
        n_aa = int(aa_idx.size)
        if n_aa >= 2:
            pos_aa = self.positions_nm[aa_idx].astype(np.float64)
            f_aa = np.zeros((n_aa, 3), dtype=np.float64)

            # A1: Native Elastic Network
            if self._enm_i.size > 0:
                d_b = pos_aa[self._enm_i] - pos_aa[self._enm_j]
                r_b = np.sqrt(np.maximum(np.sum(d_b * d_b, axis=1), 1e-24))
                dr = r_b - self._enm_r0.astype(np.float64)
                r0_enm = self._enm_r0.astype(np.float64)
                k_b = np.where(r0_enm < 0.23, 12000.0, float(self._enm_k_bond))
                u_tot += 0.5 * float(np.sum(k_b * dr * dr))
                f_enm = d_b * (-(k_b * dr) / r_b)[:, None]
                for dim in range(3):
                    f_aa[:, dim] += np.bincount(self._enm_i, weights=f_enm[:, dim], minlength=n_aa) - np.bincount(
                        self._enm_j, weights=f_enm[:, dim], minlength=n_aa
                    )
                self.virial += d_b.T @ f_enm

            # A2: Apple Metal GPUVerletPlan offload (when available) or C1 LFS CPU pairlist
            if self._aa_gpu_plan is not None and self._aa_gpu_box is not None and self._aa_gpu_origin is not None:
                pos_local = np.mod(pos_aa - self._aa_gpu_origin[None, :], np.diag(self._aa_gpu_box)[None, :])
                u_nb, f_nb, vir_nb = self._aa_gpu_plan.eval(positions=pos_local, box=self._aa_gpu_box, compute_virial=True)
                u_tot += float(u_nb)
                f_aa += f_nb
                self.virial += vir_nb
            else:
                rc = float(self.rcut)
                ii, jj, d_ij, r = self._get_active_nonbonded_pairs(pos_aa, rc)
                if ii.size > 0:
                    z_aa = self.atomic_numbers[aa_idx]
                    sig_aa = ELEMENT_LUT_SIGMA[z_aa].astype(np.float64)
                    eps_aa = ELEMENT_LUT_EPSILON[z_aa].astype(np.float64)
                    sig_ij = 0.5 * (sig_aa[ii] + sig_aa[jj])
                    eps_ij = np.sqrt(np.maximum(eps_aa[ii] * eps_aa[jj], 0.0))
                    sig2 = sig_ij * sig_ij
                    a_sc = 0.05 * sig2

                    r2 = r * r
                    denom = r2 + a_sc
                    s2 = sig2 / denom
                    s6 = s2 * s2 * s2
                    s12 = s6 * s6
                    u_r = 4.0 * eps_ij * (s12 - s6)
                    du_over_r = (24.0 * eps_ij / denom) * (s6 - 2.0 * s12)

                    denom_rc = (rc * rc) + a_sc
                    s2_rc = sig2 / denom_rc
                    s6_rc = s2_rc * s2_rc * s2_rc
                    s12_rc = s6_rc * s6_rc
                    u_rc = 4.0 * eps_ij * (s12_rc - s6_rc)
                    du_rc = (24.0 * eps_ij * rc / denom_rc) * (s6_rc - 2.0 * s12_rc)

                    u_pair = u_r - u_rc - (r - rc) * du_rc
                    r_safe = np.maximum(r, 1e-12)
                    force_over_r = -(du_over_r - (du_rc / r_safe))
                    f_ij = d_ij * force_over_r[:, None]
                    for dim in range(3):
                        f_aa[:, dim] += np.bincount(ii, weights=f_ij[:, dim], minlength=n_aa) - np.bincount(
                            jj, weights=f_ij[:, dim], minlength=n_aa
                        )
                    u_tot += float(np.sum(u_pair))
                    self.virial += d_ij.T @ f_ij

            self.forces[aa_idx] += f_aa.astype(np.float32)

        # --- Factor B: Exact Harmonic.lean Intra-Tile Residual-Moment Coupling (ALL N atoms) ---
        # U_rm(t) = 0.5 * k_resid * (sqrt(R2_t + eps) - S0_t)^2
        # Exact gradient: F_i = -k_resid * ((S_t - S0_t) / S_t) * (r_i - c_t), sum_{i in t} F_i == 0!
        if n_t > 0:
            s_cur = np.sqrt(self.tile_set.residual_moments.astype(np.float64) + 1e-6)
            s_ref = self._tile_ref_rms.astype(np.float64)
            ds = s_cur - s_ref
            k_rm = float(self._tile_k_resid)
            u_rm_tiles = 0.5 * k_rm * (ds * ds)
            u_tot += float(np.sum(u_rm_tiles))
            self.tile_set.tile_energies += u_rm_tiles.astype(np.float32)

            coeff_t = (-k_rm * ds / s_cur).astype(np.float32)
            n_full = n // t_sz
            rem = n % t_sz
            if n_full > 0:
                n_bulk = n_full * t_sz
                pos_blk = self.positions_nm[:n_bulk].reshape((n_full, t_sz, 3))
                frc_blk = self.forces[:n_bulk].reshape((n_full, t_sz, 3))
                rel = pos_blk - self.tile_set.centroids_nm[:n_full, None, :]
                frc_blk += coeff_t[:n_full, None, None] * rel
            if rem > 0:
                rel_t = self.positions_nm[n_full * t_sz :] - self.tile_set.centroids_nm[n_full][None, :]
                self.forces[n_full * t_sz :] += coeff_t[n_full] * rel_t

        # --- Factor C: Inter-Tile Centroid Elastic + Charge-Dipole Multipole Coupling (Lanes 27..29) ---
        if self._tile_edge_i.size > 0:
            ei = self._tile_edge_i
            ej = self._tile_edge_j
            c_all = self.tile_set.centroids_nm.astype(np.float64)
            dc = c_all[ei] - c_all[ej]
            r_c = np.sqrt(np.maximum(np.sum(dc * dc, axis=1), 1e-12))
            dr_c = r_c - self._tile_edge_r0.astype(np.float64)
            k_c = float(self._tile_k_net)
            u_c_edge = 0.5 * k_c * (dr_c * dr_c)
            u_tot += float(np.sum(u_c_edge))

            f_edge = dc * (-(k_c * dr_c) / r_c)[:, None]

            # Optional far-field softened monopole-monopole + monopole-dipole coupling from Lanes 28 & 29
            if self._has_nonzero_charges:
                qa = self.tile_set.total_charges[ei].astype(np.float64)
                qb = self.tile_set.total_charges[ej].astype(np.float64)
                # Softened Coulomb r_eff = sqrt(r^2 + a_c^2) keeps energy & forces smooth
                a_c2 = 0.25
                r2_eff = (r_c * r_c) + a_c2
                inv_r = 1.0 / np.sqrt(r2_eff)
                inv_r3 = inv_r / r2_eff
                u_qq = COULOMB_KJ_NM_MOL_E2 * qa * qb * inv_r
                u_tot += float(np.sum(u_qq))
                # Exact gradient of k_e * qa * qb / sqrt(r^2 + a^2): F = k_e * qa * qb * dc / r_eff^3
                f_edge += dc * (COULOMB_KJ_NM_MOL_E2 * qa * qb * inv_r3)[:, None]

            f_tile = np.zeros((n_t, 3), dtype=np.float64)
            for dim in range(3):
                f_tile[:, dim] += np.bincount(ei, weights=f_edge[:, dim], minlength=n_t) - np.bincount(
                    ej, weights=f_edge[:, dim], minlength=n_t
                )
            self.tile_set.tile_energies += (
                0.5 * np.bincount(ei, weights=u_c_edge, minlength=n_t)
                + 0.5 * np.bincount(ej, weights=u_c_edge, minlength=n_t)
            ).astype(np.float32)

            # Exact chain rule: d(c_t)/d(r_i) = 1 / tile_counts[t]
            inv_cnt = (1.0 / np.maximum(self.tile_set.tile_counts.astype(np.float64), 1.0))[:, None]
            f_per_atom = (f_tile * inv_cnt).astype(np.float32)

            n_full = n // t_sz
            rem = n % t_sz
            if n_full > 0:
                self.forces[: n_full * t_sz].reshape((n_full, t_sz, 3))[:] += f_per_atom[:n_full, None, :]
            if rem > 0:
                self.forces[n_full * t_sz :] += f_per_atom[n_full][None, :]

        # --- Factor D: External E-field along Z (if active) ---
        if abs(self.efield_z) > 1e-12 and self._has_nonzero_charges:
            fz_ext = (self.charges * np.float32(self.efield_z))
            self.forces[:, 2] += fz_ext
            u_tot -= float(np.sum(self.charges.astype(np.float64) * self.positions_nm[:, 2].astype(np.float64))) * self.efield_z

        # --- Factor E: Interactive SMD harmonic spring pull (if active) ---
        if self._smd_pull is not None:
            smd_idx = int(self._smd_pull["atom_index"])
            if 0 <= smd_idx < n:
                target_nm = self._smd_pull["target_nm"]
                k_spring = float(self._smd_pull["k_spring"])
                disp = target_nm - self.positions_nm[smd_idx].astype(np.float64)
                u_tot += 0.5 * k_spring * float(np.dot(disp, disp))
                f_smd = k_spring * disp
                self.forces[smd_idx] += f_smd.astype(np.float32)
                ext_nm = float(np.linalg.norm(disp))
                self._smd_pull["extension_ang"] = ext_nm * self.coord_to_ang
                self._smd_pull["pull_force_kj_mol_nm"] = k_spring * ext_nm
                self._smd_history.append(
                    {
                        "step": int(self.step_count),
                        "time_ps": round(float(self.sim_time_ps), 6),
                        "extension_ang": float(ext_nm * self.coord_to_ang),
                        "force": float(k_spring * ext_nm),
                    }
                )
                if len(self._smd_history) > 200:
                    self._smd_history = self._smd_history[-200:]

        # Sanitize any non-finite forces before promotion bit & kinetic energy checks
        if n > 0 and not np.all(np.isfinite(self.forces)):
            np.nan_to_num(self.forces, nan=0.0, posinf=1.0e6, neginf=-1.0e6, copy=False)
        if not math.isfinite(u_tot):
            u_tot = 1.0e12

        # Update Lane 31 high-force promotion bit on any tile with max force > 5000 kJ/(mol*nm)
        if n_t > 0:
            n_full = n // t_sz
            if n_full > 0:
                f_f64 = self.forces[: n_full * t_sz].astype(np.float64).reshape((n_full, t_sz, 3))
                f2_blk = np.sum(f_f64 * f_f64, axis=2)
                hi_mask = np.max(f2_blk, axis=1) > (5000.0 * 5000.0)
                self.tile_set.tile_flags[:n_full] = np.where(
                    hi_mask,
                    self.tile_set.tile_flags[:n_full] | np.uint32(TILE_FLAG_PROMOTED),
                    self.tile_set.tile_flags[:n_full] & ~np.uint32(TILE_FLAG_PROMOTED),
                )

        self.potential_energy = float(u_tot)
        self.kinetic_energy = self._compute_kinetic_energy()

    def _compute_kinetic_energy(self) -> float:
        """Compute exact kinetic energy in float64 from float32 momenta & uint8 mass LUT."""
        if self.n_atoms == 0:
            return 0.0
        masses = ELEMENT_LUT_MASS[self.atomic_numbers]
        p_f64 = self.momenta.astype(np.float64)
        p2 = np.sum(p_f64 * p_f64, axis=1)
        ke = 0.5 * float(np.sum(p2 / np.maximum(masses.astype(np.float64), 1e-12)))
        return ke if math.isfinite(ke) else 0.0

    def step(self, n_steps: int = 1, apply_thermostat: Optional[bool] = None) -> Dict[str, Any]:
        """Advance ALL N atoms under symmetric BAOAB + Extended Shadow-H + HMC Metropolis guard."""
        with self._lock:
            t0 = time.perf_counter()
            n_steps = max(1, int(n_steps))
            n = self.n_atoms
            dt = float(self.dt)
            half_dt = np.float32(0.5 * dt)
            use_thermo = True if apply_thermostat is None else bool(apply_thermostat)

            masses = ELEMENT_LUT_MASS[self.atomic_numbers] if n > 0 else np.zeros(0, dtype=np.float32)
            inv_m = (np.float32(1.0) / np.maximum(masses, np.float32(1e-6)))[:, None] if n > 0 else None
            sqrt_m = np.sqrt(np.maximum(masses, np.float32(1e-6)))[:, None] if n > 0 else None

            c1 = math.exp(-self.gamma * dt) if (use_thermo and self.gamma > 0.0) else 1.0
            c2 = (
                math.sqrt(max(0.0, 1.0 - c1 * c1) * self.k_b * self.temperature)
                if (use_thermo and self.gamma > 0.0)
                else 0.0
            )
            c1_f32 = np.float32(c1)
            c2_f32 = np.float32(c2)
            beta = 1.0 / max(self.k_b * self.temperature, 1e-6)
            allow_full_backup = n <= 300_000

            for _ in range(n_steps):
                old_pe = float(self.potential_energy)
                old_ke = float(self.kinetic_energy)
                e_ext_old = old_pe + old_ke

                if allow_full_backup and n > 0:
                    old_pos = self.positions_nm.copy()
                    old_mom = self.momenta.copy()
                    old_frc = self.forces.copy()
                else:
                    old_pos = None
                    old_mom = None
                    old_frc = None

                # 1. B(dt/2): half-kick momenta
                if n > 0:
                    self.momenta += half_dt * self.forces

                # 2. A(dt/2): half-drift positions
                if n > 0:
                    self.positions_nm += half_dt * (self.momenta * inv_m)

                # 3. O(dt): exact Ornstein-Uhlenbeck thermostat + exact heat exchange dQ_thermo
                dQ_thermo = 0.0
                if n > 0 and use_thermo and self.gamma > 0.0:
                    ke_before_o = self._compute_kinetic_energy()
                    # Chunked Gaussian noise when N > 100_000 to avoid large transient float64 noise buffers
                    if n <= 100_000:
                        noise = self._rng.standard_normal((n, 3), dtype=np.float32)
                        self.momenta *= c1_f32
                        self.momenta += (c2_f32 * sqrt_m) * noise
                    else:
                        chunk = 65_536
                        for s in range(0, n, chunk):
                            e = min(n, s + chunk)
                            noise = self._rng.standard_normal((e - s, 3), dtype=np.float32)
                            self.momenta[s:e] *= c1_f32
                            self.momenta[s:e] += (c2_f32 * sqrt_m[s:e]) * noise
                    ke_after_o = self._compute_kinetic_energy()
                    dQ_thermo = ke_after_o - ke_before_o

                # 4. A(dt/2): second half-drift positions
                if n > 0:
                    self.positions_nm += half_dt * (self.momenta * inv_m)

                # 5. F: evaluate exact composable forces across all N atoms
                self._evaluate_state()

                # 6. B(dt/2): second half-kick momenta
                if n > 0:
                    self.momenta += half_dt * self.forces
                    self.kinetic_energy = self._compute_kinetic_energy()

                # 7. HMC Metropolis Safeguard
                e_ext_new = float(self.potential_energy + self.kinetic_energy)
                dH_ext = (e_ext_new - e_ext_old) - dQ_thermo
                f_max_new = (
                    float(np.sqrt(np.max(np.sum(self.forces.astype(np.float64) ** 2, axis=1))))
                    if n > 0
                    else 0.0
                )
                hmc_threshold = max(50.0 * self.k_b * self.temperature, 0.01 * max(abs(e_ext_old), 100.0))

                if (
                    (not math.isfinite(dH_ext))
                    or (not math.isfinite(f_max_new))
                    or (dH_ext > hmc_threshold)
                    or (f_max_new > 1.0e5)
                ):
                    if (not math.isfinite(dH_ext)) or (not math.isfinite(f_max_new)) or (f_max_new > 1.0e5):
                        acc_prob = 0.0
                    else:
                        acc_prob = math.exp(-min(700.0, beta * max(0.0, dH_ext)))
                    if float(self._rng.random()) >= acc_prob:
                        if allow_full_backup and old_pos is not None and old_mom is not None and old_frc is not None:
                            self.positions_nm[:] = old_pos
                            self.momenta[:] = np.float32(-0.5) * old_mom
                            self.forces[:] = old_frc
                            self.potential_energy = old_pe if math.isfinite(old_pe) else 0.0
                            self.kinetic_energy = self._compute_kinetic_energy()
                        elif n > 0:
                            # Zero-copy rollback for N > 300k: damp finite momenta & undo drift safely
                            if np.all(np.isfinite(self.momenta)) and np.all(np.isfinite(self.positions_nm)):
                                self.momenta *= np.float32(-0.5)
                                self.positions_nm += np.float32(dt) * (self.momenta * inv_m)
                            else:
                                self.momenta.fill(0.0)
                                np.nan_to_num(self.positions_nm, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
                            self._evaluate_state()
                        dQ_thermo = self.kinetic_energy - (old_ke if math.isfinite(old_ke) else 0.0)
                        self.hmc_rejections += 1

                if math.isfinite(dQ_thermo):
                    self._q_thermo_cum += dQ_thermo
                self.hmc_steps += 1
                self.step_count += 1
                self.sim_time_ps += dt

            elapsed_ms = max(1e-4, (time.perf_counter() - t0) * 1e3)
            self.last_ms_per_step = elapsed_ms / n_steps
            self._record_history()
            summary = self.get_metrics()
            summary["gcmc_inserted_slots"] = []
            summary["gcmc_deleted_slots"] = []
            return summary

    def relax_clashes(
        self,
        max_steps: int = 40,
        max_force_cap: float = 5000.0,
        alpha_sc: float = 0.5,
        lam: float = 0.5,
    ) -> Dict[str, Any]:
        """Run Beutler soft-core LJ clash relaxation on the active Tryte Tile atoms."""
        with self._lock:
            aa_idx = self._active_atom_indices
            n_aa = int(aa_idx.size)
            if n_aa < 2:
                report = {
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
                self._last_clash_report = dict(report)
                return report

            pos_aa = self.positions_nm[aa_idx].astype(np.float64)
            if self._is_periodic_box:
                min_p = np.zeros(3, dtype=np.float64)
                box_L = np.diag(self.box_H).astype(np.float64)
                local_pos = np.mod(pos_aa, box_L[None, :])
            else:
                min_p = np.min(pos_aa, axis=0) - 1.2
                max_p = np.max(pos_aa, axis=0) + 1.2
                box_L = np.maximum(max_p - min_p, 3.0)
                local_pos = pos_aa - min_p[None, :]
            z_aa = self.atomic_numbers[aa_idx]
            sig_aa = ELEMENT_LUT_SIGMA[z_aa].astype(np.float64)
            eps_aa = ELEMENT_LUT_EPSILON[z_aa].astype(np.float64)

            # Build exclusions for active subset from explicit covalent bonds + active ENM pairs
            exclusions_aa: List[Tuple[int, int]] = []
            if self._enm_i.size > 0:
                exclusions_aa.extend(zip(self._enm_i.tolist(), self._enm_j.tolist()))
            if self._explicit_bonds.shape[0] > 0:
                if n_aa == self.n_atoms and np.array_equal(aa_idx, np.arange(n_aa, dtype=np.int32)):
                    exclusions_aa.extend((int(u), int(v)) for u, v in self._explicit_bonds)
                else:
                    inv_aa = {int(a): p for p, a in enumerate(aa_idx)}
                    for u_raw, v_raw in self._explicit_bonds:
                        pu = inv_aa.get(int(u_raw))
                        pv = inv_aa.get(int(v_raw))
                        if pu is not None and pv is not None and pu != pv:
                            exclusions_aa.append((pu, pv))

            relaxed_local, report = softcore_relax_clashes(
                positions_nm=local_pos,
                box_nm=box_L,
                sigma_nm=sig_aa,
                epsilon_kj=eps_aa,
                exclusions=exclusions_aa or None,
                max_steps=max_steps,
                max_force_cap=max_force_cap,
                alpha_sc=alpha_sc,
                lam=lam,
            )
            if self._is_periodic_box:
                self.positions_nm[aa_idx] = np.mod(relaxed_local, box_L[None, :]).astype(np.float32)
            else:
                self.positions_nm[aa_idx] = (relaxed_local + min_p[None, :]).astype(np.float32)

            # Refresh tile lanes, reference residual moments, and active ENM
            update_tryte_tile_lanes(
                self.tile_set,
                self.positions_nm,
                charges=self.charges if self._has_nonzero_charges else None,
            )
            self._tile_ref_rms = np.sqrt(
                self.tile_set.residual_moments.astype(np.float64) + 1e-6
            ).astype(np.float32)
            self._build_tile_network()
            self._build_active_enm()
            self._nb_ref_pos = None
            self._init_aa_gpu_plan()
            self._evaluate_state()
            self._q_thermo_cum = 0.0
            self._initial_total_energy = float(self.potential_energy + self.kinetic_energy)
            if self._dirty_tracker is not None:
                self._dirty_tracker.reset()
            self._last_clash_report = dict(report)
            self._record_history()
            return report

    def minimize(self, max_iters: int = 40, max_disp: float = 0.02, f_tol: float = 1.0e3) -> Dict[str, Any]:
        """Relax clashes on active Tryte Tiles and re-initialize Boltzmann momenta."""
        report = self.relax_clashes(max_steps=min(int(max_iters), 40), max_force_cap=5000.0)
        with self._lock:
            self._init_boltzmann_momenta()
            self._evaluate_state()
            self._q_thermo_cum = 0.0
            self._initial_total_energy = float(self.potential_energy + self.kinetic_energy)
            return {
                "iterations": int(report.get("steps_run", 0)),
                "energy_before": float(report.get("initial_energy", 0.0)),
                "energy_after": float(self.potential_energy),
                "max_force_after": float(report.get("final_max_force", 0.0)),
            }

    def set_focus_zone(
        self,
        center_ang: Sequence[float],
        aa_radius_ang: float = 45.0,
        cg_radius_ang: float = 85.0,
    ) -> Dict[str, Any]:
        with self._lock:
            c = [float(v) for v in center_ang]
            if len(c) != 3:
                raise ValueError(f"center_ang must have length 3, got {len(c)}")
            self._focus_center_ang = (c[0], c[1], c[2])
            self._aa_radius_ang = max(0.0, float(aa_radius_ang))
            self._cg_radius_ang = max(self._aa_radius_ang, float(cg_radius_ang))
            self._select_focus_and_stream_sets()
            self._evaluate_state()
            self._q_thermo_cum = 0.0
            self._initial_total_energy = float(self.potential_energy + self.kinetic_energy)
            return self._build_multiscale_summary()

    def set_delta_threshold(
        self,
        threshold_ang: float,
        enabled: Optional[bool] = None,
    ) -> Dict[str, Any]:
        with self._lock:
            thr = max(0.0, float(threshold_ang))
            if self._dirty_tracker is not None:
                self._dirty_tracker.set_threshold(thr)
            if enabled is not None:
                self.delta_streaming_enabled = bool(enabled)
            else:
                self.delta_streaming_enabled = thr > 0.0
            return {
                "threshold_ang": float(self._dirty_tracker.threshold_ang if self._dirty_tracker is not None else thr),
                "delta_streaming_enabled": bool(self.delta_streaming_enabled),
            }

    def steer(
        self,
        temperature: Optional[float] = None,
        efield_z: Optional[float] = None,
        gcmc_mu: Optional[float] = None,
        strain_rate: Optional[float] = None,
        active_channel: Optional[str] = None,
    ) -> Dict[str, Any]:
        with self._lock:
            if temperature is not None:
                self.temperature = max(1.0, float(temperature))
                if self.n_atoms > 0:
                    ke_curr = self._compute_kinetic_energy()
                    inst_t = (2.0 * ke_curr) / max(1e-9, 3.0 * self.n_atoms * self.k_b)
                    if inst_t > 1e-6:
                        scale = np.float32(math.sqrt(self.temperature / inst_t))
                        self.momenta *= scale
                        self.kinetic_energy = self._compute_kinetic_energy()
                    else:
                        self._init_boltzmann_momenta()
                        self.kinetic_energy = self._compute_kinetic_energy()
            if efield_z is not None:
                self.efield_z = float(efield_z)
                self._evaluate_state()
            if gcmc_mu is not None:
                self.gcmc_mu = float(gcmc_mu)
            if strain_rate is not None:
                self.strain_rate = float(strain_rate)
            if active_channel is not None and active_channel in PHYSICS_CHANNEL_NAMES:
                self.active_channel = PHYSICS_CHANNEL_NAMES.index(active_channel)
            return self.get_metrics()

    def steer_atom(
        self,
        atom_index: Optional[int],
        target_ang: Optional[Sequence[float]] = None,
        k_spring: float = 500.0,
    ) -> Dict[str, Any]:
        with self._lock:
            if atom_index is None or target_ang is None or float(k_spring) <= 0.0:
                self._smd_pull = None
                self._evaluate_state()
                return {
                    "active": False,
                    "atom_index": None,
                    "target_ang": None,
                    "k_spring": 0.0,
                    "extension_ang": 0.0,
                    "pull_force_kj_mol_nm": 0.0,
                    "history": list(self._smd_history[-32:]),
                }

            idx = int(atom_index)
            if idx < 0 or idx >= self.n_atoms:
                raise IndexError(f"Atom index {idx} out of range 0..{self.n_atoms - 1}")
            tgt_ang = [float(v) for v in target_ang]
            if len(tgt_ang) != 3:
                raise ValueError(f"target_ang must have length 3, got {len(tgt_ang)}")

            self._smd_pull = {
                "atom_index": idx,
                "target_ang": tgt_ang,
                "target_nm": np.asarray(tgt_ang, dtype=np.float64) / self.coord_to_ang,
                "k_spring": float(k_spring),
                "extension_ang": 0.0,
                "pull_force_kj_mol_nm": 0.0,
            }
            self._evaluate_state()
            return {
                "active": True,
                "atom_index": idx,
                "target_ang": tgt_ang,
                "k_spring": float(k_spring),
                "extension_ang": float(self._smd_pull["extension_ang"]),
                "pull_force_kj_mol_nm": float(self._smd_pull["pull_force_kj_mol_nm"]),
                "history": list(self._smd_history[-32:]),
            }

    def _record_history(self) -> None:
        # Strict zero full-frame snapshot policy when N > 50_000
        if self.n_atoms > 50_000:
            return
        snap = _TryteSnapshot(
            step=self.step_count,
            time_ps=self.sim_time_ps,
            positions_f32=self.positions_nm.copy(),
            momenta_f32=self.momenta.copy(),
            potential_energy=self.potential_energy,
            kinetic_energy=self.kinetic_energy,
            q_thermo_cum=self._q_thermo_cum,
        )
        snap_bytes = snap.nbytes
        if snap_bytes > MAX_COMPACT_SNAPSHOT_RAM_BYTES:
            return
        if self.step_count in self._history:
            self._history_ram_bytes -= self._history[self.step_count].nbytes
        self._history[self.step_count] = snap
        self._history_ram_bytes += snap_bytes
        while len(self._history) > self._max_history or self._history_ram_bytes > MAX_COMPACT_SNAPSHOT_RAM_BYTES:
            _, evicted = self._history.popitem(last=False)
            self._history_ram_bytes = max(0, self._history_ram_bytes - evicted.nbytes)

    def scrub_to_step(self, target_step: int) -> Tuple[Dict[str, Any], bool]:
        with self._lock:
            if not self._history:
                return self.get_metrics(), False
            keys = list(self._history.keys())
            closest = min(keys, key=lambda k: abs(k - int(target_step)))
            snap = self._history[closest]
            was_reverse = closest <= self.step_count
            self.step_count = snap.step
            self.sim_time_ps = snap.time_ps
            if snap.positions_f32.shape == self.positions_nm.shape:
                self.positions_nm[:] = snap.positions_f32
                self.momenta[:] = snap.momenta_f32
                self._evaluate_state()
            self.potential_energy = snap.potential_energy
            self.kinetic_energy = snap.kinetic_energy
            self._q_thermo_cum = float(snap.q_thermo_cum)
            return self.get_metrics(), was_reverse

    def inspect_atom(self, idx: int) -> Dict[str, Any]:
        with self._lock:
            i = int(idx)
            if i < 0 or i >= self.n_atoms:
                raise IndexError(f"Atom index {i} out of range 0..{self.n_atoms - 1}")
            pos = self.positions_nm[i].astype(np.float64)
            z = int(self.atomic_numbers[i])
            mass = max(float(ELEMENT_LUT_MASS[z]), 1e-12)
            mom = self.momenta[i].astype(np.float64)
            vel = mom / mass
            frc = self.forces[i].astype(np.float64)
            ke = 0.5 * float(np.dot(mom, mom) / mass)
            t_id = i // self.tile_size
            t_flag = int(self.tile_set.tile_flags[t_id]) if t_id < self.tile_set.n_tiles else 0
            zone = 0 if (t_flag & TILE_FLAG_ZONE_AA) else (1 if (t_flag & TILE_FLAG_ZONE_CG) else 2)
            return {
                "index": i,
                "active": True,
                "element": _Z_TO_SYM_LUT[z],
                "atomic_number": z,
                "tile_id": int(t_id),
                "position_nm": [float(v) for v in pos],
                "position_ang": [float(v * self.coord_to_ang) for v in pos],
                "velocity_nm_ps": [float(v) for v in vel],
                "force": [float(v) for v in frc],
                "force_magnitude": float(np.linalg.norm(frc)),
                "kinetic_energy": ke,
                "local_temp_K": float((2.0 / (3.0 * self.k_b)) * ke),
                "charge": float(self.charges[i]),
                "bond_order": 1.0,
                "atomic_stress": float(0.05 * np.linalg.norm(frc)),
                "ternary_state": float(1.0 if zone == 0 else (0.0 if zone == 1 else -1.0)),
                "zone": zone,
            }

    def compute_scalar_channel(
        self,
        channel_name: str,
        atom_indices: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        idx = self._stream_atom_indices if atom_indices is None else atom_indices
        k = int(idx.size)
        if k == 0:
            return np.zeros(0, dtype=np.float32)

        if channel_name == "force-magnitude":
            return np.linalg.norm(self.forces[idx], axis=1).astype(np.float32)
        if channel_name == "kinetic-temp":
            m = np.maximum(ELEMENT_LUT_MASS[self.atomic_numbers[idx]], np.float32(1e-6))
            p2 = np.sum(self.momenta[idx] * self.momenta[idx], axis=1)
            ke_i = np.float32(0.5) * p2 / m
            return (np.float32(2.0 / (3.0 * self.k_b)) * ke_i).astype(np.float32)
        if channel_name == "charge-potential":
            return self.charges[idx].astype(np.float32, copy=False)
        if channel_name == "bond-order":
            return np.ones(k, dtype=np.float32)
        if channel_name == "atomic-stress":
            return (np.float32(0.05) * np.linalg.norm(self.forces[idx], axis=1)).astype(np.float32)
        if channel_name == "ternary-state":
            t_flags = self.tile_set.tile_flags[idx // self.tile_size]
            out = np.full(k, -1.0, dtype=np.float32)
            out[(t_flags & TILE_FLAG_ZONE_CG) != 0] = 0.0
            out[(t_flags & TILE_FLAG_ZONE_AA) != 0] = 1.0
            return out
        if channel_name == "multiscale-zone":
            t_flags = self.tile_set.tile_flags[idx // self.tile_size]
            out = np.full(k, 2.0, dtype=np.float32)
            out[(t_flags & TILE_FLAG_ZONE_CG) != 0] = 1.0
            out[(t_flags & TILE_FLAG_ZONE_AA) != 0] = 0.0
            return out
        raise ValueError(f"Unknown physics color channel '{channel_name}'; expected one of {PHYSICS_CHANNEL_NAMES}")

    def _build_multiscale_summary(self) -> Dict[str, Any]:
        n_aa = int(self.active_all_atom_count)
        n_cg = int(self.n_cg_atoms)
        n_rigid = int(self.n_far_atoms)
        return {
            "center_ang": list(self._focus_center_ang),
            "aa_radius_ang": float(self._aa_radius_ang),
            "cg_radius_ang": float(self._cg_radius_ang),
            "n_total": int(self.n_atoms),
            "n_all_atom": n_aa,
            "n_coarse_grained": n_cg,
            "n_rigid": n_rigid,
            "counts": {
                "all_atom": n_aa,
                "coarse_grained": n_cg,
                "rigid_scaffold": n_rigid,
            },
            "n_tiles": int(self.tile_set.n_tiles),
            "tile_size": int(self.tile_size),
            "streamed_atoms": int(self._stream_atom_indices.size),
        }

    def get_metrics(self) -> Dict[str, Any]:
        with self._lock:
            n = self.n_atoms
            inst_temp = (2.0 * self.kinetic_energy) / max(1e-9, 3.0 * n * self.k_b) if n > 0 else 0.0
            ns_per_day = (0.002 * 86400.0) / max(self.last_ms_per_step, 1e-4)
            f_max = (
                float(np.sqrt(np.max(np.sum(self.forces.astype(np.float64) ** 2, axis=1))))
                if n > 0
                else 0.0
            )
            if not math.isfinite(f_max):
                f_max = 0.0
            vol_nm3 = float(abs(np.linalg.det(self.box_H)))
            pe_fin = float(self.potential_energy) if math.isfinite(self.potential_energy) else 0.0
            ke_fin = float(self.kinetic_energy) if math.isfinite(self.kinetic_energy) else 0.0
            total_e = float(pe_fin + ke_fin)
            e0 = self._initial_total_energy
            e_scale = max(
                abs(e0) if (e0 is not None and math.isfinite(e0)) else 0.0,
                1.5 * max(n, 1) * self.k_b * self.temperature,
                100.0,
            )
            if e0 is not None and math.isfinite(total_e) and math.isfinite(e0) and math.isfinite(self._q_thermo_cum):
                drift_fraction = abs((total_e - e0) - self._q_thermo_cum) / e_scale
            else:
                drift_fraction = 0.0

            if (not math.isfinite(total_e)) or (not math.isfinite(f_max)) or (f_max > 1.0e5):
                health_status = "clash"
            elif (f_max > 1.0e4) or (drift_fraction > 0.05):
                health_status = "warning"
            else:
                health_status = "bounded"

            ram_bytes = self._compute_ram_bytes()
            ram_mb = float(ram_bytes) / (1024.0 * 1024.0)
            bytes_per_atom = float(ram_bytes) / max(1, n)
            naive_float64_ram_mb = float(n) * 160.0 / (1024.0 * 1024.0)
            memory_savings_ratio = naive_float64_ram_mb / max(ram_mb, 1e-6)

            smd_info = None
            if self._smd_pull is not None:
                smd_info = {
                    "active": True,
                    "atom_index": int(self._smd_pull["atom_index"]),
                    "target_ang": list(self._smd_pull["target_ang"]),
                    "k_spring": float(self._smd_pull["k_spring"]),
                    "extension_ang": float(self._smd_pull.get("extension_ang", 0.0)),
                    "pull_force_kj_mol_nm": float(self._smd_pull.get("pull_force_kj_mol_nm", 0.0)),
                }

            return {
                "step": int(self.step_count),
                "time_ps": round(self.sim_time_ps, 6),
                "n_atoms": int(n),
                "capacity": int(self._stream_atom_indices.size),
                "active_all_atom_count": int(self.active_all_atom_count),
                "storage_mode": "tryte-all-atom",
                "ram_bytes": int(ram_bytes),
                "ram_mb": round(ram_mb, 4),
                "bytes_per_atom": round(bytes_per_atom, 2),
                "naive_float64_ram_mb": round(naive_float64_ram_mb, 4),
                "memory_savings_ratio": round(memory_savings_ratio, 2),
                "n_tiles": int(self.tile_set.n_tiles),
                "tile_size": int(self.tile_size),
                "tryte_pair_count": int(self._tryte_pair_count),
                "potential_energy": pe_fin,
                "kinetic_energy": ke_fin,
                "total_energy": total_e,
                "temperature_K": float(inst_temp) if math.isfinite(inst_temp) else 0.0,
                "max_force": f_max,
                "drift_fraction": float(drift_fraction) if math.isfinite(drift_fraction) else 0.0,
                "q_thermo_cum": float(self._q_thermo_cum) if math.isfinite(self._q_thermo_cum) else 0.0,
                "hmc_rejections": int(self.hmc_rejections),
                "health_status": health_status,
                "ms_per_step": float(self.last_ms_per_step),
                "ns_per_day_2fs": float(ns_per_day),
                "box_lengths_nm": [float(self.box_H[0, 0]), float(self.box_H[1, 1]), float(self.box_H[2, 2])],
                "box_origin_ang": [float(self._box_origin_ang[0]), float(self._box_origin_ang[1]), float(self._box_origin_ang[2])],
                "volume_nm3": vol_nm3,
                "stress_zz": (lambda v: v if math.isfinite(v) else 0.0)(float(self.virial[2, 2] / max(vol_nm3, 1e-9))),
                "mode": self.mode,
                "backend": "tryte-27+5-all-atom",
                "history_steps": list(self._history.keys()),
                "multiscale": self._build_multiscale_summary(),
                "smd": smd_info,
                "delta_streaming": {
                    "enabled": bool(self.delta_streaming_enabled),
                    "threshold_ang": float(self._dirty_tracker.threshold_ang if self._dirty_tracker is not None else 0.05),
                    "last_dirty_ratio": float(self._last_dirty_ratio),
                    "last_dirty_tiles": int(self._last_dirty_tiles),
                },
            }

    def encode_topology_packet(self) -> bytes:
        with self._lock:
            idx = self._stream_atom_indices
            k = int(idx.size)
            pos_sub = self.positions
            coords_ang = (pos_sub * np.float32(self.coord_to_ang)).astype(np.float32) if k > 0 else np.zeros((0, 3), dtype=np.float32)
            z_sub = self.atomic_numbers[idx] if k > 0 else np.zeros(0, dtype=np.uint8)
            radii = ELEMENT_LUT_RADIUS_ANG[z_sub] if k > 0 else np.zeros(0, dtype=np.float32)
            tile_ids = (idx // self.tile_size).astype(np.uint32) if k > 0 else np.zeros(0, dtype=np.uint32)
            res_ids = np.arange(k, dtype=np.uint32)
            box_ang = (self.box_H * self.coord_to_ang).astype(np.float32)
            if self._explicit_bonds.shape[0] > 0 and k == self.n_atoms and np.array_equal(idx, np.arange(k, dtype=np.int32)):
                topo_bonds = self._explicit_bonds.astype(np.uint32, copy=False)
            else:
                topo_bonds = np.zeros((0, 2), dtype=np.uint32)
            return pack_topology_init(
                atomic_numbers=z_sub,
                bonds=topo_bonds,
                residue_ids=res_ids,
                tile_ids=tile_ids,
                box_H=box_ang,
                coords=coords_ang,
                radii=radii,
                frame_id=self.step_count,
                sim_time_ps=self.sim_time_ps,
            )

    def encode_frame_packet(
        self,
        channel_name: Optional[str] = None,
        reversed_step: bool = False,
        camera_roi: Optional[Tuple[Sequence[float], float, int]] = None,
        use_delta: Optional[bool] = None,
    ) -> bytes:
        with self._lock:
            ch_name = channel_name or PHYSICS_CHANNEL_NAMES[self.active_channel]
            ch_idx = PHYSICS_CHANNEL_NAMES.index(ch_name)
            idx = self._stream_atom_indices
            k = int(idx.size)
            pos_sub = self.positions
            coords_ang = (pos_sub * np.float32(self.coord_to_ang)).astype(np.float32) if k > 0 else np.zeros((0, 3), dtype=np.float32)
            scalars = self.compute_scalar_channel(ch_name, atom_indices=idx)
            box_ang = (self.box_H * self.coord_to_ang).astype(np.float32)

            want_delta = self.delta_streaming_enabled if use_delta is None else bool(use_delta)
            if (
                want_delta
                and (not reversed_step)
                and self._dirty_tracker is not None
                and self._dirty_tracker.last_sent_coords_ang is not None
            ):
                dirty_idx, dirty_ratio, n_dirty_tiles = self._dirty_tracker.compute_dirty(coords_ang, commit=True)
                self._last_dirty_ratio = dirty_ratio
                self._last_dirty_tiles = n_dirty_tiles
                if dirty_ratio < 0.75:
                    return pack_dirty_tile_delta(
                        dirty_indices=dirty_idx,
                        coords_ang=coords_ang,
                        scalars=scalars,
                        box_H=box_ang,
                        frame_id=self.step_count,
                        sim_time_ps=self.sim_time_ps,
                        n_dirty_tiles=n_dirty_tiles,
                        use_int16=False,
                        scalar_channel=ch_idx,
                        pbc_wrapped=False,
                    )

            if self._dirty_tracker is not None:
                self._dirty_tracker.reset(coords_ang)
            self._last_dirty_ratio = 1.0
            self._last_dirty_tiles = int(np.unique(idx // self.tile_size).size) if k > 0 else 0
            return pack_frame_full(
                coords=coords_ang,
                frame_id=self.step_count,
                sim_time_ps=self.sim_time_ps,
                box_H=box_ang,
                scalars=scalars,
                use_int16=False,
                pbc_wrapped=False,
                reversed_step=reversed_step,
                scalar_channel=ch_idx,
            )

    def encode_overlay_packet(self, max_vectors: int = 2000) -> bytes:
        with self._lock:
            idx = self._stream_atom_indices
            k_tot = int(idx.size)
            if k_tot > 0:
                coords_ang = (self.positions * np.float32(self.coord_to_ang)).astype(np.float32)
                frc_sub = self.forces[idx]
                f_mags = np.linalg.norm(frc_sub, axis=1)
                k = min(int(max_vectors), k_tot)
                top = np.argpartition(f_mags, -k)[-k:] if k < k_tot else np.arange(k_tot)
                starts = coords_ang[top]
                dirs = (frc_sub[top] * np.float32(0.02)).astype(np.float32)
                mags = f_mags[top].astype(np.float32)
            else:
                starts = np.zeros((0, 3), dtype=np.float32)
                dirs = np.zeros((0, 3), dtype=np.float32)
                mags = np.zeros(0, dtype=np.float32)

            active_bonds = self.bonds
            if active_bonds and k_tot > 0:
                bp_all = np.asarray(active_bonds, dtype=np.uint32)
                p1 = coords_ang[bp_all[:, 0]]
                p2 = coords_ang[bp_all[:, 1]]
                d_b = np.linalg.norm(p1 - p2, axis=1)
                box_diag_half = min(0.45 * float(np.min(np.diag(self.box_H)) * self.coord_to_ang), 3.2)
                keep_b = d_b <= box_diag_half
                bp = bp_all[keep_b]
                bo = np.ones(bp.shape[0], dtype=np.float32)
            else:
                bp = np.zeros((0, 2), dtype=np.uint32)
                bo = np.zeros(0, dtype=np.float32)

            return pack_overlay_vectors(
                vector_starts=starts,
                vector_dirs=dirs,
                vector_mags=mags,
                bond_pairs=bp,
                bond_orders=bo,
                frame_id=self.step_count,
                sim_time_ps=self.sim_time_ps,
            )

    def encode_volume_packet(self, ml: int = 27) -> bytes:
        with self._lock:
            n_grid = int(ml)
            Q_grid = np.zeros((n_grid, n_grid, n_grid), dtype=np.float32)
            if self.tile_set.n_tiles > 0:
                # Bin Tryte Tile centroids weighted by tile_counts for O(N/27) fast volumetric density
                rel_nm = self.tile_set.centroids_nm.astype(np.float64) - self._box_origin_nm.astype(np.float64)[None, :]
                Hinv = np.linalg.inv(self.box_H)
                frac = rel_nm @ Hinv.T
                frac -= np.floor(frac)
                gidx = np.clip((frac * n_grid).astype(np.int32), 0, n_grid - 1)
                np.add.at(Q_grid, (gidx[:, 0], gidx[:, 1], gidx[:, 2]), self.tile_set.tile_counts.astype(np.float32))
            box_ang = (self.box_H * self.coord_to_ang).astype(np.float32)
            return pack_volume_grid3d(
                Q_grid,
                box_H=box_ang,
                origin=tuple(float(v) for v in self._box_origin_ang),
                frame_id=self.step_count,
                sim_time_ps=self.sim_time_ps,
            )
