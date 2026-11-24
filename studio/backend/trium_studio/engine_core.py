"""Unified low-compute, high-accuracy physics core for Biology-MM-Studio.

Connects directly to `biology_mm` native Apple M5 Metal 4 / ARM64 NEON libraries
and computes real-time observables across all 6 physics channels, Morton spatial
tile LOD filtering, GCMC dynamic slot deltas, 3D electrostatic grids, and
reversible step scrubbing.
"""

from __future__ import annotations

import ast
import ctypes
import math
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from biology_mm import (
    AtomicSystem,
    GPUProfile,
    GPUVerletPlan,
    NanostructureBuilder,
    ParameterEncoder,
    ParameterSpec,
    SystemBuilder,
    gpu_msm_stencil27,
    probe_hardware,
)
from biology_mm.native import _as_box_matrices, _encode_pair_keys, get_bridge

from .multiscale import (
    DirtyTileTracker,
    MultiscaleZonePartition,
    _find_neighbor_pairs_mic,
    assign_multiscale_zones,
    softcore_relax_clashes,
)
from .protocol import (
    pack_dirty_tile_delta,
    pack_frame_full,
    pack_frame_tile_patch,
    pack_gcmc_slot_delta,
    pack_overlay_vectors,
    pack_topology_init,
    pack_volume_grid3d,
)

PHYSICS_CHANNEL_NAMES = (
    "force-magnitude",
    "kinetic-temp",
    "charge-potential",
    "bond-order",
    "atomic-stress",
    "ternary-state",
    "multiscale-zone",
)

ELEMENT_Z: Dict[str, int] = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
    "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20,
    "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30,
    "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36, "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40,
    "Nb": 41, "Mo": 42, "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50, "Sb": 51,
    "Te": 52, "I": 53, "Xe": 54, "Cs": 55, "Ba": 56, "La": 57, "Ce": 58, "Nd": 60, "Eu": 63, "Gd": 64,
    "Hf": 72, "Ta": 73, "W": 74, "Re": 75, "Os": 76, "Ir": 77, "Pt": 78, "Au": 79, "Hg": 80, "Tl": 81,
    "Pb": 82, "Bi": 83, "Th": 90, "U": 92, "X": 6,
}

ELEMENT_RADII_ANG: Dict[int, float] = {
    1: 1.10, 2: 1.40, 3: 1.82, 4: 1.53, 5: 1.92, 6: 1.70, 7: 1.55, 8: 1.52, 9: 1.47, 10: 1.54,
    11: 2.27, 12: 1.73, 13: 1.84, 14: 2.10, 15: 1.80, 16: 1.80, 17: 1.75, 18: 1.88, 19: 2.75, 20: 2.31,
    21: 2.11, 22: 2.00, 23: 1.95, 24: 1.90, 25: 1.90, 26: 2.00, 27: 1.85, 28: 1.63, 29: 1.40, 30: 1.39,
    31: 1.87, 32: 2.11, 33: 1.85, 34: 1.90, 35: 1.85, 36: 2.02, 37: 3.03, 38: 2.49, 39: 2.19, 40: 2.06,
    41: 2.07, 42: 2.09, 44: 2.05, 45: 2.00, 46: 1.63, 47: 1.72, 48: 1.58, 49: 1.93, 50: 2.17, 51: 2.06,
    52: 2.06, 53: 1.98, 54: 2.16, 55: 3.43, 56: 2.68, 57: 2.40, 58: 2.35, 60: 2.29, 63: 2.33, 64: 2.37,
    72: 2.12, 73: 2.17, 74: 2.10, 75: 2.05, 76: 2.00, 77: 2.00, 78: 1.75, 79: 1.66, 80: 1.55, 81: 1.96,
    82: 2.02, 83: 2.07, 90: 2.37, 92: 1.86,
}

KB_KJ_MOL_K = 0.00831446261815324
MAX_SNAPSHOT_RAM_BYTES = 64 * 1024 * 1024  # 64 MB hard cap on StepSnapshot history

_SAFE_MATH: Dict[str, Any] = {
    "exp": np.exp,
    "log": np.log,
    "log10": np.log10,
    "sqrt": np.sqrt,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "sinh": np.sinh,
    "cosh": np.cosh,
    "tanh": np.tanh,
    "arcsin": np.arcsin,
    "arccos": np.arccos,
    "arctan": np.arctan,
    "abs": np.abs,
    "min": np.minimum,
    "max": np.maximum,
    "clip": np.clip,
    "pi": math.pi,
    "e": math.e,
}

_ALLOWED_AST_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Call,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.USub,
    ast.UAdd,
)


def compile_safe_radial_expr(expr: str) -> Callable[[np.ndarray, Dict[str, float]], np.ndarray]:
    """Compile a restricted mathematical `U(r)` expression using `r` and parameter names."""
    tree = ast.parse(expr.strip(), mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_AST_NODES):
            raise ValueError(f"Disallowed AST construct '{type(node).__name__}' in custom U(r) expression")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _SAFE_MATH or not callable(_SAFE_MATH[node.func.id]):
                fn_name = getattr(node.func, "id", type(node.func).__name__)
                raise ValueError(f"Disallowed function '{fn_name}' in custom U(r) expression")

    code = compile(tree, "<custom_potential_expr>", "eval")

    def evaluator(r: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        env: Dict[str, Any] = dict(_SAFE_MATH)
        env["r"] = np.asarray(r, dtype=np.float64)
        for k, v in params.items():
            env[str(k)] = float(v)
        val = eval(code, {"__builtins__": {}}, env)  # noqa: S307 - AST-restricted mathematical expression
        arr = np.broadcast_to(np.asarray(val, dtype=np.float64), np.shape(env["r"])).copy()
        return arr

    return evaluator


class CubicSplineRadialTable:
    """Exact natural C2 cubic spline evaluator matching FoundationCore's `CubicSplineTable`."""

    def __init__(self, r_grid: np.ndarray, u_grid: np.ndarray, mode: str = "lfs") -> None:
        r = np.ascontiguousarray(r_grid, dtype=np.float64)
        u = np.ascontiguousarray(u_grid, dtype=np.float64)
        if r.ndim != 1 or r.size < 4 or r.shape != u.shape:
            raise ValueError("CubicSplineRadialTable requires matching 1D arrays of length >= 4")
        if not np.all(np.diff(r) > 0):
            raise ValueError("r_grid must be strictly increasing")
        self.r = r
        self.u_raw = u
        n = r.size
        h = np.diff(r)

        alpha = np.zeros(n, dtype=np.float64)
        for i in range(1, n - 1):
            alpha[i] = (3.0 / h[i]) * (u[i + 1] - u[i]) - (3.0 / h[i - 1]) * (u[i] - u[i - 1])

        l = np.ones(n, dtype=np.float64)
        mu = np.zeros(n, dtype=np.float64)
        z = np.zeros(n, dtype=np.float64)
        for i in range(1, n - 1):
            l[i] = 2.0 * (r[i + 1] - r[i - 1]) - h[i - 1] * mu[i - 1]
            mu[i] = h[i] / l[i]
            z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]

        self.c = np.zeros(n, dtype=np.float64)
        self.b = np.zeros(n - 1, dtype=np.float64)
        self.d = np.zeros(n - 1, dtype=np.float64)
        self.a = u[:-1].copy()
        for j in range(n - 2, -1, -1):
            self.c[j] = z[j] - mu[j] * self.c[j + 1]
            self.b[j] = (u[j + 1] - u[j]) / h[j] - h[j] * (self.c[j + 1] + 2.0 * self.c[j]) / 3.0
            self.d[j] = (self.c[j + 1] - self.c[j]) / (3.0 * h[j])

        self.rc = float(r[-1])
        self.mode = mode
        if mode == "lfs":
            self.uc, self.duc = self._eval_raw_scalar(self.rc)
        else:
            self.uc, self.duc = 0.0, 0.0

    def _eval_raw_scalar(self, rv: float) -> Tuple[float, float]:
        r = self.r
        if rv <= r[0]:
            s0 = float(self.b[0])
            return float(self.a[0] + s0 * (rv - r[0])), s0
        if rv >= r[-1]:
            dx = rv - r[-2]
            du = float(self.b[-1] + 2.0 * self.c[-2] * dx + 3.0 * self.d[-1] * dx * dx)
            u_end = float(self.u_raw[-1])
            return u_end + du * (rv - r[-1]), du
        idx = int(np.searchsorted(r, rv, side="right")) - 1
        idx = max(0, min(idx, r.size - 2))
        dx = rv - r[idx]
        u = self.a[idx] + dx * (self.b[idx] + dx * (self.c[idx] + dx * self.d[idx]))
        du = self.b[idx] + dx * (2.0 * self.c[idx] + 3.0 * self.d[idx] * dx)
        return float(u), float(du)

    def eval_array(self, r_vals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        rv = np.asarray(r_vals, dtype=np.float64)
        u_out = np.zeros_like(rv)
        du_out = np.zeros_like(rv)
        active = rv < self.rc
        if not np.any(active):
            return u_out, du_out

        ra = rv[active]
        idx = np.clip(np.searchsorted(self.r, ra, side="right") - 1, 0, self.r.size - 2)
        dx = ra - self.r[idx]
        u = self.a[idx] + dx * (self.b[idx] + dx * (self.c[idx] + dx * self.d[idx]))
        du = self.b[idx] + dx * (2.0 * self.c[idx] + 3.0 * self.d[idx] * dx)

        below = ra < self.r[0]
        if np.any(below):
            db = ra[below] - self.r[0]
            du_b = self.b[0]
            u[below] = self.a[0] + du_b * db
            du[below] = du_b

        if self.mode == "lfs":
            u = u - self.uc - self.duc * (ra - self.rc)
            du = du - self.duc

        u_out[active] = u
        du_out[active] = du
        return u_out, du_out


def compute_morton_tiles(
    positions: np.ndarray,
    box_H: np.ndarray,
    tile_size: int = 32,
) -> np.ndarray:
    """
    Assign each atom to a spatial 3D Morton (Z-order) tile index (mirrored from TriumXBT
    32-atom Morton tiling) so millions of atoms stream by spatially coherent tiles.
    """
    n = int(positions.shape[0])
    if n == 0:
        return np.zeros(0, dtype=np.uint32)
    H = np.asarray(box_H, dtype=np.float64)
    Hinv = np.linalg.inv(H)
    frac = positions @ Hinv.T
    frac = frac - np.floor(frac)
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

    morton = _split_bits(gx) | (_split_bits(gy) << 1) | (_split_bits(gz) << 2)
    # Bin spatially sorted atoms into `tile_size`-sized chunks for uniform WebGL texel alignment
    order = np.argsort(morton, kind="mergesort")
    tile_ids = np.empty(n, dtype=np.uint32)
    tile_ids[order] = (np.arange(n, dtype=np.uint32) // max(1, int(tile_size)))
    return tile_ids


def select_roi_atoms(
    positions_ang: np.ndarray,
    camera_center_ang: Sequence[float],
    radius_ang: float,
    max_atoms: int = 262_144,
) -> np.ndarray:
    """Select closest ROI atom indices around `camera_center_ang` within `radius_ang`."""
    c = np.asarray(camera_center_ang, dtype=np.float64).reshape((1, 3))
    d2 = np.sum((positions_ang - c) ** 2, axis=1)
    r2 = float(radius_ang) ** 2
    inside = np.flatnonzero(d2 <= r2)
    if inside.size > max_atoms:
        sub = np.argpartition(d2[inside], max_atoms)[:max_atoms]
        inside = inside[sub]
    return np.sort(inside).astype(np.uint32)


class LatestFrameBuffer:
    """
    Bounded 2-slot frame buffer preventing WebSocket backpressure when the physics
    engine produces frames faster than the browser `requestAnimationFrame` renders.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest_binary: Optional[bytes] = None
        self._latest_metrics: Optional[Dict[str, Any]] = None
        self._pending_volume: Optional[bytes] = None
        self._dropped_frames: int = 0
        self._sequence: int = 0

    def push(
        self,
        binary_frame: bytes,
        metrics: Dict[str, Any],
        volume_packet: Optional[bytes] = None,
    ) -> None:
        with self._lock:
            if self._latest_binary is not None:
                self._dropped_frames += 1
            self._latest_binary = binary_frame
            self._latest_metrics = metrics
            if volume_packet is not None:
                self._pending_volume = volume_packet
            self._sequence += 1

    def pop_latest(self) -> Tuple[Optional[bytes], Optional[Dict[str, Any]], Optional[bytes], int]:
        with self._lock:
            frame = self._latest_binary
            metrics = self._latest_metrics
            vol = self._pending_volume
            dropped = self._dropped_frames
            self._latest_binary = None
            self._latest_metrics = None
            self._pending_volume = None
            return frame, metrics, vol, dropped


@dataclass
class StepSnapshot:
    step: int
    time_ps: float
    positions: np.ndarray
    momenta: np.ndarray
    forces: np.ndarray
    box_H: np.ndarray
    potential_energy: float
    kinetic_energy: float
    scalars: Dict[str, np.ndarray]


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


def _build_spatial_gpu_verlet_plan(
    positions: np.ndarray,
    sigma: np.ndarray,
    epsilon: np.ndarray,
    charges: Optional[np.ndarray],
    box: Optional[Union[Sequence[float], np.ndarray]],
    exclusions: Optional[Sequence[Sequence[int]]],
    rcut: float,
    skin: float = 0.2,
    alpha_dsf: float = 0.0,
    coulomb_prefactor: float = 138.935456,
    use_lfs: bool = True,
    include_self: bool = True,
    backend: str = "auto",
    min_atoms_for_cells: int = 2048,
) -> GPUVerletPlan:
    """
    Build a `GPUVerletPlan` using O(N) periodic linked-cell 13-neighbor half-shell
    spatial tile culling so >=100k-atom systems initialize in milliseconds while remaining
    bit-exact with full all-pairs tile enumeration.
    """
    pos = np.ascontiguousarray(positions, dtype=np.float64)
    n = int(pos.shape[0])
    if n < min_atoms_for_cells or box is None:
        return GPUVerletPlan(
            positions=pos,
            sigma=sigma,
            epsilon=epsilon,
            charges=charges,
            box=box,
            exclusions=exclusions,
            rcut=rcut,
            skin=skin,
            alpha_dsf=alpha_dsf,
            coulomb_prefactor=coulomb_prefactor,
            use_lfs=use_lfs,
            include_self=include_self,
            backend=backend,
        )

    H_arr = np.asarray(box, dtype=np.float64)
    if H_arr.shape == (3,):
        H_mat = np.diag(H_arr).astype(np.float64)
    elif H_arr.shape == (3, 3):
        H_mat = H_arr
    else:
        raise ValueError(f"Unsupported box shape {H_arr.shape}")

    Hinv = np.linalg.inv(H_mat)
    # Perpendicular slab thickness between crystal planes along each axis
    d_perp = 1.0 / np.maximum(np.linalg.norm(Hinv, axis=1), 1e-12)
    r_list = float(rcut) + max(float(skin), 0.02)
    nx = min(256, int(math.floor(float(d_perp[0]) / r_list)))
    ny = min(256, int(math.floor(float(d_perp[1]) / r_list)))
    nz = min(256, int(math.floor(float(d_perp[2]) / r_list)))

    if min(nx, ny, nz) < 3:
        return GPUVerletPlan(
            positions=pos,
            sigma=sigma,
            epsilon=epsilon,
            charges=charges,
            box=box,
            exclusions=exclusions,
            rcut=rcut,
            skin=skin,
            alpha_dsf=alpha_dsf,
            coulomb_prefactor=coulomb_prefactor,
            use_lfs=use_lfs,
            include_self=include_self,
            backend=backend,
        )

    bridge = get_bridge()
    want_gpu = (backend != "cpu") and bridge.has_gpu_bridge
    if (not want_gpu) and (not bridge.has_cpu_bridge):
        raise RuntimeError("No native Trium-XBT CPU or GPU bridge library available")

    sig = np.ascontiguousarray(sigma, dtype=np.float64)
    eps = np.ascontiguousarray(epsilon, dtype=np.float64)
    chg = np.zeros(n, dtype=np.float64) if charges is None else np.ascontiguousarray(charges, dtype=np.float64)

    # Sort atoms by periodic linked-cell ID and bin each cell into <= TILE_SIZE tiles
    frac = pos @ Hinv.T
    frac -= np.floor(frac)
    cx = np.clip((frac[:, 0] * nx).astype(np.int32), 0, nx - 1)
    cy = np.clip((frac[:, 1] * ny).astype(np.int32), 0, ny - 1)
    cz = np.clip((frac[:, 2] * nz).astype(np.int32), 0, nz - 1)
    cid = cx + nx * (cy + ny * cz)
    perm = np.argsort(cid, kind="mergesort")
    cid_sorted = cid[perm]
    order = (perm + 1).astype(np.int32)

    unique_cids, start_idx, counts = np.unique(cid_sorted, return_index=True, return_counts=True)
    n_cells = nx * ny * nz
    cell_t_lo = np.zeros(n_cells, dtype=np.int32)
    cell_t_hi = np.zeros(n_cells, dtype=np.int32)

    tile_size = GPUVerletPlan.TILE_SIZE
    tile_starts_list: List[int] = []
    tile_ends_list: List[int] = []
    ti_parts: List[np.ndarray] = []
    tj_parts: List[np.ndarray] = []

    next_tile1 = 1
    for u_i in range(unique_cids.size):
        c_val = int(unique_cids[u_i])
        s0 = int(start_idx[u_i])
        cnt = int(counts[u_i])
        e0 = s0 + cnt
        t_lo = next_tile1
        for a in range(s0, e0, tile_size):
            tile_starts_list.append(a + 1)
            tile_ends_list.append(min(a + tile_size, e0))
            next_tile1 += 1
        t_hi = next_tile1
        cell_t_lo[c_val] = t_lo
        cell_t_hi[c_val] = t_hi


    # Vectorized intra-cell pairs
    diff_cell = cell_t_hi - cell_t_lo
    max_intra = int(np.max(diff_cell)) if diff_cell.size else 0
    if max_intra > 1:
        mask_multi = diff_cell > 1
        lo_m = cell_t_lo[mask_multi]
        diff_m = diff_cell[mask_multi]
        for da in range(max_intra):
            for db in range(da + 1, max_intra):
                m_sub = diff_m > db
                if np.any(m_sub):
                    ti_parts.append((lo_m[m_sub] + da).astype(np.int32))
                    tj_parts.append((lo_m[m_sub] + db).astype(np.int32))

    # Same-tile diagonal pairs (1..n_tiles), emitted exactly once per tile
    all_tiles = np.arange(1, next_tile1, dtype=np.int32)
    ti_parts.append(all_tiles)
    tj_parts.append(all_tiles)

    # Cross-cell forward 13-neighbor pairs
    ucx = unique_cids % nx
    ucy = (unique_cids // nx) % ny
    ucz = unique_cids // (nx * ny)
    src_lo_all = cell_t_lo[unique_cids]
    src_hi_all = cell_t_hi[unique_cids]

    for dx, dy, dz in _HALF_SHELL_OFFSETS:
        ncx = (ucx + dx) % nx
        ncy = (ucy + dy) % ny
        ncz = (ucz + dz) % nz
        nb_cid = ncx + nx * (ncy + ny * ncz)
        nb_lo = cell_t_lo[nb_cid]
        nb_hi = cell_t_hi[nb_cid]
        mask = nb_hi > nb_lo
        if not np.any(mask):
            continue
        s_lo = src_lo_all[mask]
        s_hi = src_hi_all[mask]
        d_lo = nb_lo[mask]
        d_hi = nb_hi[mask]
        single_mask = (s_hi - s_lo == 1) & (d_hi - d_lo == 1)
        if np.all(single_mask):
            ti_parts.append(np.minimum(s_lo, d_lo))
            tj_parts.append(np.maximum(s_lo, d_lo))
        else:
            if np.any(single_mask):
                ti_parts.append(np.minimum(s_lo[single_mask], d_lo[single_mask]))
                tj_parts.append(np.maximum(s_lo[single_mask], d_lo[single_mask]))
            multi_mask = ~single_mask
            s_lo_m = s_lo[multi_mask]
            s_hi_m = s_hi[multi_mask]
            d_lo_m = d_lo[multi_mask]
            d_hi_m = d_hi[multi_mask]
            diff_s = s_hi_m - s_lo_m
            diff_d = d_hi_m - d_lo_m
            max_s = int(np.max(diff_s)) if diff_s.size else 0
            max_d = int(np.max(diff_d)) if diff_d.size else 0
            for da in range(max_s):
                mask_a = diff_s > da
                ta = s_lo_m[mask_a] + da
                for db in range(max_d):
                    mask_b = (diff_d > db)[mask_a]
                    if not np.any(mask_b):
                        continue
                    curr_ta = ta[mask_b]
                    curr_tb = d_lo_m[mask_a][mask_b] + db
                    swap = curr_ta > curr_tb
                    ti_parts.append(np.where(swap, curr_tb, curr_ta).astype(np.int32))
                    tj_parts.append(np.where(swap, curr_ta, curr_tb).astype(np.int32))

    tile_starts = np.asarray(tile_starts_list, dtype=np.int32)
    tile_ends = np.asarray(tile_ends_list, dtype=np.int32)
    tile_i = np.ascontiguousarray(np.concatenate(ti_parts), dtype=np.int32)
    tile_j = np.ascontiguousarray(np.concatenate(tj_parts), dtype=np.int32)

    is_tri, H_col, Hinv_col = _as_box_matrices(box)
    H_ptr = H_col.ctypes.data_as(ctypes.POINTER(ctypes.c_double)) if H_col is not None else ctypes.POINTER(ctypes.c_double)()
    Hinv_ptr = Hinv_col.ctypes.data_as(ctypes.POINTER(ctypes.c_double)) if Hinv_col is not None else ctypes.POINTER(ctypes.c_double)()

    excl_keys = _encode_pair_keys(exclusions, n, with_weights=False)
    excl_ptr = excl_keys.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)) if excl_keys.size else ctypes.POINTER(ctypes.c_uint64)()
    sk_ptr = ctypes.POINTER(ctypes.c_uint64)()
    sw_ptr = ctypes.POINTER(ctypes.c_double)()

    ptr = None
    using_gpu = False
    if want_gpu:
        assert bridge.gpu is not None
        c_ptr = bridge.gpu.xbt_gpu_verlet_plan_create(
            ctypes.c_size_t(n),
            pos.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            sig.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            eps.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            chg.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            H_ptr,
            Hinv_ptr,
            ctypes.c_int(is_tri),
            ctypes.c_size_t(tile_i.size),
            tile_i.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            tile_j.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            order.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            tile_starts.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            tile_ends.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ctypes.c_size_t(excl_keys.size),
            excl_ptr,
            ctypes.c_size_t(0),
            sk_ptr,
            sw_ptr,
            ctypes.c_double(0.5),
            ctypes.c_double(1.0 / 1.2),
            ctypes.c_double(float(rcut)),
            ctypes.c_double(float(skin)),
            ctypes.c_double(float(alpha_dsf)),
            ctypes.c_double(float(coulomb_prefactor)),
            ctypes.c_int(1 if use_lfs else 0),
        )
        if c_ptr:
            ptr = int(c_ptr)
            using_gpu = True

    if ptr is None:
        if not bridge.has_cpu_bridge:
            raise RuntimeError("No native Trium-XBT CPU or GPU bridge library available")
        assert bridge.mm is not None
        c_ptr = bridge.mm.xbt_verlet_plan_create(
            ctypes.c_size_t(n),
            pos.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            sig.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            eps.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            chg.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            H_ptr,
            Hinv_ptr,
            ctypes.c_int(is_tri),
            ctypes.c_size_t(tile_i.size),
            tile_i.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            tile_j.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            order.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            tile_starts.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            tile_ends.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ctypes.c_size_t(excl_keys.size),
            excl_ptr,
            ctypes.c_size_t(0),
            sk_ptr,
            sw_ptr,
            ctypes.c_double(0.5),
            ctypes.c_double(1.0 / 1.2),
            ctypes.c_double(float(rcut)),
            ctypes.c_double(float(skin)),
            ctypes.c_double(float(alpha_dsf)),
            ctypes.c_double(float(coulomb_prefactor)),
            ctypes.c_int(1 if use_lfs else 0),
        )
        if not c_ptr:
            raise RuntimeError("xbt_verlet_plan_create failed")
        ptr = int(c_ptr)
        using_gpu = False

    plan_obj = object.__new__(GPUVerletPlan)
    plan_obj._bridge = bridge
    plan_obj._ptr = ptr
    plan_obj._using_gpu = using_gpu
    plan_obj.n_atoms = n
    plan_obj.rcut = float(rcut)
    plan_obj.skin = float(skin)
    plan_obj.alpha_dsf = float(alpha_dsf)
    plan_obj.coulomb_prefactor = float(coulomb_prefactor)
    plan_obj.use_lfs = bool(use_lfs)
    plan_obj.include_self = bool(include_self)
    plan_obj._sigma = sig
    plan_obj._epsilon = eps
    plan_obj._charges = chg
    plan_obj._default_box = box
    plan_obj._keepalive = (pos, sig, eps, chg, order, tile_starts, tile_ends, tile_i, tile_j, excl_keys)
    plan_obj._self_energy = 0.0
    if include_self and alpha_dsf > 0.0 and coulomb_prefactor != 0.0:
        erfc_rc = math.erfc(float(alpha_dsf) * float(rcut))
        plan_obj._self_energy = -float(coulomb_prefactor) * (
            erfc_rc / (2.0 * float(rcut)) + float(alpha_dsf) / math.sqrt(math.pi)
        ) * float(np.sum(chg * chg))
    return plan_obj


def _extract_angles_and_dihedrals(
    bonds: Sequence[Tuple[int, int]],
) -> Tuple[List[Tuple[int, int, int]], List[Tuple[int, int, int, int]]]:
    """Extract harmonic angle triplets (i, j, k) and proper dihedral quartets (i, j, k, l) from covalent bonds."""
    from collections import defaultdict

    adj: Dict[int, Set[int]] = defaultdict(set)
    for u, v in bonds:
        if u != v:
            adj[int(u)].add(int(v))
            adj[int(v)].add(int(u))

    angles: List[Tuple[int, int, int]] = []
    for j, nbrs_set in adj.items():
        nbrs = sorted(nbrs_set)
        for idx1 in range(len(nbrs)):
            i = nbrs[idx1]
            for idx2 in range(idx1 + 1, len(nbrs)):
                k = nbrs[idx2]
                angles.append((i, j, k))

    dihedrals_set: Set[Tuple[int, int, int, int]] = set()
    for u, v in bonds:
        u_i, v_i = int(u), int(v)
        for j, k in ((u_i, v_i), (v_i, u_i)):
            for i in adj[j]:
                if i == k:
                    continue
                for l in adj[k]:
                    if l == j or l == i:
                        continue
                    quartet = (i, j, k, l) if i < l else (l, k, j, i)
                    dihedrals_set.add(quartet)

    dihedrals = sorted(list(dihedrals_set))
    return angles, dihedrals


class SimulationSession:
    """
    Stateful molecular & nanotech simulation session powering the Studio UI.

    All internal dynamics use the system's physical units (`nm, kJ/mol, amu, ps`
    or `reduced`), while `TriumStream v1` packets convert coordinates and box
    matrices cleanly by `coord_to_angstrom` (`10.0` for `nm`, `3.4` for `reduced`)
    so Mol* renders exact physical Ångström scales.
    """

    def __init__(
        self,
        system: AtomicSystem,
        mode: str = "metal_verlet",
        dt: float = 0.002,
        temperature: float = 300.0,
        gamma: float = 0.5,
        sigma: float | np.ndarray = 0.34,
        epsilon: float | np.ndarray = 0.997,
        rcut: float = 1.0,
        alpha_dsf: float = 0.0,
        use_int16: bool = False,
        max_history: int = 240,
        custom_expr: Optional[str] = None,
        custom_params: Optional[Dict[str, float]] = None,
        param_specs: Optional[List[ParameterSpec]] = None,
        gcmc_mu: float = -2.0,
        efield_z: float = 0.0,
        strain_rate: float = 0.0,
        pore_radius_nm: float = 0.45,
        max_atoms_cap: Optional[int] = None,
        structure_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.system = system.copy()
        self.mode = mode
        self.structure_metadata: Optional[Dict[str, Any]] = dict(structure_metadata) if structure_metadata else None
        self.dt = float(dt)
        self.temperature = float(temperature)
        self.gamma = float(gamma)
        self.rcut = float(rcut)
        self.alpha_dsf = float(alpha_dsf)
        self.use_int16 = bool(use_int16)
        self.custom_expr = custom_expr
        self.custom_params = dict(custom_params or {})
        self.param_encoder = ParameterEncoder(param_specs) if param_specs else None
        self.gcmc_mu = float(gcmc_mu)
        self.efield_z = float(efield_z)
        self.strain_rate = float(strain_rate)
        self.pore_radius_nm = float(pore_radius_nm)
        self._rng = np.random.default_rng(202611)
        self._lock = threading.RLock()

        n = self.system.n_atoms
        self.n_real = n
        self.capacity = max(n, int(max_atoms_cap)) if max_atoms_cap is not None else n
        self.coord_to_ang = 10.0 if self.system.units == "md" else 3.4
        self.k_b = KB_KJ_MOL_K if self.system.units == "md" else 1.0

        self.positions = np.zeros((self.capacity, 3), dtype=np.float64)
        self.momenta = np.zeros((self.capacity, 3), dtype=np.float64)
        self.forces = np.zeros((self.capacity, 3), dtype=np.float64)
        self.masses = np.ones(self.capacity, dtype=np.float64)
        self.charges = np.zeros(self.capacity, dtype=np.float64)
        self.sigma = np.full(self.capacity, float(sigma) if np.ndim(sigma) == 0 else 0.34, dtype=np.float64)
        self.epsilon = np.full(self.capacity, float(epsilon) if np.ndim(epsilon) == 0 else 0.997, dtype=np.float64)
        if np.ndim(sigma) == 0 and np.ndim(epsilon) == 0 and self.system.units == "md" and getattr(self.system, "names", None):
            try:
                from biology_mm.openff_amber_bridge import _OPEN_LJ_TABLE
                for _idx_el, _nm in enumerate(self.system.names[: self.n_real]):
                    _key = "".join(c for c in str(_nm) if c.isalpha()).upper()
                    if _key in _OPEN_LJ_TABLE:
                        _s_nm, _e_kj, _, _ = _OPEN_LJ_TABLE[_key]
                        if _s_nm > 0.0:
                            self.sigma[_idx_el] = _s_nm
                            self.epsilon[_idx_el] = max(0.02, _e_kj)
            except Exception:
                pass
        self.active_mask = np.zeros(self.capacity, dtype=bool)
        self.active_mask[:n] = True

        self.positions[:n] = self.system.positions
        self.masses[:n] = self.system.masses
        self.charges[:n] = self.system.charges
        if np.ndim(sigma) > 0:
            self.sigma[:n] = np.asarray(sigma, dtype=np.float64)
        if np.ndim(epsilon) > 0:
            self.epsilon[:n] = np.asarray(epsilon, dtype=np.float64)

        self.names = list(self.system.names) + ["Ar"] * (self.capacity - n)
        self.atomic_numbers = np.array(
            [
                ELEMENT_Z.get(
                    ("".join(c for c in str(nm) if c.isalpha())[:1].upper() + "".join(c for c in str(nm) if c.isalpha())[1:].lower()),
                    ELEMENT_Z.get(str(nm).strip(), 6),
                )
                for nm in self.names
            ],
            dtype=np.uint8,
        )
        self.radii_ang = np.array([ELEMENT_RADII_ANG.get(int(z), 1.5) for z in self.atomic_numbers], dtype=np.float32)
        if self.capacity > n:
            self.radii_ang[n:] = 0.0  # vacant GCMC slots hidden by radius 0

        lx, ly, lz = self.system.box_lengths
        self.box_H = np.diag([lx, ly, lz]).astype(np.float64)
        self.bonds = [(int(i), int(j)) for i, j in self.system.bonds]
        if not self.bonds and 1 < self.n_real <= 8000 and self.mode == "metal_verlet" and self.system.units == "md":
            try:
                from .structure_loader import _infer_bonds_pbc
                inferred = _infer_bonds_pbc(self.positions[: self.n_real], self.names[: self.n_real], box_nm=self.box_H)
                if inferred:
                    self.bonds = [(int(i), int(j)) for i, j in inferred]
            except Exception:
                pass
        self.bond_orders = np.ones(len(self.bonds), dtype=np.float32)
        if self.bonds and self.n_real > 0:
            self._bond_pairs = np.asarray(self.bonds, dtype=np.int32)
            _, r0_vec = self._mic_vectors(self._bond_pairs[:, 0], self._bond_pairs[:, 1], self.positions[: self.n_real])
            self._bond_r0 = np.clip(r0_vec, 0.08, 0.35)
            ang_list, dih_list = _extract_angles_and_dihedrals(self.bonds)
            deg_counts: Dict[int, int] = {}
            for u_b, v_b in self.bonds:
                deg_counts[int(u_b)] = deg_counts.get(int(u_b), 0) + 1
                deg_counts[int(v_b)] = deg_counts.get(int(v_b), 0) + 1

            # Filter out hyper-coordinated metallic core angles (deg > 8) and collinear 180-deg trans angles
            cand_ang = [
                (i_a, j_a, k_a)
                for (i_a, j_a, k_a) in ang_list
                if deg_counts.get(int(j_a), 0) <= 8
            ]
            if cand_ang:
                cand_tri = np.asarray(cand_ang, dtype=np.int32)
                ai, aj, ak = cand_tri[:, 0], cand_tri[:, 1], cand_tri[:, 2]
                v1, r1 = self._mic_vectors(ai, aj, self.positions[: self.n_real])
                v2, r2 = self._mic_vectors(ak, aj, self.positions[: self.n_real])
                cos_th = np.clip(
                    np.sum(v1 * v2, axis=1) / (np.maximum(r1, 1e-6) * np.maximum(r2, 1e-6)),
                    -0.999999,
                    0.999999,
                )
                # Exclude near-180-deg trans angles (|cos_th| > 0.95) that hit the arccos cusp singularity
                keep_ang = np.abs(cos_th) <= 0.95
                if np.any(keep_ang):
                    self._angle_triplets = cand_tri[keep_ang]
                    self._angle_theta0 = np.arccos(cos_th[keep_ang])
                else:
                    self._angle_triplets = np.zeros((0, 3), dtype=np.int32)
                    self._angle_theta0 = np.zeros(0, dtype=np.float64)
            else:
                self._angle_triplets = np.zeros((0, 3), dtype=np.int32)
                self._angle_theta0 = np.zeros(0, dtype=np.float64)

            if dih_list:
                filtered_dih = [
                    (i_d, j_d, k_d, l_d)
                    for (i_d, j_d, k_d, l_d) in dih_list
                    if deg_counts.get(int(j_d), 0) <= 4 and deg_counts.get(int(k_d), 0) <= 4
                ]
            else:
                filtered_dih = []

            if filtered_dih:
                cand_q = np.asarray(filtered_dih, dtype=np.int32)
                di, dj, dk, dl = cand_q[:, 0], cand_q[:, 1], cand_q[:, 2], cand_q[:, 3]
                r_ij, rij_len = self._mic_vectors(di, dj, self.positions[: self.n_real])
                r_kj, rkj_len = self._mic_vectors(dk, dj, self.positions[: self.n_real])
                r_kl, rkl_len = self._mic_vectors(dk, dl, self.positions[: self.n_real])
                m = np.cross(r_ij, r_kj)
                n_vec = np.cross(r_kj, r_kl)
                m_len = np.sqrt(np.maximum(np.sum(m * m, axis=1), 1e-12))
                n_len = np.sqrt(np.maximum(np.sum(n_vec * n_vec, axis=1), 1e-12))
                sin_ijk = m_len / np.maximum(rij_len * rkj_len, 1e-6)
                sin_jkl = n_len / np.maximum(rkj_len * rkl_len, 1e-6)
                keep_dih = (sin_ijk > 0.30) & (sin_jkl > 0.30)
                if np.any(keep_dih):
                    self._dihedral_quartets = cand_q[keep_dih]
                    m_k = m[keep_dih]
                    n_k = n_vec[keep_dih]
                    m_l = m_len[keep_dih]
                    n_l = n_len[keep_dih]
                    rkj_k = r_kj[keep_dih]
                    rkj_l = rkj_len[keep_dih]
                    cos_phi = np.clip(np.sum(m_k * n_k, axis=1) / (m_l * n_l), -0.999999, 0.999999)
                    sin_phi = np.sum(np.cross(m_k, n_k) * rkj_k, axis=1) / (m_l * n_l * np.maximum(rkj_l, 1e-6))
                    self._dihedral_phi0 = np.arctan2(sin_phi, cos_phi)
                else:
                    self._dihedral_quartets = np.zeros((0, 4), dtype=np.int32)
                    self._dihedral_phi0 = np.zeros(0, dtype=np.float64)
            else:
                self._dihedral_quartets = np.zeros((0, 4), dtype=np.int32)
                self._dihedral_phi0 = np.zeros(0, dtype=np.float64)

            # Build comprehensive 1-2 (bond) and 1-3 (valence angle) non-bonded exclusions
            excl_set = set()
            for u, v in self.bonds:
                u_i, v_i = int(u), int(v)
                if u_i != v_i:
                    excl_set.add((min(u_i, v_i), max(u_i, v_i)))
            for i, j, k in ang_list:
                i_i, k_i = int(i), int(k)
                if i_i != k_i:
                    excl_set.add((min(i_i, k_i), max(i_i, k_i)))
            self._exclusion_pairs = sorted(list(excl_set))
        else:
            self._bond_pairs = np.zeros((0, 2), dtype=np.int32)
            self._bond_r0 = np.zeros(0, dtype=np.float64)
            self._angle_triplets = np.zeros((0, 3), dtype=np.int32)
            self._angle_theta0 = np.zeros(0, dtype=np.float64)
            self._dihedral_quartets = np.zeros((0, 4), dtype=np.int32)
            self._dihedral_phi0 = np.zeros(0, dtype=np.float64)
            self._exclusion_pairs = []

        # Initialize Boltzmann momenta for real atoms with zero net COM drift
        v_scale = np.sqrt(self.k_b * max(1e-6, self.temperature) / self.masses[:n])[:, None]
        vel0 = self._rng.normal(0.0, 1.0, size=(n, 3)) * v_scale
        vel0 -= np.mean(vel0, axis=0, keepdims=True)
        self.momenta[:n] = vel0 * self.masses[:n, None]

        # Track ternary guest states (-1 water/blocked, 0 empty, +1 adsorbed gas/ion)
        self.ternary_states = np.zeros(self.capacity, dtype=np.float32)
        if mode == "nanopore_gcmc_nemd":
            self.ternary_states[:n] = 0.0  # carbon pore framework

        self.step_count = 0
        self.sim_time_ps = 0.0
        self.potential_energy = 0.0
        self.kinetic_energy = 0.0
        self.virial = np.zeros((3, 3), dtype=np.float64)
        self.bond_orders_per_atom = np.full(self.capacity, 3.0 if mode == "cnt_tersoff" else 1.0, dtype=np.float32)
        self.atomic_stress = np.zeros(self.capacity, dtype=np.float32)
        self.last_ms_per_step = 0.1
        # Safety rail for imported / user-defined systems: max displacement per MD step (nm). None = off.
        self.max_step_disp: Optional[float] = None

        self.hardware: GPUProfile = probe_hardware()
        self._gpu_plan: Optional[GPUVerletPlan] = None
        self._custom_spline: Optional[CubicSplineRadialTable] = None
        self._history: "OrderedDict[int, StepSnapshot]" = OrderedDict()
        self._max_history = max_history
        self._history_ram_bytes: int = 0
        self.MAX_SNAPSHOT_RAM_BYTES: int = MAX_SNAPSHOT_RAM_BYTES

        self.active_channel = 0
        self.tile_ids = compute_morton_tiles(self.positions[:n], self.box_H, tile_size=32)
        if self.capacity > n:
            pad_tiles = np.full(self.capacity - n, int(np.max(self.tile_ids) + 1) if n > 0 else 0, dtype=np.uint32)
            self.tile_ids = np.concatenate([self.tile_ids, pad_tiles])

        # Multiscale 3-zone partition, dirty-tile delta tracker, interactive SMD & health diagnostics
        self.multiscale_zones = np.zeros(self.capacity, dtype=np.uint8)
        self._multiscale_partition: Optional[MultiscaleZonePartition] = None
        self._dirty_tracker = DirtyTileTracker(self.capacity, self.tile_ids, threshold_ang=0.05)
        self.delta_streaming_enabled: bool = False
        self._last_dirty_ratio: float = 1.0
        self._last_dirty_tiles: int = 0
        self._smd_pull: Optional[Dict[str, Any]] = None
        self._smd_history: List[Dict[str, float]] = []
        self._last_clash_report: Optional[Dict[str, Any]] = None
        self._initial_total_energy: Optional[float] = None

        self._init_backend_resources()
        self._evaluate_state()
        self._initial_total_energy = float(self.potential_energy + self.kinetic_energy)
        self._record_history()

    def _init_backend_resources(self) -> None:
        if self.mode == "custom_spline":
            expr = self.custom_expr or "De * (1.0 - exp(-a * (r - r0)))**2"
            params = dict(self.custom_params or {"De": 180.0, "a": 2.0, "r0": 0.20})
            fn = compile_safe_radial_expr(expr)
            r_grid = np.linspace(0.08, self.rcut, 256, dtype=np.float64)
            u_grid = fn(r_grid, params)
            self._custom_spline = CubicSplineRadialTable(r_grid, u_grid, mode="lfs")

        if self.mode == "metal_verlet" and self.n_real >= 2:
            # Clamp rcut to strictly < half shortest box edge to keep MIC valid
            min_L = float(np.min(np.diag(self.box_H)))
            rc_safe = min(self.rcut, 0.45 * min_L)
            skin = min(0.18, max(0.05, 0.48 * min_L - rc_safe))
            self.rcut = rc_safe
            try:
                self._gpu_plan = _build_spatial_gpu_verlet_plan(
                    positions=self.positions[: self.n_real],
                    sigma=self.sigma[: self.n_real],
                    epsilon=self.epsilon[: self.n_real],
                    charges=self.charges[: self.n_real],
                    box=self.box_H,
                    exclusions=self._exclusion_pairs or None,
                    rcut=rc_safe,
                    skin=max(0.02, skin),
                    alpha_dsf=self.alpha_dsf,
                    use_lfs=True,
                    include_self=True,
                    backend="auto",
                )
            except RuntimeError:
                # Pure-NumPy fallback when native C++/Metal library is absent (e.g. Linux CI sandbox)
                self._gpu_plan = None

    def close(self) -> None:
        with self._lock:
            if self._gpu_plan is not None:
                self._gpu_plan.close()
                self._gpu_plan = None

    def _mic_vectors(self, i_idx: np.ndarray, j_idx: np.ndarray, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        d = q[i_idx] - q[j_idx]
        H = self.box_H
        Hinv = np.linalg.inv(H)
        frac = d @ Hinv.T
        frac -= np.rint(frac)
        d_mic = frac @ H.T
        r = np.linalg.norm(d_mic, axis=1)
        return d_mic, r

    def _evaluate_cnt_tersoff(self, q: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Smooth bond-order reactive covalent C-C network forcefield with Tersoff switching
        window `[R-D, R+D] = [0.18, 0.21] nm` and reactive bond-order updates.
        """
        n = self.n_real
        F = np.zeros((n, 3), dtype=np.float64)
        u_tot = 0.0
        vir = np.zeros((3, 3), dtype=np.float64)
        self.bond_orders_per_atom[:n] = 0.0
        self.atomic_stress[:n] = 0.0

        if not self.bonds:
            return 0.0, F, vir

        pairs = np.asarray(self.bonds, dtype=np.int32)
        ii = pairs[:, 0]
        jj = pairs[:, 1]
        d_ij, r = self._mic_vectors(ii, jj, q)
        r_safe = np.maximum(r, 1e-9)

        # Morse-Tersoff reactive C-C potential (De = 480 kJ/mol, a = 21.0 nm^-1, r0 = 0.142 nm)
        De = 480.0
        alpha = 21.0
        r0 = 0.142
        exp_term = np.exp(-alpha * (r_safe - r0))
        one_m = 1.0 - exp_term

        # Smooth Tersoff cutoff switch f_C in [0.18, 0.24] nm
        R, D = 0.21, 0.03
        fc = np.ones_like(r_safe)
        dfc = np.zeros_like(r_safe)
        in_sw = (r_safe > R - D) & (r_safe < R + D)
        if np.any(in_sw):
            arg = 0.5 * math.pi * (r_safe[in_sw] - R) / D
            fc[in_sw] = 0.5 - 0.5 * np.sin(arg)
            dfc[in_sw] = -(0.25 * math.pi / D) * np.cos(arg)
        beyond = r_safe >= R + D
        fc[beyond] = 0.0
        dfc[beyond] = 0.0

        u_pair = fc * De * (one_m * one_m - 1.0)
        du_pair = dfc * De * (one_m * one_m - 1.0) + fc * (2.0 * De * alpha * one_m * exp_term)
        u_tot += float(np.sum(u_pair))

        coeff = -du_pair / r_safe
        f_ij = d_ij * coeff[:, None]
        np.add.at(F, ii, f_ij)
        np.add.at(F, jj, -f_ij)

        # Bond orders (modulated smoothly by bond stretch) + atomic axial stress
        bo = np.clip(fc * np.exp(-2.5 * np.abs(r_safe - r0)), 0.0, 2.0)
        self.bond_orders = bo.astype(np.float32)
        np.add.at(self.bond_orders_per_atom, ii, bo)
        np.add.at(self.bond_orders_per_atom, jj, bo)

        # Virial tensor sigma = sum_b d_ij outer f_ij
        pair_stress = np.sum(d_ij * f_ij, axis=1)
        np.add.at(self.atomic_stress, ii, 0.5 * pair_stress)
        np.add.at(self.atomic_stress, jj, 0.5 * pair_stress)
        vir = d_ij.T @ f_ij

        # Simple weak nonbonded repulsion across 2nd neighbors to stabilize tube radial modes
        if n <= 400:
            diff = q[:, None, :] - q[None, :, :]
            H = self.box_H
            Hinv = np.linalg.inv(H)
            frac = diff @ Hinv.T
            frac -= np.rint(frac)
            d_all = frac @ H.T
            r2 = np.sum(d_all * d_all, axis=2)
            # Repulsive core for nonbonded pairs r < 0.28 nm
            mask = np.triu((r2 > 1e-8) & (r2 < 0.0784), k=1)
            mask[ii, jj] = False
            mask[jj, ii] = False
            ai, aj = np.where(mask)
            if ai.size > 0:
                rij = np.sqrt(r2[ai, aj])
                k_rep = 2500.0
                rc_nb = 0.28
                dr = rc_nb - rij
                u_rep = 0.5 * k_rep * dr * dr
                du_rep = -k_rep * dr
                u_tot += float(np.sum(u_rep))
                f_rep = d_all[ai, aj] * (-du_rep / rij)[:, None]
                np.add.at(F, ai, f_rep)
                np.add.at(F, aj, -f_rep)

        return u_tot, F, vir

    def _evaluate_custom_spline(self, q: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        assert self._custom_spline is not None
        n = self.n_real
        F = np.zeros((n, 3), dtype=np.float64)
        vir = np.zeros((3, 3), dtype=np.float64)
        self.atomic_stress[:n] = 0.0

        if n < 2:
            return 0.0, F, vir

        if n <= 1500:
            tri_i, tri_j = np.triu_indices(n, k=1)
            if self._bond_pairs.shape[0] > 0:
                bond_keys = set(self._bond_pairs[:, 0] * n + self._bond_pairs[:, 1])
                pair_keys = tri_i * n + tri_j
                non_bonded = np.array([k not in bond_keys for k in pair_keys], dtype=bool)
                ii = tri_i[non_bonded]
                jj = tri_j[non_bonded]
            else:
                ii = tri_i
                jj = tri_j
            d_ij, r = self._mic_vectors(ii, jj, q)
            within_rc = r <= self.rcut
            ii = ii[within_rc]
            jj = jj[within_rc]
            d_ij = d_ij[within_rc]
            r = r[within_rc]
        else:
            # Spatial linked-cell neighbor search for high-scale N > 1500
            H = self.box_H
            Hinv = np.linalg.inv(H)
            frac = q @ Hinv.T
            frac -= np.floor(frac)
            d_perp = 1.0 / np.maximum(np.linalg.norm(Hinv, axis=1), 1e-12)
            rc = float(self.rcut)
            nx = max(1, min(24, int(d_perp[0] / rc)))
            ny = max(1, min(24, int(d_perp[1] / rc)))
            nz = max(1, min(24, int(d_perp[2] / rc)))
            cx = np.clip((frac[:, 0] * nx).astype(np.int32), 0, nx - 1)
            cy = np.clip((frac[:, 1] * ny).astype(np.int32), 0, ny - 1)
            cz = np.clip((frac[:, 2] * nz).astype(np.int32), 0, nz - 1)
            cell_id = cx + nx * (cy + ny * cz)
            n_cells = nx * ny * nz

            cell_atoms = [[] for _ in range(n_cells)]
            for idx, c in enumerate(cell_id):
                cell_atoms[c].append(idx)

            offsets = [
                (0, 0, 0), (1, 0, 0), (-1, 1, 0), (0, 1, 0), (1, 1, 0),
                (-1, -1, 1), (0, -1, 1), (1, -1, 1),
                (-1, 0, 1), (0, 0, 1), (1, 0, 1),
                (-1, 1, 1), (0, 1, 1), (1, 1, 1)
            ]
            cand_i = []
            cand_j = []
            for ix in range(nx):
                for iy in range(ny):
                    for iz in range(nz):
                        c0 = ix + nx * (iy + ny * iz)
                        atoms0 = cell_atoms[c0]
                        if not atoms0:
                            continue
                        for dx, dy, dz in offsets:
                            jx = (ix + dx) % nx
                            jy = (iy + dy) % ny
                            jz = (iz + dz) % nz
                            c1 = jx + nx * (jy + ny * jz)
                            atoms1 = cell_atoms[c1]
                            if not atoms1:
                                continue
                            if c0 == c1:
                                for a_idx in range(len(atoms0)):
                                    for b_idx in range(a_idx + 1, len(atoms0)):
                                        cand_i.append(atoms0[a_idx])
                                        cand_j.append(atoms0[b_idx])
                            elif c0 < c1 or (dx, dy, dz) != (0, 0, 0):
                                for a in atoms0:
                                    for b in atoms1:
                                        cand_i.append(a)
                                        cand_j.append(b)
            if cand_i:
                ii = np.asarray(cand_i, dtype=np.int32)
                jj = np.asarray(cand_j, dtype=np.int32)
                d_ij, r = self._mic_vectors(ii, jj, q)
                within_rc = r <= self.rcut
                ii = ii[within_rc]
                jj = jj[within_rc]
                d_ij = d_ij[within_rc]
                r = r[within_rc]
            else:
                ii = np.empty(0, dtype=np.int32)
                jj = np.empty(0, dtype=np.int32)
                d_ij = np.empty((0, 3), dtype=np.float64)
                r = np.empty(0, dtype=np.float64)

        if ii.size == 0:
            return 0.0, F, vir

        r_safe = np.maximum(r, 1e-9)
        u_pair, du_pair = self._custom_spline.eval_array(r_safe)
        coeff = -du_pair / r_safe
        f_ij = d_ij * coeff[:, None]
        np.add.at(F, ii, f_ij)
        np.add.at(F, jj, -f_ij)
        p_stress = np.sum(d_ij * f_ij, axis=1)
        np.add.at(self.atomic_stress, ii, 0.5 * p_stress)
        np.add.at(self.atomic_stress, jj, 0.5 * p_stress)
        vir = d_ij.T @ f_ij
        self.bond_orders_per_atom[:n] = 1.0
        return float(np.sum(u_pair)), F, vir

    def _evaluate_nanopore_gcmc_nemd(self, q: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Graphene membrane (`charges == 0, names == 'C'`) tethered weakly around pore slab +
        mobile ions/CO2 guests with steric LJ interactions & external electric field `E_z`.
        """
        n = self.capacity
        active = self.active_mask
        F = np.zeros((n, 3), dtype=np.float64)
        vir = np.zeros((3, 3), dtype=np.float64)
        u_tot = 0.0

        # Weak harmonic tether on framework carbon atoms so pore geometry holds steady under flux
        c_idx = np.flatnonzero(active & (self.atomic_numbers == 6))
        if c_idx.size > 0:
            disp_c = q[c_idx] - self.system.positions[c_idx]
            H = self.box_H
            Hinv = np.linalg.inv(H)
            frac = disp_c @ Hinv.T
            frac -= np.rint(frac)
            disp_mic = frac @ H.T
            k_host = 2500.0
            u_tot += 0.5 * k_host * float(np.sum(disp_mic * disp_mic))
            F[c_idx] -= k_host * disp_mic

        # All active pairwise WCA/LJ repulsion + Coulomb external E_z on mobile guests
        act_idx = np.flatnonzero(active)
        m = act_idx.size
        if m >= 2 and m <= 2000:
            qa = q[act_idx]
            diff = qa[:, None, :] - qa[None, :, :]
            H = self.box_H
            Hinv = np.linalg.inv(H)
            frac = diff @ Hinv.T
            frac -= np.rint(frac)
            d_all = frac @ H.T
            r2 = np.sum(d_all * d_all, axis=2)
            sig_m = 0.5 * (self.sigma[act_idx][:, None] + self.sigma[act_idx][None, :])
            eps_m = np.sqrt(self.epsilon[act_idx][:, None] * self.epsilon[act_idx][None, :])
            rc2 = 0.75 * 0.75
            # Skip static C-C host pairs in pairwise LJ loop
            host_only = (self.atomic_numbers[act_idx] == 6)[:, None] & (self.atomic_numbers[act_idx] == 6)[None, :]
            mask = np.triu((r2 > 1e-6) & (r2 < rc2) & (~host_only), k=1)
            pi, pj = np.where(mask)
            if pi.size > 0:
                ri2 = np.maximum(r2[pi, pj], (0.55 * sig_m[pi, pj]) ** 2)
                s2 = (sig_m[pi, pj] ** 2) / ri2
                s6 = s2 * s2 * s2
                s12 = s6 * s6
                eij = eps_m[pi, pj]
                u_p = 4.0 * eij * (s12 - s6)
                du_over_r = 24.0 * eij * (s6 - 2.0 * s12) / ri2
                u_tot += float(np.sum(u_p))
                f_ij = d_all[pi, pj] * (-du_over_r)[:, None]
                gi = act_idx[pi]
                gj = act_idx[pj]
                np.add.at(F, gi, f_ij)
                np.add.at(F, gj, -f_ij)
                ps = np.sum(d_all[pi, pj] * f_ij, axis=1)
                np.add.at(self.atomic_stress, gi, 0.5 * ps)
                np.add.at(self.atomic_stress, gj, 0.5 * ps)

        # NEMD external electric field along +Z on charged species + axial pore bias
        guest_idx = np.flatnonzero(active & (self.atomic_numbers != 6))
        if guest_idx.size > 0:
            if abs(self.efield_z) > 0.0:
                # F_z = q * E_z * scale_conversion + pressure drive
                fz = self.charges[guest_idx] * self.efield_z * 96.485 + 2.5 * self.efield_z
                F[guest_idx, 2] += fz
                u_tot -= float(np.sum(fz * q[guest_idx, 2]))

        return u_tot, F[:n], vir

    def _evaluate_numpy_nonbonded(self, q: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        """Pure-NumPy periodic shifted-LJ nonbonded fallback when native CPU/Metal library is absent."""
        n = int(q.shape[0])
        f = np.zeros((n, 3), dtype=np.float64)
        vir = np.zeros((3, 3), dtype=np.float64)
        if n < 2:
            return 0.0, f, vir

        box_L = np.diag(self.box_H).astype(np.float64)
        rc = min(float(self.rcut), 0.45 * float(np.min(box_L)))
        ii, jj, d_ij, r = _find_neighbor_pairs_mic(q, box_L, rc)
        if ii.size == 0:
            return 0.0, f, vir

        if getattr(self, "_exclusion_pairs", None):
            excl_arr = np.asarray(self._exclusion_pairs, dtype=np.int64)
            excl_keys = excl_arr[:, 0] * n + excl_arr[:, 1]
            pair_keys = ii.astype(np.int64) * n + jj.astype(np.int64)
            keep = ~np.isin(pair_keys, excl_keys)
            ii, jj, d_ij, r = ii[keep], jj[keep], d_ij[keep], r[keep]
            if ii.size == 0:
                return 0.0, f, vir
        elif self._bond_pairs.shape[0] > 0:
            bp = self._bond_pairs
            b_lo = np.minimum(bp[:, 0], bp[:, 1]).astype(np.int64)
            b_hi = np.maximum(bp[:, 0], bp[:, 1]).astype(np.int64)
            bond_keys = b_lo * n + b_hi
            pair_keys = ii.astype(np.int64) * n + jj.astype(np.int64)
            keep = ~np.isin(pair_keys, bond_keys)
            ii, jj, d_ij, r = ii[keep], jj[keep], d_ij[keep], r[keep]
            if ii.size == 0:
                return 0.0, f, vir

        sig_ij = 0.5 * (self.sigma[ii] + self.sigma[jj])
        eps_ij = np.sqrt(np.maximum(self.epsilon[ii] * self.epsilon[jj], 0.0))
        r_safe = np.maximum(r, 0.25 * sig_ij)
        inv_r2 = 1.0 / (r_safe * r_safe)
        s2 = (sig_ij * sig_ij) * inv_r2
        s6 = s2 * s2 * s2
        s12 = s6 * s6

        s2_rc = (sig_ij * sig_ij) / (rc * rc)
        s6_rc = s2_rc * s2_rc * s2_rc
        u_pair = 4.0 * eps_ij * ((s12 - s6) - (s6_rc * s6_rc - s6_rc))
        du_over_r = 24.0 * eps_ij * (s6 - 2.0 * s12) * inv_r2
        f_ij = d_ij * (-du_over_r)[:, None]
        np.add.at(f, ii, f_ij)
        np.add.at(f, jj, -f_ij)
        vir = d_ij.T @ f_ij
        return float(np.sum(u_pair)), f, vir

    def _evaluate_state(self) -> None:
        if self.mode == "metal_verlet":
            if self._gpu_plan is not None:
                u, f, vir = self._gpu_plan.eval(positions=self.positions[: self.n_real], box=self.box_H, compute_virial=True)
            else:
                u, f, vir = self._evaluate_numpy_nonbonded(self.positions[: self.n_real])
            # Include harmonic covalent bonds + electric field if system carries explicit bonds/charges
            if self._bond_pairs.shape[0] > 0:
                ii = self._bond_pairs[:, 0]
                jj = self._bond_pairs[:, 1]
                d_ij, r = self._mic_vectors(ii, jj, self.positions[: self.n_real])
                r_safe = np.maximum(r, 1e-6)
                dr = r_safe - self._bond_r0
                k_bond = 12000.0 if self.system.units == "md" else 500.0
                u += 0.5 * k_bond * float(np.sum(dr * dr))
                f_ij = (-k_bond * dr / r_safe)[:, None] * d_ij
                np.add.at(f, ii, f_ij)
                np.add.at(f, jj, -f_ij)
                vir = vir + (d_ij.T @ f_ij)

            # AMBER-style Harmonic Valence Angles
            if self._angle_triplets.shape[0] > 0:
                ai = self._angle_triplets[:, 0]
                aj = self._angle_triplets[:, 1]
                ak = self._angle_triplets[:, 2]
                v1, r1 = self._mic_vectors(ai, aj, self.positions[: self.n_real])
                v2, r2 = self._mic_vectors(ak, aj, self.positions[: self.n_real])
                r1_safe = np.maximum(r1, 1e-6)
                r2_safe = np.maximum(r2, 1e-6)
                cos_th = np.clip(np.sum(v1 * v2, axis=1) / (r1_safe * r2_safe), -0.999999, 0.999999)
                th = np.arccos(cos_th)
                dth = th - self._angle_theta0
                k_th = 120.0 if self.system.units == "md" else 80.0
                u += 0.5 * k_th * float(np.sum(dth * dth))
                sin_th = np.sqrt(np.maximum(1.0 - cos_th * cos_th, 1e-8))
                coeff = (k_th * dth / sin_th)[:, None]
                g_i = (v2 / (r1_safe * r2_safe)[:, None]) - ((cos_th / (r1_safe * r1_safe))[:, None] * v1)
                g_k = (v1 / (r1_safe * r2_safe)[:, None]) - ((cos_th / (r2_safe * r2_safe))[:, None] * v2)
                f_ai = coeff * g_i
                f_ak = coeff * g_k
                f_aj = -(f_ai + f_ak)
                np.add.at(f, ai, f_ai)
                np.add.at(f, aj, f_aj)
                np.add.at(f, ak, f_ak)
                vir = vir + (v1.T @ f_ai) + (v2.T @ f_ak)

            # AMBER-style Periodic Proper Dihedral Torsions
            if self._dihedral_quartets.shape[0] > 0:
                di = self._dihedral_quartets[:, 0]
                dj = self._dihedral_quartets[:, 1]
                dk = self._dihedral_quartets[:, 2]
                dl = self._dihedral_quartets[:, 3]
                r_ij, _ = self._mic_vectors(di, dj, self.positions[: self.n_real])
                r_kj, rkj_len = self._mic_vectors(dk, dj, self.positions[: self.n_real])
                r_kl, _ = self._mic_vectors(dk, dl, self.positions[: self.n_real])
                m = np.cross(r_ij, r_kj)
                n_vec = np.cross(r_kj, r_kl)
                m2 = np.sum(m * m, axis=1)
                n2 = np.sum(n_vec * n_vec, axis=1)
                valid = (m2 > 1e-4) & (n2 > 1e-4)
                m_len = np.sqrt(np.maximum(m2, 1e-4))
                n_len = np.sqrt(np.maximum(n2, 1e-4))
                rkj_safe = np.maximum(rkj_len, 1e-6)
                rkj2 = np.maximum(rkj_len * rkj_len, 1e-12)
                cos_phi = np.clip(np.sum(m * n_vec, axis=1) / (m_len * n_len), -0.999999, 0.999999)
                sin_phi = np.sum(np.cross(m, n_vec) * r_kj, axis=1) / (m_len * n_len * rkj_safe)
                phi = np.arctan2(sin_phi, cos_phi)
                dphi = phi - self._dihedral_phi0
                k_phi = 4.0 if self.system.units == "md" else 2.0
                u += float(np.sum(np.where(valid, k_phi * (1.0 - np.cos(dphi)), 0.0)))
                dE_dphi = np.where(valid, k_phi * np.sin(dphi), 0.0)
                f_di = - (dE_dphi * (rkj_safe / np.maximum(m2, 1e-8)))[:, None] * m
                f_dl =   (dE_dphi * (rkj_safe / np.maximum(n2, 1e-8)))[:, None] * n_vec
                coef_ij = (np.sum(r_ij * r_kj, axis=1) / rkj2)[:, None]
                coef_kl = (np.sum(r_kl * r_kj, axis=1) / rkj2)[:, None]
                f_dj = (coef_ij - 1.0) * f_di - coef_kl * f_dl
                f_dk = - (f_di + f_dj + f_dl)
                np.add.at(f, di, f_di)
                np.add.at(f, dj, f_dj)
                np.add.at(f, dk, f_dk)
                np.add.at(f, dl, f_dl)
                vir = vir + (r_ij.T @ f_di) + (r_kj.T @ (f_dk + f_dl)) - (r_kl.T @ f_dl)
            if abs(self.efield_z) > 0.0 and self.n_real > 0:
                fz = self.charges[: self.n_real] * self.efield_z * 96.485
                f[:, 2] += fz
                u -= float(np.sum(fz * self.positions[: self.n_real, 2]))
            self.potential_energy = float(u)
            self.forces[: self.n_real] = f
            self.virial = vir
            f_mag = np.linalg.norm(f, axis=1)
            self.atomic_stress[: self.n_real] = (np.trace(vir) / max(1, self.n_real)) + 0.05 * f_mag
        elif self.mode == "cnt_tersoff":
            u, f, vir = self._evaluate_cnt_tersoff(self.positions[: self.n_real])
            self.potential_energy = u
            self.forces[: self.n_real] = f
            self.virial = vir
        elif self.mode == "custom_spline":
            u, f, vir = self._evaluate_custom_spline(self.positions[: self.n_real])

            # AMBER-style Harmonic Covalent Bonds
            if self._bond_pairs.shape[0] > 0:
                ii = self._bond_pairs[:, 0]
                jj = self._bond_pairs[:, 1]
                d_ij, r = self._mic_vectors(ii, jj, self.positions[: self.n_real])
                r_safe = np.maximum(r, 1e-6)
                dr = r_safe - self._bond_r0
                k_bond = 12000.0 if self.system.units == "md" else 500.0
                u += 0.5 * k_bond * float(np.sum(dr * dr))
                f_ij = (-k_bond * dr / r_safe)[:, None] * d_ij
                np.add.at(f, ii, f_ij)
                np.add.at(f, jj, -f_ij)
                vir = vir + (d_ij.T @ f_ij)

            # AMBER-style Harmonic Valence Angles
            if self._angle_triplets.shape[0] > 0:
                ai = self._angle_triplets[:, 0]
                aj = self._angle_triplets[:, 1]
                ak = self._angle_triplets[:, 2]
                v1, r1 = self._mic_vectors(ai, aj, self.positions[: self.n_real])
                v2, r2 = self._mic_vectors(ak, aj, self.positions[: self.n_real])
                r1_safe = np.maximum(r1, 1e-6)
                r2_safe = np.maximum(r2, 1e-6)
                cos_th = np.clip(np.sum(v1 * v2, axis=1) / (r1_safe * r2_safe), -0.999999, 0.999999)
                th = np.arccos(cos_th)
                dth = th - self._angle_theta0
                k_th = 120.0 if self.system.units == "md" else 80.0
                u += 0.5 * k_th * float(np.sum(dth * dth))
                sin_th = np.sqrt(np.maximum(1.0 - cos_th * cos_th, 1e-8))
                coeff = (k_th * dth / sin_th)[:, None]
                g_i = (v2 / (r1_safe * r2_safe)[:, None]) - ((cos_th / (r1_safe * r1_safe))[:, None] * v1)
                g_k = (v1 / (r1_safe * r2_safe)[:, None]) - ((cos_th / (r2_safe * r2_safe))[:, None] * v2)
                f_ai = coeff * g_i
                f_ak = coeff * g_k
                f_aj = -(f_ai + f_ak)
                np.add.at(f, ai, f_ai)
                np.add.at(f, aj, f_aj)
                np.add.at(f, ak, f_ak)

            # AMBER-style Proper Dihedrals
            if self._dihedral_quartets.shape[0] > 0:
                di = self._dihedral_quartets[:, 0]
                dj = self._dihedral_quartets[:, 1]
                dk = self._dihedral_quartets[:, 2]
                dl = self._dihedral_quartets[:, 3]
                r_ij, _ = self._mic_vectors(di, dj, self.positions[: self.n_real])
                r_kj, rkj_len = self._mic_vectors(dk, dj, self.positions[: self.n_real])
                r_kl, _ = self._mic_vectors(dk, dl, self.positions[: self.n_real])
                m = np.cross(r_ij, r_kj)
                n_vec = np.cross(r_kj, r_kl)
                m2 = np.sum(m * m, axis=1)
                n2 = np.sum(n_vec * n_vec, axis=1)
                valid = (m2 > 1e-4) & (n2 > 1e-4)
                m_len = np.sqrt(np.maximum(m2, 1e-4))
                n_len = np.sqrt(np.maximum(n2, 1e-4))
                rkj_safe = np.maximum(rkj_len, 1e-6)
                rkj2 = np.maximum(rkj_len * rkj_len, 1e-12)
                cos_phi = np.clip(np.sum(m * n_vec, axis=1) / (m_len * n_len), -0.999999, 0.999999)
                sin_phi = np.sum(np.cross(m, n_vec) * r_kj, axis=1) / (m_len * n_len * rkj_safe)
                phi = np.arctan2(sin_phi, cos_phi)
                dphi = phi - self._dihedral_phi0
                k_phi = 4.0 if self.system.units == "md" else 2.0
                u += float(np.sum(np.where(valid, k_phi * (1.0 - np.cos(dphi)), 0.0)))
                dE_dphi = np.where(valid, k_phi * np.sin(dphi), 0.0)
                f_di = - (dE_dphi * (rkj_safe / np.maximum(m2, 1e-8)))[:, None] * m
                f_dl =   (dE_dphi * (rkj_safe / np.maximum(n2, 1e-8)))[:, None] * n_vec
                coef_ij = (np.sum(r_ij * r_kj, axis=1) / rkj2)[:, None]
                coef_kl = (np.sum(r_kl * r_kj, axis=1) / rkj2)[:, None]
                f_dj = (coef_ij - 1.0) * f_di - coef_kl * f_dl
                f_dk = - (f_di + f_dj + f_dl)
                np.add.at(f, di, f_di)
                np.add.at(f, dj, f_dj)
                np.add.at(f, dk, f_dk)
                np.add.at(f, dl, f_dl)
                vir = vir + (r_ij.T @ f_di) + (r_kj.T @ (f_dk + f_dl)) - (r_kl.T @ f_dl)

            if abs(self.efield_z) > 0.0 and self.n_real > 0:
                fz = self.charges[: self.n_real] * self.efield_z * 96.485
                f[:, 2] += fz
                u -= float(np.sum(fz * self.positions[: self.n_real, 2]))

            self.potential_energy = float(u)
            self.forces[: self.n_real] = f
            self.virial = vir
            f_mag = np.linalg.norm(f, axis=1)
            self.atomic_stress[: self.n_real] = (np.trace(vir) / max(1, self.n_real)) + 0.05 * f_mag
        elif self.mode == "nanopore_gcmc_nemd":
            u, f, vir = self._evaluate_nanopore_gcmc_nemd(self.positions)
            self.potential_energy = u
            self.forces[:] = f
            self.virial = vir
        else:
            self.potential_energy = 0.0
            self.forces.fill(0.0)

        if self._smd_pull is not None:
            smd_idx = int(self._smd_pull["atom_index"])
            if 0 <= smd_idx < self.n_real and self.active_mask[smd_idx]:
                target_nm = self._smd_pull["target_nm"]
                k_spring = float(self._smd_pull["k_spring"])
                disp = target_nm - self.positions[smd_idx]
                Hinv = np.linalg.inv(self.box_H)
                frac = disp @ Hinv.T
                frac -= np.rint(frac)
                disp_mic = frac @ self.box_H.T
                u_smd = 0.5 * k_spring * float(np.dot(disp_mic, disp_mic))
                f_smd = k_spring * disp_mic
                self.potential_energy += u_smd
                self.forces[smd_idx] += f_smd
                ext_nm = float(np.linalg.norm(disp_mic))
                ext_ang = ext_nm * self.coord_to_ang
                f_pull = k_spring * ext_nm
                self._smd_pull["extension_ang"] = ext_ang
                self._smd_pull["pull_force_kj_mol_nm"] = f_pull
                self._smd_history.append(
                    {
                        "step": int(self.step_count),
                        "time_ps": round(float(self.sim_time_ps), 6),
                        "extension_ang": float(ext_ang),
                        "force": float(f_pull),
                    }
                )
                if len(self._smd_history) > 200:
                    self._smd_history = self._smd_history[-200:]

        enh_suite = getattr(self, "_enhanced_sampling", None)
        if enh_suite is not None and self.n_real > 0:
            try:
                box_diag = np.diag(self.box_H).astype(np.float64)
                bias_e, bias_f, _ = enh_suite.evaluate(
                    self.positions[: self.n_real],
                    box_nm=box_diag,
                    step=int(self.step_count),
                    time_ps=float(self.sim_time_ps),
                    record=False,
                )
                self.potential_energy += float(bias_e)
                if bias_f.shape == (self.n_real, 3):
                    self.forces[: self.n_real] += bias_f
            except Exception:
                pass

        # Optional per-atom pLDDT harmonic scaffold restraints (AlphaFold/ATLAS confidence weighting)
        plddt_k = getattr(self, "_plddt_restraint_k", None)
        plddt_ref = getattr(self, "_plddt_restraint_ref_nm", None)
        if plddt_k is not None and plddt_ref is not None and plddt_k.size == self.n_real:
            dr_res = self.positions[: self.n_real] - plddt_ref
            self.potential_energy += 0.5 * float(np.sum(plddt_k * np.sum(dr_res * dr_res, axis=1)))
            self.forces[: self.n_real] += -plddt_k[:, None] * dr_res

        act = self.active_mask
        p_act = self.momenta[act]
        m_act = self.masses[act]
        self.kinetic_energy = 0.5 * float(np.sum(np.sum(p_act * p_act, axis=1) / m_act))

    def minimize(self, max_iters: int = 300, max_disp: float = 0.02, f_tol: float = 1.0e3) -> Dict[str, Any]:
        """Adaptive steepest-descent relaxation (removes steric clashes in imported structures).

        Zeroes momenta, then re-draws Boltzmann velocities at the target temperature, and enables the
        per-step displacement safety clamp so downstream MD cannot explode.
        """
        with self._lock:
            n = self.n_real
            self._evaluate_state()
            u_prev = self.potential_energy
            step_len = 0.001
            iters = 0
            u0 = u_prev
            for iters in range(1, max_iters + 1):
                f = self.forces[:n]
                fmax = float(np.max(np.linalg.norm(f, axis=1))) if n else 0.0
                if not np.isfinite(fmax) or fmax < f_tol:
                    break
                old_pos = self.positions[:n].copy()
                cand_pos = old_pos + f * (step_len / fmax)
                if self.box_H is not None and abs(float(np.linalg.det(self.box_H))) > 1e-12:
                    _diag = np.diag(self.box_H)
                    if np.all(_diag > 0):
                        cand_pos = np.mod(cand_pos, _diag[None, :])
                self.positions[:n] = cand_pos
                self._evaluate_state()
                if np.isfinite(self.potential_energy) and self.potential_energy < u_prev:
                    u_prev = self.potential_energy
                    step_len = min(max_disp, step_len * 1.2)
                else:
                    self.positions[:n] = old_pos
                    self._evaluate_state()
                    step_len *= 0.5
                    if step_len < 1e-6:
                        break
            v_scale = np.sqrt(self.k_b * max(1e-6, self.temperature) / self.masses[:n])[:, None]
            vel0 = self._rng.normal(0.0, 1.0, size=(n, 3)) * v_scale
            vel0 -= np.mean(vel0, axis=0, keepdims=True)
            self.momenta[:n] = vel0 * self.masses[:n, None]
            self.max_step_disp = 0.02
            self._evaluate_state()
            self._initial_total_energy = float(self.potential_energy + self.kinetic_energy)
            self._dirty_tracker.reset()
            self._history.clear()
            self._history_ram_bytes = 0
            self._record_history()
            return {
                "iterations": iters,
                "energy_before": u0,
                "energy_after": float(self.potential_energy),
                "max_force_after": float(np.max(np.linalg.norm(self.forces[:n], axis=1))) if n else 0.0,
            }

    def step(self, n_steps: int = 1, apply_thermostat: Optional[bool] = None) -> Dict[str, Any]:
        """Advance simulation cleanly by `n_steps` and return a telemetry + delta summary."""
        with self._lock:
            t0 = time.perf_counter()
            n_steps = max(1, int(n_steps))
            half_dt = 0.5 * self.dt
            use_thermo = (self.mode != "custom_spline") if apply_thermostat is None else apply_thermostat
            gcmc_ins_slots: List[int] = []
            gcmc_del_slots: List[int] = []

            # Fast Metal GPU-resident batched path for metal_verlet when n_steps >= 10
            if (
                self.mode == "metal_verlet"
                and self._gpu_plan is not None
                and self._gpu_plan.using_gpu
                and n_steps >= 10
                and not self.bonds
                and abs(self.strain_rate) == 0.0
                and self._smd_pull is None
                and self._multiscale_partition is None
            ):
                res = self._gpu_plan.run_batched_steps(
                    positions=self.positions[: self.n_real],
                    momenta=self.momenta[: self.n_real],
                    masses=self.masses[: self.n_real],
                    dt=self.dt,
                    n_steps=n_steps,
                    box=self.box_H,
                )
                self.positions[: self.n_real] = np.asarray(res["positions"], dtype=np.float64)
                self.momenta[: self.n_real] = np.asarray(res["momenta"], dtype=np.float64)
                self.forces[: self.n_real] = np.asarray(res["forces"], dtype=np.float64)
                self.potential_energy = float(res["potential_energy"])
                self.kinetic_energy = float(res["kinetic_energy"])
                self.step_count += n_steps
                self.sim_time_ps += n_steps * self.dt
                if self.temperature > 0.0 and use_thermo:
                    # Gentle Bussi-like velocity rescaling toward target T for stable visual showcase
                    inst_t = (2.0 * self.kinetic_energy) / max(1e-9, 3.0 * self.n_real * self.k_b)
                    if inst_t > 1e-6:
                        alpha = math.sqrt(1.0 + 0.1 * (self.temperature / inst_t - 1.0))
                        self.momenta[: self.n_real] *= alpha
                        self.kinetic_energy *= alpha * alpha
            else:
                for _ in range(n_steps):
                    act = self.active_mask
                    if abs(self.strain_rate) > 0.0:
                        scale = 1.0 + self.strain_rate * self.dt
                        self.box_H[2, 2] *= scale
                        self.positions[act, 2] *= scale

                    # Velocity Verlet / BAOAB half-kick -> drift
                    self.momenta[act] += half_dt * self.forces[act]
                    disp = self.dt * self.momenta[act] / self.masses[act, None]
                    if self.max_step_disp is not None:
                        disp = np.clip(disp, -self.max_step_disp, self.max_step_disp)
                    if self._multiscale_partition is not None:
                        rigid_act = self.multiscale_zones[act] == 2
                        if np.any(rigid_act):
                            disp[rigid_act] = 0.0
                    self.positions[act] += disp

                    # Weak Langevin O-step when thermostatted
                    if use_thermo and self.gamma > 0.0:
                        c1 = math.exp(-self.gamma * self.dt)
                        c2 = math.sqrt(max(0.0, 1.0 - c1 * c1) * self.k_b * self.temperature)
                        noise = self._rng.normal(0.0, 1.0, size=(int(np.sum(act)), 3))
                        self.momenta[act] = c1 * self.momenta[act] + c2 * np.sqrt(self.masses[act, None]) * noise

                    # Periodic wrap in-box cleanly
                    H = self.box_H
                    Hinv = np.linalg.inv(H)
                    frac = self.positions[act] @ Hinv.T
                    frac -= np.floor(frac)
                    self.positions[act] = frac @ H.T

                    # GCMC insertion/deletion moves in nanopore_gcmc_nemd mode
                    if self.mode == "nanopore_gcmc_nemd" and (self.step_count % 5 == 0):
                        ins_s, del_s = self._attempt_gcmc_move()
                        gcmc_ins_slots.extend(ins_s)
                        gcmc_del_slots.extend(del_s)

                    self._evaluate_state()
                    act = self.active_mask
                    self.momenta[act] += half_dt * self.forces[act]
                    self.step_count += 1
                    self.sim_time_ps += self.dt

            act = self.active_mask
            p_act = self.momenta[act]
            m_act = self.masses[act]
            self.kinetic_energy = 0.5 * float(np.sum(np.sum(p_act * p_act, axis=1) / m_act))
            elapsed_ms = max(1e-4, (time.perf_counter() - t0) * 1e3)
            self.last_ms_per_step = elapsed_ms / n_steps

            self._record_history()
            summary = self.get_metrics()
            summary["gcmc_inserted_slots"] = gcmc_ins_slots
            summary["gcmc_deleted_slots"] = gcmc_del_slots
            return summary

    def _attempt_gcmc_move(self) -> Tuple[List[int], List[int]]:
        """Propose 1 GCMC insertion or deletion weighted cleanly by chemical potential `mu`."""
        inserted: List[int] = []
        deleted: List[int] = []
        free = np.flatnonzero(~self.active_mask)
        guests = np.flatnonzero(self.active_mask & (self.atomic_numbers != 6))

        propose_insert = bool(self._rng.random() < 0.55)
        if propose_insert and free.size > 0:
            slot = int(free[0])
            # Propose random position in box outside carbon core radius >= 0.22 nm
            L = np.diag(self.box_H)
            cand = self._rng.uniform(0.0, 1.0, size=3) * L
            act_q = self.positions[self.active_mask]
            if act_q.shape[0] > 0:
                d = act_q - cand[None, :]
                d -= L[None, :] * np.rint(d / L[None, :])
                min_r2 = float(np.min(np.sum(d * d, axis=1)))
            else:
                min_r2 = 1.0
            if min_r2 >= 0.048:  # >= 0.22 nm Farkas hard-core steric clearance
                is_cation = bool(self._rng.random() < 0.5)
                z_num = 11 if is_cation else 18  # Na+ (+1 ternary) or Ar/CO2-like (-1 / +1)
                self.positions[slot] = cand
                self.masses[slot] = 22.99 if is_cation else 44.01
                self.charges[slot] = 1.0 if is_cation else -0.5
                self.sigma[slot] = 0.26 if is_cation else 0.34
                self.epsilon[slot] = 0.55 if is_cation else 0.95
                self.atomic_numbers[slot] = z_num
                self.radii_ang[slot] = ELEMENT_RADII_ANG.get(z_num, 1.8)
                self.ternary_states[slot] = 1.0 if is_cation else -1.0
                v_s = math.sqrt(self.k_b * self.temperature / self.masses[slot])
                self.momenta[slot] = self._rng.normal(0.0, v_s, size=3) * self.masses[slot]
                self.active_mask[slot] = True
                inserted.append(slot)
        elif (not propose_insert) and guests.size > 2 and self.gcmc_mu < 2.0:
            # Higher chemical potential suppresses deletion cleanly
            acc = math.exp(-max(0.0, self.gcmc_mu + 4.0) * 0.25)
            if self._rng.random() < acc:
                slot = int(guests[self._rng.integers(0, guests.size)])
                self.active_mask[slot] = False
                self.radii_ang[slot] = 0.0
                self.momenta[slot] = 0.0
                self.forces[slot] = 0.0
                self.ternary_states[slot] = 0.0
                deleted.append(slot)
        return inserted, deleted

    def compute_scalar_channel(self, channel_name: str) -> np.ndarray:
        """Compute active `(capacity,)` float32 scalar values for one of the 6 physics themes."""
        if channel_name == "force-magnitude":
            out = np.linalg.norm(self.forces, axis=1).astype(np.float32)
        elif channel_name == "kinetic-temp":
            p2 = np.sum(self.momenta * self.momenta, axis=1)
            ke_i = 0.5 * p2 / np.maximum(self.masses, 1e-12)
            out = ((2.0 / (3.0 * self.k_b)) * ke_i).astype(np.float32)
        elif channel_name == "charge-potential":
            out = self.charges.astype(np.float32)
        elif channel_name == "bond-order":
            out = self.bond_orders_per_atom.astype(np.float32)
        elif channel_name == "atomic-stress":
            out = self.atomic_stress.astype(np.float32)
        elif channel_name == "ternary-state":
            out = self.ternary_states.astype(np.float32)
        elif channel_name == "multiscale-zone":
            out = self.multiscale_zones.astype(np.float32)
        else:
            raise ValueError(f"Unknown physics color channel '{channel_name}'; expected one of {PHYSICS_CHANNEL_NAMES}")
        out[~self.active_mask] = 0.0
        return out

    @staticmethod
    def _snapshot_nbytes(snap: StepSnapshot) -> int:
        total = (
            int(snap.positions.nbytes)
            + int(snap.momenta.nbytes)
            + int(snap.forces.nbytes)
            + int(snap.box_H.nbytes)
        )
        for arr in snap.scalars.values():
            total += int(arr.nbytes)
        return total

    def _compute_ram_bytes(self) -> int:
        total = 0
        for arr in (
            self.positions,
            self.momenta,
            self.forces,
            self.masses,
            self.charges,
            self.sigma,
            self.epsilon,
            self.active_mask,
            self.atomic_numbers,
            self.radii_ang,
            self.ternary_states,
            self.bond_orders_per_atom,
            self.atomic_stress,
            self.tile_ids,
            self.multiscale_zones,
            self._dirty_tracker.tile_ids,
            self._dirty_tracker.last_sent_coords_ang,
        ):
            if isinstance(arr, np.ndarray):
                total += int(arr.nbytes)
        total += int(self._history_ram_bytes)
        return total

    def _record_history(self) -> None:
        est_bytes = (
            int(self.positions.nbytes)
            + int(self.momenta.nbytes)
            + int(self.forces.nbytes)
            + int(self.box_H.nbytes)
            + int(self.capacity) * 8
        )
        if est_bytes > MAX_SNAPSHOT_RAM_BYTES:
            return
        scalars = {name: self.compute_scalar_channel(name) for name in ("force-magnitude", "bond-order")}
        snap = StepSnapshot(
            step=self.step_count,
            time_ps=self.sim_time_ps,
            positions=self.positions.copy(),
            momenta=self.momenta.copy(),
            forces=self.forces.copy(),
            box_H=self.box_H.copy(),
            potential_energy=self.potential_energy,
            kinetic_energy=self.kinetic_energy,
            scalars=scalars,
        )
        snap_bytes = self._snapshot_nbytes(snap)
        if snap_bytes > MAX_SNAPSHOT_RAM_BYTES:
            return
        if self.step_count in self._history:
            self._history_ram_bytes = max(0, self._history_ram_bytes - self._snapshot_nbytes(self._history[self.step_count]))
        self._history[self.step_count] = snap
        self._history_ram_bytes += snap_bytes
        while len(self._history) > self._max_history or self._history_ram_bytes > MAX_SNAPSHOT_RAM_BYTES:
            _, evicted = self._history.popitem(last=False)
            self._history_ram_bytes = max(0, self._history_ram_bytes - self._snapshot_nbytes(evicted))

    def scrub_to_step(self, target_step: int) -> Tuple[Dict[str, Any], bool]:
        """Restore recorded snapshot (supporting exact reversible time-scrubbing)."""
        with self._lock:
            if not self._history:
                return self.get_metrics(), False
            keys = list(self._history.keys())
            closest = min(keys, key=lambda k: abs(k - int(target_step)))
            snap = self._history[closest]
            was_reverse = closest <= self.step_count
            self.step_count = snap.step
            self.sim_time_ps = snap.time_ps
            self.positions[:] = snap.positions
            self.momenta[:] = snap.momenta
            self.forces[:] = snap.forces
            self.box_H[:] = snap.box_H
            self.potential_energy = snap.potential_energy
            self.kinetic_energy = snap.kinetic_energy
            return self.get_metrics(), was_reverse

    def get_metrics(self) -> Dict[str, Any]:
        with self._lock:
            n_act = int(np.sum(self.active_mask))
            inst_temp = (2.0 * self.kinetic_energy) / max(1e-9, 3.0 * n_act * self.k_b) if n_act > 0 else 0.0
            # Effective ns/day @ 2 fs step
            ns_per_day = (0.002 * 86400.0) / max(self.last_ms_per_step, 1e-4)
            f_max = float(np.max(np.linalg.norm(self.forces[self.active_mask], axis=1))) if n_act > 0 else 0.0
            vol_nm3 = float(abs(np.linalg.det(self.box_H)))
            total_e = float(self.potential_energy + self.kinetic_energy)
            e0 = self._initial_total_energy
            if e0 is not None and math.isfinite(total_e) and math.isfinite(e0):
                e_scale = max(
                    abs(e0),
                    abs(float(self.potential_energy)) + abs(float(self.kinetic_energy)),
                    1.5 * max(1, n_act) * self.k_b * max(float(self.temperature), 100.0),
                    1.0,
                )
                drift_fraction = abs(total_e - e0) / e_scale
            else:
                drift_fraction = 0.0 if math.isfinite(total_e) else float("inf")

            warn_drift = 0.35 if (self.gamma > 0.0 and self.mode != "custom_spline") else 0.05
            if (not math.isfinite(total_e)) or (not math.isfinite(f_max)) or (f_max > 1.0e5):
                health_status = "clash"
            elif (f_max > 1.0e4) or (drift_fraction > warn_drift):
                health_status = "warning"
            else:
                health_status = "bounded"

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

            ram_bytes = self._compute_ram_bytes()
            ram_mb = float(ram_bytes) / (1024.0 * 1024.0)
            naive_float64_ram_mb = float(n_act) * 160.0 / 1.0e6
            storage_mode = "ternary-compact" if self.n_real > 200_000 else "float64-full"
            memory_savings_ratio = (naive_float64_ram_mb / max(ram_mb, 1e-6)) if n_act > 0 else 1.0
            active_aa_count = (
                int(self._multiscale_partition.n_all_atom)
                if self._multiscale_partition is not None
                else int(n_act)
            )

            return {
                "step": int(self.step_count),
                "time_ps": round(self.sim_time_ps, 6),
                "n_atoms": n_act,
                "capacity": int(self.capacity),
                "active_all_atom_count": active_aa_count,
                "storage_mode": storage_mode,
                "ram_bytes": int(ram_bytes),
                "ram_mb": round(ram_mb, 4),
                "naive_float64_ram_mb": round(naive_float64_ram_mb, 4),
                "memory_savings_ratio": round(memory_savings_ratio, 2),
                "potential_energy": float(self.potential_energy),
                "kinetic_energy": float(self.kinetic_energy),
                "total_energy": total_e,
                "temperature_K": float(inst_temp),
                "max_force": f_max,
                "drift_fraction": float(drift_fraction),
                "health_status": health_status,
                "ms_per_step": float(self.last_ms_per_step),
                "ns_per_day_2fs": float(ns_per_day),
                "box_lengths_nm": [float(self.box_H[0, 0]), float(self.box_H[1, 1]), float(self.box_H[2, 2])],
                "box_origin_ang": [0.0, 0.0, 0.0],
                "volume_nm3": vol_nm3,
                "stress_zz": float(self.virial[2, 2] / max(vol_nm3, 1e-9)),
                "mode": self.mode,
                "backend": self.hardware.backend if (self.mode == "metal_verlet" and self.hardware.has_gpu) else "cpu",
                "history_steps": list(self._history.keys()),
                "multiscale": (
                    self._multiscale_partition.build_multiscale_summary()
                    if self._multiscale_partition is not None
                    else None
                ),
                "smd": smd_info,
                "delta_streaming": {
                    "enabled": bool(self.delta_streaming_enabled),
                    "threshold_ang": float(self._dirty_tracker.threshold_ang),
                    "last_dirty_ratio": float(self._last_dirty_ratio),
                    "last_dirty_tiles": int(self._last_dirty_tiles),
                },
            }

    def set_plddt_restraints(
        self,
        restraint_k_per_atom: Optional[np.ndarray],
    ) -> Dict[str, Any]:
        """Attach or clear per-atom pLDDT-weighted harmonic scaffold restraints (kJ/(mol*nm^2))."""
        with self._lock:
            if restraint_k_per_atom is None:
                self._plddt_restraint_k = None
                self._plddt_restraint_ref_nm = None
                self._evaluate_state()
                return {"active": False, "mean_k": 0.0, "restrained_atoms": 0}
            k_arr = np.asarray(restraint_k_per_atom, dtype=np.float64).reshape(-1)
            if k_arr.size != self.n_real:
                raise ValueError(f"restraint_k_per_atom length ({k_arr.size}) must match n_real ({self.n_real})")
            self._plddt_restraint_k = np.maximum(k_arr, 0.0).copy()
            self._plddt_restraint_ref_nm = self.positions[: self.n_real].copy()
            self._evaluate_state()
            self._q_thermo_cum = 0.0
            self._initial_total_energy = float(self.potential_energy + self.kinetic_energy)
            return {
                "active": True,
                "mean_k": float(np.mean(self._plddt_restraint_k)),
                "restrained_atoms": int(np.sum(self._plddt_restraint_k > 1e-6)),
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
                act = self.active_mask
                n_act = int(np.sum(act))
                if n_act > 0:
                    p_act = self.momenta[act]
                    m_act = self.masses[act]
                    ke_curr = 0.5 * float(np.sum(np.sum(p_act * p_act, axis=1) / m_act))
                    inst_t = (2.0 * ke_curr) / max(1e-9, 3.0 * n_act * self.k_b)
                    if inst_t > 1e-6:
                        scale = math.sqrt(self.temperature / inst_t)
                        self.momenta[act] *= scale
                    else:
                        v_scale = np.sqrt(self.k_b * self.temperature / m_act)[:, None]
                        vel = self._rng.normal(0.0, 1.0, size=(n_act, 3)) * v_scale
                        vel -= np.mean(vel, axis=0, keepdims=True)
                        self.momenta[act] = vel * m_act[:, None]
                    p_new = self.momenta[act]
                    self.kinetic_energy = 0.5 * float(np.sum(np.sum(p_new * p_new, axis=1) / m_act))
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

    def set_focus_zone(
        self,
        center_ang: Sequence[float],
        aa_radius_ang: float = 15.0,
        cg_radius_ang: float = 35.0,
    ) -> Dict[str, Any]:
        """Partition atoms into All-Atom (0), Coarse-Grained (1), and Rigid Scaffold (2) zones."""
        with self._lock:
            coords_ang = self.positions[: self.n_real] * self.coord_to_ang
            part = assign_multiscale_zones(coords_ang, center_ang, aa_radius_ang, cg_radius_ang)
            self._multiscale_partition = part
            self.multiscale_zones[: self.n_real] = part.zone_ids
            if self.capacity > self.n_real:
                self.multiscale_zones[self.n_real :] = 2
            return part.build_multiscale_summary()

    def relax_clashes(
        self,
        max_steps: int = 40,
        max_force_cap: float = 5000.0,
        alpha_sc: float = 0.5,
        lam: float = 0.5,
    ) -> Dict[str, Any]:
        """Run Beutler soft-core LJ clash relaxation on active atoms and re-evaluate state."""
        with self._lock:
            n = self.n_real
            relaxed_pos, report = softcore_relax_clashes(
                self.positions[:n],
                np.diag(self.box_H),
                self.sigma[:n],
                self.epsilon[:n],
                exclusions=getattr(self, "_exclusion_pairs", None),
                max_steps=max_steps,
                max_force_cap=max_force_cap,
                alpha_sc=alpha_sc,
                lam=lam,
            )
            self.positions[:n] = relaxed_pos
            self.max_step_disp = 0.02
            self._evaluate_state()
            self._initial_total_energy = float(self.potential_energy + self.kinetic_energy)
            self._dirty_tracker.reset()
            self._last_clash_report = dict(report)
            self._record_history()
            return report

    def set_delta_threshold(
        self,
        threshold_ang: float,
        enabled: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Configure DirtyTileTracker displacement threshold (Å) for sparse delta streaming."""
        with self._lock:
            thr = max(0.0, float(threshold_ang))
            self._dirty_tracker.set_threshold(thr)
            if enabled is not None:
                self.delta_streaming_enabled = bool(enabled)
            else:
                self.delta_streaming_enabled = thr > 0.0
            return {
                "threshold_ang": float(self._dirty_tracker.threshold_ang),
                "delta_streaming_enabled": bool(self.delta_streaming_enabled),
            }

    def steer_atom(
        self,
        atom_index: Optional[int],
        target_ang: Optional[Sequence[float]] = None,
        k_spring: float = 500.0,
    ) -> Dict[str, Any]:
        """Attach or clear an interactive SMD harmonic spring pull on a single atom."""
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
            if idx < 0 or idx >= self.n_real:
                raise IndexError(f"Atom index {idx} out of range 0..{self.n_real - 1}")
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

    def inspect_atom(self, idx: int) -> Dict[str, Any]:
        with self._lock:
            if idx < 0 or idx >= self.capacity:
                raise IndexError(f"Atom index {idx} out of range 0..{self.capacity - 1}")
            pos = self.positions[idx]
            vel = self.momenta[idx] / max(self.masses[idx], 1e-12)
            frc = self.forces[idx]
            ke = 0.5 * float(np.dot(self.momenta[idx], self.momenta[idx]) / max(self.masses[idx], 1e-12))
            return {
                "index": int(idx),
                "active": bool(self.active_mask[idx]),
                "element": self.names[idx],
                "atomic_number": int(self.atomic_numbers[idx]),
                "position_nm": [float(v) for v in pos],
                "position_ang": [float(v * self.coord_to_ang) for v in pos],
                "velocity_nm_ps": [float(v) for v in vel],
                "force": [float(v) for v in frc],
                "force_magnitude": float(np.linalg.norm(frc)),
                "kinetic_energy": ke,
                "local_temp_K": float((2.0 / (3.0 * self.k_b)) * ke),
                "charge": float(self.charges[idx]),
                "bond_order": float(self.bond_orders_per_atom[idx]),
                "atomic_stress": float(self.atomic_stress[idx]),
                "ternary_state": float(self.ternary_states[idx]),
            }

    def encode_topology_packet(self) -> bytes:
        with self._lock:
            coords_ang = (self.positions * self.coord_to_ang).astype(np.float32)
            box_ang = (self.box_H * self.coord_to_ang).astype(np.float32)
            b_arr = np.asarray(self.bonds, dtype=np.uint32) if self.bonds else np.zeros((0, 2), dtype=np.uint32)
            res_ids = np.arange(self.capacity, dtype=np.uint32)
            return pack_topology_init(
                atomic_numbers=self.atomic_numbers,
                bonds=b_arr,
                residue_ids=res_ids,
                tile_ids=self.tile_ids,
                box_H=box_ang,
                coords=coords_ang,
                radii=self.radii_ang,
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
            scalars = self.compute_scalar_channel(ch_name)
            coords_ang = (self.positions * self.coord_to_ang).astype(np.float32)
            box_ang = (self.box_H * self.coord_to_ang).astype(np.float32)

            if camera_roi is not None and self.capacity > 50_000:
                center_ang, radius_ang, max_k = camera_roi
                roi_idx = select_roi_atoms(coords_ang, center_ang, radius_ang, max_atoms=max_k)
                return pack_frame_tile_patch(
                    atom_indices=roi_idx,
                    coords=coords_ang[roi_idx],
                    scalars=scalars[roi_idx],
                    box_H=box_ang,
                    frame_id=self.step_count,
                    sim_time_ps=self.sim_time_ps,
                    use_int16=self.use_int16,
                    scalar_channel=ch_idx,
                    tile_id=0,
                )

            want_delta = self.delta_streaming_enabled if use_delta is None else bool(use_delta)
            if want_delta and (not reversed_step) and self._dirty_tracker.last_sent_coords_ang is not None:
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
                        use_int16=self.use_int16,
                        scalar_channel=ch_idx,
                    )

            self._dirty_tracker.reset(coords_ang)
            self._last_dirty_ratio = 1.0
            self._last_dirty_tiles = int(np.unique(self.tile_ids).size) if self.capacity > 0 else 0
            return pack_frame_full(
                coords=coords_ang,
                frame_id=self.step_count,
                sim_time_ps=self.sim_time_ps,
                box_H=box_ang,
                scalars=scalars,
                use_int16=self.use_int16 or (self.capacity >= 250_000),
                pbc_wrapped=True,
                reversed_step=reversed_step,
                scalar_channel=ch_idx,
            )

    def encode_overlay_packet(self, max_vectors: int = 2000) -> bytes:
        """Encode ROI force vectors + PBC-safe bonds (skipping cell-wrapping long bonds)."""
        with self._lock:
            act_idx = np.flatnonzero(self.active_mask)
            coords_ang = self.positions * self.coord_to_ang
            f_mags = np.linalg.norm(self.forces[act_idx], axis=1)
            k = min(int(max_vectors), act_idx.size)
            if k > 0:
                top_local = np.argpartition(f_mags, -k)[-k:] if k < act_idx.size else np.arange(act_idx.size)
                sel = act_idx[top_local]
                starts = coords_ang[sel].astype(np.float32)
                dirs = (self.forces[sel] * 0.02).astype(np.float32)
                # Normalize exaggerated vector length safely
                mags = f_mags[top_local].astype(np.float32)
            else:
                starts = np.zeros((0, 3), dtype=np.float32)
                dirs = np.zeros((0, 3), dtype=np.float32)
                mags = np.zeros(0, dtype=np.float32)

            if self.bonds:
                b_arr = np.asarray(self.bonds, dtype=np.uint32)
                # Filter out cross-PBC bonds longer than half box diagonal so cylinders never streak
                box_diag_half = 0.45 * float(np.min(np.diag(self.box_H))) * self.coord_to_ang
                b_len = np.linalg.norm(coords_ang[b_arr[:, 0]] - coords_ang[b_arr[:, 1]], axis=1)
                non_wrap = b_len <= min(box_diag_half, 3.2)
                bp = b_arr[non_wrap]
                bo = self.bond_orders[non_wrap]
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
        """Compute live 3^k TernaryMSM electrostatic/charge density 3D field via Metal/CPU 27-pt stencil."""
        with self._lock:
            n_grid = int(ml)
            Q_grid = np.zeros((n_grid, n_grid, n_grid), dtype=np.float64)
            H = self.box_H
            Hinv = np.linalg.inv(H)
            act = np.flatnonzero(self.active_mask)
            if act.size > 0:
                frac = self.positions[act] @ Hinv.T
                frac -= np.floor(frac)
                idx = np.clip((frac * n_grid).astype(np.int32), 0, n_grid - 1)
                wts = np.where(np.abs(self.charges[act]) > 1e-6, self.charges[act], self.ternary_states[act] + 0.1)
                np.add.at(Q_grid, (idx[:, 0], idx[:, 1], idx[:, 2]), wts)

            # Canonical isotropic 27-point Coulomb MSM stencil weights
            k27 = np.zeros(27, dtype=np.float64)
            for dz in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        r2 = dx * dx + dy * dy + dz * dz
                        idx27 = (dx + 1) + 3 * (dy + 1) + 9 * (dz + 1)
                        k27[idx27] = 1.0 if r2 == 0 else 1.0 / (math.sqrt(r2) * 3.0)

            try:
                _, phi = gpu_msm_stencil27(Q_grid, k27, prefactor=1.0, backend="auto")
            except RuntimeError:
                phi = np.zeros_like(Q_grid)
                for dz in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            w = k27[(dx + 1) + 3 * (dy + 1) + 9 * (dz + 1)]
                            phi += w * np.roll(Q_grid, shift=(-dx, -dy, -dz), axis=(0, 1, 2))
            box_ang = (self.box_H * self.coord_to_ang).astype(np.float32)
            return pack_volume_grid3d(phi.astype(np.float32), box_H=box_ang, origin=(0.0, 0.0, 0.0), frame_id=self.step_count, sim_time_ps=self.sim_time_ps)
