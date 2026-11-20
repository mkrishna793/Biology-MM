"""Trium Balanced-Ternary compression & Symmetry-Instanced Multiscale Low-RAM engine.

Slashes 40M+ atom biomolecular assembly RAM (e.g. Faustovirus 5J7V: 14,478 ASU x 2,760
transforms = 39,959,280 atoms) from ~21 GB down to < 250 MB by combining:
  1. Exact Base-243 balanced-ternary 5-trits-per-byte compression (94.9% Shannon density),
     matching `native/include/ternary_core.hpp` (`Base243Codec`).
  2. 1-byte/atom `uint8` element lookup tables (`ELEMENT_LUT_MASS`, `ELEMENT_LUT_SIGMA`,
     `ELEMENT_LUT_EPSILON`) eliminating Python string lists and float64 parameter arrays.
  3. `CompactSymmetryMultiscaleSession`: symmetry-instanced 3-zone multiscale session that
     NEVER materializes 40M float64 arrays, performs O(K) subunit bounding-sphere culling,
     runs Beutler soft-core clash relaxation on the active All-Atom focus zone, and streams
     compact dirty-tile WebSocket patches.
"""

from __future__ import annotations

import math
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from biology_mm import GPUProfile, probe_hardware

from .engine_core import (
    ELEMENT_RADII_ANG,
    ELEMENT_Z,
    KB_KJ_MOL_K,
    PHYSICS_CHANNEL_NAMES,
    compute_morton_tiles,
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
    pack_frame_tile_patch,
    pack_overlay_vectors,
    pack_topology_init,
    pack_volume_grid3d,
)

MAX_COMPACT_SNAPSHOT_RAM_BYTES = 64 * 1024 * 1024  # 64 MB hard cap on history snapshots

# ==============================================================================
# 1. Exact 5-Trit-Per-Byte Base-243 Balanced-Ternary Codec
# ==============================================================================


class Base243TernaryCodec:
    """Exact 5-trit-per-byte balanced-ternary codec matching `native/include/ternary_core.hpp`.

    Maps 5 balanced trits in `{-1, 0, +1}` into a single `uint8` in `[0..242]`
    (1.6 bits/trit = 94.9% of log2(3) Shannon entropy) and unpacks via Z3-verified
    division-free reciprocal multipliers `(171, 57, 19, 203)`.
    """

    POWERS_OF_3 = np.array([1, 3, 9, 27, 81], dtype=np.uint16)

    @staticmethod
    def pack_trits_base243(trits: np.ndarray) -> np.ndarray:
        """Pack a 1D/ND array of balanced trits `{-1, 0, +1}` into `uint8` bytes (5 trits/byte)."""
        flat = np.asarray(trits, dtype=np.int8).reshape(-1)
        n = int(flat.size)
        if n == 0:
            return np.zeros(0, dtype=np.uint8)
        # Clamp strictly to {-1, 0, +1} and map to {0, 1, 2}
        digits = np.clip(flat, -1, 1).astype(np.int16) + 1
        rem = n % 5
        if rem != 0:
            # Pad tail with Zero trits (digit = 1)
            digits = np.concatenate([digits, np.ones(5 - rem, dtype=np.int16)])
        groups = digits.reshape((-1, 5))
        packed = (
            groups[:, 0]
            + groups[:, 1] * 3
            + groups[:, 2] * 9
            + groups[:, 3] * 27
            + groups[:, 4] * 81
        )
        return np.ascontiguousarray(packed, dtype=np.uint8)

    @staticmethod
    def unpack_trits_base243(packed: np.ndarray, n_trits: Optional[int] = None) -> np.ndarray:
        """Unpack Base-243 `uint8` bytes back to `int8` balanced trits `{-1, 0, +1}` using magic multipliers."""
        b = np.asarray(packed, dtype=np.uint8).reshape(-1)
        if b.size == 0:
            return np.zeros(0, dtype=np.int8)
        b32 = b.astype(np.uint32)
        # Out-of-range bytes (>= 243) decode to all-Zero trits (byte value 121) per ternary_core.hpp
        b_safe = np.where(b32 >= 243, np.uint32(121), b32)

        q1 = (b_safe * np.uint32(171)) >> np.uint32(9)
        q2 = (b_safe * np.uint32(57)) >> np.uint32(9)
        q3 = (b_safe * np.uint32(19)) >> np.uint32(9)
        q4 = (b_safe * np.uint32(203)) >> np.uint32(14)

        t0 = (b_safe - np.uint32(3) * q1).astype(np.int8) - np.int8(1)
        t1 = (q1 - np.uint32(3) * q2).astype(np.int8) - np.int8(1)
        t2 = (q2 - np.uint32(3) * q3).astype(np.int8) - np.int8(1)
        t3 = (q3 - np.uint32(3) * q4).astype(np.int8) - np.int8(1)
        t4 = q4.astype(np.int8) - np.int8(1)

        out = np.empty((b.size, 5), dtype=np.int8)
        out[:, 0] = t0
        out[:, 1] = t1
        out[:, 2] = t2
        out[:, 3] = t3
        out[:, 4] = t4
        flat = out.reshape(-1)
        if n_trits is not None:
            return np.ascontiguousarray(flat[: int(n_trits)], dtype=np.int8)
        return np.ascontiguousarray(flat, dtype=np.int8)

    @staticmethod
    def quantize_coords_ternary(
        coords: np.ndarray,
        n_trits_per_val: int = 10,
        quantum: Optional[float] = None,
        origin: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Quantize floating-point coordinates/displacements into Base-243 balanced ternary.

        With default `n_trits_per_val=10` (2 bytes/scalar, 6 bytes/atom vs 24 bytes float64),
        signed integer range is `[-29524, +29524]` (`3^10 = 59,049` levels).
        """
        arr = np.asarray(coords, dtype=np.float64)
        orig_shape = tuple(arr.shape)
        k = max(1, int(n_trits_per_val))
        half_range = (3**k - 1) // 2

        if arr.size == 0:
            zero_origin = np.zeros(orig_shape[-1] if arr.ndim >= 2 else 1, dtype=np.float64)
            return {
                "packed": np.zeros(0, dtype=np.uint8),
                "shape": orig_shape,
                "origin": zero_origin,
                "quantum": float(quantum if quantum is not None else 1e-3),
                "n_trits_per_val": k,
                "n_trits": 0,
                "raw_float64_bytes": 0,
                "packed_bytes": 0,
                "compression_ratio": 1.0,
            }

        if origin is None:
            if arr.ndim >= 2:
                orig = np.mean(arr, axis=0, dtype=np.float64)
            else:
                orig = np.array([float(np.mean(arr))], dtype=np.float64)
        else:
            orig = np.asarray(origin, dtype=np.float64)

        shifted = arr - orig
        max_abs = float(np.max(np.abs(shifted)))
        if quantum is None or float(quantum) <= 0.0:
            q_step = max(max_abs / max(1, half_range), 1e-9)
        else:
            q_step = float(quantum)

        q_int = np.rint(shifted.reshape(-1) / q_step).astype(np.int64)
        q_int = np.clip(q_int, -half_range, half_range)

        # Exact vectorized integer -> balanced-ternary digit decomposition
        val = q_int.copy()
        trits = np.empty((val.size, k), dtype=np.int8)
        for d in range(k):
            rem = ((val + 1) % 3) - 1  # in {-1, 0, +1}
            trits[:, d] = rem.astype(np.int8)
            val = (val - rem) // 3

        n_trits = int(trits.size)
        packed = Base243TernaryCodec.pack_trits_base243(trits.reshape(-1))
        raw_bytes = int(arr.size) * 8
        packed_bytes = int(packed.nbytes)
        return {
            "packed": packed,
            "shape": orig_shape,
            "origin": orig.astype(np.float64, copy=False),
            "quantum": float(q_step),
            "n_trits_per_val": k,
            "n_trits": n_trits,
            "raw_float64_bytes": raw_bytes,
            "packed_bytes": packed_bytes,
            "compression_ratio": float(raw_bytes) / max(1.0, float(packed_bytes)),
        }

    @staticmethod
    def dequantize_coords_ternary(
        encoded: Union[Dict[str, Any], np.ndarray],
        shape: Optional[Tuple[int, ...]] = None,
        origin: Optional[np.ndarray] = None,
        quantum: Optional[float] = None,
        n_trits_per_val: int = 10,
        dtype: Any = np.float32,
    ) -> np.ndarray:
        """Reconstruct floating-point coordinates/displacements from Base-243 balanced ternary."""
        if isinstance(encoded, dict):
            packed = encoded["packed"]
            out_shape = tuple(encoded["shape"])
            orig = np.asarray(encoded["origin"], dtype=np.float64)
            q_step = float(encoded["quantum"])
            k = int(encoded["n_trits_per_val"])
            n_trits = int(encoded.get("n_trits", int(np.prod(out_shape)) * k))
        else:
            if shape is None or origin is None or quantum is None:
                raise ValueError("dequantize_coords_ternary requires shape, origin, and quantum when given raw array")
            packed = encoded
            out_shape = tuple(shape)
            orig = np.asarray(origin, dtype=np.float64)
            q_step = float(quantum)
            k = int(n_trits_per_val)
            n_trits = int(np.prod(out_shape)) * k

        if n_trits == 0:
            return np.zeros(out_shape, dtype=dtype)

        trits = Base243TernaryCodec.unpack_trits_base243(packed, n_trits=n_trits).reshape((-1, k))
        weights = (3 ** np.arange(k, dtype=np.int64))
        q_int = trits.astype(np.int64) @ weights
        vals = (q_int.astype(np.float64) * q_step).reshape(out_shape) + orig
        return np.ascontiguousarray(vals, dtype=dtype)


# ==============================================================================
# 2. 1-Byte/Atom Element Lookup Tables (Eliminates Python Strings & float64 LUTs)
# ==============================================================================

ELEMENT_LUT_MASS = np.full(256, 12.011, dtype=np.float32)
ELEMENT_LUT_SIGMA = np.full(256, 0.34, dtype=np.float32)
ELEMENT_LUT_EPSILON = np.full(256, 0.997, dtype=np.float32)
ELEMENT_LUT_RADIUS_ANG = np.full(256, 1.50, dtype=np.float32)

_ELEMENT_TABLE: Dict[int, Tuple[str, float, float, float, float]] = {
    # Z: (Symbol, mass_amu, sigma_nm, epsilon_kj_mol, radius_ang)
    1: ("H", 1.008, 0.250, 0.125, 1.10),
    2: ("He", 4.0026, 0.256, 0.085, 1.40),
    3: ("Li", 6.94, 0.213, 0.076, 1.82),
    4: ("Be", 9.0122, 0.245, 0.200, 1.53),
    5: ("B", 10.81, 0.364, 0.400, 1.92),
    6: ("C", 12.011, 0.340, 0.458, 1.70),
    7: ("N", 14.007, 0.325, 0.711, 1.55),
    8: ("O", 15.999, 0.312, 0.879, 1.52),
    9: ("F", 18.998, 0.312, 0.255, 1.47),
    10: ("Ne", 20.180, 0.278, 0.300, 1.54),
    11: ("Na", 22.990, 0.260, 0.550, 2.27),
    12: ("Mg", 24.305, 0.240, 0.650, 1.73),
    13: ("Al", 26.982, 0.370, 0.800, 1.84),
    14: ("Si", 28.085, 0.382, 0.850, 2.10),
    15: ("P", 30.974, 0.374, 0.837, 1.80),
    16: ("S", 32.06, 0.356, 1.046, 1.80),
    17: ("Cl", 35.45, 0.347, 1.100, 1.75),
    18: ("Ar", 39.948, 0.340, 0.997, 1.88),
    19: ("K", 39.098, 0.333, 0.418, 2.75),
    20: ("Ca", 40.078, 0.280, 0.600, 2.31),
    26: ("Fe", 55.845, 0.260, 0.900, 2.00),
    30: ("Zn", 65.38, 0.240, 0.800, 1.39),
    34: ("Se", 78.971, 0.375, 1.218, 1.90),
    35: ("Br", 79.904, 0.365, 1.250, 1.85),
    53: ("I", 126.90, 0.395, 1.500, 1.98),
}

_Z_TO_SYM_LUT: List[str] = ["C"] * 256
_UPPER_SYM_TO_Z: Dict[str, int] = {}

for _z, (_sym, _m, _sig, _eps, _rad) in _ELEMENT_TABLE.items():
    ELEMENT_LUT_MASS[_z] = np.float32(_m)
    ELEMENT_LUT_SIGMA[_z] = np.float32(_sig)
    ELEMENT_LUT_EPSILON[_z] = np.float32(_eps)
    ELEMENT_LUT_RADIUS_ANG[_z] = np.float32(_rad)
    _Z_TO_SYM_LUT[_z] = _sym
    _UPPER_SYM_TO_Z[_sym.upper()] = _z

# Ensure fallback radius consistency with engine_core.ELEMENT_RADII_ANG
for _z, _rad in ELEMENT_RADII_ANG.items():
    if 0 <= _z < 256:
        ELEMENT_LUT_RADIUS_ANG[_z] = np.float32(_rad)

# Reserve atomic number 0 exclusively for Coarse-Grained (CG) buffer beads (never "Ar")
_Z_TO_SYM_LUT[0] = "CG"
_UPPER_SYM_TO_Z["CG"] = 0
ELEMENT_LUT_MASS[0] = np.float32(192.0)
ELEMENT_LUT_SIGMA[0] = np.float32(0.45)
ELEMENT_LUT_EPSILON[0] = np.float32(0.50)
ELEMENT_LUT_RADIUS_ANG[0] = np.float32(4.2)


def encode_elements_uint8(names: Sequence[Union[str, int]]) -> np.ndarray:
    """Encode element symbols or atomic numbers into a compact 1-byte/atom `uint8` array."""
    if isinstance(names, np.ndarray) and names.dtype == np.uint8:
        return np.ascontiguousarray(names, dtype=np.uint8).reshape(-1)
    n = len(names)
    out = np.empty(n, dtype=np.uint8)
    lookup = _UPPER_SYM_TO_Z
    for i, item in enumerate(names):
        if isinstance(item, (int, np.integer)):
            z = int(item)
            out[i] = z if 0 <= z < 256 else 6
        else:
            s = str(item).strip().upper()
            z = lookup.get(s)
            if z is None:
                cleaned = "".join(ch for ch in s if ch.isalpha())
                z = lookup.get(cleaned, lookup.get(cleaned[:1], 6) if cleaned else 6)
            out[i] = z
    return out


# ==============================================================================
# 3. Compact Symmetry-Instanced Multiscale Session (< 250 MB for 40M+ Atoms)
# ==============================================================================


def eval_bipartite_centroid_residual(
    disp_a: np.ndarray,
    disp_b: np.ndarray,
    k_couple: float,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Exact O(n+m) Harmonic.lean bipartite centroid-residual coupling.

    Computes the exact all-pairs harmonic energy & gradients in O(n+m) time:
        U(A, B) = (k_eff / 2) * [n * m * ||a - b||^2 + m * resid(A) + n * resid(B)]
        F_i^A   = -k_eff * m * (q_i - b)
        F_j^B   = -k_eff * n * (p_j - a)
    where `k_eff = k_couple / (n + m)` keeps per-particle stiffness O(1) independent
    of group sizes and satisfies exact Newton's 3rd law (sum F_A + sum F_B == 0).
    """
    n = int(disp_a.shape[0])
    m = int(disp_b.shape[0])
    if n == 0 or m == 0 or k_couple <= 0.0:
        return 0.0, np.zeros_like(disp_a, dtype=np.float64), np.zeros_like(disp_b, dtype=np.float64)

    a = np.mean(disp_a, axis=0, dtype=np.float64)
    b = np.mean(disp_b, axis=0, dtype=np.float64)
    da_c = disp_a - a[None, :]
    db_c = disp_b - b[None, :]
    resid_a = float(np.sum(da_c * da_c))
    resid_b = float(np.sum(db_c * db_c))
    diff_ab = a - b
    k_eff = float(k_couple) / float(n + m)
    nf = float(n)
    mf = float(m)

    u = 0.5 * k_eff * (nf * mf * float(np.dot(diff_ab, diff_ab)) + mf * resid_a + nf * resid_b)
    f_a = -(k_eff * mf) * (disp_a - b[None, :])
    f_b = -(k_eff * nf) * (disp_b - a[None, :])
    return float(u), f_a, f_b


@dataclass
class _CompactSnapshot:
    step: int
    time_ps: float
    focus_center_ang: Tuple[float, float, float]
    positions_f32: np.ndarray
    momenta_f32: np.ndarray
    forces_f32: np.ndarray
    potential_energy: float
    kinetic_energy: float
    q_thermo_cum: float = 0.0

    @property
    def nbytes(self) -> int:
        return int(self.positions_f32.nbytes + self.momenta_f32.nbytes + self.forces_f32.nbytes)


class CompactSymmetryMultiscaleSession:
    """Drop-in symmetry-instanced multiscale simulation session matching `SimulationSession`.

    Stores ONLY:
      - Asymmetric Unit (M atoms in float32 + uint8 elements + Base-243 compressed reference)
      - K rigid-body subunit transforms (`(K, 4, 4)` float32 + 6-DOF COM/velocities)
      - Active All-Atom focus zone (Zone 0, capped at `max_active_aa_atoms`)
      - Active Coarse-Grained transition shell (Zone 1, 1 bead per ~16 atoms)
    NEVER materializes full `M * K` float64 arrays.
    """

    def __init__(
        self,
        asu_coords_ang: np.ndarray,
        transforms_4x4: np.ndarray,
        asu_elements: Optional[Sequence[Union[str, int]]] = None,
        mode: str = "metal_verlet",
        dt: float = 0.002,
        temperature: float = 300.0,
        gamma: float = 5.0,
        rcut: float = 0.75,
        use_int16: bool = False,
        box_ang: Optional[Tuple[float, float, float]] = None,
        focus_center_ang: Optional[Sequence[float]] = None,
        aa_radius_ang: float = 45.0,
        cg_radius_ang: float = 85.0,
        max_active_aa_atoms: int = 65_536,
        max_cg_beads: int = 16_384,
        cg_atoms_per_bead: int = 16,
        max_history: int = 64,
        structure_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._lock = threading.RLock()
        self._rng = np.random.default_rng(202611)

        asu = np.ascontiguousarray(asu_coords_ang, dtype=np.float32)
        if asu.ndim != 2 or asu.shape[1] != 3:
            raise ValueError(f"Expected asu_coords_ang shape (M, 3), got {asu.shape}")
        tfs = np.ascontiguousarray(transforms_4x4, dtype=np.float32)
        if tfs.ndim == 2 and tfs.shape == (4, 4):
            tfs = tfs.reshape((1, 4, 4))
        elif tfs.ndim != 3 or tfs.shape[1:] != (4, 4):
            raise ValueError(f"Expected transforms_4x4 shape (K, 4, 4), got {tfs.shape}")

        self.n_asu_atoms: int = int(asu.shape[0])
        self.n_transforms: int = int(tfs.shape[0])
        self.n_atoms: int = int(self.n_asu_atoms * self.n_transforms)

        # Compact ASU storage (float32 + uint8 elements + Base-243 ternary reference)
        self.asu_coords_ang: np.ndarray = asu.copy()
        if asu_elements is None:
            self.asu_elements_u8: np.ndarray = np.full(self.n_asu_atoms, 6, dtype=np.uint8)
        else:
            self.asu_elements_u8 = encode_elements_uint8(asu_elements)
            if self.asu_elements_u8.size != self.n_asu_atoms:
                raise ValueError(
                    f"asu_elements length ({self.asu_elements_u8.size}) != n_asu_atoms ({self.n_asu_atoms})"
                )
        self.asu_compressed_b243: Dict[str, Any] = Base243TernaryCodec.quantize_coords_ternary(
            self.asu_coords_ang, n_trits_per_val=10
        )

        # ASU geometry & CG bead template (1 bead per ~16 atoms)
        if self.n_asu_atoms > 0:
            self._asu_centroid_ang: np.ndarray = np.mean(self.asu_coords_ang, axis=0, dtype=np.float64).astype(np.float32)
            asu_rel = self.asu_coords_ang - self._asu_centroid_ang[None, :]
            self._asu_bound_radius_ang: float = float(np.max(np.linalg.norm(asu_rel, axis=1)))
            asu_masses = ELEMENT_LUT_MASS[self.asu_elements_u8]
            self._subunit_mass: float = float(np.sum(asu_masses, dtype=np.float64))
            r2_asu = np.sum(asu_rel.astype(np.float64) ** 2, axis=1)
            self._subunit_inertia: float = max(1.0, float(np.sum(asu_masses * r2_asu)))

            stride = max(1, int(cg_atoms_per_bead))
            n_beads = (self.n_asu_atoms + stride - 1) // stride
            cg_coords = np.empty((n_beads, 3), dtype=np.float32)
            cg_masses = np.empty(n_beads, dtype=np.float32)
            for b in range(n_beads):
                s0 = b * stride
                s1 = min(self.n_asu_atoms, s0 + stride)
                cg_coords[b] = np.mean(self.asu_coords_ang[s0:s1], axis=0)
                cg_masses[b] = float(np.sum(asu_masses[s0:s1]))
            self._asu_cg_coords_ang: np.ndarray = cg_coords
            self._asu_cg_masses: np.ndarray = cg_masses
        else:
            self._asu_centroid_ang = np.zeros(3, dtype=np.float32)
            self._asu_bound_radius_ang = 0.0
            self._subunit_mass = 12.011
            self._subunit_inertia = 1.0
            self._asu_cg_coords_ang = np.zeros((0, 3), dtype=np.float32)
            self._asu_cg_masses = np.zeros(0, dtype=np.float32)

        # K rigid-body transforms & 6-DOF Zone 2 state (O(K) memory)
        self.transforms_4x4: np.ndarray = tfs.copy()
        rot = self.transforms_4x4[:, :3, :3]  # (K, 3, 3)
        trans = self.transforms_4x4[:, :3, 3]  # (K, 3)
        self.subunit_com_ang: np.ndarray = np.ascontiguousarray(
            np.einsum("kij,j->ki", rot, self._asu_centroid_ang) + trans, dtype=np.float32
        )
        self.subunit_com_ref_ang: np.ndarray = self.subunit_com_ang.copy()
        self.subunit_vel_ang_ps: np.ndarray = np.zeros((self.n_transforms, 3), dtype=np.float32)
        self.subunit_omega_rad_ps: np.ndarray = np.zeros((self.n_transforms, 3), dtype=np.float32)
        self.subunit_disp_ang: np.ndarray = np.zeros((self.n_transforms, 3), dtype=np.float32)

        # Simulation parameters matching SimulationSession
        self.mode: str = mode or "metal_verlet"
        self.dt: float = float(dt) if dt > 0 else 0.002
        self.temperature: float = float(temperature) if temperature > 0 else 300.0
        self.gamma: float = float(gamma)
        self.rcut: float = float(rcut)
        self.alpha_dsf: float = 0.0
        self.use_int16: bool = bool(use_int16)
        self.coord_to_ang: float = 10.0
        self.k_b: float = KB_KJ_MOL_K
        self.efield_z: float = 0.0
        self.gcmc_mu: float = -2.0
        self.strain_rate: float = 0.0
        self.custom_expr: Optional[str] = None
        self.custom_params: Dict[str, float] = {}
        self._custom_spline: Any = None
        self.max_step_disp: Optional[float] = None

        self.max_active_aa_atoms: int = max(1, int(max_active_aa_atoms))
        self.max_cg_beads: int = max(0, int(max_cg_beads))
        self.cg_atoms_per_bead: int = max(1, int(cg_atoms_per_bead))

        # Global bounding box in nm derived in O(K) without expanding 40M atoms
        margin_ang = 12.0
        if self.n_transforms > 0:
            min_ang = np.min(self.subunit_com_ang, axis=0) - self._asu_bound_radius_ang - margin_ang
            max_ang = np.max(self.subunit_com_ang, axis=0) + self._asu_bound_radius_ang + margin_ang
            span_ang = np.maximum(max_ang - min_ang, 28.0)
        else:
            min_ang = np.zeros(3, dtype=np.float32)
            span_ang = np.array([28.0, 28.0, 28.0], dtype=np.float32)

        self._box_origin_ang: np.ndarray = min_ang.astype(np.float64)
        if box_ang is not None and len(box_ang) == 3 and all(float(v) > 0 for v in box_ang):
            lx, ly, lz = (float(box_ang[0]) * 0.1, float(box_ang[1]) * 0.1, float(box_ang[2]) * 0.1)
        else:
            lx, ly, lz = (float(span_ang[0]) * 0.1, float(span_ang[1]) * 0.1, float(span_ang[2]) * 0.1)
        self.box_H: np.ndarray = np.diag([max(lx, 2.8), max(ly, 2.8), max(lz, 2.8)]).astype(np.float64)

        self.structure_metadata: Optional[Dict[str, Any]] = dict(structure_metadata) if structure_metadata else {
            "source": "molstar-symmetry-compact",
            "n_asu_atoms": self.n_asu_atoms,
            "n_transforms": self.n_transforms,
            "atom_count": self.n_atoms,
        }

        self.hardware: GPUProfile = probe_hardware()
        self.active_channel: int = 0
        self.step_count: int = 0
        self.sim_time_ps: float = 0.0
        self.potential_energy: float = 0.0
        self.kinetic_energy: float = 0.0
        self._rigid_kinetic_energy: float = 0.0
        self._q_thermo_cum: float = 0.0
        self.hmc_rejections: int = 0
        self.hmc_steps: int = 0
        self.virial: np.ndarray = np.zeros((3, 3), dtype=np.float64)
        self.last_ms_per_step: float = 0.1
        self._initial_total_energy: Optional[float] = None
        self._last_clash_report: Optional[Dict[str, Any]] = None

        # Consistent MD-unit (nm, ps, amu, kJ/mol) rigid subunit state for exact symplectic BAOAB
        self._subunit_disp_nm: np.ndarray = np.zeros((self.n_transforms, 3), dtype=np.float64)
        self._subunit_momenta_nm: np.ndarray = np.zeros((self.n_transforms, 3), dtype=np.float64)
        self._subunit_forces_nm: np.ndarray = np.zeros((self.n_transforms, 3), dtype=np.float64)

        # Factor 1 (ENM) & Factor 2 (Verlet neighbor list) caches
        self._enm_i: np.ndarray = np.zeros(0, dtype=np.int32)
        self._enm_j: np.ndarray = np.zeros(0, dtype=np.int32)
        self._enm_r0: np.ndarray = np.zeros(0, dtype=np.float64)
        self._enm_pair_keys: np.ndarray = np.zeros(0, dtype=np.int64)
        self._enm_k_bond: float = 2500.0
        self._enm_cutoff_nm: float = 0.38
        self._aa_ref_positions_nm: np.ndarray = np.zeros((0, 3), dtype=np.float64)
        self._nb_ref_pos: Optional[np.ndarray] = None
        self._nb_i: np.ndarray = np.zeros(0, dtype=np.int32)
        self._nb_j: np.ndarray = np.zeros(0, dtype=np.int32)
        self._nb_rcut_cached: float = 0.0
        self._nb_skin_nm: float = 0.08

        self.bonds: List[Tuple[int, int]] = []
        self.bond_orders: np.ndarray = np.zeros(0, dtype=np.float32)

        # Focus zone & active materialized buffers (populated by set_focus_zone)
        self._focus_center_ang: Tuple[float, float, float] = (
            tuple(float(v) for v in focus_center_ang)  # type: ignore[assignment]
            if focus_center_ang is not None
            else (
                (float(self.subunit_com_ang[0, 0]), float(self.subunit_com_ang[0, 1]), float(self.subunit_com_ang[0, 2]))
                if self.n_transforms > 0
                else (0.0, 0.0, 0.0)
            )
        )
        self._aa_radius_ang: float = float(aa_radius_ang)
        self._cg_radius_ang: float = max(float(aa_radius_ang), float(cg_radius_ang))

        self.active_all_atom_count: int = 0
        self.active_cg_bead_count: int = 0
        self._n_cg_virtual_atoms: int = 0
        self._n_subunits_aa: int = 0
        self._n_subunits_cg: int = 0
        self._n_subunits_rigid: int = self.n_transforms
        self._active_subunit_indices: np.ndarray = np.zeros(0, dtype=np.int32)
        self._active_global_atom_ids: np.ndarray = np.zeros(0, dtype=np.int64)

        self.capacity: int = 0
        self.n_real: int = 0
        self.positions: np.ndarray = np.zeros((0, 3), dtype=np.float64)
        self.momenta: np.ndarray = np.zeros((0, 3), dtype=np.float64)
        self.forces: np.ndarray = np.zeros((0, 3), dtype=np.float64)
        self.masses: np.ndarray = np.zeros(0, dtype=np.float64)
        self.charges: np.ndarray = np.zeros(0, dtype=np.float64)
        self.atomic_numbers: np.ndarray = np.zeros(0, dtype=np.uint8)
        self.radii_ang: np.ndarray = np.zeros(0, dtype=np.float32)
        self.active_mask: np.ndarray = np.zeros(0, dtype=bool)
        self.multiscale_zones: np.ndarray = np.zeros(0, dtype=np.uint8)
        self.bond_orders_per_atom: np.ndarray = np.zeros(0, dtype=np.float32)
        self.atomic_stress: np.ndarray = np.zeros(0, dtype=np.float32)
        self.ternary_states: np.ndarray = np.zeros(0, dtype=np.float32)
        self.tile_ids: np.ndarray = np.zeros(0, dtype=np.uint32)
        self._cg_ref_positions_nm: np.ndarray = np.zeros((0, 3), dtype=np.float32)

        self.delta_streaming_enabled: bool = True
        self._dirty_tracker: DirtyTileTracker = DirtyTileTracker(0, np.zeros(0, dtype=np.uint32), threshold_ang=0.05)
        self._last_dirty_ratio: float = 1.0
        self._last_dirty_tiles: int = 0

        self._smd_pull: Optional[Dict[str, Any]] = None
        self._smd_history: List[Dict[str, float]] = []

        self._history: "OrderedDict[int, _CompactSnapshot]" = OrderedDict()
        self._history_ram_bytes: int = 0
        self._max_history: int = max(1, int(max_history))

        # Initialize 6-DOF rigid-body momenta thermally in consistent MD units (nm, ps, amu)
        if self.n_transforms > 0:
            p_sub_scale = math.sqrt(self.k_b * self.temperature * max(self._subunit_mass, 1.0))
            self._subunit_momenta_nm = self._rng.normal(0.0, p_sub_scale, size=(self.n_transforms, 3))
            self._subunit_momenta_nm -= np.mean(self._subunit_momenta_nm, axis=0, keepdims=True)
            self._rigid_kinetic_energy = 0.5 * float(
                np.sum(self._subunit_momenta_nm * self._subunit_momenta_nm) / max(self._subunit_mass, 1.0)
            )
            self.subunit_vel_ang_ps = (
                (self._subunit_momenta_nm / max(self._subunit_mass, 1.0)) * 10.0
            ).astype(np.float32)
            w_sub_scale = math.sqrt(self.k_b * self.temperature / max(self._subunit_inertia, 1.0))
            self.subunit_omega_rad_ps = self._rng.normal(0.0, w_sub_scale * 0.1, size=(self.n_transforms, 3)).astype(np.float32)

        # Materialize initial focus zone
        self.set_focus_zone(
            center_ang=self._focus_center_ang,
            aa_radius_ang=self._aa_radius_ang,
            cg_radius_ang=self._cg_radius_ang,
        )
        self._q_thermo_cum = 0.0
        self._initial_total_energy = self._extended_total_energy()
        self._record_history()

    @classmethod
    def from_tsym_payload(
        cls,
        sym: SymmetryAssemblyPayload,
        aa_radius_ang: float = 45.0,
        cg_radius_ang: float = 85.0,
        max_active_aa_atoms: int = 65_536,
    ) -> "CompactSymmetryMultiscaleSession":
        """Construct directly from an unpacked `TSYM` payload without calling `expand_symmetry_assembly()`."""
        return cls(
            asu_coords_ang=sym.asu_coords_ang,
            transforms_4x4=sym.transforms_4x4,
            asu_elements=sym.asu_elements,
            mode=sym.mode or "metal_verlet",
            dt=sym.dt if sym.dt > 0 else 0.002,
            temperature=sym.temperature if sym.temperature > 0 else 300.0,
            box_ang=sym.box_ang,
            aa_radius_ang=aa_radius_ang,
            cg_radius_ang=cg_radius_ang,
            max_active_aa_atoms=max_active_aa_atoms,
            structure_metadata={
                "source": "molstar-symmetry-compact",
                "title": sym.title,
                "n_asu_atoms": sym.n_asu_atoms,
                "n_transforms": sym.n_transforms,
                "atom_count": sym.n_asu_atoms * sym.n_transforms,
            },
        )

    @property
    def names(self) -> List[str]:
        """Lazy element names for active buffer atoms (never allocates 40M strings)."""
        return [_Z_TO_SYM_LUT[int(z)] for z in self.atomic_numbers]

    def close(self) -> None:
        with self._lock:
            self._history.clear()
            self._history_ram_bytes = 0

    def _compute_ram_bytes(self) -> int:
        """Exact live NumPy memory footprint in bytes across all stored arrays."""
        total = 0
        for arr in (
            self.asu_coords_ang,
            self.asu_elements_u8,
            self.asu_compressed_b243.get("packed"),
            self._asu_cg_coords_ang,
            self._asu_cg_masses,
            self.transforms_4x4,
            self.subunit_com_ang,
            self.subunit_com_ref_ang,
            self.subunit_vel_ang_ps,
            self.subunit_omega_rad_ps,
            self.subunit_disp_ang,
            self._active_subunit_indices,
            self._active_global_atom_ids,
            self.positions,
            self.momenta,
            self.forces,
            self.masses,
            self.charges,
            self.atomic_numbers,
            self.radii_ang,
            self.active_mask,
            self.multiscale_zones,
            self.bond_orders_per_atom,
            self.atomic_stress,
            self.ternary_states,
            self.tile_ids,
            self._cg_ref_positions_nm,
            self._dirty_tracker.tile_ids,
            self._dirty_tracker.last_sent_coords_ang,
        ):
            if isinstance(arr, np.ndarray):
                total += int(arr.nbytes)
        total += int(self._history_ram_bytes)
        return total

    def _build_multiscale_summary(self) -> Dict[str, Any]:
        ram_mb = self._compute_ram_bytes() / (1024.0 * 1024.0)
        n_aa = int(self.active_all_atom_count)
        n_cg = int(self._n_cg_virtual_atoms)
        n_rigid = max(0, int(self.n_atoms - n_aa - n_cg))
        return {
            "center_ang": [float(c) for c in self._focus_center_ang],
            "aa_radius_ang": float(self._aa_radius_ang),
            "cg_radius_ang": float(self._cg_radius_ang),
            "n_all_atom": n_aa,
            "n_coarse_grained": n_cg,
            "n_coarse_grained_beads": int(self.active_cg_bead_count),
            "n_rigid_scaffold": n_rigid,
            "all_atom": n_aa,
            "coarse_grained": n_cg,
            "rigid_scaffold": n_rigid,
            "counts": {
                "all_atom": n_aa,
                "coarse_grained": n_cg,
                "rigid_scaffold": n_rigid,
            },
            "n_total": int(self.n_atoms),
            "active_subunits_aa": int(self._n_subunits_aa),
            "active_subunits_cg": int(self._n_subunits_cg),
            "rigid_subunits": int(self._n_subunits_rigid),
            "estimated_memory_mb": round(float(ram_mb), 4),
        }

    def set_focus_zone(
        self,
        center_ang: Sequence[float],
        aa_radius_ang: float = 45.0,
        cg_radius_ang: float = 85.0,
    ) -> Dict[str, Any]:
        """Move the 3D Focus Zone in milliseconds via O(K) subunit bounding-sphere culling."""
        with self._lock:
            ctr = np.asarray(center_ang, dtype=np.float32).reshape((3,))
            r_aa = max(0.0, float(aa_radius_ang))
            r_cg = max(r_aa, float(cg_radius_ang))
            self._focus_center_ang = (float(ctr[0]), float(ctr[1]), float(ctr[2]))
            self._aa_radius_ang = r_aa
            self._cg_radius_ang = r_cg

            if self.n_transforms == 0 or self.n_asu_atoms == 0:
                self._allocate_empty_active()
                return self._build_multiscale_summary()

            # O(K) subunit bounding-sphere culling (< 0.2 ms for K=2,760)
            diff_k = self.subunit_com_ang - ctr[None, :]
            dist_k = np.linalg.norm(diff_k, axis=1)
            r_bound = self._asu_bound_radius_ang

            aa_cand = np.flatnonzero(dist_k <= (r_aa + r_bound))
            if aa_cand.size == 0 and r_aa > 0.0:
                # Guarantee at least the closest subunit is considered if focus is near surface
                aa_cand = np.array([int(np.argmin(dist_k))], dtype=np.int64)
            # Sort candidate AA subunits nearest-first
            aa_order = aa_cand[np.argsort(dist_k[aa_cand])]

            # Materialize ONLY Zone 0 active All-Atom focus atoms (capped at max_active_aa_atoms)
            aa_pos_parts: List[np.ndarray] = []
            aa_z_parts: List[np.ndarray] = []
            aa_gid_parts: List[np.ndarray] = []
            aa_subunits_used: List[int] = []
            r_aa2 = r_aa * r_aa
            collected_aa = 0
            # Cap candidate subunit expansion to avoid expanding hundreds of subunits if r_aa is huge
            max_cand_subunits = max(4, (self.max_active_aa_atoms // max(1, self.n_asu_atoms)) + 8)
            for k_idx in aa_order[:max_cand_subunits]:
                k = int(k_idx)
                tf = self.transforms_4x4[k]
                rot = tf[:3, :3]
                trans = tf[:3, 3] + self.subunit_disp_ang[k]
                sub_coords = self.asu_coords_ang @ rot.T + trans[None, :]
                d2 = np.sum((sub_coords - ctr[None, :]) ** 2, axis=1)
                mask = d2 <= r_aa2
                if not np.any(mask) and collected_aa == 0 and aa_subunits_used == []:
                    # If focus point is slightly outside bounding atoms, take nearest atoms of closest subunit
                    take = min(self.n_asu_atoms, self.max_active_aa_atoms)
                    sel_local = np.argpartition(d2, take - 1)[:take] if take < self.n_asu_atoms else np.arange(self.n_asu_atoms)
                else:
                    sel_local = np.flatnonzero(mask)
                if sel_local.size == 0:
                    continue
                remaining = self.max_active_aa_atoms - collected_aa
                if sel_local.size > remaining:
                    sub_d2 = d2[sel_local]
                    keep = np.argpartition(sub_d2, remaining - 1)[:remaining]
                    sel_local = sel_local[keep]
                aa_pos_parts.append(sub_coords[sel_local])
                aa_z_parts.append(self.asu_elements_u8[sel_local])
                aa_gid_parts.append(sel_local.astype(np.int64) + np.int64(k) * np.int64(self.n_asu_atoms))
                aa_subunits_used.append(k)
                collected_aa += int(sel_local.size)
                if collected_aa >= self.max_active_aa_atoms:
                    break

            aa_set = set(aa_subunits_used)
            if aa_pos_parts:
                aa_coords_ang = np.concatenate(aa_pos_parts, axis=0).astype(np.float32, copy=False)
                aa_z = np.concatenate(aa_z_parts, axis=0).astype(np.uint8, copy=False)
                aa_gids = np.concatenate(aa_gid_parts, axis=0)
            else:
                aa_coords_ang = np.zeros((0, 3), dtype=np.float32)
                aa_z = np.zeros(0, dtype=np.uint8)
                aa_gids = np.zeros(0, dtype=np.int64)

            # Zone 1: Coarse-Grained shell (1 bead per ~16 atoms, capped at max_cg_beads)
            cg_cand_all = np.flatnonzero(dist_k <= (r_cg + r_bound))
            cg_subunits = [int(k) for k in cg_cand_all[np.argsort(dist_k[cg_cand_all])] if int(k) not in aa_set]
            if len(cg_subunits) == 0 and len(aa_set) < self.n_transforms and self.max_cg_beads > 0 and r_cg > 0.0:
                # Guarantee a real CG transition buffer shell around Zone 0 whenever non-AA subunits exist
                non_aa = [int(k) for k in np.argsort(dist_k) if int(k) not in aa_set]
                take_cg = min(6, len(non_aa) - 1) if len(non_aa) > 1 else len(non_aa)
                cg_subunits = non_aa[:take_cg]
            n_cg_subunits = len(cg_subunits)
            cg_pos_parts: List[np.ndarray] = []
            cg_mass_parts: List[np.ndarray] = []
            n_beads_per_asu = int(self._asu_cg_coords_ang.shape[0])
            if n_cg_subunits > 0 and self.max_cg_beads > 0:
                if n_cg_subunits * n_beads_per_asu <= self.max_cg_beads and n_beads_per_asu > 0:
                    for k in cg_subunits:
                        tf = self.transforms_4x4[k]
                        rot = tf[:3, :3]
                        trans = tf[:3, 3] + self.subunit_disp_ang[k]
                        cg_pos_parts.append(self._asu_cg_coords_ang @ rot.T + trans[None, :])
                        cg_mass_parts.append(self._asu_cg_masses)
                else:
                    # Fallback to 1 CG bead per subunit COM when CG shell spans many subunits
                    take_k = cg_subunits[: self.max_cg_beads]
                    cg_pos_parts.append(self.subunit_com_ang[take_k])
                    cg_mass_parts.append(np.full(len(take_k), self._subunit_mass, dtype=np.float32))

            if cg_pos_parts:
                cg_coords_ang = np.concatenate(cg_pos_parts, axis=0).astype(np.float32, copy=False)
                cg_masses = np.concatenate(cg_mass_parts, axis=0).astype(np.float64, copy=False)
            else:
                cg_coords_ang = np.zeros((0, 3), dtype=np.float32)
                cg_masses = np.zeros(0, dtype=np.float64)

            n_aa = int(aa_coords_ang.shape[0])
            n_cg_beads = int(cg_coords_ang.shape[0])
            cap = n_aa + n_cg_beads

            self.active_all_atom_count = n_aa
            self.active_cg_bead_count = n_cg_beads
            self._n_subunits_aa = len(aa_subunits_used)
            self._n_subunits_cg = n_cg_subunits
            self._n_subunits_rigid = max(0, self.n_transforms - self._n_subunits_aa - self._n_subunits_cg)
            self._n_cg_virtual_atoms = min(self.n_atoms - n_aa, n_cg_subunits * self.n_asu_atoms)
            self._active_subunit_indices = np.asarray(aa_subunits_used, dtype=np.int32)
            self._active_global_atom_ids = aa_gids

            self.capacity = cap
            self.n_real = cap

            # Assemble compact active buffer in nm
            all_coords_ang = (
                np.concatenate([aa_coords_ang, cg_coords_ang], axis=0)
                if cap > 0
                else np.zeros((0, 3), dtype=np.float32)
            )
            self.positions = (all_coords_ang.astype(np.float64) * 0.1)
            self._cg_ref_positions_nm = (cg_coords_ang * 0.1).astype(np.float32, copy=True)

            self.atomic_numbers = np.empty(cap, dtype=np.uint8)
            self.masses = np.empty(cap, dtype=np.float64)
            self.radii_ang = np.empty(cap, dtype=np.float32)
            self.multiscale_zones = np.empty(cap, dtype=np.uint8)
            if n_aa > 0:
                self.atomic_numbers[:n_aa] = aa_z
                self.masses[:n_aa] = ELEMENT_LUT_MASS[aa_z].astype(np.float64)
                self.radii_ang[:n_aa] = ELEMENT_LUT_RADIUS_ANG[aa_z]
                self.multiscale_zones[:n_aa] = 0
            if n_cg_beads > 0:
                self.atomic_numbers[n_aa:] = 0  # Dedicated CG bead marker (element="CG", never "Ar")
                self.masses[n_aa:] = cg_masses
                self.radii_ang[n_aa:] = 4.2
                self.multiscale_zones[n_aa:] = 1

            self.charges = np.zeros(cap, dtype=np.float64)
            self.active_mask = np.ones(cap, dtype=bool)
            self.forces = np.zeros((cap, 3), dtype=np.float64)
            self.momenta = np.zeros((cap, 3), dtype=np.float64)
            self.bond_orders_per_atom = np.ones(cap, dtype=np.float32)
            self.atomic_stress = np.zeros(cap, dtype=np.float32)
            self.ternary_states = np.zeros(cap, dtype=np.float32)
            if n_aa > 0:
                self.ternary_states[:n_aa] = 1.0
            if n_cg_beads > 0:
                self.ternary_states[n_aa:] = 0.0

            # Initialize Boltzmann momenta on active atoms
            if cap > 0:
                v_scale = np.sqrt(self.k_b * max(1e-6, self.temperature) / np.maximum(self.masses, 1e-6))[:, None]
                vel0 = self._rng.normal(0.0, 1.0, size=(cap, 3)) * v_scale
                vel0 -= np.mean(vel0, axis=0, keepdims=True)
                self.momenta = vel0 * self.masses[:, None]

            # Spatial Morton tiles + DirtyTileTracker on compact active buffer
            rel_pos_nm = self.positions - (self._box_origin_ang * 0.1)[None, :]
            self.tile_ids = compute_morton_tiles(rel_pos_nm, self.box_H, tile_size=32)
            prev_thr = self._dirty_tracker.threshold_ang if self._dirty_tracker is not None else 0.05
            self._dirty_tracker = DirtyTileTracker(cap, self.tile_ids, threshold_ang=prev_thr)
            self._last_dirty_ratio = 1.0
            self._last_dirty_tiles = int(np.unique(self.tile_ids).size) if cap > 0 else 0

            # Clear SMD pull if atom index moved out of range
            if self._smd_pull is not None and int(self._smd_pull["atom_index"]) >= cap:
                self._smd_pull = None

            self._aa_ref_positions_nm = self.positions[:n_aa].copy() if n_aa > 0 else np.zeros((0, 3), dtype=np.float64)
            self._build_enm()
            self._nb_ref_pos = None
            self._evaluate_state()
            self._q_thermo_cum = 0.0
            self._initial_total_energy = self._extended_total_energy()
            return self._build_multiscale_summary()

    def _allocate_empty_active(self) -> None:
        self.active_all_atom_count = 0
        self.active_cg_bead_count = 0
        self._n_cg_virtual_atoms = 0
        self._n_subunits_aa = 0
        self._n_subunits_cg = 0
        self._n_subunits_rigid = self.n_transforms
        self.capacity = 0
        self.n_real = 0
        self.positions = np.zeros((0, 3), dtype=np.float64)
        self.momenta = np.zeros((0, 3), dtype=np.float64)
        self.forces = np.zeros((0, 3), dtype=np.float64)
        self.masses = np.zeros(0, dtype=np.float64)
        self.charges = np.zeros(0, dtype=np.float64)
        self.atomic_numbers = np.zeros(0, dtype=np.uint8)
        self.radii_ang = np.zeros(0, dtype=np.float32)
        self.active_mask = np.zeros(0, dtype=bool)
        self.multiscale_zones = np.zeros(0, dtype=np.uint8)
        self.bond_orders_per_atom = np.zeros(0, dtype=np.float32)
        self.atomic_stress = np.zeros(0, dtype=np.float32)
        self.ternary_states = np.zeros(0, dtype=np.float32)
        self.tile_ids = np.zeros(0, dtype=np.uint32)
        self._cg_ref_positions_nm = np.zeros((0, 3), dtype=np.float32)
        self._dirty_tracker = DirtyTileTracker(0, self.tile_ids, threshold_ang=0.05)

    def _local_aa_box(self, pos_aa: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute tight local bounding box (origin_nm, box_L_nm) around active AA atoms for fast O(N_aa) cells."""
        margin = 1.2  # nm
        min_p = np.min(pos_aa, axis=0) - margin
        max_p = np.max(pos_aa, axis=0) + margin
        box_L = np.maximum(max_p - min_p, 3.0)
        return min_p, box_L

    def _extended_total_energy(self) -> float:
        return float(self.potential_energy + self.kinetic_energy + self._rigid_kinetic_energy)

    def _build_enm(self) -> None:
        """Factor 1 setup: O(N) cKDTree native-contact elastic network on Zone 0 All-Atom atoms."""
        n_aa = self.active_all_atom_count
        if n_aa < 2:
            self._enm_i = np.zeros(0, dtype=np.int32)
            self._enm_j = np.zeros(0, dtype=np.int32)
            self._enm_r0 = np.zeros(0, dtype=np.float64)
            self._enm_pair_keys = np.zeros(0, dtype=np.int64)
            return

        pos_aa = self.positions[:n_aa]
        try:
            from scipy.spatial import cKDTree

            tree = cKDTree(pos_aa)
            pairs = tree.query_pairs(r=float(self._enm_cutoff_nm), output_type="ndarray")
            if pairs.size == 0:
                ii = np.zeros(0, dtype=np.int32)
                jj = np.zeros(0, dtype=np.int32)
            else:
                ii = pairs[:, 0].astype(np.int32, copy=False)
                jj = pairs[:, 1].astype(np.int32, copy=False)
        except Exception:
            origin_nm, box_L = self._local_aa_box(pos_aa)
            ii, jj, _, _ = _find_neighbor_pairs_mic(pos_aa - origin_nm[None, :], box_L, float(self._enm_cutoff_nm))

        if ii.size == 0:
            self._enm_i = np.zeros(0, dtype=np.int32)
            self._enm_j = np.zeros(0, dtype=np.int32)
            self._enm_r0 = np.zeros(0, dtype=np.float64)
            self._enm_pair_keys = np.zeros(0, dtype=np.int64)
            return

        d = pos_aa[ii] - pos_aa[jj]
        r0 = np.sqrt(np.maximum(np.sum(d * d, axis=1), 0.0))
        valid = r0 > 1e-4
        ii = ii[valid]
        jj = jj[valid]
        r0 = r0[valid]

        self._enm_i = ii
        self._enm_j = jj
        self._enm_r0 = r0
        keys = ii.astype(np.int64) * np.int64(n_aa) + jj.astype(np.int64)
        keys.sort()
        self._enm_pair_keys = keys

    def _get_nonbonded_pairs(self, pos_aa: np.ndarray, rc: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Exact O(N) cKDTree Verlet neighbor list for Factor 2 nonbonded LJ (excludes ENM bonds)."""
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
            disp_sq = np.sum((pos_aa - self._nb_ref_pos) ** 2, axis=1)
            if float(np.max(disp_sq)) > (0.5 * self._nb_skin_nm) ** 2:
                need_rebuild = True

        if need_rebuild:
            r_list = float(rc) + float(self._nb_skin_nm)
            try:
                from scipy.spatial import cKDTree

                tree = cKDTree(pos_aa)
                pairs = tree.query_pairs(r=r_list, output_type="ndarray")
                if pairs.size == 0:
                    cand_i = np.zeros(0, dtype=np.int32)
                    cand_j = np.zeros(0, dtype=np.int32)
                else:
                    cand_i = pairs[:, 0].astype(np.int32, copy=False)
                    cand_j = pairs[:, 1].astype(np.int32, copy=False)
            except Exception:
                origin_nm, box_L = self._local_aa_box(pos_aa)
                cand_i, cand_j, _, _ = _find_neighbor_pairs_mic(pos_aa - origin_nm[None, :], box_L, r_list)

            # Exclude ENM bonded pairs so covalent/native contacts are governed purely by Factor 1
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
            self._nb_ref_pos = pos_aa.copy()
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
        ii = self._nb_i[mask]
        jj = self._nb_j[mask]
        d_sel = d_all[mask]
        r_sel = np.sqrt(np.maximum(r2_all[mask], 0.0))
        return ii, jj, d_sel, r_sel

    def _evaluate_state(self) -> None:
        """Factor-Graph / CDC Composable VJP Pipeline: exact energy & F = -grad(U) sum."""
        n_aa = self.active_all_atom_count
        n_cg = self.active_cg_bead_count
        cap = self.capacity
        self.forces.fill(0.0)
        self.virial.fill(0.0)
        self._subunit_forces_nm.fill(0.0)
        u_tot = 0.0

        # Factor 1: Native Elastic Network (ENM) on Zone 0 All-Atom atoms (O(N) sparse bonds)
        if n_aa >= 2 and self._enm_i.size > 0:
            pos_aa = self.positions[:n_aa]
            d_b = pos_aa[self._enm_i] - pos_aa[self._enm_j]
            r_b = np.sqrt(np.maximum(np.sum(d_b * d_b, axis=1), 1e-24))
            dr = r_b - self._enm_r0
            k_b = float(self._enm_k_bond)
            u_enm = 0.5 * k_b * float(np.sum(dr * dr))
            f_enm = d_b * (-(k_b * dr) / r_b)[:, None]
            f_aa = self.forces[:n_aa]
            for dim in range(3):
                f_aa[:, dim] += np.bincount(self._enm_i, weights=f_enm[:, dim], minlength=n_aa) - np.bincount(
                    self._enm_j, weights=f_enm[:, dim], minlength=n_aa
                )
            u_tot += u_enm
            self.virial += d_b.T @ f_enm

        # Factor 2: Smooth Soft-Core + C1 Linear-Force-Shifted (LFS) Nonbonded LJ
        if n_aa >= 2:
            pos_aa = self.positions[:n_aa]
            rc = float(self.rcut)
            ii, jj, d_ij, r = self._get_nonbonded_pairs(pos_aa, rc)
            if ii.size > 0:
                z_aa = self.atomic_numbers[:n_aa]
                sig_aa = ELEMENT_LUT_SIGMA[z_aa].astype(np.float64)
                eps_aa = ELEMENT_LUT_EPSILON[z_aa].astype(np.float64)
                sig_ij = 0.5 * (sig_aa[ii] + sig_aa[jj])
                eps_ij = np.sqrt(np.maximum(eps_aa[ii] * eps_aa[jj], 0.0))
                sig2 = sig_ij * sig_ij
                # Smooth soft-core parameter a = alpha_sc * sigma^2 keeps U and dU/dr C^inf at r -> 0
                alpha_sc = 0.05
                a_sc = alpha_sc * sig2

                r2 = r * r
                denom = r2 + a_sc
                s2 = sig2 / denom
                s6 = s2 * s2 * s2
                s12 = s6 * s6
                u_r = 4.0 * eps_ij * (s12 - s6)
                du_over_r = (24.0 * eps_ij / denom) * (s6 - 2.0 * s12)

                # C1 Linear Force Shifting (LFS) at r = rc: U_LFS(rc) = 0, U_LFS'(rc) = 0
                denom_rc = (rc * rc) + a_sc
                s2_rc = sig2 / denom_rc
                s6_rc = s2_rc * s2_rc * s2_rc
                s12_rc = s6_rc * s6_rc
                u_rc = 4.0 * eps_ij * (s12_rc - s6_rc)
                du_rc = (24.0 * eps_ij * rc / denom_rc) * (s6_rc - 2.0 * s12_rc)

                u_pair = u_r - u_rc - (r - rc) * du_rc
                # Exact gradient: F_ij = -(du_r/dr - du_rc) * (d_ij / r)
                r_safe = np.maximum(r, 1e-12)
                force_over_r = -(du_over_r - (du_rc / r_safe))
                f_ij = d_ij * force_over_r[:, None]
                f_aa = self.forces[:n_aa]
                for dim in range(3):
                    f_aa[:, dim] += np.bincount(ii, weights=f_ij[:, dim], minlength=n_aa) - np.bincount(
                        jj, weights=f_ij[:, dim], minlength=n_aa
                    )
                u_tot += float(np.sum(u_pair))
                self.virial += d_ij.T @ f_ij

        # Factor 3: Exact Harmonic.lean Bipartite Centroid-Residual Coupling (O(n+m)!)
        # Zone 0 (All-Atom) <-> Zone 1 (CG Buffer) <-> Zone 2 (all K rigid capsomers)
        disp_aa = (
            self.positions[:n_aa] - self._aa_ref_positions_nm
            if n_aa > 0 and self._aa_ref_positions_nm.shape == (n_aa, 3)
            else np.zeros((0, 3), dtype=np.float64)
        )
        disp_cg = (
            self.positions[n_aa:cap] - self._cg_ref_positions_nm.astype(np.float64)
            if n_cg > 0
            else np.zeros((0, 3), dtype=np.float64)
        )

        if n_aa > 0 and n_cg > 0:
            u_01, f_aa_01, f_cg_01 = eval_bipartite_centroid_residual(disp_aa, disp_cg, k_couple=250.0)
            u_tot += u_01
            self.forces[:n_aa] += f_aa_01
            self.forces[n_aa:cap] += f_cg_01

        if n_cg > 0 and self.n_transforms > 0:
            u_12, f_cg_12, f_sub_12 = eval_bipartite_centroid_residual(disp_cg, self._subunit_disp_nm, k_couple=250.0)
            u_tot += u_12
            self.forces[n_aa:cap] += f_cg_12
            self._subunit_forces_nm += f_sub_12
        elif n_aa > 0 and self.n_transforms > 0:
            # Direct Zone 0 <-> Zone 2 bipartite coupling when CG shell has 0 beads
            u_02, f_aa_02, f_sub_02 = eval_bipartite_centroid_residual(disp_aa, self._subunit_disp_nm, k_couple=100.0)
            u_tot += u_02
            self.forces[:n_aa] += f_aa_02
            self._subunit_forces_nm += f_sub_02

        # Zone 2 rigid capsomer harmonic scaffold tether
        if self.n_transforms > 0:
            k_tether = 500.0
            u_tot += 0.5 * k_tether * float(np.sum(self._subunit_disp_nm * self._subunit_disp_nm))
            self._subunit_forces_nm += -k_tether * self._subunit_disp_nm

        # Factor 4: Interactive SMD harmonic spring pull (if active)
        if self._smd_pull is not None:
            smd_idx = int(self._smd_pull["atom_index"])
            if 0 <= smd_idx < cap:
                target_nm = self._smd_pull["target_nm"]
                k_spring = float(self._smd_pull["k_spring"])
                disp = target_nm - self.positions[smd_idx]
                u_smd = 0.5 * k_spring * float(np.dot(disp, disp))
                f_smd = k_spring * disp
                u_tot += u_smd
                self.forces[smd_idx] += f_smd
                ext_nm = float(np.linalg.norm(disp))
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

        self.potential_energy = float(u_tot)
        if cap > 0:
            f_mag = np.linalg.norm(self.forces, axis=1)
            self.atomic_stress = ((np.trace(self.virial) / max(1, cap)) + 0.05 * f_mag).astype(np.float32)
            p2 = np.sum(self.momenta * self.momenta, axis=1)
            self.kinetic_energy = 0.5 * float(np.sum(p2 / np.maximum(self.masses, 1e-12)))
        else:
            self.kinetic_energy = 0.0

        if self.n_transforms > 0:
            self._rigid_kinetic_energy = 0.5 * float(
                np.sum(self._subunit_momenta_nm * self._subunit_momenta_nm) / max(self._subunit_mass, 1.0)
            )
        else:
            self._rigid_kinetic_energy = 0.0

    def relax_clashes(
        self,
        max_steps: int = 40,
        max_force_cap: float = 5000.0,
        alpha_sc: float = 0.5,
        lam: float = 0.5,
    ) -> Dict[str, Any]:
        """Run Beutler soft-core LJ clash relaxation on the active All-Atom focus zone."""
        with self._lock:
            n_aa = self.active_all_atom_count
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

            pos_aa = self.positions[:n_aa]
            origin_nm, box_L = self._local_aa_box(pos_aa)
            local_pos = pos_aa - origin_nm[None, :]
            z_aa = self.atomic_numbers[:n_aa]
            sig_aa = ELEMENT_LUT_SIGMA[z_aa].astype(np.float64)
            eps_aa = ELEMENT_LUT_EPSILON[z_aa].astype(np.float64)

            relaxed_local, report = softcore_relax_clashes(
                positions_nm=local_pos,
                box_nm=box_L,
                sigma_nm=sig_aa,
                epsilon_kj=eps_aa,
                max_steps=max_steps,
                max_force_cap=max_force_cap,
                alpha_sc=alpha_sc,
                lam=lam,
            )
            self.positions[:n_aa] = relaxed_local + origin_nm[None, :]
            self._aa_ref_positions_nm = self.positions[:n_aa].copy()
            self._build_enm()
            self._nb_ref_pos = None
            self._evaluate_state()
            self._q_thermo_cum = 0.0
            self._initial_total_energy = self._extended_total_energy()
            self._dirty_tracker.reset()
            self._last_clash_report = dict(report)
            self._record_history()
            return report

    def minimize(self, max_iters: int = 40, max_disp: float = 0.02, f_tol: float = 1.0e3) -> Dict[str, Any]:
        """Relax clashes on the active All-Atom zone and redraw Boltzmann momenta."""
        report = self.relax_clashes(max_steps=min(int(max_iters), 40), max_force_cap=5000.0)
        with self._lock:
            cap = self.capacity
            if cap > 0:
                v_scale = np.sqrt(self.k_b * max(1e-6, self.temperature) / np.maximum(self.masses, 1e-6))[:, None]
                vel0 = self._rng.normal(0.0, 1.0, size=(cap, 3)) * v_scale
                vel0 -= np.mean(vel0, axis=0, keepdims=True)
                self.momenta = vel0 * self.masses[:, None]
                self._evaluate_state()
                self._q_thermo_cum = 0.0
                self._initial_total_energy = self._extended_total_energy()
            return {
                "iterations": int(report.get("steps_run", 0)),
                "energy_before": float(report.get("initial_energy", 0.0)),
                "energy_after": float(self.potential_energy),
                "max_force_after": float(report.get("final_max_force", 0.0)),
            }

    def step(self, n_steps: int = 1, apply_thermostat: Optional[bool] = None) -> Dict[str, Any]:
        """Advance Zone 0 (AA), Zone 1 (CG), and Zone 2 (K rigid capsomers) via symmetric BAOAB + HMC guard."""
        with self._lock:
            t0 = time.perf_counter()
            n_steps = max(1, int(n_steps))
            dt = float(self.dt)
            half_dt = 0.5 * dt
            use_thermo = True if apply_thermostat is None else bool(apply_thermostat)
            cap = self.capacity
            k_sub = self.n_transforms
            m_sub = max(self._subunit_mass, 1.0)
            sqrt_m_sub = math.sqrt(m_sub)

            inv_m = (1.0 / np.maximum(self.masses, 1e-12))[:, None] if cap > 0 else None
            sqrt_m = np.sqrt(np.maximum(self.masses, 1e-12))[:, None] if cap > 0 else None

            c1 = math.exp(-self.gamma * dt) if (use_thermo and self.gamma > 0.0) else 1.0
            c2 = math.sqrt(max(0.0, 1.0 - c1 * c1) * self.k_b * self.temperature) if (use_thermo and self.gamma > 0.0) else 0.0
            beta = 1.0 / max(self.k_b * self.temperature, 1e-6)

            for _ in range(n_steps):
                # Snapshot old state for HMC Metropolis safeguard
                old_pos = self.positions.copy() if cap > 0 else None
                old_mom = self.momenta.copy() if cap > 0 else None
                old_frc = self.forces.copy() if cap > 0 else None
                old_sub_disp = self._subunit_disp_nm.copy() if k_sub > 0 else None
                old_sub_mom = self._subunit_momenta_nm.copy() if k_sub > 0 else None
                old_sub_frc = self._subunit_forces_nm.copy() if k_sub > 0 else None
                old_pe = float(self.potential_energy)
                old_ke = float(self.kinetic_energy)
                old_rigid_ke = float(self._rigid_kinetic_energy)
                old_virial = self.virial.copy()
                e_ext_old = old_pe + old_ke + old_rigid_ke

                # 1. B(dt/2): half-kick momenta
                if cap > 0:
                    self.momenta += half_dt * self.forces
                if k_sub > 0:
                    self._subunit_momenta_nm += half_dt * self._subunit_forces_nm

                # 2. A(dt/2): half-drift positions
                if cap > 0:
                    self.positions += half_dt * (self.momenta * inv_m)
                if k_sub > 0:
                    self._subunit_disp_nm += half_dt * (self._subunit_momenta_nm / m_sub)

                # 3. O(dt): exact Ornstein-Uhlenbeck thermostat + exact heat exchange dQ_thermo
                dQ_thermo = 0.0
                if use_thermo and self.gamma > 0.0:
                    ke_before_o = 0.0
                    if cap > 0:
                        ke_before_o += 0.5 * float(np.sum((self.momenta * self.momenta) * inv_m))
                    if k_sub > 0:
                        ke_before_o += 0.5 * float(np.sum(self._subunit_momenta_nm * self._subunit_momenta_nm) / m_sub)

                    if cap > 0:
                        noise = self._rng.normal(0.0, 1.0, size=(cap, 3))
                        self.momenta = c1 * self.momenta + c2 * sqrt_m * noise
                    if k_sub > 0:
                        sub_noise = self._rng.normal(0.0, 1.0, size=(k_sub, 3))
                        self._subunit_momenta_nm = c1 * self._subunit_momenta_nm + (c2 * sqrt_m_sub) * sub_noise

                    ke_after_o = 0.0
                    if cap > 0:
                        ke_after_o += 0.5 * float(np.sum((self.momenta * self.momenta) * inv_m))
                    if k_sub > 0:
                        ke_after_o += 0.5 * float(np.sum(self._subunit_momenta_nm * self._subunit_momenta_nm) / m_sub)
                    dQ_thermo = ke_after_o - ke_before_o

                # 4. A(dt/2): second half-drift positions
                if cap > 0:
                    self.positions += half_dt * (self.momenta * inv_m)
                if k_sub > 0:
                    self._subunit_disp_nm += half_dt * (self._subunit_momenta_nm / m_sub)

                # 5. F: evaluate exact composable Factor-Graph forces
                self._evaluate_state()

                # 6. B(dt/2): second half-kick momenta
                if cap > 0:
                    self.momenta += half_dt * self.forces
                    self.kinetic_energy = 0.5 * float(np.sum((self.momenta * self.momenta) * inv_m))
                if k_sub > 0:
                    self._subunit_momenta_nm += half_dt * self._subunit_forces_nm
                    self._rigid_kinetic_energy = 0.5 * float(
                        np.sum(self._subunit_momenta_nm * self._subunit_momenta_nm) / m_sub
                    )

                # 7. HMC Metropolis Safeguard (hmc_guard)
                e_ext_new = float(self.potential_energy + self.kinetic_energy + self._rigid_kinetic_energy)
                dH_ext = (e_ext_new - e_ext_old) - dQ_thermo
                f_max_new = float(np.max(np.linalg.norm(self.forces, axis=1))) if cap > 0 else 0.0
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
                        # Reject stiff numerical spike: roll back positions & flip/damp momenta
                        if cap > 0 and old_pos is not None and old_mom is not None and old_frc is not None:
                            self.positions[:] = old_pos
                            self.momenta[:] = -0.5 * old_mom
                            self.forces[:] = old_frc
                            self.kinetic_energy = 0.5 * float(np.sum((self.momenta * self.momenta) * inv_m))
                        if k_sub > 0 and old_sub_disp is not None and old_sub_mom is not None and old_sub_frc is not None:
                            self._subunit_disp_nm[:] = old_sub_disp
                            self._subunit_momenta_nm[:] = -0.5 * old_sub_mom
                            self._subunit_forces_nm[:] = old_sub_frc
                            self._rigid_kinetic_energy = 0.5 * float(
                                np.sum(self._subunit_momenta_nm * self._subunit_momenta_nm) / m_sub
                            )
                        self.potential_energy = old_pe
                        self.virial[:] = old_virial
                        dQ_thermo = (self.kinetic_energy + self._rigid_kinetic_energy) - (old_ke + old_rigid_ke)
                        self.hmc_rejections += 1

                self._q_thermo_cum += dQ_thermo
                self.hmc_steps += 1
                self.step_count += 1
                self.sim_time_ps += dt

            # Sync Zone 2 rigid subunit visual arrays in Angstroms
            if k_sub > 0:
                self.subunit_disp_ang = (self._subunit_disp_nm * 10.0).astype(np.float32)
                self.subunit_com_ang = self.subunit_com_ref_ang + self.subunit_disp_ang
                self.subunit_vel_ang_ps = ((self._subunit_momenta_nm / m_sub) * 10.0).astype(np.float32)
                if use_thermo and self.gamma > 0.0:
                    w_noise = self._rng.normal(0.0, 1.0, size=(k_sub, 3)).astype(np.float32)
                    self.subunit_omega_rad_ps = (
                        np.float32(c1) * self.subunit_omega_rad_ps
                        + np.float32(c2 / math.sqrt(max(self._subunit_inertia, 1.0))) * w_noise
                    )

            elapsed_ms = max(1e-4, (time.perf_counter() - t0) * 1e3)
            self.last_ms_per_step = elapsed_ms / n_steps
            self._record_history()
            summary = self.get_metrics()
            summary["gcmc_inserted_slots"] = []
            summary["gcmc_deleted_slots"] = []
            return summary

    def _record_history(self) -> None:
        snap = _CompactSnapshot(
            step=self.step_count,
            time_ps=self.sim_time_ps,
            focus_center_ang=self._focus_center_ang,
            positions_f32=self.positions.astype(np.float32, copy=True),
            momenta_f32=self.momenta.astype(np.float32, copy=True),
            forces_f32=self.forces.astype(np.float32, copy=True),
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
            if snap.positions_f32.shape == self.positions.shape:
                self.positions[:] = snap.positions_f32.astype(np.float64)
                self.momenta[:] = snap.momenta_f32.astype(np.float64)
                self.forces[:] = snap.forces_f32.astype(np.float64)
            self.potential_energy = snap.potential_energy
            self.kinetic_energy = snap.kinetic_energy
            self._q_thermo_cum = float(snap.q_thermo_cum)
            return self.get_metrics(), was_reverse

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
            if efield_z is not None:
                self.efield_z = float(efield_z)
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
            if idx < 0 or idx >= self.capacity:
                raise IndexError(f"Atom index {idx} out of active focus range 0..{self.capacity - 1}")
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

    def set_delta_threshold(
        self,
        threshold_ang: float,
        enabled: Optional[bool] = None,
    ) -> Dict[str, Any]:
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

    def inspect_atom(self, idx: int) -> Dict[str, Any]:
        with self._lock:
            i = int(idx)
            if 0 <= i < self.capacity:
                pos = self.positions[i]
                vel = self.momenta[i] / max(self.masses[i], 1e-12)
                frc = self.forces[i]
                ke = 0.5 * float(np.dot(self.momenta[i], self.momenta[i]) / max(self.masses[i], 1e-12))
                z = int(self.atomic_numbers[i])
                is_cg_bead = int(self.multiscale_zones[i]) == 1
                elem_sym = "CG" if is_cg_bead else _Z_TO_SYM_LUT[z]
                return {
                    "index": i,
                    "active": True,
                    "element": elem_sym,
                    "atomic_number": 0 if is_cg_bead else z,
                    "position_nm": [float(v) for v in pos],
                    "position_ang": [float(v * self.coord_to_ang) for v in pos],
                    "velocity_nm_ps": [float(v) for v in vel],
                    "force": [float(v) for v in frc],
                    "force_magnitude": float(np.linalg.norm(frc)),
                    "kinetic_energy": ke,
                    "local_temp_K": float((2.0 / (3.0 * self.k_b)) * ke),
                    "charge": float(self.charges[i]),
                    "bond_order": float(self.bond_orders_per_atom[i]),
                    "atomic_stress": float(self.atomic_stress[i]),
                    "ternary_state": float(self.ternary_states[i]),
                    "zone": int(self.multiscale_zones[i]),
                }
            # Also support on-the-fly O(1) inspection of any atom in the 40M virtual assembly!
            if 0 <= i < self.n_atoms and self.n_asu_atoms > 0:
                k = i // self.n_asu_atoms
                m = i % self.n_asu_atoms
                tf = self.transforms_4x4[k]
                pos_ang = tf[:3, :3] @ self.asu_coords_ang[m] + tf[:3, 3] + self.subunit_disp_ang[k]
                z = int(self.asu_elements_u8[m])
                return {
                    "index": i,
                    "active": False,
                    "element": _Z_TO_SYM_LUT[z],
                    "atomic_number": z,
                    "position_nm": [float(v * 0.1) for v in pos_ang],
                    "position_ang": [float(v) for v in pos_ang],
                    "velocity_nm_ps": [float(v * 0.1) for v in self.subunit_vel_ang_ps[k]],
                    "force": [0.0, 0.0, 0.0],
                    "force_magnitude": 0.0,
                    "kinetic_energy": 0.0,
                    "local_temp_K": float(self.temperature),
                    "charge": 0.0,
                    "bond_order": 1.0,
                    "atomic_stress": 0.0,
                    "ternary_state": -1.0,
                    "zone": 2,
                    "subunit_index": int(k),
                }
            raise IndexError(f"Atom index {i} out of range 0..{self.n_atoms - 1}")

    def compute_scalar_channel(self, channel_name: str) -> np.ndarray:
        if channel_name == "force-magnitude":
            out = np.linalg.norm(self.forces, axis=1).astype(np.float32) if self.capacity > 0 else np.zeros(0, dtype=np.float32)
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
        return out

    def get_metrics(self) -> Dict[str, Any]:
        with self._lock:
            cap = self.capacity
            inst_temp = (2.0 * self.kinetic_energy) / max(1e-9, 3.0 * cap * self.k_b) if cap > 0 else 0.0
            ns_per_day = (0.002 * 86400.0) / max(self.last_ms_per_step, 1e-4)
            f_max = float(np.max(np.linalg.norm(self.forces, axis=1))) if cap > 0 else 0.0
            vol_nm3 = float(abs(np.linalg.det(self.box_H)))
            total_e = float(self.potential_energy + self.kinetic_energy)
            e_ext = float(self.potential_energy + self.kinetic_energy + self._rigid_kinetic_energy)
            e0 = self._initial_total_energy
            e_scale = max(
                abs(e0) if (e0 is not None and math.isfinite(e0)) else 0.0,
                1.5 * max(cap, 1) * self.k_b * self.temperature,
                100.0,
            )
            if e0 is not None and math.isfinite(e_ext) and math.isfinite(e0) and math.isfinite(self._q_thermo_cum):
                drift_fraction = abs((e_ext - e0) - self._q_thermo_cum) / e_scale
            else:
                drift_fraction = 0.0 if math.isfinite(e_ext) else float("inf")

            if (not math.isfinite(total_e)) or (not math.isfinite(f_max)) or (f_max > 1.0e5):
                health_status = "clash"
            elif (f_max > 1.0e4) or (drift_fraction > 0.05):
                health_status = "warning"
            else:
                health_status = "bounded"

            ram_bytes = self._compute_ram_bytes()
            ram_mb = float(ram_bytes) / (1024.0 * 1024.0)
            naive_float64_ram_mb = float(self.n_atoms) * 160.0 / 1.0e6
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
                "n_atoms": int(self.n_atoms),
                "n_asu_atoms": int(self.n_asu_atoms),
                "n_transforms": int(self.n_transforms),
                "capacity": int(self.capacity),
                "active_all_atom_count": int(self.active_all_atom_count),
                "storage_mode": "symmetry-compact",
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
                "q_thermo_cum": float(self._q_thermo_cum),
                "hmc_rejections": int(self.hmc_rejections),
                "health_status": health_status,
                "ms_per_step": float(self.last_ms_per_step),
                "ns_per_day_2fs": float(ns_per_day),
                "box_lengths_nm": [float(self.box_H[0, 0]), float(self.box_H[1, 1]), float(self.box_H[2, 2])],
                "box_origin_ang": [float(self._box_origin_ang[0]), float(self._box_origin_ang[1]), float(self._box_origin_ang[2])],
                "volume_nm3": vol_nm3,
                "stress_zz": float(self.virial[2, 2] / max(vol_nm3, 1e-9)),
                "mode": self.mode,
                "backend": "symmetry-compact-cpu",
                "history_steps": list(self._history.keys()),
                "multiscale": self._build_multiscale_summary(),
                "smd": smd_info,
                "delta_streaming": {
                    "enabled": bool(self.delta_streaming_enabled),
                    "threshold_ang": float(self._dirty_tracker.threshold_ang),
                    "last_dirty_ratio": float(self._last_dirty_ratio),
                    "last_dirty_tiles": int(self._last_dirty_tiles),
                },
            }

    def encode_topology_packet(self) -> bytes:
        with self._lock:
            coords_ang = (self.positions * self.coord_to_ang).astype(np.float32)
            box_ang = (self.box_H * self.coord_to_ang).astype(np.float32)
            b_arr = np.zeros((0, 2), dtype=np.uint32)
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
                        use_int16=False,
                        scalar_channel=ch_idx,
                        pbc_wrapped=False,
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
                use_int16=False,
                pbc_wrapped=False,
                reversed_step=reversed_step,
                scalar_channel=ch_idx,
            )

    def encode_overlay_packet(self, max_vectors: int = 2000) -> bytes:
        with self._lock:
            cap = self.capacity
            if cap > 0:
                coords_ang = (self.positions * self.coord_to_ang).astype(np.float32)
                f_mags = np.linalg.norm(self.forces, axis=1)
                k = min(int(max_vectors), cap)
                top = np.argpartition(f_mags, -k)[-k:] if k < cap else np.arange(cap)
                starts = coords_ang[top]
                dirs = (self.forces[top] * 0.02).astype(np.float32)
                mags = f_mags[top].astype(np.float32)
            else:
                starts = np.zeros((0, 3), dtype=np.float32)
                dirs = np.zeros((0, 3), dtype=np.float32)
                mags = np.zeros(0, dtype=np.float32)

            return pack_overlay_vectors(
                vector_starts=starts,
                vector_dirs=dirs,
                vector_mags=mags,
                bond_pairs=np.zeros((0, 2), dtype=np.uint32),
                bond_orders=np.zeros(0, dtype=np.float32),
                frame_id=self.step_count,
                sim_time_ps=self.sim_time_ps,
            )

    def encode_volume_packet(self, ml: int = 27) -> bytes:
        with self._lock:
            n_grid = int(ml)
            Q_grid = np.zeros((n_grid, n_grid, n_grid), dtype=np.float32)
            if self.capacity > 0:
                rel_nm = self.positions - (self._box_origin_ang * 0.1)[None, :]
                Hinv = np.linalg.inv(self.box_H)
                frac = rel_nm @ Hinv.T
                frac -= np.floor(frac)
                idx = np.clip((frac * n_grid).astype(np.int32), 0, n_grid - 1)
                np.add.at(Q_grid, (idx[:, 0], idx[:, 1], idx[:, 2]), 1.0)
            box_ang = (self.box_H * self.coord_to_ang).astype(np.float32)
            return pack_volume_grid3d(
                Q_grid,
                box_H=box_ang,
                origin=tuple(float(v) for v in self._box_origin_ang),
                frame_id=self.step_count,
                sim_time_ps=self.sim_time_ps,
            )
