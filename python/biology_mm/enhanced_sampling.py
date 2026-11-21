"""Composable Factor-Graph Enhanced Sampling & Collective Variables (PLUMED-style).

Every CollectiveVariable implements an exact analytical Vector-Jacobian Product (VJP):
    `evaluate_with_grad(positions_nm, box_nm) -> (value, d_value_d_positions)`
so any bias potential `V(s)` produces exact, energy-conserving forces:
    `F = -(dV/ds) * (ds/dr)`.

Supported Collective Variables:
- `DistanceCV`: PBC minimum-image distance between two atoms (nm)
- `COMDistanceCV`: center-of-mass distance between two atom groups (nm)
- `AngleCV`: 3-atom angle in radians [0, pi]
- `DihedralCV`: 4-atom proper torsion in radians [-pi, pi] (periodic)
- `RadiusOfGyrationCV`: mass-weighted radius of gyration R_g (nm)
- `CoordinationNumberCV`: smooth rational switching coordination number

Supported Bias Potentials:
- `UmbrellaBias`: harmonic restraint V(s) = 0.5 * kappa * (s - s0)^2 (periodic-aware)
- `SteeredMDBias`: constant-velocity pulling s0(t) = s_start + v * t with Jarzynski work W(t)
- `WellTemperedMetadynamics`: adaptive Gaussian hills + free-energy reconstruction F(s)
- `EnhancedSamplingSuite`: composable multi-bias container + `PLUMEDParser.from_text(...)`
"""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "AngleCV",
    "BiasPotential",
    "COMDistanceCV",
    "CollectiveVariable",
    "CoordinationNumberCV",
    "DihedralCV",
    "DistanceCV",
    "EnhancedSamplingSuite",
    "PLUMEDParser",
    "RadiusOfGyrationCV",
    "SteeredMDBias",
    "UmbrellaBias",
    "WellTemperedMetadynamics",
]

_KB_KJ_MOL_K = 0.00831446261815324


def _minimum_image_vec(dr: np.ndarray, box_nm: Optional[np.ndarray]) -> np.ndarray:
    """Apply minimum-image convention for orthorhombic or triclinic 3x3 box matrices."""
    if box_nm is None:
        return dr
    H = np.asarray(box_nm, dtype=np.float64)
    if H.shape == (3,):
        if np.all(H > 0):
            return dr - H * np.rint(dr / H)
        return dr
    if H.shape == (3, 3) and abs(float(np.linalg.det(H))) > 1e-12:
        frac = np.linalg.solve(H.T, dr.T).T
        frac -= np.rint(frac)
        return frac @ H
    return dr


def _wrap_periodic(diff: float, period: Optional[float]) -> float:
    if period is None or period <= 0.0:
        return float(diff)
    return float((diff + 0.5 * period) % period - 0.5 * period)


# =============================================================================
# 1. Analytical Collective Variables (with exact VJP gradients)
# =============================================================================


class CollectiveVariable(ABC):
    """Base class for differentiable collective variables s(r) -> R."""

    name: str = "cv"
    periodic: Optional[float] = None  # e.g. 2*pi for DihedralCV

    @abstractmethod
    def evaluate_with_grad(
        self,
        positions_nm: np.ndarray,
        box_nm: Optional[np.ndarray] = None,
    ) -> Tuple[float, np.ndarray]:
        """Return `(s, grad_s)` where `grad_s` has shape `(n_atoms, 3)`."""

    def evaluate(
        self,
        positions_nm: np.ndarray,
        box_nm: Optional[np.ndarray] = None,
    ) -> float:
        val, _ = self.evaluate_with_grad(positions_nm, box_nm)
        return val


class DistanceCV(CollectiveVariable):
    """Minimum-image Euclidean distance between atoms `i` and `j` (nm)."""

    def __init__(self, i: int, j: int, name: str = "dist") -> None:
        self.i = int(i)
        self.j = int(j)
        self.name = name
        self.periodic = None

    def evaluate_with_grad(
        self,
        positions_nm: np.ndarray,
        box_nm: Optional[np.ndarray] = None,
    ) -> Tuple[float, np.ndarray]:
        pos = np.asarray(positions_nm, dtype=np.float64)
        grad = np.zeros_like(pos)
        dr = _minimum_image_vec(pos[self.j] - pos[self.i], box_nm)
        r = float(np.linalg.norm(dr))
        if r > 1e-12:
            u = dr / r
            grad[self.j] = u
            grad[self.i] = -u
        return r, grad


class COMDistanceCV(CollectiveVariable):
    """Center-of-mass distance between two atom index groups (nm)."""

    def __init__(
        self,
        group_a: Sequence[int],
        group_b: Sequence[int],
        masses: Optional[Sequence[float]] = None,
        name: str = "com_dist",
    ) -> None:
        self.group_a = np.asarray(group_a, dtype=np.int64)
        self.group_b = np.asarray(group_b, dtype=np.int64)
        self.masses = None if masses is None else np.asarray(masses, dtype=np.float64)
        self.name = name
        self.periodic = None

    def _weights(self, idx: np.ndarray) -> np.ndarray:
        if self.masses is None:
            w = np.ones(len(idx), dtype=np.float64)
        else:
            w = np.asarray(self.masses[idx], dtype=np.float64)
        return w / float(np.sum(w))

    def evaluate_with_grad(
        self,
        positions_nm: np.ndarray,
        box_nm: Optional[np.ndarray] = None,
    ) -> Tuple[float, np.ndarray]:
        pos = np.asarray(positions_nm, dtype=np.float64)
        grad = np.zeros_like(pos)
        wa = self._weights(self.group_a)
        wb = self._weights(self.group_b)
        com_a = np.sum(pos[self.group_a] * wa[:, None], axis=0)
        com_b = np.sum(pos[self.group_b] * wb[:, None], axis=0)
        dr = _minimum_image_vec(com_b - com_a, box_nm)
        r = float(np.linalg.norm(dr))
        if r > 1e-12:
            u = dr / r
            grad[self.group_b] += wb[:, None] * u[None, :]
            grad[self.group_a] -= wa[:, None] * u[None, :]
        return r, grad


class AngleCV(CollectiveVariable):
    """Bond/domain angle i-j-k in radians [0, pi]."""

    def __init__(self, i: int, j: int, k: int, name: str = "angle") -> None:
        self.i = int(i)
        self.j = int(j)
        self.k = int(k)
        self.name = name
        self.periodic = None

    def evaluate_with_grad(
        self,
        positions_nm: np.ndarray,
        box_nm: Optional[np.ndarray] = None,
    ) -> Tuple[float, np.ndarray]:
        pos = np.asarray(positions_nm, dtype=np.float64)
        grad = np.zeros_like(pos)
        v1 = _minimum_image_vec(pos[self.i] - pos[self.j], box_nm)
        v2 = _minimum_image_vec(pos[self.k] - pos[self.j], box_nm)
        r1 = float(np.linalg.norm(v1))
        r2 = float(np.linalg.norm(v2))
        if r1 < 1e-12 or r2 < 1e-12:
            return 0.0, grad
        u1 = v1 / r1
        u2 = v2 / r2
        cos_t = float(np.clip(np.dot(u1, u2), -1.0, 1.0))
        theta = math.acos(cos_t)
        sin_t = math.sqrt(max(1e-12, 1.0 - cos_t * cos_t))
        g_i = -(u2 - cos_t * u1) / (r1 * sin_t)
        g_k = -(u1 - cos_t * u2) / (r2 * sin_t)
        grad[self.i] = g_i
        grad[self.k] = g_k
        grad[self.j] = -(g_i + g_k)
        return theta, grad


class DihedralCV(CollectiveVariable):
    """Proper torsion angle i-j-k-l in radians [-pi, pi] (periodic 2*pi)."""

    def __init__(self, i: int, j: int, k: int, l: int, name: str = "torsion") -> None:
        self.i = int(i)
        self.j = int(j)
        self.k = int(k)
        self.l = int(l)
        self.name = name
        self.periodic = 2.0 * math.pi

    def evaluate_with_grad(
        self,
        positions_nm: np.ndarray,
        box_nm: Optional[np.ndarray] = None,
    ) -> Tuple[float, np.ndarray]:
        # Standard Blondel & Karplus (1996) singularity-free analytical dihedral gradient
        pos = np.asarray(positions_nm, dtype=np.float64)
        grad = np.zeros_like(pos)
        b1 = _minimum_image_vec(pos[self.j] - pos[self.i], box_nm)
        b2 = _minimum_image_vec(pos[self.k] - pos[self.j], box_nm)
        b3 = _minimum_image_vec(pos[self.l] - pos[self.k], box_nm)

        b2_norm = float(np.linalg.norm(b2))
        if b2_norm < 1e-12:
            return 0.0, grad
        m = np.cross(b1, b2)
        n = np.cross(b2, b3)
        m2 = float(np.dot(m, m))
        n2 = float(np.dot(n, n))
        if m2 < 1e-16 or n2 < 1e-16:
            return 0.0, grad

        # IUPAC/PLUMED sign convention: atan2(|b2| * b1.(b2 x b3), (b1 x b2).(b2 x b3))
        y = float(np.dot(b1, n)) * b2_norm
        x = float(np.dot(m, n))
        phi = math.atan2(y, x)

        # Analytical derivatives of phi w.r.t. r_i, r_j, r_k, r_l
        g_i = -(b2_norm / m2) * m
        g_l = (b2_norm / n2) * n
        b1_dot_b2 = float(np.dot(b1, b2))
        b3_dot_b2 = float(np.dot(b3, b2))
        inv_b2_2 = 1.0 / (b2_norm * b2_norm)
        g_j = (-(b1_dot_b2 * inv_b2_2) - 1.0) * g_i + (b3_dot_b2 * inv_b2_2) * g_l
        g_k = -(g_i + g_j + g_l)

        grad[self.i] = g_i
        grad[self.j] = g_j
        grad[self.k] = g_k
        grad[self.l] = g_l
        return phi, grad


class RadiusOfGyrationCV(CollectiveVariable):
    """Mass-weighted Radius of Gyration R_g (nm) for a selected group of atoms."""

    def __init__(
        self,
        indices: Sequence[int],
        masses: Optional[Sequence[float]] = None,
        name: str = "rg",
    ) -> None:
        self.indices = np.asarray(indices, dtype=np.int64)
        self.masses = None if masses is None else np.asarray(masses, dtype=np.float64)
        self.name = name
        self.periodic = None

    def evaluate_with_grad(
        self,
        positions_nm: np.ndarray,
        box_nm: Optional[np.ndarray] = None,
    ) -> Tuple[float, np.ndarray]:
        pos = np.asarray(positions_nm, dtype=np.float64)
        grad = np.zeros_like(pos)
        sub = pos[self.indices]
        if self.masses is None:
            w = np.ones(len(self.indices), dtype=np.float64)
        else:
            w = np.asarray(self.masses[self.indices], dtype=np.float64)
        w = w / float(np.sum(w))
        com = np.sum(sub * w[:, None], axis=0)
        rel = sub - com[None, :]
        rg2 = float(np.sum(w * np.sum(rel * rel, axis=1)))
        rg = math.sqrt(max(0.0, rg2))
        if rg > 1e-12:
            grad[self.indices] = (w[:, None] * rel) / rg
        return rg, grad


class CoordinationNumberCV(CollectiveVariable):
    """Smooth rational switching coordination number between two atom groups."""

    def __init__(
        self,
        group_a: Sequence[int],
        group_b: Sequence[int],
        r0_nm: float = 0.35,
        n_exp: int = 6,
        m_exp: int = 12,
        name: str = "coord",
    ) -> None:
        self.group_a = np.asarray(group_a, dtype=np.int64)
        self.group_b = np.asarray(group_b, dtype=np.int64)
        self.r0_nm = float(r0_nm)
        self.n_exp = int(n_exp)
        self.m_exp = int(m_exp)
        self.name = name
        self.periodic = None

    def evaluate_with_grad(
        self,
        positions_nm: np.ndarray,
        box_nm: Optional[np.ndarray] = None,
    ) -> Tuple[float, np.ndarray]:
        pos = np.asarray(positions_nm, dtype=np.float64)
        grad = np.zeros_like(pos)
        total = 0.0
        n, m = self.n_exp, self.m_exp
        inv_r0 = 1.0 / self.r0_nm

        for ia in self.group_a:
            for ib in self.group_b:
                if ia == ib:
                    continue
                dr = _minimum_image_vec(pos[ib] - pos[ia], box_nm)
                r = float(np.linalg.norm(dr))
                if r < 1e-12:
                    total += 1.0
                    continue
                x = r * inv_r0
                if abs(x - 1.0) < 1e-6:
                    # Smooth L'Hopital limit at x -> 1: s = n/m, ds/dx = n*(n - m)/(2*m)
                    s_val = float(n) / float(m)
                    ds_dx = float(n * (n - m)) / float(2 * m)
                else:
                    xn = x ** n
                    xm = x ** m
                    denom = 1.0 - xm
                    s_val = (1.0 - xn) / denom
                    ds_dx = (-n * (x ** (n - 1)) * denom + (1.0 - xn) * m * (x ** (m - 1))) / (denom * denom)
                total += s_val
                g_vec = (ds_dx * inv_r0 / r) * dr
                grad[ib] += g_vec
                grad[ia] -= g_vec
        return total, grad


# =============================================================================
# 2. Composable Bias Potentials (Umbrella, Steered MD, Well-Tempered Metadynamics)
# =============================================================================


class BiasPotential(ABC):
    """Base class for collective-variable bias potentials."""

    name: str = "bias"

    @abstractmethod
    def apply(
        self,
        positions_nm: np.ndarray,
        box_nm: Optional[np.ndarray] = None,
        step: int = 0,
        time_ps: float = 0.0,
    ) -> Tuple[float, np.ndarray, Dict[str, float]]:
        """Return `(bias_energy_kj_mol, bias_forces_kj_mol_nm, observables)`."""


class UmbrellaBias(BiasPotential):
    """Harmonic umbrella sampling restraint: V(s) = 0.5 * kappa * (s - center)^2."""

    def __init__(
        self,
        cv: CollectiveVariable,
        center: float,
        kappa: float,
        name: Optional[str] = None,
    ) -> None:
        self.cv = cv
        self.center = float(center)
        self.kappa = float(kappa)
        self.name = name or f"umbrella_{cv.name}"

    def apply(
        self,
        positions_nm: np.ndarray,
        box_nm: Optional[np.ndarray] = None,
        step: int = 0,
        time_ps: float = 0.0,
    ) -> Tuple[float, np.ndarray, Dict[str, float]]:
        s, grad_s = self.cv.evaluate_with_grad(positions_nm, box_nm)
        diff = _wrap_periodic(s - self.center, self.cv.periodic)
        energy = 0.5 * self.kappa * diff * diff
        dV_ds = self.kappa * diff
        forces = -dV_ds * grad_s
        return float(energy), forces, {self.cv.name: float(s), f"{self.name}_energy": float(energy)}


class SteeredMDBias(BiasPotential):
    """Constant-velocity Steered MD pulling along any CV, tracking Jarzynski work."""

    def __init__(
        self,
        cv: CollectiveVariable,
        start: float,
        velocity_per_ps: float,
        kappa: float,
        name: Optional[str] = None,
    ) -> None:
        self.cv = cv
        self.start = float(start)
        self.velocity_per_ps = float(velocity_per_ps)
        self.kappa = float(kappa)
        self.name = name or f"smd_{cv.name}"
        self.cumulative_work_kj_mol: float = 0.0
        self._last_time_ps: Optional[float] = None
        self._last_force_cv: float = 0.0

    def apply(
        self,
        positions_nm: np.ndarray,
        box_nm: Optional[np.ndarray] = None,
        step: int = 0,
        time_ps: float = 0.0,
    ) -> Tuple[float, np.ndarray, Dict[str, float]]:
        target = self.start + self.velocity_per_ps * float(time_ps)
        s, grad_s = self.cv.evaluate_with_grad(positions_nm, box_nm)
        diff = _wrap_periodic(s - target, self.cv.periodic)
        energy = 0.5 * self.kappa * diff * diff
        # Pulling force along CV: F_pull = -kappa * (s - target)
        f_pull = -self.kappa * diff
        forces = -(self.kappa * diff) * grad_s

        if self._last_time_ps is not None:
            dt = float(time_ps) - self._last_time_ps
            if dt > 0.0:
                # Trapezoidal work integration: dW = 0.5 * (F_prev + F_curr) * v * dt
                self.cumulative_work_kj_mol += 0.5 * (self._last_force_cv + f_pull) * self.velocity_per_ps * dt
        self._last_time_ps = float(time_ps)
        self._last_force_cv = float(f_pull)

        return float(energy), forces, {
            self.cv.name: float(s),
            f"{self.name}_target": float(target),
            f"{self.name}_force": float(f_pull),
            f"{self.name}_work": float(self.cumulative_work_kj_mol),
        }


class WellTemperedMetadynamics(BiasPotential):
    """Well-Tempered Metadynamics (Barducci et al., 2008) on any 1D CollectiveVariable."""

    def __init__(
        self,
        cv: CollectiveVariable,
        height_kj_mol: float = 1.2,
        sigma: float = 0.25,
        bias_factor: float = 10.0,
        temperature_K: float = 300.0,
        stride: int = 50,
        name: Optional[str] = None,
    ) -> None:
        self.cv = cv
        self.height_kj_mol = float(height_kj_mol)
        self.sigma = float(sigma)
        self.bias_factor = max(1.01, float(bias_factor))
        self.temperature_K = float(temperature_K)
        self.stride = max(1, int(stride))
        self.name = name or f"metad_{cv.name}"
        self._centers: List[float] = []
        self._heights: List[float] = []
        self._last_deposit_step: int = -1

    @property
    def n_hills(self) -> int:
        return len(self._centers)

    def _eval_bias_and_deriv(self, s: float) -> Tuple[float, float]:
        if not self._centers:
            return 0.0, 0.0
        centers = np.asarray(self._centers, dtype=np.float64)
        heights = np.asarray(self._heights, dtype=np.float64)
        diff = s - centers
        if self.cv.periodic is not None:
            p = self.cv.periodic
            diff = (diff + 0.5 * p) % p - 0.5 * p
        inv_s2 = 1.0 / (self.sigma * self.sigma)
        gauss = heights * np.exp(-0.5 * diff * diff * inv_s2)
        v = float(np.sum(gauss))
        dv_ds = float(np.sum(-gauss * diff * inv_s2))
        return v, dv_ds

    def apply(
        self,
        positions_nm: np.ndarray,
        box_nm: Optional[np.ndarray] = None,
        step: int = 0,
        time_ps: float = 0.0,
    ) -> Tuple[float, np.ndarray, Dict[str, float]]:
        s, grad_s = self.cv.evaluate_with_grad(positions_nm, box_nm)
        v_curr, dv_ds = self._eval_bias_and_deriv(s)

        if step > 0 and (step % self.stride == 0) and step != self._last_deposit_step:
            delta_T = (self.bias_factor - 1.0) * self.temperature_K
            scaled_h = self.height_kj_mol * math.exp(-v_curr / (_KB_KJ_MOL_K * delta_T))
            self._centers.append(float(s))
            self._heights.append(float(scaled_h))
            self._last_deposit_step = int(step)
            v_curr, dv_ds = self._eval_bias_and_deriv(s)

        forces = -dv_ds * grad_s
        return float(v_curr), forces, {
            self.cv.name: float(s),
            f"{self.name}_bias": float(v_curr),
            f"{self.name}_hills": float(len(self._centers)),
        }

    def free_energy_surface(self, grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Reconstruct 1D free-energy profile F(s) = -(gamma / (gamma - 1)) * V(s) shifted to min=0."""
        g = np.asarray(grid, dtype=np.float64)
        scale = -self.bias_factor / (self.bias_factor - 1.0)
        fes = np.array([scale * self._eval_bias_and_deriv(float(x))[0] for x in g], dtype=np.float64)
        if fes.size > 0:
            fes -= float(np.min(fes))
        return g, fes


@dataclass
class EnhancedSamplingSuite:
    """Composable container of multiple CV biases + trajectory observable logging."""

    biases: List[BiasPotential] = field(default_factory=list)
    cvs: Dict[str, CollectiveVariable] = field(default_factory=dict)
    history: List[Dict[str, float]] = field(default_factory=list)

    def add_cv(self, cv: CollectiveVariable) -> None:
        self.cvs[cv.name] = cv

    def add_bias(self, bias: BiasPotential) -> None:
        self.biases.append(bias)

    def evaluate(
        self,
        positions_nm: np.ndarray,
        box_nm: Optional[np.ndarray] = None,
        step: int = 0,
        time_ps: float = 0.0,
        record: bool = False,
    ) -> Tuple[float, np.ndarray, Dict[str, float]]:
        total_energy = 0.0
        total_forces = np.zeros_like(positions_nm, dtype=np.float64)
        obs: Dict[str, float] = {"step": float(step), "time_ps": float(time_ps)}

        for b in self.biases:
            e_b, f_b, o_b = b.apply(positions_nm, box_nm=box_nm, step=step, time_ps=time_ps)
            total_energy += e_b
            total_forces += f_b
            obs.update(o_b)

        for name, cv in self.cvs.items():
            if name not in obs:
                obs[name] = cv.evaluate(positions_nm, box_nm=box_nm)

        obs["bias_energy_kj_mol"] = float(total_energy)
        if record:
            self.history.append(obs)
        return float(total_energy), total_forces, obs


# =============================================================================
# 3. PLUMED-Style Declarative Input Parser
# =============================================================================


class PLUMEDParser:
    """Parse standard PLUMED-style `.dat` directives into an `EnhancedSamplingSuite`.

    Note: PLUMED atom indices are 1-based; this parser converts them to 0-based indices
    automatically when `one_based=True` (default).
    """

    @staticmethod
    def _parse_kv(tokens: Sequence[str]) -> Dict[str, str]:
        kv: Dict[str, str] = {}
        for tok in tokens:
            if "=" in tok:
                k, v = tok.split("=", 1)
                kv[k.strip().upper()] = v.strip()
        return kv

    @staticmethod
    def _parse_int_list(raw: str, offset: int) -> List[int]:
        out: List[int] = []
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                a, b = part.split("-", 1)
                out.extend(range(int(a) - offset, int(b) - offset + 1))
            else:
                out.append(int(part) - offset)
        return out

    @classmethod
    def from_text(cls, text: str, one_based: bool = True, temperature_K: float = 300.0) -> EnhancedSamplingSuite:
        suite = EnhancedSamplingSuite()
        offset = 1 if one_based else 0

        for raw_line in text.splitlines():
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            label: Optional[str] = None
            if ":" in line:
                maybe_label, rest = line.split(":", 1)
                if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", maybe_label.strip()):
                    label = maybe_label.strip()
                    line = rest.strip()
            tokens = line.split()
            if not tokens:
                continue
            cmd = tokens[0].upper()
            kv = cls._parse_kv(tokens[1:])
            name = label or kv.get("LABEL") or cmd.lower()

            if cmd == "DISTANCE":
                atoms = cls._parse_int_list(kv["ATOMS"], offset)
                if len(atoms) != 2:
                    raise ValueError("DISTANCE requires exactly 2 atoms")
                suite.add_cv(DistanceCV(atoms[0], atoms[1], name=name))
            elif cmd == "ANGLE":
                atoms = cls._parse_int_list(kv["ATOMS"], offset)
                if len(atoms) != 3:
                    raise ValueError("ANGLE requires exactly 3 atoms")
                suite.add_cv(AngleCV(atoms[0], atoms[1], atoms[2], name=name))
            elif cmd == "TORSION":
                atoms = cls._parse_int_list(kv["ATOMS"], offset)
                if len(atoms) != 4:
                    raise ValueError("TORSION requires exactly 4 atoms")
                suite.add_cv(DihedralCV(atoms[0], atoms[1], atoms[2], atoms[3], name=name))
            elif cmd == "GYRATION":
                atoms = cls._parse_int_list(kv["ATOMS"], offset)
                suite.add_cv(RadiusOfGyrationCV(atoms, name=name))
            elif cmd == "COORDINATION":
                ga = cls._parse_int_list(kv["GROUPA"], offset)
                gb = cls._parse_int_list(kv["GROUPB"], offset)
                r0 = float(kv.get("R_0", "0.35"))
                n_exp = int(kv.get("NN", "6"))
                m_exp = int(kv.get("MM", "12"))
                suite.add_cv(CoordinationNumberCV(ga, gb, r0_nm=r0, n_exp=n_exp, m_exp=m_exp, name=name))
            elif cmd == "RESTRAINT":
                cv = suite.cvs[kv["ARG"]]
                suite.add_bias(UmbrellaBias(cv, center=float(kv["AT"]), kappa=float(kv["KAPPA"]), name=name))
            elif cmd == "MOVINGRESTRAINT":
                cv = suite.cvs[kv["ARG"]]
                at0 = float(kv.get("AT0", kv.get("AT", "0.0")))
                vel = float(kv.get("VELOCITY", "0.05"))
                kappa = float(kv.get("KAPPA0", kv.get("KAPPA", "500.0")))
                suite.add_bias(SteeredMDBias(cv, start=at0, velocity_per_ps=vel, kappa=kappa, name=name))
            elif cmd == "METAD":
                cv = suite.cvs[kv["ARG"]]
                suite.add_bias(
                    WellTemperedMetadynamics(
                        cv,
                        height_kj_mol=float(kv.get("HEIGHT", "1.2")),
                        sigma=float(kv.get("SIGMA", "0.25")),
                        bias_factor=float(kv.get("BIASFACTOR", "10.0")),
                        temperature_K=float(kv.get("TEMP", str(temperature_K))),
                        stride=int(kv.get("PACE", "50")),
                        name=name,
                    )
                )
        return suite
