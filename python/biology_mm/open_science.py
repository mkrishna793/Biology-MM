"""Biology-MM Open-Science Multi-Domain Physics, Chemistry & Biology Workbench.

Designed so researchers across biology, chemistry, and physics can:
1. Modify any physical parameter cleanly (force-field sigma/epsilon/charges/masses, bonded
   terms, cutoffs, dielectric, pair styles, integrators, thermostat/barostat, external E-field,
   or custom Python/NumPy potential functions).
2. Attach composable Enhanced Sampling CVs and biases (Umbrella, Steered MD, Well-Tempered
   Metadynamics, or PLUMED-style configs).
3. Stream trajectories directly to standard `.dcd`, `.xtc`, and `.pdb` files compatible with
   MDTraj, MDAnalysis, VMD, GROMACS, and OpenMM.
4. Measure exact physical observables: NVE/Shadow-H energy drift, radial distribution function
   g(r), equilibrium bulk density (g/cm^3), virial pressure (bar), and temperature (K).

Internal units: nm, ps, kJ/mol, amu (g/mol), elementary charge e, Kelvin, bar.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .enhanced_sampling import EnhancedSamplingSuite
from .trajectory_io import DCDWriter, XTCWriter, write_pdb_topology

__all__ = [
    "MolecularTopology",
    "OpenScienceEngine",
    "PhysicsParameters",
    "RunReport",
    "build_alanine_dipeptide",
    "build_lj_argon_fluid",
    "build_morse_diatomic",
    "build_tip3p_water_box",
    "build_yukawa_plasma",
    "radial_distribution_function",
]

_KB = 0.00831446261815324  # kJ / (mol * K)
_COULOMB_K = 138.93545764438198  # kJ * nm / (mol * e^2)
_KJ_MOL_NM3_TO_BAR = 16.605390666  # 1 kJ/(mol*nm^3) = 16.60539 bar
_AMU_PER_NM3_TO_G_CM3 = 0.00166053906660  # 1 amu/nm^3 = 1.660539e-3 g/cm^3 (1 amu/A^3 = 1.660539 g/cm^3)


@dataclass
class PhysicsParameters:
    """User-editable physical simulation parameters for biology, chemistry, and physics runs."""

    integrator: str = "baoab_nvt"  # "nve_verlet" | "baoab_nvt" | "baoab_npt"
    pair_style: str = "lj_shifted"  # "lj_shifted" | "lj_lfs" | "morse" | "buckingham" | "yukawa" | "none"
    dt_ps: float = 0.002
    temperature_K: float = 300.0
    pressure_bar: float = 1.0
    gamma_ps_inv: float = 1.0  # Langevin friction (1/ps)
    rcut_nm: float = 0.9
    dielectric: float = 1.0
    coulomb_kappa_nm_inv: float = 2.0  # Wolf / Yukawa screening parameter (1/nm)
    # Optional alternative pair-style parameters (editable for chemistry/physics experiments)
    morse_De_kj_mol: float = 420.0
    morse_a_nm_inv: float = 19.0
    morse_r0_nm: float = 0.14
    buckingham_A_kj_mol: float = 1.5e5
    buckingham_B_nm_inv: float = 34.0
    buckingham_C_kj_mol_nm6: float = 2.5e-3
    # Barostat settings (for baoab_npt)
    barostat_interval: int = 25
    barostat_max_dV_frac: float = 0.03
    # External field (kJ/(mol*nm*e)) along z
    efield_z: float = 0.0
    # Custom user potential: fn(positions_nm, box_nm, params) -> (energy_kj_mol, forces_kj_mol_nm)
    custom_potential: Optional[
        Callable[[np.ndarray, Optional[np.ndarray], "PhysicsParameters"], Tuple[float, np.ndarray]]
    ] = None
    seed: int = 42

    def override(self, **kwargs: Any) -> "PhysicsParameters":
        """Return a new `PhysicsParameters` with selected fields overridden cleanly."""
        data = {f.name: getattr(self, f.name) for f in self.__dataclass_fields__.values()}
        for k, v in kwargs.items():
            if k not in data:
                raise KeyError(f"Unknown physics parameter '{k}'")
            data[k] = v
        return PhysicsParameters(**data)


@dataclass
class MolecularTopology:
    """Complete molecular system description (bio, chem, or physics)."""

    positions_nm: np.ndarray  # (N, 3)
    masses_amu: np.ndarray  # (N,)
    sigma_nm: np.ndarray  # (N,)
    epsilon_kj_mol: np.ndarray  # (N,)
    charges_e: np.ndarray  # (N,)
    elements: List[str]
    atom_names: Optional[List[str]] = None
    residue_names: Optional[List[str]] = None
    residue_ids: Optional[List[int]] = None
    box_nm: Optional[np.ndarray] = None  # (3, 3) or None
    bonds: List[Tuple[int, int, float, float]] = field(default_factory=list)  # (i, j, k_b, r0_nm)
    angles: List[Tuple[int, int, int, float, float]] = field(default_factory=list)  # (i, j, k, k_th, th0_rad)
    dihedrals: List[Tuple[int, int, int, int, float, int, float]] = field(default_factory=list)  # (i,j,k,l,Vn,n,gamma)
    exclusions: Optional[set] = None  # set of (min(i,j), max(i,j)) excluded from nonbonded
    molecule_groups: Optional[List[np.ndarray]] = None  # optional molecule atom-index arrays for NPT COM scaling

    @property
    def n_atoms(self) -> int:
        return int(self.positions_nm.shape[0])


@dataclass
class RunReport:
    """Summary & time-series observables returned by `OpenScienceEngine.run(...)`."""

    n_steps: int
    time_ps: np.ndarray
    potential_energy_kj_mol: np.ndarray
    kinetic_energy_kj_mol: np.ndarray
    total_energy_kj_mol: np.ndarray
    bias_energy_kj_mol: np.ndarray
    temperature_K: np.ndarray
    pressure_bar: np.ndarray
    density_g_cm3: np.ndarray
    volume_nm3: np.ndarray
    relative_energy_drift: float
    mean_temperature_K: float
    mean_density_g_cm3: float
    cv_history: List[Dict[str, float]] = field(default_factory=list)
    exported_files: Dict[str, str] = field(default_factory=dict)


def _min_image(dr: np.ndarray, box_diag: Optional[np.ndarray]) -> np.ndarray:
    if box_diag is None:
        return dr
    return dr - box_diag * np.rint(dr / box_diag)


class OpenScienceEngine:
    """Clean, extensible, open-science molecular dynamics engine (NVE / NVT / NPT + CVs + DCD/XTC)."""

    def __init__(
        self,
        topology: MolecularTopology,
        params: Optional[PhysicsParameters] = None,
        enhanced_sampling: Optional[EnhancedSamplingSuite] = None,
    ) -> None:
        self.top = topology
        self.params = params or PhysicsParameters()
        self.enhanced = enhanced_sampling
        self.rng = np.random.default_rng(self.params.seed)

        self.positions = np.ascontiguousarray(topology.positions_nm, dtype=np.float64).copy()
        self.masses = np.ascontiguousarray(topology.masses_amu, dtype=np.float64).copy()
        self.sigma = np.ascontiguousarray(topology.sigma_nm, dtype=np.float64).copy()
        self.epsilon = np.ascontiguousarray(topology.epsilon_kj_mol, dtype=np.float64).copy()
        self.charges = np.ascontiguousarray(topology.charges_e, dtype=np.float64).copy()
        self.box = None if topology.box_nm is None else np.ascontiguousarray(topology.box_nm, dtype=np.float64).copy()

        # Build exclusion mask for 1-2 and 1-3 bonded pairs automatically if not supplied
        if topology.exclusions is not None:
            self._exclusions = set(topology.exclusions)
        else:
            ex = set()
            for i, j, _, _ in topology.bonds:
                ex.add((min(i, j), max(i, j)))
            for i, j, k, _, _ in topology.angles:
                ex.add((min(i, k), max(i, k)))
            self._exclusions = ex

        self.velocities = np.zeros_like(self.positions)
        self.forces = np.zeros_like(self.positions)
        self.step_count = 0
        self.time_ps = 0.0
        self._last_pot = 0.0
        self._last_bias = 0.0
        self._last_virial = 0.0
        self._init_velocities(self.params.temperature_K)
        self._last_pot, self._last_bias, self.forces, self._last_virial, _ = self.compute_forces(
            self.positions, self.box, step=0, time_ps=0.0
        )

    def _box_diag(self, box: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if box is None:
            return None
        if box.shape == (3,):
            return box if np.all(box > 0) else None
        diag = np.diag(box)
        return diag if np.all(diag > 0) else None

    def _init_velocities(self, temperature_K: float) -> None:
        n = self.top.n_atoms
        if n == 0 or temperature_K <= 0.0:
            self.velocities[:] = 0.0
            return
        # v_std = sqrt(kB * T / m) in nm/ps
        std = np.sqrt(_KB * float(temperature_K) / np.maximum(1e-8, self.masses))[:, None]
        v = self.rng.normal(0.0, 1.0, size=(n, 3)) * std
        if n > 1:
            p_tot = np.sum(self.masses[:, None] * v, axis=0)
            v -= p_tot / float(np.sum(self.masses))
        self.velocities = v

    def set_parameters(self, **kwargs: Any) -> PhysicsParameters:
        """Modify any simulation parameter on the fly."""
        self.params = self.params.override(**kwargs)
        return self.params

    def kinetic_energy(self) -> float:
        return float(0.5 * np.sum(self.masses[:, None] * (self.velocities ** 2)))

    def temperature(self) -> float:
        n = self.top.n_atoms
        dof = max(1, 3 * n - (3 if n > 1 else 0))
        return float(2.0 * self.kinetic_energy() / (dof * _KB))

    def volume_nm3(self) -> float:
        if self.box is None:
            return 0.0
        return float(abs(np.linalg.det(self.box))) if self.box.shape == (3, 3) else float(np.prod(self.box))

    def density_g_cm3(self) -> float:
        vol = self.volume_nm3()
        if vol <= 0.0:
            return 0.0
        total_mass_amu = float(np.sum(self.masses))
        return float((total_mass_amu / vol) * _AMU_PER_NM3_TO_G_CM3)

    def pressure_bar(self) -> float:
        vol = self.volume_nm3()
        if vol <= 0.0:
            return 0.0
        ke = self.kinetic_energy()
        # P = (2*KE + virial) / (3*V)
        p_internal = (2.0 * ke + self._last_virial) / (3.0 * vol)
        return float(p_internal * _KJ_MOL_NM3_TO_BAR)

    def _compute_nonbonded(
        self,
        pos: np.ndarray,
        box_diag: Optional[np.ndarray],
    ) -> Tuple[float, np.ndarray, float]:
        n = pos.shape[0]
        forces = np.zeros_like(pos)
        pot = 0.0
        virial = 0.0
        style = self.params.pair_style
        if style == "none" or n <= 1:
            return pot, forces, virial

        rcut = float(self.params.rcut_nm)
        rcut2 = rcut * rcut
        inv_rc = 1.0 / rcut
        kappa = float(self.params.coulomb_kappa_nm_inv)
        has_q = bool(np.any(self.charges != 0.0))
        inv_diel = _COULOMB_K / max(1e-6, float(self.params.dielectric))

        # Vectorized upper-triangle pair calculation (fast for N <= 2500 validation & research systems)
        iu, ju = np.triu_indices(n, k=1)
        if self._exclusions:
            keep = np.array([(int(i), int(j)) not in self._exclusions for i, j in zip(iu, ju)], dtype=bool)
            iu, ju = iu[keep], ju[keep]
        if iu.size == 0:
            return pot, forces, virial

        dr = pos[iu] - pos[ju]
        dr = _min_image(dr, box_diag)
        r2 = np.sum(dr * dr, axis=1)
        mask = (r2 > 1e-10) & (r2 < rcut2)
        if not np.any(mask):
            return pot, forces, virial

        iu, ju, dr, r2 = iu[mask], ju[mask], dr[mask], r2[mask]
        r = np.sqrt(r2)
        inv_r = 1.0 / r
        f_scalar = np.zeros_like(r)  # -(dU/dr) / r so F_i = f_scalar * dr

        if style in ("lj_shifted", "lj_lfs"):
            sig_ij = 0.5 * (self.sigma[iu] + self.sigma[ju])
            eps_ij = np.sqrt(np.maximum(0.0, self.epsilon[iu] * self.epsilon[ju]))
            active = (eps_ij > 0.0) & (sig_ij > 0.0)
            if np.any(active):
                sr = sig_ij[active] * inv_r[active]
                sr2 = sr * sr
                sr6 = sr2 * sr2 * sr2
                sr12 = sr6 * sr6
                u_lj = 4.0 * eps_ij[active] * (sr12 - sr6)
                du_dr = 4.0 * eps_ij[active] * (-12.0 * sr12 + 6.0 * sr6) * inv_r[active]

                src = sig_ij[active] * inv_rc
                src6 = src ** 6
                src12 = src6 * src6
                uc = 4.0 * eps_ij[active] * (src12 - src6)
                if style == "lj_lfs":
                    # Linear-Force-Shifted (C1 continuous force at rcut, matches Biology-MM :lfs)
                    duc = 4.0 * eps_ij[active] * (-12.0 * src12 + 6.0 * src6) * inv_rc
                    u_pair = u_lj - uc - (r[active] - rcut) * duc
                    du_pair = du_dr - duc
                else:
                    u_pair = u_lj - uc
                    du_pair = du_dr
                pot += float(np.sum(u_pair))
                f_scalar[active] += -du_pair * inv_r[active]

        elif style == "morse":
            De = float(self.params.morse_De_kj_mol)
            a = float(self.params.morse_a_nm_inv)
            r0 = float(self.params.morse_r0_nm)
            ex = np.exp(-a * (r - r0))
            ex_c = math.exp(-a * (rcut - r0))
            u_pair = De * ((1.0 - ex) ** 2 - (1.0 - ex_c) ** 2)
            du_dr = 2.0 * De * a * (1.0 - ex) * ex
            pot += float(np.sum(u_pair))
            f_scalar += -du_dr * inv_r

        elif style == "buckingham":
            A = float(self.params.buckingham_A_kj_mol)
            B = float(self.params.buckingham_B_nm_inv)
            C = float(self.params.buckingham_C_kj_mol_nm6)
            ex = np.exp(-B * r)
            inv_r6 = inv_r ** 6
            u_c = A * math.exp(-B * rcut) - C * (inv_rc ** 6)
            u_pair = A * ex - C * inv_r6 - u_c
            du_dr = -A * B * ex + 6.0 * C * inv_r6 * inv_r
            pot += float(np.sum(u_pair))
            f_scalar += -du_dr * inv_r

        elif style == "yukawa":
            qq = self.charges[iu] * self.charges[ju]
            ex = np.exp(-kappa * r)
            ex_c = math.exp(-kappa * rcut)
            u_pair = inv_diel * qq * (ex * inv_r - ex_c * inv_rc)
            du_dr = -inv_diel * qq * ex * (kappa * inv_r + inv_r * inv_r)
            pot += float(np.sum(u_pair))
            f_scalar += -du_dr * inv_r

        # Damped-shifted-force (Fennell-Gezelter / Wolf) electrostatics for non-Yukawa styles
        if has_q and style != "yukawa":
            qq = self.charges[iu] * self.charges[ju]
            qmask = qq != 0.0
            if np.any(qmask):
                rq = r[qmask]
                inv_rq = inv_r[qmask]
                qq_act = qq[qmask]
                # C1 shifted-force damped Coulomb (erfc-approximated via smooth Yukawa damping)
                ex = np.exp(-kappa * rq)
                ex_c = math.exp(-kappa * rcut)
                u_c = ex_c * inv_rc
                du_c = -ex_c * (kappa * inv_rc + inv_rc * inv_rc)
                u_el = inv_diel * qq_act * (ex * inv_rq - u_c - (rq - rcut) * du_c)
                du_el = inv_diel * qq_act * (-ex * (kappa * inv_rq + inv_rq * inv_rq) - du_c)
                pot += float(np.sum(u_el))
                f_scalar[qmask] += -du_el * inv_rq

        f_vec = f_scalar[:, None] * dr
        virial += float(np.sum(f_vec * dr))
        np.add.at(forces, iu, f_vec)
        np.add.at(forces, ju, -f_vec)
        return pot, forces, virial

    def _compute_bonded(
        self,
        pos: np.ndarray,
        box_diag: Optional[np.ndarray],
    ) -> Tuple[float, np.ndarray, float]:
        pot = 0.0
        forces = np.zeros_like(pos)
        virial = 0.0

        # Harmonic bonds: U = 0.5 * k_b * (r - r0)^2
        for i, j, kb, r0 in self.top.bonds:
            dr = _min_image(pos[i] - pos[j], box_diag)
            r = float(np.linalg.norm(dr))
            if r > 1e-12:
                diff = r - r0
                pot += 0.5 * kb * diff * diff
                f_vec = -(kb * diff / r) * dr
                forces[i] += f_vec
                forces[j] -= f_vec
                virial += float(np.dot(f_vec, dr))

        # Harmonic angles: U = 0.5 * k_th * (theta - th0)^2
        for i, j, k, kth, th0 in self.top.angles:
            v1 = _min_image(pos[i] - pos[j], box_diag)
            v2 = _min_image(pos[k] - pos[j], box_diag)
            r1 = float(np.linalg.norm(v1))
            r2 = float(np.linalg.norm(v2))
            if r1 > 1e-12 and r2 > 1e-12:
                u1, u2 = v1 / r1, v2 / r2
                cos_t = float(np.clip(np.dot(u1, u2), -1.0, 1.0))
                theta = math.acos(cos_t)
                sin_t = math.sqrt(max(1e-12, 1.0 - cos_t * cos_t))
                diff = theta - th0
                pot += 0.5 * kth * diff * diff
                dU_dth = kth * diff
                fi = dU_dth * (u2 - cos_t * u1) / (r1 * sin_t)
                fk = dU_dth * (u1 - cos_t * u2) / (r2 * sin_t)
                forces[i] += fi
                forces[k] += fk
                forces[j] -= fi + fk

        # Proper dihedrals: U = Vn * (1 + cos(n*phi - gamma))
        for i, j, k, l, vn, mult, gamma in self.top.dihedrals:
            b1 = _min_image(pos[j] - pos[i], box_diag)
            b2 = _min_image(pos[k] - pos[j], box_diag)
            b3 = _min_image(pos[l] - pos[k], box_diag)
            b2n = float(np.linalg.norm(b2))
            if b2n < 1e-12:
                continue
            m = np.cross(b1, b2)
            n_vec = np.cross(b2, b3)
            m2, n2 = float(np.dot(m, m)), float(np.dot(n_vec, n_vec))
            if m2 < 1e-16 or n2 < 1e-16:
                continue
            phi = math.atan2(float(np.dot(b1, n_vec)) * b2n, float(np.dot(m, n_vec)))
            pot += vn * (1.0 + math.cos(mult * phi - gamma))
            dU_dphi = -vn * mult * math.sin(mult * phi - gamma)
            g_i = -(b2n / m2) * m
            g_l = (b2n / n2) * n_vec
            inv_b2_2 = 1.0 / (b2n * b2n)
            g_j = (-(float(np.dot(b1, b2)) * inv_b2_2) - 1.0) * g_i + (float(np.dot(b3, b2)) * inv_b2_2) * g_l
            g_k = -(g_i + g_j + g_l)
            forces[i] -= dU_dphi * g_i
            forces[j] -= dU_dphi * g_j
            forces[k] -= dU_dphi * g_k
            forces[l] -= dU_dphi * g_l

        return pot, forces, virial

    def compute_forces(
        self,
        pos: np.ndarray,
        box: Optional[np.ndarray],
        step: int = 0,
        time_ps: float = 0.0,
        record_cv: bool = False,
    ) -> Tuple[float, float, np.ndarray, float, Dict[str, float]]:
        box_diag = self._box_diag(box)
        pot_nb, f_nb, vir_nb = self._compute_nonbonded(pos, box_diag)
        pot_b, f_b, vir_b = self._compute_bonded(pos, box_diag)
        pot = pot_nb + pot_b
        forces = f_nb + f_b
        virial = vir_nb + vir_b

        if abs(self.params.efield_z) > 0.0:
            pot += float(-np.sum(self.charges * pos[:, 2] * self.params.efield_z))
            forces[:, 2] += self.charges * self.params.efield_z

        if self.params.custom_potential is not None:
            u_c, f_c = self.params.custom_potential(pos, box, self.params)
            pot += float(u_c)
            forces += np.asarray(f_c, dtype=np.float64)

        bias_e = 0.0
        cv_obs: Dict[str, float] = {}
        if self.enhanced is not None:
            bias_e, f_bias, cv_obs = self.enhanced.evaluate(
                pos, box_nm=box, step=step, time_ps=time_ps, record=record_cv
            )
            forces += f_bias

        return float(pot), float(bias_e), forces, float(virial), cv_obs

    def _mc_barostat_attempt(self) -> bool:
        """Isotropic Monte Carlo barostat move (OpenMM/AMBER NPT style)."""
        if self.box is None:
            return False
        vol_old = self.volume_nm3()
        if vol_old <= 0.0:
            return False
        dV = self.rng.uniform(-1.0, 1.0) * self.params.barostat_max_dV_frac * vol_old
        vol_new = vol_old + dV
        if vol_new <= 0.1 * vol_old:
            return False
        scale = (vol_new / vol_old) ** (1.0 / 3.0)
        box_new = self.box * scale
        pos_new = self.positions.copy()

        # Scale molecular centers of mass (preserving rigid/bonded internal geometries)
        if self.top.molecule_groups:
            for grp in self.top.molecule_groups:
                w = self.masses[grp]
                com = np.sum(pos_new[grp] * w[:, None], axis=0) / float(np.sum(w))
                pos_new[grp] += com * (scale - 1.0)
            n_mol = len(self.top.molecule_groups)
        else:
            pos_new *= scale
            n_mol = self.top.n_atoms

        pot_new, bias_new, f_new, vir_new, _ = self.compute_forces(
            pos_new, box_new, step=self.step_count, time_ps=self.time_ps, record_cv=False
        )
        dU = (pot_new + bias_new) - (self._last_pot + self._last_bias)
        p_ext_kj = self.params.pressure_bar / _KJ_MOL_NM3_TO_BAR
        kT = _KB * max(1.0, self.params.temperature_K)
        dW = dU + p_ext_kj * dV - n_mol * kT * math.log(vol_new / vol_old)
        if dW <= 0.0 or self.rng.random() < math.exp(-dW / kT):
            self.box = box_new
            self.positions = pos_new
            self._last_pot = pot_new
            self._last_bias = bias_new
            self.forces = f_new
            self._last_virial = vir_new
            return True
        return False

    def step(self, record_cv: bool = False) -> Dict[str, float]:
        """Advance one integration step (Velocity Verlet NVE or symmetric BAOAB NVT/NPT)."""
        dt = float(self.params.dt_ps)
        inv_m = 1.0 / np.maximum(1e-8, self.masses)[:, None]

        # B (half kick)
        self.velocities += (0.5 * dt) * self.forces * inv_m

        if self.params.integrator == "nve_verlet":
            # A (full drift)
            self.positions += dt * self.velocities
        else:
            # BAOAB: A (half) -> O (exact OU thermostat) -> A (half)
            self.positions += (0.5 * dt) * self.velocities
            gamma = max(0.0, float(self.params.gamma_ps_inv))
            c1 = math.exp(-gamma * dt)
            c2 = math.sqrt(max(0.0, 1.0 - c1 * c1))
            sigma_v = np.sqrt(_KB * max(0.0, self.params.temperature_K) * inv_m)
            self.velocities = c1 * self.velocities + c2 * sigma_v * self.rng.normal(0.0, 1.0, size=self.velocities.shape)
            self.positions += (0.5 * dt) * self.velocities

        # Wrap coordinates into PBC box if periodic
        box_diag = self._box_diag(self.box)
        if box_diag is not None:
            self.positions = np.mod(self.positions, box_diag[None, :])

        self.step_count += 1
        self.time_ps += dt

        self._last_pot, self._last_bias, self.forces, self._last_virial, cv_obs = self.compute_forces(
            self.positions, self.box, step=self.step_count, time_ps=self.time_ps, record_cv=record_cv
        )
        # B (half kick)
        self.velocities += (0.5 * dt) * self.forces * inv_m

        if (
            self.params.integrator == "baoab_npt"
            and self.step_count % max(1, int(self.params.barostat_interval)) == 0
        ):
            self._mc_barostat_attempt()

        return cv_obs

    def minimize(self, max_steps: int = 200, step_size_nm: float = 0.002, force_tol: float = 50.0) -> float:
        """Steepest-descent energy minimization (clash relaxation) prior to MD."""
        for _ in range(max_steps):
            pot, bias, forces, vir, _ = self.compute_forces(self.positions, self.box, step=0, time_ps=0.0)
            f_norm = np.linalg.norm(forces, axis=1)
            max_f = float(np.max(f_norm)) if f_norm.size > 0 else 0.0
            if max_f <= force_tol:
                self._last_pot, self._last_bias, self.forces, self._last_virial = pot, bias, forces, vir
                break
            scale = step_size_nm / max(1.0, max_f)
            self.positions += np.clip(forces * scale, -0.02, 0.02)
        self._last_pot, self._last_bias, self.forces, self._last_virial, _ = self.compute_forces(
            self.positions, self.box, step=0, time_ps=0.0
        )
        return float(self._last_pot + self._last_bias)

    def run(
        self,
        n_steps: int,
        report_interval: int = 10,
        dcd_path: Optional[Union[str, Path]] = None,
        xtc_path: Optional[Union[str, Path]] = None,
        pdb_path: Optional[Union[str, Path]] = None,
        xtc_precision: float = 1000.0,
    ) -> RunReport:
        """Execute `n_steps` of MD, optionally streaming `.dcd`, `.xtc`, and `.pdb` files."""
        exported: Dict[str, str] = {}
        if pdb_path is not None:
            p = Path(pdb_path)
            p.write_text(
                write_pdb_topology(
                    self.positions,
                    elements=self.top.elements,
                    atom_names=self.top.atom_names,
                    residue_names=self.top.residue_names,
                    residue_ids=self.top.residue_ids,
                    box=self.box,
                    units="nm",
                ),
                encoding="utf-8",
            )
            exported["pdb"] = str(p)

        dcd_writer = (
            DCDWriter(
                dcd_path,
                n_atoms=self.top.n_atoms,
                dt_ps=self.params.dt_ps,
                nsavc=report_interval,
                has_unit_cell=self.box is not None,
            )
            if dcd_path is not None
            else None
        )
        xtc_writer = XTCWriter(xtc_path, precision=xtc_precision) if xtc_path is not None else None
        if dcd_path is not None:
            exported["dcd"] = str(dcd_path)
        if xtc_path is not None:
            exported["xtc"] = str(xtc_path)

        times: List[float] = []
        pots: List[float] = []
        kins: List[float] = []
        tots: List[float] = []
        biases: List[float] = []
        temps: List[float] = []
        presses: List[float] = []
        densities: List[float] = []
        vols: List[float] = []

        try:
            for s in range(1, int(n_steps) + 1):
                is_rep = (s % max(1, int(report_interval)) == 0) or (s == n_steps)
                self.step(record_cv=is_rep)
                if is_rep:
                    ke = self.kinetic_energy()
                    pe = self._last_pot
                    be = self._last_bias
                    times.append(self.time_ps)
                    pots.append(pe)
                    kins.append(ke)
                    tots.append(pe + be + ke)
                    biases.append(be)
                    temps.append(self.temperature())
                    presses.append(self.pressure_bar())
                    densities.append(self.density_g_cm3())
                    vols.append(self.volume_nm3())
                    if dcd_writer is not None:
                        dcd_writer.write_frame(self.positions, box=self.box, units="nm")
                    if xtc_writer is not None:
                        xtc_writer.write_frame(
                            self.positions, step=self.step_count, time_ps=self.time_ps, box=self.box, units="nm"
                        )
        finally:
            if dcd_writer is not None:
                dcd_writer.close()
            if xtc_writer is not None:
                xtc_writer.close()

        tot_arr = np.asarray(tots, dtype=np.float64)
        if tot_arr.size >= 2 and abs(tot_arr[0]) > 1e-12:
            rel_drift = float(np.std(tot_arr) / abs(tot_arr[0]))
        else:
            rel_drift = 0.0

        return RunReport(
            n_steps=int(n_steps),
            time_ps=np.asarray(times, dtype=np.float64),
            potential_energy_kj_mol=np.asarray(pots, dtype=np.float64),
            kinetic_energy_kj_mol=np.asarray(kins, dtype=np.float64),
            total_energy_kj_mol=tot_arr,
            bias_energy_kj_mol=np.asarray(biases, dtype=np.float64),
            temperature_K=np.asarray(temps, dtype=np.float64),
            pressure_bar=np.asarray(presses, dtype=np.float64),
            density_g_cm3=np.asarray(densities, dtype=np.float64),
            volume_nm3=np.asarray(vols, dtype=np.float64),
            relative_energy_drift=rel_drift,
            mean_temperature_K=float(np.mean(temps)) if temps else 0.0,
            mean_density_g_cm3=float(np.mean(densities)) if densities else 0.0,
            cv_history=list(self.enhanced.history) if self.enhanced is not None else [],
            exported_files=exported,
        )


def radial_distribution_function(
    frames_nm: np.ndarray,
    boxes_nm: np.ndarray,
    indices: Optional[Sequence[int]] = None,
    r_max_nm: float = 0.75,
    n_bins: int = 75,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the pair radial distribution function g(r) averaged over trajectory frames."""
    pos = np.asarray(frames_nm, dtype=np.float64)
    if pos.ndim == 2:
        pos = pos[np.newaxis, :, :]
    boxes = np.asarray(boxes_nm, dtype=np.float64)
    if boxes.ndim == 2:
        boxes = np.repeat(boxes[np.newaxis, :, :], pos.shape[0], axis=0)

    edges = np.linspace(0.0, float(r_max_nm), int(n_bins) + 1)
    r_centers = 0.5 * (edges[:-1] + edges[1:])
    hist_total = np.zeros(int(n_bins), dtype=np.float64)
    vol_sum = 0.0
    n_sel = 0

    for f_idx in range(pos.shape[0]):
        sub = pos[f_idx] if indices is None else pos[f_idx, indices]
        n_sel = sub.shape[0]
        box_diag = np.diag(boxes[f_idx])
        vol_sum += float(np.prod(box_diag))
        iu, ju = np.triu_indices(n_sel, k=1)
        dr = _min_image(sub[iu] - sub[ju], box_diag)
        dists = np.linalg.norm(dr, axis=1)
        counts, _ = np.histogram(dists, bins=edges)
        hist_total += 2.0 * counts  # count both i-j and j-i

    n_frames = pos.shape[0]
    mean_vol = vol_sum / max(1, n_frames)
    rho = n_sel / max(1e-12, mean_vol)
    shell_vols = (4.0 / 3.0) * math.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    norm = max(1e-12, n_frames * n_sel * rho) * shell_vols
    return r_centers, hist_total / norm


# =============================================================================
# Built-In Multi-Domain Scientific Builders (Biology, Chemistry, Physics)
# =============================================================================


def build_lj_argon_fluid(n_side: int = 4, density_g_cm3: float = 1.374) -> MolecularTopology:
    """Physics benchmark: Periodic Lennard-Jones liquid Argon (sigma=0.3405 nm, eps=0.996 kJ/mol)."""
    n = n_side ** 3
    mass_ar = 39.948
    vol_nm3 = (n * mass_ar * _AMU_PER_NM3_TO_G_CM3) / float(density_g_cm3)
    L = vol_nm3 ** (1.0 / 3.0)
    grid = (np.arange(n_side, dtype=np.float64) + 0.5) * (L / n_side)
    xx, yy, zz = np.meshgrid(grid, grid, grid, indexing="ij")
    pos = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
    return MolecularTopology(
        positions_nm=pos,
        masses_amu=np.full(n, mass_ar),
        sigma_nm=np.full(n, 0.3405),
        epsilon_kj_mol=np.full(n, 0.996),
        charges_e=np.zeros(n),
        elements=["Ar"] * n,
        atom_names=["AR"] * n,
        residue_names=["ARG"] * n,
        residue_ids=list(range(1, n + 1)),
        box_nm=np.diag([L, L, L]),
    )


def build_tip3p_water_box(n_side: int = 4, box_nm: Optional[float] = None) -> MolecularTopology:
    """Biology benchmark: Flexible TIP3P water box (O-H=0.09572 nm, H-O-H=104.52 deg, ~0.997 g/cm^3)."""
    n_mol = n_side ** 3
    if box_nm is None:
        # Target 0.997 g/cm^3: M_H2O = 18.01528 amu
        vol_nm3 = (n_mol * 18.01528 * _AMU_PER_NM3_TO_G_CM3) / 0.997
        box_nm = vol_nm3 ** (1.0 / 3.0)
    L = float(box_nm)
    spacing = L / n_side

    r_oh = 0.09572
    half_ang = math.radians(104.52 * 0.5)
    h1_rel = np.array([r_oh * math.sin(half_ang), r_oh * math.cos(half_ang), 0.0])
    h2_rel = np.array([-r_oh * math.sin(half_ang), r_oh * math.cos(half_ang), 0.0])

    pos_list: List[np.ndarray] = []
    masses: List[float] = []
    sigs: List[float] = []
    epss: List[float] = []
    qs: List[float] = []
    elems: List[str] = []
    anames: List[str] = []
    rnames: List[str] = []
    rids: List[int] = []
    bonds: List[Tuple[int, int, float, float]] = []
    angles: List[Tuple[int, int, int, float, float]] = []
    mol_groups: List[np.ndarray] = []

    mol_id = 0
    for ix in range(n_side):
        for iy in range(n_side):
            for iz in range(n_side):
                o_pos = np.array([(ix + 0.5) * spacing, (iy + 0.5) * spacing, (iz + 0.5) * spacing])
                base = len(pos_list)
                pos_list.extend([o_pos, o_pos + h1_rel, o_pos + h2_rel])
                masses.extend([15.9994, 1.00794, 1.00794])
                sigs.extend([0.315061, 0.0, 0.0])
                epss.extend([0.636386, 0.0, 0.0])
                qs.extend([-0.834, 0.417, 0.417])
                elems.extend(["O", "H", "H"])
                anames.extend(["OW", "HW1", "HW2"])
                mol_id += 1
                rnames.extend(["HOH", "HOH", "HOH"])
                rids.extend([mol_id, mol_id, mol_id])
                bonds.append((base, base + 1, 376560.0, r_oh))
                bonds.append((base, base + 2, 376560.0, r_oh))
                angles.append((base + 1, base, base + 2, 460.24, math.radians(104.52)))
                mol_groups.append(np.array([base, base + 1, base + 2], dtype=np.int64))

    return MolecularTopology(
        positions_nm=np.asarray(pos_list, dtype=np.float64),
        masses_amu=np.asarray(masses, dtype=np.float64),
        sigma_nm=np.asarray(sigs, dtype=np.float64),
        epsilon_kj_mol=np.asarray(epss, dtype=np.float64),
        charges_e=np.asarray(qs, dtype=np.float64),
        elements=elems,
        atom_names=anames,
        residue_names=rnames,
        residue_ids=rids,
        box_nm=np.diag([L, L, L]),
        bonds=bonds,
        angles=angles,
        molecule_groups=mol_groups,
    )


def build_alanine_dipeptide() -> MolecularTopology:
    """Biology enhanced-sampling benchmark: 10-heavy-atom Ace-Ala-Nme backbone with phi/psi torsions.

    Backbone phi = dihedral(1, 3, 4, 6) [C_ace - N - CA - C]
    Backbone psi = dihedral(3, 4, 6, 8) [N - CA - C - N_nme]
    """
    # Clean 3D coordinates (nm) for Ace-Ala-Nme heavy atoms:
    # 0:CH3_ace, 1:C_ace, 2:O_ace, 3:N_ala, 4:CA_ala, 5:CB_ala, 6:C_ala, 7:O_ala, 8:N_nme, 9:CH3_nme
    pos = np.array([
        [-0.360,  0.080,  0.000],  # 0 CH3_ace
        [-0.220,  0.030,  0.000],  # 1 C_ace
        [-0.180, -0.090,  0.020],  # 2 O_ace
        [-0.130,  0.130, -0.010],  # 3 N_ala
        [ 0.010,  0.100, -0.020],  # 4 CA_ala
        [ 0.060,  0.160, -0.150],  # 5 CB_ala
        [ 0.090,  0.150,  0.100],  # 6 C_ala
        [ 0.040,  0.240,  0.170],  # 7 O_ala
        [ 0.210,  0.090,  0.120],  # 8 N_nme
        [ 0.300,  0.130,  0.230],  # 9 CH3_nme
    ], dtype=np.float64)
    elems = ["C", "C", "O", "N", "C", "C", "C", "O", "N", "C"]
    anames = ["CH3", "C", "O", "N", "CA", "CB", "C", "O", "N", "CH3"]
    rnames = ["ACE", "ACE", "ACE", "ALA", "ALA", "ALA", "ALA", "ALA", "NME", "NME"]
    rids = [1, 1, 1, 2, 2, 2, 2, 2, 3, 3]
    masses = np.array([15.035, 12.011, 15.999, 14.007, 13.019, 15.035, 12.011, 15.999, 14.007, 15.035])
    sigma = np.array([0.34, 0.34, 0.30, 0.325, 0.34, 0.34, 0.34, 0.30, 0.325, 0.34])
    eps = np.array([0.45, 0.45, 0.88, 0.71, 0.45, 0.45, 0.45, 0.88, 0.71, 0.45])
    charges = np.array([0.0, 0.55, -0.55, -0.40, 0.15, 0.0, 0.55, -0.55, -0.40, 0.65])
    charges -= np.mean(charges)

    bond_pairs = [(0, 1), (1, 2), (1, 3), (3, 4), (4, 5), (4, 6), (6, 7), (6, 8), (8, 9)]
    bonds = [(i, j, 260000.0, float(np.linalg.norm(pos[j] - pos[i]))) for i, j in bond_pairs]
    angle_triples = [(0, 1, 2), (0, 1, 3), (2, 1, 3), (1, 3, 4), (3, 4, 5), (3, 4, 6), (5, 4, 6), (4, 6, 7), (4, 6, 8), (7, 6, 8), (6, 8, 9)]
    angles = []
    for i, j, k in angle_triples:
        v1, v2 = pos[i] - pos[j], pos[k] - pos[j]
        th0 = math.acos(float(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0)))
        angles.append((i, j, k, 420.0, th0))

    # Realistic double-well backbone phi/psi torsion barriers (AMBER-like)
    dihedrals = [
        (1, 3, 4, 6, 6.5, 2, 0.0),  # phi barrier
        (3, 4, 6, 8, 5.5, 2, 0.0),  # psi barrier
    ]
    return MolecularTopology(
        positions_nm=pos,
        masses_amu=masses,
        sigma_nm=sigma,
        epsilon_kj_mol=eps,
        charges_e=charges,
        elements=elems,
        atom_names=anames,
        residue_names=rnames,
        residue_ids=rids,
        box_nm=None,
        bonds=bonds,
        angles=angles,
        dihedrals=dihedrals,
    )


def build_morse_diatomic(r_init_nm: float = 0.16) -> MolecularTopology:
    """Chemistry benchmark: Anharmonic Morse bond oscillator (CO/H2-like dissociation)."""
    pos = np.array([[0.0, 0.0, 0.0], [float(r_init_nm), 0.0, 0.0]], dtype=np.float64)
    return MolecularTopology(
        positions_nm=pos,
        masses_amu=np.array([12.011, 15.9994]),
        sigma_nm=np.array([0.34, 0.30]),
        epsilon_kj_mol=np.array([0.45, 0.88]),
        charges_e=np.zeros(2),
        elements=["C", "O"],
        atom_names=["C1", "O1"],
        residue_names=["CO", "CO"],
        residue_ids=[1, 1],
        box_nm=None,
    )


def build_yukawa_plasma(n_side: int = 3, box_nm: float = 2.0) -> MolecularTopology:
    """Physics benchmark: Screened-Coulomb Yukawa plasma (alternating +1/-1 charges)."""
    n = n_side ** 3
    grid = (np.arange(n_side, dtype=np.float64) + 0.5) * (box_nm / n_side)
    xx, yy, zz = np.meshgrid(grid, grid, grid, indexing="ij")
    pos = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
    charges = np.array([1.0 if (i % 2 == 0) else -1.0 for i in range(n)], dtype=np.float64)
    if n % 2 == 1:
        charges[-1] = 0.0
    return MolecularTopology(
        positions_nm=pos,
        masses_amu=np.full(n, 23.0),
        sigma_nm=np.full(n, 0.25),
        epsilon_kj_mol=np.full(n, 0.35),
        charges_e=charges,
        elements=["Na" if q >= 0 else "Cl" for q in charges],
        box_nm=np.diag([box_nm, box_nm, box_nm]),
    )
