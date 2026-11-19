"""Atomic system and nanostructure builders for Biology MM."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .parameters import CustomPotential

__all__ = [
    "AtomicSystem",
    "NanostructureBuilder",
    "SystemBuilder",
]


def _pbc_bonds(
    positions: np.ndarray,
    box_lengths: Tuple[float, float, float],
    cutoff: float,
) -> List[Tuple[int, int]]:
    """Find all 0-based atom pairs `(i, j)` with `i < j` within minimum-image `cutoff`."""
    n = positions.shape[0]
    if n < 2:
        return []
    lx, ly, lz = box_lengths
    diff = positions[:, None, :] - positions[None, :, :]
    diff[:, :, 0] -= lx * np.rint(diff[:, :, 0] / lx)
    diff[:, :, 1] -= ly * np.rint(diff[:, :, 1] / ly)
    diff[:, :, 2] -= lz * np.rint(diff[:, :, 2] / lz)
    dist2 = np.sum(diff * diff, axis=2)
    mask = np.triu((dist2 > 1e-12) & (dist2 <= cutoff * cutoff), k=1)
    ii, jj = np.where(mask)
    return [(int(i), int(j)) for i, j in zip(ii, jj)]


@dataclass
class AtomicSystem:
    """Container for an atomic configuration, topology, box, and potential factors."""

    positions: np.ndarray
    masses: np.ndarray
    charges: np.ndarray
    box_lengths: Tuple[float, float, float]
    names: List[str] = field(default_factory=list)
    bonds: List[Tuple[int, int]] = field(default_factory=list)
    angles: List[Tuple[int, int, int]] = field(default_factory=list)
    potentials: List[CustomPotential] = field(default_factory=list)
    units: str = "reduced"

    def __post_init__(self) -> None:
        self.positions = np.ascontiguousarray(self.positions, dtype=np.float64)
        if self.positions.ndim != 2 or self.positions.shape[1] != 3:
            raise ValueError(f"positions must have shape (N, 3), got {self.positions.shape}")
        n = self.positions.shape[0]
        self.masses = np.ascontiguousarray(self.masses, dtype=np.float64)
        self.charges = np.ascontiguousarray(self.charges, dtype=np.float64)
        if self.masses.shape != (n,):
            raise ValueError(f"masses must have shape ({n},), got {self.masses.shape}")
        if self.charges.shape != (n,):
            raise ValueError(f"charges must have shape ({n},), got {self.charges.shape}")
        if np.any(self.masses <= 0.0) or not np.all(np.isfinite(self.masses)):
            raise ValueError("masses must be positive and finite")
        if len(self.box_lengths) != 3:
            raise ValueError("box_lengths must be a 3-tuple (Lx, Ly, Lz)")
        self.box_lengths = tuple(float(v) for v in self.box_lengths)  # type: ignore[assignment]
        if any((not math.isfinite(v) or v <= 0.0) for v in self.box_lengths):
            raise ValueError(f"box_lengths must be positive and finite, got {self.box_lengths}")
        if not self.names:
            self.names = ["X"] * n
        elif len(self.names) != n:
            raise ValueError(f"names length {len(self.names)} != n_atoms {n}")

    @property
    def n_atoms(self) -> int:
        return int(self.positions.shape[0])

    def copy(self) -> "AtomicSystem":
        return AtomicSystem(
            positions=self.positions.copy(),
            masses=self.masses.copy(),
            charges=self.charges.copy(),
            box_lengths=self.box_lengths,
            names=list(self.names),
            bonds=list(self.bonds),
            angles=list(self.angles),
            potentials=list(self.potentials),
            units=self.units,
        )

    def center_of_mass(self) -> np.ndarray:
        total_m = float(np.sum(self.masses))
        return np.sum(self.positions * self.masses[:, None], axis=0) / total_m

    def minimum_distance(self, periodic: bool = True) -> float:
        n = self.n_atoms
        if n < 2:
            return math.inf
        diff = self.positions[:, None, :] - self.positions[None, :, :]
        if periodic:
            lx, ly, lz = self.box_lengths
            diff[:, :, 0] -= lx * np.rint(diff[:, :, 0] / lx)
            diff[:, :, 1] -= ly * np.rint(diff[:, :, 1] / ly)
            diff[:, :, 2] -= lz * np.rint(diff[:, :, 2] / lz)
        dist2 = np.sum(diff * diff, axis=2)
        iu = np.triu_indices(n, k=1)
        return float(np.sqrt(np.min(dist2[iu])))


class SystemBuilder:
    """Fluent builder for general atomic systems and crystalline lattices."""

    def __init__(
        self,
        box_lengths: Tuple[float, float, float] = (10.0, 10.0, 10.0),
        units: str = "reduced",
    ) -> None:
        self._positions: List[Tuple[float, float, float]] = []
        self._masses: List[float] = []
        self._charges: List[float] = []
        self._names: List[str] = []
        self._bonds: List[Tuple[int, int]] = []
        self._angles: List[Tuple[int, int, int]] = []
        self._potentials: List[CustomPotential] = []
        self._box_lengths = (float(box_lengths[0]), float(box_lengths[1]), float(box_lengths[2]))
        self._units = str(units)

    def set_box(self, lx: float, ly: float, lz: float) -> "SystemBuilder":
        self._box_lengths = (float(lx), float(ly), float(lz))
        return self

    def add_atom(
        self,
        position: Sequence[float],
        mass: float = 1.0,
        charge: float = 0.0,
        name: str = "Ar",
    ) -> int:
        if len(position) != 3:
            raise ValueError("position must have 3 coordinates")
        idx = len(self._positions)
        self._positions.append((float(position[0]), float(position[1]), float(position[2])))
        self._masses.append(float(mass))
        self._charges.append(float(charge))
        self._names.append(str(name))
        return idx

    def add_bond(
        self,
        i: int,
        j: int,
        k: Optional[float] = None,
        r0: Optional[float] = None,
    ) -> "SystemBuilder":
        self._bonds.append((int(i), int(j)))
        if k is not None and r0 is not None:
            self._potentials.append(CustomPotential.harmonic_bond(i, j, k, r0))
        return self

    def add_angle(self, i: int, j: int, k: int) -> "SystemBuilder":
        self._angles.append((int(i), int(j), int(k)))
        return self

    def add_potential(self, potential: CustomPotential) -> "SystemBuilder":
        self._potentials.append(potential)
        return self

    def build(self) -> AtomicSystem:
        if not self._positions:
            raise ValueError("Cannot build an empty AtomicSystem")
        return AtomicSystem(
            positions=np.array(self._positions, dtype=np.float64),
            masses=np.array(self._masses, dtype=np.float64),
            charges=np.array(self._charges, dtype=np.float64),
            box_lengths=self._box_lengths,
            names=list(self._names),
            bonds=list(self._bonds),
            angles=list(self._angles),
            potentials=list(self._potentials),
            units=self._units,
        )

    @classmethod
    def fcc_lattice(
        cls,
        n_cells: int,
        rho: float = 0.8,
        mass: float = 1.0,
        charge: float = 0.0,
        name: str = "Ar",
        units: str = "reduced",
    ) -> AtomicSystem:
        """Build an `n_cells x n_cells x n_cells` FCC crystal at number density `rho`."""
        nc = int(n_cells)
        if nc < 1:
            raise ValueError("n_cells must be >= 1")
        if rho <= 0.0:
            raise ValueError("rho must be positive")
        a = (4.0 / float(rho)) ** (1.0 / 3.0)
        basis = ((0.0, 0.0, 0.0), (0.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0))
        L = nc * a
        sb = cls(box_lengths=(L, L, L), units=units)
        for ix in range(nc):
            for iy in range(nc):
                for iz in range(nc):
                    for bx, by, bz in basis:
                        sb.add_atom(
                            ((ix + bx) * a, (iy + by) * a, (iz + bz) * a),
                            mass=mass,
                            charge=charge,
                            name=name,
                        )
        return sb.build()

    @classmethod
    def simple_cubic(
        cls,
        n_per_side: int,
        spacing: float = 1.0,
        mass: float = 1.0,
        charge: float = 0.0,
        name: str = "Ar",
        units: str = "reduced",
    ) -> AtomicSystem:
        """Build an `n_per_side^3` simple cubic lattice with spacing `spacing`."""
        ns = int(n_per_side)
        if ns < 1:
            raise ValueError("n_per_side must be >= 1")
        if spacing <= 0.0:
            raise ValueError("spacing must be positive")
        L = ns * float(spacing)
        sb = cls(box_lengths=(L, L, L), units=units)
        for ix in range(ns):
            for iy in range(ns):
                for iz in range(ns):
                    sb.add_atom(
                        ((ix + 0.5) * spacing, (iy + 0.5) * spacing, (iz + 0.5) * spacing),
                        mass=mass,
                        charge=charge,
                        name=name,
                    )
        return sb.build()


class NanostructureBuilder:
    """Builders for carbon nanotubes, graphene sheets, nanopore membranes, and silicon crystals."""

    @staticmethod
    def carbon_nanotube(
        n: int,
        m: int,
        n_cells: int = 2,
        bond_length_nm: float = 0.142,
        pad_xy_nm: float = 2.0,
        mass: float = 12.011,
    ) -> AtomicSystem:
        """
        Construct an `(n, m)` single-walled carbon nanotube periodic along the z-axis.
        Uses exact chiral vector `C_h = n a1 + m a2` and orthogonal translational vector
        `T = t1 a1 + t2 a2`.
        """
        if n < 0 or m < 0 or (n == 0 and m == 0):
            raise ValueError(f"Chiral indices (n, m) must be non-negative and not both zero, got ({n}, {m})")
        if n_cells < 1:
            raise ValueError(f"n_cells must be >= 1, got {n_cells}")
        if bond_length_nm <= 0.0:
            raise ValueError("bond_length_nm must be positive")

        a_cc = float(bond_length_nm)
        a_lat = math.sqrt(3.0) * a_cc

        d_r = math.gcd(2 * n + m, n + 2 * m)
        t1 = (2 * m + n) // d_r
        t2 = -(2 * n + m) // d_r

        # |D| = number of graphene hexagons per 1D CNT unit cell
        det = n * t2 - m * t1
        n_hex = abs(det)
        expected_per_cell = 2 * n_hex

        c_h = a_lat * math.sqrt(n * n + n * m + m * m)
        radius = c_h / (2.0 * math.pi)
        t_len = a_lat * math.sqrt(t1 * t1 + t1 * t2 + t2 * t2)

        # Bounding box in (i, j) covering unit square (u, v) in [0, 1]^2
        corners_i = [0, n, t1, n + t1]
        corners_j = [0, m, t2, m + t2]
        i_min, i_max = min(corners_i) - 2, max(corners_i) + 2
        j_min, j_max = min(corners_j) - 2, max(corners_j) + 2

        unit_uv: List[Tuple[float, float]] = []
        seen = set()
        for ii in range(i_min, i_max + 1):
            for jj in range(j_min, j_max + 1):
                for b1, b2 in ((0.0, 0.0), (1.0 / 3.0, 1.0 / 3.0)):
                    ri = ii + b1
                    rj = jj + b2
                    u = (t2 * ri - t1 * rj) / det
                    v = (-m * ri + n * rj) / det
                    u_w = u % 1.0
                    v_w = v % 1.0
                    if abs(u_w - 1.0) < 1e-9:
                        u_w = 0.0
                    if abs(v_w - 1.0) < 1e-9:
                        v_w = 0.0
                    key = (round(u_w * 1e7), round(v_w * 1e7))
                    if key not in seen:
                        seen.add(key)
                        unit_uv.append((u_w, v_w))

        if len(unit_uv) != expected_per_cell:
            raise RuntimeError(
                f"CNT ({n},{m}) unit cell enumeration mismatch: got {len(unit_uv)} atoms, expected {expected_per_cell}"
            )

        unit_uv.sort(key=lambda uv: (uv[1], uv[0]))

        lx = 2.0 * radius + float(pad_xy_nm)
        ly = lx
        lz = n_cells * t_len
        cx, cy = 0.5 * lx, 0.5 * ly

        coords: List[Tuple[float, float, float]] = []
        for cell in range(n_cells):
            for u_w, v_w in unit_uv:
                theta = 2.0 * math.pi * u_w
                x = cx + radius * math.cos(theta)
                y = cy + radius * math.sin(theta)
                z = (v_w + cell) * t_len
                coords.append((x, y, z))

        pos = np.array(coords, dtype=np.float64)
        n_atoms = pos.shape[0]
        box = (lx, ly, lz)
        bonds = _pbc_bonds(pos, box, cutoff=1.22 * a_cc)
        return AtomicSystem(
            positions=pos,
            masses=np.full(n_atoms, float(mass), dtype=np.float64),
            charges=np.zeros(n_atoms, dtype=np.float64),
            box_lengths=box,
            names=["C"] * n_atoms,
            bonds=bonds,
            units="md",
        )

    @staticmethod
    def graphene_sheet(
        nx: int,
        ny: int,
        bond_length_nm: float = 0.142,
        vacuum_z_nm: float = 3.0,
        mass: float = 12.011,
    ) -> AtomicSystem:
        """Construct an orthorhombic periodic graphene sheet of `nx x ny` 4-atom rectangular cells."""
        if nx < 1 or ny < 1:
            raise ValueError("nx and ny must be >= 1")
        if bond_length_nm <= 0.0 or vacuum_z_nm <= 0.0:
            raise ValueError("bond_length_nm and vacuum_z_nm must be positive")

        a_cc = float(bond_length_nm)
        ax = math.sqrt(3.0) * a_cc
        ay = 3.0 * a_cc
        z0 = 0.5 * float(vacuum_z_nm)

        basis = (
            (0.0, 0.0),
            (0.5 * ax, 0.5 * a_cc),
            (0.5 * ax, 1.5 * a_cc),
            (0.0, 2.0 * a_cc),
        )
        coords: List[Tuple[float, float, float]] = []
        for ix in range(nx):
            for iy in range(ny):
                for bx, by in basis:
                    coords.append((ix * ax + bx, iy * ay + by, z0))

        pos = np.array(coords, dtype=np.float64)
        n_atoms = pos.shape[0]
        box = (nx * ax, ny * ay, float(vacuum_z_nm))
        bonds = _pbc_bonds(pos, box, cutoff=1.2 * a_cc)
        return AtomicSystem(
            positions=pos,
            masses=np.full(n_atoms, float(mass), dtype=np.float64),
            charges=np.zeros(n_atoms, dtype=np.float64),
            box_lengths=box,
            names=["C"] * n_atoms,
            bonds=bonds,
            units="md",
        )

    @staticmethod
    def nanopore_membrane(
        nx: int,
        ny: int,
        pore_radius_nm: float = 0.45,
        bond_length_nm: float = 0.142,
        box_z_nm: float = 4.0,
        mass: float = 12.011,
    ) -> AtomicSystem:
        """Construct a graphene membrane with a circular nanopore centered in XY."""
        if pore_radius_nm <= 0.0:
            raise ValueError("pore_radius_nm must be positive")
        sheet = NanostructureBuilder.graphene_sheet(
            nx=nx,
            ny=ny,
            bond_length_nm=bond_length_nm,
            vacuum_z_nm=box_z_nm,
            mass=mass,
        )
        lx, ly, lz = sheet.box_lengths
        cx, cy = 0.5 * lx, 0.5 * ly
        dx = sheet.positions[:, 0] - cx
        dy = sheet.positions[:, 1] - cy
        r_xy = np.sqrt(dx * dx + dy * dy)
        keep = r_xy >= float(pore_radius_nm)
        pos_kept = sheet.positions[keep]
        if pos_kept.shape[0] == 0:
            raise ValueError("pore_radius_nm removed all atoms from the graphene sheet")
        n_atoms = pos_kept.shape[0]
        bonds = _pbc_bonds(pos_kept, sheet.box_lengths, cutoff=1.2 * float(bond_length_nm))
        return AtomicSystem(
            positions=pos_kept,
            masses=np.full(n_atoms, float(mass), dtype=np.float64),
            charges=np.zeros(n_atoms, dtype=np.float64),
            box_lengths=sheet.box_lengths,
            names=["C"] * n_atoms,
            bonds=bonds,
            units="md",
        )

    @staticmethod
    def silicon_diamond(
        nx: int,
        ny: int,
        nz: int,
        a_nm: float = 0.5431,
        mass: float = 28.085,
    ) -> AtomicSystem:
        """Construct an `nx x ny x nz` silicon crystal in the conventional 8-atom diamond cubic cell."""
        if nx < 1 or ny < 1 or nz < 1:
            raise ValueError("nx, ny, nz must be >= 1")
        if a_nm <= 0.0:
            raise ValueError("a_nm must be positive")

        a = float(a_nm)
        basis = (
            (0.00, 0.00, 0.00),
            (0.00, 0.50, 0.50),
            (0.50, 0.00, 0.50),
            (0.50, 0.50, 0.00),
            (0.25, 0.25, 0.25),
            (0.25, 0.75, 0.75),
            (0.75, 0.25, 0.75),
            (0.75, 0.75, 0.25),
        )
        coords: List[Tuple[float, float, float]] = []
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    for bx, by, bz in basis:
                        coords.append(((ix + bx) * a, (iy + by) * a, (iz + bz) * a))

        pos = np.array(coords, dtype=np.float64)
        n_atoms = pos.shape[0]
        box = (nx * a, ny * a, nz * a)
        nn_dist = (math.sqrt(3.0) / 4.0) * a
        bonds = _pbc_bonds(pos, box, cutoff=1.2 * nn_dist)
        return AtomicSystem(
            positions=pos,
            masses=np.full(n_atoms, float(mass), dtype=np.float64),
            charges=np.zeros(n_atoms, dtype=np.float64),
            box_lengths=box,
            names=["Si"] * n_atoms,
            bonds=bonds,
            units="md",
        )
