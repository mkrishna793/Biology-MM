#!/usr/bin/env python3
"""Generate the external AMBER parameter file for the `membrane` fixture.

The repository does not ship a membrane prmtop: `wrappers/python/openmm/app/data/DPPC.pdb`
is a structure only. This script builds the exact OpenMM `amber19/lipid21.xml` +
`amber19/tip3p.xml` system, exports it with ParmEd, and writes the pair
`dppc.prmtop` + `dppc.inpcrd` into an external cache directory. The generated
prmtop is ~12 MB and must never be committed.

The reference generator (`make_reference.py`) loads the generated prmtop through
`AmberPrmtopFile`, so this export is the independent parameter input for the
membrane reference.

Usage:
    python chemistry/tools/make_generated_fixtures.py [--out DIR] [--force]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

from openmm import Context, Platform, VerletIntegrator, Vec3
from openmm.app import (
    AmberInpcrdFile,
    AmberPrmtopFile,
    ForceField,
    NoCutoff,
    PDBFile,
    PME,
)
from openmm.unit import angstrom, kilojoule_per_mole, nanometer
import parmed

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "wrappers" / "python" / "openmm" / "app" / "data"
SOURCE_PDB = DATA_DIR / "DPPC.pdb"
DEFAULT_OUT = Path.home() / ".cache" / "biology-mm" / "chemistry-fixtures"

CUTOFF_NM = 1.0
EWALD_TOLERANCE = 5.0e-4
GENERATOR = "chemistry/tools/make_generated_fixtures.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _box_rows(box_vectors) -> np.ndarray:
    """Rows are lattice vectors, values in nm."""
    return np.array([[v.x, v.y, v.z] for v in box_vectors], dtype=np.float64)


def _lengths_angles(box_nm: np.ndarray):
    lengths = np.linalg.norm(box_nm, axis=1)
    angles = []
    for j, k in ((1, 2), (0, 2), (0, 1)):
        cosine = float(np.dot(box_nm[j], box_nm[k]) / (lengths[j] * lengths[k]))
        angles.append(math.degrees(math.acos(max(-1.0, min(1.0, cosine)))))
    return lengths, angles


def write_inpcrd(path: Path, positions_angstrom: np.ndarray, box_nm: np.ndarray, title: str) -> None:
    """Write an Amber ASCII restart file (6 fields of 12 chars per line).

    Velocities are omitted; the box line always carries lengths *and* angles so
    OpenMM's fixed-width parser recognizes it.
    """
    flat = np.asarray(positions_angstrom, dtype=np.float64).reshape(-1)
    lengths, angles = _lengths_angles(np.asarray(box_nm, dtype=np.float64))
    lines = [title, "%6d" % (flat.size // 3)]
    for start in range(0, flat.size, 6):
        lines.append("".join("%12.7f" % value for value in flat[start:start + 6]))
    box_fields = list(lengths * 10.0) + list(angles)  # nm -> Angstrom
    lines.append("".join("%12.7f" % value for value in box_fields))
    path.write_text("\n".join(lines) + "\n")


def _system_from_prmtop(prmtop: AmberPrmtopFile):
    system = prmtop.createSystem(
        nonbondedMethod=PME,
        nonbondedCutoff=CUTOFF_NM * nanometer,
        constraints=None,
        rigidWater=False,
        removeCMMotion=False,
        hydrogenMass=None,
        ewaldErrorTolerance=EWALD_TOLERANCE,
    )
    for force in system.getForces():
        if hasattr(force, "setUseDispersionCorrection"):
            force.setUseDispersionCorrection(False)
    return system


def verify(out_dir: Path) -> dict:
    """Load the generated pair back through AmberPrmtopFile/AmberInpcrdFile."""
    prmtop_path = out_dir / "dppc.prmtop"
    inpcrd_path = out_dir / "dppc.inpcrd"
    prmtop = AmberPrmtopFile(str(prmtop_path))
    inpcrd = AmberInpcrdFile(str(inpcrd_path))
    n_atoms = prmtop.topology.getNumAtoms()
    if len(inpcrd.positions) != n_atoms:
        raise RuntimeError(
            "generated coordinate count %d != prmtop atom count %d"
            % (len(inpcrd.positions), n_atoms)
        )
    box = prmtop.topology.getPeriodicBoxVectors()
    if box is None:
        raise RuntimeError("generated prmtop lost its periodic box")
    system = _system_from_prmtop(prmtop)
    integrator = VerletIntegrator(0.001)
    platform = Platform.getPlatformByName("CPU")
    context = Context(system, integrator, platform, {"Threads": "1", "DeterministicForces": "true"})
    context.setPeriodicBoxVectors(*[Vec3(*row) for row in _box_rows(box)])
    context.setPositions(inpcrd.positions)
    energy = context.getState(energy=True, enforcePeriodicBox=True).getPotentialEnergy()
    energy_kj = energy.value_in_unit(kilojoule_per_mole)
    if not math.isfinite(energy_kj):
        raise RuntimeError("generated membrane system produced a non-finite energy")
    return {
        "prmtop": str(prmtop_path),
        "inpcrd": str(inpcrd_path),
        "n_atoms": n_atoms,
        "box_nm": _box_rows(box).tolist(),
        "energy_kj_mol": energy_kj,
        "prmtop_sha256": sha256(prmtop_path),
        "inpcrd_sha256": sha256(inpcrd_path),
        "prmtop_bytes": prmtop_path.stat().st_size,
        "inpcrd_bytes": inpcrd_path.stat().st_size,
        "openmm_version": __import__("openmm").__version__,
    }


def generate(out_dir: Path = DEFAULT_OUT, force: bool = False) -> dict:
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    prmtop_path = out_dir / "dppc.prmtop"
    inpcrd_path = out_dir / "dppc.inpcrd"

    if prmtop_path.exists() and inpcrd_path.exists() and not force:
        return verify(out_dir)

    if not SOURCE_PDB.exists():
        raise FileNotFoundError("missing source structure: %s" % SOURCE_PDB)

    pdb = PDBFile(str(SOURCE_PDB))
    box = pdb.topology.getPeriodicBoxVectors()
    if box is None:
        raise RuntimeError("DPPC.pdb has no periodic box")
    positions_angstrom = np.array(pdb.positions.value_in_unit(angstrom), dtype=np.float64)

    forcefield = ForceField("amber19/lipid21.xml", "amber19/tip3p.xml")
    system = forcefield.createSystem(
        pdb.topology,
        nonbondedMethod=PME,
        nonbondedCutoff=CUTOFF_NM * nanometer,
        constraints=None,
        rigidWater=False,
        removeCMMotion=False,
    )
    for force_ in system.getForces():
        if hasattr(force_, "setUseDispersionCorrection"):
            force_.setUseDispersionCorrection(False)

    structure = parmed.openmm.load_topology(pdb.topology, system)
    structure.save(str(prmtop_path), format="amber", overwrite=True)
    write_inpcrd(inpcrd_path, positions_angstrom, _box_rows(box), "DPPC bilayer from OpenMM amber19/lipid21.xml + tip3p.xml")

    result = verify(out_dir)
    result["source_pdb"] = str(SOURCE_PDB)
    result["source_pdb_sha256"] = sha256(SOURCE_PDB)
    result["generator"] = GENERATOR
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="output directory (external cache)")
    parser.add_argument("--force", action="store_true", help="rebuild even if the files exist")
    args = parser.parse_args(argv)

    result = generate(Path(args.out), force=args.force)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
