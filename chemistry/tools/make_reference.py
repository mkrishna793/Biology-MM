#!/usr/bin/env python3
"""Produce independent OpenMM reference energies and forces for the fixture set.

The reference loads the *parameter file itself* (AmberPrmtopFile, or
CharmmPsfFile + CharmmParameterSet) -- never a normalized intermediate produced
by this repository -- so it is an independent oracle for the Phase-2 importer.

Protocol (matches chemistry/INTERFACE.md and the engine defaults):

* periodic fixtures: PME, nonbondedCutoff 1.0 nm, ewaldErrorTolerance 1e-6,
  constraints=None, rigidWater=False, removeCMMotion=False, hydrogenMass=None,
  dispersion correction disabled on every NonbondedForce.
* DNA (no prmtop box) and the CHARMM adapter: NoCutoff, same constraint flags.
* The periodic box is always the PRMTOP box (authoritative), applied with
  Context.setPeriodicBoxVectors. A coordinate-file box is only recorded, never
  used.
* Platform CPU, Threads=1 and DeterministicForces=true when available.

Configurations: config 0 is the source coordinates. Every config c >= 1 applies
the INTERFACE perturbation to all three components, with 0-based atom index i and
double precision:

    q[d,i] += 0.02 * config * sin(0.37*(i+1) + 1.1*(d+1))      config = 1,2,...

So config 1 has amplitude 0.02 nm and config 2 amplitude 0.04 nm; all five
fixtures store configs 0, 1 and 2. No coordinates are stored: Julia regenerates
them from the source file with the same rule.

Artifacts per fixture, under chemistry/reference/:
    <id>.json   metadata, protocol, per-config energies and force summaries
    <id>.f32    little-endian Float32 forces, config-major; within a config the
                flat C-order sequence of the OpenMM (N,3) array
                (x1,y1,z1,x2,y2,z2,...), i.e. Julia reshape(flat,3,N)

Usage:
    python chemistry/tools/make_reference.py --fixture water_ions [--force]
    python chemistry/tools/make_reference.py --fixture all --force
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

try:
    from openmm import Context, Platform, Vec3, VerletIntegrator
    from openmm.app import (
        AmberInpcrdFile,
        AmberPrmtopFile,
        CharmmParameterSet,
        CharmmPsfFile,
        NoCutoff,
        PDBFile,
        PME,
    )
    from openmm.unit import kilojoule_per_mole, nanometer
except ImportError:
    Context = Platform = Vec3 = VerletIntegrator = None
    AmberInpcrdFile = AmberPrmtopFile = CharmmParameterSet = CharmmPsfFile = None
    NoCutoff = PDBFile = PME = None
    kilojoule_per_mole = nanometer = None

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = REPO_ROOT / "chemistry" / "fixtures"
REFERENCE_DIR = REPO_ROOT / "chemistry" / "reference"
MANIFEST_PATH = FIXTURE_DIR / "manifest.json"
GENERATOR = "chemistry/tools/make_reference.py"
SCHEMA = "1.0.0"
CUTOFF_NM = 1.0
EWALD_TOLERANCE = 1.0e-6
CHARMM_SYSTEM_DIR = REPO_ROOT / "wrappers" / "python" / "tests" / "systems"

# Total configurations stored per fixture (0-based config index 0..n-1).
CONFIG_COUNT = {
    "water_ions": 3,
    "solvated_protein": 3,
    "dna": 3,
    "membrane": 3,
    "charmm_adapter": 3,
    "protein_ligand": 3,
}

PROTOCOLS = {
    "water_ions": True,
    "solvated_protein": True,
    "dna": False,
    "membrane": True,
    "charmm_adapter": False,
    "protein_ligand": False,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def box_rows(box_vectors) -> list[list[float]] | None:
    if box_vectors is None:
        return None
    return [[float(v.x), float(v.y), float(v.z)] for v in box_vectors]


def load_manifest() -> dict:
    with open(MANIFEST_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _numpy_box(box: list[list[float]]) -> list[Vec3]:
    return [Vec3(*row) for row in box]


def perturbed(base: np.ndarray, config: int) -> np.ndarray:
    """INTERFACE perturbation, all components, amplitude 0.02*config nm."""
    if config == 0:
        return base.copy()
    index = np.arange(base.shape[0], dtype=np.float64)
    positions = base.copy()
    for d in range(3):
        positions[:, d] += 0.02 * config * np.sin(
            0.37 * (index + 1.0) + 1.1 * (d + 1.0)
        )
    return positions


def build_amber(parm_path: Path, crd_path: Path, periodic: bool):
    prmtop = AmberPrmtopFile(str(parm_path))
    inpcrd = AmberInpcrdFile(str(crd_path))
    n_atoms = prmtop.topology.getNumAtoms()
    if len(inpcrd.positions) != n_atoms:
        raise RuntimeError("%s: coordinate count != prmtop atom count" % parm_path.name)

    prmtop_box = box_rows(prmtop.topology.getPeriodicBoxVectors())
    coordinate_box = box_rows(inpcrd.boxVectors)

    if periodic:
        if prmtop_box is None:
            raise RuntimeError("%s: periodic fixture but prmtop has no box" % parm_path.name)
        system = prmtop.createSystem(
            nonbondedMethod=PME,
            nonbondedCutoff=CUTOFF_NM * nanometer,
            constraints=None,
            rigidWater=False,
            removeCMMotion=False,
            hydrogenMass=None,
            ewaldErrorTolerance=EWALD_TOLERANCE,
        )
    else:
        if prmtop_box is not None:
            raise RuntimeError("%s: non-periodic fixture but prmtop has a box" % parm_path.name)
        system = prmtop.createSystem(
            nonbondedMethod=NoCutoff,
            constraints=None,
            rigidWater=False,
            removeCMMotion=False,
            hydrogenMass=None,
        )

    positions = np.array(inpcrd.positions.value_in_unit(nanometer), dtype=np.float64)
    return system, positions, prmtop_box, coordinate_box


def build_charmm(psf_path: Path, pdb_path: Path, parameter_sets):
    psf = CharmmPsfFile(str(psf_path))
    params = CharmmParameterSet(*[str(path) for path in parameter_sets])
    pdb = PDBFile(str(pdb_path))
    if psf.topology.getNumAtoms() != pdb.topology.getNumAtoms():
        raise RuntimeError("PSF and PDB atom counts differ")
    psf_names = [atom.name for atom in psf.topology.atoms()]
    pdb_names = [atom.name for atom in pdb.topology.atoms()]
    if psf_names != pdb_names:
        raise RuntimeError("PSF and PDB atom order differs; refusing to pair them")

    system = psf.createSystem(
        params,
        nonbondedMethod=NoCutoff,
        constraints=None,
        rigidWater=False,
        removeCMMotion=False,
    )
    positions = np.array(pdb.positions.value_in_unit(nanometer), dtype=np.float64)
    coordinate_box = box_rows(pdb.topology.getPeriodicBoxVectors())
    return system, positions, None, coordinate_box


def configure(system) -> list[dict]:
    """Group bonded forces (0) apart from nonbonded forces (1)."""
    from openmm import CustomNonbondedForce, NonbondedForce

    nonbonded_types = (NonbondedForce, CustomNonbondedForce)
    groups = []
    for index, force in enumerate(system.getForces()):
        if isinstance(force, nonbonded_types):
            group = 1
        else:
            group = 0
        force.setForceGroup(group)
        if isinstance(force, NonbondedForce):
            force.setUseDispersionCorrection(False)
            if force.getNonbondedMethod() == NonbondedForce.PME:
                force.setEwaldErrorTolerance(EWALD_TOLERANCE)
        # OpenMM enables a long-range tail correction on the AMBER 1-6-4
        # CustomNonbondedForce by default. The declared protocol is "no
        # dispersion correction", and the engine implements the truncated
        # -C/r^4 pair sum; leaving the correction on silently adds a ~15
        # kJ/mol constant to charged 1-6-4 fixtures and mixes a model choice
        # into a parameter/force comparison.
        if isinstance(force, CustomNonbondedForce):
            force.setUseLongRangeCorrection(False)
        groups.append({"index": index, "class": type(force).__name__, "group": group})
    return groups


def evaluate(system, positions, box, config_index):
    integrator = VerletIntegrator(0.001)
    platform = Platform.getPlatformByName("CPU")
    properties = {"Threads": "1", "DeterministicForces": "true"}
    context = Context(system, integrator, platform, properties)
    if box is not None:
        context.setPeriodicBoxVectors(*_numpy_box(box))
    context.setPositions(positions * nanometer)
    enforce = box is not None

    state = context.getState(energy=True, forces=True, enforcePeriodicBox=enforce)
    energy = state.getPotentialEnergy().value_in_unit(kilojoule_per_mole)
    forces = state.getForces(asNumpy=True).value_in_unit(
        kilojoule_per_mole / nanometer
    )
    forces = np.asarray(forces, dtype=np.float64)

    bonded = context.getState(energy=True, groups={0}, enforcePeriodicBox=enforce)
    bonded_energy = bonded.getPotentialEnergy().value_in_unit(kilojoule_per_mole)
    nonbonded = context.getState(energy=True, groups={1}, enforcePeriodicBox=enforce)
    nonbonded_energy = nonbonded.getPotentialEnergy().value_in_unit(kilojoule_per_mole)

    total = float(energy)
    bonded_energy = float(bonded_energy)
    nonbonded_energy = float(nonbonded_energy)
    split_error = abs(bonded_energy + nonbonded_energy - total)
    scale = max(abs(total), abs(bonded_energy), abs(nonbonded_energy), 1e-12)
    if not split_error <= 1e-6 * scale:
        raise RuntimeError(
            "config %d: bonded+nonbonded != total (%.3e vs %.3e)"
            % (config_index, bonded_energy + nonbonded_energy, total)
        )
    if not all(math.isfinite(v) for v in (total, bonded_energy, nonbonded_energy)):
        raise RuntimeError("config %d: non-finite energy" % config_index)
    if not np.isfinite(forces).all():
        raise RuntimeError("config %d: non-finite forces" % config_index)

    force_absmax = float(np.abs(forces).max())
    if not force_absmax > 1e-3:
        raise RuntimeError("config %d: max|F| = %.3e is not a real force field" % (config_index, force_absmax))

    return {
        "config_index": config_index,
        "energy_kj_mol": total,
        "bonded_kj_mol": bonded_energy,
        "nonbonded_kj_mol": nonbonded_energy,
        "force_norm": float(np.linalg.norm(forces)),
        "force_absmax": force_absmax,
    }, forces, split_error


def encode(forces_list) -> bytes:
    chunks = []
    for forces in forces_list:
        n_atoms = forces.shape[0]
        # The stored bytes are the flat C-order sequence of the (N,3) array:
        # x1,y1,z1,x2,y2,z2,...  Julia reads them with reshape(flat, 3, N).
        blocks = forces.reshape(3, n_atoms)  # view only; no reordering
        chunks.append(np.ascontiguousarray(blocks, dtype="<f4").tobytes(order="C"))
    return b"".join(chunks)


def resolve_sources(fixture_id: str, manifest: dict) -> dict:
    entries = {entry["id"]: entry for entry in manifest["fixtures"]}

    if fixture_id == "membrane":
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import make_generated_fixtures

        generated = make_generated_fixtures.generate()
        return {
            "kind": "amber",
            "class": "lipid membrane",
            "parameter_file": Path(generated["prmtop"]),
            "coordinate_file": Path(generated["inpcrd"]),
            "force_field": entries["membrane"]["force_field"],
            "periodic": True,
            "extra_source": {
                "generator": "make_generated_fixtures.py",
                "source_structure": entries["membrane"]["source_structure"],
                "source_structure_sha256": generated.get("source_pdb_sha256"),
                "generated_parameter_sha256": generated["prmtop_sha256"],
                "generated_coordinate_sha256": generated["inpcrd_sha256"],
            },
        }

    if fixture_id == "charmm_adapter":
        return {
            "kind": "charmm",
            "class": "CHARMM adapter evidence (not one of the five classes)",
            "psf": CHARMM_SYSTEM_DIR / "methanol_ions.psf",
            "coordinate_file": CHARMM_SYSTEM_DIR / "methanol_ions.pdb",
            "parameter_files": [
                CHARMM_SYSTEM_DIR / "par_all36_cgenff.prm",
                CHARMM_SYSTEM_DIR / "water_methanol.prm",
                CHARMM_SYSTEM_DIR / "toppar_water_ions.str",
            ],
            "force_field": "CHARMM36 CGenFF + water/methanol + water/ions toppar (OpenMM test-suite data)",
            "periodic": False,
            "extra_source": None,
        }

    entry = entries[fixture_id]
    if entry.get("status") == "blocked":
        raise SystemExit("fixture %s is blocked: %s" % (fixture_id, entry.get("blocker")))
    return {
        "kind": "amber",
        "class": entry["class"],
        "parameter_file": REPO_ROOT / entry["parameter_file"],
        "coordinate_file": REPO_ROOT / entry["coordinate_file"],
        "force_field": entry["force_field"],
        "periodic": PROTOCOLS[fixture_id],
        "extra_source": None,
    }


def make_reference(fixture_id: str, manifest: dict, force: bool = False) -> dict:
    json_path = REFERENCE_DIR / ("%s.json" % fixture_id)
    force_path = REFERENCE_DIR / ("%s.f32" % fixture_id)
    if json_path.exists() and force_path.exists() and not force:
        with open(json_path, "r", encoding="utf-8") as handle:
            record = json.load(handle)
        record["_skipped"] = True
        return record

    sources = resolve_sources(fixture_id, manifest)
    periodic = sources["periodic"]

    if sources["kind"] == "amber":
        system, positions, box, coordinate_box = build_amber(
            sources["parameter_file"], sources["coordinate_file"], periodic
        )
        parameter_files = [sources["parameter_file"]]
        coordinate_file = sources["coordinate_file"]
    else:
        system, positions, box, coordinate_box = build_charmm(
            sources["psf"], sources["coordinate_file"], sources["parameter_files"]
        )
        parameter_files = list(sources["parameter_files"])
        coordinate_file = sources["coordinate_file"]

    force_groups = configure(system)
    n_atoms = positions.shape[0]

    n_configs = CONFIG_COUNT[fixture_id]
    configs = []
    force_sets = []
    split_errors = []
    for config in range(n_configs):
        config_positions = perturbed(positions, config)
        entry, forces, split_error = evaluate(system, config_positions, box, config)
        if forces.shape != (n_atoms, 3):
            raise RuntimeError("config %d: force shape %s" % (config, forces.shape))
        configs.append(entry)
        force_sets.append(forces)
        split_errors.append(split_error)

    for config in range(1, n_configs):
        delta_energy = abs(configs[config]["energy_kj_mol"] - configs[0]["energy_kj_mol"])
        if not delta_energy > 1e-3:
            raise RuntimeError("config %d does not change the energy (%.3e)" % (config, delta_energy))

    blob = encode(force_sets)
    expected = n_configs * 3 * n_atoms * 4
    if len(blob) != expected:
        raise RuntimeError("force blob %d bytes, expected %d" % (len(blob), expected))

    if sources["kind"] == "amber":
        source_block = {
            "parameter_file": str(parameter_files[0].relative_to(REPO_ROOT))
            if parameter_files[0].is_relative_to(REPO_ROOT)
            else str(parameter_files[0]),
            "parameter_sha256": sha256(parameter_files[0]),
            "coordinate_file": str(coordinate_file.relative_to(REPO_ROOT))
            if coordinate_file.is_relative_to(REPO_ROOT)
            else str(coordinate_file),
            "coordinate_sha256": sha256(coordinate_file),
            "force_field": sources["force_field"],
        }
    else:
        relative_parameters = [str(path.relative_to(REPO_ROOT)) for path in parameter_files]
        source_block = {
            "parameter_file": " ".join(relative_parameters),
            "parameter_files": relative_parameters,
            "parameter_sha256": [sha256(path) for path in parameter_files],
            "coordinate_file": str(coordinate_file.relative_to(REPO_ROOT)),
            "coordinate_sha256": sha256(coordinate_file),
            "psf_file": str(sources["psf"].relative_to(REPO_ROOT)),
            "psf_sha256": sha256(sources["psf"]),
            "force_field": sources["force_field"],
        }
    if sources["extra_source"]:
        source_block.update(sources["extra_source"])

    protocol = {
        "nonbonded": "PME" if periodic else "NoCutoff",
        "cutoff_nm": CUTOFF_NM if periodic else None,
        "ewald_error_tolerance": EWALD_TOLERANCE if periodic else None,
        "dispersion_correction": False,
        "constraints": "none",
        "rigid_water": False,
        "remove_cmmotion": False,
    }

    record = {
        "schema": SCHEMA,
        "fixture": fixture_id,
        "class": sources["class"],
        "source": source_block,
        "generated_by": GENERATOR,
        "openmm_version": __import__("openmm").__version__,
        "platform": "CPU",
        "platform_properties": {"Threads": "1", "DeterministicForces": "true"},
        "protocol": protocol,
        "periodic": periodic,
        "box_nm": box,
        "coordinate_box_nm": coordinate_box,
        "n_atoms": n_atoms,
        "n_configs": n_configs,
        "perturbation": {
            "formula": "q[d,i] += 0.02*config*sin(0.37*(i+1) + 1.1*(d+1))",
            "index_base": "i is 0-based; d is the Cartesian component (0=x, 1=y, 2=z); config = 1,2,...",
            "applies_to": "all three components for every config >= 1; amplitude 0.02*config nm",
            "units": "nm",
            "note": (
                "config 0 is unperturbed. config 1 uses amplitude 0.02 nm, "
                "config 2 uses 0.04 nm, etc., in double precision. "
                "No coordinates are stored."
            ),
        },
        "force_groups": force_groups,
        "configs": configs,
        "force_file": "%s.f32" % fixture_id,
        "force_encoding": (
            "float32 little-endian, config-major; per config the flat C-order "
            "sequence of the OpenMM (N,3) force array (x1,y1,z1,x2,...), i.e. "
            "Julia reshape(flat,3,N)"
        ),
    }

    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    force_path.write_bytes(blob)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")

    record["_bytes"] = {"force_file": len(blob), "json": json_path.stat().st_size}
    record["_split_error_max"] = max(split_errors)
    return record


def self_check(fixture_id: str) -> dict:
    json_path = REFERENCE_DIR / ("%s.json" % fixture_id)
    force_path = REFERENCE_DIR / ("%s.f32" % fixture_id)
    with open(json_path, "r", encoding="utf-8") as handle:
        record = json.load(handle)
    n_configs = record["n_configs"]
    n_atoms = record["n_atoms"]
    expected = n_configs * 3 * n_atoms * 4
    actual = force_path.stat().st_size
    if actual != expected:
        raise RuntimeError("%s: %d bytes, expected %d" % (force_path.name, actual, expected))
    data = np.fromfile(force_path, dtype="<f4")
    if data.size != n_configs * 3 * n_atoms:
        raise RuntimeError("%s: wrong float32 count" % force_path.name)
    if not np.isfinite(data).all():
        raise RuntimeError("%s: non-finite forces" % force_path.name)
    energies = [config["energy_kj_mol"] for config in record["configs"]]
    if not all(math.isfinite(value) for value in energies):
        raise RuntimeError("%s: non-finite energy" % json_path.name)
    if not abs(energies[1] - energies[0]) > 1e-3:
        raise RuntimeError("%s: config 1 does not differ from config 0" % fixture_id)
    return {"bytes": actual, "json_bytes": json_path.stat().st_size, "energies": energies}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture",
        required=True,
        choices=sorted(CONFIG_COUNT) + ["all"],
    )
    parser.add_argument("--force", action="store_true", help="rebuild existing artifacts")
    args = parser.parse_args(argv)

    manifest = load_manifest()
    fixture_ids = (
        list(CONFIG_COUNT) if args.fixture == "all" else [args.fixture]
    )

    failures = []
    for fixture_id in fixture_ids:
        try:
            record = make_reference(fixture_id, manifest, force=args.force)
            check = self_check(fixture_id)
            print(
                "%s: N=%d configs=%d energies=%s f32=%d bytes json=%d bytes"
                % (
                    fixture_id,
                    record["n_atoms"],
                    record["n_configs"],
                    ["%.6f" % value for value in check["energies"]],
                    check["bytes"],
                    check["json_bytes"],
                )
            )
            print("  json: %s" % (REFERENCE_DIR / ("%s.json" % fixture_id)))
            print("  f32 : %s" % (REFERENCE_DIR / ("%s.f32" % fixture_id)))
            if record.get("_skipped"):
                print("  (skipped: existing artifacts; use --force to rebuild)")
        except Exception as exc:  # noqa: BLE001 - report every fixture, then fail
            failures.append((fixture_id, exc))
            print("%s: FAIL %s: %s" % (fixture_id, type(exc).__name__, exc), file=sys.stderr)

    if failures:
        print("failed fixtures: %s" % ", ".join(item[0] for item in failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
