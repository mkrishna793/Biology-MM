"""Regression checks for the reference-generation protocol itself.

The 2026-11-13 audit found two reference-side choices that made the own-engine
comparison look worse than it was and mixed a model choice into a parameter
check:

* OpenMM enables a long-range tail correction on the AMBER 1-6-4
  ``CustomNonbondedForce`` by default. With it on, the water+ions reference was
  15.4 kJ/mol more negative than the truncated ``-C/r^4`` model the engine
  implements.
* ``ewaldErrorTolerance = 5e-4`` left a 5-6 kJ/mol PME error in the
  solvated-protein/membrane references. At 1e-6 both converge and the engine
  agrees with the converged value.

These tests pin the corrected protocol without needing a network or a rebuild.
Run with the fixture Python environment:
    python -m pytest -q chemistry/tools/test_reference_protocol.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import make_reference as mr  # noqa: E402


def test_periodic_reference_protocol_flags():
    pytest.importorskip("openmm")
    from openmm import CustomNonbondedForce, NonbondedForce

    manifest = mr.load_manifest()
    sources = mr.resolve_sources("water_ions", manifest)
    system, positions, box, _ = mr.build_amber(
        sources["parameter_file"], sources["coordinate_file"], periodic=True
    )
    mr.configure(system)
    assert mr.EWALD_TOLERANCE == 1.0e-6
    for force in system.getForces():
        if isinstance(force, NonbondedForce):
            assert force.getUseDispersionCorrection() is False
            assert force.getEwaldErrorTolerance() == pytest.approx(1.0e-6)
        if isinstance(force, CustomNonbondedForce):
            assert force.getUseLongRangeCorrection() is False
    assert positions.shape[0] == 2164


def test_reference_metadata_matches_corrected_protocol():
    ref = Path(mr.REFERENCE_DIR)
    water = json.loads((ref / "water_ions.json").read_text())
    assert water["protocol"]["ewald_error_tolerance"] == pytest.approx(1.0e-6)
    assert water["protocol"]["dispersion_correction"] is False
    # The corrected 1-6-4-truncated reference energy at the source coordinates.
    assert water["configs"][0]["energy_kj_mol"] == pytest.approx(-25746.3917, abs=0.05)

    ligand_path = ref / "protein_ligand.json"
    assert ligand_path.is_file(), "protein_ligand reference artifact missing"
    ligand = json.loads(ligand_path.read_text())
    assert ligand["n_atoms"] == 2615
    assert ligand["periodic"] is False
    assert ligand["protocol"]["nonbonded"] == "NoCutoff"


def test_ligand_fixture_is_registered_ready():
    manifest = mr.load_manifest()
    entries = {entry["id"]: entry for entry in manifest["fixtures"]}
    entry = entries["protein_ligand"]
    assert entry["status"] == "ready"
    assert Path(entry["parameter_file"]).is_file()
    assert Path(entry["coordinate_file"]).is_file()
