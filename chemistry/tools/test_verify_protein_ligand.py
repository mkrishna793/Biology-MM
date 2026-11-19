"""Offline pytest gate for the ``protein_ligand`` fixture verifier.

Run from the repository root:

    ~/.cache/biology-mm/.venv/bin/python -m pytest -q \
        chemistry/tools/test_verify_protein_ligand.py

Nothing here touches the network or rewrites the fixture: every assertion is
computed from the committed prmtop/inpcrd/pdb and the committed OpenMM
reference.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

TOOLS_DIR = Path(__file__).resolve().parent


def _load_verifier():
    spec = importlib.util.spec_from_file_location(
        "verify_protein_ligand", TOOLS_DIR / "verify_protein_ligand.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


verify = _load_verifier()


@pytest.fixture(scope="module")
def report() -> dict:
    pytest.importorskip("openmm")
    return verify.run_verification()


def test_report_passes(report: dict) -> None:
    failures = [check for check in report["checks"] if not check["ok"]]
    assert report["ok"] is True, failures
    assert len(report["checks"]) >= 35


def test_expected_checks_are_present(report: dict) -> None:
    names = {check["check"] for check in report["checks"]}
    required = {
        "parmed:atom-count",
        "ligand:single-UNL",
        "ligand:atom-count",
        "ligand:elements",
        "charge:net",
        "pdb:atom-names-order",
        "geometry:bond-length-median",
        "geometry:bond-length-max",
        "geometry:ligand-protein-min-heavy-distance",
        "ligand:ring",
        "ligand:one-cycle",
        "parameters:ff14sb-charges",
        "config0:energy-vs-reference",
        "config0:max-force-vs-reference",
        "config0:f32-vs-double",
    }
    assert required <= names


def test_atom_and_ligand_counts(report: dict) -> None:
    values = report["values"]
    assert values["n_atoms"] == 2615
    ligand = values["ligand"]
    assert ligand["residue"] == "UNL"
    assert ligand["n_atoms"] == 14
    assert ligand["elements"] == {"C": 6, "H": 6, "O": 2}


def test_net_charge(report: dict) -> None:
    assert abs(report["values"]["net_charge_e"] - 8.0) <= 0.01


def test_geometry_envelope(report: dict) -> None:
    bonds = report["values"]["bonds"]
    assert 0.09 <= bonds["median_nm"] <= 0.15
    assert bonds["max_nm"] < 0.20
    assert 0.14 <= report["values"]["ligand_protein_min_heavy_distance_nm"] <= 0.35


def test_ligand_graph(report: dict) -> None:
    graph = report["values"]["ligand_graph"]
    assert graph["bonds"] == 14
    assert graph["cycle_rank"] == 1
    assert graph["carbons_in_ring"] == 6


def test_protein_parameters_are_ff14sb(report: dict) -> None:
    residues = report["values"]["ff14sb_residues"]
    assert residues["compared"] >= 150
    assert residues["matched"] == residues["compared"]
    assert residues["max_charge_delta_e"] < 1e-4


def test_config0_matches_reference(report: dict) -> None:
    config0 = report["values"]["config0"]
    assert config0["energy_delta_kj_mol"] <= verify.ENERGY_TOL_KJMOL
    assert config0["max_force_delta_kj_mol_per_nm"] <= verify.FORCE_TOL_KJMOL_PER_NM
    assert report["values"]["config0_f32"]["max_abs_delta_kj_mol_per_nm"] <= (
        verify.FORCE_TOL_KJMOL_PER_NM
    )


def test_hashes_match_reference_record(report: dict) -> None:
    hashes = report["values"]["sha256"]
    assert set(hashes) == {
        "protein_ligand.prmtop",
        "protein_ligand.inpcrd",
        "protein_ligand.pdb",
    }
    for digest in hashes.values():
        assert len(digest) == 64
