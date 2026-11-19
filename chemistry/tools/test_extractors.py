"""Extractor tests for the Phase-2 normalized chemistry record.

Run from the repository root with the isolated interpreter:

    ~/.cache/biology-mm/.venv/bin/python -m pytest -q \
        chemistry/tools/test_extractors.py

The tests never write into the repository and never use the network: every
output goes into pytest's ``tmp_path``.
"""

from __future__ import annotations

import copy
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parent
REPO = TOOLS.parents[1]
SYSTEMS = REPO / "wrappers" / "python" / "tests" / "systems"

sys.path.insert(0, str(TOOLS))

import common  # noqa: E402
import extract_amber  # noqa: E402
import extract_charmm  # noqa: E402


def _systems(name: str) -> str:
    path = SYSTEMS / name
    assert path.is_file(), f"missing repository fixture {path}"
    return str(path)


def _json(path) -> dict:
    with open(path, "r", encoding="ascii") as fh:
        return json.load(fh)


def _by_pair(terms):
    return {(min(t["i"], t["j"]), max(t["i"], t["j"])): t for t in terms}


def _term_map(record):
    out = {}
    for d in record["dihedrals"]:
        out[(d["i"], d["j"], d["k"], d["l"])] = d["terms"]
    return out


# ---------------------------------------------------------------------------
# AMBER
# ---------------------------------------------------------------------------

def test_mg_water_extraction(tmp_path):
    parm = _systems("Mg_water.prmtop")
    crd = _systems("Mg_water.inpcrd")
    record = extract_amber.extract(parm, crd, "test water/ions")
    common.validate_record(record)

    assert set(record) == set(common.TOP_KEYS)
    assert record["schema"] == "1.0.0"
    assert record["family"] == "amber"
    assert record["units"] == "nm-kj-mol-e-amu"
    assert len(record["atoms"]) == 2164
    assert record["dihedrals"] == []
    assert record["pair14"] == []

    oxygen = next(a for a in record["atoms"]
                  if a["element"] == "O" and a["name"].strip() == "O")
    assert abs(oxygen["charge"] + 0.834) <= 1e-6
    assert oxygen["sigma_nm"] > 0.0 and oxygen["epsilon_kj"] > 0.0

    mg = record["atoms"][0]
    assert mg["sigma_nm"] > 0.0 and mg["epsilon_kj"] > 0.0

    codes = [d["code"] for d in record["input_diagnostics"]]
    assert "box_mismatch" in codes
    mismatch = next(d for d in record["input_diagnostics"]
                    if d["code"] == "box_mismatch")
    assert mismatch["severity"] == "warning"
    assert "differs from coordinate box" in mismatch["message"]
    assert mismatch["atoms"] == []

    # The output file is deterministic and validates after a read.
    out = tmp_path / "mg.json"
    common.write_record(record, str(out))
    assert common.read_record(str(out)) == record
    text = out.read_text(encoding="ascii")
    assert text == json.dumps(record, sort_keys=True,
                              separators=(",", ":")) + "\n"


def test_amber_bond_and_dihedral_coefficients(tmp_path, monkeypatch):
    """k is the coefficient of OUR factor, not the native prmtop key."""
    monkeypatch.setenv(common.ALLOW_NBFIX_ENV, "1")
    from parmed.amber import AmberParm, ChamberParm

    parm = _systems("Mg_water.prmtop")
    crd = _systems("Mg_water.inpcrd")
    struct = AmberParm(parm, crd)
    record = extract_amber.extract(parm, crd, "test water/ions")
    json_bonds = _by_pair(record["bonds"])
    for bond in struct.bonds:
        key = (min(bond.atom1.idx + 1, bond.atom2.idx + 1),
               max(bond.atom1.idx + 1, bond.atom2.idx + 1))
        expected = 2.0 * float(bond.type.k) * common.BOND_FACTOR
        assert key in json_bonds
        assert math.isclose(json_bonds[key]["k"], expected,
                            rel_tol=1e-12, abs_tol=0.0)

    # ala3_solv is a ChamberParm with multi-term torsions.  phi_k is already
    # the coefficient of k*(1+cos(n phi - phase)); only kcal -> kJ applies.
    ala_parm = _systems("ala3_solv.parm7")
    ala_crd = _systems("ala3_solv.crd")
    ala = ChamberParm(ala_parm, ala_crd)
    record = extract_amber.extract(ala_parm, ala_crd, "test chamber")
    actual = _term_map(record)

    expected = defaultdict(list)
    for d in ala.dihedrals:
        key = (d.atom1.idx + 1, d.atom2.idx + 1,
               d.atom3.idx + 1, d.atom4.idx + 1)
        expected[key].append({
            "k": float(d.type.phi_k) * common.KCAL_TO_KJ,
            "n": int(d.type.per),
            "phase_rad": math.radians(float(d.type.phase)),
        })
    assert set(actual) == set(expected)
    nonzero = 0
    for key, terms in expected.items():
        assert len(actual[key]) == len(terms)
        for got, want in zip(actual[key], terms):
            assert math.isclose(got["k"], want["k"], rel_tol=1e-12,
                                abs_tol=0.0)
            assert got["n"] == want["n"]
            assert math.isclose(got["phase_rad"], want["phase_rad"],
                                rel_tol=1e-12, abs_tol=1e-15)
            nonzero += got["k"] != 0.0
    assert nonzero > 0


def test_ala3_multiplicity_impropers_and_determinism(tmp_path, monkeypatch):
    monkeypatch.setenv(common.ALLOW_NBFIX_ENV, "1")
    parm = _systems("ala3_solv.parm7")
    crd = _systems("ala3_solv.crd")
    record = extract_amber.extract(parm, crd, "test chamber")
    common.validate_record(record)

    assert len(record["atoms"]) == 2776
    assert len(record["impropers"]) == 5
    assert len(record["dihedrals"]) == 74
    assert sum(len(d["terms"]) for d in record["dihedrals"]) == 76
    assert sum(len(d["terms"]) for d in record["dihedrals"]) > len(
        record["dihedrals"])
    # The fixture has one CHARMM CMAP; it must be carried, not dropped.
    assert len(record["cmaps"]) == 1
    assert record["cmaps"][0]["resolution"] == 24
    assert len(record["cmaps"][0]["grid"]) == 24 * 24

    # NBFIX is refused by default and only downgraded to a warning on request.
    assert any(d["code"] == "nbfix_ignored"
               for d in record["input_diagnostics"])

    excl = {tuple(p) for p in record["exclusions"]}
    p14 = {(p["i"], p["j"]) for p in record["pair14"]}
    assert excl.isdisjoint(p14)
    assert len(excl) == len(record["exclusions"])
    assert len(p14) == len(record["pair14"])

    first = tmp_path / "ala3-a.json"
    second = tmp_path / "ala3-b.json"
    common.write_record(record, str(first))
    common.write_record(common.read_record(str(first)), str(second))
    assert first.read_bytes() == second.read_bytes()
    expected = json.dumps(record, sort_keys=True,
                          separators=(",", ":")) + "\n"
    assert first.read_text(encoding="ascii") == expected


def test_dna_exclusions_and_pair14(tmp_path):
    parm = _systems("DNA_mbondi3.prmtop")
    crd = _systems("DNA_mbondi3.inpcrd")
    record = extract_amber.extract(parm, crd, "test DNA")
    common.validate_record(record)

    assert len(record["atoms"]) == 628
    assert record["box"]["kind"] == "none"
    assert abs(record["net_charge"] + 18.0) <= 1e-5

    excl = [tuple(p) for p in record["exclusions"]]
    p14 = [(p["i"], p["j"]) for p in record["pair14"]]
    assert len(set(excl)) == len(excl)
    assert len(set(p14)) == len(p14)
    assert set(excl).isdisjoint(p14)

    # 2500 is the number of DIHEDRALS entries; the unique proper 1-4 pair
    # count is 1578.  exclusions + pair14 must partition all OpenMM
    # NonbondedForce exceptions (3472 for this fixture).
    assert len(p14) == 1578
    assert len(excl) + len(p14) == 3472


# ---------------------------------------------------------------------------
# CMAP convention
# ---------------------------------------------------------------------------

def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _torsion(points, i, j, k, l):
    """IUPAC signed torsion, identical to Factors.dihedral_angle_grad."""
    b1 = tuple(points[j][d] - points[i][d] for d in range(3))
    b2 = tuple(points[k][d] - points[j][d] for d in range(3))
    b3 = tuple(points[l][d] - points[k][d] for d in range(3))
    n1 = _cross(b1, b2)
    n2 = _cross(b2, b3)
    m = _cross(n1, n2)
    y = _dot(m, b2) / math.sqrt(_dot(b2, b2))
    return math.atan2(y, _dot(n1, n2))


def _openmm_shift(raw, n):
    """amber_file_parser's CMAP permutation: M[s+n*t] = P[(s+h),(t+h)]."""
    h = n // 2
    return [raw[((s + h) % n) * n + ((t + h) % n)]
            for t in range(n) for s in range(n)]


def _openmm_cmap_energy(values, n, points):
    import openmm as mm

    force = mm.CMAPTorsionForce()
    index = force.addMap(n, values)
    # (a,b,c,d,b,c,d,e): the eight-slot CMAP support repeats b,c,d.
    force.addTorsion(index, 0, 1, 2, 3, 1, 2, 3, 4)
    system = mm.System()
    for _ in range(5):
        system.addParticle(1.0)
    system.addForce(force)
    names = [p.getName() for p in
             (mm.Platform.getPlatform(i)
              for i in range(mm.Platform.getNumPlatforms()))]
    platform = mm.Platform.getPlatformByName("CPU" if "CPU" in names
                                             else "Reference")
    context = mm.Context(system, mm.VerletIntegrator(0.01), platform)
    context.setPositions([list(p) for p in points])
    return context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        mm.unit.kilojoule_per_mole)


def test_cmap_analytic_map_uses_positive_signed_torsion():
    """A map E=cos(phi)+0.5*sin(psi) must be read at (+phi,+psi).

    We build the raw prmtop grid in the schema's -pi..pi order, apply the
    OpenMM AMBER permutation, and evaluate OpenMM at geometries whose signed
    torsions are known analytically.  The un-reversed grid matches E(phi,psi);
    the reversed grid does not.  This is the empirical check behind the
    convention documented in extract_amber._cmaps.
    """
    pytest.importorskip("openmm")
    n = 36
    delta = 2.0 * math.pi / n

    def energy(phi, psi):
        return math.cos(phi) + 0.5 * math.sin(psi)

    raw = [[energy(-math.pi + a * delta, -math.pi + b * delta)
            for b in range(n)] for a in range(n)]
    flat = [raw[a][b] for a in range(n) for b in range(n)]
    openmm_map = _openmm_shift(flat, n)

    reversed_raw = [[raw[(n - 1 - a) % n][(n - 1 - b) % n]
                     for b in range(n)] for a in range(n)]
    reversed_flat = [reversed_raw[a][b] for a in range(n) for b in range(n)]
    reversed_map = _openmm_shift(reversed_flat, n)

    geometries = [
        [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
         [1.0, 0.0, 1.0], [0.5, -0.5, 1.0]],
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0],
         [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]],
        [[0.0, 0.0, 0.0], [1.0, 0.2, 0.1], [0.9, 1.0, -0.2],
         [1.1, 1.2, 0.9], [-0.3, 0.8, 1.1]],
    ]
    saw_reversal_difference = False
    for points in geometries:
        phi = _torsion(points, 0, 1, 2, 3)
        psi = _torsion(points, 1, 2, 3, 4)
        expected = energy(phi, psi)
        got = _openmm_cmap_energy(openmm_map, n, points)
        assert abs(got - expected) < 2e-3
        rev = _openmm_cmap_energy(reversed_map, n, points)
        if abs(rev - expected) > 0.1:
            saw_reversal_difference = True
    assert saw_reversal_difference, "reversal test did not distinguish the sign"


def test_18protein_cmap_matches_openmm_map(tmp_path):
    parm = _systems("18protein.parm7")
    crd = _systems("18protein.rst7")
    record = extract_amber.extract(parm, crd, "test solvated protein")
    assert len(record["cmaps"]) == 16

    from parmed.amber import AmberParm
    struct = AmberParm(parm)
    saw_permutation = False
    for k, cmap in enumerate(record["cmaps"]):
        source = struct.cmaps[k]
        n = int(source.type.resolution)
        assert cmap["resolution"] == n
        assert len(cmap["grid"]) == n ** 2
        raw = [float(x) * common.KCAL_TO_KJ for x in source.type.grid]
        expected = _openmm_shift(raw, n)
        assert cmap["grid"] == pytest.approx(expected, rel=0.0, abs=1e-12)
        if any(abs(a - b) > 1e-9 for a, b in zip(cmap["grid"], raw)):
            saw_permutation = True
        atoms = [source.atom1.idx + 1, source.atom2.idx + 1,
                 source.atom3.idx + 1, source.atom4.idx + 1,
                 source.atom5.idx + 1]
        assert cmap["i"] == [atoms[0], atoms[1], atoms[2], atoms[3],
                             atoms[1], atoms[2], atoms[3], atoms[4]]
        assert len(set(cmap["i"])) == 5
    assert saw_permutation, "the AMBER half-grid permutation was not applied"

    # Print the count/first values for the run report (pytest -q keeps it).
    first = record["cmaps"][0]
    print("18protein cmaps:", len(record["cmaps"]),
          "resolution:", first["resolution"],
          "grid0..2:", first["grid"][:3],
          "i:", first["i"])



def test_amber_cli_contract(tmp_path):
    parm = _systems("DNA_mbondi3.prmtop")
    crd = _systems("DNA_mbondi3.inpcrd")
    out = tmp_path / "dna.json"
    result = subprocess.run(
        [sys.executable, str(TOOLS / "extract_amber.py"),
         "--parm", parm, "--crd", crd, "--out", str(out),
         "--force-field", "test DNA", "--repo-root", str(REPO)],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert result.stdout.strip() == str(out)
    record = _json(out)
    assert record["provenance"]["source"] == (
        "wrappers/python/tests/systems/DNA_mbondi3.prmtop")
    assert len(record["provenance"]["source_sha256"]) == 64


# ---------------------------------------------------------------------------
# CHARMM
# ---------------------------------------------------------------------------

def _charmm_inputs():
    psf = _systems("methanol_ions.psf")
    params = [
        _systems("par_all36_cgenff.prm"),
        _systems("water_methanol.prm"),
        _systems("toppar_water_ions.str"),
    ]
    pdb = _systems("methanol_ions.pdb")
    return psf, params, pdb


def test_charmm_methanol_adapter(tmp_path, monkeypatch):
    monkeypatch.setenv(common.ALLOW_NBFIX_ENV, "1")
    psf, params, pdb = _charmm_inputs()
    record = extract_charmm.extract(psf, params, pdb, "test CHARMM")
    common.validate_record(record)

    assert len(record["atoms"]) == 8
    assert len(record["urey_bradleys"]) == 3
    assert record["box"]["kind"] == "orthorhombic"
    assert record["box"]["lengths_nm"] == pytest.approx(
        [1.2009, 1.2338, 1.1510])
    assert any(d["code"] == "nbfix_ignored"
               for d in record["input_diagnostics"])

    out = tmp_path / "methanol.json"
    common.write_record(record, str(out))
    assert common.read_record(str(out)) == record


def test_charmm_nbfix_is_refused_by_default(monkeypatch):
    monkeypatch.delenv(common.ALLOW_NBFIX_ENV, raising=False)
    psf, params, pdb = _charmm_inputs()
    with pytest.raises(common.ExtractionError) as info:
        extract_charmm.extract(psf, params, pdb, "test CHARMM")
    assert info.value.code == "unsupported_term"
    assert "NBFIX" in info.value.message


def test_charmm_coefficient_conventions(monkeypatch):
    monkeypatch.setenv(common.ALLOW_NBFIX_ENV, "1")
    from parmed.charmm import CharmmParameterSet, CharmmPsfFile

    psf_path, params, pdb = _charmm_inputs()
    psf = CharmmPsfFile(psf_path)
    psf.load_parameters(CharmmParameterSet(*params))
    record = extract_charmm.extract(psf_path, params, pdb, "test CHARMM")

    json_bonds = _by_pair(record["bonds"])
    for bond in psf.bonds:
        key = (min(bond.atom1.idx + 1, bond.atom2.idx + 1),
               max(bond.atom1.idx + 1, bond.atom2.idx + 1))
        assert math.isclose(json_bonds[key]["k"],
                            2.0 * float(bond.type.k) * common.BOND_FACTOR,
                            rel_tol=1e-12, abs_tol=0.0)

    json_angles = {(a["i"], a["j"], a["k"]): a for a in record["angles"]}
    for angle in psf.angles:
        key = (angle.atom1.idx + 1, angle.atom2.idx + 1,
               angle.atom3.idx + 1)
        assert math.isclose(json_angles[key]["k_theta"],
                            2.0 * float(angle.type.k) * common.ANGLE_FACTOR,
                            rel_tol=1e-12, abs_tol=0.0)

    json_ub = _by_pair(record["urey_bradleys"])
    for ub in psf.urey_bradleys:
        key = (min(ub.atom1.idx + 1, ub.atom2.idx + 1),
               max(ub.atom1.idx + 1, ub.atom2.idx + 1))
        assert math.isclose(json_ub[key]["k"],
                            float(ub.type.k) * common.BOND_FACTOR,
                            rel_tol=1e-12, abs_tol=0.0)

    json_dihedrals = _term_map(record)
    for d in psf.dihedrals:
        key = (d.atom1.idx + 1, d.atom2.idx + 1,
               d.atom3.idx + 1, d.atom4.idx + 1)
        terms = list(d.type) if not hasattr(d.type, "phi_k") else [d.type]
        assert len(json_dihedrals[key]) == len(terms)
        for got, want in zip(json_dihedrals[key], terms):
            assert math.isclose(got["k"],
                                float(want.phi_k) * common.KCAL_TO_KJ,
                                rel_tol=1e-12, abs_tol=0.0)
            assert got["n"] == int(want.per)


# ---------------------------------------------------------------------------
# shared validator
# ---------------------------------------------------------------------------

def test_validator_rejects_exclusion_pair14_overlap(tmp_path, monkeypatch):
    monkeypatch.setenv(common.ALLOW_NBFIX_ENV, "1")
    record = extract_amber.extract(
        _systems("ala3_solv.parm7"), _systems("ala3_solv.crd"), "test")
    tampered = copy.deepcopy(record)
    pair = tampered["pair14"][0]
    tampered["exclusions"].append([pair["i"], pair["j"]])
    with pytest.raises(common.ValidationError) as info:
        common.validate_record(tampered)
    assert "both an exclusion and a 1-4" in str(info.value)


def test_validator_rejects_bad_charge_sum(tmp_path, monkeypatch):
    monkeypatch.setenv(common.ALLOW_NBFIX_ENV, "1")
    record = extract_amber.extract(
        _systems("ala3_solv.parm7"), _systems("ala3_solv.crd"), "test")
    tampered = copy.deepcopy(record)
    tampered["net_charge"] += 0.5
    with pytest.raises(common.ValidationError) as info:
        common.validate_record(tampered)
    assert "charges sum" in str(info.value)
