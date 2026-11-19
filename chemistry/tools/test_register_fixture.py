"""Offline tests for the fixture registration workflow.

    python -m pytest -q chemistry/tools/test_register_fixture.py

Everything is written under pytest's ``tmp_path``.  The extractor is never run
(``--no-extract`` with an existing normalized record), so no ParmEd, no network
and no writes into the repository.  The Julia leg is skipped when no Julia is
installed; it is only a reader of the record via ``FoundationCore.Chemistry``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parent
REPO = TOOLS.parents[1]
PRMTOP = "wrappers/python/tests/systems/Mg_water.prmtop"
INPCRD = "wrappers/python/tests/systems/Mg_water.inpcrd"
JULIA_FALLBACK = Path.home() / ".cache/biology-mm/julia-1.13.3/bin/julia"

sys.path.insert(0, str(TOOLS))
import common  # noqa: E402
import register_fixture  # noqa: E402


def methanol_record() -> dict:
    """A 6-atom methanol record: bonds, angles, a 2-term torsion, 1-4 pairs."""
    atoms = [
        ("C1", 12.011, -0.2, "C"), ("O1", 15.999, -0.6, "O"),
        ("H1", 1.008, 0.1, "H"), ("H2", 1.008, 0.1, "H"),
        ("H3", 1.008, 0.1, "H"), ("HO", 1.008, 0.5, "H"),
    ]
    return {
        "schema": "1.0.0", "family": "amber", "units": "nm-kj-mol-e-amu",
        "provenance": {"source": "synthetic/methanol.prmtop",
                       "force_field": "synthetic 6-atom methanol (test only)",
                       "source_sha256": "0" * 64,
                       "coordinate_source": "synthetic/methanol.inpcrd",
                       "coordinate_sha256": "1" * 64,
                       "generator": "chemistry/tools/test_register_fixture.py",
                       "generator_version": "1.0.0", "notes": "synthetic"},
        "atoms": [{"name": n, "mass": m, "charge": q, "element": e, "residue_index": 1,
                   "residue_name": "MOH", "sigma_nm": 0.3, "epsilon_kj": 0.4}
                  for n, m, q, e in atoms],
        "box": {"kind": "none"},
        "bonds": [{"i": 1, "j": 2, "k": 251040.0, "r0_nm": 0.143},
                  {"i": 1, "j": 3, "k": 276144.0, "r0_nm": 0.109},
                  {"i": 1, "j": 4, "k": 276144.0, "r0_nm": 0.109},
                  {"i": 1, "j": 5, "k": 276144.0, "r0_nm": 0.109},
                  {"i": 2, "j": 6, "k": 276144.0, "r0_nm": 0.096}],
        "angles": [{"i": 2, "j": 1, "k": 3, "k_theta": 418.4, "theta0_rad": 1.9111},
                   {"i": 2, "j": 1, "k": 4, "k_theta": 418.4, "theta0_rad": 1.9111},
                   {"i": 2, "j": 1, "k": 5, "k_theta": 418.4, "theta0_rad": 1.9111},
                   {"i": 1, "j": 2, "k": 6, "k_theta": 418.4, "theta0_rad": 1.8925}],
        "urey_bradleys": [],
        "dihedrals": [{"i": 3, "j": 1, "k": 2, "l": 6,
                       "terms": [{"k": 1.0, "n": 3, "phase_rad": 0.0},
                                 {"k": 0.4, "n": 1, "phase_rad": 3.141592653589793}]}],
        "impropers": [], "cmaps": [],
        "exclusions": [[1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [2, 3], [2, 4],
                       [2, 5], [2, 6], [3, 4], [3, 5], [4, 5]],
        "pair14": [{"i": 3, "j": 6, "coulomb_scale": 5 / 6, "lj_scale": 0.5},
                   {"i": 4, "j": 6, "coulomb_scale": 5 / 6, "lj_scale": 0.5},
                   {"i": 5, "j": 6, "coulomb_scale": 5 / 6, "lj_scale": 0.5}],
        "lj146": [], "virtual_sites": [], "constraints": [],
        "input_diagnostics": [], "net_charge": 0.0,
    }


def write_spec(tmp_path: Path, **overrides) -> Path:
    spec = {"id": "synthetic_methanol_probe", "class": "water+ions", "family": "amber",
            "parameter_file": PRMTOP, "coordinate_file": INPCRD,
            "force_field": "synthetic record, not a force field", "status": "ready"}
    spec.update(overrides)
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    return path


def run_cli(spec_path: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(TOOLS / "register_fixture.py"), "--spec", str(spec_path),
         "--no-extract", *extra], capture_output=True, text=True)


def julia() -> str | None:
    for candidate in (os.environ.get("BIOLOGY_MM_JULIA"),
                      os.environ.get("FOUNDATION_JULIA"),
                      os.environ.get("JULIA"),
                      shutil.which("julia"),
                      str(JULIA_FALLBACK)):
        if candidate and Path(candidate).is_file():
            return candidate
    return None


def run_julia(script: str, *args: str) -> subprocess.CompletedProcess:
    binary = julia()
    if binary is None:
        pytest.skip("no Julia binary; the ResolvedSystem leg is BLOCKED, not passed")
    env = dict(os.environ)
    return subprocess.run([binary, "--project=.", "-e", script, *args],
                          cwd=REPO / "julia", env=env,
                          capture_output=True, text=True, timeout=600)


def test_valid_registration_reports_terms_and_capabilities(tmp_path):
    record = methanol_record()
    common.validate_record(record)
    normalized = tmp_path / "methanol.json"
    normalized.write_text(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
    proc = run_cli(write_spec(tmp_path), "--normalized", str(normalized), "--json")
    assert proc.returncode == 0, proc.stderr
    report = json.loads(proc.stdout)
    assert report["counts"]["atoms"] == 6
    assert report["counts"]["dihedral_terms"] == 2
    assert report["counts"]["pair14"] == 3
    assert report["hashes"] == {"parameter_file": "not-run", "coordinate_file": "not-run"}
    assert report["term_capabilities"]["nbfix"]["status"] == "refused"
    assert report["reference"]["exists"] is False
    assert report["consumer_extraction"] == "ok"


def test_roundtrip_builds_resolved_system(tmp_path):
    normalized = tmp_path / "methanol.json"
    normalized.write_text(json.dumps(methanol_record(), sort_keys=True) + "\n")
    script = ("using FoundationCore; const CH = FoundationCore.Chemistry; "
              "const RS = FoundationCore.ResolvedSystem; "
              "ns = CH.read_normalized(ARGS[1]); rs = CH.build_resolved(ns); "
              'println("OK ", CH.n_atoms(ns), " ", RS.n_atoms(rs), " ", length(rs.factors))')
    proc = run_julia(script, str(normalized))
    assert proc.returncode == 0, proc.stderr
    tag, atoms, resolved, factors = proc.stdout.split()
    assert (tag, int(atoms), int(resolved)) == ("OK", 6, 6)
    assert int(factors) > 0


def test_triclinic_pme_build_and_small_cutoff_refusal(tmp_path):
    # Valid triclinic cell with min(widths) > 2 * cutoff_nm builds and evaluates PME + box_derivative
    record = methanol_record()
    record["box"] = {"kind": "triclinic",
                     "vectors_nm": [[3.0, 0.0, 0.0], [0.3, 3.0, 0.0], [0.1, 0.2, 3.0]]}
    normalized = tmp_path / "triclinic_ok.json"
    normalized.write_text(json.dumps(record, sort_keys=True) + "\n")
    script = (
        "using FoundationCore; "
        "const CH = FoundationCore.Chemistry; "
        "const RS = FoundationCore.ResolvedSystem; "
        "const SC = FoundationCore.SystemContract; "
        "const T = FoundationCore.Types; "
        "ns = CH.read_normalized(ARGS[1]); "
        "prot = CH.default_protocol(ns; lj_mode = :lfs); "
        "rs = CH.build_resolved(ns; protocol = prot); "
        "@assert rs.box isa T.TriclinicBox; "
        "@assert :box_derivative in RS.engine_capabilities(rs); "
        "q = [0.10 0.24 0.06 0.06 0.06 0.32; 0.10 0.10 0.20 0.05 0.05 0.12; 0.10 0.10 0.10 0.20 0.00 0.15]; "
        "U, F = SC.energy_forces(rs, q); "
        "G = SC.box_derivative(rs, q); "
        "@assert isfinite(U) && all(isfinite, F) && all(isfinite, G); "
        "S = SC.fractional_coordinates(rs, q); "
        "H = copy(rs.box.H); eps = 1e-6; max_err = 0.0; "
        "for a in 1:3, b in 1:3; "
        "  Hp = copy(H); Hp[a, b] += eps; Hm = copy(H); Hm[a, b] -= eps; "
        "  rsp = RS.restart_system(rs; box = T.TriclinicBox(Hp)); "
        "  rsm = RS.restart_system(rs; box = T.TriclinicBox(Hm)); "
        "  fd = (SC.energy(rsp, Hp * S) - SC.energy(rsm, Hm * S)) / (2 * eps); "
        "  global max_err = max(max_err, abs(G[a, b] - fd)); "
        "end; "
        "@assert max_err < 1e-6 * max(1.0, maximum(abs, G)); "
        'println("OK ", max_err)'
    )
    proc = run_julia(script, str(normalized))
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().startswith("OK ")

    # Small triclinic cell where default cutoff_nm = 1.0 >= 0.5 * min(width) refuses as :protocol
    record_small = methanol_record()
    record_small["box"] = {"kind": "triclinic",
                           "vectors_nm": [[2.0, 0.0, 0.0], [0.3, 2.0, 0.0], [0.1, 0.2, 2.0]]}
    normalized_small = tmp_path / "triclinic_small.json"
    normalized_small.write_text(json.dumps(record_small, sort_keys=True) + "\n")
    script_small = (
        "using FoundationCore; const CH = FoundationCore.Chemistry; "
        "try CH.build_resolved(CH.read_normalized(ARGS[1])); println(\"NO-REFUSAL\") "
        "catch e; println(\"REFUSAL \", e isa CH.ChemistryError ? e.code : typeof(e)); end"
    )
    proc_small = run_julia(script_small, str(normalized_small))
    assert proc_small.returncode == 0, proc_small.stderr
    assert proc_small.stdout.strip() == "REFUSAL protocol"


@pytest.mark.parametrize("overrides, code", [
    ({"bogus_key": 1}, "schema"),
    ({"id": "water_ions"}, "duplicate"),
    ({"parameter_file": "no/such/file.prmtop"}, "missing_file"),
    ({"parameter_file": "/etc/passwd"}, "path"),
    ({"id": "Bad-Id"}, "id"),
])
def test_spec_refusals(tmp_path, overrides, code):
    proc = run_cli(write_spec(tmp_path, **overrides))
    assert proc.returncode == 2
    assert "error[%s]" % code in proc.stderr


def test_bad_hash_is_refused(tmp_path):
    record = methanol_record()
    record["provenance"]["source_sha256"] = "0" * 64
    spec = {"parameter_file": PRMTOP, "coordinate_file": INPCRD}
    with pytest.raises(register_fixture.RegistrationError) as excinfo:
        register_fixture.verify_hashes(record, spec)
    assert excinfo.value.code == "bad_hash"
    record["provenance"]["source_sha256"] = "not-a-hash"
    with pytest.raises(register_fixture.RegistrationError) as excinfo:
        register_fixture.verify_hashes(record, spec)
    assert excinfo.value.code == "bad_hash"
