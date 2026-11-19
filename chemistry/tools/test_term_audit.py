#!/usr/bin/env python3
"""Gate for the per-class / per-term audit of the Phase-2 fixtures.

Run with ``pytest chemistry/tools/test_term_audit.py -s`` or directly
``python chemistry/tools/test_term_audit.py``.  It audits ``dna``,
``protein_ligand`` and ``water_ions`` (periodic, so only the per-class and PME
checks apply) plus ``charmm_adapter`` when its OpenMM test-suite inputs are
present.  ``membrane`` and ``solvated_protein`` are exercised by
``term_audit.py --fixture all``; set ``TERM_AUDIT_ALL=1`` to include them here.

Every assertion prints the measured number, so a failure shows how far off the
value was instead of only which threshold was crossed.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

pytest.importorskip("openmm")

TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import term_audit as ta  # noqa: E402

# Thresholds from the audit contract: class energies are compared at 1e-5
# relative, the total at 1e-3, and the class-sum identity at 1e-8.  Force
# vectors are compared at 1e-4 because OpenMM's nonbonded kernel evaluates the
# pair math in single precision (measured floor: 7.8e-7 on one LJ pair).
CLASS_TOL = 1e-5
FORCE_TOL = 1e-4
TOTAL_TOL = 1e-3
SUM_TOL = 1e-8

REQUIRED = ("dna", "protein_ligand", "water_ions", "charmm_adapter")
OPTIONAL = ("membrane", "solvated_protein")
AUDITED = [fid for fid in REQUIRED if ta.available(fid)]
if os.environ.get("TERM_AUDIT_ALL") == "1":
    AUDITED += [fid for fid in OPTIONAL if ta.available(fid)]
SKIPPED = [fid for fid in REQUIRED + OPTIONAL if fid not in AUDITED]

_CACHE = {}
for _fid in SKIPPED:
    print("skipping %s: fixture inputs are absent" % _fid)


def audit_of(fixture_id: str) -> dict:
    if fixture_id not in _CACHE:
        with tempfile.TemporaryDirectory(prefix="term-audit-test-") as tmp:
            _CACHE[fixture_id] = ta.audit(fixture_id, Path(tmp))
    return _CACHE[fixture_id]


def test_required_fixtures_are_audited():
    assert "dna" in AUDITED and "protein_ligand" in AUDITED and "water_ions" in AUDITED
    print("audited: %s (skipped: %s)" % (", ".join(AUDITED), ", ".join(SKIPPED) or "none"))


def _each() -> list:
    return AUDITED


def test_class_energies_sum_to_total():
    for fixture_id in _each():
        result = audit_of(fixture_id)
        for config in result["configs"]:
            row = config["class_sum"]
            print("%-15s config %d: sum %.9f total %.9f rel %.3e force rel %.3e"
                  % (fixture_id, config["config"], row["energy"],
                     config["total"]["energy"], row["energy_rel_err"],
                     row["force_rel_err"]))
            assert row["energy_rel_err"] < SUM_TOL, (fixture_id, config["config"],
                                                    row["energy_rel_err"])
            assert row["force_rel_err"] < FORCE_TOL, (fixture_id, config["config"],
                                                      row["force_rel_err"])


def test_exact_reference_agreement():
    for fixture_id in _each():
        result = audit_of(fixture_id)
        if result["periodic"]:
            print("%-15s periodic PME: exact NumPy reference not applicable"
                  % fixture_id)
            continue
        for config in result["configs"]:
            exact = config["exact"]
            for name, row in sorted(exact["classes"].items()):
                print("%-15s config %d %-13s openmm %18.9f numpy %18.9f rel %.2e"
                      % (fixture_id, config["config"], name, row["openmm"],
                         row["numpy"], row["rel_err"]))
                assert row["rel_err"] < CLASS_TOL, (fixture_id, config["config"],
                                                    name, row["rel_err"])
                if "force_norm_rel_err" in row:
                    assert row["force_norm_rel_err"] < FORCE_TOL, (
                        fixture_id, config["config"], name,
                        row["force_norm_rel_err"])
            print("%-15s config %d total rel %.2e (numpy %.6f)"
                  % (fixture_id, config["config"], exact["total"]["rel_err"],
                     exact["total"]["numpy"]))
            assert exact["total"]["rel_err"] < TOTAL_TOL, (fixture_id, config["config"],
                                                           exact["total"]["rel_err"])


def test_term_inventory_matches_normalized_json():
    for fixture_id in _each():
        result = audit_of(fixture_id)
        counts = result["counts"]
        for name in ("bonds", "angles", "urey_bradleys", "impropers",
                     "exclusions", "pair14", "lj146"):
            row = counts[name]
            print("%-15s %-13s parmed %6d json %6d missing %d extra %d"
                  % (fixture_id, name, row["parmed"], row["json"],
                     row["missing"], row["extra"]))
            assert row["missing"] == 0 and row["extra"] == 0, (fixture_id, name, row)
        dihedrals = counts["dihedrals"]
        print("%-15s %-13s quartets %6d/%6d terms %6d/%6d multiplicity %d"
              % (fixture_id, "dihedrals", dihedrals["parmed_quartets"],
                 dihedrals["json_quartets"], dihedrals["parmed_terms"],
                 dihedrals["json_terms"], dihedrals["multiplicity_mismatch"]))
        assert dihedrals["missing"] == 0 and dihedrals["extra"] == 0
        assert dihedrals["multiplicity_mismatch"] == 0
        assert dihedrals["parmed_terms"] == dihedrals["json_terms"]
        assert counts["cmaps"]["parmed"] == counts["cmaps"]["json"]
        assert counts["virtual_sites"]["parmed"] == counts["virtual_sites"]["json"]
        assert counts["ok"], fixture_id


def test_exclusions_and_14_pairs_cover_every_exception():
    """The normalized exclusion/1-4 lists must tile the NonbondedForce exceptions."""
    for fixture_id in _each():
        result = audit_of(fixture_id)
        counts = result["counts"]
        normalized = counts["exclusions"]["json"] + counts["pair14"]["json"]
        exceptions = sum(row["count"] for row in result["nonbonded_exceptions"])
        print("%-15s exclusions %d + 1-4 %d = %d ; NonbondedForce exceptions %d"
              % (fixture_id, counts["exclusions"]["json"],
                 counts["pair14"]["json"], normalized, exceptions))
        assert normalized == exceptions, (fixture_id, normalized, exceptions)


def test_each_perturbation_changes_each_class():
    for fixture_id in _each():
        result = audit_of(fixture_id)
        base = result["configs"][0]["classes"]
        for config in result["configs"][1:]:
            for label, row in sorted(config["classes"].items()):
                reference = base[label]
                scale = max(abs(reference["energy"]), 1.0)
                delta = abs(row["energy"] - reference["energy"])
                force_delta = abs(row["force_norm"] - reference["force_norm"])
                moved = delta > 1e-7 * scale or force_delta > 1e-7 * max(
                    reference["force_norm"], 1.0)
                if reference["force_absmax"] == 0.0 and row["force_absmax"] == 0.0:
                    print("%-15s config %d %-32s empty class (skipped)"
                          % (fixture_id, config["config"], label))
                    continue
                print("%-15s config %d %-32s dE %+.6e (scale %.3e)"
                      % (fixture_id, config["config"], label,
                         row["energy"] - reference["energy"], scale))
                assert moved, (fixture_id, config["config"], label, delta, force_delta)


def test_pme_convergence_is_below_the_topology_scale():
    for fixture_id in _each():
        result = audit_of(fixture_id)
        sweep = result.get("pme_convergence")
        if not sweep:
            continue
        stored_tolerance = ta.mr.EWALD_TOLERANCE
        print("%-15s PME stored tolerance %.0e" % (fixture_id, stored_tolerance))
        for row in sweep:
            print("   tol %.0e energy %18.6f |F| %14.6f dE_vs_stored %+.3e"
                  % (row["ewald_error_tolerance"], row["energy"],
                     row["force_norm"], row.get("energy_delta_vs_stored", 0.0)))
        # the tolerance stored in the reference must reproduce the stored config 0
        matching = [row for row in sweep
                    if abs(row["ewald_error_tolerance"] - stored_tolerance) <= 1e-12]
        assert matching, (fixture_id, "stored tolerance not in the sweep")
        assert abs(matching[0].get("energy_delta_vs_stored", 1.0)) < 1e-6, \
            (fixture_id, matching[0].get("energy_delta_vs_stored"))
        energies = [row["energy"] for row in sweep]
        spread = (max(energies) - min(energies)) / max(abs(min(energies)), 1.0)
        print("%-15s PME spread over 5e-4..1e-8: %.3e relative" % (fixture_id, spread))
        assert spread < 1e-3, (fixture_id, spread)


def test_nbfix_declaration_is_quantified():
    """A declared NBFIX gap must be real; an undeclared one must be harmless."""
    for fixture_id in _each():
        result = audit_of(fixture_id)
        impact = result.get("nbfix_impact")
        if impact is None:
            continue
        declarations = [d["code"] for d in result["counts"]["normalized"]["declarations"]]
        print("%-15s declarations %s | LB matrix dev %.2e pairs>1e-8 %d | "
              "normal-pair LJ rel %.2e"
              % (fixture_id, declarations or "none",
                 impact["matrix"]["max_relative_deviation"],
                 impact["matrix"]["pairs_over_1e-8"], impact["rel_error"]))
        if "nbfix_ignored" in declarations:
            assert impact["matrix"]["pairs_over_1e-8"] > 0
            assert impact["rel_error"] > 1e-6, (fixture_id, impact["rel_error"])
        else:
            assert impact["matrix"]["pairs_over_1e-8"] == 0, (fixture_id, impact["matrix"])
            assert impact["rel_error"] < CLASS_TOL, (fixture_id, impact["rel_error"])


def main() -> int:
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    failures = []
    for test in tests:
        try:
            test()
            print("PASS %s" % test.__name__)
        except AssertionError as exc:
            failures.append((test.__name__, exc))
            print("FAIL %s: %s" % (test.__name__, exc))
    print("\n%d/%d checks passed" % (len(tests) - len(failures), len(tests)))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
