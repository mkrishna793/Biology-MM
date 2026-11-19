#!/usr/bin/env python3
"""Per-Force-class / per-term audit of the Phase-2 chemistry fixtures.

The stored reference energies (``make_reference.py``) only compare a *total*
plus a bonded/nonbonded split.  A total is a bad gate: bonded energies cancel
nothing but the nonbonded sum does, and the bonded forces are two orders of
magnitude larger than the nonbonded ones, so a nonbonded mapping error can hide
inside a 1e-6 total-energy tolerance.

This audit therefore does three independent things per fixture:

1. **Per-class OpenMM state.**  Every Force is placed in its own force group and
   evaluated from a fresh Context, so bonds / angles / torsions / impropers /
   CMAP / Coulomb / LJ / 1-6-4 each get their own energy, force norm and
   abs-max.  The class sum must reproduce the total, and the class force vector
   sum must reproduce the total force vector.
   Where one Force carries several physical classes (``NonbondedForce`` holds
   Coulomb + LJ + 1-4), the force is re-evaluated twice more from in-memory
   copies with the charges zeroed and the LJ parameters zeroed, which splits it
   into the two classes without touching any file.

2. **Term inventory.**  ParmEd term sets (bonds, angles, proper/improper
   torsions with every multiplicity term, CMAPs, exclusions, 1-4 pairs, 1-6-4
   pairs, virtual sites) are compared *tuple by tuple* against the normalized
   ``mdparams`` JSON that ``extract_amber.py`` / ``extract_charmm.py`` actually
   write, so a dropped or duplicated term is reported with its atom indices
   rather than inferred from an energy difference.  Extractor refusals and
   warnings (NBFIX, unsupported terms) are recorded verbatim.

3. **Independent NumPy oracle.**  For the NoCutoff fixtures
   (``exact_reference.py``) recomputes every class in double precision from the
   ParmEd parameters and compares against OpenMM class by class.  This is the
   only check here that does not share a code path with the reference.

Periodic fixtures additionally get a PME convergence sweep
(``ewaldErrorTolerance`` 5e-4 / 1e-6 / 1e-8 at config 0) so the numerical PME
floor is separated from a topology or unit error.

Usage::

    python chemistry/tools/term_audit.py --fixture dna
    python chemistry/tools/term_audit.py --fixture all --write
    python chemistry/tools/term_audit.py --fixture dna --print
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import exact_reference as xr  # noqa: E402
import make_reference as mr  # noqa: E402

from openmm import (  # noqa: E402
    CMAPTorsionForce,
    Context,
    CustomNonbondedForce,
    CustomTorsionForce,
    HarmonicAngleForce,
    HarmonicBondForce,
    NonbondedForce,
    PeriodicTorsionForce,
    Platform,
    VerletIntegrator,
)
from openmm.unit import kilojoule_per_mole, nanometer  # noqa: E402

AUDIT_DIR = mr.REFERENCE_DIR / "term_audit"
AUDIT_SCHEMA = "1.0.0"
FIXTURES = ("water_ions", "solvated_protein", "dna", "membrane",
            "charmm_adapter", "protein_ligand")
PERIODIC = dict(mr.PROTOCOLS)
EWALD_TOLERANCES = (5.0e-4, 1.0e-6, 1.0e-8)


# ---------------------------------------------------------------------------
# system construction (identical protocol to make_reference.py)
# ---------------------------------------------------------------------------

def fixture_sources(fixture_id: str) -> dict:
    return mr.resolve_sources(fixture_id, mr.load_manifest())


def build_system(fixture_id: str):
    """Return (system, positions, box, params, kind) exactly as the reference."""
    sources = fixture_sources(fixture_id)
    if sources["kind"] == "amber":
        system, positions, box, _ = mr.build_amber(
            sources["parameter_file"], sources["coordinate_file"], sources["periodic"])
        params = xr.load_amber(sources["parameter_file"])
    else:
        system, positions, box, _ = mr.build_charmm(
            sources["psf"], sources["coordinate_file"], sources["parameter_files"])
        params = xr.load_charmm(sources["psf"], sources["coordinate_file"],
                                sources["parameter_files"])
    mr.configure(system)
    return system, positions, box, params, sources["kind"], sources


def available(fixture_id: str) -> bool:
    if fixture_id == "membrane":
        try:
            fixture_sources("membrane")
            return True
        except Exception:
            return False
    try:
        sources = fixture_sources(fixture_id)
    except Exception:
        return False
    paths = [sources["parameter_file"], sources["coordinate_file"]] if sources["kind"] == "amber" \
        else [sources["psf"], *sources["parameter_files"], sources["coordinate_file"]]
    return all(Path(p).exists() for p in paths)


# ---------------------------------------------------------------------------
# OpenMM per-class state
# ---------------------------------------------------------------------------

def make_context(system, positions, box):
    context = Context(system, VerletIntegrator(0.001), Platform.getPlatformByName("CPU"),
                      {"Threads": "1", "DeterministicForces": "true"})
    if box is not None:
        context.setPeriodicBoxVectors(*[mr.Vec3(*row) for row in box])
    context.setPositions(positions * nanometer)
    return context


def state_of(context, group, periodic):
    state = context.getState(energy=True, forces=True, groups={group},
                            enforcePeriodicBox=periodic)
    energy = state.getPotentialEnergy().value_in_unit(kilojoule_per_mole)
    forces = np.asarray(state.getForces(asNumpy=True).value_in_unit(
        kilojoule_per_mole / nanometer), dtype=np.float64)
    return float(energy), forces


def force_label(force, index: int) -> str:
    if isinstance(force, CustomNonbondedForce):
        expression = force.getEnergyFunction()
        if "ccoef" in expression:
            return "f%d:%s[1-6-4]" % (index, type(force).__name__)
        if "acoef" in expression:
            return "f%d:%s[12-6]" % (index, type(force).__name__)
    return "f%d:%s" % (index, type(force).__name__)


def per_force_state(system, positions, box, periodic):
    """One group per Force; return {(label,index): {energy,force_norm,...}}."""
    for index, force in enumerate(system.getForces()):
        force.setForceGroup(index)
    context = make_context(system, positions, box)
    out = {}
    for index, force in enumerate(system.getForces()):
        energy, forces = state_of(context, index, periodic)
        out[index] = {
            "label": force_label(force, index),
            "energy": energy,
            "force_norm": float(np.linalg.norm(forces)),
            "force_absmax": float(np.abs(forces).max()),
            "_forces": forces,
        }
    state = context.getState(energy=True, forces=True, enforcePeriodicBox=periodic)
    total_energy = float(state.getPotentialEnergy().value_in_unit(kilojoule_per_mole))
    total_forces = np.asarray(state.getForces(asNumpy=True).value_in_unit(
        kilojoule_per_mole / nanometer), dtype=np.float64)
    return out, total_energy, total_forces


def isolated_nonbonded(system, positions, box, periodic, mode):
    """Split a NonbondedForce into Coulomb-only ('coul') or LJ-only ('lj')."""
    clone = copy.deepcopy(system)
    for force in clone.getForces():
        if not isinstance(force, NonbondedForce):
            continue
        for i in range(force.getNumParticles()):
            charge, sigma, epsilon = force.getParticleParameters(i)
            if mode == "coul":
                force.setParticleParameters(i, charge, sigma, 0.0)
            else:
                force.setParticleParameters(i, 0.0, sigma, epsilon)
        for i in range(force.getNumExceptions()):
            a, b, chargeprod, sigma, epsilon = force.getExceptionParameters(i)
            if mode == "coul":
                force.setExceptionParameters(i, a, b, chargeprod, sigma, 0.0)
            else:
                force.setExceptionParameters(i, a, b, 0.0, sigma, epsilon)
    for index, force in enumerate(clone.getForces()):
        if isinstance(force, NonbondedForce):
            force.setForceGroup(1)
        else:
            force.setForceGroup(2)
    context = make_context(clone, positions, box)
    energy, forces = state_of(context, 1, periodic)
    return energy, forces


def kappa(force) -> str:
    if isinstance(force, CustomNonbondedForce):
        expression = force.getEnergyFunction()
        if "ccoef" in expression:
            return "lj146"
        if "acoef" in expression:
            return "lj"
        return "custom"
    return "lj"


# ---------------------------------------------------------------------------
# ParmEd term inventory
# ---------------------------------------------------------------------------

def _pair(i: int, j: int) -> tuple:
    return (i, j) if i < j else (j, i)


def parmed_terms(fixture_id: str) -> dict:
    """Independent term inventory straight from ParmEd (1-based indices)."""
    import parmed

    sources = fixture_sources(fixture_id)
    out = {"bonds": set(), "angles": set(), "urey_bradleys": set(),
           "dihedrals": {}, "impropers": set(), "cmaps": 0,
           "exclusions": set(), "pair14": set(), "lj146": set(),
           "virtual_sites": 0}

    if sources["kind"] == "amber":
        struct = parmed.load_file(str(sources["parameter_file"]))
        for bond in struct.bonds:
            out["bonds"].add(_pair(bond.atom1.idx + 1, bond.atom2.idx + 1))
        for angle in struct.angles:
            out["angles"].add((angle.atom1.idx + 1, angle.atom2.idx + 1, angle.atom3.idx + 1))
        for urey in struct.urey_bradleys:
            out["urey_bradleys"].add(_pair(urey.atom1.idx + 1, urey.atom2.idx + 1))
        for dihedral in struct.dihedrals:
            key = (dihedral.atom1.idx + 1, dihedral.atom2.idx + 1,
                   dihedral.atom3.idx + 1, dihedral.atom4.idx + 1)
            terms = xr._dihedral_type_terms(dihedral.type)
            out["dihedrals"][key] = out["dihedrals"].get(key, 0) + len(terms)
            if not dihedral.improper and not dihedral.ignore_end:
                out["pair14"].add(_pair(dihedral.atom1.idx + 1, dihedral.atom4.idx + 1))
        for improper in struct.impropers:
            out["impropers"].add((improper.atom1.idx + 1, improper.atom2.idx + 1,
                                  improper.atom3.idx + 1, improper.atom4.idx + 1))
        out["cmaps"] = len(getattr(struct, "cmaps", []) or [])
        out["virtual_sites"] = sum(1 for atom in struct.atoms if float(atom.mass) == 0.0)
        parm_data = struct.parm_data
        listed = parm_data.get("EXCLUDED_ATOMS_LIST", [])
        counts = parm_data.get("NUMBER_EXCLUDED_ATOMS", [])
        pos = 0
        raw = set()
        for atom0, count in enumerate(counts):
            for j in listed[pos:pos + count]:
                pos += 1
                if j <= 0:
                    continue
                raw.add(_pair(atom0 + 1, int(j)))
        out["exclusions"] = raw - out["pair14"]

        if "LENNARD_JONES_CCOEF" in parm_data:
            ccoef = [float(x) for x in parm_data["LENNARD_JONES_CCOEF"]]
            if any(value != 0.0 for value in ccoef):
                ntypes = int(struct.pointers["NTYPES"])
                nbi = [int(x) for x in parm_data["NONBONDED_PARM_INDEX"]]
                type_of = [int(atom.nb_idx) - 1 for atom in struct.atoms]
                type_c = {}
                for t1 in range(ntypes):
                    for t2 in range(t1, ntypes):
                        index = nbi[ntypes * t1 + t2] - 1
                        if index < 0 or ccoef[index] == 0.0:
                            continue
                        type_c[(t1, t2)] = True
                excepted = out["exclusions"] | out["pair14"]
                by_type = {}
                for atom in struct.atoms:
                    by_type.setdefault(int(atom.nb_idx) - 1, []).append(atom.idx + 1)
                for (t1, t2) in type_c:
                    for i0 in by_type.get(t1, []):
                        for j0 in by_type.get(t2, []):
                            if i0 < j0 and (i0, j0) not in excepted:
                                out["lj146"].add((i0, j0))
    else:
        from parmed.charmm import CharmmParameterSet
        psf = parmed.charmm.CharmmPsfFile(str(sources["psf"]))
        params = CharmmParameterSet(*[str(path) for path in sources["parameter_files"]])
        psf.load_parameters(params)
        for bond in psf.bonds:
            out["bonds"].add(_pair(bond.atom1.idx + 1, bond.atom2.idx + 1))
        for angle in psf.angles:
            out["angles"].add((angle.atom1.idx + 1, angle.atom2.idx + 1, angle.atom3.idx + 1))
        for urey in psf.urey_bradleys:
            out["urey_bradleys"].add(_pair(urey.atom1.idx + 1, urey.atom2.idx + 1))
        for dihedral in psf.dihedrals:
            key = (dihedral.atom1.idx + 1, dihedral.atom2.idx + 1,
                   dihedral.atom3.idx + 1, dihedral.atom4.idx + 1)
            terms = xr._dihedral_type_terms(dihedral.type)
            out["dihedrals"][key] = out["dihedrals"].get(key, 0) + len(terms)
            out["pair14"].add(_pair(dihedral.atom1.idx + 1, dihedral.atom4.idx + 1))
        for improper in psf.impropers:
            out["impropers"].add((improper.atom1.idx + 1, improper.atom2.idx + 1,
                                  improper.atom3.idx + 1, improper.atom4.idx + 1))
        out["cmaps"] = len(getattr(psf, "cmaps", []) or [])
        out["virtual_sites"] = len(getattr(psf, "lonepair_list", []) or []) + \
            len(getattr(psf, "drudepair_list", []) or [])
        out["exclusions"] = (out["bonds"] | {(a, c) for a, b, c in out["angles"]}) - out["pair14"]
    return out


def normalized_terms(fixture_id: str, workdir: Path) -> dict:
    """Run the real extractor and return its term sets plus declarations."""
    sources = fixture_sources(fixture_id)
    out_path = workdir / ("%s.json" % fixture_id)
    env = dict(os.environ)
    env.pop("FOUNDATION_EXTRACT_ALLOW_NBFIX", None)
    if sources["kind"] == "amber":
        command = [sys.executable, str(HERE / "extract_amber.py"),
                   "--parm", str(sources["parameter_file"]),
                   "--crd", str(sources["coordinate_file"]),
                   "--force-field", "audit", "--out", str(out_path)]
    else:
        command = [sys.executable, str(HERE / "extract_charmm.py"),
                   "--parm", str(sources["psf"]), "--crd", str(sources["coordinate_file"]),
                   "--force-field", "audit", "--out", str(out_path)]
        for path in sources["parameter_files"]:
            command.extend(["--prm", str(path)])

    strict = subprocess.run(command, capture_output=True, text=True, env=env)
    refusals = []
    allow_nbfix = False
    if strict.returncode != 0:
        refusals.append({"mode": "strict", "returncode": strict.returncode,
                         "stderr": strict.stderr.strip()[-400:]})
        env["FOUNDATION_EXTRACT_ALLOW_NBFIX"] = "1"
        relaxed = subprocess.run(command, capture_output=True, text=True, env=env)
        if relaxed.returncode != 0:
            raise RuntimeError("extractor failed even with the NBFIX override: %s"
                               % relaxed.stderr.strip()[-800:])
        allow_nbfix = True
    with open(out_path, "r", encoding="ascii") as handle:
        record = json.load(handle)

    terms = {
        "bonds": {_pair(t["i"], t["j"]) for t in record["bonds"]},
        "angles": {(t["i"], t["j"], t["k"]) for t in record["angles"]},
        "urey_bradleys": {_pair(t["i"], t["j"]) for t in record["urey_bradleys"]},
        "dihedrals": {(t["i"], t["j"], t["k"], t["l"]): len(t["terms"])
                      for t in record["dihedrals"]},
        "impropers": {(t["i"], t["j"], t["k"], t["l"]) for t in record["impropers"]},
        "cmaps": len(record["cmaps"]),
        "exclusions": {_pair(*pair) for pair in record["exclusions"]},
        "pair14": {_pair(t["i"], t["j"]) for t in record["pair14"]},
        "lj146": {(t["i"], t["j"]) for t in record["lj146"]},
        "virtual_sites": len(record["virtual_sites"]),
        "dihedral_term_count": sum(len(t["terms"]) for t in record["dihedrals"]),
        "net_charge": record["net_charge"],
        "n_atoms": len(record["atoms"]),
    }
    return {"terms": terms, "record": record, "refusals": refusals,
            "allow_nbfix": allow_nbfix,
            "diagnostics": record["input_diagnostics"]}


def _set_report(parmed_set, json_set, limit=6) -> dict:
    missing = sorted(parmed_set - json_set)
    extra = sorted(json_set - parmed_set)
    return {
        "parmed": len(parmed_set),
        "json": len(json_set),
        "missing": len(missing),
        "extra": len(extra),
        "missing_examples": [list(x) if isinstance(x, tuple) else [x] for x in missing[:limit]],
        "extra_examples": [list(x) if isinstance(x, tuple) else [x] for x in extra[:limit]],
    }


def compare_terms(fixture_id: str, workdir: Path) -> dict:
    pm = parmed_terms(fixture_id)
    norm = normalized_terms(fixture_id, workdir)
    js = norm["terms"]
    report = {}
    for name in ("bonds", "angles", "urey_bradleys", "impropers",
                 "exclusions", "pair14", "lj146"):
        report[name] = _set_report(pm[name], js[name])
    dihedral_missing = sorted(set(pm["dihedrals"]) - set(js["dihedrals"]))
    dihedral_extra = sorted(set(js["dihedrals"]) - set(pm["dihedrals"]))
    multiplicity = sorted(key for key in set(pm["dihedrals"]) & set(js["dihedrals"])
                          if pm["dihedrals"][key] != js["dihedrals"][key])
    report["dihedrals"] = {
        "parmed_quartets": len(pm["dihedrals"]),
        "json_quartets": len(js["dihedrals"]),
        "parmed_terms": sum(pm["dihedrals"].values()),
        "json_terms": js["dihedral_term_count"],
        "missing": len(dihedral_missing),
        "extra": len(dihedral_extra),
        "multiplicity_mismatch": len(multiplicity),
        "missing_examples": [list(x) for x in dihedral_missing[:6]],
        "extra_examples": [list(x) for x in dihedral_extra[:6]],
        "multiplicity_examples": [list(x) for x in multiplicity[:6]],
    }
    report["cmaps"] = {"parmed": pm["cmaps"], "json": js["cmaps"]}
    report["virtual_sites"] = {"parmed": pm["virtual_sites"], "json": js["virtual_sites"]}
    report["normalized"] = {
        "n_atoms": js["n_atoms"], "net_charge": js["net_charge"],
        "declarations": [{"code": d["code"], "severity": d["severity"],
                          "message": d["message"][:200]} for d in norm["diagnostics"]],
        "strict_refusals": norm["refusals"],
        "nbfix_override_used": norm["allow_nbfix"],
    }
    report["ok"] = all(
        report[name]["missing"] == 0 and report[name]["extra"] == 0
        for name in ("bonds", "angles", "urey_bradleys", "impropers",
                     "exclusions", "pair14", "lj146")) and \
        report["dihedrals"]["missing"] == 0 and report["dihedrals"]["extra"] == 0 and \
        report["dihedrals"]["multiplicity_mismatch"] == 0 and \
        report["cmaps"]["parmed"] == report["cmaps"]["json"] and \
        report["virtual_sites"]["parmed"] == report["virtual_sites"]["json"]
    return report


# ---------------------------------------------------------------------------
# exact-reference comparison
# ---------------------------------------------------------------------------

def _rel(a: float, b: float) -> float:
    scale = max(abs(a), abs(b), 1e-12)
    return abs(a - b) / scale


def compare_exact(params, positions, classes, has_custom_lj: bool) -> dict:
    """Compare per-class OpenMM values against the NumPy oracle."""
    energies, forces = xr.evaluate(params, positions)
    out = {"classes": {}, "skipped": []}

    def note(name, openmm_energy, openmm_forces, exact_energy, exact_forces=None):
        row = {
            "openmm": float(openmm_energy),
            "numpy": float(exact_energy),
            "abs_err": abs(float(openmm_energy) - float(exact_energy)),
            "rel_err": _rel(float(openmm_energy), float(exact_energy)),
        }
        if exact_forces is not None:
            row["force_norm_openmm"] = float(np.linalg.norm(openmm_forces))
            row["force_norm_numpy"] = float(np.linalg.norm(exact_forces))
            row["force_norm_rel_err"] = _rel(row["force_norm_openmm"],
                                             row["force_norm_numpy"])
        out["classes"][name] = row

    bond_seen = 0
    for index in sorted(classes):
        entry = classes[index]
        label = entry["label"].split(":", 1)[1]
        if label == "HarmonicBondForce":
            name = "bonds" if bond_seen == 0 else "urey_bradley"
            bond_seen += 1
            note(name, entry["energy"], entry["_forces"], energies[name], forces[name])
        elif label == "HarmonicAngleForce":
            note("angles", entry["energy"], entry["_forces"],
                 energies["angles"], forces["angles"])
        elif label == "PeriodicTorsionForce":
            note("torsions", entry["energy"], entry["_forces"],
                 energies["torsions"], forces["torsions"])
        elif label == "CustomTorsionForce":
            note("impropers", entry["energy"], entry["_forces"],
                 energies["impropers"], forces["impropers"])
        elif label == "CMAPTorsionForce":
            note("cmaps", entry["energy"], entry["_forces"], 0.0)
            if entry["force_absmax"] > 0.0:
                out["skipped"].append("CMAPTorsionForce is not covered by exact_reference")
        elif label.startswith("CustomNonbondedForce"):
            if "[1-6-4]" in entry["label"]:
                note("lj146", entry["energy"], entry["_forces"],
                     energies["lj146"], forces["lj146"])
            elif "[12-6]" in entry["label"]:
                note("lj", entry["energy"], entry["_forces"], energies["lj"], forces["lj"])
            else:
                out["skipped"].append("unrecognised CustomNonbondedForce %s" % entry["label"])
        elif label == "NonbondedForce":
            note("coulomb", entry["coul"][0], entry["coul"][1],
                 energies["coulomb"], forces["coulomb"])
            if has_custom_lj:
                note("lj14", entry["lj"][0], entry["lj"][1],
                     energies["lj14"], forces["lj14"])
            else:
                note("lj", entry["lj"][0], entry["lj"][1],
                     energies["lj"] + energies["lj14"],
                     forces["lj"] + forces["lj14"])
    out["total"] = {
        "numpy": xr.total(energies),
        "sum_of_classes": sum(row["openmm"] for row in out["classes"].values()),
        "rel_err": _rel(sum(row["openmm"] for row in out["classes"].values()),
                        xr.total(energies)),
        "abs_err": abs(sum(row["openmm"] for row in out["classes"].values())
                       - xr.total(energies)),
    }
    out["numpy_components"] = {name: float(energies[name]) for name in xr.COMPONENTS}
    out["term_counts"] = params.n_terms()
    return out


def nbfix_impact(params, positions) -> dict:
    """Cost of collapsing pair-specific 12-6 terms onto per-atom LB parameters."""
    import dataclasses

    lb_a, lb_b = xr.lb_matrices(params)
    deviation = xr.matrix_deviation(params)
    lb_params = dataclasses.replace(params, lj_a=lb_a, lj_b=lb_b)
    energies_true, _ = xr.evaluate(params, positions, want_forces=False)
    energies_lb, _ = xr.evaluate(lb_params, positions, want_forces=False)
    true_lj = energies_true["lj"]
    lb_lj = energies_lb["lj"]
    return {
        "matrix": deviation,
        "normal_pair_lj_parameters": true_lj,
        "normal_pair_lj_lorentz_berthelot": lb_lj,
        "abs_error": abs(lb_lj - true_lj),
        "rel_error": _rel(lb_lj, true_lj),
    }


# ---------------------------------------------------------------------------
# PME convergence
# ---------------------------------------------------------------------------

def pme_convergence(system, positions, box, periodic, stored_energy=None) -> list:
    rows = []
    for tolerance in EWALD_TOLERANCES:
        clone = copy.deepcopy(system)
        for force in clone.getForces():
            if isinstance(force, NonbondedForce) and force.getNonbondedMethod() == NonbondedForce.PME:
                force.setEwaldErrorTolerance(tolerance)
        context = make_context(clone, positions, box)
        state = context.getState(energy=True, forces=True, enforcePeriodicBox=periodic)
        energy = float(state.getPotentialEnergy().value_in_unit(kilojoule_per_mole))
        forces = np.asarray(state.getForces(asNumpy=True).value_in_unit(
            kilojoule_per_mole / nanometer), dtype=np.float64)
        rows.append({
            "ewald_error_tolerance": tolerance,
            "energy": energy,
            "force_norm": float(np.linalg.norm(forces)),
            "force_absmax": float(np.abs(forces).max()),
        })
    base = rows[0]
    for row in rows:
        row["energy_delta_vs_loosest"] = row["energy"] - base["energy"]
        row["force_norm_delta_vs_loosest"] = row["force_norm"] - base["force_norm"]
        if stored_energy is not None:
            row["energy_delta_vs_stored"] = row["energy"] - stored_energy
    return rows


# ---------------------------------------------------------------------------
# audit driver
# ---------------------------------------------------------------------------

def audit(fixture_id: str, workdir: Path, check_pme: bool = True) -> dict:
    system, base_positions, box, params, kind, sources = build_system(fixture_id)
    periodic = PERIODIC[fixture_id]
    n_atoms = base_positions.shape[0]

    configs = []
    for config in range(mr.CONFIG_COUNT[fixture_id]):
        positions = mr.perturbed(base_positions, config)
        classes, total_energy, total_forces = per_force_state(system, positions, box, periodic)

        # a NonbondedForce that still carries LJ is split into Coulomb and LJ
        mixed = [i for i, f in enumerate(system.getForces())
                 if isinstance(f, NonbondedForce)]
        for index in mixed:
            entry = classes[index]
            entry["coul"] = isolated_nonbonded(system, positions, box, periodic, "coul")
            entry["lj"] = isolated_nonbonded(system, positions, box, periodic, "lj")

        clean = {}
        for index, entry in classes.items():
            row = {key: entry[key] for key in ("label", "energy", "force_norm", "force_absmax")}
            if "coul" in entry:
                row["coulomb_energy"] = entry["coul"][0]
                row["lj_energy"] = entry["lj"][0]
            clean[entry["label"]] = row

        class_energy_sum = sum(entry["energy"] for entry in classes.values())
        force_sum = sum(entry["_forces"] for entry in classes.values())
        record = {
            "config": config,
            "classes": clean,
            "total": {
                "energy": total_energy,
                "force_norm": float(np.linalg.norm(total_forces)),
                "force_absmax": float(np.abs(total_forces).max()),
            },
            "class_sum": {
                "energy": class_energy_sum,
                "energy_abs_err": abs(class_energy_sum - total_energy),
                "energy_rel_err": _rel(class_energy_sum, total_energy),
                "force_abs_err": float(np.abs(force_sum - total_forces).max()),
                "force_rel_err": _rel(float(np.linalg.norm(force_sum)),
                                     float(np.linalg.norm(total_forces))),
            },
        }

        if not periodic:
            has_custom_lj = any(
                "[12-6]" in entry["label"] for entry in classes.values())
            record["exact"] = compare_exact(params, positions, classes, has_custom_lj)
        configs.append(record)

    result = {
        "schema": AUDIT_SCHEMA,
        "fixture": fixture_id,
        "n_atoms": int(n_atoms),
        "periodic": periodic,
        "protocol": {
            "nonbonded": "PME" if periodic else "NoCutoff",
            "cutoff_nm": mr.CUTOFF_NM if periodic else None,
            "ewald_error_tolerance": mr.EWALD_TOLERANCE if periodic else None,
            "dispersion_correction": False,
            "constraints": "none",
        },
        "nonbonded_exceptions": [
            {"label": force_label(force, index),
             "count": force.getNumExceptions(),
             "particles": force.getNumParticles()}
            for index, force in enumerate(system.getForces())
            if isinstance(force, NonbondedForce)],
        "counts": compare_terms(fixture_id, workdir),
        "nbfix_impact": None if periodic else nbfix_impact(params, mr.perturbed(base_positions, 0)),
        "configs": configs,
    }
    if periodic and check_pme:
        stored = None
        stored_path = mr.REFERENCE_DIR / ("%s.json" % fixture_id)
        if stored_path.exists():
            with open(stored_path, "r", encoding="utf-8") as handle:
                stored_record = json.load(handle)
            stored = stored_record.get("configs", [{}])[0].get("energy_kj_mol")
        result["pme_convergence"] = pme_convergence(
            system, mr.perturbed(base_positions, 0), box, periodic, stored)
    return result


def summarize(result: dict) -> str:
    lines = []
    counts = result["counts"]
    lines.append("%s: N=%d %s" % (result["fixture"], result["n_atoms"],
                                  "PME" if result["periodic"] else "NoCutoff"))
    if counts["ok"]:
        lines.append("  terms: all %d categories match the normalized JSON"
                     % len([k for k in counts if isinstance(counts[k], dict)
                            and "parmed" in counts[k]]))
    else:
        for name in ("bonds", "angles", "urey_bradleys", "impropers",
                     "exclusions", "pair14", "lj146"):
            row = counts[name]
            if row["missing"] or row["extra"]:
                lines.append("  TERM DEFECT %s: parmed=%d json=%d missing=%s extra=%s"
                             % (name, row["parmed"], row["json"],
                                row["missing_examples"], row["extra_examples"]))
        d = counts["dihedrals"]
        if d["missing"] or d["extra"] or d["multiplicity_mismatch"]:
            lines.append("  TERM DEFECT dihedrals: %s" % d)
    for config in result["configs"]:
        row = config["class_sum"]
        lines.append("  config %d: E=%.6f |F|=%.6f class-sum rel %.2e force rel %.2e"
                     % (config["config"], config["total"]["energy"],
                        config["total"]["force_norm"], row["energy_rel_err"],
                        row["force_rel_err"]))
        if "exact" in config:
            worst = sorted(config["exact"]["classes"].items(),
                           key=lambda kv: -kv[1]["rel_err"])[:3]
            lines.append("    exact worst rel: " + ", ".join(
                "%s %.2e" % (name, row["rel_err"]) for name, row in worst))
            lines.append("    exact total rel %.2e (numpy %.6f vs openmm classes %.6f)"
                         % (config["exact"]["total"]["rel_err"],
                            config["exact"]["total"]["numpy"],
                            config["exact"]["total"]["sum_of_classes"]))
    impact = result.get("nbfix_impact")
    if impact:
        lines.append("  12-6 LB vs pair-specific: max matrix dev %.2e (%d used-type pairs >1e-8),"
                     " normal-pair LJ rel err %.2e"
                     % (impact["matrix"]["max_relative_deviation"],
                        impact["matrix"]["pairs_over_1e-8"],
                        impact["rel_error"]))
    pme = result.get("pme_convergence")
    if pme:
        lines.append("  PME (|F| per tolerance, dE vs stored config 0): " + ", ".join(
            "tol %.0e dE=%+.3e |F|=%.6f" % (row["ewald_error_tolerance"],
                                            row.get("energy_delta_vs_stored", 0.0),
                                            row["force_norm"])
            for row in pme))
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", default="all",
                        choices=list(FIXTURES) + ["all"])
    parser.add_argument("--write", action="store_true",
                        help="write chemistry/reference/term_audit/<id>.json")
    parser.add_argument("--print", dest="verbose", action="store_true",
                        help="print the per-class table")
    parser.add_argument("--no-pme", action="store_true", help="skip the PME sweep")
    args = parser.parse_args(argv)

    ids = list(FIXTURES) if args.fixture == "all" else [args.fixture]
    failures = []
    with tempfile.TemporaryDirectory(prefix="term-audit-") as tmp:
        for fixture_id in ids:
            if not available(fixture_id):
                print("%s: skipped (inputs absent)" % fixture_id)
                continue
            result = audit(fixture_id, Path(tmp), check_pme=not args.no_pme)
            print(summarize(result))
            if args.verbose:
                for config in result["configs"]:
                    for label, row in config["classes"].items():
                        print("    %-34s E=%16.6f |F|=%14.6f max=%14.6f"
                              % (label, row["energy"], row["force_norm"],
                                 row["force_absmax"]))
            if not result["counts"]["ok"]:
                failures.append((fixture_id, "term inventory"))
            if args.write:
                AUDIT_DIR.mkdir(parents=True, exist_ok=True)
                path = AUDIT_DIR / ("%s.json" % fixture_id)
                with open(path, "w", encoding="utf-8") as handle:
                    json.dump(result, handle, indent=1, sort_keys=True, allow_nan=False)
                    handle.write("\n")
                print("  wrote %s" % path.relative_to(mr.REPO_ROOT))
    if failures:
        print("failures: %s" % failures, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
