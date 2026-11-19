#!/usr/bin/env python
"""Per-term energies and forces from OpenMM, for the frozen bonded fixture.

Run with the OpenMM environment (not the InterMol one):

    ~/.cache/biology-mm/.venv/bin/python \
        reference/openmm_terms.py > reference/openmm_terms.tsv

Input: reference/terms.tsv (parameters + geometry, produced by
extract_reference.py).  This script never re-parses the GROMACS text: it takes
the **ParmEd** columns and converts them to OpenMM's units with the declared
mapping, so the engine comparison is against a real external evaluator, not
against a second implementation of the same formula.

Declared conversion (ParmEd -> OpenMM), from the convention table:
  bond    OpenMM k [kJ/mol/nm^2] = 836.8 * kp [kcal/mol/A^2]
  angle   OpenMM k [kJ/mol/rad^2] = 8.368 * kth_p [kcal/mol/rad^2]
  torsion OpenMM k [kJ/mol]       = 4.184 * phi_k [kcal/mol]
  r0 [nm] = req [A] * 0.1 ; theta0 [rad] = theteq [deg] ; phase [rad] = phase [deg]

OpenMM's standard bonded terms are exactly the three reference forms:
  HarmonicBondForce       E = 0.5 k (r - r0)^2
  HarmonicAngleForce      E = 0.5 k (theta - theta0)^2
  PeriodicTorsionForce    E = k [1 + cos(n phi - phase)]
"""
import math
import sys

import numpy as np
import openmm
from openmm import unit

KJ = unit.kilojoules_per_mole
NM = unit.nanometer
RAD = unit.radian


def read_terms(path):
    atoms = {}
    bonds = []
    angles = []
    dihs = []
    for line in open(path):
        if line.startswith("#"):
            continue
        w = line.split()
        if not w:
            continue
        if w[0] == "ATOM":
            # ATOM <case> <idx> <name> <x> <y> <z>   (nm)
            atoms.setdefault(w[1], []).append((int(w[2]), w[3], *map(float, w[4:7])))
        elif w[0] == "BOND":
            # BOND i j k_gmx r0 kp r0p_A case
            bonds.append(dict(case=w[7], i=int(w[1]), j=int(w[2]),
                              kp=float(w[5]), r0p=float(w[6])))
        elif w[0] == "ANGLE":
            # ANGLE i j k k_gmx th0 kp th0p case
            angles.append(dict(case=w[8], i=int(w[1]), j=int(w[2]), k=int(w[3]),
                               kp=float(w[6]), th0p=float(w[7])))
        elif w[0] == "DIH":
            # DIH i j k l k_gmx n phis kp np phisp case fc0..fc6
            dihs.append(dict(case=w[11], i=int(w[1]), j=int(w[2]), k=int(w[3]), l=int(w[4]),
                             kp=float(w[8]), n=int(w[9]), phisp=float(w[10])))
    return atoms, bonds, angles, dihs


def terms_for(case, bonds, angles, dihs):
    """(kind, indices, builder) for every term of one case."""
    out = []
    for b in bonds:
        if b["case"] == case:
            out.append(("bond", (b["i"], b["j"]), 0, lambda f, b=b: f.addBond(
                b["i"] - 1, b["j"] - 1, b["r0p"] * 0.1 * NM, 836.8 * b["kp"] * KJ / NM ** 2)))
    for a in angles:
        if a["case"] == case:
            out.append(("angle", (a["i"], a["j"], a["k"]), 0, lambda f, a=a: f.addAngle(
                a["i"] - 1, a["j"] - 1, a["k"] - 1,
                math.radians(a["th0p"]) * RAD, 8.368 * a["kp"] * KJ / RAD ** 2)))
    for d in dihs:
        if d["case"] == case:
            out.append(("dih", (d["i"], d["j"], d["k"], d["l"]), d["n"], lambda f, d=d: f.addTorsion(
                d["i"] - 1, d["j"] - 1, d["k"] - 1, d["l"] - 1,
                d["n"], math.radians(d["phisp"]) * RAD, 4.184 * d["kp"] * KJ)))
    return out


def eval_one(coords, kind, build, platform):
    n = len(coords)
    system = openmm.System()
    for _ in range(n):
        system.addParticle(1.0)
    f = {"bond": openmm.HarmonicBondForce,
         "angle": openmm.HarmonicAngleForce,
         "dih": openmm.PeriodicTorsionForce}[kind]()
    build(f)
    system.addForce(f)
    ctx = openmm.Context(system, openmm.VerletIntegrator(0.001),
                         openmm.Platform.getPlatformByName(platform))
    ctx.setPositions(np.array(coords) * NM)
    st = ctx.getState(energy=True, forces=True)
    e = st.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    fv = st.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
    return e, fv


def eval_total(coords, terms, platform):
    n = len(coords)
    system = openmm.System()
    for _ in range(n):
        system.addParticle(1.0)
    for kind, idx, nterm, build in terms:
        f = {"bond": openmm.HarmonicBondForce,
             "angle": openmm.HarmonicAngleForce,
             "dih": openmm.PeriodicTorsionForce}[kind]()
        build(f)
        system.addForce(f)
    ctx = openmm.Context(system, openmm.VerletIntegrator(0.001),
                         openmm.Platform.getPlatformByName(platform))
    ctx.setPositions(np.array(coords) * NM)
    st = ctx.getState(energy=True, forces=True)
    e = st.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    fv = st.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
    return e, fv


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "reference/terms.tsv"
    platform = "Reference"
    atoms, bonds, angles, dihs = read_terms(path)
    print("# OpenMM per-term energies and forces on the frozen bonded fixture")
    print("# generator: reference/openmm_terms.py, OpenMM %s, platform %s" %
          (openmm.__version__, platform))
    print("# columns: kind i j k l, energy kJ/mol, then 3*N force components kJ/mol/nm")
    for case in sorted(atoms):
        coords = [a[2:] for a in sorted(atoms[case])]
        terms = terms_for(case, bonds, angles, dihs)
        print("#CASE %s platform=%s openmm=%s natoms=%d nterms=%d" %
              (case, platform, openmm.__version__, len(coords), len(terms)))
        for kind, idx, nterm, build in terms:
            e, fv = eval_one(coords, kind, build, platform)
            i, j, k, *rest = list(idx) + [0] * (4 - len(idx))
            l = rest[0] if rest else 0
            flat = " ".join("%.17g" % v for v in fv.reshape(-1))
            n = nterm
            print("OMME %s %s %d %d %d %d %d %.17g" % (case, kind, i, j, k, l, n, e))
            print("OMMF %s %s %d %d %d %d %d %s" % (case, kind, i, j, k, l, n, flat))
        et, ft = eval_total(coords, terms, platform)
        print("OMMT %s %.17g %s" % (case, et, " ".join("%.17g" % v for v in ft.reshape(-1))))


if __name__ == "__main__":
    main()
