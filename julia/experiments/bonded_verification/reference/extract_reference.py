#!/usr/bin/env python
"""Extract per-term bonded parameters from two independent external parsers.

Run with the isolated InterMol environment (outside the repository):

    ~/.cache/biology-mm/intermol-verify/bin/python \
        reference/extract_reference.py > reference/terms.tsv

Inputs are two GROMACS vacuum topologies shipped inside the InterMol
distribution (`intermol/tests/gromacs/unit_tests/`).  They are parsed twice,
by InterMol (0.1.2) and by ParmEd (4.3.1), and the two views are cross-checked
here before a single table is written.  Nothing in this script invents a
parameter: every number is copied from a parser's object model.

Reference conventions (documented, not inferred):

  GROMACS funct 1 bond   V = (1/2) k_b (r - b0)^2      [kJ/mol, nm]
  GROMACS funct 1 angle  V = (1/2) k_th (th - th0)^2   [kJ/mol, rad]
  GROMACS funct 1/9 dih  V = k_phi [1 + cos(n phi - phi_s)]

  ParmEd BondType        V = k_b (r - b0)^2            [kcal/mol, A]
  ParmEd AngleType       V = k_th (th - th0)^2         [kcal/mol, rad]
  ParmEd DihedralType    V = phi_k [1 + cos(n phi - phase)]

The factor-of-two between the two bond/angle k's is therefore a *convention*
difference (the 1/2), and the kJ/kcal and nm/A differences are units.  The
torsion has no 1/2 in either convention.
"""
import hashlib
import os
import sys

import parmed as pmd
import intermol

from intermol.gromacs.gromacs_parser import GromacsParser

DEFAULT_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(intermol.__file__)),
    "tests", "gromacs", "unit_tests")

CASES = ["dihedral9_vacuum/dihedral9_vacuum", "dihedral1_vacuum/dihedral1_vacuum"]

# ParmEd internal units -> GROMACS units (exact definitions, not fitted):
#   1 kcal/mol = 4.184 kJ/mol
#   1 A = 0.1 nm  =>  (kcal/mol)/A^2 -> kJ/mol/nm^2 is 4.184 * 100 and the
#   missing factor 2 belongs to the 1/2 in the GROMACS convention.
KCAL_TO_KJ = 4.184
ANG2_PER_NM2 = 100.0
A2_TO_NM2 = 0.01


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read())
    return h.hexdigest()


def parse_intermol(base):
    m = GromacsParser(base + ".top", base + ".gro").read().molecule_types["Ethanol"]
    bonds = []
    for b in m.bond_forces:
        bonds.append(dict(i=b.atom1, j=b.atom2,
                          k=float(b.k._value), r0=float(b.length._value)))
    angles = []
    for a in m.angle_forces:
        angles.append(dict(i=a.atom1, j=a.atom2, k=a.atom3,
                           kth=float(a.k._value), th0=float(a.theta._value)))
    dihedrals = []
    for d in m.dihedral_forces:
        fc = [float(getattr(d, "fc%d" % i)._value) for i in range(7)]
        # InterMol stores GROMACS funct 1/9 as fc0 = k and fc_n = k with
        # phi = phi_s.  Recover the triple from that representation.
        nz = [i for i in range(1, 7) if abs(fc[i]) > 0.0]
        if len(nz) != 1 or abs(fc[0] - fc[nz[0]]) > 1e-9:
            raise SystemExit("unexpected InterMol dihedral form: %s" % fc)
        dihedrals.append(dict(i=d.atom1, j=d.atom2, k=d.atom3, l=d.atom4,
                              kphi=fc[nz[0]], n=nz[0], phis=float(d.phi._value),
                              fc=fc, phi=float(d.phi._value)))
    return bonds, angles, dihedrals


def parse_parmed(base):
    s = pmd.load_file(base + ".top", xyz=base + ".gro")
    bonds = [dict(i=b.atom1.idx + 1, j=b.atom2.idx + 1,
                  k=float(b.type.k), r0=float(b.type.req)) for b in s.bonds]
    angles = [dict(i=a.atom1.idx + 1, j=a.atom2.idx + 1, k=a.atom3.idx + 1,
                   kth=float(a.type.k), th0=float(a.type.theteq)) for a in s.angles]
    dihedrals = []
    for d in s.dihedrals:
        types = d.type if isinstance(d.type, (list, tuple)) else [d.type]
        for t in types:
            dihedrals.append(dict(i=d.atom1.idx + 1, j=d.atom2.idx + 1,
                                  k=d.atom3.idx + 1, l=d.atom4.idx + 1,
                                  kphi=float(t.phi_k), n=int(t.per),
                                  phis=float(t.phase)))
    return s, bonds, angles, dihedrals


def key(t):
    return (t["i"], t["j"], t["k"], t["l"], t.get("n", 0))


def main():
    root = os.environ.get("INTERMOL_TESTS", DEFAULT_ROOT)
    if not os.path.isdir(root):
        raise SystemExit("InterMol test corpus not found at %s; set INTERMOL_TESTS" % root)

    print("# external-source extraction for the bonded convention lane")
    print("# generator: reference/extract_reference.py (InterMol + ParmEd)")
    print("# units: length nm, angle rad, energy kJ/mol")
    print("# NOTE: the torsions below are written as (k_phi, n, phi_s) with")
    print("#       V = k_phi*[1 + cos(n*phi - phi_s)], i.e. the GROMACS funct 1/9")
    print("#       form; InterMol stores the same numbers as (fc0=k, fc_n=k, phi=phi_s)")
    print("#       and ParmEd as (phi_k=k, per=n, phase=phi_s).")

    for case in CASES:
        base = os.path.join(root, case)
        tag = os.path.basename(case)
        top, gro = base + ".top", base + ".gro"
        print("#CASE %s sha256_top=%s sha256_gro=%s" %
              (tag, sha256(top), sha256(gro)))

        ib, ia, idih = parse_intermol(base)
        s, pb, pa, pdih = parse_parmed(base)

        # geometry from the .gro that came with the topology, tagged by case so a
        # consumer cannot concatenate the same 9 atoms twice
        for at in s.atoms:
            print("ATOM %s %d %s %.10f %.10f %.10f" %
                  (tag, at.idx + 1, at.name, at.xx * 0.1, at.xy * 0.1, at.xz * 0.1))

        # ---- bonds: InterMol (GROMACS units) must pair with ParmEd (kcal/A) ----
        imol = {(b["i"], b["j"]): b for b in ib}
        if len(imol) != len(ib):
            raise SystemExit("ambiguous InterMol bond table")
        if len(pb) != len(ib):
            raise SystemExit("bond count mismatch: intermol %d parmed %d" % (len(ib), len(pb)))
        for b in sorted(pb, key=lambda x: (x["i"], x["j"])):
            r = imol[(b["i"], b["j"])]
            k_conv = b["k"] * KCAL_TO_KJ * ANG2_PER_NM2
            if abs(k_conv - 0.5 * r["k"]) > 1e-6 * max(1.0, r["k"]):
                raise SystemExit("bond k convention mismatch %s: %.6f vs %.6f" %
                                 ((b["i"], b["j"]), k_conv, 0.5 * r["k"]))
            if abs(b["r0"] * 0.1 - r["r0"]) > 1e-9:
                raise SystemExit("bond r0 mismatch %s" % ((b["i"], b["j"]),))
            print("BOND %d %d %.17g %.17g %.17g %.17g %s" %
                  (r["i"], r["j"], r["k"], r["r0"],
                   b["k"], b["r0"], tag))

        # ---- angles ----
        imola = {(a["i"], a["j"], a["k"]): a for a in ia}
        if len(imola) != len(ia) or len(pa) != len(ia):
            raise SystemExit("angle table not one-to-one")
        for a in sorted(pa, key=lambda x: (x["i"], x["j"], x["k"])):
            r = imola[(a["i"], a["j"], a["k"])]
            k_conv = a["kth"] * KCAL_TO_KJ
            if abs(k_conv - 0.5 * r["kth"]) > 1e-6 * max(1.0, r["kth"]):
                raise SystemExit("angle k convention mismatch %s: %.6f vs %.6f" %
                                 ((a["i"], a["j"], a["k"]), k_conv, 0.5 * r["kth"]))
            if abs(a["th0"] - r["th0"]) > 1e-9:
                raise SystemExit("angle theta0 mismatch %s" % ((a["i"], a["j"], a["k"]),))
            print("ANGLE %d %d %d %.17g %.17g %.17g %.17g %s" %
                  (r["i"], r["j"], r["k"], r["kth"], r["th0"],
                   a["kth"], a["th0"], tag))

        # ---- torsions: pair by (i,j,k,l,n); no 1/2 in either convention ----
        pd = {key(t): t for t in pdih}
        if len(pd) != len(pdih):
            raise SystemExit("duplicate ParmEd dihedral key")
        if len(idih) != len(pdih):
            raise SystemExit("dihedral count mismatch: intermol %d parmed %d" %
                             (len(idih), len(pdih)))
        for t in sorted(idih, key=lambda x: (x["i"], x["j"], x["k"], x["l"], x["n"])):
            p = pd.pop(key(t), None)
            if p is None:
                raise SystemExit("no ParmEd partner for %s" % (key(t),))
            k_conv = p["kphi"] * KCAL_TO_KJ
            if abs(k_conv - t["kphi"]) > 1e-6 * max(1.0, t["kphi"]):
                raise SystemExit("dihedral k mismatch %s: %.9f vs %.9f" %
                                 (key(t), k_conv, t["kphi"]))
            if abs(p["phis"] - t["phis"]) > 1e-9:
                raise SystemExit("dihedral phase mismatch %s" % (key(t),))
            print("DIH %d %d %d %d %.17g %d %.17g %.17g %d %.17g %s %s" %
                  (t["i"], t["j"], t["k"], t["l"], t["kphi"], t["n"], t["phis"],
                   p["kphi"], p["n"], p["phis"], tag,
                   " ".join("%.6f" % v for v in t["fc"])))
        if pd:
            raise SystemExit("unmatched ParmEd dihedrals: %s" % (sorted(pd),))

        # ---- an independent dihedral-angle implementation on the same geometry ----
        from parmed.geometry import dihedral as pmd_dihedral
        from parmed.topologyobjects import Atom
        for t in sorted({key(x): x for x in idih}.values(),
                        key=lambda x: (x["i"], x["j"], x["k"], x["l"])):
            phi_deg = pmd_dihedral(s.atoms[t["i"] - 1], s.atoms[t["j"] - 1],
                                   s.atoms[t["k"] - 1], s.atoms[t["l"] - 1])
            print("PHIPARMED %d %d %d %d %.10f" %
                  (t["i"], t["j"], t["k"], t["l"], phi_deg))


if __name__ == "__main__":
    main()
