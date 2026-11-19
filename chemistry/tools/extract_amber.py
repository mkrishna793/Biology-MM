#!/usr/bin/env python3
"""AMBER prmtop/inpcrd -> normalized JSON (mdparams 1.0.0).

The force-field convention is fixed by ``chemistry/INTERFACE.md``:

* bonds/angles are stored with the coefficient used by the foundation factors
  ``U = (k/2)(x - x0)^2``; AMBER and OpenMM define the prmtop coefficients with
  ``U = K (x - x0)^2``, so the extractor doubles them (unit factors are
  ``418.4`` for bonds and ``4.184`` for angles).
* dihedral terms are ``k (1 + cos(n phi - phase))``.  ParmEd's
  ``type.phi_k`` is already the native coefficient of that form (the prmtop
  ``Vn/2``), so ``k = phi_k * 4.184`` with no extra factor of one half.  Every
  DIHEDRALS entry is kept; entries
  with ``ignore_end=True`` still contribute energy but do not create a 1-4
  pair.  AMBER's periodic "improper" entries are DIHEDRALS entries with a
  negative fourth atom; OpenMM puts them in its periodic torsion force, so
  they stay in ``dihedrals`` here as well.
* CHARMM-style harmonic impropers in an AMBER prmtop (``CHARMM_IMPROPERS``,
  written by ParmEd's ``ChamberParm``) are the only entries placed in
  ``impropers``: ParmEd exposes ``type.psi_k`` in kcal/mol/rad^2 and
  ``type.psi_eq`` in degrees.  Thus ``k_phi = psi_k * 4.184`` and
  ``phase_rad = radians(psi_eq)``.  For a plain AMBER prmtop ``p.impropers`` is
  empty and the periodic improper entries are already in ``dihedrals``.
* atom sigma/epsilon come from the raw diagonal A/B coefficients exactly as
  in OpenMM's ``getNonbondTerms``::

      rMin = (2 A / B)^(1/6)                 (angstrom, LJ minimum)
      sigma = rMin * 2^(-1/6) * 0.1          (nm, zero crossing)
      epsilon = B^2 / (4 A) * 4.184          (kJ/mol)

  With ``A == 0`` or ``B == 0`` OpenMM falls back to sigma = 0.5 A = 0.05 nm
  and epsilon = 0; the same fallback is used here.

NBFIX is refused, not diagonalized: any pair of used atom types whose stored
A/B coefficient is not Lorentz-Berthelot from the diagonal within 1e-8
relative raises ``unsupported_term``.  The test suite can opt into a recorded
warning (``FOUNDATION_EXTRACT_ALLOW_NBFIX=1``) for fixtures that contain an
ion-pair NBFIX; the default remains a hard refusal.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

from common import (ANGLE_FACTOR, BOND_FACTOR, ExtractionError, KCAL_TO_KJ,
                    SCHEMA, UNITS, cli_error, diagnostic, fatal, finish,
                    nbfix_allowed, relative_source, sha256_file)

GENERATOR = "chemistry/tools/extract_amber.py"
GENERATOR_VERSION = "1.0.0"
NBFIX_MESSAGE = "NBFIX/off-diagonal LJ is not supported in schema v1.0.0"


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def _flags(path: str) -> set:
    flags = set()
    with open(path, "r", encoding="ascii", errors="replace") as fh:
        for line in fh:
            if line.startswith("%FLAG"):
                flags.add(line[6:].strip())
    return flags


def load_structure(parm: str, crd: str):
    try:
        from parmed.amber import AmberParm, ChamberParm
    except Exception as exc:  # pragma: no cover - import failure is fatal
        fatal("io", f"cannot import ParmEd: {exc}")
    flags = _flags(parm)
    cls = ChamberParm if "CHARMM_IMPROPERS" in flags else AmberParm
    try:
        return cls(parm, crd)
    except Exception as exc:
        fatal("parameter", f"ParmEd could not read {parm} + {crd}: {exc}")


# ---------------------------------------------------------------------------
# box
# ---------------------------------------------------------------------------

def _box_vectors(alpha: float, a: float, b: float, c: float,
                 beta: float, gamma: float):
    alpha, beta, gamma = (math.radians(x) for x in (alpha, beta, gamma))
    ca, cb, cg = math.cos(alpha), math.cos(beta), math.cos(gamma)
    sg = math.sin(gamma)
    if abs(sg) < 1e-12:
        fatal("box", "degenerate box: gamma is 0 or 180 degrees")
    va = (a, 0.0, 0.0)
    vb = (b * cg, b * sg, 0.0)
    cx = c * cb
    cy = c * (ca - cb * cg) / sg
    cz2 = c * c - cx * cx - cy * cy
    if cz2 < 0.0 and cz2 > -1e-8:
        cz2 = 0.0
    if cz2 <= 0.0:
        fatal("box", "degenerate box: non-positive cell height")
    vc = (cx, cy, math.sqrt(cz2))
    return [list(va), list(vb), list(vc)]


def _prmtop_box(parm_data) -> dict:
    raw = parm_data.get("BOX_DIMENSIONS")
    if not raw or len(raw) < 4:
        return {"kind": "none"}
    alpha = float(raw[0])
    a, b, c = float(raw[1]), float(raw[2]), float(raw[3])
    beta = float(raw[4]) if len(raw) > 4 else 90.0
    gamma = float(raw[5]) if len(raw) > 5 else 90.0
    if min(a, b, c) <= 0:
        fatal("box", "prmtop BOX_DIMENSIONS has a non-positive length")
    if abs(alpha - 90) < 1e-6 and abs(beta - 90) < 1e-6 and abs(gamma - 90) < 1e-6:
        return {"kind": "orthorhombic",
                "lengths_nm": [a * 0.1, b * 0.1, c * 0.1]}
    return {"kind": "triclinic",
            "vectors_nm": _box_vectors(alpha, a, b, c, beta, gamma)}


def _amber_coord_box(struct):
    box = getattr(struct, "box", None)
    if box is None or len(box) < 6:
        return None
    a, b, c = (float(x) * 0.1 for x in box[:3])
    alpha, beta, gamma = (float(x) for x in box[3:6])
    return _box_vectors(alpha, a, b, c, beta, gamma), \
        (a, b, c, alpha, beta, gamma)


def _box_diagnostic(prmtop_box: dict, coord) -> dict | None:
    if coord is None:
        return None
    vectors, lengths_angles = coord
    if prmtop_box["kind"] == "none":
        return None
    if prmtop_box["kind"] == "orthorhombic":
        pvec = [[prmtop_box["lengths_nm"][0], 0.0, 0.0],
                [0.0, prmtop_box["lengths_nm"][1], 0.0],
                [0.0, 0.0, prmtop_box["lengths_nm"][2]]]
        pdesc = (f"lengths_nm=({prmtop_box['lengths_nm'][0]:.6f},"
                 f"{prmtop_box['lengths_nm'][1]:.6f},"
                 f"{prmtop_box['lengths_nm'][2]:.6f})")
    else:
        pvec = prmtop_box["vectors_nm"]
        pdesc = "vectors_nm=(" + ",".join(
            "(" + ",".join(f"{v:.6f}" for v in row) + ")" for row in pvec) + ")"
    diff = max(abs(pvec[r][q] - vectors[r][q])
               for r in range(3) for q in range(3))
    if diff <= 1e-4:
        return None
    a, b, c, alpha, beta, gamma = lengths_angles
    cdesc = (f"lengths_nm=({a:.6f},{b:.6f},{c:.6f}) "
             f"angles_deg=({alpha:.6f},{beta:.6f},{gamma:.6f})")
    return diagnostic(
        "box_mismatch", "warning",
        f"prmtop box {pdesc} differs from coordinate box {cdesc} by "
        f"{diff:.6e} nm; the prmtop box is used (OpenMM parity)")


# ---------------------------------------------------------------------------
# atoms and LJ
# ---------------------------------------------------------------------------

def _element(atom) -> str:
    name = getattr(atom, "element_name", "") or ""
    if name:
        return name
    z = int(getattr(atom, "atomic_number", 0) or 0)
    if z > 0:
        import parmed.periodic_table as pt
        if z < len(pt.Element):
            return pt.Element[z - 1]
    fatal("atom", f"cannot determine an element symbol for atom "
                  f"{atom.idx + 1} ({atom.name})")


def _diagonal_lj(parm_data, ntypes: int, nb_idx: int):
    acoef = parm_data["LENNARD_JONES_ACOEF"]
    bcoef = parm_data["LENNARD_JONES_BCOEF"]
    index = int(parm_data["NONBONDED_PARM_INDEX"][(nb_idx - 1) * (ntypes + 1)]) - 1
    if index < 0 or index >= len(acoef):
        fatal("parameter", f"atom type {nb_idx} has no diagonal LJ entry")
    a = float(acoef[index])
    b = float(bcoef[index])
    if a == 0.0 or b == 0.0:
        # OpenMM's ZeroDivisionError branch in getNonbondTerms.
        return 0.05, 0.0
    rmin = (2.0 * a / b) ** (1.0 / 6.0)
    sigma = rmin * 2.0 ** (-1.0 / 6.0) * 0.1
    epsilon = 0.25 * b * b / a * KCAL_TO_KJ
    return sigma, epsilon


def _atoms(struct):
    parm_data = struct.parm_data
    ntypes = int(struct.pointers["NTYPES"])
    out = []
    for atom in struct.atoms:
        if float(atom.mass) == 0.0 or int(getattr(atom, "atomic_number", 0) or 0) == 0:
            fatal("unsupported_term",
                  f"atom {atom.idx + 1} ({atom.name}) is massless or has no "
                  "atomic number; virtual sites/lone pairs are not supported "
                  "by extract_amber.py")
        nb_idx = int(atom.nb_idx)
        if not 1 <= nb_idx <= ntypes:
            fatal("parameter",
                  f"atom {atom.idx + 1} ({atom.name}) has invalid type index {nb_idx}")
        sigma, epsilon = _diagonal_lj(parm_data, ntypes, nb_idx)
        out.append({
            "name": atom.name,
            "mass": float(atom.mass),
            "charge": float(atom.charge),
            "element": _element(atom),
            "residue_index": int(atom.residue.idx) + 1,
            "residue_name": atom.residue.name,
            "sigma_nm": sigma,
            "epsilon_kj": epsilon,
        })
    return out


# ---------------------------------------------------------------------------
# NBFIX / HBOND
# ---------------------------------------------------------------------------

def _type_labels(struct) -> dict:
    labels = {}
    parm_data = struct.parm_data
    names = parm_data.get("AMBER_ATOM_TYPE", [])
    for atom in struct.atoms:
        nb = int(atom.nb_idx)
        if nb not in labels:
            labels[nb] = names[atom.idx] if atom.idx < len(names) else atom.type
    return labels


def check_hbond(parm_data) -> None:
    a = parm_data.get("HBOND_ACOEF", [])
    b = parm_data.get("HBOND_BCOEF", [])
    for x, y in zip(a, b):
        if float(x) or float(y):
            fatal("unsupported_term",
                  "10-12 (HBOND) interactions are not supported in schema "
                  "v1.0.0")


def check_nbfix(struct) -> list:
    """Return diagnostics for NBFIX-like pairs, or refuse when not allowed."""
    parm_data = struct.parm_data
    if "LENNARD_JONES_ACOEF" not in parm_data:
        fatal("parameter", "prmtop has no LENNARD_JONES_ACOEF section")
    acoef = parm_data["LENNARD_JONES_ACOEF"]
    bcoef = parm_data["LENNARD_JONES_BCOEF"]
    nbindex = parm_data["NONBONDED_PARM_INDEX"]
    ntypes = int(struct.pointers["NTYPES"])
    used = sorted({int(a.nb_idx) for a in struct.atoms})
    diag = {}
    for t in used:
        index = int(nbindex[(t - 1) * (ntypes + 1)]) - 1
        if index < 0:
            diag[t] = None
            continue
        a = float(acoef[index])
        b = float(bcoef[index])
        if a == 0.0 or b == 0.0:
            diag[t] = None
        else:
            rmin = (2.0 * a / b) ** (1.0 / 6.0)
            eps = 0.25 * b * b / a
            diag[t] = (rmin, eps)

    labels = _type_labels(struct)
    deviations = []
    for i_pos, i in enumerate(used):
        for j in used[i_pos:]:
            index = int(nbindex[(i - 1) * ntypes + (j - 1)]) - 1
            if index < 0:
                continue
            a = float(acoef[index])
            b = float(bcoef[index])
            if diag[i] is None or diag[j] is None:
                if a != 0.0 or b != 0.0:
                    deviations.append((i, j, float("inf")))
                continue
            if a == 0.0 or b == 0.0:
                deviations.append((i, j, float("inf")))
                continue
            rmin = 0.5 * (diag[i][0] + diag[j][0])
            eps = math.sqrt(diag[i][1] * diag[j][1])
            expected_a = eps * rmin ** 12
            expected_b = 2.0 * eps * rmin ** 6
            rel = max(abs((a - expected_a) / expected_a),
                      abs((b - expected_b) / expected_b))
            if rel > 1e-8:
                deviations.append((i, j, rel))

    if not deviations:
        return []

    pieces = []
    for i, j, rel in deviations[:8]:
        li = labels.get(i, str(i))
        lj = labels.get(j, str(j))
        pieces.append(f"types {i}/{li} and {j}/{lj} deviate by {rel:.3e} "
                      "relative")
    detail = "; ".join(pieces)
    if len(deviations) > 8:
        detail += f"; and {len(deviations) - 8} more pairs"

    message = f"{NBFIX_MESSAGE} ({detail})"
    if not nbfix_allowed():
        fatal("unsupported_term", message)
    return [diagnostic(
        "nbfix_ignored", "warning",
        message + "; diagonal per-atom LJ parameters were used because "
        f"{os.environ.get('FOUNDATION_EXTRACT_ALLOW_NBFIX')} is set")]


# ---------------------------------------------------------------------------
# bonded terms
# ---------------------------------------------------------------------------

def _bonds(struct):
    out = []
    for bond in struct.bonds:
        t = bond.type
        k = float(t.k)
        r0 = float(t.req)
        if not math.isfinite(k) or not math.isfinite(r0) or r0 <= 0:
            fatal("parameter", f"invalid bond type for atoms "
                               f"{bond.atom1.idx + 1}-{bond.atom2.idx + 1}")
        # foundation HarmonicBond uses U = (k/2) dr^2
        out.append({"i": bond.atom1.idx + 1, "j": bond.atom2.idx + 1,
                    "k": 2.0 * k * BOND_FACTOR, "r0_nm": r0 * 0.1})
    return out


def _angles(struct):
    out = []
    for angle in struct.angles:
        t = angle.type
        k = float(t.k)
        theta = float(t.theteq)
        if not math.isfinite(k) or not math.isfinite(theta):
            fatal("parameter", f"invalid angle type for atoms "
                               f"{angle.atom1.idx + 1}-{angle.atom2.idx + 1}-"
                               f"{angle.atom3.idx + 1}")
        # foundation HarmonicAngle uses U = (k/2) dtheta^2
        out.append({"i": angle.atom1.idx + 1, "j": angle.atom2.idx + 1,
                    "k": angle.atom3.idx + 1,
                    "k_theta": 2.0 * k * ANGLE_FACTOR,
                    "theta0_rad": math.radians(theta)})
    return out


def _urey_bradleys(struct):
    out = []
    for ub in getattr(struct, "urey_bradleys", []):
        t = ub.type
        k = float(t.k)
        r0 = float(t.req)
        if not math.isfinite(k) or not math.isfinite(r0) or r0 <= 0:
            fatal("parameter", f"invalid Urey-Bradley type for atoms "
                               f"{ub.atom1.idx + 1}-{ub.atom2.idx + 1}")
        # ChemistryFactors.UreyBradley uses U = k dr^2 (no 1/2).
        out.append({"i": ub.atom1.idx + 1, "j": ub.atom2.idx + 1,
                    "k": k * BOND_FACTOR, "r0_nm": r0 * 0.1})
    return out


def _dihedral_terms(d, path: str):
    """Return the prmtop torsion terms for one ParmEd Dihedral."""
    t = d.type
    if hasattr(t, "phi_k"):
        types = [t]
    else:
        try:
            types = list(t)
        except TypeError:
            fatal("parameter", f"{path}: cannot read dihedral type")
    terms = []
    for item in types:
        k = float(item.phi_k)
        n = int(item.per)
        phase = float(item.phase)
        if not math.isfinite(k) or not math.isfinite(phase):
            fatal("parameter", f"{path}: non-finite dihedral parameter")
        if n < 1:
            fatal("parameter", f"{path}: periodicity must be >= 1, got {n}")
        # ParmEd phi_k is already the coefficient of k*(1+cos(n phi-phase)).
        terms.append({"k": k * KCAL_TO_KJ, "n": n,
                      "phase_rad": math.radians(phase)})
    if not terms:
        fatal("parameter", f"{path}: empty dihedral type")
    return terms


def _dihedrals(struct):
    grouped = {}
    order = []
    for pos, d in enumerate(struct.dihedrals, 1):
        key = (d.atom1.idx + 1, d.atom2.idx + 1,
               d.atom3.idx + 1, d.atom4.idx + 1)
        terms = _dihedral_terms(d, f"dihedral {pos}")
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].extend(terms)
    return [{"i": key[0], "j": key[1], "k": key[2], "l": key[3],
             "terms": grouped[key]} for key in order]


def _impropers(struct):
    out = []
    for pos, imp in enumerate(getattr(struct, "impropers", []), 1):
        t = imp.type
        k = float(t.psi_k)
        phase = float(t.psi_eq)
        if not math.isfinite(k) or not math.isfinite(phase):
            fatal("parameter", f"improper {pos}: non-finite parameter")
        out.append({"i": imp.atom1.idx + 1, "j": imp.atom2.idx + 1,
                    "k": imp.atom3.idx + 1, "l": imp.atom4.idx + 1,
                    "k_phi": k * KCAL_TO_KJ,
                    "phase_rad": math.radians(phase)})
    return out


def _cmaps(struct):
    """Extract CMAP grids in the foundation/OpenMM flat-map convention.

    ``grid`` is the n*n flat OpenMM map order ``M[s + n*t]`` where

        phi_s = s*delta,   psi_t = t*delta,   delta = 2*pi/n,

    ``phi`` is the first torsion ``(a,b,c,d)`` and ``psi`` the second
    ``(b,c,d,e)``, both measured with the signed atan2 convention of
    ``Factors.dihedral_angle_grad``.  This is exactly what
    ``ChemistryCMAP.CMAPCorrection`` expects (row index = phi, column = psi,
    row-major flat).

    ParmEd's ``Cmap.type.grid`` is the raw prmtop ``CMAP_PARAMETER`` array in
    -pi..pi order, so the required transform is the AMBER parser's half-grid
    shift, with no transpose and no axis reversal:

        M[s + n*t] = P[((s+n//2) % n), ((t+n//2) % n)]      (0-based)

    Sign verification: an analytic map ``E = cos(phi) + 0.5*sin(psi)``,
    evaluated through OpenMM on both the Reference and CPU platforms, gives
    ``E(phi_here, psi_here)``, not ``E(-phi_here, -psi_here)``.  The reference
    kernel calls ``ReferenceForce::getDeltaR(a2, a1)``, which returns
    ``a1 - a2`` (not ``a2 - a1``), so its three delta vectors are all negated
    relative to the sign convention quoted in the follow-up and the resulting
    torsion equals ``Factors.dihedral_angle_grad``.  Reversing both axes would
    therefore be wrong for OpenMM 8.6.1, and is deliberately NOT applied.
    See ``test_cmap_analytic_map_uses_positive_signed_torsion``.
    """
    out = []
    for pos, cmap in enumerate(getattr(struct, "cmaps", []), 1):
        res = int(cmap.type.resolution)
        raw = [float(x) for x in cmap.type.grid]
        if res < 2 or len(raw) != res * res:
            fatal("unsupported_term",
                  f"cmap {pos}: resolution {res} does not match "
                  f"{len(raw)} grid values")
        shift = res // 2
        grid = [raw[((s + shift) % res) * res + ((t + shift) % res)]
                for t in range(res) for s in range(res)]
        atoms = []
        for name in ("atom1", "atom2", "atom3", "atom4", "atom5"):
            atom = getattr(cmap, name, None)
            if atom is None:
                fatal("unsupported_term", f"cmap {pos}: missing {name}")
            atoms.append(atom.idx + 1)
        indices = [atoms[0], atoms[1], atoms[2], atoms[3],
                   atoms[1], atoms[2], atoms[3], atoms[4]]
        out.append({"i": indices, "resolution": res,
                    "grid": [x * KCAL_TO_KJ for x in grid]})
    return out


def _cmap_section_present(parm_data) -> bool:
    return ("CMAP_COUNT" in parm_data or "CHARMM_CMAP_COUNT" in parm_data)


# ---------------------------------------------------------------------------
# 1-4 pairs and exclusions
# ---------------------------------------------------------------------------

def _pair14(struct):
    scales = {}
    for d in struct.dihedrals:
        if d.improper or d.ignore_end:
            continue
        i = d.atom1.idx + 1
        j = d.atom4.idx + 1
        key = (min(i, j), max(i, j))
        scee = float(d.type.scee)
        scnb = float(d.type.scnb)
        if scee < 0 or scnb < 0:
            fatal("parameter",
                  f"dihedral {i}-{j}: negative SCEE/SCNB scale factor")
        coulomb = 0.0 if scee == 0 else 1.0 / scee
        lj = 0.0 if scnb == 0 else 1.0 / scnb
        if key in scales:
            old = scales[key]
            if abs(old[0] - coulomb) > 1e-12 or abs(old[1] - lj) > 1e-12:
                fatal("parameter",
                      f"1-4 pair {key} has conflicting SCEE/SCNB scales "
                      f"({old} vs {(coulomb, lj)})")
        else:
            scales[key] = (coulomb, lj)
    return [{"i": key[0], "j": key[1],
             "coulomb_scale": scales[key][0],
             "lj_scale": scales[key][1]}
            for key in sorted(scales)]


def _raw_exclusions(parm_data):
    counts = parm_data.get("NUMBER_EXCLUDED_ATOMS")
    listed = parm_data.get("EXCLUDED_ATOMS_LIST")
    if counts is None or listed is None:
        return set()
    counts = [int(x) for x in counts]
    listed = [int(x) for x in listed]
    if sum(counts) != len(listed):
        fatal("parameter",
              "NUMBER_EXCLUDED_ATOMS does not match EXCLUDED_ATOMS_LIST")
    pairs = set()
    pos = 0
    for atom0, count in enumerate(counts):
        for j in listed[pos:pos + count]:
            pos += 1
            if j <= 0:
                continue
            i = atom0 + 1
            pairs.add((min(i, j), max(i, j)))
    return pairs


def _exclusions(struct, pair14):
    pairs = _raw_exclusions(struct.parm_data)
    pair14_keys = {(p["i"], p["j"]) for p in pair14}
    for key in pair14_keys:
        if key in pairs:
            pairs.discard(key)
    return [list(key) for key in sorted(pairs)]


def _check_pair14_not_bonded(struct, pair14) -> None:
    bonded12 = {(min(b.atom1.idx + 1, b.atom2.idx + 1),
                 max(b.atom1.idx + 1, b.atom2.idx + 1))
                for b in struct.bonds}
    bonded13 = {(min(a.atom1.idx + 1, a.atom3.idx + 1),
                 max(a.atom1.idx + 1, a.atom3.idx + 1))
                for a in struct.angles}
    for p in pair14:
        key = (p["i"], p["j"])
        if key in bonded12 or key in bonded13:
            fatal("parameter",
                  f"1-4 pair {key} is also a 1-2 or 1-3 pair; the prmtop "
                  "dihedral and exclusion lists are inconsistent")


# ---------------------------------------------------------------------------
# record
# ---------------------------------------------------------------------------


def _lj146(struct, pair14, exclusions):
    """AMBER 1-6-4 coefficients as a nonbonded pair list.

    prmtop `LENNARD_JONES_CCOEF` entries are kcal/mol * A^4; the normalized
    schema stores kJ/mol * nm^4. OpenMM applies `-c/r^4` only to pairs that are
    not NonbondedForce exceptions, so excluded and 1-4 pairs are removed here.
    """
    parm_data = struct.parm_data
    if "LENNARD_JONES_CCOEF" not in parm_data:
        return []
    ccoef = parm_data["LENNARD_JONES_CCOEF"]
    if not any(float(x) != 0.0 for x in ccoef):
        return []
    ntypes = int(max(parm_data["ATOM_TYPE_INDEX"]))
    nbidx = [int(x) for x in parm_data["NONBONDED_PARM_INDEX"]]
    type_of = {}
    for a in struct.atoms:
        # ParmEd exposes the AMBER type index through atom.type or the raw array.
        type_of[a.idx] = int(parm_data["ATOM_TYPE_INDEX"][a.idx]) - 1
    # Canonical type pairs with nonzero C.
    type_c = {}
    for t1 in range(ntypes):
        for t2 in range(t1, ntypes):
            idx = nbidx[ntypes * t1 + t2] - 1
            if idx < 0:
                continue
            c = float(ccoef[idx])
            if c != 0.0:
                type_c[(t1, t2)] = c * 4.184e-4      # kcal/A^4 -> kJ/nm^4
    if not type_c:
        return []
    excepted = {(min(i, j), max(i, j)) for (i, j) in exclusions}
    excepted |= {(min(p["i"], p["j"]), max(p["i"], p["j"])) for p in pair14}
    out = []
    by_type = {}
    for a in struct.atoms:
        by_type.setdefault(type_of[a.idx], []).append(a.idx)
    for (t1, t2), c in sorted(type_c.items()):
        for i0 in by_type.get(t1, []):
            for j0 in by_type.get(t2, []):
                if i0 >= j0:
                    continue
                i, j = i0 + 1, j0 + 1
                if (i, j) in excepted:
                    continue
                out.append({"i": i, "j": j, "c": c})
    return out

def extract(parm: str, crd: str, force_field: str,
            repo_root: str | None = None) -> dict:
    struct = load_structure(parm, crd)
    parm_data = struct.parm_data

    check_hbond(parm_data)
    nbfix_diags = check_nbfix(struct)

    atoms = _atoms(struct)
    box = _prmtop_box(parm_data)
    diagnostics = []
    coord = _amber_coord_box(struct)
    box_diag = _box_diagnostic(box, coord)
    if box_diag is not None:
        diagnostics.append(box_diag)
    diagnostics.extend(nbfix_diags)

    pair14 = _pair14(struct)
    _check_pair14_not_bonded(struct, pair14)
    exclusions = _exclusions(struct, pair14)
    lj146 = _lj146(struct, pair14, exclusions)

    cmaps = _cmaps(struct)
    if _cmap_section_present(parm_data) and not cmaps:
        fatal("unsupported_term",
              "the prmtop declares a CMAP section but ParmEd produced no "
              "cmap terms; refusing to drop it")

    impropers = _impropers(struct)
    if "CHARMM_IMPROPERS" in parm_data and len(impropers) != len(
            parm_data["CHARMM_IMPROPERS"]) // 5:
        fatal("parameter", "CHARMM_IMPROPERS count does not match the parsed "
                           "improper list")

    title = "".join(parm_data.get("TITLE", [])).strip()
    record = {
        "schema": SCHEMA,
        "family": "amber",
        "units": UNITS,
        "provenance": {
            "source": relative_source(parm, repo_root),
            "force_field": force_field,
            "source_sha256": sha256_file(parm),
            "coordinate_source": relative_source(crd, repo_root),
            "coordinate_sha256": sha256_file(crd),
            "generator": GENERATOR,
            "generator_version": GENERATOR_VERSION,
            "notes": title,
        },
        "atoms": atoms,
        "box": box,
        "bonds": _bonds(struct),
        "angles": _angles(struct),
        "urey_bradleys": _urey_bradleys(struct),
        "dihedrals": _dihedrals(struct),
        "impropers": impropers,
        "cmaps": cmaps,
        "exclusions": exclusions,
        "pair14": pair14,
        "lj146": lj146,
        "virtual_sites": [],
        "constraints": [],
        "input_diagnostics": diagnostics,
        "net_charge": sum(a["charge"] for a in atoms),
    }
    return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="AMBER prmtop/inpcrd -> mdparams 1.0.0 JSON")
    parser.add_argument("--parm", required=True)
    parser.add_argument("--crd", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--force-field", required=True)
    parser.add_argument("--repo-root", default=None)
    args = parser.parse_args(argv)
    try:
        record = extract(args.parm, args.crd, args.force_field, args.repo_root)
        return finish(record, args.out)
    except ExtractionError as exc:
        return cli_error(exc)
    except Exception as exc:  # pragma: no cover - last-resort typed failure
        return cli_error(ExtractionError("io", f"{type(exc).__name__}: {exc}"))


if __name__ == "__main__":
    sys.exit(main())
