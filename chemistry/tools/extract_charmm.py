#!/usr/bin/env python3
"""CHARMM PSF + parameter files (+ optional coordinates) -> mdparams 1.0.0.

Coefficient conventions follow ``chemistry/INTERFACE.md`` and the
foundation factor forms:

* bonds / angles: ``U = (k/2) delta^2``.  ParmEd's ``BondType.k`` and
  ``AngleType.k`` are the CHARMM ``K`` of ``U = K delta^2``, so the schema
  value is ``2 * K`` after the unit conversion (418.4 for bonds,
  4.184 for angles).
* Urey-Bradley: ``U = k (r - r0)^2`` (no one-half), so ``k = K * 418.4``.
* dihedrals: ``U = k (1 + cos(n phi - phase))``.  ``DihedralType.phi_k`` is
  already the coefficient of that form, so ``k = phi_k * 4.184``.
* impropers: ``U = k (wrap(phi - phase))^2``.  ``ImproperType.psi_k`` is the
  native CHARMM coefficient (possibly negative), so ``k = psi_k * 4.184`` and
  ``phase = radians(psi_eq)``.

CHARMM atom sigma/epsilon are read from ParmEd's ``Atom.sigma`` / ``.epsilon``,
which are in angstrom and kcal/mol (and epsilon is negative for the usual
CHARMM convention); they are converted to nm and kJ/mol with the sign removed.

NBFIX is refused by default.  The tests may set
``FOUNDATION_EXTRACT_ALLOW_NBFIX=1`` to continue for a fixture that contains a
pair-specific ion term; in that case an ``nbfix_ignored`` warning naming the
type pair is written into ``input_diagnostics`` instead of dropping it
silently.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

from common import (ANGLE_FACTOR, BOND_FACTOR, ExtractionError, KCAL_TO_KJ,
                    SCHEMA, UNITS, cli_error, diagnostic, fatal, finish,
                    nbfix_allowed, relative_source, sha256_file)

GENERATOR = "chemistry/tools/extract_charmm.py"
GENERATOR_VERSION = "1.0.0"
NBFIX_MESSAGE = "NBFIX/off-diagonal LJ is not supported in schema v1.0.0"


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def load_psf(parm: str, prm_files):
    try:
        from parmed.charmm import CharmmPsfFile, CharmmParameterSet
    except Exception as exc:  # pragma: no cover
        fatal("io", f"cannot import ParmEd: {exc}")
    try:
        psf = CharmmPsfFile(parm)
    except Exception as exc:
        fatal("io", f"cannot read PSF {parm}: {exc}")
    try:
        prm = CharmmParameterSet(*prm_files)
    except Exception as exc:
        fatal("parameter", f"cannot read CHARMM parameters: {exc}")
    try:
        psf.load_parameters(prm)
    except Exception as exc:
        fatal("parameter",
              "CHARMM parameter assignment failed (numeric atom types or a "
              f"missing parameter are the usual causes): {exc}")
    return psf, prm


def load_coordinates(psf, crd: str) -> None:
    """Attach coordinates and a box from a PDB or CHARMM CRD file."""
    if crd is None:
        return
    suffix = os.path.splitext(crd)[1].lower()
    coords = None
    box = None
    try:
        if suffix in (".pdb", ".ent"):
            import parmed
            other = parmed.load_file(crd)
            if len(other.atoms) != len(psf.atoms):
                fatal("shape", f"{crd} has {len(other.atoms)} atoms; PSF has "
                               f"{len(psf.atoms)}")
            for a, b in zip(psf.atoms, other.atoms):
                if a.name.strip() != b.name.strip():
                    fatal("parameter",
                          f"coordinate atom order does not match the PSF "
                          f"({a.name} vs {b.name})")
            coords = other.coordinates
            box = other.box
        else:
            from parmed.charmm import CharmmCrdFile, CharmmRstFile
            reader = CharmmRstFile if suffix == ".rst" else CharmmCrdFile
            other = reader(crd)
            coords = other.coordinates
            if coords is not None and getattr(coords, "ndim", 2) == 3:
                coords = coords[0]
            box = getattr(other, "box", None)
    except ExtractionError:
        raise
    except Exception as exc:
        fatal("io", f"cannot read coordinates {crd}: {exc}")

    if coords is None:
        fatal("io", f"{crd} contains no coordinates")
    if len(coords) != len(psf.atoms):
        fatal("shape", f"{crd} has {len(coords)} coordinates; PSF has "
                       f"{len(psf.atoms)}")
    psf.coordinates = coords
    if box is not None:
        psf.box = box


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
    return [list(va), list(vb), [cx, cy, math.sqrt(cz2)]]


def _box(psf) -> dict:
    box = getattr(psf, "box", None)
    if box is None or len(box) < 3:
        return {"kind": "none"}
    a, b, c = (float(x) for x in box[:3])
    if len(box) >= 6:
        alpha, beta, gamma = (float(x) for x in box[3:6])
    else:
        alpha = beta = gamma = 90.0
    if min(a, b, c) <= 0:
        fatal("box", "coordinate box has a non-positive length")
    if abs(alpha - 90) < 1e-6 and abs(beta - 90) < 1e-6 and abs(gamma - 90) < 1e-6:
        return {"kind": "orthorhombic",
                "lengths_nm": [a * 0.1, b * 0.1, c * 0.1]}
    a, b, c = a * 0.1, b * 0.1, c * 0.1
    return {"kind": "triclinic",
            "vectors_nm": _box_vectors(alpha, a, b, c, beta, gamma)}


# ---------------------------------------------------------------------------
# atoms / NBFIX
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


def _atoms(psf):
    out = []
    for atom in psf.atoms:
        if float(atom.mass) == 0.0 or int(getattr(atom, "atomic_number", 0) or 0) == 0:
            fatal("unsupported_term",
                  f"atom {atom.idx + 1} ({atom.name}) is massless or has no "
                  "atomic number; lone pairs/virtual sites are not supported")
        sigma = getattr(atom, "sigma", None)
        epsilon = getattr(atom, "epsilon", None)
        if sigma is None or epsilon is None:
            fatal("parameter",
                  f"atom {atom.idx + 1} ({atom.name}) has no LJ parameters; "
                  "a CHARMM atom type was not resolved")
        sigma = float(sigma)
        epsilon = float(epsilon)
        if not math.isfinite(sigma) or not math.isfinite(epsilon) or sigma < 0:
            fatal("parameter",
                  f"atom {atom.idx + 1} ({atom.name}) has invalid LJ parameters")
        out.append({
            "name": atom.name,
            "mass": float(atom.mass),
            "charge": float(atom.charge),
            "element": _element(atom),
            "residue_index": int(atom.residue.idx) + 1,
            "residue_name": atom.residue.name,
            # ParmEd stores CHARMM sigma in angstrom and epsilon in kcal/mol.
            "sigma_nm": sigma * 0.1,
            "epsilon_kj": abs(epsilon) * KCAL_TO_KJ,
        })
    return out


def check_nbfix(psf, prm) -> list:
    nbfix = getattr(prm, "nbfix_types", {}) or {}
    if not nbfix:
        return []
    used = {str(atom.type).upper() for atom in psf.atoms}
    hits = []
    for key in nbfix:
        if not isinstance(key, tuple) or len(key) != 2:
            continue
        left, right = str(key[0]).upper(), str(key[1]).upper()
        if left in used and right in used:
            hits.append((str(key[0]), str(key[1])))
    if not hits:
        return []
    detail = ", ".join(f"{a}-{b}" for a, b in sorted(hits)[:8])
    if len(hits) > 8:
        detail += f", and {len(hits) - 8} more"
    message = f"{NBFIX_MESSAGE} (CHARMM NBFIX pairs in this system: {detail})"
    if not nbfix_allowed():
        fatal("unsupported_term", message)
    return [diagnostic(
        "nbfix_ignored", "warning",
        message + "; diagonal per-atom LJ parameters were used because "
        f"{os.environ.get('FOUNDATION_EXTRACT_ALLOW_NBFIX')} is set")]


# ---------------------------------------------------------------------------
# bonded terms
# ---------------------------------------------------------------------------

def _bonds(psf):
    out = []
    for bond in psf.bonds:
        k = float(bond.type.k)
        r0 = float(bond.type.req)
        if not math.isfinite(k) or not math.isfinite(r0) or r0 <= 0:
            fatal("parameter", f"invalid bond type for atoms "
                               f"{bond.atom1.idx + 1}-{bond.atom2.idx + 1}")
        out.append({"i": bond.atom1.idx + 1, "j": bond.atom2.idx + 1,
                    "k": 2.0 * k * BOND_FACTOR, "r0_nm": r0 * 0.1})
    return out


def _angles(psf):
    out = []
    for angle in psf.angles:
        k = float(angle.type.k)
        theta = float(angle.type.theteq)
        if not math.isfinite(k) or not math.isfinite(theta):
            fatal("parameter", f"invalid angle type for atoms "
                               f"{angle.atom1.idx + 1}-{angle.atom2.idx + 1}-"
                               f"{angle.atom3.idx + 1}")
        out.append({"i": angle.atom1.idx + 1, "j": angle.atom2.idx + 1,
                    "k": angle.atom3.idx + 1,
                    "k_theta": 2.0 * k * ANGLE_FACTOR,
                    "theta0_rad": math.radians(theta)})
    return out


def _urey_bradleys(psf):
    out = []
    for ub in getattr(psf, "urey_bradleys", []):
        k = float(ub.type.k)
        r0 = float(ub.type.req)
        if not math.isfinite(k) or not math.isfinite(r0) or r0 <= 0:
            fatal("parameter", f"invalid Urey-Bradley type for atoms "
                               f"{ub.atom1.idx + 1}-{ub.atom2.idx + 1}")
        out.append({"i": ub.atom1.idx + 1, "j": ub.atom2.idx + 1,
                    "k": k * BOND_FACTOR, "r0_nm": r0 * 0.1})
    return out


def _dihedral_terms(d, path: str):
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
        terms.append({"k": k * KCAL_TO_KJ, "n": n,
                      "phase_rad": math.radians(phase)})
    if not terms:
        fatal("parameter", f"{path}: empty dihedral type")
    return terms


def _dihedrals(psf):
    grouped = {}
    order = []
    for pos, d in enumerate(psf.dihedrals, 1):
        key = (d.atom1.idx + 1, d.atom2.idx + 1,
               d.atom3.idx + 1, d.atom4.idx + 1)
        terms = _dihedral_terms(d, f"dihedral {pos}")
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].extend(terms)
    return [{"i": key[0], "j": key[1], "k": key[2], "l": key[3],
             "terms": grouped[key]} for key in order]


def _impropers(psf):
    out = []
    for pos, imp in enumerate(getattr(psf, "impropers", []), 1):
        k = float(imp.type.psi_k)
        phase = float(imp.type.psi_eq)
        if not math.isfinite(k) or not math.isfinite(phase):
            fatal("parameter", f"improper {pos}: non-finite parameter")
        out.append({"i": imp.atom1.idx + 1, "j": imp.atom2.idx + 1,
                    "k": imp.atom3.idx + 1, "l": imp.atom4.idx + 1,
                    "k_phi": k * KCAL_TO_KJ,
                    "phase_rad": math.radians(phase)})
    return out


def _cmaps(psf):
    out = []
    for pos, cmap in enumerate(getattr(psf, "cmaps", []), 1):
        res = int(cmap.type.resolution)
        grid = list(cmap.type.grid)
        if res < 2 or len(grid) != res * res:
            fatal("unsupported_term",
                  f"cmap {pos}: resolution {res} does not match "
                  f"{len(grid)} grid values")
        atoms = []
        for name in ("atom1", "atom2", "atom3", "atom4", "atom5"):
            atom = getattr(cmap, name, None)
            if atom is None:
                fatal("unsupported_term", f"cmap {pos}: missing {name}")
            atoms.append(atom.idx + 1)
        out.append({"i": [atoms[0], atoms[1], atoms[2], atoms[3],
                          atoms[1], atoms[2], atoms[3], atoms[4]],
                    "resolution": res,
                    "grid": [float(x) * KCAL_TO_KJ for x in grid]})
    return out


# ---------------------------------------------------------------------------
# 1-4 pairs and exclusions
# ---------------------------------------------------------------------------

def _pair14(psf):
    pairs = set()
    for d in psf.dihedrals:
        i = d.atom1.idx + 1
        j = d.atom4.idx + 1
        pairs.add((min(i, j), max(i, j)))
    return [{"i": i, "j": j, "coulomb_scale": 1.0, "lj_scale": 1.0}
            for i, j in sorted(pairs)]


def _exclusions(psf, pair14):
    pairs = set()
    for bond in psf.bonds:
        i, j = bond.atom1.idx + 1, bond.atom2.idx + 1
        pairs.add((min(i, j), max(i, j)))
    for angle in psf.angles:
        i, j = angle.atom1.idx + 1, angle.atom3.idx + 1
        pairs.add((min(i, j), max(i, j)))
    pair14_keys = {(p["i"], p["j"]) for p in pair14}
    for key in pair14_keys:
        if key in pairs:
            fatal("parameter",
                  f"1-4 pair {key} is also a 1-2 or 1-3 pair; the PSF "
                  "dihedral and angle lists are inconsistent")
    return [list(key) for key in sorted(pairs - pair14_keys)]


# ---------------------------------------------------------------------------
# record
# ---------------------------------------------------------------------------

def extract(parm: str, prm_files, crd: str | None, force_field: str,
            repo_root: str | None = None) -> dict:
    psf, prm = load_psf(parm, prm_files)
    load_coordinates(psf, crd)

    nbfix_diags = check_nbfix(psf, prm)
    atoms = _atoms(psf)
    pair14 = _pair14(psf)
    exclusions = _exclusions(psf, pair14)

    title = ""
    psf_title = getattr(psf, "title", None)
    if psf_title:
        title = "; ".join(str(line).strip() for line in psf_title if str(line).strip())

    record = {
        "schema": SCHEMA,
        "family": "charmm",
        "units": UNITS,
        "provenance": {
            "source": relative_source(parm, repo_root),
            "force_field": force_field,
            "source_sha256": sha256_file(parm),
            "coordinate_source": (relative_source(crd, repo_root)
                                  if crd else ""),
            "coordinate_sha256": sha256_file(crd) if crd else "",
            "generator": GENERATOR,
            "generator_version": GENERATOR_VERSION,
            "notes": title,
        },
        "atoms": atoms,
        "box": _box(psf),
        "bonds": _bonds(psf),
        "angles": _angles(psf),
        "urey_bradleys": _urey_bradleys(psf),
        "dihedrals": _dihedrals(psf),
        "impropers": _impropers(psf),
        "cmaps": _cmaps(psf),
        "exclusions": exclusions,
        "pair14": pair14,
        "lj146": [],
        "virtual_sites": [],
        "constraints": [],
        "input_diagnostics": nbfix_diags,
        "net_charge": sum(a["charge"] for a in atoms),
    }
    return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="CHARMM PSF + parameter files -> mdparams 1.0.0 JSON")
    parser.add_argument("--parm", required=True, help="CHARMM PSF file")
    parser.add_argument("--prm", action="append", required=True,
                        help="CHARMM parameter/stream file (repeatable)")
    parser.add_argument("--crd", default=None,
                        help="optional PDB or CHARMM CRD coordinate file")
    parser.add_argument("--out", required=True)
    parser.add_argument("--force-field", required=True)
    parser.add_argument("--repo-root", default=None)
    args = parser.parse_args(argv)
    try:
        record = extract(args.parm, args.prm, args.crd, args.force_field,
                         args.repo_root)
        return finish(record, args.out)
    except ExtractionError as exc:
        return cli_error(exc)
    except Exception as exc:  # pragma: no cover
        return cli_error(ExtractionError("io", f"{type(exc).__name__}: {exc}"))


if __name__ == "__main__":
    sys.exit(main())
