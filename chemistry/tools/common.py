"""Shared helpers for the Phase-2 parameter-file extractors.

The record produced here is the ``mdparams`` schema 1.0.0 described by
``chemistry/INTERFACE.md``: nm, kJ/mol, elementary charge, amu.
This module deliberately contains no engine code.  ``validate_record`` is a
Python gate in addition to the Julia parser in ``ChemistrySchema.jl`` so that
a malformed extractor cannot write a file that only fails much later.

All numbers are written with ``json.dump(..., sort_keys=True,
separators=(',', ':'), allow_nan=False)`` plus a trailing newline, which is the
deterministic serialization required by the interface contract.

Coefficient conventions (they are not interchangeable):

* ``bonds`` / ``angles`` are coefficients of ``U = (k/2) delta^2``, so the
  extractors double ParmEd's native ``BondType.k`` / ``AngleType.k`` after the
  unit conversion (``418.4`` and ``4.184`` respectively).  OpenMM's AMBER and
  CHARMM parsers add ``2*k`` to ``HarmonicBondForce``/``HarmonicAngleForce``
  for exactly that reason.
* ``dihedrals`` are coefficients of ``k (1 + cos(n phi - phase))``.  ParmEd's
  ``DihedralType.phi_k`` is already that coefficient (the prmtop ``Vn/2``), so
  converting kcal -> kJ is the only change.
* ``impropers`` are coefficients of ``k (wrap(phi - phase))^2``.  ParmEd's
  ``ImproperType.psi_k`` is the native coefficient; convert kcal -> kJ and
  keep the sign (CHARMM may use a negative ``psi_k``).
* ``urey_bradleys`` are coefficients of ``k (r - r0)^2`` (no one-half).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys

SCHEMA = "1.0.0"
UNITS = "nm-kj-mol-e-amu"
KCAL_TO_KJ = 4.184
ANGSTROM_TO_NM = 0.1
# 1 kcal/mol/A^2 = 4.184 kJ/mol / (0.1 nm)^2
BOND_FACTOR = KCAL_TO_KJ / ANGSTROM_TO_NM ** 2  # 418.4
ANGLE_FACTOR = KCAL_TO_KJ

TOP_KEYS = (
    "schema", "family", "units", "provenance", "atoms", "box", "bonds",
    "angles", "urey_bradleys", "dihedrals", "impropers", "cmaps",
    "exclusions", "pair14", "lj146", "virtual_sites", "constraints",
    "input_diagnostics", "net_charge",
)

LJ146_KEYS = ("i", "j", "c")

ATOM_KEYS = ("name", "mass", "charge", "element", "residue_index",
             "residue_name", "sigma_nm", "epsilon_kj")
PROVENANCE_KEYS = ("source", "force_field", "source_sha256",
                   "coordinate_source", "coordinate_sha256", "generator",
                   "generator_version", "notes")
DIAGNOSTIC_KEYS = ("code", "severity", "message", "atoms", "residues")

# The override exists only so the extractor tests can exercise fixtures whose
# parameters contain a pair-specific LJ term that schema v1.0.0 cannot encode.
# The default is a hard refusal; when the override is set the extractor records
# a warning naming the affected atom-type pair instead of using it silently.
ALLOW_NBFIX_ENV = "FOUNDATION_EXTRACT_ALLOW_NBFIX"


class ExtractionError(Exception):
    """Typed extraction refusal with a schema error code."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class ValidationError(ValueError):
    """Raised by :func:`validate_record`; also a ``ValueError`` for tests."""

    def __init__(self, path: str, message: str):
        super().__init__(f"{path}: {message}")
        self.path = path
        self.message = message


def nbfix_allowed() -> bool:
    return os.environ.get(ALLOW_NBFIX_ENV, "").strip() == "1"


def fatal(code: str, message: str) -> "None":
    raise ExtractionError(code, message)


def _fail(path: str, message: str) -> "None":
    raise ValidationError(path, message)


def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _number(x, path: str, *, positive: bool = False,
            nonnegative: bool = False) -> float:
    if not _is_number(x):
        _fail(path, "expected a number")
    v = float(x)
    if not math.isfinite(v):
        _fail(path, "must be finite")
    if positive and not v > 0:
        _fail(path, "must be > 0")
    if nonnegative and v < 0:
        _fail(path, "must be >= 0")
    return v


def _integer(x, path: str, *, minimum: int = -(2 ** 63)) -> int:
    if not isinstance(x, int) or isinstance(x, bool):
        _fail(path, "expected an integer")
    if x < minimum:
        _fail(path, f"must be >= {minimum}")
    return x


def _string(x, path: str, *, nonempty: bool = False) -> str:
    if not isinstance(x, str):
        _fail(path, "expected a string")
    if nonempty and not x.strip():
        _fail(path, "must be non-empty")
    return x


def _object(x, path: str) -> dict:
    if not isinstance(x, dict):
        _fail(path, "expected an object")
    return x


def _array(x, path: str) -> list:
    if not isinstance(x, list):
        _fail(path, "expected an array")
    return x


def _keys(d: dict, allowed, path: str) -> None:
    for key in d:
        if key not in allowed:
            _fail(path, f"unknown key '{key}'")
    for key in allowed:
        if key not in d:
            _fail(path, f"missing required key '{key}'")


def _canonical_pair(i: int, j: int) -> tuple:
    if i == j:
        _fail("pair", "pair must reference distinct atoms")
    return (i, j) if i < j else (j, i)


def _check_index(i: int, n: int, path: str) -> None:
    if not 1 <= i <= n:
        _fail(path, f"atom index {i} is outside 1:{n}")


def _check_support(indices, n: int, path: str) -> None:
    if len(set(indices)) != len(indices):
        _fail(path, "support repeats an atom index")
    for i in indices:
        _check_index(i, n, path)


def _check_duplicates(keys, path: str) -> None:
    seen = set()
    for key in keys:
        if key in seen:
            _fail(path, f"duplicate term {key}")
        seen.add(key)


# ---------------------------------------------------------------------------
# file helpers
# ---------------------------------------------------------------------------

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def relative_source(path: str, repo_root: str | None) -> str:
    absolute = os.path.abspath(path)
    if repo_root is None:
        return absolute.replace(os.sep, "/")
    root = os.path.abspath(repo_root)
    try:
        inside = os.path.commonpath([absolute, root]) == root
    except ValueError:
        inside = False
    rel = os.path.relpath(absolute, root) if inside else absolute
    return rel.replace(os.sep, "/")


def json_text(record: dict) -> str:
    return json.dumps(record, sort_keys=True, separators=(",", ":"),
                      allow_nan=False) + "\n"


def write_record(record: dict, path: str) -> str:
    validate_record(record)
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    text = json_text(record)
    with open(path, "w", encoding="ascii", newline="\n") as fh:
        fh.write(text)
    return path


def read_record(path: str) -> dict:
    with open(path, "r", encoding="ascii") as fh:
        return json.load(fh)


def diagnostic(code: str, severity: str, message: str,
               atoms=None, residues=None) -> dict:
    return {
        "code": code,
        "severity": severity,
        "message": message,
        "atoms": list(atoms or []),
        "residues": list(residues or []),
    }


def finish(record: dict, out_path: str) -> int:
    """Validate, write, and print only the output path (the CLI contract)."""
    write_record(record, out_path)
    print(out_path)
    return 0


def cli_error(exc: ExtractionError) -> int:
    print(f"error[{exc.code}]: {exc.message}", file=sys.stderr)
    return 2


# ---------------------------------------------------------------------------
# schema validation
# ---------------------------------------------------------------------------

def validate_record(d: dict) -> dict:
    """Validate one normalized record and return it.

    This mirrors the checks in ``ChemistrySchema.jl`` that can be applied
    without an engine: exact key sets, unit strings, 1-based indices,
    duplicate detection, exclusion/1-4 disjointness, finite numbers, sha256
    shape, and the atom-charge sum.  It raises :class:`ValidationError`.
    """
    d = _object(d, "<root>")
    _keys(d, TOP_KEYS, "<root>")

    _string(d["schema"], "schema", nonempty=True)
    if d["schema"] != SCHEMA:
        _fail("schema", f"expected '{SCHEMA}', got '{d['schema']}'")
    _string(d["family"], "family", nonempty=True)
    if d["family"] not in ("amber", "charmm"):
        _fail("family", "must be amber or charmm")
    _string(d["units"], "units", nonempty=True)
    if d["units"] != UNITS:
        _fail("units", f"expected '{UNITS}', got '{d['units']}'")

    prov = _object(d["provenance"], "provenance")
    _keys(prov, PROVENANCE_KEYS, "provenance")
    _string(prov["source"], "provenance.source", nonempty=True)
    _string(prov["force_field"], "provenance.force_field", nonempty=True)
    for key in ("source_sha256", "coordinate_sha256"):
        s = _string(prov[key], f"provenance.{key}")
        if s and not (len(s) == 64 and all(c in "0123456789abcdef" for c in s)):
            _fail(f"provenance.{key}", "must be 64 lowercase hex characters")
    _string(prov["coordinate_source"], "provenance.coordinate_source")
    _string(prov["generator"], "provenance.generator")
    _string(prov["generator_version"], "provenance.generator_version")
    _string(prov["notes"], "provenance.notes")

    pair14 = _array(d["pair14"], "pair14")
    pair14_keys = set()
    for k, term in enumerate(pair14, 1):
        path = f"pair14[{k}]"
        term = _object(term, path)
        _keys(term, ("i", "j", "coulomb_scale", "lj_scale"), path)
        i = _integer(term["i"], f"{path}.i", minimum=1)
        j = _integer(term["j"], f"{path}.j", minimum=1)
        if i == j:
            _fail(path, "pair must reference distinct atoms")
        pair14_keys.add((min(i, j), max(i, j)))
        for key in ("coulomb_scale", "lj_scale"):
            value = _number(term[key], f"{path}.{key}", nonnegative=True)
            if value > 1:
                _fail(f"{path}.{key}", "scale must lie in [0, 1]")

    lj146 = _array(d["lj146"], "lj146")
    lj146_keys = set()
    for k, term in enumerate(lj146, 1):
        path = f"lj146[{k}]"
        term = _object(term, path)
        _keys(term, LJ146_KEYS, path)
        i = _integer(term["i"], f"{path}.i", minimum=1)
        j = _integer(term["j"], f"{path}.j", minimum=1)
        if i == j:
            _fail(path, "pair must reference distinct atoms")
        key = (min(i, j), max(i, j))
        if key in lj146_keys:
            _fail(path, "duplicate 1-6-4 pair")
        if key in pair14_keys or key in {tuple(p) for p in d["exclusions"]}:
            _fail(path, "1-6-4 pair overlaps an exclusion or 1-4 pair")
        lj146_keys.add(key)
        _number(term["c"], f"{path}.c")

    atoms = _array(d["atoms"], "atoms")
    n = len(atoms)
    if n < 1:
        _fail("atoms", "at least one atom is required")
    last_res = 0
    charge_sum = 0.0
    for k, atom in enumerate(atoms, 1):
        path = f"atoms[{k}]"
        atom = _object(atom, path)
        _keys(atom, ATOM_KEYS, path)
        _string(atom["name"], f"{path}.name", nonempty=True)
        _number(atom["mass"], f"{path}.mass", positive=True)
        charge_sum += _number(atom["charge"], f"{path}.charge")
        _string(atom["element"], f"{path}.element", nonempty=True)
        res = _integer(atom["residue_index"], f"{path}.residue_index", minimum=1)
        if res < last_res:
            _fail(f"{path}.residue_index",
                  "residue indices must be non-decreasing in atom order")
        last_res = res
        _string(atom["residue_name"], f"{path}.residue_name", nonempty=True)
        _number(atom["sigma_nm"], f"{path}.sigma_nm", nonnegative=True)
        _number(atom["epsilon_kj"], f"{path}.epsilon_kj", nonnegative=True)

    box = _object(d["box"], "box")
    kind = _string(box.get("kind"), "box.kind", nonempty=True)
    if kind == "none":
        _keys(box, ("kind",), "box")
    elif kind == "orthorhombic":
        _keys(box, ("kind", "lengths_nm"), "box")
        lengths = _array(box["lengths_nm"], "box.lengths_nm")
        if len(lengths) != 3:
            _fail("box.lengths_nm", "must have three entries")
        for m, value in enumerate(lengths, 1):
            _number(value, f"box.lengths_nm[{m}]", positive=True)
    elif kind == "triclinic":
        _keys(box, ("kind", "vectors_nm"), "box")
        rows = _array(box["vectors_nm"], "box.vectors_nm")
        if len(rows) != 3:
            _fail("box.vectors_nm", "must have three rows")
        for m, row in enumerate(rows, 1):
            row = _array(row, f"box.vectors_nm[{m}]")
            if len(row) != 3:
                _fail(f"box.vectors_nm[{m}]", "must have three entries")
            for q, value in enumerate(row, 1):
                _number(value, f"box.vectors_nm[{m}][{q}]")
        a, b, c = rows
        det = (a[0] * (b[1] * c[2] - b[2] * c[1])
               - a[1] * (b[0] * c[2] - b[2] * c[0])
               + a[2] * (b[0] * c[1] - b[1] * c[0]))
        if not det > 0:
            _fail("box.vectors_nm", "cell must have positive volume")
    else:
        _fail("box.kind", "must be none/orthorhombic/triclinic")

    bonds = _array(d["bonds"], "bonds")
    for k, term in enumerate(bonds, 1):
        path = f"bonds[{k}]"
        term = _object(term, path)
        _keys(term, ("i", "j", "k", "r0_nm"), path)
        i = _integer(term["i"], f"{path}.i", minimum=1)
        j = _integer(term["j"], f"{path}.j", minimum=1)
        _check_support((i, j), n, path)
        _number(term["k"], f"{path}.k")
        _number(term["r0_nm"], f"{path}.r0_nm", positive=True)
    _check_duplicates(((t["i"], t["j"]) for t in bonds), "bonds")

    angles = _array(d["angles"], "angles")
    for k, term in enumerate(angles, 1):
        path = f"angles[{k}]"
        term = _object(term, path)
        _keys(term, ("i", "j", "k", "k_theta", "theta0_rad"), path)
        i = _integer(term["i"], f"{path}.i", minimum=1)
        j = _integer(term["j"], f"{path}.j", minimum=1)
        kk = _integer(term["k"], f"{path}.k", minimum=1)
        _check_support((i, j, kk), n, path)
        _number(term["k_theta"], f"{path}.k_theta")
        theta = _number(term["theta0_rad"], f"{path}.theta0_rad")
        if not 0 < theta <= math.pi:
            _fail(f"{path}.theta0_rad", "must lie in (0, pi]")
    _check_duplicates(((t["i"], t["j"], t["k"]) for t in angles), "angles")

    ureys = _array(d["urey_bradleys"], "urey_bradleys")
    for k, term in enumerate(ureys, 1):
        path = f"urey_bradleys[{k}]"
        term = _object(term, path)
        _keys(term, ("i", "j", "k", "r0_nm"), path)
        i = _integer(term["i"], f"{path}.i", minimum=1)
        j = _integer(term["j"], f"{path}.j", minimum=1)
        _check_support((i, j), n, path)
        _number(term["k"], f"{path}.k")
        _number(term["r0_nm"], f"{path}.r0_nm", positive=True)
    _check_duplicates(((t["i"], t["j"]) for t in ureys), "urey_bradleys")

    dihedrals = _array(d["dihedrals"], "dihedrals")
    for k, term in enumerate(dihedrals, 1):
        path = f"dihedrals[{k}]"
        term = _object(term, path)
        _keys(term, ("i", "j", "k", "l", "terms"), path)
        i = _integer(term["i"], f"{path}.i", minimum=1)
        j = _integer(term["j"], f"{path}.j", minimum=1)
        kk = _integer(term["k"], f"{path}.k", minimum=1)
        l = _integer(term["l"], f"{path}.l", minimum=1)
        _check_support((i, j, kk, l), n, path)
        terms = _array(term["terms"], f"{path}.terms")
        if not terms:
            _fail(f"{path}.terms", "must not be empty")
        for m, torsion in enumerate(terms, 1):
            tpath = f"{path}.terms[{m}]"
            torsion = _object(torsion, tpath)
            _keys(torsion, ("k", "n", "phase_rad"), tpath)
            _number(torsion["k"], f"{tpath}.k")
            _integer(torsion["n"], f"{tpath}.n", minimum=1)
            _number(torsion["phase_rad"], f"{tpath}.phase_rad")
    _check_duplicates(((t["i"], t["j"], t["k"], t["l"]) for t in dihedrals),
                      "dihedrals")

    impropers = _array(d["impropers"], "impropers")
    for k, term in enumerate(impropers, 1):
        path = f"impropers[{k}]"
        term = _object(term, path)
        _keys(term, ("i", "j", "k", "l", "k_phi", "phase_rad"), path)
        i = _integer(term["i"], f"{path}.i", minimum=1)
        j = _integer(term["j"], f"{path}.j", minimum=1)
        kk = _integer(term["k"], f"{path}.k", minimum=1)
        l = _integer(term["l"], f"{path}.l", minimum=1)
        _check_support((i, j, kk, l), n, path)
        _number(term["k_phi"], f"{path}.k_phi")
        _number(term["phase_rad"], f"{path}.phase_rad")
    _check_duplicates(((t["i"], t["j"], t["k"], t["l"]) for t in impropers),
                      "impropers")

    cmaps = _array(d["cmaps"], "cmaps")
    for k, term in enumerate(cmaps, 1):
        path = f"cmaps[{k}]"
        term = _object(term, path)
        _keys(term, ("i", "resolution", "grid"), path)
        idx = _array(term["i"], f"{path}.i")
        if len(idx) != 8:
            _fail(f"{path}.i", "needs eight atom indices")
        idx = [_integer(v, f"{path}.i[{m}]", minimum=1)
               for m, v in enumerate(idx, 1)]
        # A CMAP is two overlapping torsions: (a,b,c,d) and (b,c,d,e).  Real
        # CMAP arrays therefore repeat the three bridge atoms.  The schema
        # example uses eight distinct indices, so accept that form too; both
        # forms are bounds-checked, and a repeated-index array must have the
        # bridge pattern.
        for m, value in enumerate(idx, 1):
            _check_index(value, n, f"{path}.i[{m}]")
        if len(set(idx)) != 8 and not (
                idx[1] == idx[4] and idx[2] == idx[5] and idx[3] == idx[6]):
            _fail(f"{path}.i",
                  "must be eight distinct atoms or (a,b,c,d,b,c,d,e)")
        res = _integer(term["resolution"], f"{path}.resolution", minimum=2)
        grid = _array(term["grid"], f"{path}.grid")
        if len(grid) != res * res:
            _fail(f"{path}.grid",
                  f"expected {res * res} values, got {len(grid)}")
        for m, value in enumerate(grid, 1):
            _number(value, f"{path}.grid[{m}]")

    exclusions = _array(d["exclusions"], "exclusions")
    excl = set()
    for k, pair in enumerate(exclusions, 1):
        path = f"exclusions[{k}]"
        pair = _array(pair, path)
        if len(pair) != 2:
            _fail(path, "must be an (i,j) pair")
        i = _integer(pair[0], f"{path}[1]", minimum=1)
        j = _integer(pair[1], f"{path}[2]", minimum=1)
        key = _canonical_pair(i, j)
        _check_index(key[0], n, path)
        _check_index(key[1], n, path)
        if key in excl:
            _fail(path, f"duplicate exclusion {key}")
        excl.add(key)

    pair14 = _array(d["pair14"], "pair14")
    p14 = set()
    for k, term in enumerate(pair14, 1):
        path = f"pair14[{k}]"
        term = _object(term, path)
        _keys(term, ("i", "j", "coulomb_scale", "lj_scale"), path)
        i = _integer(term["i"], f"{path}.i", minimum=1)
        j = _integer(term["j"], f"{path}.j", minimum=1)
        key = _canonical_pair(i, j)
        _check_index(key[0], n, path)
        _check_index(key[1], n, path)
        if key in p14:
            _fail(path, f"duplicate 1-4 pair {key}")
        p14.add(key)
        cs = _number(term["coulomb_scale"], f"{path}.coulomb_scale",
                     nonnegative=True)
        ls = _number(term["lj_scale"], f"{path}.lj_scale", nonnegative=True)
        if not (0 <= cs <= 1 and 0 <= ls <= 1):
            _fail(path, "scales must lie in [0, 1]")
        if key in excl:
            _fail(path,
                  f"pair {key} is both an exclusion and a 1-4 interaction")

    virtual_sites = _array(d["virtual_sites"], "virtual_sites")
    sites = set()
    for k, term in enumerate(virtual_sites, 1):
        path = f"virtual_sites[{k}]"
        term = _object(term, path)
        _keys(term, ("kind", "site", "parents", "coefficients"), path)
        kind = _string(term["kind"], f"{path}.kind", nonempty=True)
        if kind not in ("linear", "planar", "average"):
            _fail(f"{path}.kind", "linear/planar/average required")
        site = _integer(term["site"], f"{path}.site", minimum=1)
        _check_index(site, n, path)
        if site in sites:
            _fail(f"{path}.site", "site declared twice")
        sites.add(site)
        parents = _array(term["parents"], f"{path}.parents")
        parents = [_integer(v, f"{path}.parents[{m}]", minimum=1)
                   for m, v in enumerate(parents, 1)]
        _check_support(parents, n, f"{path}.parents")
        coefs = _array(term["coefficients"], f"{path}.coefficients")
        if len(coefs) != len(parents):
            _fail(f"{path}.coefficients", "one coefficient per parent")
        coefs = [_number(v, f"{path}.coefficients[{m}]")
                 for m, v in enumerate(coefs, 1)]
        if abs(sum(coefs) - 1.0) > 1e-12:
            _fail(f"{path}.coefficients", "coefficients must sum to 1")
        if site in parents:
            _fail(f"{path}.parents", "site may not be its own parent")
        if atoms[site - 1]["mass"] != 0.0:
            _fail(f"atoms[{site}].mass", "a virtual site must be massless")
    for k, atom in enumerate(atoms, 1):
        if atom["mass"] == 0.0 and k not in sites:
            _fail(f"atoms[{k}].mass",
                  "massless atom is not a declared virtual site")

    constraints = _array(d["constraints"], "constraints")
    for k, term in enumerate(constraints, 1):
        path = f"constraints[{k}]"
        term = _object(term, path)
        _keys(term, ("i", "j", "distance_nm"), path)
        i = _integer(term["i"], f"{path}.i", minimum=1)
        j = _integer(term["j"], f"{path}.j", minimum=1)
        _check_support((i, j), n, path)
        _number(term["distance_nm"], f"{path}.distance_nm", positive=True)
    _check_duplicates((tuple(sorted((t["i"], t["j"]))) for t in constraints),
                      "constraints")

    diagnostics = _array(d["input_diagnostics"], "input_diagnostics")
    for k, term in enumerate(diagnostics, 1):
        path = f"input_diagnostics[{k}]"
        term = _object(term, path)
        _keys(term, DIAGNOSTIC_KEYS, path)
        _string(term["code"], f"{path}.code", nonempty=True)
        severity = _string(term["severity"], f"{path}.severity", nonempty=True)
        if severity not in ("error", "warning", "info"):
            _fail(f"{path}.severity", "must be error/warning/info")
        _string(term["message"], f"{path}.message")
        for name in ("atoms", "residues"):
            values = _array(term[name], f"{path}.{name}")
            for m, value in enumerate(values, 1):
                _integer(value, f"{path}.{name}[{m}]", minimum=1)
        for value in term["atoms"]:
            _check_index(value, n, f"{path}.atoms")

    net = _number(d["net_charge"], "net_charge")
    if abs(charge_sum - net) > 1e-6:
        _fail("net_charge",
              f"declared {net} but atom charges sum to {charge_sum}")

    # A final non-finite sweep catches anything the per-field checks missed
    # (for example a NaN hidden in a grid that was never indexed above).
    def _sweep(value, path):
        if isinstance(value, float):
            if not math.isfinite(value):
                _fail(path, "must be finite")
        elif isinstance(value, dict):
            for key, item in value.items():
                _sweep(item, f"{path}.{key}")
        elif isinstance(value, list):
            for m, item in enumerate(value, 1):
                _sweep(item, f"{path}[{m}]")

    _sweep(d, "<root>")
    return d
