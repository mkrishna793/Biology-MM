"""Independent pure-NumPy evaluator for the NoCutoff chemistry fixtures.

This is an oracle that does not call OpenMM at all: it reads the parameter
files with ParmEd, converts them itself, and evaluates the potential and its
gradient with numpy.  The OpenMM reference in ``make_reference.py`` is one
implementation of the same physics; this module is the second one, so a
mapping error in either shows up as a disagreement instead of cancelling out.

Covered terms: harmonic bonds, Urey-Bradley, harmonic angles, periodic proper
torsions (every term of every multi-term torsion), harmonic (CHARMM-style)
impropers, 12-6 Lennard-Jones with the raw prmtop type-pair A/B coefficients
(so off-diagonal / NBFIX-like pairs are exact, not Lorentz-Berthelot), the
AMBER 1-6-4 ``-C/r^4`` term, 1-4 pairs with their explicit scale factors, and
explicit 1-2/1-3 exclusions.  Electrostatics use the OpenMM prefactor in
kJ/mol nm e^2.

Not covered: CMAP torsions and virtual sites.  No NoCutoff fixture uses them;
:class:`UnsupportedTerm` is raised rather than silently dropping a term.

Every routine is vectorized over terms and, for the pair list, over pair
blocks, so the 2615-atom protein-ligand complex is 3.4M pairs of numpy work.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

KCAL_TO_KJ = 4.184
ANGSTROM_TO_NM = 0.1
# 1 kcal/mol/A^2 -> kJ/mol/nm^2
BOND_KCAL = KCAL_TO_KJ / ANGSTROM_TO_NM ** 2
# OpenMM's Coulomb prefactor, kJ/mol nm / e^2
COULOMB_KJ_NM = 138.9354576449822

COMPONENTS = (
    "bonds",
    "urey_bradley",
    "angles",
    "torsions",
    "impropers",
    "cmaps",
    "coulomb",
    "lj",
    "lj14",
    "lj146",
)


class UnsupportedTerm(Exception):
    """A term this oracle deliberately does not evaluate."""


@dataclass
class Parameters:
    n_atoms: int
    charges: np.ndarray
    type_index: np.ndarray
    lj_a: np.ndarray
    lj_b: np.ndarray
    lj_c: np.ndarray
    bonds: dict
    ureys: dict
    angles: dict
    torsions: dict
    impropers: dict
    exclusions: np.ndarray
    pair14: dict
    cmaps: list = field(default_factory=list)

    def n_terms(self) -> dict:
        return {
            "bonds": int(self.bonds["i"].size),
            "urey_bradley": int(self.ureys["i"].size),
            "angles": int(self.angles["i"].size),
            "torsion_terms": int(self.torsions["i"].size),
            "improper_terms": int(self.impropers["i"].size),
            "cmaps": len(self.cmaps),
            "exclusions": int(self.exclusions.shape[0]),
            "pair14": int(self.pair14["i"].size),
        }


def _empty_terms() -> dict:
    return {
        "i": np.zeros(0, dtype=np.int64),
        "j": np.zeros(0, dtype=np.int64),
        "k": np.zeros(0, dtype=np.int64),
        "l": np.zeros(0, dtype=np.int64),
        "a": np.zeros(0),
        "b": np.zeros(0),
        "c": np.zeros(0),
        "d": np.zeros(0),
    }


def _bond_terms(pairs) -> dict:
    out = _empty_terms()
    out["i"] = np.array([t[0] for t in pairs], dtype=np.int64)
    out["j"] = np.array([t[1] for t in pairs], dtype=np.int64)
    out["a"] = np.array([t[2] for t in pairs], dtype=float)
    out["b"] = np.array([t[3] for t in pairs], dtype=float)
    return out


# ---------------------------------------------------------------------------
# AMBER
# ---------------------------------------------------------------------------

def _amber_exclusions(parm_data) -> np.ndarray:
    """1-2/1-3 exclusion pairs from EXCLUDED_ATOMS_LIST, 1-based."""
    if "EXCLUDED_ATOMS_LIST" not in parm_data:
        return np.zeros((0, 2), dtype=np.int64)
    listed = parm_data["EXCLUDED_ATOMS_LIST"]
    counts = parm_data["NUMBER_EXCLUDED_ATOMS"]
    pairs = set()
    pos = 0
    for atom0, count in enumerate(counts):
        for j in listed[pos:pos + count]:
            pos += 1
            if j <= 0:
                continue
            i, jj = atom0 + 1, int(j)
            pairs.add((min(i, jj), max(i, jj)))
    return np.array(sorted(pairs), dtype=np.int64) if pairs else np.zeros((0, 2), dtype=np.int64)


def _dihedral_type_terms(dtype) -> list:
    if hasattr(dtype, "phi_k"):
        types = [dtype]
    else:
        types = list(dtype)
    return [(float(t.phi_k), int(t.per), float(t.phase)) for t in types]


def load_amber(parm_path) -> Parameters:
    import parmed

    struct = parmed.load_file(str(parm_path))
    parm_data = struct.parm_data
    n_atoms = len(struct.atoms)
    ntypes = int(struct.pointers["NTYPES"])
    nbi = np.asarray([int(x) for x in parm_data["NONBONDED_PARM_INDEX"]], dtype=np.int64)
    idx = (nbi.reshape(ntypes, ntypes) - 1).T  # [i][j] -> nbi[i*ntypes+j]
    valid = idx >= 0

    def matrix(key, factor):
        raw = np.asarray([float(x) for x in parm_data[key]], dtype=float) if key in parm_data else np.zeros(0)
        out = np.zeros((ntypes, ntypes))
        if raw.size:
            safe = np.where(valid, idx, 0)
            out[valid] = raw[safe[valid]] * factor
        return out

    lj_a = matrix("LENNARD_JONES_ACOEF", KCAL_TO_KJ * ANGSTROM_TO_NM ** 12)
    lj_b = matrix("LENNARD_JONES_BCOEF", KCAL_TO_KJ * ANGSTROM_TO_NM ** 6)
    lj_c = matrix("LENNARD_JONES_CCOEF", KCAL_TO_KJ * ANGSTROM_TO_NM ** 4)

    bonds = []
    for bond in struct.bonds:
        bonds.append((bond.atom1.idx, bond.atom2.idx,
                      float(bond.type.k) * BOND_KCAL, float(bond.type.req) * ANGSTROM_TO_NM))
    ureys = []
    for urey in struct.urey_bradleys:
        ureys.append((urey.atom1.idx, urey.atom2.idx,
                      float(urey.type.k) * BOND_KCAL, float(urey.type.req) * ANGSTROM_TO_NM))
    angles = []
    for angle in struct.angles:
        angles.append((angle.atom1.idx, angle.atom2.idx, angle.atom3.idx,
                       float(angle.type.k) * KCAL_TO_KJ, math.radians(float(angle.type.theteq))))

    torsions = []
    pair14 = {}
    for dihedral in struct.dihedrals:
        for k, n, phase in _dihedral_type_terms(dihedral.type):
            torsions.append((dihedral.atom1.idx, dihedral.atom2.idx,
                             dihedral.atom3.idx, dihedral.atom4.idx,
                             k * KCAL_TO_KJ, n, math.radians(phase)))
        if dihedral.improper or dihedral.ignore_end:
            continue
        i, j = dihedral.atom1.idx + 1, dihedral.atom4.idx + 1
        key = (min(i, j), max(i, j))
        scee = float(dihedral.type.scee)
        scnb = float(dihedral.type.scnb)
        scale = (0.0 if scee == 0 else 1.0 / scee, 0.0 if scnb == 0 else 1.0 / scnb)
        old = pair14.get(key)
        if old is not None and (abs(old[0] - scale[0]) > 1e-12 or abs(old[1] - scale[1]) > 1e-12):
            raise UnsupportedTerm(f"1-4 pair {key} has conflicting SCEE/SCNB scales")
        pair14[key] = scale

    impropers = []
    for improper in struct.impropers:
        impropers.append((improper.atom1.idx, improper.atom2.idx,
                          improper.atom3.idx, improper.atom4.idx,
                          float(improper.type.psi_k) * KCAL_TO_KJ,
                          math.radians(float(improper.type.psi_eq))))

    cmaps = list(struct.cmaps) if getattr(struct, "cmaps", None) else []
    exclusions = _amber_exclusions(parm_data)
    p14_keys = sorted(pair14)
    if p14_keys:
        excl = {(int(a), int(b)) for a, b in exclusions}
        for key in p14_keys:
            excl.discard(key)
        exclusions = np.array(sorted(excl), dtype=np.int64) if excl else np.zeros((0, 2), dtype=np.int64)

    type_index = np.array([int(atom.nb_idx) - 1 for atom in struct.atoms], dtype=np.int64)
    charges = np.array([float(atom.charge) for atom in struct.atoms])

    return Parameters(
        n_atoms=n_atoms,
        charges=charges,
        type_index=type_index,
        lj_a=lj_a,
        lj_b=lj_b,
        lj_c=lj_c,
        bonds=_bond_terms(bonds),
        ureys=_bond_terms(ureys),
        angles=_angle_terms(angles),
        torsions=_torsion_terms(torsions),
        impropers=_torsion_terms([(t[0], t[1], t[2], t[3], t[4], 1, t[5]) for t in impropers]),
        exclusions=exclusions,
        pair14=_pair14_terms(p14_keys, pair14, type_index, lj_a, lj_b),
        cmaps=cmaps,
    )


def _angle_terms(rows) -> dict:
    out = _empty_terms()
    out["i"] = np.array([r[0] for r in rows], dtype=np.int64)
    out["j"] = np.array([r[1] for r in rows], dtype=np.int64)
    out["k"] = np.array([r[2] for r in rows], dtype=np.int64)
    out["a"] = np.array([r[3] for r in rows], dtype=float)
    out["b"] = np.array([r[4] for r in rows], dtype=float)
    return out


def _torsion_terms(rows) -> dict:
    out = _empty_terms()
    out["i"] = np.array([r[0] for r in rows], dtype=np.int64)
    out["j"] = np.array([r[1] for r in rows], dtype=np.int64)
    out["k"] = np.array([r[2] for r in rows], dtype=np.int64)
    out["l"] = np.array([r[3] for r in rows], dtype=np.int64)
    out["a"] = np.array([r[4] for r in rows], dtype=float)   # k
    out["b"] = np.array([r[5] for r in rows], dtype=float)   # n
    out["c"] = np.array([r[6] for r in rows], dtype=float)   # phase
    return out


def _pair14_terms(keys, scales, type_index, lj_a, lj_b) -> dict:
    n = len(keys)
    out = {
        "i": np.array([k[0] for k in keys], dtype=np.int64),
        "j": np.array([k[1] for k in keys], dtype=np.int64),
        "coul_scale": np.array([scales[k][0] for k in keys], dtype=float),
        "a": np.zeros(n),
        "b": np.zeros(n),
    }
    for pos, key in enumerate(keys):
        ti = type_index[key[0] - 1]
        tj = type_index[key[1] - 1]
        out["a"][pos] = lj_a[ti, tj] * scales[key][1]
        out["b"][pos] = lj_b[ti, tj] * scales[key][1]
    return out


# ---------------------------------------------------------------------------
# CHARMM
# ---------------------------------------------------------------------------

def load_charmm(psf_path, coordinate_path, parameter_files) -> Parameters:
    import parmed
    from parmed.charmm import CharmmParameterSet

    psf = parmed.charmm.CharmmPsfFile(str(psf_path))
    params = CharmmParameterSet(*[str(path) for path in parameter_files])
    psf.load_parameters(params)

    n_atoms = len(psf.atoms)
    names = [atom.atom_type.name for atom in psf.atoms]
    order = sorted(set(names))
    label_index = {name: pos for pos, name in enumerate(order)}
    type_index = np.array([label_index[name] for name in names], dtype=np.int64)
    ntypes = len(order)
    by_name = {atom.atom_type.name: atom.atom_type for atom in psf.atoms}

    sigma = np.array([by_name[name].sigma * ANGSTROM_TO_NM for name in order])
    epsilon = np.array([-by_name[name].epsilon * KCAL_TO_KJ for name in order])
    sigma14 = np.array([by_name[name].sigma_14 * ANGSTROM_TO_NM for name in order])
    epsilon14 = np.array([-by_name[name].epsilon_14 * KCAL_TO_KJ for name in order])

    def lj_matrix(sig, eps):
        rmin = sig * 2.0 ** (1.0 / 6.0)  # Rmin from sigma
        eps_ij = np.sqrt(np.outer(eps, eps))
        rmin_ij = 0.5 * (np.outer(rmin, np.ones(ntypes)) + np.outer(np.ones(ntypes), rmin))
        a = eps_ij * rmin_ij ** 12
        b = 2.0 * eps_ij * rmin_ij ** 6
        return a, b

    lj_a, lj_b = lj_matrix(sigma, epsilon)
    for i, name_i in enumerate(order):
        nbfix = getattr(by_name[name_i], "nbfix", None) or {}
        for j, name_j in enumerate(order):
            entry = nbfix.get(name_j)
            if entry is None and i == j:
                entry = nbfix.get(name_i)
            if entry is None:
                continue
            rmin_nm = entry[0] * ANGSTROM_TO_NM
            eps_kj = entry[1] * KCAL_TO_KJ
            lj_a[i, j] = eps_kj * rmin_nm ** 12
            lj_b[i, j] = 2.0 * eps_kj * rmin_nm ** 6

    bonds = [(b.atom1.idx, b.atom2.idx, float(b.type.k) * BOND_KCAL,
              float(b.type.req) * ANGSTROM_TO_NM) for b in psf.bonds]
    ureys = [(u.atom1.idx, u.atom2.idx, float(u.type.k) * BOND_KCAL,
              float(u.type.req) * ANGSTROM_TO_NM) for u in psf.urey_bradleys]
    angles = [(a.atom1.idx, a.atom2.idx, a.atom3.idx,
               float(a.type.k) * KCAL_TO_KJ, math.radians(float(a.type.theteq)))
              for a in psf.angles]

    torsions = []
    for dihedral in psf.dihedrals:
        for k, n, phase in _dihedral_type_terms(dihedral.type):
            torsions.append((dihedral.atom1.idx, dihedral.atom2.idx,
                             dihedral.atom3.idx, dihedral.atom4.idx,
                             k * KCAL_TO_KJ, n, math.radians(phase)))
    impropers = [(d.atom1.idx, d.atom2.idx, d.atom3.idx, d.atom4.idx,
                  float(d.type.psi_k) * KCAL_TO_KJ, math.radians(float(d.type.psi_eq)))
                 for d in psf.impropers]

    excl = set()
    for bond in psf.bonds:
        i, j = bond.atom1.idx + 1, bond.atom2.idx + 1
        excl.add((min(i, j), max(i, j)))
    for angle in psf.angles:
        i, j = angle.atom1.idx + 1, angle.atom3.idx + 1
        excl.add((min(i, j), max(i, j)))
    pair14 = {}
    for dihedral in psf.dihedrals:
        i, j = dihedral.atom1.idx + 1, dihedral.atom4.idx + 1
        pair14[(min(i, j), max(i, j))] = float(params.default_scee)
    for key in pair14:
        excl.discard(key)
    p14_keys = sorted(pair14)

    out14 = {
        "i": np.array([k[0] for k in p14_keys], dtype=np.int64),
        "j": np.array([k[1] for k in p14_keys], dtype=np.int64),
        "coul_scale": np.array([1.0 / pair14[k] for k in p14_keys], dtype=float),
        "a": np.zeros(len(p14_keys)),
        "b": np.zeros(len(p14_keys)),
    }
    lj_a14, lj_b14 = lj_matrix(sigma14, epsilon14)
    for pos, key in enumerate(p14_keys):
        ti = type_index[key[0] - 1]
        tj = type_index[key[1] - 1]
        out14["a"][pos] = lj_a14[ti, tj]
        out14["b"][pos] = lj_b14[ti, tj]

    return Parameters(
        n_atoms=n_atoms,
        charges=np.array([float(atom.charge) for atom in psf.atoms]),
        type_index=type_index,
        lj_a=lj_a,
        lj_b=lj_b,
        lj_c=np.zeros((ntypes, ntypes)),
        bonds=_bond_terms(bonds),
        ureys=_bond_terms(ureys),
        angles=_angle_terms(angles),
        torsions=_torsion_terms(torsions),
        impropers=_torsion_terms([(r[0], r[1], r[2], r[3], r[4], 1, r[5]) for r in impropers]),
        exclusions=np.array(sorted(excl), dtype=np.int64) if excl else np.zeros((0, 2), dtype=np.int64),
        pair14=out14,
        cmaps=list(getattr(psf, "cmaps", []) or []),
    )


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

def _add(arr, idx, vec):
    for d in range(3):
        arr[:, d] += np.bincount(idx, weights=vec[:, d], minlength=arr.shape[0])


def _bond_energy(params, pos, terms, force):
    i, j = terms["i"], terms["j"]
    if i.size == 0:
        return 0.0
    delta = pos[j] - pos[i]
    r = np.sqrt(np.einsum("ij,ij->i", delta, delta))
    dr = r - terms["b"]
    energy = float(np.sum(terms["a"] * dr * dr))
    if force is not None:
        g = (2.0 * terms["a"] * dr / r)[:, None] * delta
        _add(force, i, g)
        _add(force, j, -g)
    return energy


def _angle_energy(params, pos, terms, force):
    i, j, k = terms["i"], terms["j"], terms["k"]
    if i.size == 0:
        return 0.0
    a = pos[i] - pos[j]
    b = pos[k] - pos[j]
    la = np.sqrt(np.einsum("ij,ij->i", a, a))
    lb = np.sqrt(np.einsum("ij,ij->i", b, b))
    cos = np.einsum("ij,ij->i", a, b) / (la * lb)
    cos = np.clip(cos, -1.0, 1.0)
    theta = np.arccos(cos)
    dth = theta - terms["b"]
    energy = float(np.sum(terms["a"] * dth * dth))
    if force is not None:
        sin = np.sqrt(np.maximum(1.0 - cos * cos, 1e-16))
        ded = 2.0 * terms["a"] * dth
        ga = -(ded / sin)[:, None] * (b / (la * lb)[:, None] - cos[:, None] * a / (la * la)[:, None])
        gb = -(ded / sin)[:, None] * (a / (la * lb)[:, None] - cos[:, None] * b / (lb * lb)[:, None])
        _add(force, i, ga)
        _add(force, k, gb)
        _add(force, j, -(ga + gb))
    return energy


def _dihedral_geometry(pos, i, j, k, l):
    """OpenMM's torsion convention (platforms/common/src/kernels/torsionForce.cc)."""
    v0 = pos[i] - pos[j]
    v1 = pos[k] - pos[j]
    v2 = pos[k] - pos[l]
    cp0 = np.cross(v0, v1)
    cp1 = np.cross(v1, v2)
    nn0 = np.einsum("ij,ij->i", cp0, cp0)
    nn1 = np.einsum("ij,ij->i", cp1, cp1)
    cos = np.einsum("ij,ij->i", cp0, cp1) / np.sqrt(nn0 * nn1)
    cos = np.clip(cos, -1.0, 1.0)
    sign = np.where(np.einsum("ij,ij->i", v0, cp1) < 0.0, -1.0, 1.0)
    angle = np.arccos(cos) * sign
    return v0, v1, v2, cp0, cp1, nn0, nn1, angle


def _torsion_forces(pos, force, d0, d1, d2, c0, c1, nn0, nn1, ded, i, j, k, l):
    w = np.sqrt(np.einsum("ij,ij->i", d1, d1))
    f0 = ((-ded * w / nn0)[:, None]) * c0
    f3 = ((ded * w / nn1)[:, None]) * c1
    f1 = np.einsum("ij,ij->i", d0, d1) / (w * w)
    f2 = np.einsum("ij,ij->i", d2, d1) / (w * w)
    s = f1[:, None] * f0 - f2[:, None] * f3
    _add(force, i, f0)
    _add(force, j, s - f0)
    _add(force, k, -s - f3)
    _add(force, l, f3)


def _torsion_energy(params, pos, terms, force):
    i, j, k, l = terms["i"], terms["j"], terms["k"], terms["l"]
    if i.size == 0:
        return 0.0
    d0, d1, d2, c0, c1, nn0, nn1, angle = _dihedral_geometry(pos, i, j, k, l)
    arg = terms["b"] * angle - terms["c"]
    energy = float(np.sum(terms["a"] * (1.0 + np.cos(arg))))
    if force is not None:
        ded = -terms["a"] * terms["b"] * np.sin(arg)
        _torsion_forces(pos, force, d0, d1, d2, c0, c1, nn0, nn1, ded, i, j, k, l)
    return energy


def _improper_energy(params, pos, terms, force):
    i = terms["i"]
    if i.size == 0:
        return 0.0
    d0, d1, d2, c0, c1, nn0, nn1, angle = _dihedral_geometry(
        pos, terms["i"], terms["j"], terms["k"], terms["l"])
    delta = np.abs(angle - terms["c"])
    two_pi = 2.0 * math.pi
    inner = delta <= (two_pi - delta)
    u = np.where(inner, delta, two_pi - delta)
    energy = float(np.sum(terms["a"] * u * u))
    if force is not None:
        sgn = np.sign(angle - terms["c"])
        ded = 2.0 * terms["a"] * u * np.where(inner, sgn, -sgn)
        _torsion_forces(pos, force, d0, d1, d2, c0, c1, nn0, nn1, ded,
                        terms["i"], terms["j"], terms["k"], terms["l"])
    return energy


def _pair_energy(params, pos, chunk=256):
    n = params.n_atoms
    q = params.charges
    ti = params.type_index
    excl = params.exclusions
    ex_key = (excl[:, 0] - 1) * n + (excl[:, 1] - 1) if excl.size else np.zeros(0, dtype=np.int64)
    p14 = params.pair14
    raw14 = (p14["i"] - 1) * n + (p14["j"] - 1) if p14["i"].size else np.zeros(0, dtype=np.int64)
    order = np.argsort(raw14)
    p14_sorted = raw14[order]
    scale14 = p14["coul_scale"][order]
    a14 = p14["a"][order]
    b14 = p14["b"][order]
    ex_sorted = np.sort(ex_key)

    lj_a, lj_b, lj_c = params.lj_a, params.lj_b, params.lj_c
    e_coul = e_lj = e_lj14 = e_c = 0.0
    f_coul = np.zeros_like(pos)
    f_lj = np.zeros_like(pos)
    f_lj14 = np.zeros_like(pos)
    f_c = np.zeros_like(pos)
    cols = np.arange(n)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        rows = np.arange(start, stop)
        mask = cols[None, :] > rows[:, None]
        ii = np.broadcast_to(rows[:, None], (stop - start, n))[mask]
        jj = np.broadcast_to(cols[None, :], (stop - start, n))[mask]
        if ii.size == 0:
            continue
        delta = pos[jj] - pos[ii]
        r2 = np.einsum("ij,ij->i", delta, delta)
        r2 = np.maximum(r2, 1e-8)
        r = np.sqrt(r2)
        key = ii * n + jj

        excluded = np.zeros(ii.size, dtype=bool)
        if ex_sorted.size:
            p = np.clip(np.searchsorted(ex_sorted, key), 0, ex_sorted.size - 1)
            excluded = ex_sorted[p] == key
        fourteen = np.zeros(ii.size, dtype=bool)
        if p14_sorted.size:
            p = np.clip(np.searchsorted(p14_sorted, key), 0, p14_sorted.size - 1)
            fourteen = p14_sorted[p] == key
        normal = ~excluded & ~fourteen

        a = np.where(normal, lj_a[ti[ii], ti[jj]], 0.0)
        b = np.where(normal, lj_b[ti[ii], ti[jj]], 0.0)
        c = np.where(normal, lj_c[ti[ii], ti[jj]], 0.0)
        qq = np.where(normal, q[ii] * q[jj], 0.0)
        if fourteen.any():
            p = np.clip(np.searchsorted(p14_sorted, key[fourteen]), 0, max(p14_sorted.size - 1, 0))
            a[fourteen] = a14[p]
            b[fourteen] = b14[p]
            qq[fourteen] = q[ii[fourteen]] * q[jj[fourteen]] * scale14[p]

        inv2 = 1.0 / r2
        inv6 = inv2 * inv2 * inv2
        inv12 = inv6 * inv6
        normal_lj = float(np.sum(a[normal] * inv12[normal] - b[normal] * inv6[normal]))
        e_lj += normal_lj
        e_lj14 += float(np.sum(a[fourteen] * inv12[fourteen] - b[fourteen] * inv6[fourteen]))
        e_c -= float(np.sum(c * inv2 * inv2))
        e_coul += float(np.sum(COULOMB_KJ_NM * qq / r))

        flj = ((-12.0 * a * inv12 + 6.0 * b * inv6) / r2)[:, None] * delta
        fc = (COULOMB_KJ_NM * qq * inv2 / r)[:, None] * delta
        fcc = (4.0 * c * inv6)[:, None] * delta
        _add(f_lj, ii[normal], flj[normal])
        _add(f_lj, jj[normal], -flj[normal])
        _add(f_lj14, ii[fourteen], flj[fourteen])
        _add(f_lj14, jj[fourteen], -flj[fourteen])
        _add(f_coul, ii, -fc)
        _add(f_coul, jj, fc)
        _add(f_c, ii, fcc)
        _add(f_c, jj, -fcc)
    return (e_coul, e_lj, e_lj14, e_c), (f_coul, f_lj, f_lj14, f_c)


def lb_matrices(params: Parameters):
    """The Lorentz-Berthelot 12-6 matrices implied by the per-atom parameters.

    ``mdparams`` schema 1.0.0 stores one sigma/epsilon per atom, so a consumer
    necessarily combines them with Lorentz-Berthelot.  Reconstructing those
    matrices here is what lets the audit measure the cost of any pair-specific
    (NBFIX) term the normalized record cannot express.
    """
    ntypes = params.lj_a.shape[0]
    a_ii = np.diagonal(params.lj_a).copy()
    b_ii = np.diagonal(params.lj_b).copy()
    rmin = np.zeros(ntypes)
    epsilon = np.zeros(ntypes)
    ok = (a_ii > 0.0) & (b_ii > 0.0)
    rmin[ok] = (2.0 * a_ii[ok] / b_ii[ok]) ** (1.0 / 6.0)
    epsilon[ok] = b_ii[ok] * b_ii[ok] / (4.0 * a_ii[ok])
    combo = 0.5 * (rmin[:, None] + rmin[None, :])
    mixed = np.sqrt(np.outer(epsilon, epsilon))
    return mixed * combo ** 12, 2.0 * mixed * combo ** 6


def matrix_deviation(params: Parameters):
    """Max relative A/B deviation of the parameter matrices from LB (used pairs)."""
    lb_a, lb_b = lb_matrices(params)
    used = np.unique(params.type_index)
    ti = used[:, None]
    tj = used[None, :]
    a = params.lj_a[ti, tj]
    b = params.lj_b[ti, tj]
    la = lb_a[ti, tj]
    lb = lb_b[ti, tj]
    scale = np.maximum(np.maximum(np.abs(la), np.abs(lb)), 1e-30)
    rel = np.maximum(np.abs(a - la), np.abs(b - lb)) / scale
    rel = np.where((la == 0.0) & (lb == 0.0), 0.0, rel)
    worst = float(rel.max()) if rel.size else 0.0
    over = rel > 1e-8
    pairs = int(np.count_nonzero(np.triu(over, 0)))
    return {"max_relative_deviation": worst, "pairs_over_1e-8": pairs,
            "used_types": int(used.size), "used_pairs": int(a.size)}


def evaluate(params: Parameters, positions, want_forces: bool = True):
    """Return ({component: energy}, {component: (N,3) forces})."""
    if params.cmaps:
        raise UnsupportedTerm("CMAP torsions are not implemented in exact_reference")
    pos = np.asarray(positions, dtype=float)
    if pos.shape != (params.n_atoms, 3):
        raise ValueError("positions shape %s does not match %d atoms" % (pos.shape, params.n_atoms))
    energies = {}
    forces = {}
    for name, terms, fn in (
        ("bonds", params.bonds, _bond_energy),
        ("urey_bradley", params.ureys, _bond_energy),
        ("angles", params.angles, _angle_energy),
        ("torsions", params.torsions, _torsion_energy),
        ("impropers", params.impropers, _improper_energy),
    ):
        buf = np.zeros_like(pos) if want_forces else None
        energies[name] = fn(params, pos, terms, buf)
        if want_forces:
            forces[name] = buf
    pe, pf = _pair_energy(params, pos)
    energies["coulomb"] = pe[0]
    energies["lj"] = pe[1]
    energies["lj14"] = pe[2]
    energies["lj146"] = pe[3]
    energies["lj_all"] = pe[1] + pe[2]
    energies["cmaps"] = 0.0
    if want_forces:
        forces["coulomb"] = pf[0]
        forces["lj"] = pf[1]
        forces["lj14"] = pf[2]
        forces["lj146"] = pf[3]
        forces["cmaps"] = np.zeros_like(pos)
    return energies, forces


def total(energies: dict) -> float:
    return float(sum(energies[name] for name in COMPONENTS))
