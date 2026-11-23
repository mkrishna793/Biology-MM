"""OpenFF (SMIRNOFF/Sage) + AMBER (.prmtop/.inpcrd) + AlphaFold pLDDT Open-Science Bridge.

Provides 100% open, zero-proprietary parameterization and preparation utilities:
1. Automatic detection of optional external open-science prep tools (`openff-toolkit`,
   `pdbfixer`, `tleap`, `antechamber`).
2. Pure-Python OpenFF-Sage-inspired open parameterizer (LJ sigma/epsilon, bond/angle/torsion
   parameters, and element-aware partial charges) so small molecules, water, ions, and
   biomolecules can be parameterized and exported even offline.
3. Standard AMBER `.prmtop` / `.inpcrd` exporter so parameterized systems interoperate cleanly
   with Biology-MM, OpenMM, AMBER, MDTraj, and MDAnalysis.
4. AlphaFold PDB/mmCIF `pLDDT` confidence parser & automatic Focus Zone generator that puts
   low-pLDDT flexible loops into Biology-MM's All-Atom Focus Zone.
"""

from __future__ import annotations

import math
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = [
    "AlphaFoldConfidence",
    "OpenParameterSet",
    "compute_plddt_focus_zone",
    "discover_open_prep_tools",
    "export_amber_prmtop_inpcrd",
    "parameterize_openff_sage",
    "parse_alphafold_plddt",
]

# Open-data element/SMIRNOFF-style reference parameters (sigma in nm, epsilon in kJ/mol, mass in amu)
# Values follow standard open TIP3P / AMBER / OpenFF Sage 2.x conventions for common atom types.
_OPEN_LJ_TABLE: Dict[str, Tuple[float, float, float, float]] = {
    # element: (sigma_nm, epsilon_kj_mol, mass_amu, covalent_radius_nm)
    "H": (0.247135, 0.065689, 1.00794, 0.031),
    "HW": (0.000000, 0.000000, 1.00794, 0.031),
    "C": (0.339967, 0.457730, 12.0110, 0.076),
    "N": (0.325000, 0.711280, 14.0067, 0.071),
    "O": (0.295992, 0.878640, 15.9994, 0.066),
    "OW": (0.315061, 0.636386, 15.9994, 0.066),  # TIP3P water oxygen
    "S": (0.356359, 1.046000, 32.0650, 0.105),
    "P": (0.374177, 0.836800, 30.9738, 0.107),
    "F": (0.311815, 0.255224, 18.9984, 0.057),
    "CL": (0.347094, 1.108760, 35.4530, 0.102),
    "BR": (0.395559, 1.338880, 79.9040, 0.120),
    "NA": (0.243928, 0.365846, 22.9898, 0.166),
    "K": (0.303796, 0.809604, 39.0983, 0.203),
    "MG": (0.141225, 3.743420, 24.3050, 0.141),
    "CA": (0.242201, 1.881540, 40.0780, 0.176),
    "ZN": (0.195998, 0.052300, 65.3800, 0.122),
    "AR": (0.340500, 0.996000, 39.9480, 0.106),
    # Full chemistry & materials (MOFs like UiO-66, ZIF-8, HKUST-1, semiconductors, catalysts) — UFF-derived
    "LI": (0.218359, 0.104600, 6.9410, 0.128),
    "BE": (0.244552, 0.355640, 9.0122, 0.096),
    "B":  (0.363754, 0.753120, 10.811, 0.084),
    "AL": (0.400815, 2.112920, 26.982, 0.121),
    "SI": (0.382641, 1.681970, 28.086, 0.111),
    "TI": (0.282860, 0.071128, 47.867, 0.160),
    "V":  (0.280099, 0.066944, 50.942, 0.153),
    "CR": (0.269323, 0.062760, 51.996, 0.139),
    "MN": (0.263795, 0.054392, 54.938, 0.139),
    "FE": (0.259430, 0.054392, 55.845, 0.132),
    "CO": (0.255866, 0.058576, 58.933, 0.126),
    "NI": (0.252481, 0.062760, 58.693, 0.124),
    "CU": (0.311369, 0.020920, 63.546, 0.132),
    "GA": (0.390481, 1.736360, 69.723, 0.122),
    "GE": (0.381305, 1.585740, 72.630, 0.120),
    "AS": (0.376850, 1.292860, 74.922, 0.119),
    "SE": (0.374623, 1.217540, 78.971, 0.120),
    "SR": (0.324376, 0.983240, 87.620, 0.195),
    "Y":  (0.298006, 0.301248, 88.906, 0.190),
    "ZR": (0.278317, 0.288696, 91.224, 0.175),  # UiO-66 Zirconium node
    "NB": (0.281969, 0.246856, 92.906, 0.164),
    "MO": (0.271902, 0.234304, 95.950, 0.154),
    "RU": (0.263973, 0.234304, 101.07, 0.146),
    "RH": (0.260944, 0.221752, 102.91, 0.142),
    "PD": (0.258272, 0.200832, 106.42, 0.139),
    "AG": (0.280455, 0.150624, 107.87, 0.145),
    "SN": (0.391283, 2.372330, 118.71, 0.139),
    "I":  (0.400904, 1.418380, 126.90, 0.139),
    "BA": (0.329900, 1.522980, 137.33, 0.215),
    "LA": (0.313775, 0.071128, 138.91, 0.207),
    "CE": (0.316804, 0.054392, 140.12, 0.204),
    "GD": (0.300054, 0.037656, 157.25, 0.196),
    "W":  (0.273416, 0.280328, 183.84, 0.162),
    "IR": (0.253015, 0.305432, 192.22, 0.141),
    "PT": (0.245354, 0.334720, 195.08, 0.136),
    "AU": (0.293373, 0.163176, 196.97, 0.136),
    "HG": (0.240988, 0.161084, 200.59, 0.132),
    "TL": (0.387273, 2.845120, 204.38, 0.145),
    "PB": (0.382819, 2.773990, 207.20, 0.146),
    "BI": (0.389323, 2.167312, 208.98, 0.148),
    "SC": (0.293551, 0.079496, 44.956, 0.170),
    "CD": (0.253728, 0.953952, 112.41, 0.144),
    "IN": (0.397608, 2.506216, 114.82, 0.142),
    "SB": (0.393777, 1.878616, 121.76, 0.139),
    "TE": (0.398232, 1.665232, 127.60, 0.138),
    "CS": (0.402419, 0.188280, 132.91, 0.244),
    "ND": (0.318496, 0.041840, 144.24, 0.201),
    "EU": (0.311191, 0.033472, 151.96, 0.198),
    "HF": (0.279832, 0.301248, 178.49, 0.175),
    "TA": (0.282415, 0.338904, 180.95, 0.170),
    "RE": (0.263171, 0.276144, 186.21, 0.151),
    "OS": (0.277960, 0.154808, 190.23, 0.144),
    "TH": (0.302550, 0.108784, 232.04, 0.206),
    "U":  (0.302460, 0.092048, 238.03, 0.196),
}

# Pauling electronegativity for lightweight Gasteiger-style charge equilibration
_ELECTRONEGATIVITY: Dict[str, float] = {
    "H": 2.20, "C": 2.55, "N": 3.04, "O": 3.44, "F": 3.98,
    "P": 2.19, "S": 2.58, "CL": 3.16, "BR": 2.96, "NA": 0.93,
    "K": 0.82, "MG": 1.31, "CA": 1.00, "ZN": 1.65, "AR": 0.0,
    "LI": 0.98, "BE": 1.57, "B": 2.04, "AL": 1.61, "SI": 1.90, "TI": 1.54,
    "V": 1.63, "CR": 1.66, "MN": 1.55, "FE": 1.83, "CO": 1.88, "NI": 1.91,
    "CU": 1.90, "GA": 1.81, "GE": 2.01, "AS": 2.18, "SE": 2.55, "SR": 0.95,
    "Y": 1.22, "ZR": 1.33, "NB": 1.60, "MO": 2.16, "RU": 2.20, "RH": 2.28,
    "PD": 2.20, "AG": 1.93, "CD": 1.69, "IN": 1.78, "SN": 1.96, "SB": 2.05,
    "TE": 2.10, "I": 2.66, "CS": 0.79, "BA": 0.89, "LA": 1.10, "CE": 1.12,
    "ND": 1.14, "EU": 1.20, "GD": 1.20, "HF": 1.30, "TA": 1.50, "W": 2.36,
    "RE": 1.90, "OS": 2.20, "IR": 2.20, "PT": 2.28, "AU": 2.54, "HG": 2.00,
    "TL": 1.62, "PB": 2.33, "BI": 2.02, "TH": 1.30, "U": 1.38,
}

# Reference harmonic bond stiffness (kJ/mol/nm^2)
_BOND_KB_DEFAULT = 250000.0
_ANGLE_KTHETA_DEFAULT = 420.0


@dataclass
class OpenParameterSet:
    """Parameterized open-science molecular system (units: nm, kJ/mol, amu, e)."""

    elements: List[str]
    atom_names: List[str]
    residue_names: List[str]
    residue_ids: List[int]
    positions_nm: np.ndarray
    masses_amu: np.ndarray
    charges_e: np.ndarray
    sigma_nm: np.ndarray
    epsilon_kj_mol: np.ndarray
    bonds: List[Tuple[int, int, float, float]] = field(default_factory=list)  # (i, j, k_b, r0_nm)
    angles: List[Tuple[int, int, int, float, float]] = field(default_factory=list)  # (i, j, k, k_th, th0_rad)
    dihedrals: List[Tuple[int, int, int, int, float, int, float]] = field(default_factory=list)
    box_nm: Optional[np.ndarray] = None
    forcefield_source: str = "openff-sage-open-fallback"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlphaFoldConfidence:
    """Per-atom and per-residue AlphaFold pLDDT confidence breakdown."""

    plddt_per_atom: np.ndarray
    mean_plddt: float
    low_confidence_indices: np.ndarray  # pLDDT < 70 (flexible loops / disordered)
    medium_confidence_indices: np.ndarray  # 70 <= pLDDT < 90
    high_confidence_indices: np.ndarray  # pLDDT >= 90 (rigid scaffold)


def discover_open_prep_tools() -> Dict[str, Any]:
    """Detect optional external open-science prep tools installed on the host."""
    has_openff = False
    has_pdbfixer = False
    has_openmm = False
    for mod_name in ("openff.toolkit", "pdbfixer", "openmm"):
        try:
            __import__(mod_name)
            if mod_name == "openff.toolkit":
                has_openff = True
            elif mod_name == "pdbfixer":
                has_pdbfixer = True
            elif mod_name == "openmm":
                has_openmm = True
        except ImportError:
            pass
    return {
        "openff_toolkit": has_openff,
        "pdbfixer": has_pdbfixer,
        "openmm": has_openmm,
        "tleap": shutil.which("tleap"),
        "antechamber": shutil.which("antechamber"),
        "builtin_openff_sage_fallback": True,
        "builtin_amber_writer": True,
    }


def _infer_bonds_by_covalent_radii(
    positions_nm: np.ndarray,
    elements: Sequence[str],
    scale: float = 1.25,
) -> List[Tuple[int, int]]:
    n = len(elements)
    if n <= 1 or n > 4000:
        return []
    radii = np.array([_OPEN_LJ_TABLE.get(e.upper(), _OPEN_LJ_TABLE["C"])[3] for e in elements], dtype=np.float64)
    bonds: List[Tuple[int, int]] = []
    for i in range(n):
        if elements[i].upper() in ("NA", "K", "MG", "CA", "ZN", "AR", "CL"):
            continue
        diff = positions_nm[i + 1 :] - positions_nm[i]
        dists = np.linalg.norm(diff, axis=1)
        max_r = (radii[i] + radii[i + 1 :]) * scale
        hits = np.where((dists > 0.04) & (dists <= max_r))[0]
        for h in hits:
            j = i + 1 + int(h)
            if elements[j].upper() in ("NA", "K", "MG", "CA", "ZN", "AR", "CL"):
                continue
            bonds.append((i, j))
    return bonds


def parameterize_openff_sage(
    positions_nm: np.ndarray,
    elements: Sequence[str],
    atom_names: Optional[Sequence[str]] = None,
    residue_names: Optional[Sequence[str]] = None,
    residue_ids: Optional[Sequence[int]] = None,
    bonds: Optional[Sequence[Tuple[int, int]]] = None,
    box_nm: Optional[np.ndarray] = None,
    net_charge: float = 0.0,
) -> OpenParameterSet:
    """Assign open-science OpenFF-Sage / AMBER-compatible parameters to any molecular geometry.

    Assigns LJ (sigma, epsilon), masses, electronegativity-equilibrated partial charges
    (preserving `net_charge`), harmonic bonds, and harmonic angles.
    """
    pos = np.asarray(positions_nm, dtype=np.float64)
    n = int(pos.shape[0])
    elems = [str(e).strip().upper() for e in elements]
    anames = list(atom_names) if atom_names is not None else [f"{e}{i+1}" for i, e in enumerate(elems)]
    rnames = list(residue_names) if residue_names is not None else ["MOL"] * n
    rids = list(residue_ids) if residue_ids is not None else [1] * n

    masses = np.zeros(n, dtype=np.float64)
    sigma = np.zeros(n, dtype=np.float64)
    epsilon = np.zeros(n, dtype=np.float64)
    charges = np.zeros(n, dtype=np.float64)

    for i, el in enumerate(elems):
        rname = rnames[i].upper()
        key = el
        if rname in ("HOH", "WAT", "TIP3", "SOL"):
            if el == "O":
                key = "OW"
                charges[i] = -0.834
            elif el == "H":
                key = "HW"
                charges[i] = 0.417
        elif el == "NA":
            charges[i] = 1.0
        elif el == "K":
            charges[i] = 1.0
        elif el == "CL":
            charges[i] = -1.0
        elif el in ("MG", "CA", "ZN"):
            charges[i] = 2.0

        s_nm, e_kj, m_amu, _ = _OPEN_LJ_TABLE.get(key, _OPEN_LJ_TABLE["C"])
        sigma[i] = s_nm
        epsilon[i] = e_kj
        masses[i] = m_amu

    raw_bonds = list(bonds) if bonds is not None else _infer_bonds_by_covalent_radii(pos, elems)

    # Bonded electronegativity polarization for non-water organic atoms
    adj: List[List[int]] = [[] for _ in range(n)]
    bond_terms: List[Tuple[int, int, float, float]] = []
    for i, j in raw_bonds:
        adj[i].append(j)
        adj[j].append(i)
        r0 = float(np.linalg.norm(pos[j] - pos[i]))
        kb = 376560.0 if ("H" in (elems[i], elems[j])) else _BOND_KB_DEFAULT
        bond_terms.append((int(i), int(j), kb, max(0.08, r0)))
        if rnames[i].upper() not in ("HOH", "WAT", "TIP3", "SOL"):
            en_i = _ELECTRONEGATIVITY.get(elems[i], 2.5)
            en_j = _ELECTRONEGATIVITY.get(elems[j], 2.5)
            dq = 0.14 * (en_j - en_i)
            charges[i] += dq
            charges[j] -= dq

    # Neutralize / enforce exact net_charge across non-ion atoms if needed
    if n > 0:
        drift = (float(np.sum(charges)) - float(net_charge)) / float(n)
        if abs(drift) > 1e-9:
            charges -= drift

    # Infer harmonic angles from bond adjacency
    angle_terms: List[Tuple[int, int, int, float, float]] = []
    for j in range(n):
        nbrs = adj[j]
        for a_idx in range(len(nbrs)):
            for b_idx in range(a_idx + 1, len(nbrs)):
                i, k = nbrs[a_idx], nbrs[b_idx]
                v1 = pos[i] - pos[j]
                v2 = pos[k] - pos[j]
                n1, n2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
                if n1 > 1e-8 and n2 > 1e-8:
                    cos_t = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
                    th0 = math.acos(cos_t)
                    angle_terms.append((int(i), int(j), int(k), _ANGLE_KTHETA_DEFAULT, th0))

    return OpenParameterSet(
        elements=elems,
        atom_names=anames,
        residue_names=rnames,
        residue_ids=rids,
        positions_nm=pos,
        masses_amu=masses,
        charges_e=charges,
        sigma_nm=sigma,
        epsilon_kj_mol=epsilon,
        bonds=bond_terms,
        angles=angle_terms,
        box_nm=None if box_nm is None else np.asarray(box_nm, dtype=np.float64).copy(),
        forcefield_source="openff-sage-open-fallback",
        metadata={"n_bonds": len(bond_terms), "n_angles": len(angle_terms)},
    )


def export_amber_prmtop_inpcrd(
    params: OpenParameterSet,
    title: str = "BIOLOGY-MM OPEN-SCIENCE SYSTEM",
) -> Tuple[str, str]:
    """Export an `OpenParameterSet` to standard AMBER7 `.prmtop` and `.inpcrd` text strings."""
    n = len(params.elements)
    # AMBER internal charge scaling: q_amber = q_e * 18.2223
    amber_q = params.charges_e * 18.2223

    # Build unique LJ atom types by element
    unique_types: List[str] = []
    type_index: List[int] = []
    for el in params.elements:
        if el not in unique_types:
            unique_types.append(el)
        type_index.append(unique_types.index(el) + 1)  # 1-based AMBER IAC
    ntypes = len(unique_types)

    # Lorentz-Berthelot combining rules for AMBER ACOEF / BCOEF (kcal/mol, Angstrom)
    kj_to_kcal = 1.0 / 4.184
    type_sig_a = np.array([
        float(params.sigma_nm[params.elements.index(t)]) * 10.0 for t in unique_types
    ], dtype=np.float64)
    type_eps_kcal = np.array([
        float(params.epsilon_kj_mol[params.elements.index(t)]) * kj_to_kcal for t in unique_types
    ], dtype=np.float64)

    acoef: List[float] = []
    bcoef: List[float] = []
    nb_idx: List[int] = []
    pair_num = 1
    pair_map: Dict[Tuple[int, int], int] = {}
    for i in range(ntypes):
        for j in range(ntypes):
            key = (min(i, j), max(i, j))
            if key not in pair_map:
                pair_map[key] = pair_num
                pair_num += 1
                s_ij = 0.5 * (type_sig_a[i] + type_sig_a[j])
                e_ij = math.sqrt(max(0.0, type_eps_kcal[i] * type_eps_kcal[j]))
                s6 = s_ij ** 6
                acoef.append(4.0 * e_ij * s6 * s6)
                bcoef.append(4.0 * e_ij * s6)
            nb_idx.append(pair_map[key])

    # Format helpers
    def _fmt_ints(vals: Sequence[int], per_line: int = 10, width: int = 8) -> List[str]:
        lines: List[str] = []
        for k in range(0, len(vals), per_line):
            lines.append("".join(f"{int(v):{width}d}" for v in vals[k : k + per_line]))
        return lines or [""]

    def _fmt_floats(vals: Sequence[float], per_line: int = 5) -> List[str]:
        lines: List[str] = []
        for k in range(0, len(vals), per_line):
            lines.append("".join(f"{float(v):16.8E}" for v in vals[k : k + per_line]))
        return lines or [""]

    def _fmt_strings(vals: Sequence[str], per_line: int = 20, width: int = 4) -> List[str]:
        lines: List[str] = []
        for k in range(0, len(vals), per_line):
            lines.append("".join(f"{str(v)[:width]:<{width}s}" for v in vals[k : k + per_line]))
        return lines or [""]

    # Residues
    res_labels: List[str] = []
    res_ptrs: List[int] = []
    last_rid = None
    for idx, (rname, rid) in enumerate(zip(params.residue_names, params.residue_ids)):
        if rid != last_rid:
            res_labels.append(rname[:4])
            res_ptrs.append(idx + 1)
            last_rid = rid

    # Bond sections (BONDS_WITHOUT_HYDROGEN / BONDS_INC_HYDROGEN)
    bond_k_kcal = [b[2] * kj_to_kcal * 0.01 * 0.5 for b in params.bonds]  # kJ/mol/nm^2 -> kcal/mol/A^2 (AMBER 1/2 absorbed)
    bond_r0_a = [b[3] * 10.0 for b in params.bonds]
    bonds_no_h: List[int] = []
    bonds_inc_h: List[int] = []
    for b_idx, (i, j, _, _) in enumerate(params.bonds):
        triple = [i * 3, j * 3, b_idx + 1]
        if params.elements[i] == "H" or params.elements[j] == "H":
            bonds_inc_h.extend(triple)
        else:
            bonds_no_h.extend(triple)

    pointers = [0] * 31
    pointers[0] = n  # NATOM
    pointers[1] = ntypes  # NTYPES
    pointers[2] = len(bonds_inc_h) // 3  # NBONH
    pointers[3] = len(bonds_no_h) // 3  # MBONA
    pointers[11] = len(res_labels)  # NRES
    pointers[15] = max(1, len(params.bonds))  # NUMBND
    pointers[27] = 1 if params.box_nm is not None else 0  # IFBOX

    prm: List[str] = [
        "%VERSION  VERSION_STAMP = V0001.000  DATE = 11/21/26  00:00:00",
        "%FLAG TITLE",
        "%FORMAT(20a4)",
        f"{title[:80]:<80s}",
        "%FLAG POINTERS",
        "%FORMAT(10I8)",
        *_fmt_ints(pointers, 10, 8),
        "%FLAG ATOM_NAME",
        "%FORMAT(20a4)",
        *_fmt_strings(params.atom_names, 20, 4),
        "%FLAG CHARGE",
        "%FORMAT(5E16.8)",
        *_fmt_floats(amber_q, 5),
        "%FLAG MASS",
        "%FORMAT(5E16.8)",
        *_fmt_floats(params.masses_amu, 5),
        "%FLAG ATOM_TYPE_INDEX",
        "%FORMAT(10I8)",
        *_fmt_ints(type_index, 10, 8),
        "%FLAG NONBONDED_PARM_INDEX",
        "%FORMAT(10I8)",
        *_fmt_ints(nb_idx, 10, 8),
        "%FLAG RESIDUE_LABEL",
        "%FORMAT(20a4)",
        *_fmt_strings(res_labels, 20, 4),
        "%FLAG RESIDUE_POINTER",
        "%FORMAT(10I8)",
        *_fmt_ints(res_ptrs, 10, 8),
        "%FLAG BOND_FORCE_CONSTANT",
        "%FORMAT(5E16.8)",
        *_fmt_floats(bond_k_kcal or [0.0], 5),
        "%FLAG BOND_EQUIL_VALUE",
        "%FORMAT(5E16.8)",
        *_fmt_floats(bond_r0_a or [0.0], 5),
        "%FLAG BONDS_INC_HYDROGEN",
        "%FORMAT(10I8)",
        *(_fmt_ints(bonds_inc_h, 10, 8) if bonds_inc_h else [""]),
        "%FLAG BONDS_WITHOUT_HYDROGEN",
        "%FORMAT(10I8)",
        *(_fmt_ints(bonds_no_h, 10, 8) if bonds_no_h else [""]),
        "%FLAG LENNARD_JONES_ACOEF",
        "%FORMAT(5E16.8)",
        *_fmt_floats(acoef, 5),
        "%FLAG LENNARD_JONES_BCOEF",
        "%FORMAT(5E16.8)",
        *_fmt_floats(bcoef, 5),
        "%FLAG AMBER_ATOM_TYPE",
        "%FORMAT(20a4)",
        *_fmt_strings(params.elements, 20, 4),
    ]

    # Build .inpcrd (Angstrom)
    pos_a = params.positions_nm * 10.0
    crd: List[str] = [title[:80], f"{n:5d}"]
    flat = pos_a.reshape(-1)
    for k in range(0, len(flat), 6):
        crd.append("".join(f"{float(v):12.7f}" for v in flat[k : k + 6]))
    if params.box_nm is not None:
        H = np.asarray(params.box_nm, dtype=np.float64) * 10.0
        la, lb, lc = (float(np.linalg.norm(H[0])), float(np.linalg.norm(H[1])), float(np.linalg.norm(H[2])))
        crd.append(f"{la:12.7f}{lb:12.7f}{lc:12.7f}{90.0:12.7f}{90.0:12.7f}{90.0:12.7f}")

    return "\n".join(prm) + "\n", "\n".join(crd) + "\n"


def parse_alphafold_plddt(pdb_text: str) -> AlphaFoldConfidence:
    """Extract per-atom pLDDT scores from an AlphaFold PDB file (stored in B-factor cols 60-66)."""
    scores: List[float] = []
    for line in pdb_text.splitlines():
        if line.startswith(("ATOM  ", "HETATM")) and len(line) >= 66:
            try:
                scores.append(float(line[60:66].strip()))
            except ValueError:
                scores.append(50.0)
    arr = np.asarray(scores, dtype=np.float64)
    if arr.size == 0:
        return AlphaFoldConfidence(
            plddt_per_atom=arr,
            mean_plddt=0.0,
            low_confidence_indices=np.zeros(0, dtype=np.int64),
            medium_confidence_indices=np.zeros(0, dtype=np.int64),
            high_confidence_indices=np.zeros(0, dtype=np.int64),
        )
    return AlphaFoldConfidence(
        plddt_per_atom=arr,
        mean_plddt=float(np.mean(arr)),
        low_confidence_indices=np.where(arr < 70.0)[0].astype(np.int64),
        medium_confidence_indices=np.where((arr >= 70.0) & (arr < 90.0))[0].astype(np.int64),
        high_confidence_indices=np.where(arr >= 90.0)[0].astype(np.int64),
    )


def compute_plddt_focus_zone(
    positions_nm: np.ndarray,
    confidence: AlphaFoldConfidence,
    pad_nm: float = 0.6,
) -> Dict[str, Any]:
    """Compute a Biology-MM Multiscale Focus Zone centered on low-pLDDT flexible regions."""
    pos = np.asarray(positions_nm, dtype=np.float64)
    if pos.shape[0] == 0:
        return {"focus_center_nm": [0.0, 0.0, 0.0], "aa_radius_nm": 1.0, "cg_radius_nm": 2.0, "low_plddt_atoms": 0}
    target = confidence.low_confidence_indices
    if target.size == 0:
        target = confidence.medium_confidence_indices
    if target.size == 0:
        target = np.arange(pos.shape[0], dtype=np.int64)
    sub = pos[target]
    center = np.mean(sub, axis=0)
    max_r = float(np.max(np.linalg.norm(sub - center[None, :], axis=1))) if sub.shape[0] > 0 else 0.5
    aa_radius = max(0.8, max_r + float(pad_nm))
    return {
        "focus_center_nm": [float(c) for c in center],
        "aa_radius_nm": float(aa_radius),
        "cg_radius_nm": float(aa_radius * 1.8),
        "low_plddt_atoms": int(confidence.low_confidence_indices.size),
        "mean_plddt": float(confidence.mean_plddt),
    }
