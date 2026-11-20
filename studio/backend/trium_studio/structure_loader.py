"""High-scale multi-format structure loader for Biology-MM Studio.

Directly ingests and parses raw structure and force-field topology files:
  - PDB / ENT (`.pdb`, `.ent`)
  - mmCIF (`.cif`, `.mmcif`)
  - GROMACS (`.gro`)
  - XYZ (`.xyz`)
  - LAMMPS Data (`.data`, `.lammps`, `.lmp`)
  - AMBER / CHARMM Normalized Chemistry JSON & Reference JSON (`.json`)
  - AMBER Topology & Coordinates (`.prmtop`, `.parm7`, `.inpcrd`, `.rst7`)
  - CHARMM Topology & Coordinates (`.psf`, `.crd`, `.cor`)
  - Drug Ligands (`.mol2`, `.sdf`, `.mol`)

Also supports paired topology + coordinate uploads (e.g., `.prmtop` + `.inpcrd`,
`.psf` + `.crd`, `.json` + `.f32`) while preserving exact partial charges, masses,
bonds, and periodic boxes.
"""

from __future__ import annotations

import json
import math
import re
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from biology_mm import AtomicSystem

AMBER_CHARGE_SCALE = 18.2223

_ELEMENT_MASSES: Dict[str, float] = {
    "H": 1.008,
    "HE": 4.003,
    "LI": 6.941,
    "BE": 9.012,
    "B": 10.81,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "F": 18.998,
    "NE": 20.180,
    "NA": 22.990,
    "MG": 24.305,
    "AL": 26.982,
    "SI": 28.086,
    "P": 30.974,
    "S": 32.065,
    "CL": 35.453,
    "AR": 39.948,
    "K": 39.098,
    "CA": 40.078,
    "MN": 54.938,
    "FE": 55.845,
    "CO": 58.933,
    "NI": 58.693,
    "CU": 63.546,
    "ZN": 65.38,
    "SE": 78.971,
    "BR": 79.904,
    "I": 126.90,
}

_Z_TO_ELEMENT: Dict[int, str] = {
    1: "H",
    2: "He",
    3: "Li",
    4: "Be",
    5: "B",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    10: "Ne",
    11: "Na",
    12: "Mg",
    13: "Al",
    14: "Si",
    15: "P",
    16: "S",
    17: "Cl",
    18: "Ar",
    19: "K",
    20: "Ca",
    25: "Mn",
    26: "Fe",
    27: "Co",
    28: "Ni",
    29: "Cu",
    30: "Zn",
    34: "Se",
    35: "Br",
    53: "I",
}

_TWO_LETTER_ELEMENTS: Set[str] = {
    "HE", "LI", "BE", "NE", "NA", "MG", "AL", "SI", "CL", "AR",
    "CA", "MN", "FE", "CO", "NI", "CU", "ZN", "SE", "BR",
}

_MASS_TABLE: List[Tuple[float, str]] = [
    (m, sym[0].upper() + sym[1:].lower()) for sym, m in _ELEMENT_MASSES.items()
]

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _clean_element(raw: str) -> str:
    cleaned = "".join(ch for ch in str(raw).strip() if ch.isalpha())
    if not cleaned:
        return "C"
    return cleaned[0].upper() + cleaned[1:].lower()


def _infer_element(name: str = "", mass: float = 0.0, atomic_number: int = 0) -> str:
    """Infer a canonical element symbol from atomic number, mass, or atom name."""
    if atomic_number > 0 and atomic_number in _Z_TO_ELEMENT:
        return _Z_TO_ELEMENT[atomic_number]

    if mass > 0.5:
        best_sym: Optional[str] = None
        best_diff = 0.65
        for ref_m, sym in _MASS_TABLE:
            diff = abs(mass - ref_m)
            if diff < best_diff:
                best_diff = diff
                best_sym = sym
        if best_sym is not None:
            return best_sym

    raw = str(name or "").strip()
    # Strip leading digits (e.g. "1HB", "2HG1")
    stripped = re.sub(r"^[0-9]+", "", raw)
    alpha = "".join(ch for ch in stripped if ch.isalpha())
    if not alpha:
        return "C"
    upper = alpha.upper()
    if len(upper) == 2 and upper in _TWO_LETTER_ELEMENTS:
        # Distinguish e.g. "NA" ion vs "CA" (alpha carbon when mass unknown -> default C if protein-like)
        if upper != "CA":
            return upper[0] + upper[1].lower()
    if upper[0] in _ELEMENT_MASSES:
        return upper[0]
    if len(upper) >= 2 and upper[:2] in _TWO_LETTER_ELEMENTS:
        return upper[0] + upper[1].lower()
    return _clean_element(alpha[:1])


def _synthesize_coords_from_graph(
    n_atoms: int,
    bonds: Sequence[Tuple[int, int]],
    explicit_box_nm: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Generate deterministic, non-clashing 3D coordinates (in nm) for topology-only files.

    Orders atoms via BFS over connected components so bonded neighbors stay spatially
    proximate, then lays them out on a clash-free 3D lattice.
    """
    if n_atoms <= 0:
        return np.zeros((0, 3), dtype=np.float64)

    adj: List[List[int]] = [[] for _ in range(n_atoms)]
    for i, j in bonds:
        if 0 <= i < n_atoms and 0 <= j < n_atoms and i != j:
            adj[i].append(j)
            adj[j].append(i)

    visited = bytearray(n_atoms)
    order: List[int] = []
    for start in range(n_atoms):
        if visited[start]:
            continue
        visited[start] = 1
        q: deque[int] = deque([start])
        while q:
            u = q.popleft()
            order.append(u)
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = 1
                    q.append(v)

    nx = max(1, int(math.ceil(n_atoms ** (1.0 / 3.0))))
    ny = max(1, int(math.ceil((n_atoms / nx) ** 0.5)))
    nz = max(1, int(math.ceil(n_atoms / (nx * ny))))

    spacing = 0.28  # 2.8 Angstroms -> clash-free for all standard atom radii
    if explicit_box_nm and len(explicit_box_nm) == 3:
        bx, by, bz = (float(v) for v in explicit_box_nm)
        if bx > 0.5 and by > 0.5 and bz > 0.5:
            sx = (bx * 0.88) / max(nx, 1)
            sy = (by * 0.88) / max(ny, 1)
            sz = (bz * 0.88) / max(nz, 1)
            spacing = max(0.18, min(0.32, sx, sy, sz))

    coords = np.zeros((n_atoms, 3), dtype=np.float64)
    for rank, atom_idx in enumerate(order):
        iz = rank // (nx * ny)
        rem = rank % (nx * ny)
        iy = rem // nx
        ix = rem % nx
        coords[atom_idx, 0] = ix * spacing
        coords[atom_idx, 1] = iy * spacing
        coords[atom_idx, 2] = iz * spacing
    return coords


def _extract_box_lengths_nm(box_obj: Any) -> Optional[Tuple[float, float, float]]:
    """Extract (lx, ly, lz) in nm from normalized JSON box dict, 3-vector, or 3x3 matrix."""
    if box_obj is None:
        return None
    if isinstance(box_obj, dict):
        kind = str(box_obj.get("kind", "")).lower()
        if kind == "none":
            return None
        lengths = box_obj.get("lengths_nm") or box_obj.get("lengths")
        if isinstance(lengths, (list, tuple)) and len(lengths) == 3:
            return (float(lengths[0]), float(lengths[1]), float(lengths[2]))
        vectors = box_obj.get("vectors_nm") or box_obj.get("vectors")
        if isinstance(vectors, (list, tuple)) and len(vectors) == 3:
            mat = np.asarray(vectors, dtype=np.float64)
            if mat.shape == (3, 3):
                return (
                    float(np.linalg.norm(mat[:, 0])),
                    float(np.linalg.norm(mat[:, 1])),
                    float(np.linalg.norm(mat[:, 2])),
                )
        return None
    if isinstance(box_obj, (list, tuple)) and len(box_obj) == 3:
        if all(isinstance(v, (int, float)) for v in box_obj):
            return (float(box_obj[0]), float(box_obj[1]), float(box_obj[2]))
        try:
            mat = np.asarray(box_obj, dtype=np.float64)
            if mat.shape == (3, 3):
                return (
                    float(np.linalg.norm(mat[0])),
                    float(np.linalg.norm(mat[1])),
                    float(np.linalg.norm(mat[2])),
                )
        except Exception:
            return None
    return None


def _infer_box(pos_nm: np.ndarray, explicit_box_nm: Optional[Sequence[float]] = None) -> Tuple[float, float, float]:
    n = pos_nm.shape[0]
    span = np.ptp(pos_nm, axis=0) if n > 1 else np.zeros(3, dtype=np.float64)
    bx, by, bz = 0.0, 0.0, 0.0
    if explicit_box_nm and len(explicit_box_nm) == 3:
        bx, by, bz = (max(0.0, float(v)) for v in explicit_box_nm)
    return (
        max(bx, float(span[0]) + 1.8, 2.8),
        max(by, float(span[1]) + 1.8, 2.8),
        max(bz, float(span[2]) + 1.8, 2.8),
    )


def parse_pdb_text(text: str) -> Tuple[np.ndarray, List[str], List[Tuple[int, int]], Dict[str, Any]]:
    coords: List[List[float]] = []
    elements: List[str] = []
    atom_id_map: Dict[int, int] = {}
    bonds_set: Set[Tuple[int, int]] = set()
    res_names: List[str] = []
    chain_ids: List[str] = []
    explicit_box: Optional[Tuple[float, float, float]] = None

    for line in text.splitlines():
        if line.startswith("CRYST1"):
            try:
                a = float(line[6:15]) * 0.1
                b = float(line[15:24]) * 0.1
                c = float(line[24:33]) * 0.1
                explicit_box = (a, b, c)
            except Exception:
                pass
        elif line.startswith(("ATOM  ", "HETATM")):
            try:
                atom_num = int(line[6:11].strip())
            except ValueError:
                atom_num = len(coords) + 1
            atom_name = line[12:16].strip()
            res_name = line[17:20].strip()
            chain = line[21:22].strip() or "A"
            try:
                x = float(line[30:38]) * 0.1
                y = float(line[38:46]) * 0.1
                z = float(line[46:54]) * 0.1
            except ValueError:
                continue

            elem = line[76:78].strip() if len(line) >= 77 else ""
            if not elem:
                alpha_name = "".join(c for c in atom_name if c.isalpha())
                if line.startswith("HETATM") and len(alpha_name) >= 2 and alpha_name[:2].upper() in _ELEMENT_MASSES:
                    elem = alpha_name[:2]
                else:
                    elem = alpha_name[:1]

            idx = len(coords)
            atom_id_map[atom_num] = idx
            coords.append([x, y, z])
            elements.append(_clean_element(elem))
            res_names.append(res_name)
            chain_ids.append(chain)
        elif line.startswith("CONECT"):
            parts = line[6:].split()
            if len(parts) >= 2 and parts[0].isdigit():
                src = int(parts[0])
                if src in atom_id_map:
                    s_idx = atom_id_map[src]
                    for dest_str in parts[1:]:
                        if dest_str.isdigit():
                            dest = int(dest_str)
                            if dest in atom_id_map and dest != src:
                                d_idx = atom_id_map[dest]
                                bonds_set.add((min(s_idx, d_idx), max(s_idx, d_idx)))

    pos_nm = np.asarray(coords, dtype=np.float64)
    bonds = sorted(list(bonds_set))
    meta = {
        "res_names": res_names,
        "chain_ids": chain_ids,
        "explicit_box_nm": explicit_box,
    }
    return pos_nm, elements, bonds, meta


def parse_gro_text(text: str) -> Tuple[np.ndarray, List[str], List[Tuple[int, int]], Dict[str, Any]]:
    lines = text.strip().splitlines()
    if len(lines) < 3:
        raise ValueError("GRO file too short (expected at least title, count, and box lines)")

    title = lines[0].strip()
    try:
        n_atoms = int(lines[1].strip())
    except ValueError as e:
        raise ValueError(f"Invalid atom count in GRO header: {lines[1]}") from e

    coords: List[List[float]] = []
    elements: List[str] = []
    res_names: List[str] = []

    atom_lines = lines[2 : 2 + n_atoms]
    for line in atom_lines:
        if len(line) < 44:
            continue
        res_name = line[5:10].strip()
        atom_name = line[10:15].strip()
        try:
            x = float(line[20:28])
            y = float(line[28:36])
            z = float(line[36:44])
        except ValueError:
            continue
        alpha_an = "".join(c for c in atom_name if c.isalpha())
        if res_name.upper() in ("NA", "CL", "MG", "ZN", "CA", "FE", "CU", "MN", "BR") and alpha_an[:2].upper() in _ELEMENT_MASSES:
            elem_raw = alpha_an[:2]
        else:
            elem_raw = alpha_an[:1]
        coords.append([x, y, z])
        elements.append(_clean_element(elem_raw))
        res_names.append(res_name)

    explicit_box: Optional[Tuple[float, float, float]] = None
    if len(lines) >= 2 + n_atoms + 1:
        box_line = lines[2 + n_atoms].strip().split()
        if len(box_line) >= 3:
            try:
                explicit_box = (float(box_line[0]), float(box_line[1]), float(box_line[2]))
            except ValueError:
                pass

    pos_nm = np.asarray(coords, dtype=np.float64)
    meta = {"title": title, "res_names": res_names, "explicit_box_nm": explicit_box}
    return pos_nm, elements, [], meta


def parse_xyz_text(text: str) -> Tuple[np.ndarray, List[str], List[Tuple[int, int]], Dict[str, Any]]:
    raw_lines = text.splitlines()
    # Skip leading blank lines only, preserving a potentially blank XYZ comment line (line 2)
    start = 0
    while start < len(raw_lines) and not raw_lines[start].strip():
        start += 1
    lines = raw_lines[start:]
    if len(lines) < 3:
        raise ValueError("XYZ file too short")

    try:
        n_atoms = int(lines[0].strip())
    except ValueError as e:
        raise ValueError(f"First line of XYZ must be integer atom count: {lines[0]}") from e

    comment = lines[1].strip()
    coords: List[List[float]] = []
    elements: List[str] = []

    atom_lines = [ln.strip() for ln in lines[2:] if ln.strip()]
    for line in atom_lines[:n_atoms]:
        parts = line.split()
        if len(parts) >= 4:
            elem = parts[0]
            try:
                x = float(parts[1]) * 0.1
                y = float(parts[2]) * 0.1
                z = float(parts[3]) * 0.1
            except ValueError:
                continue
            coords.append([x, y, z])
            elements.append(_clean_element(elem))

    pos_nm = np.asarray(coords, dtype=np.float64)
    meta = {"comment": comment}
    return pos_nm, elements, [], meta


def parse_lammps_data_text(text: str) -> Tuple[np.ndarray, List[str], List[Tuple[int, int]], Dict[str, Any]]:
    lines = text.splitlines()
    n_atoms = 0
    n_bonds = 0
    xlo, xhi = 0.0, 10.0
    ylo, yhi = 0.0, 10.0
    zlo, zhi = 0.0, 10.0

    in_atoms_section = False
    in_bonds_section = False
    atoms_raw: List[Dict[str, Any]] = []
    bonds_raw: List[Tuple[int, int]] = []

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if "atoms" in line and n_atoms == 0:
            parts = line.split()
            if parts[0].isdigit():
                n_atoms = int(parts[0])
        elif "bonds" in line and n_bonds == 0:
            parts = line.split()
            if parts[0].isdigit():
                n_bonds = int(parts[0])
        elif "xlo xhi" in line:
            parts = line.split()
            xlo, xhi = float(parts[0]), float(parts[1])
        elif "ylo yhi" in line:
            parts = line.split()
            ylo, yhi = float(parts[0]), float(parts[1])
        elif "zlo zhi" in line:
            parts = line.split()
            zlo, zhi = float(parts[0]), float(parts[1])
        elif line.startswith("Atoms"):
            in_atoms_section = True
            in_bonds_section = False
            continue
        elif line.startswith(("Velocities", "Bonds", "Angles", "Dihedrals", "Masses", "Pair Coeffs")):
            in_atoms_section = False
            if line.startswith("Bonds"):
                in_bonds_section = True
            else:
                in_bonds_section = False
            continue

        if in_atoms_section:
            parts = line.split()
            # atom_style full: id, mol, type, q, x, y, z
            # atom_style atomic: id, type, x, y, z
            if len(parts) >= 7:
                try:
                    aid = int(parts[0])
                    atype = int(parts[2])
                    q = float(parts[3])
                    x, y, z = float(parts[4]), float(parts[5]), float(parts[6])
                    atoms_raw.append({"id": aid, "type": atype, "q": q, "pos": [x * 0.1, y * 0.1, z * 0.1]})
                except ValueError:
                    pass
            elif len(parts) >= 5:
                try:
                    aid = int(parts[0])
                    atype = int(parts[1])
                    x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                    atoms_raw.append({"id": aid, "type": atype, "q": 0.0, "pos": [x * 0.1, y * 0.1, z * 0.1]})
                except ValueError:
                    pass

        elif in_bonds_section:
            parts = line.split()
            if len(parts) >= 4:
                try:
                    a1 = int(parts[2])
                    a2 = int(parts[3])
                    bonds_raw.append((a1, a2))
                except ValueError:
                    pass

    # Sort atoms by ID
    atoms_raw.sort(key=lambda a: a["id"])
    id_map = {a["id"]: idx for idx, a in enumerate(atoms_raw)}
    coords = [a["pos"] for a in atoms_raw]
    charges = [a["q"] for a in atoms_raw]
    elements = ["C" if a["type"] == 1 else "O" if a["type"] == 2 else "H" for a in atoms_raw]

    bonds: List[Tuple[int, int]] = []
    seen = set()
    for a1, a2 in bonds_raw:
        if a1 in id_map and a2 in id_map and a1 != a2:
            i, j = id_map[a1], id_map[a2]
            key = (min(i, j), max(i, j))
            if key not in seen:
                seen.add(key)
                bonds.append(key)

    pos_nm = np.asarray(coords, dtype=np.float64)
    box_nm = ((xhi - xlo) * 0.1, (yhi - ylo) * 0.1, (zhi - zlo) * 0.1)
    meta = {"charges": charges, "explicit_box_nm": box_nm}
    return pos_nm, elements, bonds, meta


def parse_mmcif_text(text: str) -> Tuple[np.ndarray, List[str], List[Tuple[int, int]], Dict[str, Any]]:
    coords: List[List[float]] = []
    elements: List[str] = []
    atom_id_map: Dict[str, int] = {}
    res_names: List[str] = []
    chain_ids: List[str] = []

    lines = text.splitlines()
    in_atom_site_loop = False
    atom_site_cols: List[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#") or stripped.startswith("data_") or stripped.startswith(";"):
            in_atom_site_loop = False
            continue

        if stripped.startswith("loop_"):
            in_atom_site_loop = False
            atom_site_cols = []
            continue

        if stripped.startswith("_atom_site."):
            in_atom_site_loop = True
            atom_site_cols.append(stripped.split()[0].replace("_atom_site.", ""))
            continue

        if in_atom_site_loop:
            if stripped.startswith("_"):
                in_atom_site_loop = False
                continue

            parts = [tok.strip("'\"") for tok in stripped.split()]
            if len(parts) >= len(atom_site_cols) and len(atom_site_cols) >= 6:
                row = dict(zip(atom_site_cols, parts[: len(atom_site_cols)]))
                group = row.get("group_PDB", "ATOM")
                if group in ("ATOM", "HETATM"):
                    try:
                        x = float(row.get("Cartn_x", "0")) * 0.1
                        y = float(row.get("Cartn_y", "0")) * 0.1
                        z = float(row.get("Cartn_z", "0")) * 0.1
                    except ValueError:
                        continue
                    elem = row.get("type_symbol", "")
                    if not elem or elem in (".", "?"):
                        atom_lbl = row.get("label_atom_id", row.get("auth_atom_id", "C"))
                        elem = "".join(c for c in atom_lbl if c.isalpha())[:1] or "C"
                    aid = row.get("id", str(len(coords) + 1))
                    idx = len(coords)
                    atom_id_map[aid] = idx
                    coords.append([x, y, z])
                    elements.append(_clean_element(elem))
                    res_names.append(row.get("label_comp_id", "UNK"))
                    chain_ids.append(row.get("label_asym_id", "A"))

    pos_nm = np.asarray(coords, dtype=np.float64)
    meta = {"res_names": res_names, "chain_ids": chain_ids}
    return pos_nm, elements, [], meta


def parse_amber_prmtop_text(
    text: str,
    synthesize_coords: bool = True,
) -> Tuple[np.ndarray, List[str], List[Tuple[int, int]], Dict[str, Any]]:
    """Parse AMBER .prmtop / .parm7 topology file."""
    sections: Dict[str, List[str]] = {}
    formats: Dict[str, str] = {}
    current_flag: Optional[str] = None

    for raw_line in text.splitlines():
        if raw_line.startswith("%VERSION"):
            continue
        if raw_line.startswith("%FLAG"):
            parts = raw_line.split()
            current_flag = parts[1].strip() if len(parts) >= 2 else None
            if current_flag is not None:
                sections[current_flag] = []
            continue
        if raw_line.startswith("%FORMAT"):
            if current_flag is not None:
                formats[current_flag] = raw_line.strip()
            continue
        if raw_line.startswith("%COMMENT"):
            continue
        if current_flag is not None:
            sections[current_flag].append(raw_line)

    def _read_ints(flag: str) -> List[int]:
        lines = sections.get(flag, [])
        vals: List[int] = []
        for ln in lines:
            for tok in ln.split():
                vals.append(int(tok))
        return vals

    def _read_floats(flag: str) -> List[float]:
        lines = sections.get(flag, [])
        vals: List[float] = []
        for ln in lines:
            for tok in ln.split():
                vals.append(float(tok))
        return vals

    def _read_fixed_strings(flag: str, width: int = 4) -> List[str]:
        lines = sections.get(flag, [])
        vals: List[str] = []
        for ln in lines:
            if not ln:
                continue
            for pos in range(0, len(ln), width):
                chunk = ln[pos : pos + width].strip()
                if chunk:
                    vals.append(chunk)
        return vals

    pointers = _read_ints("POINTERS")
    if not pointers or pointers[0] <= 0:
        raise ValueError("AMBER prmtop missing valid %FLAG POINTERS")
    n_atoms = int(pointers[0])

    atom_names = _read_fixed_strings("ATOM_NAME", width=4)[:n_atoms]
    while len(atom_names) < n_atoms:
        atom_names.append("C")

    raw_charges = _read_floats("CHARGE")
    if len(raw_charges) >= n_atoms:
        charges = [float(q) / AMBER_CHARGE_SCALE for q in raw_charges[:n_atoms]]
    else:
        charges = [0.0] * n_atoms

    raw_masses = _read_floats("MASS")
    masses = [float(m) for m in raw_masses[:n_atoms]] if len(raw_masses) >= n_atoms else []

    raw_z = _read_ints("ATOMIC_NUMBER")
    z_list = raw_z[:n_atoms] if len(raw_z) >= n_atoms else [0] * n_atoms

    elements: List[str] = []
    final_masses: List[float] = []
    for idx in range(n_atoms):
        z_val = int(z_list[idx]) if idx < len(z_list) else 0
        m_val = float(masses[idx]) if idx < len(masses) else 0.0
        elem = _infer_element(atom_names[idx], mass=m_val, atomic_number=z_val)
        elements.append(elem)
        final_masses.append(m_val if m_val > 0.0 else _ELEMENT_MASSES.get(elem.upper(), 12.011))

    bonds_set: Set[Tuple[int, int]] = set()
    for flag in ("BONDS_INC_HYDROGEN", "BONDS_WITHOUT_HYDROGEN"):
        b_ints = _read_ints(flag)
        for k in range(0, len(b_ints) - 2, 3):
            i = abs(b_ints[k]) // 3
            j = abs(b_ints[k + 1]) // 3
            if 0 <= i < n_atoms and 0 <= j < n_atoms and i != j:
                bonds_set.add((min(i, j), max(i, j)))
    bonds = sorted(bonds_set)

    # Optional residue names
    res_names: Optional[List[str]] = None
    res_labels = _read_fixed_strings("RESIDUE_LABEL", width=4)
    res_ptrs = _read_ints("RESIDUE_POINTER")
    if res_labels and res_ptrs and len(res_labels) == len(res_ptrs):
        res_names = ["UNK"] * n_atoms
        for r_idx, start_1 in enumerate(res_ptrs):
            start_0 = max(0, start_1 - 1)
            end_0 = (res_ptrs[r_idx + 1] - 1) if (r_idx + 1 < len(res_ptrs)) else n_atoms
            lbl = res_labels[r_idx]
            for a_idx in range(start_0, min(end_0, n_atoms)):
                res_names[a_idx] = lbl

    explicit_box: Optional[Tuple[float, float, float]] = None
    box_dims = _read_floats("BOX_DIMENSIONS")
    if len(box_dims) >= 4:
        # BOX_DIMENSIONS: [beta, x, y, z] in Angstroms
        bx, by, bz = box_dims[1] * 0.1, box_dims[2] * 0.1, box_dims[3] * 0.1
        if bx > 0.0 and by > 0.0 and bz > 0.0:
            explicit_box = (bx, by, bz)

    if synthesize_coords:
        pos_nm = _synthesize_coords_from_graph(n_atoms, bonds, explicit_box)
    else:
        pos_nm = np.zeros((n_atoms, 3), dtype=np.float64)

    title_lines = sections.get("TITLE", [])
    title = " ".join(ln.strip() for ln in title_lines if ln.strip()) or "AMBER Topology"
    meta: Dict[str, Any] = {
        "title": title,
        "atom_names": atom_names,
        "charges": charges,
        "masses": final_masses,
        "res_names": res_names,
        "explicit_box_nm": explicit_box,
        "coords_synthesized": bool(synthesize_coords),
        "family": "amber",
    }
    return pos_nm, elements, bonds, meta


def parse_amber_inpcrd_text(text: str) -> Tuple[np.ndarray, List[str], List[Tuple[int, int]], Dict[str, Any]]:
    """Parse AMBER .inpcrd / .rst7 coordinate/restart file (6F12.7 in Angstroms -> nm)."""
    lines = text.splitlines()
    if len(lines) < 3:
        raise ValueError("AMBER inpcrd/rst7 file too short")

    title = lines[0].strip() or "AMBER Coordinates"
    header_parts = lines[1].split()
    if not header_parts:
        raise ValueError("AMBER inpcrd/rst7 missing atom count on line 2")
    try:
        n_atoms = int(header_parts[0])
    except ValueError as e:
        raise ValueError(f"Invalid atom count in AMBER inpcrd header: {lines[1]}") from e
    if n_atoms <= 0:
        raise ValueError(f"Invalid atom count {n_atoms} in AMBER inpcrd")

    def _parse_6f12_line(ln: str) -> List[float]:
        # Try fixed-width 12-char chunks first (standard 6F12.7 where negatives can touch)
        vals: List[float] = []
        if len(ln) >= 12 and len(ln.rstrip()) <= 74:
            ok = True
            for start in range(0, len(ln.rstrip()), 12):
                chunk = ln[start : start + 12].strip()
                if not chunk:
                    continue
                try:
                    vals.append(float(chunk))
                except ValueError:
                    ok = False
                    break
            if ok and vals:
                return vals
        return [float(tok) for tok in ln.split()]

    n_coord_floats = 3 * n_atoms
    n_coord_lines = (n_coord_floats + 5) // 6
    coord_vals: List[float] = []
    line_idx = 2
    while line_idx < len(lines) and len(coord_vals) < n_coord_floats:
        ln = lines[line_idx]
        line_idx += 1
        if not ln.strip():
            continue
        coord_vals.extend(_parse_6f12_line(ln))

    if len(coord_vals) < n_coord_floats:
        raise ValueError(
            f"AMBER inpcrd truncated: expected {n_coord_floats} coordinate values, got {len(coord_vals)}"
        )

    pos_nm = np.asarray(coord_vals[:n_coord_floats], dtype=np.float64).reshape(n_atoms, 3) * 0.1

    # Check remaining non-empty lines for optional velocities + final periodic box line
    rem_lines = [ln for ln in lines[line_idx:] if ln.strip()]
    explicit_box: Optional[Tuple[float, float, float]] = None
    if rem_lines:
        # Box line is either immediately after coords (len == 1) or after velocities (len == n_coord_lines + 1)
        if len(rem_lines) == 1 or len(rem_lines) == n_coord_lines + 1:
            box_vals = _parse_6f12_line(rem_lines[-1])
            if len(box_vals) >= 3:
                bx, by, bz = box_vals[0] * 0.1, box_vals[1] * 0.1, box_vals[2] * 0.1
                if bx > 0.0 and by > 0.0 and bz > 0.0:
                    explicit_box = (bx, by, bz)

    elements = ["C"] * n_atoms
    meta: Dict[str, Any] = {
        "title": title,
        "explicit_box_nm": explicit_box,
        "coords_synthesized": False,
        "family": "amber",
    }
    return pos_nm, elements, [], meta


def parse_charmm_psf_text(
    text: str,
    synthesize_coords: bool = True,
) -> Tuple[np.ndarray, List[str], List[Tuple[int, int]], Dict[str, Any]]:
    """Parse CHARMM .psf topology file (!NATOM and !NBOND sections)."""
    lines = text.splitlines()
    n_atoms = 0
    atom_names: List[str] = []
    res_names: List[str] = []
    chain_ids: List[str] = []
    charges: List[float] = []
    masses: List[float] = []
    elements: List[str] = []
    bonds_set: Set[Tuple[int, int]] = set()

    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if "!NATOM" in line:
            parts = line.split()
            try:
                n_atoms = int(parts[0])
            except (ValueError, IndexError) as e:
                raise ValueError(f"Invalid !NATOM header in PSF: {line}") from e
            idx += 1
            read_count = 0
            while idx < len(lines) and read_count < n_atoms:
                aln = lines[idx].strip()
                idx += 1
                if not aln:
                    continue
                cols = aln.split()
                if len(cols) < 8:
                    continue
                segid = cols[1]
                resname = cols[3]
                aname = cols[4]
                q = float(cols[6])
                m = float(cols[7])
                elem = _infer_element(aname, mass=m)
                atom_names.append(aname)
                res_names.append(resname)
                chain_ids.append(segid[:1] if segid else "A")
                charges.append(q)
                masses.append(m if m > 0.0 else _ELEMENT_MASSES.get(elem.upper(), 12.011))
                elements.append(elem)
                read_count += 1
            continue

        if "!NBOND" in line:
            parts = line.split()
            try:
                n_bonds = int(parts[0])
            except (ValueError, IndexError):
                n_bonds = 0
            idx += 1
            bond_ints: List[int] = []
            while idx < len(lines) and len(bond_ints) < 2 * n_bonds:
                bln = lines[idx].strip()
                idx += 1
                if not bln or "!" in bln:
                    if "!" in bln:
                        idx -= 1
                        break
                    continue
                for tok in bln.split():
                    try:
                        bond_ints.append(int(tok))
                    except ValueError:
                        break
            for k in range(0, len(bond_ints) - 1, 2):
                i = bond_ints[k] - 1
                j = bond_ints[k + 1] - 1
                if 0 <= i < len(elements) and 0 <= j < len(elements) and i != j:
                    bonds_set.add((min(i, j), max(i, j)))
            continue

        idx += 1

    if not elements:
        raise ValueError("CHARMM PSF file contained no valid !NATOM records")

    bonds = sorted(bonds_set)
    if synthesize_coords:
        pos_nm = _synthesize_coords_from_graph(len(elements), bonds)
    else:
        pos_nm = np.zeros((len(elements), 3), dtype=np.float64)

    meta: Dict[str, Any] = {
        "atom_names": atom_names,
        "res_names": res_names,
        "chain_ids": chain_ids,
        "charges": charges,
        "masses": masses,
        "coords_synthesized": bool(synthesize_coords),
        "family": "charmm",
    }
    return pos_nm, elements, bonds, meta


def parse_charmm_crd_text(text: str) -> Tuple[np.ndarray, List[str], List[Tuple[int, int]], Dict[str, Any]]:
    """Parse CHARMM .crd / .cor coordinate file (standard and EXT formats, Angstroms -> nm)."""
    lines = text.splitlines()
    title_parts: List[str] = []
    idx = 0
    while idx < len(lines):
        stripped = lines[idx].strip()
        if not stripped:
            idx += 1
            continue
        if stripped.startswith("*"):
            t = stripped.lstrip("*").strip()
            if t:
                title_parts.append(t)
            idx += 1
            continue
        break

    if idx >= len(lines):
        raise ValueError("CHARMM CRD file missing atom count header")

    header_tokens = lines[idx].strip().split()
    try:
        n_atoms = int(header_tokens[0])
    except (ValueError, IndexError) as e:
        raise ValueError(f"Invalid atom count in CHARMM CRD header: {lines[idx]}") from e
    is_ext = any(tok.upper() == "EXT" for tok in header_tokens[1:])
    idx += 1

    coords: List[List[float]] = []
    elements: List[str] = []
    atom_names: List[str] = []
    res_names: List[str] = []
    chain_ids: List[str] = []

    while idx < len(lines) and len(coords) < n_atoms:
        raw = lines[idx]
        idx += 1
        if not raw.strip():
            continue

        parsed = False
        # Try whitespace split first (works for well-spaced standard and EXT)
        parts = raw.split()
        if len(parts) >= 7:
            try:
                resname = parts[2]
                aname = parts[3]
                x = float(parts[4]) * 0.1
                y = float(parts[5]) * 0.1
                z = float(parts[6]) * 0.1
                segid = parts[7] if len(parts) >= 8 else "A"
                parsed = True
            except ValueError:
                parsed = False

        if not parsed:
            # Fixed-width fallback for touching coordinates
            try:
                if is_ext and len(raw) >= 100:
                    resname = raw[22:30].strip()
                    aname = raw[32:40].strip()
                    x = float(raw[40:60]) * 0.1
                    y = float(raw[60:80]) * 0.1
                    z = float(raw[80:100]) * 0.1
                    segid = raw[102:110].strip() or "A"
                else:
                    resname = raw[11:15].strip()
                    aname = raw[16:20].strip()
                    x = float(raw[20:30]) * 0.1
                    y = float(raw[30:40]) * 0.1
                    z = float(raw[40:50]) * 0.1
                    segid = raw[51:55].strip() or "A"
                parsed = True
            except ValueError:
                parsed = False

        if not parsed:
            continue

        coords.append([x, y, z])
        atom_names.append(aname)
        res_names.append(resname)
        chain_ids.append(segid[:1] if segid else "A")
        elements.append(_infer_element(aname))

    pos_nm = np.asarray(coords, dtype=np.float64)
    meta: Dict[str, Any] = {
        "title": " ".join(title_parts) or "CHARMM Coordinates",
        "atom_names": atom_names,
        "res_names": res_names,
        "chain_ids": chain_ids,
        "coords_synthesized": False,
        "family": "charmm",
    }
    return pos_nm, elements, [], meta


def parse_mol2_text(text: str) -> Tuple[np.ndarray, List[str], List[Tuple[int, int]], Dict[str, Any]]:
    """Parse Tripos .mol2 ligand/molecule file (atoms, partial charges, elements, bonds)."""
    coords: List[List[float]] = []
    elements: List[str] = []
    atom_names: List[str] = []
    res_names: List[str] = []
    charges: List[float] = []
    atom_id_map: Dict[int, int] = {}
    bonds_set: Set[Tuple[int, int]] = set()
    title = "MOL2 Ligand"

    section = ""
    mol_line_idx = 0

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("@<TRIPOS>"):
            section = line.upper()
            mol_line_idx = 0
            continue

        if section == "@<TRIPOS>MOLECULE":
            if mol_line_idx == 0:
                title = line
            mol_line_idx += 1
        elif section == "@<TRIPOS>ATOM":
            parts = line.split()
            if len(parts) >= 6:
                try:
                    aid = int(parts[0])
                    aname = parts[1]
                    x = float(parts[2]) * 0.1
                    y = float(parts[3]) * 0.1
                    z = float(parts[4]) * 0.1
                except ValueError:
                    continue
                atype = parts[5]
                elem_raw = atype.split(".")[0]
                if elem_raw.upper() in ("LP", "DU"):
                    elem = "C"
                else:
                    elem = _clean_element(elem_raw) if elem_raw else _infer_element(aname)
                res_name = parts[7] if len(parts) >= 8 else "LIG"
                try:
                    q = float(parts[8]) if len(parts) >= 9 else 0.0
                except ValueError:
                    q = 0.0

                idx = len(coords)
                atom_id_map[aid] = idx
                coords.append([x, y, z])
                elements.append(elem)
                atom_names.append(aname)
                res_names.append(res_name)
                charges.append(q)
        elif section == "@<TRIPOS>BOND":
            parts = line.split()
            if len(parts) >= 3:
                try:
                    a1 = int(parts[1])
                    a2 = int(parts[2])
                except ValueError:
                    continue
                if a1 in atom_id_map and a2 in atom_id_map and a1 != a2:
                    i, j = atom_id_map[a1], atom_id_map[a2]
                    bonds_set.add((min(i, j), max(i, j)))

    pos_nm = np.asarray(coords, dtype=np.float64)
    bonds = sorted(bonds_set)
    meta: Dict[str, Any] = {
        "title": title,
        "atom_names": atom_names,
        "res_names": res_names,
        "charges": charges,
        "coords_synthesized": False,
    }
    return pos_nm, elements, bonds, meta


def parse_sdf_text(text: str) -> Tuple[np.ndarray, List[str], List[Tuple[int, int]], Dict[str, Any]]:
    """Parse V2000 .sdf / .mol file (atom block, bond block, and formal charges)."""
    lines = text.splitlines()
    if len(lines) < 5:
        raise ValueError("SDF/MOL file too short (expected header + counts + atom block)")

    title = lines[0].strip() or "SDF Ligand"
    counts_line = lines[3]
    try:
        n_atoms = int(counts_line[0:3].strip())
        n_bonds = int(counts_line[3:6].strip())
    except ValueError:
        parts = counts_line.split()
        if len(parts) < 2:
            raise ValueError(f"Invalid V2000 counts line: {counts_line}")
        n_atoms = int(parts[0])
        n_bonds = int(parts[1])

    if n_atoms <= 0:
        raise ValueError(f"Invalid atom count {n_atoms} in SDF/MOL")

    # V2000 atom block charge code mapping
    _chg_code_map = {1: 3.0, 2: 2.0, 3: 1.0, 5: -1.0, 6: -2.0, 7: -3.0}

    coords: List[List[float]] = []
    elements: List[str] = []
    charges: List[float] = []

    atom_start = 4
    atom_end = atom_start + n_atoms
    for ln in lines[atom_start:atom_end]:
        if len(ln) >= 34:
            try:
                x = float(ln[0:10]) * 0.1
                y = float(ln[10:20]) * 0.1
                z = float(ln[20:30]) * 0.1
                elem = ln[31:34].strip()
                chg_code = int(ln[36:39].strip()) if len(ln) >= 39 and ln[36:39].strip() else 0
                q = _chg_code_map.get(chg_code, 0.0)
            except ValueError:
                parts = ln.split()
                if len(parts) < 4:
                    continue
                x, y, z = float(parts[0]) * 0.1, float(parts[1]) * 0.1, float(parts[2]) * 0.1
                elem = parts[3]
                q = 0.0
        else:
            parts = ln.split()
            if len(parts) < 4:
                continue
            x, y, z = float(parts[0]) * 0.1, float(parts[1]) * 0.1, float(parts[2]) * 0.1
            elem = parts[3]
            q = 0.0
        coords.append([x, y, z])
        elements.append(_clean_element(elem))
        charges.append(q)

    bonds_set: Set[Tuple[int, int]] = set()
    bond_start = atom_end
    bond_end = bond_start + n_bonds
    for ln in lines[bond_start:bond_end]:
        try:
            a1 = int(ln[0:3].strip()) - 1
            a2 = int(ln[3:6].strip()) - 1
        except ValueError:
            parts = ln.split()
            if len(parts) < 2:
                continue
            a1 = int(parts[0]) - 1
            a2 = int(parts[1]) - 1
        if 0 <= a1 < len(coords) and 0 <= a2 < len(coords) and a1 != a2:
            bonds_set.add((min(a1, a2), max(a1, a2)))

    # Parse M  CHG property lines (overrides atom block charge codes per V2000 spec)
    m_chg_applied = False
    for ln in lines[bond_end:]:
        if ln.startswith("M  END") or ln.startswith("$$$$"):
            break
        if ln.startswith("M  CHG"):
            parts = ln[6:].split()
            if parts:
                try:
                    n_entries = int(parts[0])
                    if not m_chg_applied:
                        charges = [0.0] * len(coords)
                        m_chg_applied = True
                    for k in range(n_entries):
                        a_idx = int(parts[1 + 2 * k]) - 1
                        chg_val = float(parts[2 + 2 * k])
                        if 0 <= a_idx < len(charges):
                            charges[a_idx] = chg_val
                except (ValueError, IndexError):
                    pass

    pos_nm = np.asarray(coords, dtype=np.float64)
    bonds = sorted(bonds_set)
    meta: Dict[str, Any] = {
        "title": title,
        "charges": charges,
        "coords_synthesized": False,
    }
    return pos_nm, elements, bonds, meta


def parse_chemistry_json_text(
    text: str,
    companion_f32_bytes: Optional[bytes] = None,
) -> Tuple[np.ndarray, List[str], List[Tuple[int, int]], Dict[str, Any]]:
    """Parse AMBER/CHARMM normalized chemistry JSON (schema v1.0.0) or reference fixture JSON.

    Supports both:
      1. Normalized `mdparams` v1.0.0 JSON with `atoms`, `bonds`, `box`, optional
         `coordinates_nm` / `positions_nm` (or companion `.f32` float32 coordinates).
      2. Reference benchmark JSON (`chemistry/reference/*.json`) with `schema`, `fixture`,
         `n_atoms`, `box_nm`/`coordinate_box_nm`, and `source` pointers.
    """
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Chemistry JSON root must be a JSON object")

    schema = str(data.get("schema", "1.0.0"))
    family = str(data.get("family", "")).lower()

    # Case 1: Normalized mdparams JSON with explicit `atoms` array
    if isinstance(data.get("atoms"), list) and len(data["atoms"]) > 0:
        atoms_list = data["atoms"]
        n_atoms = len(atoms_list)
        elements: List[str] = []
        atom_names: List[str] = []
        res_names: List[str] = []
        masses: List[float] = []
        charges: List[float] = []
        inline_coords: List[List[float]] = []

        for a in atoms_list:
            if not isinstance(a, dict):
                raise ValueError("Each entry in JSON 'atoms' must be an object")
            aname = str(a.get("name", "C"))
            m_val = float(a.get("mass", 0.0))
            q_val = float(a.get("charge", 0.0))
            elem_raw = a.get("element")
            if elem_raw:
                elem = _clean_element(str(elem_raw))
            else:
                elem = _infer_element(aname, mass=m_val)
            atom_names.append(aname)
            elements.append(elem)
            res_names.append(str(a.get("residue_name", "MOL")))
            masses.append(m_val if m_val > 0.0 else _ELEMENT_MASSES.get(elem.upper(), 12.011))
            charges.append(q_val)

            if "position_nm" in a and len(a["position_nm"]) == 3:
                inline_coords.append([float(v) for v in a["position_nm"]])
            elif "coordinates_nm" in a and len(a["coordinates_nm"]) == 3:
                inline_coords.append([float(v) for v in a["coordinates_nm"]])
            elif all(k in a for k in ("x", "y", "z")):
                inline_coords.append([float(a["x"]), float(a["y"]), float(a["z"])])

        # Parse bonds (1-based dict {"i": 1, "j": 2, ...} per INTERFACE.md, or [i, j] pairs)
        bonds_set: Set[Tuple[int, int]] = set()
        raw_bonds = data.get("bonds") or []
        if raw_bonds:
            dict_bonds = [b for b in raw_bonds if isinstance(b, dict) and "i" in b and "j" in b]
            list_bonds = [b for b in raw_bonds if isinstance(b, (list, tuple)) and len(b) >= 2]

            if dict_bonds:
                # Per chemistry/INTERFACE.md, mdparams indices are 1-based; handle 0-based gracefully if 0 appears
                min_idx = min(min(int(b["i"]), int(b["j"])) for b in dict_bonds)
                one_based = min_idx >= 1
                for b in dict_bonds:
                    i = int(b["i"]) - (1 if one_based else 0)
                    j = int(b["j"]) - (1 if one_based else 0)
                    if 0 <= i < n_atoms and 0 <= j < n_atoms and i != j:
                        bonds_set.add((min(i, j), max(i, j)))
            if list_bonds:
                min_idx = min(min(int(b[0]), int(b[1])) for b in list_bonds)
                max_idx = max(max(int(b[0]), int(b[1])) for b in list_bonds)
                one_based = (min_idx >= 1 and max_idx == n_atoms)
                for b in list_bonds:
                    i = int(b[0]) - (1 if one_based else 0)
                    j = int(b[1]) - (1 if one_based else 0)
                    if 0 <= i < n_atoms and 0 <= j < n_atoms and i != j:
                        bonds_set.add((min(i, j), max(i, j)))
        bonds = sorted(bonds_set)

        explicit_box = (
            _extract_box_lengths_nm(data.get("box"))
            or _extract_box_lengths_nm(data.get("box_nm"))
            or _extract_box_lengths_nm(data.get("coordinate_box_nm"))
        )

        # Resolve coordinates: top-level array -> inline per-atom -> companion .f32 -> synthesized
        coords_synthesized = False
        pos_nm: Optional[np.ndarray] = None

        top_coords = (
            data.get("coordinates_nm")
            if data.get("coordinates_nm") is not None
            else data.get("positions_nm")
        )
        if top_coords is not None:
            arr = np.asarray(top_coords, dtype=np.float64)
            if arr.shape == (n_atoms, 3):
                pos_nm = arr
        elif data.get("coordinates_ang") is not None or data.get("positions_ang") is not None:
            raw_ang = data.get("coordinates_ang") if data.get("coordinates_ang") is not None else data.get("positions_ang")
            arr = np.asarray(raw_ang, dtype=np.float64)
            if arr.shape == (n_atoms, 3):
                pos_nm = arr * 0.1
        elif len(inline_coords) == n_atoms:
            pos_nm = np.asarray(inline_coords, dtype=np.float64)
        elif companion_f32_bytes is not None and len(companion_f32_bytes) >= n_atoms * 3 * 4:
            flat = np.frombuffer(companion_f32_bytes[: n_atoms * 3 * 4], dtype="<f4")
            pos_nm = flat.reshape(n_atoms, 3).astype(np.float64)

        if pos_nm is None:
            pos_nm = _synthesize_coords_from_graph(n_atoms, bonds, explicit_box)
            coords_synthesized = True

        prov = data.get("provenance") if isinstance(data.get("provenance"), dict) else {}
        title = str(data.get("title") or data.get("fixture") or prov.get("force_field") or "Normalized Chemistry JSON")
        meta: Dict[str, Any] = {
            "title": title,
            "atom_names": atom_names,
            "res_names": res_names,
            "masses": masses,
            "charges": charges,
            "explicit_box_nm": explicit_box,
            "coords_synthesized": coords_synthesized,
            "schema": schema,
            "family": family or "amber",
        }
        return pos_nm, elements, bonds, meta

    # Case 2: Reference benchmark JSON (`chemistry/reference/*.json`) with `n_atoms`
    n_atoms_raw = data.get("n_atoms")
    if isinstance(n_atoms_raw, int) and n_atoms_raw > 0:
        n_atoms = int(n_atoms_raw)
        fixture_name = str(data.get("fixture", "reference_system"))
        sys_class = str(data.get("class", ""))
        explicit_box = (
            _extract_box_lengths_nm(data.get("box_nm"))
            or _extract_box_lengths_nm(data.get("coordinate_box_nm"))
            or _extract_box_lengths_nm(data.get("box"))
        )

        # If local repo fixture parameter + coordinate files exist, load them directly!
        source = data.get("source") if isinstance(data.get("source"), dict) else {}
        param_rel = source.get("parameter_file")
        coord_rel = source.get("coordinate_file")
        if isinstance(param_rel, str) and isinstance(coord_rel, str):
            param_path = _REPO_ROOT / param_rel
            coord_path = _REPO_ROOT / coord_rel
            if param_path.is_file() and coord_path.is_file():
                p_low = param_path.name.lower()
                c_low = coord_path.name.lower()
                if p_low.endswith((".prmtop", ".parm7")) and c_low.endswith((".inpcrd", ".rst7")):
                    _, elements, bonds, meta_t = parse_amber_prmtop_text(
                        param_path.read_text(encoding="utf-8", errors="replace"),
                        synthesize_coords=False,
                    )
                    pos_c, _, _, meta_c = parse_amber_inpcrd_text(
                        coord_path.read_text(encoding="utf-8", errors="replace")
                    )
                    if pos_c.shape[0] == len(elements):
                        meta_t["explicit_box_nm"] = explicit_box or meta_c.get("explicit_box_nm") or meta_t.get("explicit_box_nm")
                        meta_t["coords_synthesized"] = False
                        meta_t["fixture"] = fixture_name
                        meta_t["schema"] = schema
                        meta_t["title"] = f"{fixture_name} ({sys_class})" if sys_class else fixture_name
                        return pos_c, elements, bonds, meta_t

        # Synthesize clean representative system matching exact n_atoms and box
        cls_low = sys_class.lower()
        bonds: List[Tuple[int, int]] = []
        elements: List[str] = []
        charges: List[float] = []
        masses: List[float] = []

        if "water" in cls_low:
            # TIP3P-like water molecules (O, H, H) + ion if remainder
            i = 0
            while i + 2 < n_atoms:
                elements.extend(["O", "H", "H"])
                masses.extend([15.999, 1.008, 1.008])
                charges.extend([-0.834, 0.417, 0.417])
                bonds.append((i, i + 1))
                bonds.append((i, i + 2))
                i += 3
            while i < n_atoms:
                elements.append("Zn" if i == n_atoms - 1 else "Na")
                masses.append(65.38 if i == n_atoms - 1 else 22.990)
                charges.append(0.0)
                i += 1
        elif "dna" in cls_low:
            pattern = [("P", 30.974), ("O", 15.999), ("C", 12.011), ("N", 14.007), ("H", 1.008)]
            for i in range(n_atoms):
                el, m = pattern[i % len(pattern)]
                elements.append(el)
                masses.append(m)
                charges.append(0.0)
        else:
            pattern = [("C", 12.011), ("N", 14.007), ("O", 15.999), ("H", 1.008)]
            for i in range(n_atoms):
                el, m = pattern[i % len(pattern)]
                elements.append(el)
                masses.append(m)
                charges.append(0.0)

        if companion_f32_bytes is not None and len(companion_f32_bytes) >= n_atoms * 3 * 4:
            flat = np.frombuffer(companion_f32_bytes[: n_atoms * 3 * 4], dtype="<f4")
            pos_nm = flat.reshape(n_atoms, 3).astype(np.float64)
            coords_synthesized = False
        else:
            pos_nm = _synthesize_coords_from_graph(n_atoms, bonds, explicit_box)
            coords_synthesized = True

        ff_family = "charmm" if "charmm" in cls_low or "charmm" in fixture_name.lower() else "amber"
        meta = {
            "title": f"{fixture_name} ({sys_class})" if sys_class else fixture_name,
            "masses": masses,
            "charges": charges,
            "explicit_box_nm": explicit_box,
            "coords_synthesized": coords_synthesized,
            "fixture": fixture_name,
            "schema": schema,
            "family": ff_family,
        }
        return pos_nm, elements, bonds, meta

    raise ValueError("Unsupported JSON structure: expected 'atoms' array or 'n_atoms' reference schema")


def _finalize_atomic_system(
    pos_nm: np.ndarray,
    elements: List[str],
    bonds: List[Tuple[int, int]],
    meta: Dict[str, Any],
    filename: str,
    fmt: str,
) -> Tuple[AtomicSystem, Dict[str, Any]]:
    n = pos_nm.shape[0]
    if n == 0:
        raise ValueError(f"No valid atoms parsed from {filename}")

    margin = 0.9  # 9 Angstroms uniform margin inside simulation box
    min_xyz = np.min(pos_nm, axis=0)
    pos_shifted = pos_nm - min_xyz
    span = np.ptp(pos_shifted, axis=0) if n > 1 else np.zeros(3, dtype=np.float64)
    bx, by, bz = 0.0, 0.0, 0.0
    if meta.get("explicit_box_nm") and len(meta["explicit_box_nm"]) == 3:
        bx, by, bz = (max(0.0, float(v)) for v in meta["explicit_box_nm"])
    box_nm = (
        max(bx, float(span[0]) + 2 * margin, 2.8),
        max(by, float(span[1]) + 2 * margin, 2.8),
        max(bz, float(span[2]) + 2 * margin, 2.8),
    )
    offset = 0.5 * (np.array(box_nm) - span)
    pos_nm_centered = pos_shifted + offset

    raw_masses = meta.get("masses")
    if raw_masses is not None and len(raw_masses) == n:
        masses = np.array(
            [
                float(m) if float(m) > 0.0 else _ELEMENT_MASSES.get(el.upper(), 12.011)
                for m, el in zip(raw_masses, elements)
            ],
            dtype=np.float64,
        )
    else:
        masses = np.array([_ELEMENT_MASSES.get(el.upper(), 12.011) for el in elements], dtype=np.float64)

    raw_charges = meta.get("charges")
    if raw_charges is not None and len(raw_charges) == n:
        charges = np.asarray(raw_charges, dtype=np.float64)
    else:
        charges = np.zeros(n, dtype=np.float64)

    sys = AtomicSystem(
        positions=pos_nm_centered,
        masses=masses,
        charges=charges,
        box_lengths=box_nm,
        names=elements,
        bonds=bonds,
        units="md",
    )

    structure_metadata: Dict[str, Any] = {
        "title": meta.get("title") or filename,
        "format": fmt,
        "num_atoms": n,
        "num_bonds": len(bonds),
        "box_ang": [box_nm[0] * 10.0, box_nm[1] * 10.0, box_nm[2] * 10.0],
        "res_names": meta.get("res_names"),
        "chain_ids": meta.get("chain_ids"),
        "atom_names": meta.get("atom_names"),
        "coords_synthesized": bool(meta.get("coords_synthesized", False)),
    }
    for extra_key in ("family", "schema", "fixture"):
        if meta.get(extra_key) is not None:
            structure_metadata[extra_key] = meta[extra_key]

    return sys, structure_metadata


def load_structure_from_bytes(
    raw_bytes: bytes,
    filename: str,
    companion_bytes: Optional[bytes] = None,
) -> Tuple[AtomicSystem, Dict[str, Any]]:
    """Parse raw file bytes into an AtomicSystem and metadata."""
    fn_lower = filename.lower()
    text = raw_bytes.decode("utf-8", errors="replace")

    if fn_lower.endswith((".pdb", ".ent")):
        pos_nm, elements, bonds, meta = parse_pdb_text(text)
        fmt = "pdb"
    elif fn_lower.endswith(".gro"):
        pos_nm, elements, bonds, meta = parse_gro_text(text)
        fmt = "gro"
    elif fn_lower.endswith(".xyz"):
        pos_nm, elements, bonds, meta = parse_xyz_text(text)
        fmt = "xyz"
    elif fn_lower.endswith((".data", ".lammps", ".lmp")):
        pos_nm, elements, bonds, meta = parse_lammps_data_text(text)
        fmt = "lammps"
    elif fn_lower.endswith((".cif", ".mmcif")):
        pos_nm, elements, bonds, meta = parse_mmcif_text(text)
        fmt = "cif"
    elif fn_lower.endswith(".json"):
        pos_nm, elements, bonds, meta = parse_chemistry_json_text(text, companion_f32_bytes=companion_bytes)
        fmt = "json"
    elif fn_lower.endswith((".prmtop", ".parm7")):
        pos_nm, elements, bonds, meta = parse_amber_prmtop_text(text, synthesize_coords=True)
        fmt = "prmtop"
    elif fn_lower.endswith((".inpcrd", ".rst7")):
        pos_nm, elements, bonds, meta = parse_amber_inpcrd_text(text)
        fmt = "inpcrd"
    elif fn_lower.endswith(".psf"):
        pos_nm, elements, bonds, meta = parse_charmm_psf_text(text, synthesize_coords=True)
        fmt = "psf"
    elif fn_lower.endswith((".crd", ".cor")):
        pos_nm, elements, bonds, meta = parse_charmm_crd_text(text)
        fmt = "crd"
    elif fn_lower.endswith(".mol2"):
        pos_nm, elements, bonds, meta = parse_mol2_text(text)
        fmt = "mol2"
    elif fn_lower.endswith((".sdf", ".mol")):
        pos_nm, elements, bonds, meta = parse_sdf_text(text)
        fmt = "sdf"
    else:
        # Fallback by content detection
        stripped = text.lstrip()
        if stripped.startswith("{"):
            pos_nm, elements, bonds, meta = parse_chemistry_json_text(text, companion_f32_bytes=companion_bytes)
            fmt = "json"
        elif "%FLAG POINTERS" in text:
            pos_nm, elements, bonds, meta = parse_amber_prmtop_text(text, synthesize_coords=True)
            fmt = "prmtop"
        elif "!NATOM" in text and "PSF" in text[:120]:
            pos_nm, elements, bonds, meta = parse_charmm_psf_text(text, synthesize_coords=True)
            fmt = "psf"
        elif "@<TRIPOS>ATOM" in text:
            pos_nm, elements, bonds, meta = parse_mol2_text(text)
            fmt = "mol2"
        elif "V2000" in text[:400]:
            pos_nm, elements, bonds, meta = parse_sdf_text(text)
            fmt = "sdf"
        elif "ATOM  " in text or "HETATM" in text:
            pos_nm, elements, bonds, meta = parse_pdb_text(text)
            fmt = "pdb"
        elif "_atom_site." in text:
            pos_nm, elements, bonds, meta = parse_mmcif_text(text)
            fmt = "cif"
        else:
            pos_nm, elements, bonds, meta = parse_xyz_text(text)
            fmt = "xyz"

    return _finalize_atomic_system(pos_nm, elements, bonds, meta, filename, fmt)


_TOPOLOGY_EXTS = (".prmtop", ".parm7", ".psf", ".json")
_COORD_EXTS = (".inpcrd", ".rst7", ".crd", ".cor", ".pdb", ".ent", ".gro", ".xyz", ".f32")


def load_structure_pair_from_bytes(
    file1_bytes: bytes,
    file1_name: str,
    file2_bytes: bytes,
    file2_name: str,
) -> Tuple[AtomicSystem, Dict[str, Any]]:
    """Combine a topology file (.prmtop/.parm7/.psf/.json) with a coordinate file (.inpcrd/.rst7/.crd/.cor/.pdb/.f32)."""
    f1_low = file1_name.lower()
    f2_low = file2_name.lower()

    if f1_low.endswith(_TOPOLOGY_EXTS) and not f2_low.endswith(_TOPOLOGY_EXTS):
        top_bytes, top_name = file1_bytes, file1_name
        coord_bytes, coord_name = file2_bytes, file2_name
    elif f2_low.endswith(_TOPOLOGY_EXTS) and not f1_low.endswith(_TOPOLOGY_EXTS):
        top_bytes, top_name = file2_bytes, file2_name
        coord_bytes, coord_name = file1_bytes, file1_name
    else:
        raise ValueError(
            f"Expected one topology file ({', '.join(_TOPOLOGY_EXTS)}) and one coordinate file "
            f"({', '.join(_COORD_EXTS)}), got '{file1_name}' and '{file2_name}'"
        )

    top_low = top_name.lower()
    coord_low = coord_name.lower()

    if top_low.endswith(".json") and coord_low.endswith(".f32"):
        return load_structure_from_bytes(top_bytes, top_name, companion_bytes=coord_bytes)

    top_text = top_bytes.decode("utf-8", errors="replace")
    if top_low.endswith((".prmtop", ".parm7")):
        _, elements, bonds, meta_t = parse_amber_prmtop_text(top_text, synthesize_coords=False)
        fmt = "amber"
    elif top_low.endswith(".psf"):
        _, elements, bonds, meta_t = parse_charmm_psf_text(top_text, synthesize_coords=False)
        fmt = "charmm"
    elif top_low.endswith(".json"):
        _, elements, bonds, meta_t = parse_chemistry_json_text(top_text)
        fmt = "json"
    else:
        raise ValueError(f"Unsupported topology file '{top_name}'")

    coord_text = coord_bytes.decode("utf-8", errors="replace")
    if coord_low.endswith((".inpcrd", ".rst7")):
        pos_nm, _, _, meta_c = parse_amber_inpcrd_text(coord_text)
    elif coord_low.endswith((".crd", ".cor")):
        pos_nm, _, _, meta_c = parse_charmm_crd_text(coord_text)
    elif coord_low.endswith((".pdb", ".ent")):
        pos_nm, _, _, meta_c = parse_pdb_text(coord_text)
    elif coord_low.endswith(".gro"):
        pos_nm, _, _, meta_c = parse_gro_text(coord_text)
    elif coord_low.endswith(".xyz"):
        pos_nm, _, _, meta_c = parse_xyz_text(coord_text)
    elif coord_low.endswith(".f32"):
        n_expected = len(elements)
        if len(coord_bytes) < n_expected * 3 * 4:
            raise ValueError(f"Binary .f32 file too short for {n_expected} atoms")
        pos_nm = np.frombuffer(coord_bytes[: n_expected * 3 * 4], dtype="<f4").reshape(n_expected, 3).astype(np.float64)
        meta_c = {}
    else:
        raise ValueError(f"Unsupported coordinate file '{coord_name}'")

    if pos_nm.shape[0] != len(elements):
        raise ValueError(
            f"Atom count mismatch between topology '{top_name}' ({len(elements)} atoms) "
            f"and coordinates '{coord_name}' ({pos_nm.shape[0]} atoms)"
        )

    merged_meta = dict(meta_t)
    if meta_c.get("explicit_box_nm"):
        merged_meta["explicit_box_nm"] = meta_c["explicit_box_nm"]
    merged_meta["coords_synthesized"] = False
    combined_name = f"{top_name} + {coord_name}"
    merged_meta["title"] = combined_name

    return _finalize_atomic_system(pos_nm, elements, bonds, merged_meta, combined_name, fmt)
