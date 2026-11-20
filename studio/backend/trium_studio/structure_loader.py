"""High-scale structure loader for Biology-MM Studio.

Directly ingests and parses raw structure files (.pdb, .cif/.mmcif, .gro, .xyz, .data)
on the server without browser-side JSON serialization bottlenecks, scaling to
hundreds of thousands of atoms smoothly.
"""

from __future__ import annotations

import io
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from biology_mm import AtomicSystem

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
    "FE": 55.845,
    "CU": 63.546,
    "ZN": 65.38,
    "BR": 79.904,
    "I": 126.90,
}


def _clean_element(raw: str) -> str:
    cleaned = "".join(ch for ch in str(raw).strip() if ch.isalpha())
    if not cleaned:
        return "C"
    return cleaned[0].upper() + cleaned[1:].lower()


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

            elem = line[76:78].strip() if len(line) >= 78 else ""
            if not elem:
                elem = "".join(c for c in atom_name if c.isalpha())[:1]

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
        elem_raw = "".join(c for c in atom_name if c.isalpha())[:1]
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
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if len(lines) < 3:
        raise ValueError("XYZ file too short")

    try:
        n_atoms = int(lines[0])
    except ValueError as e:
        raise ValueError(f"First line of XYZ must be integer atom count: {lines[0]}") from e

    comment = lines[1]
    coords: List[List[float]] = []
    elements: List[str] = []

    for line in lines[2 : 2 + n_atoms]:
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
        if not stripped or stripped.startswith("#"):
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

            parts = stripped.split()
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


def load_structure_from_bytes(
    raw_bytes: bytes,
    filename: str,
) -> Tuple[AtomicSystem, Dict[str, Any]]:
    """Parse raw file bytes into an AtomicSystem and metadata."""
    fn_lower = filename.lower()
    text = raw_bytes.decode("utf-8", errors="replace")

    if fn_lower.endswith(".pdb") or fn_lower.endswith(".ent"):
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
    else:
        # Fallback by content detection
        if "ATOM  " in text or "HETATM" in text:
            pos_nm, elements, bonds, meta = parse_pdb_text(text)
            fmt = "pdb"
        elif "_atom_site." in text:
            pos_nm, elements, bonds, meta = parse_mmcif_text(text)
            fmt = "cif"
        else:
            pos_nm, elements, bonds, meta = parse_xyz_text(text)
            fmt = "xyz"

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
    pos_nm = pos_shifted + offset

    masses = np.array([_ELEMENT_MASSES.get(el.upper(), 12.011) for el in elements], dtype=np.float64)
    charges = np.asarray(meta.get("charges", np.zeros(n, dtype=np.float64)), dtype=np.float64)

    sys = AtomicSystem(
        positions=pos_nm,
        masses=masses,
        charges=charges,
        box_lengths=box_nm,
        names=elements,
        bonds=bonds,
        units="md",
    )

    structure_metadata = {
        "title": filename,
        "format": fmt,
        "num_atoms": n,
        "num_bonds": len(bonds),
        "box_ang": [box_nm[0] * 10.0, box_nm[1] * 10.0, box_nm[2] * 10.0],
        "res_names": meta.get("res_names"),
        "chain_ids": meta.get("chain_ids"),
    }
    return sys, structure_metadata
