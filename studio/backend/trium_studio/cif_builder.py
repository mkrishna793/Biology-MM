"""mmCIF builder for Biology-MM-Studio simulation sessions."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_Z_TO_SYMBOL: Dict[int, str] = {
    1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 10: "Ne",
    11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P", 16: "S", 17: "Cl", 18: "Ar", 19: "K", 20: "Ca",
    21: "Sc", 22: "Ti", 23: "V", 24: "Cr", 25: "Mn", 26: "Fe", 27: "Co", 28: "Ni", 29: "Cu", 30: "Zn",
    31: "Ga", 32: "Ge", 33: "As", 34: "Se", 35: "Br", 36: "Kr", 37: "Rb", 38: "Sr", 39: "Y", 40: "Zr",
    41: "Nb", 42: "Mo", 44: "Ru", 45: "Rh", 46: "Pd", 47: "Ag", 48: "Cd", 49: "In", 50: "Sn", 51: "Sb",
    52: "Te", 53: "I", 54: "Xe", 55: "Cs", 56: "Ba", 57: "La", 58: "Ce", 60: "Nd", 63: "Eu", 64: "Gd",
    72: "Hf", 73: "Ta", 74: "W", 75: "Re", 76: "Os", 77: "Ir", 78: "Pt", 79: "Au", 80: "Hg", 81: "Tl",
    82: "Pb", 83: "Bi", 90: "Th", 92: "U",
}

_ION_SYMBOLS = {"NA", "CL", "K", "MG", "CA", "FE", "ZN"}


def _cif_token(val: Any, fallback: str) -> str:
    txt = str(val if val is not None else fallback).strip().replace(" ", "_")
    return txt or fallback


def _safe_element_symbol(name: str, atomic_number: int) -> str:
    """Return a clean mmCIF element symbol for `_atom_site.type_symbol`."""
    sym = _Z_TO_SYMBOL.get(int(atomic_number))
    if sym:
        return sym
    cleaned = "".join(ch for ch in str(name).strip() if ch.isalpha())
    if not cleaned:
        return "C"
    return cleaned[0].upper() + cleaned[1:].lower()


def _cell_parameters_from_box_ang(box_ang: np.ndarray) -> Tuple[float, float, float, float, float, float]:
    """Derive (a, b, c, alpha, beta, gamma) from a 3x3 triclinic box matrix in Ångströms.

    Columns of ``box_ang`` are the lattice vectors **a**, **b**, **c** (matching
    ``positions = frac @ H.T`` in ``engine_core.py``).
    """
    H = np.asarray(box_ang, dtype=np.float64)
    if H.shape != (3, 3):
        raise ValueError(f"Expected 3x3 box_H matrix, got shape {H.shape}")

    a_vec = H[:, 0]
    b_vec = H[:, 1]
    c_vec = H[:, 2]

    len_a = float(np.linalg.norm(a_vec))
    len_b = float(np.linalg.norm(b_vec))
    len_c = float(np.linalg.norm(c_vec))

    def _angle_deg(u: np.ndarray, lu: float, v: np.ndarray, lv: float) -> float:
        if lu <= 1e-12 or lv <= 1e-12:
            return 90.0
        cos_theta = float(np.clip(np.dot(u, v) / (lu * lv), -1.0, 1.0))
        return float(np.degrees(np.arccos(cos_theta)))

    alpha = _angle_deg(b_vec, len_b, c_vec, len_c)
    beta = _angle_deg(a_vec, len_a, c_vec, len_c)
    gamma = _angle_deg(a_vec, len_a, b_vec, len_b)
    return len_a, len_b, len_c, alpha, beta, gamma


def build_mmcif_from_session(session, channel: Optional[str] = None) -> str:
    """Build a valid multi-entity mmCIF string from a ``SimulationSession``.

    Emits:
      - ``data_biology_mm`` header
      - ``_cell.length_a/b/c`` and ``_cell.angle_alpha/beta/gamma`` from ``box_H * coord_to_ang``
      - ``_entity``, ``_entity_poly``, ``_entity_poly_seq``, and ``_struct_asym`` tables so Mol*
        automatically creates layered Polymer (Cartoon), Ligand, Ion, and Water representations
      - ``_atom_site`` records in Ångströms strictly for active slots (``active_mask == True``)
        with remapped 1-based contiguous IDs and ``B_iso_or_equiv`` set to ``channel`` scalars when supplied
      - ``_struct_conn`` covalent bond rows for active atoms, skipping cell-wrapping bonds
    """
    lock = getattr(session, "_lock", None)
    ctx = lock if lock is not None else nullcontext()

    with ctx:
        coord_to_ang = float(session.coord_to_ang)
        box_ang = np.asarray(session.box_H, dtype=np.float64) * coord_to_ang
        len_a, len_b, len_c, alpha, beta, gamma = _cell_parameters_from_box_ang(box_ang)

        active_mask = np.asarray(session.active_mask, dtype=bool)
        active_slots = np.flatnonzero(active_mask)
        coords_ang = np.asarray(session.positions, dtype=np.float64) * coord_to_ang
        raw_z = np.asarray(session.atomic_numbers, dtype=np.int64)
        stream_idx = getattr(session, "_stream_atom_indices", None)
        if stream_idx is not None and raw_z.shape[0] != active_mask.shape[0]:
            atomic_numbers = raw_z[np.asarray(stream_idx, dtype=np.int64)]
        else:
            atomic_numbers = raw_z
        names = list(session.names)
        bonds = list(getattr(session, "bonds", []))
        meta = getattr(session, "structure_metadata", None)

        scalars: Optional[np.ndarray] = None
        if channel:
            try:
                scalars = session.compute_scalar_channel(channel)
            except Exception:
                scalars = None

    lines: List[str] = [
        "data_biology_mm",
        "#",
        f"_cell.length_a           {len_a:.6f}",
        f"_cell.length_b           {len_b:.6f}",
        f"_cell.length_c           {len_c:.6f}",
        f"_cell.angle_alpha        {alpha:.6f}",
        f"_cell.angle_beta         {beta:.6f}",
        f"_cell.angle_gamma        {gamma:.6f}",
        "_symmetry.space_group_name_H-M   'P 1'",
        "#",
    ]

    # Resolve per-active-slot mmCIF annotation fields
    meta_group = meta.get("group_pdb") if isinstance(meta, dict) else None
    meta_atom = meta.get("atom_id") if isinstance(meta, dict) else None
    meta_comp = meta.get("comp_id") if isinstance(meta, dict) else None
    meta_asym = meta.get("asym_id") if isinstance(meta, dict) else None
    meta_ent = meta.get("entity_id") if isinstance(meta, dict) else None
    meta_seq = meta.get("seq_id") if isinstance(meta, dict) else None
    meta_entities = meta.get("entities") if isinstance(meta, dict) else None

    slot_records: List[Tuple[int, str, str, str, str, str, str, int, float]] = []
    # Maps for _entity, _entity_poly_seq, and _struct_asym
    entities_seen: Dict[str, Tuple[str, str]] = {}
    if isinstance(meta_entities, list):
        for item in meta_entities:
            if isinstance(item, dict):
                eid = _cif_token(item.get("id"), "1")
                etype = _cif_token(item.get("type"), "non-polymer")
                edesc = _cif_token(item.get("description"), "Biology-MM_Component")
                entities_seen[eid] = (etype, edesc)

    asyms_seen: Dict[str, str] = {}
    poly_seq_seen: Dict[str, List[Tuple[int, str]]] = {}
    poly_res_seen: Dict[Tuple[str, int], str] = {}

    ion_seq_counter = 0
    for idx, slot in enumerate(active_slots):
        slot_int = int(slot)
        cif_id = idx + 1
        sym = _safe_element_symbol(names[slot_int], int(atomic_numbers[slot_int]))
        b_val = float(scalars[slot_int]) if scalars is not None and slot_int < scalars.shape[0] else 0.0

        if meta_atom and slot_int < len(meta_atom):
            group_pdb = _cif_token(meta_group[slot_int] if meta_group and slot_int < len(meta_group) else "ATOM", "ATOM")
            atom_label = _cif_token(meta_atom[slot_int], f"{sym}{cif_id}")
            comp_id = _cif_token(meta_comp[slot_int] if meta_comp and slot_int < len(meta_comp) else "MOL", "MOL")
            asym_id = _cif_token(meta_asym[slot_int] if meta_asym and slot_int < len(meta_asym) else "A", "A")
            ent_id = _cif_token(meta_ent[slot_int] if meta_ent and slot_int < len(meta_ent) else "1", "1")
            seq_id = int(meta_seq[slot_int]) if meta_seq and slot_int < len(meta_seq) else 1
            if ent_id not in entities_seen:
                etype = "polymer" if group_pdb == "ATOM" else ("water" if comp_id in ("HOH", "WAT") else "non-polymer")
                entities_seen[ent_id] = (etype, comp_id)
        else:
            sym_u = sym.upper()
            if sym_u in _ION_SYMBOLS:
                ion_seq_counter += 1
                group_pdb = "HETATM"
                atom_label = sym_u
                comp_id = sym_u
                asym_id = "I"
                ent_id = "2"
                seq_id = ion_seq_counter
                entities_seen.setdefault("2", ("non-polymer", "Ions"))
            else:
                group_pdb = "HETATM"
                atom_label = f"{sym}{cif_id}"
                comp_id = "MOL"
                asym_id = "A"
                ent_id = "1"
                seq_id = 1
                entities_seen.setdefault("1", ("non-polymer", "Simulation_System"))

        asyms_seen.setdefault(asym_id, ent_id)
        ent_type = entities_seen[ent_id][0]
        if ent_type == "polymer" and (ent_id, seq_id) not in poly_res_seen:
            poly_res_seen[(ent_id, seq_id)] = comp_id
            poly_seq_seen.setdefault(ent_id, []).append((seq_id, comp_id))

        slot_records.append((slot_int, cif_id, group_pdb, sym, atom_label, comp_id, asym_id, ent_id, seq_id, b_val))

    if entities_seen:
        lines.extend([
            "loop_",
            "_entity.id",
            "_entity.type",
            "_entity.pdbx_description",
        ])
        for eid, (etype, edesc) in entities_seen.items():
            lines.append(f"{eid} {etype} '{edesc}'")
        lines.append("#")

    if poly_seq_seen:
        lines.extend([
            "loop_",
            "_entity_poly.entity_id",
            "_entity_poly.type",
        ])
        for eid in poly_seq_seen:
            lines.append(f"{eid} 'polypeptide(L)'")
        lines.append("#")

        lines.extend([
            "loop_",
            "_entity_poly_seq.entity_id",
            "_entity_poly_seq.num",
            "_entity_poly_seq.mon_id",
            "_entity_poly_seq.hetero",
        ])
        for eid, res_list in poly_seq_seen.items():
            for seq_num, mon_id in sorted(res_list, key=lambda x: x[0]):
                lines.append(f"{eid} {seq_num} {mon_id} n")
        lines.append("#")

    if asyms_seen:
        lines.extend([
            "loop_",
            "_struct_asym.id",
            "_struct_asym.entity_id",
        ])
        for aid, eid in asyms_seen.items():
            lines.append(f"{aid} {eid}")
        lines.append("#")

    slot_to_info: Dict[int, Tuple[int, str, str, int, str]] = {}

    if slot_records:
        lines.extend([
            "loop_",
            "_atom_site.group_PDB",
            "_atom_site.id",
            "_atom_site.type_symbol",
            "_atom_site.label_atom_id",
            "_atom_site.label_comp_id",
            "_atom_site.label_asym_id",
            "_atom_site.label_entity_id",
            "_atom_site.label_seq_id",
            "_atom_site.Cartn_x",
            "_atom_site.Cartn_y",
            "_atom_site.Cartn_z",
            "_atom_site.occupancy",
            "_atom_site.B_iso_or_equiv",
            "_atom_site.auth_seq_id",
            "_atom_site.auth_comp_id",
            "_atom_site.auth_asym_id",
            "_atom_site.auth_atom_id",
            "_atom_site.pdbx_PDB_model_num",
        ])

        for slot_int, cif_id, group_pdb, sym, atom_label, comp_id, asym_id, ent_id, seq_id, b_val in slot_records:
            slot_to_info[slot_int] = (cif_id, atom_label, comp_id, seq_id, asym_id)
            x, y, z = coords_ang[slot_int]
            lines.append(
                f"{group_pdb} {cif_id} {sym} {atom_label} {comp_id} {asym_id} {ent_id} {seq_id} "
                f"{x:.6f} {y:.6f} {z:.6f} 1.00 {b_val:.4f} {seq_id} {comp_id} {asym_id} {atom_label} 1"
            )
        lines.append("#")

    # Filter covalent bonds: both endpoints active and length <= 0.45 * min(a, b, c)
    min_box_len = min(len_a, len_b, len_c)
    max_bond_len = min(0.45 * min_box_len, 3.2) if min_box_len > 0.0 else 3.2
    bond_rows: List[Tuple[str, str, int, str, str, str, int, str, float]] = []

    for i_raw, j_raw in bonds:
        i = int(i_raw)
        j = int(j_raw)
        info1 = slot_to_info.get(i)
        info2 = slot_to_info.get(j)
        if info1 is None or info2 is None or info1[0] == info2[0]:
            continue
        dist = float(np.linalg.norm(coords_ang[i] - coords_ang[j]))
        if dist > max_bond_len:
            continue
        _id1, lbl1, comp1, seq1, asym1 = info1
        _id2, lbl2, comp2, seq2, asym2 = info2
        bond_rows.append((asym1, comp1, seq1, lbl1, asym2, comp2, seq2, lbl2, dist))

    if bond_rows:
        lines.extend([
            "loop_",
            "_struct_conn.id",
            "_struct_conn.conn_type_id",
            "_struct_conn.ptnr1_label_asym_id",
            "_struct_conn.ptnr1_label_comp_id",
            "_struct_conn.ptnr1_label_seq_id",
            "_struct_conn.ptnr1_label_atom_id",
            "_struct_conn.ptnr1_auth_asym_id",
            "_struct_conn.ptnr1_auth_comp_id",
            "_struct_conn.ptnr1_auth_seq_id",
            "_struct_conn.ptnr1_symmetry",
            "_struct_conn.ptnr2_label_asym_id",
            "_struct_conn.ptnr2_label_comp_id",
            "_struct_conn.ptnr2_label_seq_id",
            "_struct_conn.ptnr2_label_atom_id",
            "_struct_conn.ptnr2_auth_asym_id",
            "_struct_conn.ptnr2_auth_comp_id",
            "_struct_conn.ptnr2_auth_seq_id",
            "_struct_conn.ptnr2_symmetry",
            "_struct_conn.pdbx_dist_value",
        ])
        for b_idx, (a1, c1, s1, l1, a2, c2, s2, l2, dist) in enumerate(bond_rows, start=1):
            lines.append(
                f"covale{b_idx} covale "
                f"{a1} {c1} {s1} {l1} {a1} {c1} {s1} 1_555 "
                f"{a2} {c2} {s2} {l2} {a2} {c2} {s2} 1_555 "
                f"{dist:.6f}"
            )
        lines.append("#")

    return "\n".join(lines) + "\n"
