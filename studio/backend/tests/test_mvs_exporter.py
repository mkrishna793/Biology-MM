"""Tests for mmCIF builder and MolViewSpec (MVSJ/MVSX) exporter."""

from __future__ import annotations

import io
import json
import re
import zipfile

import molviewspec as mvs
import numpy as np
import pytest

from biology_mm import NanostructureBuilder, SystemBuilder
from trium_studio.cif_builder import build_mmcif_from_session
from trium_studio.engine_core import PHYSICS_CHANNEL_NAMES, SimulationSession
from trium_studio.mvs_exporter import (
    build_annotations_from_session,
    export_mvsj,
    export_mvsx_bytes,
)


def _make_metal_verlet_session(max_atoms_cap: int | None = None) -> SimulationSession:
    system = SystemBuilder.fcc_lattice(n_cells=2, rho=0.8, name="Ar", units="reduced")
    return SimulationSession(system, mode="metal_verlet", max_atoms_cap=max_atoms_cap)


def _make_cnt_tersoff_session() -> SimulationSession:
    system = NanostructureBuilder.carbon_nanotube(n=4, m=4, n_cells=2)
    return SimulationSession(system, mode="cnt_tersoff")


def _parse_cif_kv(cif_text: str, key: str) -> float:
    m = re.search(rf"^{re.escape(key)}\s+([+-]?\d+(?:\.\d+)?)", cif_text, flags=re.MULTILINE)
    assert m is not None, f"Missing mmCIF key {key}"
    return float(m.group(1))


def _extract_atom_site_rows(cif_text: str) -> list[list[str]]:
    lines = cif_text.splitlines()
    rows: list[list[str]] = []
    in_atom_loop = False
    for line in lines:
        stripped = line.strip()
        if stripped == "loop_":
            in_atom_loop = False
            continue
        if stripped.startswith("_atom_site."):
            in_atom_loop = True
            continue
        if in_atom_loop:
            if stripped == "#" or stripped == "" or stripped.startswith("_") or stripped == "loop_":
                in_atom_loop = False
                continue
            rows.append(stripped.split())
    return rows


def _extract_struct_conn_rows(cif_text: str) -> list[list[str]]:
    lines = cif_text.splitlines()
    rows: list[list[str]] = []
    in_conn_loop = False
    for line in lines:
        stripped = line.strip()
        if stripped == "loop_":
            in_conn_loop = False
            continue
        if stripped.startswith("_struct_conn."):
            in_conn_loop = True
            continue
        if in_conn_loop:
            if stripped == "#" or stripped == "" or stripped.startswith("_") or stripped == "loop_":
                in_conn_loop = False
                continue
            rows.append(stripped.split())
    return rows


def test_mmcif_cell_lengths_and_triclinic_angles():
    session = _make_metal_verlet_session()
    # Apply a triclinic shear to box_H: a=(4,0,0), b=(1,4,0), c=(0,0,5) in reduced units
    session.box_H = np.array(
        [
            [4.0, 1.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 0.0, 5.0],
        ],
        dtype=np.float64,
    )
    cif = build_mmcif_from_session(session)
    assert cif.startswith("data_biology_mm\n")

    scale = session.coord_to_ang  # 3.4 for reduced units
    assert _parse_cif_kv(cif, "_cell.length_a") == pytest.approx(4.0 * scale, rel=1e-5)
    assert _parse_cif_kv(cif, "_cell.length_b") == pytest.approx(np.hypot(1.0, 4.0) * scale, rel=1e-5)
    assert _parse_cif_kv(cif, "_cell.length_c") == pytest.approx(5.0 * scale, rel=1e-5)
    assert _parse_cif_kv(cif, "_cell.angle_alpha") == pytest.approx(90.0, abs=1e-4)
    assert _parse_cif_kv(cif, "_cell.angle_beta") == pytest.approx(90.0, abs=1e-4)
    expected_gamma = np.degrees(np.arccos(1.0 / np.hypot(1.0, 4.0)))
    assert _parse_cif_kv(cif, "_cell.angle_gamma") == pytest.approx(expected_gamma, abs=1e-3)


def test_mmcif_active_slot_gcmc_filtering_and_contiguous_ids():
    session = _make_metal_verlet_session(max_atoms_cap=48)
    n_real = session.n_real
    assert session.capacity == 48
    assert n_real == 32

    # Deactivate slots 2 and 5, activate GCMC slot 40
    session.active_mask[2] = False
    session.active_mask[5] = False
    session.active_mask[40] = True
    session.positions[40] = np.array([1.25, 2.5, 0.75], dtype=np.float64)
    session.names[40] = "C"
    session.atomic_numbers[40] = 6

    cif = build_mmcif_from_session(session)
    atom_rows = _extract_atom_site_rows(cif)
    expected_active = int(np.sum(session.active_mask))
    assert len(atom_rows) == expected_active == 31

    # Verify 1-based contiguous _atom_site.id
    ids = [int(row[1]) for row in atom_rows]
    assert ids == list(range(1, expected_active + 1))

    # Last active slot (40) should have its coordinates in Ångströms and element C
    last_row = atom_rows[-1]
    assert last_row[2] == "C"
    assert float(last_row[8]) == pytest.approx(1.25 * session.coord_to_ang, abs=1e-4)
    assert float(last_row[9]) == pytest.approx(2.50 * session.coord_to_ang, abs=1e-4)
    assert float(last_row[10]) == pytest.approx(0.75 * session.coord_to_ang, abs=1e-4)


def test_mmcif_non_wrapping_bonds_in_cnt_tersoff():
    session = _make_cnt_tersoff_session()
    assert len(session.bonds) > 0

    cif = build_mmcif_from_session(session)
    atom_rows = _extract_atom_site_rows(cif)
    conn_rows = _extract_struct_conn_rows(cif)
    assert len(atom_rows) == session.n_real
    assert len(conn_rows) > 0

    # Ensure every emitted bond distance is <= 0.45 * min(box_H) * coord_to_ang
    max_allowed = 0.45 * float(np.min(np.diag(session.box_H))) * session.coord_to_ang
    for row in conn_rows:
        dist = float(row[-1])
        assert dist <= max_allowed + 1e-6


def test_export_mvsj_validates_with_molviewspec():
    for session in (_make_metal_verlet_session(), _make_cnt_tersoff_session()):
        mvsj = export_mvsj(
            session,
            channel="force-magnitude",
            palette="plasma",
            title="Test Scene",
            cif_url="structure.cif",
        )
        assert isinstance(mvsj, dict)
        assert mvsj["metadata"]["title"] == "Test Scene"
        # Must pass molviewspec's official Pydantic schema validator
        mvs.validate_state_tree(json.dumps(mvsj))

        # Check expected v6 node hierarchy
        root_children = mvsj["root"]["children"]
        download_nodes = [n for n in root_children if n["kind"] == "download"]
        assert len(download_nodes) == 1
        assert download_nodes[0]["params"]["url"] == "structure.cif"


@pytest.mark.parametrize("session_factory", [_make_metal_verlet_session, _make_cnt_tersoff_session])
def test_export_mvsx_bytes_zip_contents(session_factory):
    session = session_factory()
    # Advance one step so forces/momenta are populated
    session.step(1)

    data = export_mvsx_bytes(
        session,
        channel="kinetic-temp",
        palette="thermal-div",
        title="Studio MVSX",
    )
    assert isinstance(data, bytes)
    assert len(data) > 0

    with zipfile.ZipFile(io.BytesIO(data), mode="r") as zf:
        names = set(zf.namelist())
        assert {"index.mvsj", "structure.cif", "annotations.json"}.issubset(names)

        mvsj_raw = zf.read("index.mvsj").decode("utf-8")
        cif_raw = zf.read("structure.cif").decode("utf-8")
        ann_raw = zf.read("annotations.json").decode("utf-8")

    assert len(mvsj_raw) > 0
    assert len(cif_raw) > 0
    assert len(ann_raw) > 0

    mvs.validate_state_tree(mvsj_raw)
    assert cif_raw.startswith("data_biology_mm\n")

    annotations = json.loads(ann_raw)
    n_active = int(np.sum(session.active_mask))
    assert len(annotations) == n_active
    for idx, row in enumerate(annotations):
        assert row["atom_id"] == idx + 1
        assert row["atom_index"] == idx
        assert row["channel"] == "kinetic-temp"
        assert isinstance(row["scalar"], float)
        assert re.fullmatch(r"#[0-9a-f]{6}", row["color"]) is not None


def test_all_channels_and_palettes_and_invalid_inputs():
    session = _make_metal_verlet_session()
    palettes = ("plasma", "thermal-div", "electrostatic-bwr", "bond-viridis", "stress-inferno", "ternary-triad")
    for ch, pal in zip(PHYSICS_CHANNEL_NAMES, palettes):
        ann = build_annotations_from_session(session, channel=ch, palette=pal)
        assert len(ann) == session.n_real

    with pytest.raises(ValueError):
        export_mvsj(session, channel="nonexistent-channel")
    with pytest.raises(ValueError):
        export_mvsj(session, palette="nonexistent-palette")
