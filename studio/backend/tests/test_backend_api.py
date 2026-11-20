"""Comprehensive tests for Biology-MM-Studio backend API, presets, sandbox, and WebSocket streaming."""

from __future__ import annotations

import io
import json
import zipfile

import pytest
from fastapi.testclient import TestClient

from trium_studio.app import app, create_app
from trium_studio.presets import create_preset_session, list_presets
from trium_studio.protocol import (
    MSG_FRAME_FULL,
    MSG_OVERLAY_VECTORS,
    MSG_TOPOLOGY_INIT,
    MSG_VOLUME_GRID3D,
    unpack_frame,
)
from trium_studio.sandbox import execute_user_code


EXPECTED_PRESET_IDS = {
    "biomolecular-complex",
    "cnt-tersoff-tensile",
    "nanopore-gcmc-ion",
    "argon-metal-verlet",
    "graphene-custom-spline",
    "silicon-crystal-phonons",
    "argon-100k-gpu-scale",
}


def test_presets_metadata_and_smoke():
    presets = list_presets()
    assert len(presets) == 7
    ids = {p["id"] for p in presets}
    assert ids == EXPECTED_PRESET_IDS

    p100k = next(p for p in presets if p["id"] == "argon-100k-gpu-scale")
    assert p100k["n_atoms"] >= 100_000

    for preset in presets:
        pid = preset["id"]
        sim = create_preset_session(pid)
        try:
            assert sim.n_real == preset["n_atoms"]
            if pid == "argon-100k-gpu-scale":
                assert sim.n_real == 108_000
            m = sim.step(n_steps=1)
            assert m["step"] == 1
            assert m["n_atoms"] > 0
            pkt = sim.encode_frame_packet(channel_name=preset["default_channel"])
            decoded = unpack_frame(pkt)
            assert decoded["header"].msg_type == MSG_FRAME_FULL
            assert decoded["header"].n_atoms == sim.capacity
        finally:
            sim.close()


@pytest.mark.parametrize(
    "bad_code",
    [
        "import os\nos.listdir('.')",
        "import sys\nprint(sys.version)",
        "import subprocess\nsubprocess.run(['ls'])",
        "import socket",
        "from pathlib import Path\nPath('.').resolve()",
        "f = open('/etc/passwd', 'r')",
        "__import__('os').system('echo hi')",
        "eval('1 + 1')",
        "exec('x = 1')",
        "sub = ().__class__.__bases__[0].__subclasses__()",
    ],
)
def test_sandbox_blocks_unsafe_code(bad_code: str):
    res, sim = execute_user_code(bad_code, timeout_s=2.0)
    assert res["ok"] is False
    assert res.get("error")
    assert sim is None


def test_sandbox_timeout():
    res, sim = execute_user_code("while True:\n    pass\n", timeout_s=0.15)
    assert res["ok"] is False
    assert "timed out" in res["error"].lower()
    assert sim is None


def test_sandbox_runs_preset_template_and_creates_session():
    cnt = next(p for p in list_presets() if p["id"] == "cnt-tersoff-tensile")
    res, sim = execute_user_code(cnt["code_template"], timeout_s=5.0)
    assert res["ok"] is True, res
    assert sim is not None
    try:
        assert "CNT atoms=320" in res["stdout"]
        assert res["metrics"]["n_atoms"] == 320
    finally:
        sim.close()


def test_rest_api_full_lifecycle():
    with TestClient(app) as client:
        r_health = client.get("/api/health")
        assert r_health.status_code == 200
        assert r_health.json()["status"] == "ok"

        r_cap = client.get("/api/capabilities")
        assert r_cap.status_code == 200
        cap = r_cap.json()
        assert cap["presets_count"] == 7
        assert cap["supports_mvsx"] is True
        assert len(cap["channels"]) == 7
        assert "multiscale-zone" in cap["channels"]
        assert len(cap["palettes"]) == 6

        r_presets = client.get("/api/presets")
        assert r_presets.status_code == 200
        assert len(r_presets.json()) == 7

        r_404 = client.post("/api/sessions", json={"preset_id": "nonexistent"})
        assert r_404.status_code == 404

        r_sess = client.post("/api/sessions", json={"preset_id": "cnt-tersoff-tensile"})
        assert r_sess.status_code == 200
        sid = r_sess.json()["session_id"]
        assert r_sess.json()["metrics"]["n_atoms"] == 320

        r_met = client.get(f"/api/sessions/{sid}/metrics")
        assert r_met.status_code == 200
        assert r_met.json()["session_id"] == sid

        r_step = client.post(f"/api/sessions/{sid}/step", json={"n_steps": 4})
        assert r_step.status_code == 200
        assert r_step.json()["step"] == 4

        r_steer = client.post(
            f"/api/sessions/{sid}/steer",
            json={"temperature": 450.0, "active_channel": "atomic-stress", "palette": "stress-inferno"},
        )
        assert r_steer.status_code == 200

        r_scrub = client.post(f"/api/sessions/{sid}/scrub", json={"target_step": 0})
        assert r_scrub.status_code == 200
        assert r_scrub.json()["reversed"] is True
        assert r_scrub.json()["metrics"]["step"] == 0

        r_atom = client.get(f"/api/sessions/{sid}/atom/0")
        assert r_atom.status_code == 200
        atom = r_atom.json()
        assert atom["index"] == 0
        assert atom["element"] == "C"
        assert len(atom["position_ang"]) == 3

        assert client.get(f"/api/sessions/{sid}/atom/99999").status_code == 404

        r_custom = client.post(
            f"/api/sessions/{sid}/custom-physics",
            json={
                "potential_expr": "De * (1.0 - exp(-a * (r - r0)))**2",
                "r_min": 0.08,
                "r_max": 0.30,
                "temperature": 320.0,
                "efield_z": 0.0,
                "gcmc_mu": -2.0,
                "strain_rate": 0.0,
            },
        )
        assert r_custom.status_code == 200
        assert r_custom.json()["ok"] is True
        assert len(r_custom.json()["r_grid"]) == 256

        r_bad_custom = client.post(
            f"/api/sessions/{sid}/custom-physics",
            json={
                "potential_expr": "__import__('os').system('ls')",
                "r_min": 0.08,
                "r_max": 0.30,
            },
        )
        assert r_bad_custom.status_code == 400

        r_cif = client.get(f"/api/sessions/{sid}/structure.cif")
        assert r_cif.status_code == 200
        assert "data_biology_mm" in r_cif.text
        assert "_atom_site.Cartn_x" in r_cif.text

        r_mvsj = client.get(f"/api/sessions/{sid}/export/mvsj")
        assert r_mvsj.status_code == 200
        assert "root" in r_mvsj.json()

        r_mvsx = client.get(f"/api/sessions/{sid}/export/mvsx")
        assert r_mvsx.status_code == 200
        with zipfile.ZipFile(io.BytesIO(r_mvsx.content), "r") as zf:
            names = set(zf.namelist())
            assert {"index.mvsj", "structure.cif", "annotations.json"} <= names
            mvsj_inner = json.loads(zf.read("index.mvsj"))
            assert "root" in mvsj_inner

        r_code = client.post(
            "/api/code/run",
            json={
                "code": (
                    "from biology_mm import NanostructureBuilder\n"
                    "from trium_studio.engine_core import SimulationSession\n"
                    "s = NanostructureBuilder.graphene_sheet(4, 4)\n"
                    "session = SimulationSession(s, mode='custom_spline', rcut=0.25)\n"
                    "print('OK', s.n_atoms)\n"
                )
            },
        )
        assert r_code.status_code == 200
        code_res = r_code.json()
        assert code_res["ok"] is True
        assert code_res["session_id"]
        assert "OK 64" in code_res["stdout"]


def test_websocket_stream_and_commands():
    with TestClient(app) as client:
        r_sess = client.post("/api/sessions", json={"preset_id": "nanopore-gcmc-ion"})
        assert r_sess.status_code == 200
        sid = r_sess.json()["session_id"]

        with client.websocket_connect(f"/ws/stream/{sid}") as ws:
            # Initial handshake: TOPOLOGY_INIT, FRAME_FULL, OVERLAY_VECTORS, metrics JSON, state JSON
            pkt_topo = unpack_frame(ws.receive_bytes())
            assert pkt_topo["header"].msg_type == MSG_TOPOLOGY_INIT
            assert pkt_topo["coords"] is not None

            pkt_frame = unpack_frame(ws.receive_bytes())
            assert pkt_frame["header"].msg_type == MSG_FRAME_FULL

            pkt_ov = unpack_frame(ws.receive_bytes())
            assert pkt_ov["header"].msg_type == MSG_OVERLAY_VECTORS

            msg_metrics = ws.receive_json()
            assert msg_metrics["type"] == "metrics"
            assert msg_metrics["metrics"]["session_id"] == sid

            msg_state = ws.receive_json()
            assert msg_state["type"] == "state"
            assert msg_state["playing"] is False

            # Request 3D volume grid
            ws.send_json({"type": "request_volume", "ml": 15})
            pkt_vol = unpack_frame(ws.receive_bytes())
            assert pkt_vol["header"].msg_type == MSG_VOLUME_GRID3D
            assert pkt_vol["dims"] == (15, 15, 15)

            # Request overlays
            ws.send_json({"type": "request_overlays", "max_vectors": 100})
            pkt_ov2 = unpack_frame(ws.receive_bytes())
            assert pkt_ov2["header"].msg_type == MSG_OVERLAY_VECTORS

            # Inspect atom
            ws.send_json({"type": "inspect_atom", "index": 0})
            atom_msg = ws.receive_json()
            assert atom_msg["type"] == "atom_inspection"
            assert atom_msg["atom"]["index"] == 0

            # Step & receive streamed frame + metrics
            ws.send_json({"type": "step", "n_steps": 2})
            stepped_frame = unpack_frame(ws.receive_bytes())
            assert stepped_frame["header"].msg_type == MSG_FRAME_FULL
            assert stepped_frame["header"].frame_id == 2
            stepped_metrics = None
            for _ in range(5):
                msg = ws.receive()
                if "text" in msg and msg["text"]:
                    data = json.loads(msg["text"])
                    if data.get("type") == "metrics":
                        stepped_metrics = data
                        break
            assert stepped_metrics is not None
            assert stepped_metrics["metrics"]["step"] == 2

            # Scrub back
            ws.send_json({"type": "scrub", "target_step": 0})
            scrub_frame = unpack_frame(ws.receive_bytes())
            assert scrub_frame["header"].frame_id == 0
            scrub_metrics = None
            for _ in range(5):
                msg = ws.receive()
                if "text" in msg and msg["text"]:
                    data = json.loads(msg["text"])
                    if data.get("type") == "metrics":
                        scrub_metrics = data
                        break
            assert scrub_metrics is not None
            assert scrub_metrics["metrics"]["step"] == 0

            # Set channel & palette
            ws.send_json({"type": "set_channel", "channel": "kinetic-temp"})
            got_ch_state = False
            for _ in range(10):
                msg = ws.receive()
                if "text" in msg and msg["text"]:
                    data = json.loads(msg["text"])
                    if data.get("type") == "state" and data.get("active_channel") == "kinetic-temp":
                        got_ch_state = True
                        break
            assert got_ch_state

            ws.send_json({"type": "set_palette", "palette": "thermal-div"})
            got_pal_state = False
            for _ in range(10):
                msg = ws.receive()
                if "text" in msg and msg["text"]:
                    data = json.loads(msg["text"])
                    if data.get("type") == "state" and data.get("palette") == "thermal-div":
                        got_pal_state = True
                        break
            assert got_pal_state

            # Play and pause
            ws.send_json({"type": "play"})
            got_play = False
            for _ in range(10):
                msg = ws.receive()
                if "text" in msg and msg["text"]:
                    data = json.loads(msg["text"])
                    if data.get("type") == "state" and data.get("playing") is True:
                        got_play = True
                        break
            assert got_play

            ws.send_json({"type": "pause"})
            got_pause = False
            for _ in range(30):
                msg = ws.receive()
                if "text" in msg and msg["text"]:
                    data = json.loads(msg["text"])
                    if data.get("type") == "state" and data.get("playing") is False:
                        got_pause = True
                        break
            assert got_pause


def test_spa_mount_does_not_intercept_api(tmp_path):
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<!doctype html><html><body>Studio SPA</body></html>", encoding="utf-8")

    spa_app = create_app(dist_dir=dist)
    with TestClient(spa_app) as client:
        r_root = client.get("/")
        assert r_root.status_code == 200
        assert "Studio SPA" in r_root.text

        r_api = client.get("/api/health")
        assert r_api.status_code == 200
        assert r_api.json()["status"] == "ok"


def test_biomolecular_complex_multilayer_and_from_structure_and_active_code():
    with TestClient(app) as client:
        # 1. Verify biomolecular-complex preset emits polymer + non-polymer + water mmCIF entities
        r_bio = client.post("/api/sessions", json={"preset_id": "biomolecular-complex"})
        assert r_bio.status_code == 200
        sid_bio = r_bio.json()["session_id"]
        r_cif = client.get(f"/api/sessions/{sid_bio}/structure.cif?channel=force-magnitude")
        assert r_cif.status_code == 200
        cif_txt = r_cif.text
        assert "_entity.type" in cif_txt
        assert "polymer" in cif_txt
        assert "non-polymer" in cif_txt
        assert "water" in cif_txt
        assert "_entity_poly_seq.mon_id" in cif_txt
        assert "LIG" in cif_txt
        assert "HOH" in cif_txt

        # Verify harmonic bonds + nonbonded step cleanly in metal_verlet
        r_step = client.post(f"/api/sessions/{sid_bio}/step", json={"n_steps": 3})
        assert r_step.status_code == 200
        assert r_step.json()["step"] == 3

        # 2. Import arbitrary Mol* structure coordinates via /api/sessions/from-structure
        r_imp = client.post(
            "/api/sessions/from-structure",
            json={
                "structure": {
                    "title": "Custom PDB Fragment",
                    "coords_ang": [
                        [0.0, 0.0, 0.0],
                        [1.46, 0.0, 0.0],
                        [2.05, 1.40, 0.0],
                        [3.28, 1.40, 0.0],
                        [5.00, 5.00, 5.00],
                    ],
                    "elements": ["N", "C", "C", "O", "Na"],
                    "charges": [-0.3, 0.1, 0.4, -0.4, 1.0],
                    "bonds": [[0, 1], [1, 2], [2, 3]],
                    "structure_metadata": {
                        "group_pdb": ["ATOM", "ATOM", "ATOM", "ATOM", "HETATM"],
                        "atom_id": ["N", "CA", "C", "O", "NA"],
                        "comp_id": ["ALA", "ALA", "ALA", "ALA", "NA"],
                        "asym_id": ["A", "A", "A", "A", "B"],
                        "entity_id": ["1", "1", "1", "1", "2"],
                        "seq_id": [1, 1, 1, 1, 1],
                        "entities": [
                            {"id": "1", "type": "polymer", "description": "Peptide"},
                            {"id": "2", "type": "non-polymer", "description": "Sodium Ion"},
                        ],
                    },
                },
                "temperature": 300.0,
            },
        )
        assert r_imp.status_code == 200
        imp_data = r_imp.json()
        assert imp_data["metrics"]["n_atoms"] == 5
        r_imp_cif = client.get(f"/api/sessions/{imp_data['session_id']}/structure.cif")
        assert r_imp_cif.status_code == 200
        assert "_entity_poly_seq.mon_id" in r_imp_cif.text
        assert "NA" in r_imp_cif.text

        # 3. Run Code Lab script referencing injected `active_structure`
        r_code = client.post(
            "/api/code/run",
            json={
                "code": (
                    "from trium_studio.engine_core import SimulationSession\n"
                    "session = SimulationSession(\n"
                    "    active_structure,\n"
                    "    mode='metal_verlet',\n"
                    "    temperature=310.0,\n"
                    "    structure_metadata=active_structure_metadata,\n"
                    ")\n"
                    "m = session.step(2)\n"
                    "print('ACTIVE_OK', m['n_atoms'], m['step'])\n"
                ),
                "active_structure": {
                    "title": "Code Lab Import",
                    "coords_ang": [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.0, 0.0, 0.0]],
                    "elements": ["C", "C", "O"],
                    "bonds": [[0, 1], [1, 2]],
                },
            },
        )
        assert r_code.status_code == 200
        code_res = r_code.json()
        assert code_res["ok"] is True, code_res
        assert "ACTIVE_OK 3 2" in code_res["stdout"]
        assert code_res["session_id"]


def test_imported_clashing_structure_is_minimized_and_stable():
    import numpy as np
    from fastapi.testclient import TestClient
    from trium_studio.app import create_app

    rng = np.random.default_rng(1)
    n = 60
    coords = (rng.random((n, 3)) * 8.0).tolist()  # random Å positions -> severe steric clashes
    payload = {
        "structure": {
            "title": "clash",
            "coords_ang": coords,
            "elements": ["C"] * n,
            "bonds": [[i, i + 1] for i in range(n - 1)],
            "box_ang": [30.0, 30.0, 30.0],
        },
        "mode": "metal_verlet",
        "temperature": 300.0,
        "dt": 0.001,
    }
    with TestClient(create_app()) as client:
        res = client.post("/api/sessions/from-structure", json=payload)
        assert res.status_code == 200
        body = res.json()
        mini = body["minimization"]
        assert mini["energy_after"] < mini["energy_before"]
        sid = body["session_id"]
        m = client.post(f"/api/sessions/{sid}/step", json={"n_steps": 50}).json()
        assert np.isfinite(m["total_energy"])
        assert abs(m["total_energy"]) < 1e9


def test_biomolecular_preset_is_stable_over_hundreds_of_steps():
    import numpy as np
    from trium_studio.presets import create_preset_session

    s = create_preset_session("biomolecular-complex")
    for _ in range(3):
        m = s.step(100)
        assert np.isfinite(m["total_energy"])
        assert np.isfinite(m["temperature_K"])


def test_direct_file_upload_endpoints(tmp_path):
    import numpy as np
    from fastapi.testclient import TestClient
    from trium_studio.app import create_app

    app = create_app(dist_dir=tmp_path)
    with TestClient(app) as client:
        # PDB upload
        raw_pdb = (
            "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N\n"
            "ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00 20.00           C\n"
            "ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00 20.00           C\n"
            "CONECT    1    2\n"
            "CONECT    2    3\n"
        ).encode("utf-8")
        res = client.post("/api/sessions/upload?filename=alanine.pdb", content=raw_pdb)
        assert res.status_code == 200, res.text
        data = res.json()
        assert data["num_atoms"] == 3
        assert data["num_bonds"] == 2
        sid = data["session_id"]
        # Step the session
        s_res = client.post(f"/api/sessions/{sid}/step", json={"n_steps": 10})
        assert s_res.status_code == 200
        metrics = s_res.json()
        assert np.isfinite(metrics["total_energy"])

        # XYZ upload
        raw_xyz = "2\nWater dimer test\nO 0.0 0.0 0.0\nH 0.95 0.0 0.0\n".encode("utf-8")
        res_xyz = client.post("/api/sessions/upload?filename=sample.xyz", content=raw_xyz)
        assert res_xyz.status_code == 200
        assert res_xyz.json()["num_atoms"] == 2

        # Binary protocol test (TBIN format with 4-byte aligned coords)
        import struct
        n_test = 10
        title_b = b"Binary Test System"
        mode_b = b"metal_verlet"
        hdr = struct.pack(
            "<4sIIffIIfffII",
            b"TBIN",
            n_test,
            len(title_b),
            300.0,
            0.001,
            len(mode_b),
            1,  # flags: has_elements
            20.0, 20.0, 20.0,
            0, 0,
        )
        coords = np.zeros((n_test, 3), dtype=np.float32)
        coords[:, 0] = np.arange(n_test, dtype=np.float32) * 1.5
        elems = b"CCCCCCCCCC"
        bin_payload = hdr + coords.tobytes() + elems + title_b + mode_b
        res_bin = client.post(
            "/api/sessions/create-from-binary",
            content=bin_payload,
            headers={"Content-Type": "application/octet-stream"},
        )
        assert res_bin.status_code == 200, res_bin.text
        bin_data = res_bin.json()
        assert bin_data["session_id"]
        assert bin_data["metrics"]["n_atoms"] == n_test


