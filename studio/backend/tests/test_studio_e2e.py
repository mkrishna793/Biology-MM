from __future__ import annotations

import io
import json
import time
import zipfile
from typing import Any, Dict, List

import numpy as np
import pytest
from fastapi.testclient import TestClient

from trium_studio.app import create_app
from trium_studio.protocol import (
    FLAG_REVERSED,
    MAGIC,
    MSG_FRAME_FULL,
    MSG_OVERLAY_VECTORS,
    MSG_TOPOLOGY_INIT,
    MSG_VOLUME_GRID3D,
    decode_int16_fractional,
    encode_int16_fractional,
    pack_frame_full,
    unpack_frame,
)

REQUIRED_PRESETS = (
    "cnt-tersoff-tensile",
    "nanopore-gcmc-ion",
    "argon-metal-verlet",
    "graphene-custom-spline",
)


@pytest.fixture()
def client() -> TestClient:
    app = create_app()
    with TestClient(app) as tc:
        yield tc


def test_rest_workflow_across_presets_and_exports(client: TestClient) -> None:
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    caps = client.get("/api/capabilities")
    assert caps.status_code == 200
    caps_data = caps.json()
    assert caps_data["supports_mvsx"] is True
    assert len(caps_data["channels"]) == 6
    assert len(caps_data["palettes"]) == 6

    presets_resp = client.get("/api/presets")
    assert presets_resp.status_code == 200
    preset_ids = {p["id"] for p in presets_resp.json()}
    for required in REQUIRED_PRESETS:
        assert required in preset_ids

    for pid in REQUIRED_PRESETS:
        created = client.post("/api/sessions", json={"preset_id": pid})
        assert created.status_code == 200
        body = created.json()
        sid = body["session_id"]
        assert body["metrics"]["n_atoms"] > 0

        # Step forward
        stepped = client.post(f"/api/sessions/{sid}/step", json={"n_steps": 5})
        assert stepped.status_code == 200
        stepped_metrics = stepped.json()
        assert stepped_metrics["step"] > body["metrics"]["step"]
        history = stepped_metrics["history_steps"]
        assert len(history) >= 2

        # Steer parameters, channel, palette
        steered = client.post(
            f"/api/sessions/{sid}/steer",
            json={
                "temperature": 350.0,
                "efield_z": 0.25,
                "active_channel": "kinetic-temp",
                "palette": "thermal-div",
            },
        )
        assert steered.status_code == 200

        # Reversible time scrub backward then forward
        target_back = history[0]
        scrub_back = client.post(
            f"/api/sessions/{sid}/scrub",
            json={"target_step": target_back},
        )
        assert scrub_back.status_code == 200
        sb_data = scrub_back.json()
        assert sb_data["reversed"] is True
        assert sb_data["metrics"]["step"] == target_back

        # Atom inspection
        atom = client.get(f"/api/sessions/{sid}/atom/0")
        assert atom.status_code == 200
        atom_data = atom.json()
        assert atom_data["index"] == 0
        assert len(atom_data["position_nm"]) == 3
        assert len(atom_data["force"]) == 3

        # mmCIF export validation
        cif_resp = client.get(f"/api/sessions/{sid}/structure.cif")
        assert cif_resp.status_code == 200
        cif_text = cif_resp.text
        assert "data_biology_mm" in cif_text
        assert "_atom_site.Cartn_x" in cif_text

        # .mvsx ZIP validation
        mvsx_resp = client.get(
            f"/api/sessions/{sid}/export/mvsx",
            params={"channel": "force-magnitude", "palette": "plasma"},
        )
        assert mvsx_resp.status_code == 200
        zf = zipfile.ZipFile(io.BytesIO(mvsx_resp.content))
        names = set(zf.namelist())
        assert "index.mvsj" in names
        assert "structure.cif" in names
        assert "annotations.json" in names
        mvsj = json.loads(zf.read("index.mvsj").decode("utf-8"))
        assert "root" in mvsj

    # Custom spline hot-swap on graphene-custom-spline
    g_resp = client.post("/api/sessions", json={"preset_id": "graphene-custom-spline"})
    assert g_resp.status_code == 200
    g_sid = g_resp.json()["session_id"]

    hot_swap = client.post(
        f"/api/sessions/{g_sid}/custom-physics",
        json={
            "potential_expr": "4 * 0.996 * ((0.34 / r)**12 - (0.34 / r)**6)",
            "r_min": 0.25,
            "r_max": 1.0,
            "n_points": 128,
            "temperature": 320.0,
            "efield_z": 0.0,
            "gcmc_mu": -0.2,
            "strain_rate": 0.01,
        },
    )
    assert hot_swap.status_code == 200
    hs_data = hot_swap.json()
    assert hs_data["ok"] is True
    assert hs_data["metrics"]["mode"] == "custom_spline"
    assert len(hs_data["r_grid"]) == 128
    assert len(hs_data["u_grid"]) == 128
    assert len(hs_data["force_grid"]) == 128


def _collect_ws_messages(
    ws: Any, min_binary: int = 1, min_json: int = 1, max_messages: int = 8
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    binary_frames: List[Dict[str, Any]] = []
    json_msgs: List[Dict[str, Any]] = []
    for _ in range(max_messages):
        msg = ws.receive()
        if msg.get("bytes") is not None:
            raw: bytes = msg["bytes"]
            assert raw[:4] == MAGIC
            binary_frames.append(unpack_frame(raw))
        elif msg.get("text") is not None:
            json_msgs.append(json.loads(msg["text"]))
        if len(binary_frames) >= min_binary and len(json_msgs) >= min_json:
            break
    return binary_frames, json_msgs


def test_websocket_binary_triumstream_and_commands(client: TestClient) -> None:
    created = client.post("/api/sessions", json={"preset_id": "cnt-tersoff-tensile"})
    assert created.status_code == 200
    sid = created.json()["session_id"]

    with client.websocket_connect(f"/ws/stream/{sid}") as ws:
        # Initial burst: TOPOLOGY_INIT, FRAME_FULL, OVERLAY_VECTORS, metrics JSON, state JSON
        binary_frames, json_msgs = _collect_ws_messages(
            ws, min_binary=3, min_json=1, max_messages=8
        )
        msg_types = {f["header"].msg_type for f in binary_frames}
        assert MSG_TOPOLOGY_INIT in msg_types
        assert MSG_FRAME_FULL in msg_types
        assert MSG_OVERLAY_VECTORS in msg_types
        assert any(m.get("type") == "metrics" for m in json_msgs)

        # Request 3D volume grid explicitly
        ws.send_json({"action": "request_volume", "ml": 2})
        vol_frames, _ = _collect_ws_messages(ws, min_binary=1, min_json=0, max_messages=3)
        assert any(f["header"].msg_type == MSG_VOLUME_GRID3D for f in vol_frames)

        # Client commands: set_channel, set_palette, step, scrub, play, pause
        ws.send_json({"action": "set_channel", "channel": "atomic-stress"})
        b2, j2 = _collect_ws_messages(ws, min_binary=1, min_json=2, max_messages=8)
        assert any(f["header"].msg_type == MSG_FRAME_FULL for f in b2)
        assert any(
            m.get("type") == "state" and m.get("active_channel") == "atomic-stress"
            for m in j2
        )

        ws.send_json({"action": "set_palette", "palette": "stress-inferno"})
        found_palette = False
        for _ in range(6):
            _, j3 = _collect_ws_messages(ws, min_binary=0, min_json=1, max_messages=4)
            if any(
                m.get("type") == "state" and m.get("palette") == "stress-inferno"
                for m in j3
            ):
                found_palette = True
                break
        assert found_palette

        ws.send_json({"action": "step", "n_steps": 3})
        b4, j4 = _collect_ws_messages(ws, min_binary=1, min_json=1, max_messages=6)
        assert any(f["header"].msg_type == MSG_FRAME_FULL for f in b4)
        assert any(m.get("type") == "metrics" for m in j4)

        ws.send_json({"action": "scrub", "target_step": 0})
        b5, j5 = _collect_ws_messages(ws, min_binary=1, min_json=1, max_messages=6)
        assert any(
            f["header"].msg_type == MSG_FRAME_FULL and (f["header"].flags & FLAG_REVERSED)
            for f in b5
        )
        assert any(
            m.get("type") == "metrics" and m.get("metrics", {}).get("step") == 0
            for m in j5
        )

        ws.send_json({"action": "play"})
        ws.send_json({"action": "pause"})


def test_sandbox_safety_and_valid_execution(client: TestClient) -> None:
    unsafe_snippets = [
        "import os\nos.system('echo pwned')",
        "f = open('/etc/passwd', 'r')",
        "x = ().__class__.__base__.__subclasses__()",
    ]
    for code in unsafe_snippets:
        resp = client.post("/api/code/run", json={"code": code, "timeout_s": 5.0})
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is False
        assert data.get("error")

    valid_code = """
from biology_mm import NanostructureBuilder
from trium_studio.engine_core import SimulationSession

system = NanostructureBuilder.carbon_nanotube(6, 6, n_cells=4)
session = SimulationSession(system, mode='cnt_tersoff', dt=0.001, temperature=300.0)
session.step(3)
print("CNT_OK", system.n_atoms)
"""
    ok_resp = client.post("/api/code/run", json={"code": valid_code, "timeout_s": 10.0})
    assert ok_resp.status_code == 200
    ok_data = ok_resp.json()
    assert ok_data["ok"] is True
    assert "CNT_OK" in ok_data["stdout"]
    assert ok_data["session_id"]
    assert ok_data["metrics"]["n_atoms"] > 0

    # Verify the created session is live via REST
    m_resp = client.get(f"/api/sessions/{ok_data['session_id']}/metrics")
    assert m_resp.status_code == 200
    assert m_resp.json()["n_atoms"] == ok_data["metrics"]["n_atoms"]


def test_triumstream_1m_atom_int16_benchmark() -> None:
    n_atoms = 1_000_000
    box_h = np.diag([50.0, 50.0, 50.0]).astype(np.float32)
    rng = np.random.default_rng(42)
    coords = rng.uniform(0.0, 50.0, size=(n_atoms, 3)).astype(np.float32)
    scalars = rng.uniform(-1.0, 1.0, size=(n_atoms,)).astype(np.float32)

    t0 = time.perf_counter()
    packet = pack_frame_full(
        coords=coords,
        frame_id=1,
        sim_time_ps=0.5,
        box_H=box_h,
        scalars=scalars,
        use_int16=True,
        scalar_channel=0,
    )
    t_encode = time.perf_counter() - t0

    t1 = time.perf_counter()
    unpacked = unpack_frame(packet)
    t_decode = time.perf_counter() - t1

    # Verify Int16 quantization accuracy: max error per axis <= box_L / 65534 + float32 eps
    decoded_coords = unpacked["coords"]
    assert decoded_coords.shape == (n_atoms, 3)
    max_err = float(np.max(np.abs(decoded_coords - coords)))
    assert max_err < 50.0 / 65534.0 + 1e-4

    # Direct encode/decode throughput check
    q16 = encode_int16_fractional(coords, box_h)
    reconstructed = decode_int16_fractional(q16, box_h)
    assert float(np.max(np.abs(reconstructed - coords))) < 50.0 / 65534.0 + 1e-4

    # Both pack and unpack should complete well within 2.5s for 1M atoms
    assert t_encode < 2.5, f"1M encode too slow: {t_encode:.3f}s"
    assert t_decode < 2.5, f"1M decode too slow: {t_decode:.3f}s"
