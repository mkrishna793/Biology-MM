"""Comprehensive tests for multiscale assembly, soft-core clash relaxation, dirty-tile streaming, and API endpoints."""

from __future__ import annotations

import math
import unittest

import numpy as np
from fastapi.testclient import TestClient

from biology_mm import AtomicSystem
from trium_studio.app import create_app
from trium_studio.engine_core import (
    PHYSICS_CHANNEL_NAMES,
    SimulationSession,
    compute_morton_tiles,
)
from trium_studio.multiscale import (
    TSYM_HEADER_SIZE,
    DirtyTileTracker,
    assign_multiscale_zones,
    expand_symmetry_assembly,
    pack_tsym_binary,
    softcore_relax_clashes,
    unpack_tsym_binary,
)
from trium_studio.protocol import (
    HEADER_SIZE,
    MSG_FRAME_FULL,
    MSG_FRAME_TILE_PATCH,
    pack_dirty_tile_delta,
    pack_frame_tile_patch,
    unpack_frame,
)


def _rotation_z_4x4(theta_rad: float, tx: float = 0.0, ty: float = 0.0, tz: float = 0.0) -> np.ndarray:
    c = math.cos(theta_rad)
    s = math.sin(theta_rad)
    return np.array(
        [
            [c, -s, 0.0, tx],
            [s, c, 0.0, ty],
            [0.0, 0.0, 1.0, tz],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


class TestSymmetryAssemblyAndTSYM(unittest.TestCase):
    def test_tsym_roundtrip_odd_title_and_exact_expansion(self) -> None:
        # Odd atom count (M=5) and odd title length (9 bytes) + odd mode length (13 bytes)
        asu_coords = np.array(
            [
                [1.0, 0.0, 0.0],
                [2.5, 1.0, 0.5],
                [3.2, -0.5, 1.2],
                [0.4, 2.2, -0.8],
                [4.0, 1.5, 2.0],
            ],
            dtype=np.float32,
        )
        asu_elements = ["C", "N", "O", "S", "H"]
        # 4-fold C4 symmetry with z-translations
        transforms = np.stack(
            [
                _rotation_z_4x4(0.0, 0.0, 0.0, 0.0),
                _rotation_z_4x4(0.5 * math.pi, 10.0, 0.0, 1.0),
                _rotation_z_4x4(math.pi, 10.0, 10.0, 2.0),
                _rotation_z_4x4(1.5 * math.pi, 0.0, 10.0, 3.0),
            ],
            axis=0,
        )
        odd_title = "Capsid99X"  # 9 bytes (odd)
        mode_name = "metal_verlet"  # 12 bytes
        packet = pack_tsym_binary(
            asu_coords_ang=asu_coords,
            transforms_4x4=transforms,
            asu_elements=asu_elements,
            title=odd_title,
            mode=mode_name,
            temperature=310.0,
            dt=0.002,
            box_ang=(60.0, 60.0, 60.0),
        )
        self.assertEqual(TSYM_HEADER_SIZE, 48)
        self.assertEqual(TSYM_HEADER_SIZE % 8, 0)

        decoded = unpack_tsym_binary(packet)
        self.assertEqual(decoded.n_asu_atoms, 5)
        self.assertEqual(decoded.n_transforms, 4)
        self.assertEqual(decoded.title, odd_title)
        self.assertEqual(decoded.mode, mode_name)
        self.assertAlmostEqual(decoded.temperature, 310.0, places=4)
        self.assertAlmostEqual(decoded.dt, 0.002, places=6)
        self.assertEqual(decoded.asu_elements, asu_elements)
        np.testing.assert_allclose(decoded.asu_coords_ang, asu_coords, atol=1e-6)
        np.testing.assert_allclose(decoded.transforms_4x4, transforms, atol=1e-6)

        expanded_coords, expanded_elems = expand_symmetry_assembly(
            decoded.asu_coords_ang,
            decoded.asu_elements,
            decoded.transforms_4x4,
        )
        self.assertEqual(expanded_coords.shape, (20, 3))
        self.assertEqual(len(expanded_elems), 20)
        self.assertEqual(expanded_elems, asu_elements * 4)

        # Verify exact 90-degree rotation + translation on copy 1
        for m_idx in range(5):
            x, y, z = asu_coords[m_idx]
            expected = np.array([-y + 10.0, x + 0.0, z + 1.0], dtype=np.float32)
            np.testing.assert_allclose(expanded_coords[5 + m_idx], expected, atol=1e-5)


class TestSoftcoreClashRelaxation(unittest.TestCase):
    def test_relax_clashing_dimer_and_cluster(self) -> None:
        # Deliberate severe clash at r = 0.02 nm (would be > 1e10 kJ/mol in raw LJ)
        pos = np.array(
            [
                [1.50, 1.50, 1.50],
                [1.52, 1.50, 1.50],  # 0.02 nm from atom 0
                [1.50, 1.52, 1.50],  # 0.02 nm from atom 0
                [1.50, 1.50, 1.50],  # exact r = 0.0 overlap with atom 0
            ],
            dtype=np.float64,
        )
        box_nm = (4.0, 4.0, 4.0)
        relaxed, report = softcore_relax_clashes(
            positions_nm=pos,
            box_nm=box_nm,
            sigma_nm=0.34,
            epsilon_kj=0.997,
            max_steps=40,
            max_force_cap=5000.0,
            alpha_sc=0.5,
            lam=0.5,
        )
        self.assertEqual(report["status"], "relaxed")
        self.assertGreater(report["steps_run"], 0)
        self.assertTrue(np.all(np.isfinite(relaxed)))
        self.assertTrue(math.isfinite(report["initial_energy"]))
        self.assertTrue(math.isfinite(report["final_energy"]))
        self.assertTrue(math.isfinite(report["final_max_force"]))
        self.assertLess(report["final_energy"], report["initial_energy"] * 0.25)
        self.assertLess(report["final_max_force"], report["initial_max_force"] * 0.25)
        self.assertGreater(report["clashes_resolved"], 0)

        # Verify pairwise distances increased safely beyond severe core overlap
        for i in range(4):
            for j in range(i + 1, 4):
                d = relaxed[i] - relaxed[j]
                d -= 4.0 * np.rint(d / 4.0)
                self.assertGreater(float(np.linalg.norm(d)), 0.20)

    def test_clean_system_returns_clean_status(self) -> None:
        pos = np.array([[0.5, 0.5, 0.5], [2.5, 2.5, 2.5]], dtype=np.float64)
        relaxed, report = softcore_relax_clashes(pos, (4.0, 4.0, 4.0), 0.34, 0.997)
        self.assertEqual(report["status"], "clean")
        self.assertEqual(report["steps_run"], 0)
        np.testing.assert_allclose(relaxed, pos, atol=1e-9)


class TestMultiscaleZones(unittest.TestCase):
    def test_zone_partitioning_and_scalar_channel(self) -> None:
        self.assertIn("multiscale-zone", PHYSICS_CHANNEL_NAMES)
        coords_ang = np.array(
            [
                [0.0, 0.0, 0.0],   # r = 0 -> Zone 0 (AA)
                [10.0, 0.0, 0.0],  # r = 10 -> Zone 0 (AA)
                [20.0, 0.0, 0.0],  # r = 20 -> Zone 1 (CG)
                [30.0, 0.0, 0.0],  # r = 30 -> Zone 1 (CG)
                [45.0, 0.0, 0.0],  # r = 45 -> Zone 2 (Rigid)
                [60.0, 0.0, 0.0],  # r = 60 -> Zone 2 (Rigid)
            ],
            dtype=np.float64,
        )
        part = assign_multiscale_zones(coords_ang, center_ang=(0.0, 0.0, 0.0), aa_radius_ang=15.0, cg_radius_ang=35.0)
        np.testing.assert_array_equal(part.zone_ids, np.array([0, 0, 1, 1, 2, 2], dtype=np.uint8))
        self.assertEqual(part.n_all_atom, 2)
        self.assertEqual(part.n_coarse_grained, 2)
        self.assertEqual(part.n_rigid_scaffold, 2)
        summary = part.build_multiscale_summary()
        self.assertEqual(summary["n_total"], 6)
        self.assertGreater(summary["estimated_memory_mb"], 0.0)

        # Verify integration inside SimulationSession + scalar channel + rigid scaffold freeze
        pos_nm = coords_ang * 0.1 + 1.0  # shift inside box
        sys = AtomicSystem(
            positions=pos_nm,
            masses=np.full(6, 12.011),
            charges=np.zeros(6),
            box_lengths=(10.0, 10.0, 10.0),
            names=["C"] * 6,
            bonds=[],
            units="md",
        )
        sim = SimulationSession(sys, mode="metal_verlet", dt=0.002, temperature=300.0)
        sim.set_focus_zone(center_ang=[10.0, 10.0, 10.0], aa_radius_ang=15.0, cg_radius_ang=35.0)
        ch = sim.compute_scalar_channel("multiscale-zone")
        np.testing.assert_array_equal(ch, np.array([0.0, 0.0, 1.0, 1.0, 2.0, 2.0], dtype=np.float32))

        rigid_before = sim.positions[4:].copy()
        sim.step(n_steps=3)
        np.testing.assert_allclose(sim.positions[4:], rigid_before, atol=1e-12)


class TestDirtyTileStreaming(unittest.TestCase):
    def test_dirty_tile_tracker_and_aligned_patch_roundtrip(self) -> None:
        n_atoms = 96  # 3 Morton tiles of 32 atoms
        rng = np.random.default_rng(42)
        box_H = np.diag([50.0, 50.0, 50.0]).astype(np.float32)
        coords0 = rng.uniform(5.0, 45.0, size=(n_atoms, 3)).astype(np.float32)
        tile_ids = compute_morton_tiles(coords0, box_H, tile_size=32)

        tracker = DirtyTileTracker(n_atoms=n_atoms, tile_ids=tile_ids, threshold_ang=0.05)
        # First call transmits all atoms
        dirty0, ratio0, n_tiles0 = tracker.compute_dirty(coords0)
        self.assertEqual(dirty0.size, n_atoms)
        self.assertAlmostEqual(ratio0, 1.0)
        self.assertEqual(n_tiles0, 3)

        # Move an ODD number of atoms (5 atoms) above threshold, and nudge others below threshold
        coords1 = coords0.copy()
        moved_indices = np.array([3, 17, 42, 65, 91], dtype=np.uint32)
        coords1[moved_indices, 0] += 0.25  # >= 0.05 Angstrom
        coords1[0, 1] += 0.01  # < 0.05 Angstrom (must NOT be transmitted, must NOT update last_sent)

        dirty1, ratio1, n_tiles1 = tracker.compute_dirty(coords1)
        self.assertEqual(dirty1.size, 5)
        self.assertEqual(set(dirty1.tolist()), set(moved_indices.tolist()))
        self.assertAlmostEqual(ratio1, 5.0 / n_atoms)
        self.assertGreaterEqual(n_tiles1, 1)
        # Untouched atom 0 should retain original last_sent_coords_ang
        np.testing.assert_allclose(tracker.last_sent_coords_ang[0], coords0[0], atol=1e-6)
        # Transmitted atoms should have updated last_sent_coords_ang
        np.testing.assert_allclose(tracker.last_sent_coords_ang[moved_indices], coords1[moved_indices], atol=1e-6)

        # Pack sparse delta patch with use_int16=True for ODD count (k=5) and verify strict 4-byte alignment
        scalars = np.arange(n_atoms, dtype=np.float32) * 0.5
        pkt = pack_dirty_tile_delta(
            dirty_indices=dirty1,
            coords_ang=coords1,
            scalars=scalars,
            box_H=box_H,
            frame_id=7,
            sim_time_ps=0.014,
            n_dirty_tiles=n_tiles1,
            use_int16=True,
            scalar_channel=2,
        )
        self.assertEqual(len(pkt) % 4, 0)
        self.assertEqual((len(pkt) - HEADER_SIZE) % 4, 0)

        decoded = unpack_frame(pkt)
        self.assertEqual(decoded["header"].msg_type, MSG_FRAME_TILE_PATCH)
        self.assertEqual(decoded["header"].n_atoms, 5)
        self.assertEqual(decoded["tile_id"], n_tiles1)
        self.assertEqual(decoded["scalar_channel"], 2)
        np.testing.assert_array_equal(decoded["atom_indices"], dirty1)
        np.testing.assert_allclose(decoded["scalars"], scalars[dirty1], atol=1e-6)

        # Apply patch to receiver buffer and verify untouched atoms remain exact
        client_coords = coords0.copy()
        client_coords[decoded["atom_indices"]] = decoded["coords"]
        np.testing.assert_allclose(client_coords[dirty1], coords1[dirty1], atol=0.02)
        untouched = np.setdiff1d(np.arange(n_atoms), dirty1)
        np.testing.assert_allclose(client_coords[untouched], coords0[untouched], atol=1e-6)

        # Verify SimulationSession.encode_frame_packet delta mode
        sys = AtomicSystem(
            positions=coords0.astype(np.float64) * 0.1,
            masses=np.full(n_atoms, 12.011),
            charges=np.zeros(n_atoms),
            box_lengths=(5.0, 5.0, 5.0),
            names=["C"] * n_atoms,
            bonds=[],
            units="md",
        )
        sim = SimulationSession(sys, mode="metal_verlet", use_int16=True)
        sim.set_delta_threshold(0.05, enabled=True)
        full_pkt = sim.encode_frame_packet()
        self.assertEqual(unpack_frame(full_pkt)["header"].msg_type, MSG_FRAME_FULL)
        # Displace 3 atoms -> next frame should emit a compact MSG_FRAME_TILE_PATCH
        sim.positions[[1, 2, 3], 0] += 0.05  # 0.5 Angstrom
        delta_pkt = sim.encode_frame_packet()
        dec_delta = unpack_frame(delta_pkt)
        self.assertEqual(dec_delta["header"].msg_type, MSG_FRAME_TILE_PATCH)
        self.assertEqual(dec_delta["header"].n_atoms, 3)
        self.assertEqual(len(delta_pkt) % 4, 0)


class TestFastAPIMultiscaleEndpoints(unittest.TestCase):
    def setUp(self) -> None:
        self.app = create_app()
        self.client = TestClient(self.app)

    def tearDown(self) -> None:
        self.app.state.session_manager.close_all()

    def test_symmetry_relax_multiscale_steer_and_ws_e2e(self) -> None:
        # Build a TSYM packet with odd title and clashing symmetry copy
        asu_coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [5.0, 0.0, 0.0],
                [20.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        transforms = np.stack(
            [
                _rotation_z_4x4(0.0, 0.0, 0.0, 0.0),
                _rotation_z_4x4(0.0, 0.2, 0.0, 0.0),  # 0.2 Angstrom = 0.02 nm clash with copy 0
                _rotation_z_4x4(0.0, 45.0, 0.0, 0.0),
            ],
            axis=0,
        )
        tsym_bytes = pack_tsym_binary(
            asu_coords_ang=asu_coords,
            transforms_4x4=transforms,
            asu_elements=["C", "N", "O"],
            title="Viral7X",  # 7 bytes (odd)
            mode="metal_verlet",
            temperature=300.0,
            dt=0.002,
        )

        # 1. POST /api/sessions/create-from-symmetry
        resp = self.client.post(
            "/api/sessions/create-from-symmetry",
            content=tsym_bytes,
            headers={"Content-Type": "application/octet-stream"},
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        data = resp.json()
        sid = data["session_id"]
        self.assertEqual(data["title"], "Viral7X")
        self.assertEqual(data["n_asu_atoms"], 3)
        self.assertEqual(data["n_transforms"], 3)
        self.assertEqual(data["n_atoms"], 9)
        self.assertEqual(data["relaxation"]["status"], "relaxed")
        self.assertIn("health_status", data["metrics"])
        self.assertIn("drift_fraction", data["metrics"])

        # 2. POST /api/sessions/{sid}/relax
        resp_relax = self.client.post(f"/api/sessions/{sid}/relax", json={"max_steps": 20})
        self.assertEqual(resp_relax.status_code, 200, resp_relax.text)
        self.assertIn(resp_relax.json()["relaxation"]["status"], ("relaxed", "clean"))

        # 3. POST /api/sessions/{sid}/multiscale
        resp_ms = self.client.post(
            f"/api/sessions/{sid}/multiscale",
            json={"center_ang": [10.0, 10.0, 10.0], "aa_radius_ang": 12.0, "cg_radius_ang": 28.0},
        )
        self.assertEqual(resp_ms.status_code, 200, resp_ms.text)
        ms = resp_ms.json()["multiscale"]
        self.assertEqual(ms["n_total"], 9)
        self.assertEqual(
            ms["n_all_atom"] + ms["n_coarse_grained"] + ms["n_rigid_scaffold"],
            9,
        )

        # 4. POST /api/sessions/{sid}/steer-atom (attach + clear)
        resp_smd = self.client.post(
            f"/api/sessions/{sid}/steer-atom",
            json={"atom_index": 0, "target_ang": [25.0, 10.0, 10.0], "k_spring": 600.0},
        )
        self.assertEqual(resp_smd.status_code, 200, resp_smd.text)
        smd = resp_smd.json()["smd"]
        self.assertTrue(smd["active"])
        self.assertEqual(smd["atom_index"], 0)
        self.assertGreater(smd["pull_force_kj_mol_nm"], 0.0)

        resp_smd_clear = self.client.post(
            f"/api/sessions/{sid}/steer-atom",
            json={"atom_index": None, "target_ang": None, "k_spring": 0.0},
        )
        self.assertEqual(resp_smd_clear.status_code, 200, resp_smd_clear.text)
        self.assertFalse(resp_smd_clear.json()["smd"]["active"])

        # 5. WebSocket handlers: set_focus_zone, relax_clashes, steer_atom, set_delta_threshold
        import json as _json

        def _recv_json_type(ws_conn, expected_type: str, max_reads: int = 20):
            for _ in range(max_reads):
                raw = ws_conn.receive()
                if "text" in raw and raw["text"] is not None:
                    parsed = _json.loads(raw["text"])
                    if parsed.get("type") == expected_type:
                        return parsed
            raise AssertionError(f"Did not receive WS JSON message of type '{expected_type}'")

        with self.client.websocket_connect(f"/ws/stream/{sid}") as ws:
            # Consume initial handshake: topology, frame, overlay, metrics, state
            _topo = ws.receive_bytes()
            _frame = ws.receive_bytes()
            _overlay = ws.receive_bytes()
            _init_metrics = ws.receive_json()
            _init_state = ws.receive_json()

            ws.send_json(
                {
                    "type": "set_focus_zone",
                    "center_ang": [12.0, 12.0, 12.0],
                    "aa_radius_ang": 10.0,
                    "cg_radius_ang": 25.0,
                }
            )
            msg = _recv_json_type(ws, "multiscale")
            self.assertEqual(msg["multiscale"]["n_total"], 9)

            ws.send_json({"type": "set_delta_threshold", "threshold_ang": 0.08, "enabled": True})
            msg = _recv_json_type(ws, "delta_threshold")
            self.assertAlmostEqual(msg["threshold_ang"], 0.08, places=5)
            self.assertTrue(msg["delta_streaming_enabled"])

            ws.send_json({"type": "relax_clashes", "max_steps": 15})
            msg = _recv_json_type(ws, "relax_result")
            self.assertIn(msg["relaxation"]["status"], ("relaxed", "clean"))

            ws.send_json({"type": "steer_atom", "atom_index": 1, "target_ang": [30.0, 10.0, 10.0], "k_spring": 400.0})
            msg = _recv_json_type(ws, "smd_state")
            self.assertTrue(msg["smd"]["active"])
            self.assertEqual(msg["smd"]["atom_index"], 1)

        # 6. Export endpoint returns 501 gracefully when molviewspec is not installed
        caps = self.client.get("/api/capabilities").json()
        self.assertIn("multiscale-zone", caps["channels"])
        if not caps["supports_mvsx"]:
            resp_mvsj = self.client.get(f"/api/sessions/{sid}/export/mvsj")
            self.assertEqual(resp_mvsj.status_code, 501)


    def test_compact_symmetry_world_space_streaming_and_elements(self) -> None:
        """Verify TSYM -> CompactSymmetryMultiscaleSession preserves protein elements & world-space Float32 coords."""
        from trium_studio.protocol import FLAG_INT16_FRACTIONAL, FLAG_PBC_WRAPPED
        from trium_studio.ternary_compact import CompactSymmetryMultiscaleSession

        asu_coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [3.6, 0.0, 0.0],
                [0.0, 3.6, 0.0],
                [3.6, 3.6, 0.0],
                [0.0, 0.0, 3.6],
                [3.6, 3.6, 3.6],
            ],
            dtype=np.float32,
        )
        asu_elements = ["C", "N", "O", "S", "H", "P"]
        transforms = np.stack(
            [
                _rotation_z_4x4(0.0, 500.0, 200.0, -300.0),
                _rotation_z_4x4(0.5 * math.pi, 560.0, 200.0, -300.0),
                _rotation_z_4x4(math.pi, 700.0, 200.0, -300.0),
            ],
            axis=0,
        )
        pkt = pack_tsym_binary(
            asu_coords_ang=asu_coords,
            transforms_4x4=transforms,
            asu_elements=asu_elements,
            title="ProteinCapsidTest",
            mode="metal_verlet",
            temperature=300.0,
            dt=0.002,
        )
        sym = unpack_tsym_binary(pkt)
        self.assertEqual(sym.asu_elements, asu_elements)

        sim = CompactSymmetryMultiscaleSession.from_tsym_payload(
            sym, aa_radius_ang=25.0, cg_radius_ang=75.0
        )
        try:
            # Verify no Argon anywhere and exact elements in Zone 0 / 'CG' in Zone 1
            self.assertNotIn(18, set(sim.atomic_numbers.tolist()))
            z0_elems = [sim.inspect_atom(i)["element"] for i in range(sim.active_all_atom_count)]
            self.assertEqual(set(z0_elems), set(asu_elements))
            if sim.active_cg_bead_count > 0:
                self.assertEqual(sim.inspect_atom(sim.active_all_atom_count)["element"], "CG")

            # Verify exact world-space coordinates around (500, 200, -300) A (never wrapped to [0, L])
            sim._dirty_tracker.reset()
            dec = unpack_frame(sim.encode_frame_packet())
            self.assertEqual(dec["header"].flags & FLAG_INT16_FRACTIONAL, 0)
            self.assertEqual(dec["header"].flags & FLAG_PBC_WRAPPED, 0)
            ctr_stream = np.mean(dec["coords"][: sim.active_all_atom_count], axis=0)
            self.assertGreater(float(ctr_stream[0]), 490.0)
            self.assertLess(float(ctr_stream[2]), -290.0)

            # Verify 3-zone metrics sum to n_atoms
            m = sim.get_metrics()
            ms = m["multiscale"]
            self.assertEqual(ms["all_atom"] + ms["coarse_grained"] + ms["rigid_scaffold"], 18)
            self.assertGreater(ms["all_atom"], 0)
            self.assertGreater(ms["coarse_grained"], 0)
            self.assertGreater(ms["rigid_scaffold"], 0)
        finally:
            sim.close()

if __name__ == "__main__":
    unittest.main()
