"""Verification suite for Trium Base-243 ternary compression & 40M-atom low-RAM symmetry-compact engine."""

from __future__ import annotations

import math
import time
import unittest

import numpy as np
from fastapi.testclient import TestClient

from biology_mm import AtomicSystem
from trium_studio.app import create_app
from trium_studio.engine_core import MAX_SNAPSHOT_RAM_BYTES, SimulationSession
from trium_studio.multiscale import pack_tsym_binary
from trium_studio.multiscale import unpack_tsym_binary
from trium_studio.protocol import (
    FLAG_INT16_FRACTIONAL,
    FLAG_PBC_WRAPPED,
    MSG_FRAME_FULL,
    MSG_FRAME_TILE_PATCH,
    MSG_TOPOLOGY_INIT,
    unpack_frame,
)
from trium_studio.ternary_compact import (
    ELEMENT_LUT_EPSILON,
    ELEMENT_LUT_MASS,
    ELEMENT_LUT_SIGMA,
    Base243TernaryCodec,
    CompactSymmetryMultiscaleSession,
    encode_elements_uint8,
    eval_bipartite_centroid_residual,
)


def _build_faustovirus_scale_tsym() -> tuple[bytes, np.ndarray, np.ndarray]:
    """Build a realistic 5J7V Faustovirus-scale 39,959,280-atom assembly (14,478 ASU x 2,760 transforms)."""
    n_asu = 14_478
    n_transforms = 2_760
    rng = np.random.default_rng(577)

    # Realistic capsomer ASU spaced on a ~3.6 A 3D protein-like lattice (~88 A x 88 A x 88 A)
    side = int(math.ceil(n_asu ** (1.0 / 3.0)))  # 25
    grid = np.mgrid[0:side, 0:side, 0:side].reshape(3, -1).T[:n_asu].astype(np.float32)
    asu_coords = (grid - (side * 0.5)) * np.float32(3.8)
    asu_coords += rng.normal(0.0, 0.25, size=asu_coords.shape).astype(np.float32)
    # Deliberate steric clash on atom pair (0, 1) to verify relax_clashes
    asu_coords[1] = asu_coords[0] + np.array([0.25, 0.0, 0.0], dtype=np.float32)

    elem_cycle = ["C", "N", "O", "C", "H", "S"]
    asu_elements = [elem_cycle[i % len(elem_cycle)] for i in range(n_asu)]

    # 2,760 capsomer transforms on a 1,100 A radius icosahedral-like spherical capsid shell
    radius = 1100.0
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    idx = np.arange(n_transforms, dtype=np.float64)
    y = 1.0 - (2.0 * (idx + 0.5) / n_transforms)
    r_xy = np.sqrt(np.maximum(0.0, 1.0 - y * y))
    theta = golden_angle * idx
    nx = np.cos(theta) * r_xy
    ny = y
    nz = np.sin(theta) * r_xy

    transforms = np.zeros((n_transforms, 4, 4), dtype=np.float32)
    transforms[:, 3, 3] = 1.0
    # Construct right-handed orthonormal frame aligned with outward normal per capsomer
    z_axis = np.stack([nx, ny, nz], axis=1)
    up = np.where(np.abs(ny)[:, None] < 0.9, np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0]))
    x_axis = np.cross(up, z_axis)
    x_axis /= np.maximum(np.linalg.norm(x_axis, axis=1, keepdims=True), 1e-12)
    y_axis = np.cross(z_axis, x_axis)

    transforms[:, :3, 0] = x_axis.astype(np.float32)
    transforms[:, :3, 1] = y_axis.astype(np.float32)
    transforms[:, :3, 2] = z_axis.astype(np.float32)
    transforms[:, :3, 3] = (z_axis * radius).astype(np.float32)

    packet = pack_tsym_binary(
        asu_coords_ang=asu_coords,
        transforms_4x4=transforms,
        asu_elements=asu_elements,
        title="5J7V_Faustovirus_40M",
        mode="metal_verlet",
        temperature=300.0,
        dt=0.002,
    )
    return packet, asu_coords, transforms


class TestBase243TernaryCodecAndLUT(unittest.TestCase):
    def test_exhaustive_243_pack_unpack_roundtrip(self) -> None:
        # Enumerate all 3^5 = 243 balanced-ternary 5-trit tuples
        all_tuples = []
        for b in range(243):
            v = b
            row = []
            for _ in range(5):
                row.append((v % 3) - 1)
                v //= 3
            all_tuples.append(row)
        trits = np.array(all_tuples, dtype=np.int8)

        packed = Base243TernaryCodec.pack_trits_base243(trits)
        self.assertEqual(packed.dtype, np.uint8)
        self.assertEqual(packed.shape, (243,))
        np.testing.assert_array_equal(packed, np.arange(243, dtype=np.uint8))

        unpacked = Base243TernaryCodec.unpack_trits_base243(packed, n_trits=243 * 5)
        self.assertEqual(unpacked.dtype, np.int8)
        np.testing.assert_array_equal(unpacked.reshape((243, 5)), trits)

    def test_odd_lengths_and_5_trits_per_byte_density(self) -> None:
        rng = np.random.default_rng(1234)
        for n in (1, 4, 5, 7, 13, 1003):
            trits = rng.integers(-1, 2, size=n, dtype=np.int8)
            packed = Base243TernaryCodec.pack_trits_base243(trits)
            self.assertEqual(packed.nbytes, math.ceil(n / 5))
            decoded = Base243TernaryCodec.unpack_trits_base243(packed, n_trits=n)
            np.testing.assert_array_equal(decoded, trits)

        # Verify 1.6 bits/trit = 94.9% of log2(3) Shannon density
        bits_per_trit = 8.0 / 5.0
        shannon_eff = math.log2(3.0) / bits_per_trit
        self.assertGreater(shannon_eff, 0.99 * 0.949)

    def test_quantize_dequantize_coords_ternary(self) -> None:
        rng = np.random.default_rng(99)
        coords = rng.uniform(-50.0, 50.0, size=(4096, 3)).astype(np.float64)
        enc = Base243TernaryCodec.quantize_coords_ternary(coords, n_trits_per_val=10)
        # 10 trits/scalar -> 2 bytes/scalar -> 6 bytes/atom vs 24 bytes float64 = 4.0x compression
        self.assertGreaterEqual(enc["compression_ratio"], 3.99)
        recon = Base243TernaryCodec.dequantize_coords_ternary(enc, dtype=np.float64)
        self.assertEqual(recon.shape, coords.shape)
        max_err = float(np.max(np.abs(recon - coords)))
        self.assertLessEqual(max_err, enc["quantum"] * 0.51)

    def test_uint8_element_lut(self) -> None:
        encoded = encode_elements_uint8(["H", "C", "N", "O", "S", "Fe", "Zn"])
        self.assertEqual(encoded.dtype, np.uint8)
        np.testing.assert_array_equal(encoded, np.array([1, 6, 7, 8, 16, 26, 30], dtype=np.uint8))
        self.assertAlmostEqual(float(ELEMENT_LUT_MASS[6]), 12.011, places=3)
        self.assertAlmostEqual(float(ELEMENT_LUT_MASS[8]), 15.999, places=3)
        self.assertGreater(float(ELEMENT_LUT_SIGMA[6]), 0.0)
        self.assertGreater(float(ELEMENT_LUT_EPSILON[8]), 0.0)


class TestSimulationSessionRamCap(unittest.TestCase):
    def test_snapshot_cap_and_metrics_fields(self) -> None:
        self.assertEqual(MAX_SNAPSHOT_RAM_BYTES, 64 * 1024 * 1024)
        sys = AtomicSystem(
            positions=np.array([[1.0, 1.0, 1.0], [1.5, 1.0, 1.0], [2.0, 1.0, 1.0]], dtype=np.float64),
            masses=np.full(3, 12.011),
            charges=np.zeros(3),
            box_lengths=(4.0, 4.0, 4.0),
            names=["C", "N", "O"],
            bonds=[],
            units="md",
        )
        sim = SimulationSession(sys, mode="metal_verlet", dt=0.002, temperature=300.0)
        m = sim.get_metrics()
        self.assertEqual(m["storage_mode"], "float64-full")
        self.assertIn("ram_mb", m)
        self.assertIn("memory_savings_ratio", m)
        self.assertEqual(m["active_all_atom_count"], 3)
        self.assertLessEqual(sim._history_ram_bytes, MAX_SNAPSHOT_RAM_BYTES)
        sim.close()


class TestFaustovirus40MLowRAM(unittest.TestCase):
    def setUp(self) -> None:
        self.app = create_app()
        self.client = TestClient(self.app)

    def tearDown(self) -> None:
        self.app.state.session_manager.close_all()

    def test_faustovirus_39_959_280_atoms_e2e(self) -> None:
        t_start = time.perf_counter()
        tsym_bytes, asu_coords, transforms = _build_faustovirus_scale_tsym()

        # 1. Upload 14,478 ASU x 2,760 transforms = 39,959,280 atoms via FastAPI
        resp = self.client.post(
            "/api/sessions/create-from-symmetry",
            content=tsym_bytes,
            headers={"Content-Type": "application/octet-stream"},
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        data = resp.json()
        sid = data["session_id"]
        metrics = data["metrics"]

        self.assertEqual(data["n_asu_atoms"], 14_478)
        self.assertEqual(data["n_transforms"], 2_760)
        self.assertEqual(data["n_atoms"], 39_959_280)
        self.assertEqual(metrics["n_atoms"], 39_959_280)
        self.assertEqual(metrics["storage_mode"], "symmetry-compact")
        self.assertLess(metrics["ram_mb"], 250.0)
        self.assertGreater(metrics["memory_savings_ratio"], 80.0)
        self.assertGreater(metrics["active_all_atom_count"], 0)
        self.assertLessEqual(metrics["active_all_atom_count"], 65_536)
        self.assertEqual(metrics["multiscale"]["n_total"], 39_959_280)
        self.assertIn("health_status", metrics)
        self.assertIn("drift_fraction", metrics)
        self.assertTrue(math.isfinite(metrics["total_energy"]))

        # Verify /api/capabilities exposes symmetry-compact & base243 flags + live ram_mb
        caps = self.client.get("/api/capabilities").json()
        self.assertTrue(caps["supports_symmetry_compact"])
        self.assertTrue(caps["supports_base243_ternary"])
        self.assertIn("storage_mode", caps)
        self.assertIn("ram_mb", caps)
        self.assertLess(caps["ram_mb"], 250.0)

        # 2. Run soft-core clash relaxation on active All-Atom focus zone
        resp_relax = self.client.post(f"/api/sessions/{sid}/relax", json={"max_steps": 15})
        self.assertEqual(resp_relax.status_code, 200, resp_relax.text)
        self.assertIn(resp_relax.json()["relaxation"]["status"], ("relaxed", "clean"))

        # 3. Move Focus Zone live to a completely different capsomer (index 1380, opposite side of 40M capsid!)
        asu_centroid = np.mean(asu_coords, axis=0)
        k_target = 1380
        new_center = (transforms[k_target, :3, :3] @ asu_centroid + transforms[k_target, :3, 3]).tolist()
        t_focus = time.perf_counter()
        resp_ms = self.client.post(
            f"/api/sessions/{sid}/multiscale",
            json={"center_ang": new_center, "aa_radius_ang": 45.0, "cg_radius_ang": 85.0},
        )
        focus_ms = (time.perf_counter() - t_focus) * 1e3
        self.assertEqual(resp_ms.status_code, 200, resp_ms.text)
        ms_data = resp_ms.json()
        self.assertEqual(ms_data["multiscale"]["n_total"], 39_959_280)
        np.testing.assert_allclose(ms_data["multiscale"]["center_ang"], new_center, atol=1e-2)
        self.assertGreater(ms_data["multiscale"]["n_all_atom"], 0)
        self.assertLess(ms_data["metrics"]["ram_mb"], 250.0)
        self.assertLess(focus_ms, 2000.0)

        # 4. Run 3 dynamics steps cleanly on the 40M symmetry-compact session
        resp_step = self.client.post(f"/api/sessions/{sid}/step", json={"n_steps": 3})
        self.assertEqual(resp_step.status_code, 200, resp_step.text)
        step_metrics = resp_step.json()
        self.assertEqual(step_metrics["step"], 3)
        self.assertEqual(step_metrics["n_atoms"], 39_959_280)
        self.assertTrue(math.isfinite(step_metrics["total_energy"]))
        self.assertLess(step_metrics["ram_mb"], 250.0)

        # 5. Verify encode_topology_packet and encode_frame_packet (full + dirty-tile delta)
        record = self.app.state.session_manager.get_session(sid)
        self.assertIsNotNone(record)
        sim = record.session
        self.assertIsInstance(sim, CompactSymmetryMultiscaleSession)

        topo_pkt = sim.encode_topology_packet()
        self.assertEqual(unpack_frame(topo_pkt)["header"].msg_type, MSG_TOPOLOGY_INIT)

        # Reset dirty tracker so first encode emits full frame, then displace 5 atoms for compact delta patch
        sim._dirty_tracker.reset()
        full_pkt = sim.encode_frame_packet()
        dec_full = unpack_frame(full_pkt)
        self.assertEqual(dec_full["header"].msg_type, MSG_FRAME_FULL)
        self.assertEqual(dec_full["header"].n_atoms, sim.capacity)

        sim.positions[:5, 0] += 0.05  # +0.5 Angstrom displacement on 5 atoms
        delta_pkt = sim.encode_frame_packet()
        dec_delta = unpack_frame(delta_pkt)
        self.assertEqual(dec_delta["header"].msg_type, MSG_FRAME_TILE_PATCH)
        self.assertEqual(dec_delta["header"].n_atoms, 5)
        self.assertLess(len(delta_pkt), 1024)  # < 1 KB sparse delta packet!

        total_elapsed_s = time.perf_counter() - t_start
        self.assertLess(total_elapsed_s, 30.0)


    def test_faustovirus_40m_world_space_elements_baoab_shadow_h_and_3zone_metrics(self) -> None:
        """Regression suite for 5J7V live bugs:
        1) Real protein elements (C/N/O/S/H in Zone 0, 'CG' in Zone 1, never 'Ar').
        2) Exact world-space Float32 Angstrom streaming (no 10 A cube collapse).
        3) 50-step symmetric BAOAB + Bipartite Centroid + ENM + HMC guard keeps
           Shadow-H drift < 1% ('bounded') and Temp near 300 K (no runaway to 550 K).
        4) Exact 3-zone multiscale counts summing to 39,959,280 with RAM < 250 MB.
        """
        tsym_bytes, asu_coords, _transforms = _build_faustovirus_scale_tsym()
        sym = unpack_tsym_binary(tsym_bytes)
        sim = CompactSymmetryMultiscaleSession.from_tsym_payload(sym)
        try:
            sim.relax_clashes(max_steps=20)

            # 1) Real protein elements in Zone 0, 'CG' (Z=0) in Zone 1, NEVER 'Ar' (Z=18)
            n_aa = sim.active_all_atom_count
            n_cg = sim.active_cg_bead_count
            self.assertGreater(n_aa, 0)
            self.assertGreater(n_cg, 0)
            self.assertNotIn(18, set(sim.atomic_numbers.tolist()))
            zone0_syms = {sim.inspect_atom(i)["element"] for i in range(min(n_aa, 256))}
            self.assertTrue({"C", "N", "O"}.issubset(zone0_syms))
            self.assertNotIn("Ar", zone0_syms)

            cg_info = sim.inspect_atom(n_aa)
            self.assertEqual(cg_info["element"], "CG")
            self.assertEqual(cg_info["atomic_number"], 0)
            self.assertEqual(cg_info["zone"], 1)

            # 2) World-space Float32 Angstrom streaming (full frame & dirty-tile delta)
            sim._dirty_tracker.reset()
            full_pkt = sim.encode_frame_packet()
            dec_full = unpack_frame(full_pkt)
            self.assertEqual(dec_full["header"].msg_type, MSG_FRAME_FULL)
            self.assertEqual(dec_full["header"].flags & FLAG_INT16_FRACTIONAL, 0)
            self.assertEqual(dec_full["header"].flags & FLAG_PBC_WRAPPED, 0)
            streamed_aa = dec_full["coords"][:n_aa]
            true_aa = (sim.positions[:n_aa] * 10.0).astype(np.float32)
            np.testing.assert_allclose(streamed_aa, true_aa, atol=1e-4)
            # Capsomer focus zone must preserve full ~80-90 A 3D span, never collapse into <= 10 A cube
            span_ang = np.ptp(streamed_aa, axis=0)
            self.assertGreater(float(np.min(span_ang)), 40.0)

            # Sparse dirty-tile delta must also stream exact world-space Float32 Angstroms
            sim.positions[:3, 0] += 0.05  # +0.5 A
            delta_pkt = sim.encode_frame_packet()
            dec_delta = unpack_frame(delta_pkt)
            self.assertEqual(dec_delta["header"].msg_type, MSG_FRAME_TILE_PATCH)
            self.assertEqual(dec_delta["header"].flags & FLAG_INT16_FRACTIONAL, 0)
            self.assertEqual(dec_delta["header"].flags & FLAG_PBC_WRAPPED, 0)
            np.testing.assert_allclose(
                dec_delta["coords"],
                (sim.positions[dec_delta["atom_indices"]] * 10.0).astype(np.float32),
                atol=1e-4,
            )
            # Restore perturbation & reset reference energy before 50-step stability test
            sim.positions[:3, 0] -= 0.05
            sim._evaluate_state()
            sim._q_thermo_cum = 0.0
            sim._initial_total_energy = sim._extended_total_energy()

            # 3) 50 steps of symmetric BAOAB + Bipartite Centroid + ENM + HMC guard
            m50 = sim.step(n_steps=50)
            self.assertEqual(m50["step"], 50)
            self.assertLess(m50["drift_fraction"], 0.01)
            self.assertEqual(m50["health_status"], "bounded")
            self.assertGreater(m50["temperature_K"], 180.0)
            self.assertLess(m50["temperature_K"], 380.0)
            self.assertTrue(math.isfinite(m50["total_energy"]))

            # 4) Exact 3-zone multiscale metrics summing to 39,959,280 & RAM < 250 MB
            ms = m50["multiscale"]
            self.assertGreater(ms["all_atom"], 0)
            self.assertGreater(ms["coarse_grained"], 0)
            self.assertGreater(ms["rigid_scaffold"], 39_000_000)
            self.assertEqual(ms["all_atom"] + ms["coarse_grained"] + ms["rigid_scaffold"], 39_959_280)
            self.assertEqual(ms["n_all_atom"], ms["all_atom"])
            self.assertEqual(ms["n_coarse_grained"], ms["coarse_grained"])
            self.assertEqual(ms["n_rigid_scaffold"], ms["rigid_scaffold"])
            self.assertLess(m50["ram_mb"], 250.0)
        finally:
            sim.close()

    def test_bipartite_centroid_residual_identity_and_hmc_guard(self) -> None:
        """Verify exact Harmonic.lean O(n+m) identity, Newton's 3rd law, and HMC stiff-spike rollback."""
        rng = np.random.default_rng(2026)
        disp_a = rng.normal(0.0, 0.15, size=(11, 3))
        disp_b = rng.normal(0.0, 0.15, size=(17, 3))
        k_c = 250.0
        u_fast, fa, fb = eval_bipartite_centroid_residual(disp_a, disp_b, k_c)

        k_eff = k_c / float(len(disp_a) + len(disp_b))
        diff = disp_a[:, None, :] - disp_b[None, :, :]
        u_exact = 0.5 * k_eff * float(np.sum(diff * diff))
        fa_exact = -k_eff * np.sum(diff, axis=1)
        fb_exact = k_eff * np.sum(diff, axis=0)

        self.assertAlmostEqual(u_fast, u_exact, places=11)
        np.testing.assert_allclose(fa, fa_exact, atol=1e-11)
        np.testing.assert_allclose(fb, fb_exact, atol=1e-11)
        # Exact Newton's 3rd law: sum(F_A) + sum(F_B) == 0
        np.testing.assert_allclose(np.sum(fa, axis=0) + np.sum(fb, axis=0), np.zeros(3), atol=1e-11)

        # Verify HMC Metropolis safeguard rejects an injected unphysical velocity explosion
        asu = np.array([[0.0, 0.0, 0.0], [3.5, 0.0, 0.0], [0.0, 3.5, 0.0], [0.0, 0.0, 3.5]], dtype=np.float32)
        tfs = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], 4, axis=0)
        tfs[1, 0, 3] = 30.0
        tfs[2, 0, 3] = 60.0
        tfs[3, 0, 3] = 120.0
        sim = CompactSymmetryMultiscaleSession(
            asu_coords_ang=asu,
            transforms_4x4=tfs,
            asu_elements=["C", "N", "O", "S"],
            dt=0.002,
            temperature=300.0,
        )
        try:
            pos_before = sim.positions.copy()
            # Inject an extreme unphysical momentum spike that triggers stiff ENM explosion
            sim.momenta[0] = np.array([1.0e5, 0.0, 0.0], dtype=np.float64)
            sim._evaluate_state()
            m_after = sim.step(n_steps=1, apply_thermostat=False)
            self.assertGreaterEqual(m_after["hmc_rejections"], 1)
            np.testing.assert_allclose(sim.positions, pos_before, atol=1e-12)
            self.assertTrue(np.all(np.isfinite(sim.positions)))
            self.assertTrue(np.all(np.isfinite(sim.momenta)))
        finally:
            sim.close()

if __name__ == "__main__":
    unittest.main()
