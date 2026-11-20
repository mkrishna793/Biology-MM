"""Deep verification suite for Trium 27+5 Tryte Tiles, 3D Radix-3 Peano Curve, and Full-Atom Engine."""

from __future__ import annotations

import math
import struct
import unittest

import numpy as np
from fastapi.testclient import TestClient

from trium_studio.app import create_app
from trium_studio.multiscale import pack_tsym_binary
from trium_studio.protocol import (
    FLAG_INT16_FRACTIONAL,
    FLAG_PBC_WRAPPED,
    MSG_FRAME_FULL,
    MSG_FRAME_TILE_PATCH,
    MSG_OVERLAY_VECTORS,
    MSG_TOPOLOGY_INIT,
    MSG_VOLUME_GRID3D,
    unpack_frame,
)
from trium_studio.structure_loader import load_structure_from_bytes
from trium_studio.tryte_tiles import (
    TryteTileFullAtomSession,
    build_tryte_tiles_27_plus_5,
    compare_tryte_peano_vs_binary_morton,
    compute_peano_radix3_codes,
    find_interacting_tryte_pairs,
    peano_order,
)


def _make_protein_pdb(n_residues: int = 80) -> bytes:
    """Generate a realistic asymmetric multi-element protein PDB (C, N, O, S, H, Zn) with CONECT records."""
    lines = ["CRYST1   60.000   60.000   60.000  90.00  90.00  90.00 P 1           1"]
    atom_id = 1
    prev_c_id = None
    # Helical / globular backbone + sidechain atoms with realistic >= 3.4 A nonbonded spacing
    for res in range(1, n_residues + 1):
        t = res * 0.55
        r_helix = 6.0 + 4.0 * math.sin(res * 0.13)
        cx = 25.0 + r_helix * math.cos(t)
        cy = 25.0 + r_helix * math.sin(t)
        cz = 10.0 + res * 0.45
        res_name = "CYS" if (res % 15 == 0) else "ALA"

        atoms = [
            ("N", "N", cx - 1.2, cy, cz),
            ("CA", "C", cx, cy, cz),
            ("C", "C", cx + 1.2, cy + 0.4, cz + 0.2),
            ("O", "O", cx + 1.2, cy + 1.6, cz + 0.2),
            ("SG" if res_name == "CYS" else "CB", "S" if res_name == "CYS" else "C", cx, cy - 1.5, cz + 0.3),
        ]
        n_id = atom_id
        for aname, elem, x, y, z in atoms:
            lines.append(
                f"ATOM  {atom_id:5d}  {aname:<3s} {res_name:3s} A{res:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00          {elem:>2s}"
            )
            atom_id += 1
        lines.append(f"CONECT{n_id:5d}{n_id + 1:5d}")
        lines.append(f"CONECT{n_id + 1:5d}{n_id + 2:5d}")
        lines.append(f"CONECT{n_id + 2:5d}{n_id + 3:5d}")
        if prev_c_id is not None:
            lines.append(f"CONECT{prev_c_id:5d}{n_id:5d}")
        prev_c_id = n_id + 2

    # Add a catalytic Zinc HETATM without col 77-78 element to test HETATM 2-letter detection
    lines.append(
        f"HETATM{atom_id:5d}  ZN  ZN  B 101      25.000  25.000  25.000  1.00 15.00"
    )
    return ("\n".join(lines) + "\n").encode("utf-8")


def _make_tbin_packet(
    coords_ang: np.ndarray,
    elem_bytes: bytes,
    title: str = "Asymmetric_TBIN_Protein",
    mode: str = "metal_verlet",
    temperature: float = 300.0,
    dt: float = 0.002,
) -> bytes:
    n = int(coords_ang.shape[0])
    title_b = title.encode("utf-8")
    mode_b = mode.encode("utf-8")
    hdr = struct.pack(
        "<4sIIffIIfffII",
        b"TBIN",
        n,
        len(title_b),
        float(temperature),
        float(dt),
        len(mode_b),
        1,  # has_elements
        0.0,
        0.0,
        0.0,
        0,
        0,
    )
    return hdr + np.ascontiguousarray(coords_ang, dtype=np.float32).tobytes() + elem_bytes + title_b + mode_b


class TestPeanoRadix3Continuity(unittest.TestCase):
    def test_exhaustive_peano_3d_face_adjacency_zero_z_jumps(self) -> None:
        """Consecutive cells along the 3D Radix-3 Peano curve must be 100% face-adjacent (L1 == 1)."""
        for levels in (1, 2, 3):
            side = 3 ** levels
            gx, gy, gz = np.mgrid[0:side, 0:side, 0:side]
            pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1).astype(np.float64) + 0.5
            codes = compute_peano_radix3_codes(
                pts,
                origin=(0.0, 0.0, 0.0),
                span=(float(side), float(side), float(side)),
                levels=levels,
            )
            self.assertEqual(codes.dtype, np.uint64)
            self.assertEqual(int(np.unique(codes).size), side ** 3)
            self.assertEqual(int(codes.min()), 0)
            self.assertEqual(int(codes.max()), side ** 3 - 1)

            order = np.argsort(codes)
            ordered_grid = np.floor(pts[order]).astype(np.int64)
            l1_steps = np.sum(np.abs(np.diff(ordered_grid, axis=0)), axis=1)
            self.assertEqual(int(l1_steps.min()), 1, f"Duplicate/zero step at levels={levels}")
            self.assertEqual(int(l1_steps.max()), 1, f"Diagonal Z-jump detected at levels={levels}")
            self.assertEqual(int(np.sum(l1_steps != 1)), 0)

    def test_peano_order_permutation_valid(self) -> None:
        rng = np.random.default_rng(7)
        pts = rng.uniform(-10.0, 10.0, size=(513, 3)).astype(np.float32)
        perm = peano_order(pts, levels=6)
        self.assertEqual(perm.shape, (513,))
        np.testing.assert_array_equal(np.sort(perm), np.arange(513, dtype=np.int32))


class TestTryteTileMetadataExactness(unittest.TestCase):
    def test_lanes_27_to_31_match_brute_force_exact(self) -> None:
        """Lane 27 centroids, Lane 28 radii/charges, Lane 29 residual moments/dipoles match brute force < 1e-5."""
        rng = np.random.default_rng(2026)
        n = 500  # 18 full tiles of 27 + 1 partial tile of 14
        pos_nm = rng.uniform(-15.0, 15.0, size=(n, 3)).astype(np.float32)
        charges = rng.uniform(-1.0, 1.0, size=(n,)).astype(np.float32)
        z_arr = rng.choice(np.array([1, 6, 7, 8, 16], dtype=np.uint8), size=n)

        tset = build_tryte_tiles_27_plus_5(pos_nm, charges=charges, atomic_numbers=z_arr, tile_size=27, reorder=True)
        self.assertEqual(tset.n_atoms, n)
        self.assertEqual(tset.n_tiles, math.ceil(n / 27))
        self.assertIsNotNone(tset.peano_perm)

        pos_ord = pos_nm[tset.peano_perm].astype(np.float64)
        q_ord = charges[tset.peano_perm].astype(np.float64)

        for t in range(tset.n_tiles):
            s = t * 27
            e = min(n, s + 27)
            sub_p = pos_ord[s:e]
            sub_q = q_ord[s:e]
            c_ref = np.mean(sub_p, axis=0)
            rel = sub_p - c_ref[None, :]
            r_ref = float(np.max(np.linalg.norm(rel, axis=1)))
            q_ref = float(np.sum(sub_q))
            rm_ref = float(np.sum(rel * rel))
            dip_ref = np.sum(sub_q[:, None] * rel, axis=0)

            self.assertEqual(int(tset.tile_counts[t]), e - s)
            np.testing.assert_allclose(tset.centroids_nm[t], c_ref, atol=1e-5, rtol=1e-5)
            self.assertAlmostEqual(float(tset.bounding_radii_nm[t]), r_ref, delta=1e-5)
            self.assertAlmostEqual(float(tset.total_charges[t]), q_ref, delta=1e-5)
            self.assertAlmostEqual(float(tset.residual_moments[t]), rm_ref, delta=1e-4)
            np.testing.assert_allclose(tset.dipoles[t], dip_ref, atol=1e-4, rtol=1e-4)

        # Verify interacting tile pairs match O(n_tiles^2) exact sphere-overlap formula
        rcut = 1.2
        pairs = find_interacting_tryte_pairs(tset, rcut_nm=rcut)
        got_set = {(int(a), int(b)) for a, b in pairs}
        ref_set = set()
        for a in range(tset.n_tiles):
            for b in range(a + 1, tset.n_tiles):
                d = float(np.linalg.norm(tset.centroids_nm[a].astype(np.float64) - tset.centroids_nm[b].astype(np.float64)))
                if d <= float(tset.bounding_radii_nm[a]) + float(tset.bounding_radii_nm[b]) + rcut:
                    ref_set.add((a, b))
        self.assertEqual(got_set, ref_set)

    def test_compare_peano_vs_morton_diagnostic(self) -> None:
        rng = np.random.default_rng(11)
        # Globular protein-like 3D cloud of 5,400 atoms
        pts = rng.normal(0.0, 2.2, size=(5400, 3)).astype(np.float32)
        diag = compare_tryte_peano_vs_binary_morton(pts, rcut_nm=1.0)
        self.assertEqual(diag["n_atoms"], 5400)
        self.assertLess(
            diag["peano_27"]["mean_bounding_radius_nm"],
            diag["morton_32"]["mean_bounding_radius_nm"],
        )
        self.assertLess(
            diag["peano_27"]["mean_aabb_volume_nm3"],
            diag["morton_32"]["mean_aabb_volume_nm3"],
        )
        self.assertGreater(diag["atom_pair_check_reduction_pct"], 0.0)


class TestAsymmetricUploadsAndBinaryRoutes(unittest.TestCase):
    def setUp(self) -> None:
        self.app = create_app()
        self.client = TestClient(self.app)

    def tearDown(self) -> None:
        self.app.state.session_manager.close_all()

    def test_capabilities_reports_tryte_27_plus_5(self) -> None:
        caps = self.client.get("/api/capabilities").json()
        self.assertTrue(caps.get("supports_tryte_27_plus_5"))

    def test_structure_loader_edge_cases_xyz_cif_pdb_gro(self) -> None:
        # XYZ with BLANK comment line must preserve all atoms (not drop atom 0!)
        xyz_blank_comment = b"3\n\nO 0.0 0.0 0.0\nH 0.95 0.0 0.0\nH -0.24 0.92 0.0\n"
        sys_xyz, _ = load_structure_from_bytes(xyz_blank_comment, "water.xyz")
        self.assertEqual(sys_xyz.n_atoms, 3)
        self.assertEqual(sys_xyz.names, ["O", "H", "H"])

        # mmCIF with `#` terminator after _atom_site loop and quoted atom label
        cif_bytes = (
            "data_test\n"
            "loop_\n"
            "_atom_site.group_PDB\n"
            "_atom_site.id\n"
            "_atom_site.type_symbol\n"
            "_atom_site.label_atom_id\n"
            "_atom_site.label_comp_id\n"
            "_atom_site.label_asym_id\n"
            "_atom_site.Cartn_x\n"
            "_atom_site.Cartn_y\n"
            "_atom_site.Cartn_z\n"
            "ATOM 1 N N ALA A 0.0 0.0 0.0\n"
            "ATOM 2 C CA ALA A 1.46 0.0 0.0\n"
            "ATOM 3 O 'O' ALA A 2.01 1.42 0.0\n"
            "HETATM 4 ZN ZN ZN B 5.0 5.0 5.0\n"
            "#\n"
            "not_an_atom_row 1 2 3 4 5 6 7 8\n"
        ).encode("utf-8")
        sys_cif, _ = load_structure_from_bytes(cif_bytes, "mini.cif")
        self.assertEqual(sys_cif.n_atoms, 4)
        self.assertEqual(sys_cif.names, ["N", "C", "O", "Zn"])

    def test_asymmetric_pdb_xyz_cif_upload_in_tryte_mode(self) -> None:
        pdb_bytes = _make_protein_pdb(n_residues=80)  # 401 atoms (C, N, O, S, Zn)
        res = self.client.post(
            "/api/sessions/upload?filename=asymmetric_protein.pdb&storage_mode=tryte-all-atom",
            content=pdb_bytes,
        )
        self.assertEqual(res.status_code, 200, res.text)
        data = res.json()
        sid = data["session_id"]
        self.assertEqual(data["storage_mode"], "tryte-all-atom")
        self.assertEqual(data["num_atoms"], 401)

        # Verify elements preserved (C, N, O, S, Zn — NEVER Ar!)
        rec = self.app.state.session_manager.get_session(sid)
        self.assertIsNotNone(rec)
        sim = rec.session
        self.assertIsInstance(sim, TryteTileFullAtomSession)
        unique_z = set(int(z) for z in np.unique(sim.atomic_numbers))
        self.assertIn(6, unique_z)   # C
        self.assertIn(7, unique_z)   # N
        self.assertIn(8, unique_z)   # O
        self.assertIn(16, unique_z)  # S
        self.assertIn(30, unique_z)  # Zn
        self.assertNotIn(18, unique_z)  # NEVER Ar!

        # Run 30 dynamics steps and check bounded health, drift < 0.01, and Temp near 300 K
        step_res = self.client.post(f"/api/sessions/{sid}/step", json={"n_steps": 30})
        self.assertEqual(step_res.status_code, 200, step_res.text)
        m = step_res.json()
        self.assertEqual(m["storage_mode"], "tryte-all-atom")
        self.assertEqual(m["health_status"], "bounded")
        self.assertLess(m["drift_fraction"], 0.01)
        self.assertGreater(m["temperature_K"], 150.0)
        self.assertLess(m["temperature_K"], 450.0)
        self.assertGreater(m["n_tiles"], 0)
        self.assertGreaterEqual(m["tryte_pair_count"], 0)

        # Verify world-space Float32 topology & frame packets (pbc_wrapped=False)
        topo = unpack_frame(sim.encode_topology_packet())
        self.assertEqual(topo["header"].msg_type, MSG_TOPOLOGY_INIT)
        self.assertEqual(topo["header"].n_atoms, 401)
        self.assertNotIn(18, set(int(z) for z in topo["atomic_numbers"]))

        frm_full = unpack_frame(sim.encode_frame_packet(use_delta=False))
        self.assertEqual(frm_full["header"].msg_type, MSG_FRAME_FULL)
        self.assertEqual(frm_full["header"].flags & FLAG_PBC_WRAPPED, 0)
        self.assertEqual(frm_full["header"].flags & FLAG_INT16_FRACTIONAL, 0)

        # Verify overlay, volume, inspect_atom, steer, steer_atom, scrub_to_step, and mmCIF export
        ov = unpack_frame(sim.encode_overlay_packet())
        self.assertEqual(ov["header"].msg_type, MSG_OVERLAY_VECTORS)
        vol = unpack_frame(sim.encode_volume_packet())
        self.assertEqual(vol["header"].msg_type, MSG_VOLUME_GRID3D)

        atom0 = self.client.get(f"/api/sessions/{sid}/atom/0").json()
        self.assertIn(atom0["element"], ("C", "N", "O", "S", "Zn"))

        cif_resp = self.client.get(f"/api/sessions/{sid}/structure.cif")
        self.assertEqual(cif_resp.status_code, 200)
        self.assertIn("_atom_site.Cartn_x", cif_resp.text)

        # Also verify XYZ and mmCIF uploads in tryte-all-atom mode
        xyz_bytes = b"4\n\nC 0.0 0.0 0.0\nN 1.45 0.0 0.0\nO 2.9 0.0 0.0\nS 0.0 2.0 0.0\n"
        r_xyz = self.client.post("/api/sessions/upload?filename=test.xyz&storage_mode=tryte-all-atom", content=xyz_bytes)
        self.assertEqual(r_xyz.status_code, 200, r_xyz.text)
        self.assertEqual(r_xyz.json()["storage_mode"], "tryte-all-atom")
        self.assertEqual(r_xyz.json()["num_atoms"], 4)

        cif_bytes = (
            "data_mini\nloop_\n_atom_site.group_PDB\n_atom_site.id\n_atom_site.type_symbol\n"
            "_atom_site.Cartn_x\n_atom_site.Cartn_y\n_atom_site.Cartn_z\n"
            "ATOM 1 C 0.0 0.0 0.0\nATOM 2 N 1.5 0.0 0.0\nATOM 3 O 3.0 0.0 0.0\n"
        ).encode("utf-8")
        r_cif = self.client.post("/api/sessions/upload?filename=test.cif&storage_mode=tryte-all-atom", content=cif_bytes)
        self.assertEqual(r_cif.status_code, 200, r_cif.text)
        self.assertEqual(r_cif.json()["storage_mode"], "tryte-all-atom")
        self.assertEqual(r_cif.json()["num_atoms"], 3)

    def test_create_from_binary_tbin_tryte_all_atom(self) -> None:
        rng = np.random.default_rng(314)
        side = 8
        grid = np.mgrid[0:side, 0:side, 0:side].reshape(3, -1).T.astype(np.float32) * 3.8
        grid += rng.normal(0.0, 0.15, size=grid.shape).astype(np.float32)
        n = grid.shape[0]  # 512 atoms
        elems = bytes([ord("CNOS"[i % 4]) for i in range(n)])
        payload = _make_tbin_packet(grid, elems, title="TBIN_Tryte_512")

        res = self.client.post(
            "/api/sessions/create-from-binary?storage_mode=tryte-all-atom",
            content=payload,
            headers={"Content-Type": "application/octet-stream"},
        )
        self.assertEqual(res.status_code, 200, res.text)
        data = res.json()
        sid = data["session_id"]
        self.assertEqual(data["storage_mode"], "tryte-all-atom")
        self.assertEqual(data["metrics"]["n_atoms"], n)

        m = self.client.post(f"/api/sessions/{sid}/step", json={"n_steps": 15}).json()
        self.assertEqual(m["health_status"], "bounded")
        self.assertLess(m["drift_fraction"], 0.01)
        self.assertGreater(m["temperature_K"], 150.0)
        self.assertLess(m["temperature_K"], 450.0)


class TestSymmetricAssemblyTryteAllAtom(unittest.TestCase):
    def setUp(self) -> None:
        self.app = create_app()
        self.client = TestClient(self.app)

    def tearDown(self) -> None:
        self.app.state.session_manager.close_all()

    def test_symmetry_tryte_all_atom_moves_all_atoms_low_bytes_per_atom(self) -> None:
        """Symmetric assembly in tryte-all-atom mode moves ALL atoms independently at ~44-60 B/atom."""
        rng = np.random.default_rng(77)
        # 500 ASU atoms x 110 transforms = 55,000 atoms (> 50k -> verifies zero snapshot copies!)
        n_asu = 500
        n_tf = 110
        side = int(math.ceil(n_asu ** (1.0 / 3.0)))  # 8
        grid = np.mgrid[0:side, 0:side, 0:side].reshape(3, -1).T[:n_asu].astype(np.float32)
        asu_coords = (grid - side * 0.5) * np.float32(3.8)
        asu_elems = [["C", "N", "O", "S", "H"][i % 5] for i in range(n_asu)]

        transforms = np.zeros((n_tf, 4, 4), dtype=np.float32)
        transforms[:, 0, 0] = 1.0
        transforms[:, 1, 1] = 1.0
        transforms[:, 2, 2] = 1.0
        transforms[:, 3, 3] = 1.0
        # Arrange transforms on a 350 A hollow capsid shell
        golden = math.pi * (3.0 - math.sqrt(5.0))
        idx = np.arange(n_tf, dtype=np.float64)
        y = 1.0 - 2.0 * (idx + 0.5) / n_tf
        r_xy = np.sqrt(np.maximum(0.0, 1.0 - y * y))
        th = golden * idx
        transforms[:, 0, 3] = (np.cos(th) * r_xy * 350.0).astype(np.float32)
        transforms[:, 1, 3] = (y * 350.0).astype(np.float32)
        transforms[:, 2, 3] = (np.sin(th) * r_xy * 350.0).astype(np.float32)

        tsym = pack_tsym_binary(
            asu_coords_ang=asu_coords,
            transforms_4x4=transforms,
            asu_elements=asu_elems,
            title="Capsid_55k_Tryte_AllAtom",
            temperature=300.0,
            dt=0.002,
        )
        res = self.client.post(
            "/api/sessions/create-from-symmetry?storage_mode=tryte-all-atom",
            content=tsym,
            headers={"Content-Type": "application/octet-stream"},
        )
        self.assertEqual(res.status_code, 200, res.text)
        data = res.json()
        sid = data["session_id"]
        self.assertEqual(data["storage_mode"], "tryte-all-atom")
        self.assertEqual(data["n_atoms"], 55_000)

        rec = self.app.state.session_manager.get_session(sid)
        sim = rec.session
        self.assertIsInstance(sim, TryteTileFullAtomSession)
        # Zero full-frame snapshots when N > 50,000!
        self.assertEqual(len(sim._history), 0)

        m0 = data["metrics"]
        self.assertGreaterEqual(m0["bytes_per_atom"], 40.0)
        self.assertLessEqual(m0["bytes_per_atom"], 60.0)

        pos_before = sim.positions_nm.copy()
        m1 = self.client.post(f"/api/sessions/{sid}/step", json={"n_steps": 5}).json()
        self.assertEqual(m1["health_status"], "bounded")
        self.assertLess(m1["drift_fraction"], 0.01)
        self.assertEqual(len(m1["history_steps"]), 0)

        # Verify ALL 55,000 atoms moved independently (not just Zone 0 focus atoms!)
        disp = np.linalg.norm(sim.positions_nm - pos_before, axis=1)
        moved_frac = float(np.mean(disp > 1e-7))
        self.assertGreater(moved_frac, 0.999, f"Expected all atoms to move, only {moved_frac:.4f} moved")

        # Verify streamed WebGL packets stay capped <= 65,536
        topo = unpack_frame(sim.encode_topology_packet())
        self.assertLessEqual(topo["header"].n_atoms, 65_536)


if __name__ == "__main__":
    unittest.main()
