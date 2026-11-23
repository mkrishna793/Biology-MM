"""End-to-end verification of Biology-MM v2.0 Open-Science capabilities.

Verifies:
1. Standard CHARMM/NAMD .dcd, GROMACS .xtc (both uncompressed <=9 and 3DF compressed >9), and .pdb I/O.
2. Exact analytical VJP gradients vs central finite differences for all 6 Collective Variables.
3. Umbrella Sampling, Steered MD (Jarzynski work), Well-Tempered Metadynamics, and PLUMEDParser.
4. OpenFF-Sage parameterization + AMBER .prmtop/.inpcrd round-trip into structure_loader + AlphaFold pLDDT Focus Zone.
5. Multi-domain OpenScienceEngine (Biology, Chemistry, Physics) custom parameter overrides & user potential callbacks.
6. Studio REST API endpoints (/export/dcd, /export/xtc, /export/pdb, /enhanced-sampling, /physics-params) with zero PyMOL hooks.
"""

from __future__ import annotations

import io
import math
import unittest

import numpy as np
from fastapi.testclient import TestClient

from biology_mm import (
    AngleCV,
    COMDistanceCV,
    CoordinationNumberCV,
    DihedralCV,
    DistanceCV,
    EnhancedSamplingSuite,
    OpenScienceEngine,
    PLUMEDParser,
    PhysicsParameters,
    RadiusOfGyrationCV,
    SteeredMDBias,
    UmbrellaBias,
    WellTemperedMetadynamics,
    build_alanine_dipeptide,
    build_lj_argon_fluid,
    build_morse_diatomic,
    build_tip3p_water_box,
    build_yukawa_plasma,
    compute_plddt_focus_zone,
    export_amber_prmtop_inpcrd,
    parameterize_openff_sage,
    parse_alphafold_plddt,
    read_dcd,
    read_xtc,
    write_dcd,
    write_pdb_topology,
    write_xtc,
)
from trium_studio.app import create_app
from trium_studio.structure_loader import load_structure_pair_from_bytes


def _numerical_grad(cv, pos: np.ndarray, box=None, eps: float = 1e-6) -> np.ndarray:
    grad = np.zeros_like(pos)
    for i in range(pos.shape[0]):
        for d in range(3):
            pp = pos.copy()
            pm = pos.copy()
            pp[i, d] += eps
            pm[i, d] -= eps
            diff = cv.evaluate(pp, box) - cv.evaluate(pm, box)
            if cv.periodic is not None:
                diff = (diff + 0.5 * cv.periodic) % cv.periodic - 0.5 * cv.periodic
            grad[i, d] = diff / (2.0 * eps)
    return grad


class TestStandardTrajectoryIO(unittest.TestCase):
    def test_dcd_and_xtc_roundtrip_compressed_and_uncompressed(self) -> None:
        rng = np.random.default_rng(7)
        for n_atoms in (6, 42):  # 6 tests uncompressed XTC (<=9), 42 tests 3DF compressed XTC (>9)
            frames = rng.uniform(0.2, 2.8, size=(5, n_atoms, 3))
            boxes = np.repeat(np.diag([3.0, 3.2, 3.4])[None, :, :], 5, axis=0)

            dcd_buf = io.BytesIO()
            write_dcd(dcd_buf, frames, boxes=boxes, dt_ps=0.002, units="nm")
            dcd_raw = dcd_buf.getvalue()
            self.assertEqual(dcd_raw[4:8], b"CORD")
            dcd_out = read_dcd(dcd_raw)
            self.assertEqual(dcd_out.n_frames, 5)
            self.assertEqual(dcd_out.n_atoms, n_atoms)
            self.assertLess(float(np.max(np.abs(dcd_out.positions_nm - frames))), 1e-5)
            self.assertLess(float(np.max(np.abs(dcd_out.box_nm - boxes))), 1e-5)

            xtc_buf = io.BytesIO()
            write_xtc(xtc_buf, frames, boxes=boxes, dt_ps=0.002, precision=1000.0, units="nm")
            xtc_raw = xtc_buf.getvalue()
            xtc_out = read_xtc(xtc_raw)
            self.assertEqual(xtc_out.n_frames, 5)
            self.assertEqual(xtc_out.n_atoms, n_atoms)
            tol = 1e-5 if n_atoms <= 9 else 0.0006  # 1/(2*1000) nm = 0.0005 nm
            self.assertLess(float(np.max(np.abs(xtc_out.positions_nm - frames))), tol)
            self.assertLess(float(np.max(np.abs(xtc_out.box_nm - boxes))), 1e-5)

    def test_pdb_topology_writer(self) -> None:
        pos = np.array([[0.1, 0.2, 0.3], [0.25, 0.2, 0.3]])
        pdb = write_pdb_topology(pos, elements=["C", "O"], box=np.diag([2.0, 2.0, 2.0]), units="nm")
        self.assertIn("CRYST1", pdb)
        self.assertIn("ATOM      1", pdb)
        self.assertIn("END", pdb)


class TestEnhancedSamplingVJP(unittest.TestCase):
    def test_all_six_cvs_match_finite_differences(self) -> None:
        rng = np.random.default_rng(19)
        pos = rng.uniform(0.3, 1.5, size=(8, 3))
        box = np.diag([3.0, 3.0, 3.0])
        masses = np.array([12.0, 14.0, 16.0, 12.0, 12.0, 16.0, 1.0, 1.0])

        cvs = [
            DistanceCV(0, 1),
            COMDistanceCV([0, 1, 2], [3, 4, 5], masses=masses),
            AngleCV(0, 1, 2),
            DihedralCV(0, 1, 2, 3),
            RadiusOfGyrationCV([0, 1, 2, 3, 4], masses=masses),
            CoordinationNumberCV([0, 1], [2, 3, 4], r0_nm=0.65),
        ]
        for cv in cvs:
            val, g_ana = cv.evaluate_with_grad(pos, box)
            self.assertTrue(math.isfinite(val))
            g_num = _numerical_grad(cv, pos, box)
            err = float(np.max(np.abs(g_ana - g_num)))
            self.assertLess(err, 1e-5, f"VJP mismatch for {cv.__class__.__name__}: {err}")

    def test_umbrella_smd_metad_and_plumed_parser(self) -> None:
        plumed_cfg = """
        # Alanine dipeptide PLUMED-style enhanced sampling config
        phi: TORSION ATOMS=2,4,5,7
        d1: DISTANCE ATOMS=1,10
        umb: RESTRAINT ARG=phi AT=-1.2 KAPPA=150.0
        pull: MOVINGRESTRAINT ARG=d1 AT0=0.65 VELOCITY=0.05 KAPPA=200.0
        metad: METAD ARG=phi HEIGHT=1.0 SIGMA=0.3 BIASFACTOR=8.0 PACE=10
        """
        suite = PLUMEDParser.from_text(plumed_cfg, one_based=True)
        self.assertEqual(len(suite.cvs), 2)
        self.assertEqual(len(suite.biases), 3)

        top = build_alanine_dipeptide()
        eng = OpenScienceEngine(top, PhysicsParameters(dt_ps=0.001, seed=11), enhanced_sampling=suite)
        rep = eng.run(n_steps=40, report_interval=5)
        self.assertEqual(len(rep.cv_history), 8)
        self.assertIn("phi", rep.cv_history[-1])
        self.assertIn("pull_work", rep.cv_history[-1])


class TestOpenFFAmberAndAlphaFoldBridge(unittest.TestCase):
    def test_openff_parameterize_and_amber_roundtrip_into_studio_loader(self) -> None:
        pos = np.array([
            [0.00, 0.00, 0.00],  # C
            [0.15, 0.00, 0.00],  # O
            [0.00, 0.11, 0.00],  # H
        ])
        pset = parameterize_openff_sage(pos, elements=["C", "O", "H"], box_nm=np.diag([2.0, 2.0, 2.0]))
        self.assertAlmostEqual(float(np.sum(pset.charges_e)), 0.0, places=6)
        self.assertGreaterEqual(len(pset.bonds), 2)

        prmtop, inpcrd = export_amber_prmtop_inpcrd(pset)
        loaded, _meta = load_structure_pair_from_bytes(
            prmtop.encode("utf-8"), "mol.prmtop", inpcrd.encode("utf-8"), "mol.inpcrd"
        )
        self.assertEqual(loaded.n_atoms, 3)

    def test_alphafold_plddt_focus_zone(self) -> None:
        pos = np.array([[0.1, 0.1, 0.1], [0.2, 0.1, 0.1], [1.0, 1.0, 1.0], [1.1, 1.0, 1.0]])
        pdb = write_pdb_topology(pos, elements=["C", "N", "C", "O"], bfactors=[55.0, 62.0, 95.0, 96.0], units="nm")
        conf = parse_alphafold_plddt(pdb)
        self.assertEqual(list(conf.low_confidence_indices), [0, 1])
        self.assertEqual(list(conf.high_confidence_indices), [2, 3])
        zone = compute_plddt_focus_zone(pos, conf)
        self.assertEqual(zone["low_plddt_atoms"], 2)
        self.assertLess(zone["focus_center_nm"][0], 0.3)


class TestMultiDomainCustomPhysicsAndStudioAPI(unittest.TestCase):
    def test_custom_user_potential_and_parameter_override(self) -> None:
        top = build_morse_diatomic(r_init_nm=0.16)

        def harmonic_trap(pos, box, params):
            k = 100.0
            u = 0.5 * k * float(np.sum(pos * pos))
            f = -k * pos
            return u, f

        params = PhysicsParameters(
            integrator="nve_verlet",
            pair_style="morse",
            dt_ps=0.0005,
            temperature_K=0.0,
            custom_potential=harmonic_trap,
        )
        eng = OpenScienceEngine(top, params)
        eng.set_parameters(morse_De_kj_mol=500.0)
        self.assertEqual(eng.params.morse_De_kj_mol, 500.0)
        rep = eng.run(n_steps=40, report_interval=5)
        self.assertLess(rep.relative_energy_drift, 5e-3)

    def test_studio_open_science_rest_endpoints_no_pymol(self) -> None:
        app = create_app()
        with TestClient(app) as client:
            cap = client.get("/api/capabilities").json()
            self.assertTrue(cap["supports_dcd_export"])
            self.assertTrue(cap["supports_xtc_export"])
            self.assertTrue(cap["supports_enhanced_sampling"])
            self.assertNotIn("pymol", str(cap).lower())

            created = client.post("/api/sessions", json={"preset": "lj_argon_fluid", "temperature": 94.4}).json()
            sid = created["session_id"]
            client.post(f"/api/sessions/{sid}/step", json={"steps": 4})

            # Standard .dcd, .xtc, and .pdb export
            dcd_resp = client.get(f"/api/sessions/{sid}/export/dcd")
            self.assertEqual(dcd_resp.status_code, 200)
            dcd_traj = read_dcd(dcd_resp.content)
            self.assertGreaterEqual(dcd_traj.n_frames, 2)

            xtc_resp = client.get(f"/api/sessions/{sid}/export/xtc")
            self.assertEqual(xtc_resp.status_code, 200)
            xtc_traj = read_xtc(xtc_resp.content)
            self.assertEqual(xtc_traj.n_frames, dcd_traj.n_frames)

            pdb_resp = client.get(f"/api/sessions/{sid}/export/pdb")
            self.assertEqual(pdb_resp.status_code, 200)
            self.assertIn("ATOM", pdb_resp.text)

            # Live enhanced sampling & physics parameter updates
            es_resp = client.post(
                f"/api/sessions/{sid}/enhanced-sampling",
                json={"plumed_text": "d: DISTANCE ATOMS=1,2\nRESTRAINT ARG=d AT=0.4 KAPPA=100.0"},
            )
            self.assertEqual(es_resp.status_code, 200)
            self.assertTrue(es_resp.json()["ok"])

            pp_resp = client.post(
                f"/api/sessions/{sid}/physics-params",
                json={"temperature": 120.0, "epsilon_scale": 1.1, "efield_z": 0.5},
            )
            self.assertEqual(pp_resp.status_code, 200)
            self.assertEqual(pp_resp.json()["updated"]["temperature"], 120.0)


    def test_all_symmetry_and_nonsymmetry_modes_export_and_sample_cleanly(self) -> None:
        from trium_studio.multiscale import pack_tsym_binary

        app = create_app()
        with TestClient(app) as client:
            asu_pos = np.array([[1.0, 1.0, 1.0], [2.5, 1.0, 1.0], [1.0, 2.5, 1.0], [2.5, 2.5, 1.0]], dtype=np.float32)
            asu_elems = ["C", "N", "O", "C"]
            ops = np.stack([
                np.eye(4, dtype=np.float32),
                np.array([[1, 0, 0, 10], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32),
            ], axis=0)
            tsym = pack_tsym_binary(asu_pos, ops, asu_elems, title="SymAudit")
            pdb_str = write_pdb_topology(asu_pos, elements=asu_elems, units="angstrom")

            sids = [
                client.post("/api/sessions/create-from-symmetry?storage_mode=symmetry-compact&temperature=300", content=tsym, headers={"Content-Type": "application/octet-stream"}).json()["session_id"],
                client.post("/api/sessions/create-from-symmetry?storage_mode=tryte-all-atom&temperature=300", content=tsym, headers={"Content-Type": "application/octet-stream"}).json()["session_id"],
                client.post("/api/sessions/upload?storage_mode=tryte-all-atom&temperature=300", files={"file": ("t.pdb", pdb_str.encode("utf-8"), "chemical/x-pdb")}).json()["session_id"],
            ]
            for sid in sids:
                self.assertEqual(client.post(f"/api/sessions/{sid}/step", json={"steps": 2}).status_code, 200)
                self.assertEqual(client.get(f"/api/sessions/{sid}/export/dcd").status_code, 200)
                self.assertEqual(client.get(f"/api/sessions/{sid}/export/xtc").status_code, 200)
                self.assertEqual(client.get(f"/api/sessions/{sid}/export/pdb").status_code, 200)
                self.assertEqual(
                    client.post(
                        f"/api/sessions/{sid}/enhanced-sampling",
                        json={"plumed_text": "d: DISTANCE ATOMS=1,2\nRESTRAINT ARG=d AT=0.3 KAPPA=50.0"},
                    ).status_code,
                    200,
                )

    def test_uio66_crystallographic_cif_upload_and_simulation(self) -> None:
        uio66_cif = """data_UiO-66_mini
_cell_length_a   20.7004(5)
_cell_length_b   20.7004(5)
_cell_length_c   20.7004(5)
_cell_angle_alpha 90.0
_cell_angle_beta  90.0
_cell_angle_gamma 90.0
loop_
_symmetry_equiv_pos_as_xyz
'x, y, z'
'-x, -y, z'
'-x, y, -z'
'x, -y, -z'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Zr1 Zr 0.1200(2) 0.0000 0.0000
O1  O  0.0560(3) 0.0560 0.0560
C1  C  0.1550(4) 0.1550 0.0000
H1  H  0.2050(5) 0.1550 0.0000
"""
        from trium_studio.structure_loader import load_structure_from_bytes

        sys_obj, meta = load_structure_from_bytes(uio66_cif.encode("utf-8"), "UiO-66.cif")
        self.assertTrue(meta.get("crystallographic"))
        self.assertGreaterEqual(sys_obj.n_atoms, 10)
        # Verify real Zirconium atomic mass (~91.224 amu) is preserved
        zr_idx = [i for i, e in enumerate(sys_obj.names) if str(e).upper().startswith("ZR")]
        self.assertGreater(len(zr_idx), 0)
        self.assertAlmostEqual(float(sys_obj.masses[zr_idx[0]]), 91.224, places=2)

        # Verify OpenFF/UFF parameterizer assigns real Zr parameters
        pset = parameterize_openff_sage(sys_obj.positions, elements=["Zr", "O", "C", "H"] + ["C"] * (sys_obj.n_atoms - 4))
        self.assertAlmostEqual(float(pset.masses_amu[0]), 91.224, places=2)

        # Verify end-to-end upload, live MD step, and .dcd/.xtc export via Studio API
        app = create_app()
        with TestClient(app) as client:
            up = client.post(
                "/api/sessions/upload?storage_mode=tryte-all-atom&temperature=300",
                files={"file": ("UiO-66.cif", uio66_cif.encode("utf-8"), "chemical/x-cif")},
            )
            self.assertEqual(up.status_code, 200, up.text)
            sid = up.json()["session_id"]
            self.assertEqual(client.post(f"/api/sessions/{sid}/step", json={"steps": 2}).status_code, 200)
            self.assertEqual(client.get(f"/api/sessions/{sid}/export/dcd").status_code, 200)
            self.assertEqual(client.get(f"/api/sessions/{sid}/export/xtc").status_code, 200)

    def test_code_lab_live_custom_drug_and_material_script(self) -> None:
        code = """
import numpy as np
from biology_mm import AtomicSystem, parameterize_openff_sage
from trium_studio.engine_core import SimulationSession

# Build a custom Zr-O-C hybrid material / drug cluster directly in Code Lab
pos = np.array([
    [0.50, 0.50, 0.50],
    [0.71, 0.50, 0.50],
    [0.84, 0.50, 0.50],
    [0.92, 0.58, 0.50],
], dtype=np.float64)
pset = parameterize_openff_sage(pos, elements=["Zr", "O", "C", "H"])
sys = AtomicSystem(
    positions=pset.positions_nm,
    masses=pset.masses_amu,
    charges=pset.charges_e,
    box_lengths=(2.0, 2.0, 2.0),
    names=pset.elements,
)
session = SimulationSession(sys, temperature=300.0, dt=0.001)
"""
        app = create_app()
        with TestClient(app) as client:
            resp = client.post("/api/code/run", json={"code": code})
            self.assertEqual(resp.status_code, 200, resp.text)
            data = resp.json()
            self.assertTrue(data.get("ok"), data)
            sid = data["session_id"]
            self.assertEqual(client.post(f"/api/sessions/{sid}/step", json={"steps": 2}).status_code, 200)
            self.assertEqual(client.get(f"/api/sessions/{sid}/export/dcd").status_code, 200)

    def test_universal_disorder_filtering_and_mof_zeolite_organic_bonds(self) -> None:
        from trium_studio.structure_loader import load_structure_from_bytes
        from trium_studio.engine_core import SimulationSession

        # 1. Disordered Zr-MOF CIF (partial-occupancy O1B sits 0.04 nm from O1A -> must be filtered!)
        disordered_mof = """data_Disordered_MOF
_cell_length_a 10.0
_cell_length_b 10.0
_cell_length_c 10.0
_cell_angle_alpha 90.0
_cell_angle_beta  90.0
_cell_angle_gamma 90.0
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_occupancy
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Zr1 Zr 1.00 0.500 0.500 0.500
O1A O  1.00 0.710 0.500 0.500
O1B O  0.50 0.735 0.510 0.500
C1  C  1.00 0.840 0.500 0.500
O2  O  1.00 0.900 0.610 0.500
"""
        sys_mof, _ = load_structure_from_bytes(disordered_mof.encode("utf-8"), "mof.cif")
        # O1B overlapping site must be filtered out -> 4 clean atoms instead of 5
        self.assertEqual(sys_mof.n_atoms, 4)
        # Zr-O (2.1 A), O-C (1.3 A), C-O (1.25 A) bonds must be inferred automatically
        self.assertGreaterEqual(len(sys_mof.bonds), 3)
        # Running MD must stay bounded (no clash explosion!)
        sim = SimulationSession(sys_mof, temperature=300.0, dt=0.001)
        sim.step(10)
        m = sim.get_metrics()
        self.assertTrue(math.isfinite(m["potential_energy"]))

        # 2. Inorganic SiO2 / Zeolite crystal CIF
        sio2_cif = """data_Quartz_mini
_cell_length_a 6.0
_cell_length_b 6.0
_cell_length_c 6.0
_cell_angle_alpha 90.0
_cell_angle_beta  90.0
_cell_angle_gamma 90.0
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Si1 Si 0.500 0.500 0.500
O1  O  0.770 0.500 0.500
O2  O  0.230 0.500 0.500
"""
        sys_sio2, _ = load_structure_from_bytes(sio2_cif.encode("utf-8"), "sio2.cif")
        self.assertGreaterEqual(len(sys_sio2.bonds), 2)

        # 3. Organic chemistry .xyz molecule (water/methanol-like) -> covalent bonds auto-inferred
        xyz_text = """4
Formaldehyde
C  0.000  0.000  0.000
O  1.210  0.000  0.000
H -0.580  0.940  0.000
H -0.580 -0.940  0.000
"""
        sys_org, _ = load_structure_from_bytes(xyz_text.encode("utf-8"), "ch2o.xyz")
        self.assertEqual(len(sys_org.bonds), 3)

    def test_molstar_watermark_logo_hidden_in_studio_css(self) -> None:
        from pathlib import Path
        root = Path(__file__).resolve().parents[2]
        scss = (root / "frontend" / "biology-mm-studio" / "src" / "styles" / "studio.scss").read_text(encoding="utf-8")
        html = (root / "frontend" / "biology-mm-studio" / "index.html").read_text(encoding="utf-8")
        self.assertIn(".msp-viewport-logo", scss)
        self.assertIn(".msp-viewport-logo", html)

    def test_codelab_and_crystal_periodic_integrity(self) -> None:
        from fastapi.testclient import TestClient
        from trium_studio.app import create_app

        app = create_app()
        with TestClient(app) as client:
            # 1. Code Lab executing OpenScienceEngine -> auto-registers live SimulationSession
            code1 = (
                "from biology_mm import OpenScienceEngine, PhysicsParameters, build_alanine_dipeptide\n"
                "eng = OpenScienceEngine(build_alanine_dipeptide(), PhysicsParameters(temperature_K=310.0, dt_ps=0.001))\n"
                "eng.run(10)\n"
            )
            r1 = client.post("/api/code/run", json={"code": code1})
            self.assertEqual(r1.status_code, 200, r1.text)
            d1 = r1.json()
            self.assertTrue(d1["ok"])
            sid = d1["session_id"]
            self.assertEqual(d1["metrics"]["step"], 10)

            # 2. Code Lab mutating the active session in-place using session_id
            code2 = (
                "session.steer(temperature=450.0, efield_z=0.3)\n"
                "session.rcut = 0.90\n"
                "session.step(5)\n"
            )
            r2 = client.post("/api/code/run", json={"code": code2, "session_id": sid})
            self.assertEqual(r2.status_code, 200, r2.text)
            d2 = r2.json()
            self.assertTrue(d2["ok"])
            self.assertEqual(d2["metrics"]["step"], 15)

            # 3. Live physics-params update rebuilds GPU plan and rescales temperature
            r3 = client.post(
                f"/api/sessions/{sid}/physics-params",
                json={"temperature": 500.0, "rcut": 0.85, "sigma_scale": 1.02, "epsilon_scale": 1.05},
            )
            self.assertEqual(r3.status_code, 200, r3.text)
            self.assertAlmostEqual(r3.json()["metrics"]["temperature_K"], 500.0, delta=5.0)

if __name__ == "__main__":
    unittest.main()
