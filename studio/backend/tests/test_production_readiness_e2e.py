"""Exhaustive Production Readiness & Force-Field Integration E2E Audit Suite.

Covers:
  1. All 11 chemistry/structure file formats through POST /api/sessions/upload
     (including real repo files in chemistry/reference/ and chemistry/fixtures/,
     normalized mdparams v1.0.0 JSON, paired AMBER prmtop+inpcrd and CHARMM psf+crd,
     and storage_mode in {auto, tryte-all-atom, full}).
  2. Every REST endpoint and every WebSocket command (all 7 channels including
     multiscale-zone, all 6 palettes, structure.cif, custom-physics, etc.) against
     ALL THREE session engines:
       - SimulationSession
       - CompactSymmetryMultiscaleSession
       - TryteTileFullAtomSession
"""

from __future__ import annotations

import json
import struct
import unittest
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from fastapi.testclient import TestClient

from trium_studio.app import create_app
from trium_studio.engine_core import PHYSICS_CHANNEL_NAMES
from trium_studio.multiscale import pack_tsym_binary
from trium_studio.protocol import (
    MSG_FRAME_FULL,
    MSG_FRAME_TILE_PATCH,
    MSG_OVERLAY_VECTORS,
    MSG_TOPOLOGY_INIT,
    MSG_VOLUME_GRID3D,
)
from trium_studio.structure_loader import (
    AMBER_CHARGE_SCALE,
    load_structure_from_bytes,
    load_structure_pair_from_bytes,
)
from trium_studio.ws_manager import PALETTE_NAMES

REPO_ROOT = Path(__file__).resolve().parents[3]
CHEM_REF_DIR = REPO_ROOT / "chemistry" / "reference"
PROTEIN_LIGAND_DIR = REPO_ROOT / "chemistry" / "fixtures" / "protein_ligand"


SAMPLE_PDB = b"""CRYST1   30.000   30.000   30.000  90.00  90.00  90.00 P 1           1
ATOM      1  O   HOH A   1       1.000   2.000   3.000  1.00  0.00           O
ATOM      2  H1  HOH A   1       1.957   2.000   3.000  1.00  0.00           H
ATOM      3  H2  HOH A   1       0.760   2.927   3.000  1.00  0.00           H
CONECT    1    2    3
END
"""

SAMPLE_CIF = b"""data_test_water
_cell.length_a 30.0
_cell.length_b 30.0
_cell.length_c 30.0
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
ATOM 1 O O1 HOH A 1.0 2.0 3.0
ATOM 2 H H1 HOH A 1.957 2.0 3.0
ATOM 3 H H2 HOH A 0.760 2.927 3.0
"""

SAMPLE_GRO = b"""TIP3P water sample
3
    1WAT     OW    1   0.500   0.500   0.500
    1WAT    HW1    2   0.596   0.500   0.500
    1WAT    HW2    3   0.476   0.593   0.500
   3.00000   3.00000   3.00000
"""

SAMPLE_XYZ = b"""3
Water molecule
O 0.000 0.000 0.000
H 0.957 0.000 0.000
H -0.240 0.927 0.000
"""

SAMPLE_LAMMPS = b"""LAMMPS water data

3 atoms
2 bonds

2 atom types
1 bond types

0.0 30.0 xlo xhi
0.0 30.0 ylo yhi
0.0 30.0 zlo zhi

Atoms # full

1 1 2 -0.834 5.0 5.0 5.0
2 1 3  0.417 5.957 5.0 5.0
3 1 3  0.417 4.760 5.927 5.0

Bonds

1 1 1 2
2 1 1 3
"""

# Exact AMBER TIP3P water prmtop (charges scaled by 18.2223)
_Q_O_AMBER = -0.834 * AMBER_CHARGE_SCALE
_Q_H_AMBER = 0.417 * AMBER_CHARGE_SCALE
SAMPLE_PRMTOP = (
    "%VERSION  VERSION_STAMP = V0001.000  DATE = 11/20/26  12:00:00\n"
    "%FLAG TITLE\n"
    "%FORMAT(20a4)\n"
    "TIP3P_WATER\n"
    "%FLAG POINTERS\n"
    "%FORMAT(10I8)\n"
    "       3       2       2       0       0       0       0       0       0       0\n"
    "       3       1       0       0       0       1       0       0       0       0\n"
    "%FLAG ATOM_NAME\n"
    "%FORMAT(20a4)\n"
    "O   H1  H2  \n"
    "%FLAG CHARGE\n"
    "%FORMAT(5E16.8)\n"
    f"{_Q_O_AMBER:16.8E}{_Q_H_AMBER:16.8E}{_Q_H_AMBER:16.8E}\n"
    "%FLAG ATOMIC_NUMBER\n"
    "%FORMAT(10I8)\n"
    "       8       1       1\n"
    "%FLAG MASS\n"
    "%FORMAT(5E16.8)\n"
    "  1.59994000E+01  1.00800000E+00  1.00800000E+00\n"
    "%FLAG BONDS_INC_HYDROGEN\n"
    "%FORMAT(10I8)\n"
    "       0       3       1       0       6       1\n"
    "%FLAG BONDS_WITHOUT_HYDROGEN\n"
    "%FORMAT(10I8)\n"
    "\n"
    "%FLAG RESIDUE_LABEL\n"
    "%FORMAT(20a4)\n"
    "WAT \n"
    "%FLAG RESIDUE_POINTER\n"
    "%FORMAT(10I8)\n"
    "       1\n"
).encode("utf-8")

SAMPLE_INPCRD = (
    "TIP3P_WATER\n"
    "     3\n"
    "   5.0000000   5.0000000   5.0000000   5.9570000   5.0000000   5.0000000\n"
    "   4.7600000   5.9270000   5.0000000\n"
    "  30.0000000  30.0000000  30.0000000  90.0000000  90.0000000  90.0000000\n"
).encode("utf-8")

SAMPLE_PSF = b"""PSF EXT

       1 !NTITLE
* CHARMM TIP3P WATER

       3 !NATOM
       1 WAT  1    TIP3 OH2  OT    -0.834000       15.9994           0
       2 WAT  1    TIP3 H1   HT     0.417000        1.0080           0
       3 WAT  1    TIP3 H2   HT     0.417000        1.0080           0

       2 !NBOND: bonds
       1       2       1       3
"""

SAMPLE_CRD = b"""* CHARMM Standard CRD
*
    3
    1    1 TIP3 OH2    5.00000   5.00000   5.00000 WAT  1      0.00000
    2    1 TIP3 H1     5.95700   5.00000   5.00000 WAT  1      0.00000
    3    1 TIP3 H2     4.76000   5.92700   5.00000 WAT  1      0.00000
"""

SAMPLE_CRD_EXT = b"""* CHARMM EXT CRD
*
         3  EXT
         1         1  TIP3      OH2              5.0000000000        5.0000000000        5.0000000000  WAT       1               0.0000000000
         2         1  TIP3      H1               5.9570000000        5.0000000000        5.0000000000  WAT       1               0.0000000000
         3         1  TIP3      H2               4.7600000000        5.9270000000        5.0000000000  WAT       1               0.0000000000
"""

SAMPLE_MOL2 = b"""@<TRIPOS>MOLECULE
ethanol_ligand
 4 3 0 0 0
SMALL
GASTEIGER

@<TRIPOS>ATOM
      1 C1          0.0000    0.0000    0.0000 C.3     1 LIG        -0.0650
      2 C2          1.5200    0.0000    0.0000 C.3     1 LIG         0.1450
      3 O1          2.0500    1.3200    0.0000 O.3     1 LIG        -0.3980
      4 H1          3.0100    1.3200    0.0000 H       1 LIG         0.3180
@<TRIPOS>BOND
     1     1     2    1
     2     2     3    1
     3     3     4    1
"""

SAMPLE_SDF = b"""aspirin_frag
  BiologyMM 3D

  4  3  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.5000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    2.2500    1.2000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
    2.2500   -1.2000    0.0000 O   0  5  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  2  3  2  0  0  0  0
  2  4  1  0  0  0  0
M  CHG  1   4  -1
M  END
$$$$
"""

SAMPLE_NORMALIZED_JSON = json.dumps({
    "schema": "1.0.0",
    "family": "amber",
    "units": "nm-kj-mol-e-amu",
    "provenance": {
        "source": "unit_test.prmtop",
        "force_field": "TIP3P water",
    },
    "atoms": [
        {"name": "O", "mass": 15.9994, "charge": -0.834, "element": "O", "residue_index": 1, "residue_name": "WAT"},
        {"name": "H1", "mass": 1.008, "charge": 0.417, "element": "H", "residue_index": 1, "residue_name": "WAT"},
        {"name": "H2", "mass": 1.008, "charge": 0.417, "element": "H", "residue_index": 1, "residue_name": "WAT"},
    ],
    "box": {"kind": "orthorhombic", "lengths_nm": [3.2, 3.1, 3.0]},
    "bonds": [
        {"i": 1, "j": 2, "k": 418400.0, "r0_nm": 0.09572},
        {"i": 1, "j": 3, "k": 418400.0, "r0_nm": 0.09572},
    ],
    "coordinates_nm": [
        [0.500, 0.500, 0.500],
        [0.596, 0.500, 0.500],
        [0.476, 0.593, 0.500],
    ],
    "net_charge": 0.0,
}).encode("utf-8")


def _make_tsym_assembly(n_asu: int = 12, n_transforms: int = 8) -> bytes:
    """Build a small valid TSYM biological assembly packet."""
    rng = np.random.default_rng(42)
    asu_coords = rng.uniform(0.0, 4.0, size=(n_asu, 3)).astype(np.float32)
    asu_elems = ["N", "C", "O", "C", "H", "S"] * ((n_asu + 5) // 6)
    asu_elems = asu_elems[:n_asu]

    transforms = np.zeros((n_transforms, 4, 4), dtype=np.float32)
    for k in range(n_transforms):
        transforms[k] = np.eye(4, dtype=np.float32)
        transforms[k, 0, 3] = float((k % 4) * 10.0)
        transforms[k, 1, 3] = float((k // 4) * 10.0)
        transforms[k, 2, 3] = 0.0

    return pack_tsym_binary(
        asu_coords_ang=asu_coords,
        asu_elements=asu_elems,
        transforms_4x4=transforms,
        title="AuditSymmetricAssembly",
        mode="metal_verlet",
        temperature=300.0,
        dt=0.002,
    )


def _ws_drain_until_json_type(ws: Any, target_type: str, max_reads: int = 30) -> Dict[str, Any]:
    """Read WS frames (skipping binary/unrelated JSON) until a JSON frame with matching `type` arrives."""
    for _ in range(max_reads):
        msg = ws.receive()
        if "text" in msg and msg["text"] is not None:
            data = json.loads(msg["text"])
            if data.get("type") == "error":
                raise AssertionError(f"WS returned error while awaiting '{target_type}': {data}")
            if data.get("type") == target_type:
                return data
    raise AssertionError(f"Timed out waiting for WS JSON type='{target_type}'")


def _packet_msg_type(raw: bytes) -> int:
    assert len(raw) >= 40 and raw[:4] == b"TRM1", f"Invalid TriumStream packet header: {raw[:8]!r}"
    return int(struct.unpack_from("<H", raw, 4)[0])


def _ws_drain_until_binary_msg(ws: Any, expected_msg_types: Tuple[int, ...], max_reads: int = 30) -> bytes:
    """Read WS frames until a binary packet whose TriumStream msg_type is in expected_msg_types arrives."""
    for _ in range(max_reads):
        msg = ws.receive()
        if "bytes" in msg and msg["bytes"] is not None:
            raw = msg["bytes"]
            if len(raw) >= 40 and _packet_msg_type(raw) in expected_msg_types:
                return raw
        elif "text" in msg and msg["text"] is not None:
            data = json.loads(msg["text"])
            if data.get("type") == "error":
                raise AssertionError(f"WS returned error while awaiting binary {expected_msg_types}: {data}")
    raise AssertionError(f"Timed out waiting for WS binary types {expected_msg_types}")


class TestAllElevenFormatsUpload(unittest.TestCase):
    """Test all 11 structure/topology formats through structure_loader AND POST /api/sessions/upload."""

    def setUp(self) -> None:
        self.app = create_app()
        self.client = TestClient(self.app)

    def tearDown(self) -> None:
        self.app.state.session_manager.close_all()

    def test_unit_parsers_exact_charges_masses_bonds(self) -> None:
        # 1. Normalized JSON
        sys_j, meta_j = load_structure_from_bytes(SAMPLE_NORMALIZED_JSON, "tip3p.json")
        self.assertEqual(sys_j.n_atoms, 3)
        self.assertEqual(len(sys_j.bonds), 2)
        np.testing.assert_allclose(sys_j.charges, [-0.834, 0.417, 0.417], atol=1e-6)
        np.testing.assert_allclose(sys_j.masses, [15.9994, 1.008, 1.008], atol=1e-4)
        self.assertFalse(meta_j["coords_synthesized"])

        # 2. AMBER prmtop standalone (charge / 18.2223, bonds idx//3, synthesized coords)
        sys_p, meta_p = load_structure_from_bytes(SAMPLE_PRMTOP, "water.prmtop")
        self.assertEqual(sys_p.n_atoms, 3)
        self.assertEqual(sys_p.bonds, [(0, 1), (0, 2)])
        np.testing.assert_allclose(sys_p.charges, [-0.834, 0.417, 0.417], atol=1e-5)
        np.testing.assert_allclose(sys_p.masses, [15.9994, 1.008, 1.008], atol=1e-4)
        self.assertTrue(meta_p["coords_synthesized"])

        # 3. AMBER prmtop + inpcrd pair
        sys_ap, meta_ap = load_structure_pair_from_bytes(SAMPLE_PRMTOP, "water.prmtop", SAMPLE_INPCRD, "water.inpcrd")
        self.assertEqual(sys_ap.n_atoms, 3)
        self.assertEqual(sys_ap.bonds, [(0, 1), (0, 2)])
        np.testing.assert_allclose(sys_ap.charges, [-0.834, 0.417, 0.417], atol=1e-5)
        self.assertFalse(meta_ap["coords_synthesized"])

        # 4. CHARMM psf standalone + paired with crd (standard and EXT)
        sys_psf, meta_psf = load_structure_from_bytes(SAMPLE_PSF, "water.psf")
        self.assertEqual(sys_psf.n_atoms, 3)
        self.assertEqual(sys_psf.bonds, [(0, 1), (0, 2)])
        np.testing.assert_allclose(sys_psf.charges, [-0.834, 0.417, 0.417], atol=1e-6)
        self.assertTrue(meta_psf["coords_synthesized"])

        sys_cp, meta_cp = load_structure_pair_from_bytes(SAMPLE_PSF, "water.psf", SAMPLE_CRD_EXT, "water.crd")
        self.assertEqual(sys_cp.n_atoms, 3)
        self.assertEqual(sys_cp.bonds, [(0, 1), (0, 2)])
        np.testing.assert_allclose(sys_cp.charges, [-0.834, 0.417, 0.417], atol=1e-6)
        self.assertFalse(meta_cp["coords_synthesized"])

        # 5. MOL2 partial charges & bonds
        sys_m2, _ = load_structure_from_bytes(SAMPLE_MOL2, "ethanol.mol2")
        self.assertEqual(sys_m2.n_atoms, 4)
        self.assertEqual(sys_m2.bonds, [(0, 1), (1, 2), (2, 3)])
        np.testing.assert_allclose(sys_m2.charges, [-0.065, 0.145, -0.398, 0.318], atol=1e-6)

        # 6. SDF V2000 bonds & M CHG formal charges
        sys_sdf, _ = load_structure_from_bytes(SAMPLE_SDF, "ligand.sdf")
        self.assertEqual(sys_sdf.n_atoms, 4)
        self.assertEqual(sys_sdf.bonds, [(0, 1), (1, 2), (1, 3)])
        np.testing.assert_allclose(sys_sdf.charges, [0.0, 0.0, 0.0, -1.0], atol=1e-6)

    def test_upload_all_11_formats_via_rest(self) -> None:
        single_cases: List[Tuple[str, bytes, int]] = [
            ("water.pdb", SAMPLE_PDB, 3),
            ("water.cif", SAMPLE_CIF, 3),
            ("water.gro", SAMPLE_GRO, 3),
            ("water.xyz", SAMPLE_XYZ, 3),
            ("water.data", SAMPLE_LAMMPS, 3),
            ("water.prmtop", SAMPLE_PRMTOP, 3),
            ("water.inpcrd", SAMPLE_INPCRD, 3),
            ("water.psf", SAMPLE_PSF, 3),
            ("water.crd", SAMPLE_CRD, 3),
            ("water_ext.cor", SAMPLE_CRD_EXT, 3),
            ("ethanol.mol2", SAMPLE_MOL2, 4),
            ("ligand.sdf", SAMPLE_SDF, 4),
            ("normalized_tip3p.json", SAMPLE_NORMALIZED_JSON, 3),
        ]
        for fname, content, expected_n in single_cases:
            with self.subTest(filename=fname):
                res = self.client.post(f"/api/sessions/upload?filename={fname}", content=content)
                self.assertEqual(res.status_code, 200, f"Upload failed for {fname}: {res.text}")
                data = res.json()
                self.assertEqual(data["num_atoms"], expected_n)
                self.assertIn("session_id", data)
                self.assertIn("metrics", data)

    def test_upload_real_repo_chemistry_reference_files_and_fixtures(self) -> None:
        ref_files = [
            ("water_ions.json", 2164),
            ("dna.json", 628),
            ("protein_ligand.json", 2615),
            ("charmm_adapter.json", 8),
        ]
        for fname, expected_n in ref_files:
            fpath = CHEM_REF_DIR / fname
            self.assertTrue(fpath.is_file(), f"Missing repo file {fpath}")
            with self.subTest(reference_file=fname):
                res = self.client.post(
                    f"/api/sessions/upload?filename={fname}&storage_mode=tryte-all-atom",
                    content=fpath.read_bytes(),
                )
                self.assertEqual(res.status_code, 200, f"Upload failed for {fname}: {res.text}")
                data = res.json()
                self.assertEqual(data["num_atoms"], expected_n)
                self.assertEqual(data["storage_mode"], "tryte-all-atom")

        # Real AMBER protein_ligand.prmtop + protein_ligand.inpcrd pair via multipart upload
        prmtop_bytes = (PROTEIN_LIGAND_DIR / "protein_ligand.prmtop").read_bytes()
        inpcrd_bytes = (PROTEIN_LIGAND_DIR / "protein_ligand.inpcrd").read_bytes()
        res_pair = self.client.post(
            "/api/sessions/upload?storage_mode=tryte-all-atom",
            files=[
                ("files", ("protein_ligand.prmtop", prmtop_bytes, "application/octet-stream")),
                ("files", ("protein_ligand.inpcrd", inpcrd_bytes, "application/octet-stream")),
            ],
        )
        self.assertEqual(res_pair.status_code, 200, res_pair.text)
        pdata = res_pair.json()
        self.assertEqual(pdata["num_atoms"], 2615)
        self.assertGreater(pdata["num_bonds"], 2000)
        self.assertFalse(pdata["coords_synthesized"])
        self.assertEqual(pdata["storage_mode"], "tryte-all-atom")

    def test_upload_paired_amber_charmm_and_json_f32_across_storage_modes(self) -> None:
        # Paired CHARMM .psf + .crd across auto, tryte-all-atom, full
        for smode in ("auto", "tryte-all-atom", "full"):
            with self.subTest(pair="charmm", storage_mode=smode):
                res = self.client.post(
                    f"/api/sessions/upload?storage_mode={smode}",
                    files=[
                        ("files", ("water.psf", SAMPLE_PSF, "text/plain")),
                        ("files", ("water.crd", SAMPLE_CRD, "text/plain")),
                    ],
                )
                self.assertEqual(res.status_code, 200, res.text)
                d = res.json()
                self.assertEqual(d["num_atoms"], 3)
                self.assertEqual(d["num_bonds"], 2)
                self.assertFalse(d["coords_synthesized"])

        # Normalized JSON without coordinates + companion .f32 float32 coords
        json_no_coords = json.loads(SAMPLE_NORMALIZED_JSON.decode("utf-8"))
        del json_no_coords["coordinates_nm"]
        coords_f32 = np.array(
            [[0.5, 0.5, 0.5], [0.596, 0.5, 0.5], [0.476, 0.593, 0.5]],
            dtype="<f4",
        ).tobytes()
        res_jf = self.client.post(
            "/api/sessions/upload?storage_mode=auto",
            files=[
                ("files", ("tip3p.json", json.dumps(json_no_coords).encode("utf-8"), "application/json")),
                ("files", ("tip3p.f32", coords_f32, "application/octet-stream")),
            ],
        )
        self.assertEqual(res_jf.status_code, 200, res_jf.text)
        self.assertEqual(res_jf.json()["num_atoms"], 3)
        self.assertFalse(res_jf.json()["coords_synthesized"])


class TestAllThreeEnginesFullRestAndWsAudit(unittest.TestCase):
    """Test EVERY REST endpoint and EVERY WebSocket command against ALL THREE engines."""

    def setUp(self) -> None:
        self.app = create_app()
        self.client = TestClient(self.app)

    def tearDown(self) -> None:
        self.app.state.session_manager.close_all()

    def _create_three_engine_sessions(self) -> List[Tuple[str, str]]:
        sessions: List[Tuple[str, str]] = []

        # 1. SimulationSession (float64-full via upload auto)
        r1 = self.client.post("/api/sessions/upload?filename=ethanol.mol2&storage_mode=auto", content=SAMPLE_MOL2)
        self.assertEqual(r1.status_code, 200, r1.text)
        sessions.append(("SimulationSession", r1.json()["session_id"]))

        # 2. CompactSymmetryMultiscaleSession (symmetry-compact via TSYM)
        tsym_bytes = _make_tsym_assembly(n_asu=12, n_transforms=8)
        r2 = self.client.post(
            "/api/sessions/create-from-symmetry?storage_mode=symmetry-compact",
            content=tsym_bytes,
        )
        self.assertEqual(r2.status_code, 200, r2.text)
        self.assertEqual(r2.json()["storage_mode"], "symmetry-compact")
        sessions.append(("CompactSymmetryMultiscaleSession", r2.json()["session_id"]))

        # 3. TryteTileFullAtomSession (tryte-all-atom via TSYM)
        r3 = self.client.post(
            "/api/sessions/create-from-symmetry?storage_mode=tryte-all-atom",
            content=tsym_bytes,
        )
        self.assertEqual(r3.status_code, 200, r3.text)
        self.assertEqual(r3.json()["storage_mode"], "tryte-all-atom")
        sessions.append(("TryteTileFullAtomSession", r3.json()["session_id"]))

        return sessions

    def _audit_all_rest_endpoints(self, engine_name: str, sid: str) -> None:
        # 1. GET /metrics
        r = self.client.get(f"/api/sessions/{sid}/metrics")
        self.assertEqual(r.status_code, 200, f"{engine_name} GET metrics: {r.text}")
        m0 = r.json()
        self.assertEqual(m0["session_id"], sid)

        # 2. POST /step
        r = self.client.post(f"/api/sessions/{sid}/step", json={"n_steps": 2})
        self.assertEqual(r.status_code, 200, f"{engine_name} POST step: {r.text}")
        step_after = r.json()["step"]
        self.assertGreaterEqual(step_after, 2)

        # 3. POST /steer
        r = self.client.post(
            f"/api/sessions/{sid}/steer",
            json={
                "temperature": 310.0,
                "efield_z": 0.05,
                "active_channel": "multiscale-zone",
                "palette": "ternary-triad",
            },
        )
        self.assertEqual(r.status_code, 200, f"{engine_name} POST steer: {r.text}")

        # 4. POST /scrub
        r = self.client.post(f"/api/sessions/{sid}/scrub", json={"target_step": 0})
        self.assertEqual(r.status_code, 200, f"{engine_name} POST scrub: {r.text}")

        # 5. POST /relax
        r = self.client.post(f"/api/sessions/{sid}/relax", json={"max_steps": 5})
        self.assertEqual(r.status_code, 200, f"{engine_name} POST relax: {r.text}")

        # 6. POST /multiscale
        r = self.client.post(
            f"/api/sessions/{sid}/multiscale",
            json={"center_ang": [5.0, 5.0, 5.0], "aa_radius_ang": 15.0, "cg_radius_ang": 30.0},
        )
        self.assertEqual(r.status_code, 200, f"{engine_name} POST multiscale: {r.text}")

        # 7. POST /steer-atom (set + clear)
        r = self.client.post(
            f"/api/sessions/{sid}/steer-atom",
            json={"atom_index": 0, "target_ang": [6.0, 6.0, 6.0], "k_spring": 250.0},
        )
        self.assertEqual(r.status_code, 200, f"{engine_name} POST steer-atom set: {r.text}")
        r = self.client.post(f"/api/sessions/{sid}/steer-atom", json={"atom_index": None})
        self.assertEqual(r.status_code, 200, f"{engine_name} POST steer-atom clear: {r.text}")

        # 8. GET /atom/0
        r = self.client.get(f"/api/sessions/{sid}/atom/0")
        self.assertEqual(r.status_code, 200, f"{engine_name} GET atom/0: {r.text}")
        atom0 = r.json()
        self.assertIn("element", atom0)
        self.assertIn("position_ang", atom0)

        # 9. GET /structure.cif (default + explicit channels)
        for ch in (None, "force-magnitude", "multiscale-zone", "charge-potential"):
            url = f"/api/sessions/{sid}/structure.cif" + (f"?channel={ch}" if ch else "")
            r = self.client.get(url)
            self.assertEqual(r.status_code, 200, f"{engine_name} GET structure.cif ({ch}): {r.text}")
            self.assertIn("data_biology_mm", r.text)
            self.assertIn("_atom_site.Cartn_x", r.text)

        # 10. POST /custom-physics
        r = self.client.post(
            f"/api/sessions/{sid}/custom-physics",
            json={
                "potential_expr": "4 * eps * ((sig / r)**12 - (sig / r)**6)",
                "r_min": 0.22,
                "r_max": 1.0,
                "n_points": 64,
                "temperature": 300.0,
            },
        )
        self.assertEqual(r.status_code, 200, f"{engine_name} POST custom-physics: {r.text}")
        self.assertTrue(r.json().get("ok"))
        r_after_cp = self.client.post(f"/api/sessions/{sid}/step", json={"n_steps": 1})
        self.assertEqual(r_after_cp.status_code, 200, f"{engine_name} step after custom-physics: {r_after_cp.text}")

    def _audit_all_ws_commands(self, engine_name: str, sid: str) -> None:
        with self.client.websocket_connect(f"/ws/stream/{sid}") as ws:
            # Initial handshake: topology, frame, overlays, metrics, state
            topo = ws.receive_bytes()
            self.assertEqual(_packet_msg_type(topo), MSG_TOPOLOGY_INIT, f"{engine_name} initial topology")
            frame0 = ws.receive_bytes()
            self.assertIn(_packet_msg_type(frame0), (MSG_FRAME_FULL, MSG_FRAME_TILE_PATCH), f"{engine_name} initial frame")
            ov0 = ws.receive_bytes()
            self.assertEqual(_packet_msg_type(ov0), MSG_OVERLAY_VECTORS, f"{engine_name} initial overlay")
            m0 = ws.receive_json()
            self.assertEqual(m0["type"], "metrics")
            st0 = ws.receive_json()
            self.assertEqual(st0["type"], "state")

            # play & pause
            ws.send_json({"type": "play"})
            st_play = _ws_drain_until_json_type(ws, "state")
            self.assertTrue(st_play["playing"], f"{engine_name} play")

            ws.send_json({"type": "pause"})
            st_pause = _ws_drain_until_json_type(ws, "state")
            self.assertFalse(st_pause["playing"], f"{engine_name} pause")

            # step
            ws.send_json({"type": "step", "n_steps": 2})
            m_step = _ws_drain_until_json_type(ws, "metrics")
            self.assertGreaterEqual(m_step["metrics"]["step"], 1)

            # scrub
            ws.send_json({"type": "scrub", "target_step": 0})
            _ws_drain_until_json_type(ws, "metrics")

            # steer
            ws.send_json({"type": "steer", "temperature": 305.0, "efield_z": 0.02})
            _ws_drain_until_json_type(ws, "metrics")

            # set_channel across ALL 7 channels (including multiscale-zone)
            self.assertEqual(len(PHYSICS_CHANNEL_NAMES), 7)
            self.assertIn("multiscale-zone", PHYSICS_CHANNEL_NAMES)
            for ch in PHYSICS_CHANNEL_NAMES:
                ws.send_json({"type": "set_channel", "channel": ch})
                st_ch = _ws_drain_until_json_type(ws, "state")
                self.assertEqual(st_ch["active_channel"], ch, f"{engine_name} set_channel {ch}")

            # set_palette across ALL 6 palettes
            self.assertEqual(len(PALETTE_NAMES), 6)
            for pal in PALETTE_NAMES:
                ws.send_json({"type": "set_palette", "palette": pal})
                st_pal = _ws_drain_until_json_type(ws, "state")
                self.assertEqual(st_pal["palette"], pal, f"{engine_name} set_palette {pal}")

            # request_volume
            ws.send_json({"type": "request_volume", "ml": 9, "enabled": False})
            vol_pkt = _ws_drain_until_binary_msg(ws, (MSG_VOLUME_GRID3D,))
            self.assertEqual(_packet_msg_type(vol_pkt), MSG_VOLUME_GRID3D)

            # request_overlays
            ws.send_json({"type": "request_overlays", "max_vectors": 100, "enabled": False})
            ov_pkt = _ws_drain_until_binary_msg(ws, (MSG_OVERLAY_VECTORS,))
            self.assertEqual(_packet_msg_type(ov_pkt), MSG_OVERLAY_VECTORS)

            # camera_roi (set + clear)
            ws.send_json({"type": "camera_roi", "center_ang": [5.0, 5.0, 5.0], "radius_ang": 20.0, "max_atoms": 1000})
            _ws_drain_until_json_type(ws, "metrics")
            ws.send_json({"type": "camera_roi", "center_ang": None})
            _ws_drain_until_json_type(ws, "metrics")

            # inspect_atom
            ws.send_json({"type": "inspect_atom", "index": 0})
            insp = _ws_drain_until_json_type(ws, "atom_inspection")
            self.assertIn("element", insp["atom"])

            # set_focus_zone
            ws.send_json({"type": "set_focus_zone", "center_ang": [5.0, 5.0, 5.0], "aa_radius_ang": 15.0, "cg_radius_ang": 30.0})
            ms_res = _ws_drain_until_json_type(ws, "multiscale")
            self.assertIn("multiscale", ms_res)

            # relax_clashes
            ws.send_json({"type": "relax_clashes", "max_steps": 5})
            rel_res = _ws_drain_until_json_type(ws, "relax_result")
            self.assertIn("relaxation", rel_res)

            # steer_atom (set + clear)
            ws.send_json({"type": "steer_atom", "atom_index": 0, "target_ang": [6.0, 5.0, 5.0], "k_spring": 200.0})
            smd_res = _ws_drain_until_json_type(ws, "smd_state")
            self.assertTrue(smd_res["smd"]["active"])
            ws.send_json({"type": "steer_atom", "atom_index": None})
            smd_clr = _ws_drain_until_json_type(ws, "smd_state")
            self.assertFalse(smd_clr["smd"]["active"])

            # set_delta_threshold
            ws.send_json({"type": "set_delta_threshold", "threshold_ang": 0.08, "enabled": True})
            dt_res = _ws_drain_until_json_type(ws, "delta_threshold")
            self.assertAlmostEqual(dt_res["threshold_ang"], 0.08, places=4)

    def test_exhaustive_rest_and_ws_across_all_three_engines(self) -> None:
        # Global endpoints
        self.assertEqual(self.client.get("/api/health").status_code, 200)
        self.assertEqual(self.client.get("/api/capabilities").status_code, 200)
        self.assertEqual(self.client.get("/api/presets").status_code, 200)

        for engine_name, sid in self._create_three_engine_sessions():
            with self.subTest(engine=engine_name, phase="REST"):
                self._audit_all_rest_endpoints(engine_name, sid)
            with self.subTest(engine=engine_name, phase="WS"):
                self._audit_all_ws_commands(engine_name, sid)


    def test_tryte_cif_bonds_subset_elements_and_multiscale_counts(self) -> None:
        # 1. Verify TryteTileFullAtomSession exports _struct_conn bonds in structure.cif
        r = self.client.post(
            "/api/sessions/upload?filename=ethanol.mol2&storage_mode=tryte-all-atom",
            content=SAMPLE_MOL2,
        )
        self.assertEqual(r.status_code, 200, r.text)
        sid = r.json()["session_id"]
        ms = r.json()["metrics"]["multiscale"]
        self.assertIn("counts", ms)
        self.assertIn("all_atom", ms["counts"])
        self.assertIn("coarse_grained", ms["counts"])
        self.assertIn("rigid_scaffold", ms["counts"])

        cif_res = self.client.get(f"/api/sessions/{sid}/structure.cif?channel=charge-potential")
        self.assertEqual(cif_res.status_code, 200, cif_res.text)
        self.assertIn("_struct_conn.conn_type_id", cif_res.text)
        self.assertIn("covale1", cif_res.text)

        # 2. Verify TryteTileFullAtomSession cif_builder slices atomic_numbers to _stream_atom_indices
        from trium_studio.cif_builder import build_mmcif_from_session
        from trium_studio.tryte_tiles import TryteTileFullAtomSession

        # 40 atoms: first 20 H, last 20 O; stream only 10 atoms
        pos = np.zeros((40, 3), dtype=np.float32)
        for i in range(40):
            pos[i] = [0.3 * i, 0.0, 0.0]
        z = np.array([1] * 20 + [8] * 20, dtype=np.uint8)
        sim = TryteTileFullAtomSession(
            positions_nm=pos,
            atomic_numbers=z,
            max_stream_atoms=10,
            focus_center_ang=(float(pos[-1, 0] * 10.0), 0.0, 0.0),
            aa_radius_ang=15.0,
            cg_radius_ang=30.0,
        )
        cif_sub = build_mmcif_from_session(sim, channel="multiscale-zone")
        # All streamed atoms around the focus at the O-end must have element O matching sim.names
        atom_lines = [ln for ln in cif_sub.splitlines() if ln.startswith(("ATOM ", "HETATM "))]
        self.assertEqual(len(atom_lines), len(sim.names))
        for ln, expected_el in zip(atom_lines, sim.names):
            cif_el = ln.split()[2]
            self.assertEqual(cif_el, expected_el)
        sim.close()

        # 3. Verify CompactSymmetryMultiscaleSession also exposes multiscale.counts
        tsym_bytes = _make_tsym_assembly(n_asu=12, n_transforms=8)
        r_sc = self.client.post(
            "/api/sessions/create-from-symmetry?storage_mode=symmetry-compact",
            content=tsym_bytes,
        )
        self.assertEqual(r_sc.status_code, 200, r_sc.text)
        ms_sc = r_sc.json()["metrics"]["multiscale"]
        self.assertIn("counts", ms_sc)
        self.assertIn("all_atom", ms_sc["counts"])
        self.assertIn("coarse_grained", ms_sc["counts"])
        self.assertIn("rigid_scaffold", ms_sc["counts"])

class TestBackendReliabilityRegressions(unittest.TestCase):
    """Regression tests for hollow-capsid focus, startup clash, RCSB proxy, and WS worker."""

    def setUp(self) -> None:
        self.app = create_app()
        self.client = TestClient(self.app)

    def tearDown(self) -> None:
        self.app.state.session_manager.close_all()

    def test_hollow_capsid_auto_centers_on_wall_and_exposes_box_origin(self) -> None:
        from trium_studio.tryte_tiles import TryteTileFullAtomSession

        # Build a 5J7V-like hollow spherical capsid shell (R = 20 nm = 200 A, empty center at 0,0,0)
        n = 2700
        idx = np.arange(n, dtype=np.float64)
        phi = np.arccos(1.0 - 2.0 * (idx + 0.5) / n)
        theta = np.pi * (1.0 + 5.0 ** 0.5) * idx
        r_nm = 20.0
        pos_nm = np.stack(
            [
                r_nm * np.sin(phi) * np.cos(theta),
                r_nm * np.sin(phi) * np.sin(theta),
                r_nm * np.cos(phi),
            ],
            axis=1,
        ).astype(np.float32)
        z = np.full(n, 6, dtype=np.uint8)

        sim = TryteTileFullAtomSession(
            positions_nm=pos_nm,
            atomic_numbers=z,
            max_active_aa_atoms=540,
            aa_radius_ang=45.0,
            cg_radius_ang=85.0,
        )
        try:
            metrics = sim.get_metrics()
            ms = metrics["multiscale"]
            # Must auto-center on the capsid wall (first surface tile centroid), NOT empty (0, 0, 0)
            expected_ctr = sim.tile_set.centroids_nm[0].astype(np.float64) * 10.0
            np.testing.assert_allclose(ms["center_ang"], expected_ctr, atol=1e-3)
            self.assertGreater(float(np.linalg.norm(ms["center_ang"])), 150.0)

            # Both All-Atom and Coarse-Grained zones must be populated (> 0)
            self.assertGreater(ms["counts"]["all_atom"], 0)
            self.assertGreater(ms["counts"]["coarse_grained"], 0)
            self.assertGreater(ms["n_all_atom"], 0)
            self.assertGreater(ms["n_coarse_grained"], 0)

            # box_origin_ang must be exposed and match min_xyz (with the 9 A box margin)
            self.assertIn("box_origin_ang", metrics)
            expected_origin = (np.min(sim.positions_nm, axis=0).astype(np.float64) - 0.9) * 10.0
            np.testing.assert_allclose(metrics["box_origin_ang"], expected_origin, atol=1e-2)
            # Verify origin is negative (around -209 A), not (0, 0, 0)
            self.assertLess(metrics["box_origin_ang"][0], -150.0)
        finally:
            sim.close()

        # Also verify box_origin_ang is exposed on CompactSymmetryMultiscaleSession and SimulationSession
        tsym_bytes = _make_tsym_assembly(n_asu=12, n_transforms=8)
        r_sc = self.client.post(
            "/api/sessions/create-from-symmetry?storage_mode=symmetry-compact",
            content=tsym_bytes,
        )
        self.assertEqual(r_sc.status_code, 200, r_sc.text)
        self.assertEqual(len(r_sc.json()["metrics"]["box_origin_ang"]), 3)

        r_sim = self.client.post("/api/sessions/upload?filename=ethanol.mol2&storage_mode=auto", content=SAMPLE_MOL2)
        self.assertEqual(r_sim.status_code, 200, r_sim.text)
        self.assertEqual(len(r_sim.json()["metrics"]["box_origin_ang"]), 3)

    def test_biomolecular_complex_preset_boots_without_clash(self) -> None:
        res = self.client.post("/api/sessions", json={"preset_id": "biomolecular-complex"})
        self.assertEqual(res.status_code, 200, res.text)
        metrics = res.json()["metrics"]
        self.assertNotEqual(metrics["health_status"], "clash")
        self.assertEqual(metrics["health_status"], "bounded")

    def test_rcsb_proxy_validation_and_cors_headers(self) -> None:
        from unittest.mock import MagicMock, patch
        import urllib.error

        # Invalid PDB IDs must return HTTP 400 cleanly
        for bad_id in ("BAD!ID", "AB", "1234567890123"):
            r_bad = self.client.get(f"/api/structures/fetch/{bad_id}")
            self.assertEqual(r_bad.status_code, 400, f"Expected 400 for {bad_id}, got {r_bad.status_code}")

        # Invalid format returns HTTP 400
        r_fmt = self.client.get("/api/structures/fetch/3J3Q?fmt=xyz")
        self.assertEqual(r_fmt.status_code, 400)

        # Mocked successful fetch returns 200 + Access-Control-Allow-Origin: *
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"data_3J3Q\n"
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.__exit__.return_value = False
        with patch("urllib.request.urlopen", return_value=mock_resp) as m_open:
            r_ok = self.client.get("/api/structures/fetch/3J3Q?fmt=cif")
            self.assertEqual(r_ok.status_code, 200, r_ok.text)
            self.assertEqual(r_ok.headers.get("access-control-allow-origin"), "*")
            self.assertIn(b"data_3J3Q", r_ok.content)
            req_arg = m_open.call_args[0][0]
            self.assertEqual(req_arg.full_url, "https://files.rcsb.org/download/3J3Q.cif")

        with patch("urllib.request.urlopen", return_value=mock_resp) as m_open:
            r_bcif = self.client.get("/api/structures/fetch/3J3Q?fmt=bcif")
            self.assertEqual(r_bcif.status_code, 200, r_bcif.text)
            self.assertEqual(r_bcif.headers.get("access-control-allow-origin"), "*")
            req_arg = m_open.call_args[0][0]
            self.assertEqual(req_arg.full_url, "https://models.rcsb.org/v1/3J3Q/full?encoding=bcif")

        # Upstream error -> 502, timeout -> 504
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("upstream down")):
            r_502 = self.client.get("/api/structures/fetch/3J3Q")
            self.assertEqual(r_502.status_code, 502)

        with patch("urllib.request.urlopen", side_effect=TimeoutError("timed out")):
            r_504 = self.client.get("/api/structures/fetch/3J3Q")
            self.assertEqual(r_504.status_code, 504)

    def test_ws_manager_adaptive_steps_and_worker_error_surfacing(self) -> None:
        from types import SimpleNamespace
        from trium_studio.ws_manager import SessionManager

        # Adaptive step counts for >500k, >50k, and smaller metal_verlet / non-verlet systems
        self.assertEqual(SessionManager._steps_per_tick(SimpleNamespace(n_atoms=40_000_000, mode="metal_verlet")), 1)
        self.assertEqual(SessionManager._steps_per_tick(SimpleNamespace(n_atoms=500_001, mode="metal_verlet")), 1)
        self.assertEqual(SessionManager._steps_per_tick(SimpleNamespace(n_atoms=100_000, mode="metal_verlet")), 2)
        self.assertEqual(SessionManager._steps_per_tick(SimpleNamespace(n_atoms=1_000, mode="metal_verlet")), 10)
        self.assertEqual(SessionManager._steps_per_tick(SimpleNamespace(n_atoms=1_000, mode="cnt_tersoff")), 2)

        # Worker error surfacing via ManagedSession.last_error -> WS {"type": "error", "error": ...}
        r = self.client.post("/api/sessions/upload?filename=ethanol.mol2&storage_mode=auto", content=SAMPLE_MOL2)
        self.assertEqual(r.status_code, 200, r.text)
        sid = r.json()["session_id"]
        record = self.app.state.session_manager.get_session(sid)
        self.assertIsNotNone(record)

        def _failing_step(*_args: Any, **_kwargs: Any) -> Dict[str, Any]:
            raise RuntimeError("synthetic worker boom")

        record.session.step = _failing_step  # type: ignore[method-assign]

        with self.client.websocket_connect(f"/ws/stream/{sid}") as ws:
            # Drain initial handshake (3 binary + 2 json)
            ws.receive_bytes()
            ws.receive_bytes()
            ws.receive_bytes()
            ws.receive_json()
            ws.receive_json()

            ws.send_json({"type": "play"})
            saw_error = None
            for _ in range(30):
                msg = ws.receive()
                if "text" in msg and msg["text"] is not None:
                    data = json.loads(msg["text"])
                    if data.get("type") == "error":
                        saw_error = data
                        break
            self.assertIsNotNone(saw_error, "Expected worker exception to be surfaced over WS")
            self.assertIn("synthetic worker boom", saw_error["error"])
            self.assertFalse(record.playing)


if __name__ == "__main__":
    unittest.main()
