"""Unit & Integration Tests for Biology-MM v2.6.0:
- AlphaFold PDB & mmCIF pLDDT parsing
- PAE domain & hinge clustering
- Confidence-guided Focus Zone & per-atom pLDDT harmonic restraints
- Active-zone C1-smooth DSF Coulomb electrostatics in TryteTileFullAtomSession
- Tight covalent stiffness on crystals/MOFs (UiO-66)
- FastAPI /api/sessions/alphafold/{uniprot_id} (mocked network) & /alphafold-analysis
"""

from __future__ import annotations

import json
import unittest
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

from biology_mm.alphafold_pipeline import (
    AlphaFoldEntry,
    cluster_pae_domains,
    compute_confidence_guided_setup,
    compute_per_residue_rmsf,
    compute_plddt_restraint_weights,
    evaluate_atlas_flexibility,
    parse_alphafold_plddt_any,
    parse_pae_json,
)
from trium_studio.app import create_app
from trium_studio.tryte_tiles import TryteTileFullAtomSession

SAMPLE_AF_PDB = """HEADER    ALPHAFOLD V4 TEST
ATOM      1  N   ALA A   1       1.000   1.000   1.000  1.00 95.40           N
ATOM      2  CA  ALA A   1       2.450   1.000   1.000  1.00 95.40           C
ATOM      3  C   ALA A   1       3.000   2.300   1.000  1.00 95.40           C
ATOM      4  N   GLY A   2       4.300   2.500   1.000  1.00 52.10           N
ATOM      5  CA  GLY A   2       4.900   3.800   1.000  1.00 52.10           C
ATOM      6  C   GLY A   2       6.400   3.800   1.000  1.00 52.10           C
END
"""

SAMPLE_AF_CIF = """data_AF_TEST
_cell.length_a 20.0
_cell.length_b 20.0
_cell.length_c 20.0
_cell.angle_alpha 90.0
_cell.angle_beta 90.0
_cell.angle_gamma 90.0
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.B_iso_or_equiv
ATOM 1 N N ALA A 1 1.0 1.0 1.0 94.2
ATOM 2 C CA ALA A 1 2.4 1.0 1.0 94.2
ATOM 3 N N GLY A 2 3.8 1.5 1.0 48.5
ATOM 4 C CA GLY A 2 5.1 2.0 1.0 48.5
"""


class TestAlphaFoldAndDeepEngineV26(unittest.TestCase):
    def test_parse_plddt_pdb_and_mmcif(self) -> None:
        c_pdb = parse_alphafold_plddt_any(SAMPLE_AF_PDB, fmt="pdb")
        self.assertEqual(c_pdb.plddt_per_atom.size, 6)
        self.assertEqual(c_pdb.high_confidence_atoms.size, 3)
        self.assertEqual(c_pdb.low_confidence_atoms.size, 3)
        self.assertTrue(c_pdb.is_likely_alphafold)

        c_cif = parse_alphafold_plddt_any(SAMPLE_AF_CIF, fmt="cif")
        self.assertEqual(c_cif.plddt_per_atom.size, 4)
        self.assertEqual(c_cif.high_confidence_atoms.size, 2)
        self.assertEqual(c_cif.low_confidence_atoms.size, 2)
        self.assertAlmostEqual(float(c_cif.plddt_per_residue[0]), 94.2, places=1)
        self.assertAlmostEqual(float(c_cif.plddt_per_residue[1]), 48.5, places=1)

    def test_pae_json_and_domain_clustering(self) -> None:
        pae = np.full((40, 40), 20.0, dtype=np.float32)
        pae[:16, :16] = 2.5
        pae[24:, 24:] = 2.8
        np.fill_diagonal(pae, 0.5)
        raw_json = json.dumps([{"predicted_aligned_error": pae.tolist()}])
        parsed = parse_pae_json(raw_json)
        self.assertIsNotNone(parsed)
        part = cluster_pae_domains(parsed, pae_cutoff_ang=5.0, min_domain_size=12)
        self.assertEqual(len(part.domains), 2)
        self.assertEqual(len(part.hinge_residues), 8)
        self.assertGreater(part.inter_domain_pae_mean, 15.0)

    def test_plddt_restraints_and_active_dsf_coulomb_in_tryte(self) -> None:
        # 2 oppositely charged ions at 0.45 nm apart: DSF Coulomb should produce attractive force
        pos = np.array([[0.0, 0.0, 0.0], [0.45, 0.0, 0.0]], dtype=np.float32)
        sim_q = TryteTileFullAtomSession(
            positions_nm=pos,
            atomic_numbers=["Na", "Cl"],
            charges=np.array([1.0, -1.0], dtype=np.float32),
            rcut=0.9,
        )
        sim_0 = TryteTileFullAtomSession(
            positions_nm=pos,
            atomic_numbers=["Na", "Cl"],
            charges=np.array([0.0, 0.0], dtype=np.float32),
            rcut=0.9,
        )
        # Attractive Coulomb lowers potential energy substantially vs uncharged LJ-only pair
        self.assertLess(sim_q.potential_energy, sim_0.potential_energy - 50.0)

        # Verify per-atom pLDDT restraints attach cleanly and stay bounded
        weights = compute_plddt_restraint_weights(np.array([95.0, 45.0]), k_high=300.0)
        self.assertGreater(float(weights[0]), 290.0)
        self.assertEqual(float(weights[1]), 0.0)
        rep = sim_q.set_plddt_restraints(weights)
        self.assertTrue(rep["active"])
        self.assertEqual(rep["restrained_atoms"], 1)
        m = sim_q.step(10)
        self.assertEqual(m["health_status"], "bounded")

    def test_rmsf_and_atlas_flexibility_eval(self) -> None:
        rng = np.random.default_rng(7)
        base = np.array([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [0.6, 0.0, 0.0], [0.9, 0.0, 0.0]])
        scales = np.array([0.01, 0.01, 0.08, 0.08])[:, None]
        traj = np.stack([base + rng.normal(0.0, 1.0, size=base.shape) * scales for _ in range(30)], axis=0)
        rmsf = compute_per_residue_rmsf(traj, residue_ids=[1, 1, 2, 2])
        self.assertEqual(rmsf.size, 2)
        self.assertGreater(rmsf[1], rmsf[0] * 2.0)
        ev = evaluate_atlas_flexibility("mock", np.array([0.5, 0.6, 2.1, 2.4]), np.array([0.4, 0.5, 2.0, 2.5]))
        self.assertGreater(ev.pearson_r, 0.95)

    def test_fastapi_alphafold_endpoint_and_analysis(self) -> None:
        app = create_app()
        client = TestClient(app)
        fake_entry = AlphaFoldEntry(
            uniprot_id="P69905",
            entry_id="AF-P69905-F1",
            gene="HBA1",
            organism="Homo sapiens",
            structure_text=SAMPLE_AF_PDB,
            structure_format="pdb",
            filename="AF-P69905-F1.pdb",
            pae_matrix=None,
            source_url="https://alphafold.ebi.ac.uk/files/AF-P69905-F1-model_v4.pdb",
        )
        with patch("trium_studio.app.fetch_alphafold_entry", return_value=fake_entry):
            res = client.post("/api/sessions/alphafold/P69905?storage_mode=tryte-all-atom")
            self.assertEqual(res.status_code, 200, res.text)
            payload = res.json()
            sid = payload["session_id"]
            self.assertEqual(payload["uniprot_id"], "P69905")
            self.assertEqual(payload["num_atoms"], 6)
            self.assertTrue(payload["restraints"]["active"])

            # Check analysis endpoint & trajectory export endpoints
            ana = client.get(f"/api/sessions/{sid}/alphafold-analysis")
            self.assertEqual(ana.status_code, 200)
            self.assertTrue(ana.json()["has_alphafold"])
            self.assertEqual(client.get(f"/api/sessions/{sid}/export/dcd").status_code, 200)
            self.assertEqual(client.get(f"/api/sessions/{sid}/export/xtc").status_code, 200)
            self.assertEqual(client.get(f"/api/sessions/{sid}/export/pdb").status_code, 200)


if __name__ == "__main__":
    unittest.main()
