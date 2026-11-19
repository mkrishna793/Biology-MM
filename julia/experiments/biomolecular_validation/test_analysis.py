import unittest

import numpy as np

import analyze


class AnalysisTests(unittest.TestCase):
    def test_rdf_uniform_ideal_gas_normalization(self):
        rng = np.random.default_rng(1234)
        frames = rng.uniform(0, 20, size=(100, 160, 3))
        r, g = analyze.rdf(frames, [20, 20, 20], bins=16)
        # Statistical sanity check, not an acceptance interval for molecular RDFs.
        bulk = (r > 3.0) & (r < 9.0)
        self.assertLess(abs(float(np.mean(g[bulk])) - 1.0), 0.12)

    def test_rdf_lattice_peak_and_frame_blocks(self):
        grid = np.array([[x, y, z] for x in (0.0, 5.0) for y in (0.0, 5.0) for z in (0.0, 5.0)])
        frames = np.stack([grid + np.random.default_rng(seed).normal(0, 0.01, grid.shape)
                           for seed in range(12)]) % 10
        result = analyze.rdf_diagnostics(frames, [10, 10, 10], bins=100, block_size=3)
        self.assertEqual(result["n_blocks"], 4)
        self.assertAlmostEqual(result["peak_r"], 5.0, delta=0.2)
        self.assertIsNotNone(result["peak_r_stderr"])
        self.assertIsNotNone(result["peak_height_stderr"])

    def test_cross_rdf_and_minimum_image(self):
        x = np.array([[[0.2, 1, 1], [9.8, 1, 1], [5, 5, 5]]], dtype=float)
        r, g = analyze.rdf(x, [10, 10, 10], [0], [1], bins=10, r_max=2)
        self.assertEqual(int(np.argmax(g)), int(np.argmin(abs(r - 0.4))))

    def test_brownian_msd_recovers_diffusion_scale(self):
        rng = np.random.default_rng(43)
        diffusion = 0.35
        nframes, natoms, dt = 900, 240, 0.01
        increments = rng.normal(0, np.sqrt(2 * diffusion * dt), (nframes - 1, natoms, 3))
        xyz = np.concatenate([np.zeros((1, natoms, 3)), np.cumsum(increments, axis=0)])
        result = analyze.msd_diffusion(xyz, np.arange(nframes) * dt)
        self.assertLess(abs(result["internal_diffusion"] - diffusion) / diffusion, 0.10)
        self.assertIn("lab_diffusion", result)

    def test_com_drift_removal_preserves_internal_diffusion(self):
        rng = np.random.default_rng(8)
        increments = rng.normal(0, np.sqrt(2 * 0.2 * 0.01), (299, 20, 3))
        xyz = np.concatenate([np.zeros((1, 20, 3)), np.cumsum(increments, axis=0)])
        xyz += np.arange(300)[:, None, None] * np.array([0.04, -0.02, 0.01])
        result = analyze.msd_diffusion(xyz, np.arange(300) * 0.01)
        self.assertGreater(result["lab_diffusion"], result["internal_diffusion"])
        self.assertIsNotNone(result["internal_diffusion_stderr"])

    def test_blocking_and_scalar_diagnostics(self):
        x = np.arange(100, dtype=float)
        result = analyze.block_error(x, 10)
        self.assertEqual(result["n_blocks"], 10)
        self.assertAlmostEqual(result["mean"], 49.5)
        diag = analyze.scalar_diagnostics(np.ones(20))
        self.assertEqual(diag["tau_int_frames"], 0.5)
        self.assertAlmostEqual(diag["effective_samples_estimate"], 20)
        self.assertAlmostEqual(diag["autocorrelation_stderr"], 0)


if __name__ == "__main__":
    unittest.main()
