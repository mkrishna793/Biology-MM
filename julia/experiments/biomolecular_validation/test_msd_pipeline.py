"""Tests for the all-origin MSD pipeline; synthetic data only.

Run with the supplied environment:
    python -m unittest discover -s julia/experiments/biomolecular_validation -v
"""

import unittest

import numpy as np

import msd_pipeline as msd
import audit_engine_com as audit


def _brownian(nframes, natoms, dt, diffusion, seed, start=None):
    rng = np.random.default_rng(seed)
    increments = rng.normal(0.0, np.sqrt(2.0 * diffusion * dt),
                            (nframes - 1, natoms, 3))
    xyz = np.concatenate([np.zeros((1, natoms, 3)), np.cumsum(increments, axis=0)])
    if start is not None:
        xyz = xyz + np.asarray(start, dtype=float)
    return xyz


class MsdDefinitionTests(unittest.TestCase):
    def test_com_denominator_suppression_is_f_squared(self):
        # A uniform COM scale s scales MSD (and any linear MSD slope) by s^2.
        # This is the identity behind the engine's N*m_O/M_mol suppression.
        rng = np.random.default_rng(4)
        nframes, nmolecules, dt = 240, 200, 0.5
        increments = rng.normal(0.0, np.sqrt(2.0 * 0.004 * dt),
                                (nframes - 1, nmolecules, 3))
        com = np.concatenate([np.zeros((1, nmolecules, 3)),
                              np.cumsum(increments, axis=0)])
        times = np.arange(nframes) * dt
        factor = nmolecules * 15.9994 / 18.0154
        correct, _, _ = audit.fit_stream(times, com)
        buggy, _, _ = audit.fit_stream(times, com / factor)
        self.assertAlmostEqual(correct["diffusion"] / buggy["diffusion"],
                               factor ** 2, places=6)
        self.assertAlmostEqual(np.log10(factor ** 2), 4.50, delta=0.05)

    def test_msd_matches_direct_definition_and_sample_counts(self):
        frames = np.array([
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0], [1.0, 2.0, 0.0]],
            [[3.0, 0.0, 0.0], [1.0, 2.0, 3.0]],
            [[6.0, 0.0, 0.0], [1.0, 2.0, 6.0]],
        ])
        curve = msd.msd_all_origins(frames)
        self.assertEqual(curve["lags"].tolist(), [1, 2, 3])
        # Lag 1: three time origins per atom.
        self.assertAlmostEqual(curve["msd"][0], 6.0)
        # Lag 3: one origin per atom; atom 0 contributes 36, atom 1 40.
        self.assertAlmostEqual(curve["msd"][2], 38.0)
        self.assertEqual(curve["n_samples"].tolist(), [6, 4, 2])

    def test_fit_recovers_analytic_slope_and_reports_window(self):
        times = np.arange(10, dtype=float) * 0.5
        lags = np.arange(1, 10)
        msd_values = 6.0 * 0.25 * times[1:]
        fit = msd.fit_diffusion(lags, msd_values, times, window=(2, 8))
        self.assertAlmostEqual(fit["diffusion"], 0.25, places=12)
        self.assertAlmostEqual(fit["loglog_slope"], 1.0, places=12)
        self.assertEqual(fit["fit_lag_index"], [2, 8])
        self.assertEqual(fit["fit_lag_count"], 7)


class BrownianDiffusionTests(unittest.TestCase):
    def test_known_diffusion_recovered(self):
        diffusion = 0.35
        rng = np.random.default_rng(12345)
        estimates = []
        for _ in range(8):
            seed = int(rng.integers(0, 2**31 - 1))
            xyz = _brownian(800, 100, 0.01, diffusion, seed=seed)
            report = msd.diffusion_summary(xyz, np.arange(800) * 0.01)
            estimates.append(report["diffusion_internal"]["value"])
            self.assertLess(abs(report["diffusion_lab"]["value"]
                                - report["diffusion_internal"]["value"]) / diffusion,
                            0.05)
        estimates = np.asarray(estimates)
        mean = float(estimates.mean())
        se = float(estimates.std(ddof=1) / np.sqrt(len(estimates)))
        # Recover within the empirical replica standard error rather than a
        # hard-coded percentage: a single trajectory has finite-sample scatter.
        self.assertGreater(se, 0.0)
        self.assertLess(se, 0.05 * diffusion)
        self.assertLessEqual(abs(mean - diffusion), 2.0 * se)
        self.assertEqual(report["unwrap"]["unwrapped"], False)

    def test_drift_lab_vs_internal(self):
        diffusion = 0.20
        xyz = _brownian(1200, 96, 0.01, diffusion, seed=8)
        velocity = np.array([0.04, -0.02, 0.01])
        xyz = xyz + (np.arange(1200) * 0.01)[:, None, None] * velocity
        report = msd.diffusion_summary(xyz, np.arange(1200) * 0.01)
        self.assertGreater(report["diffusion_lab"]["value"],
                           report["diffusion_internal"]["value"])
        self.assertLess(abs(report["diffusion_internal"]["value"] - diffusion) / diffusion,
                        0.10)
        np.testing.assert_allclose(report["drift"]["velocity"], velocity, atol=0.04)

    def test_ideal_gas_ballistic_slope_is_two(self):
        # Constant per-atom velocities with kT/m = 1 give MSD(tau) = 3 tau^2:
        # a log-log slope of 2, not the diffusive slope of 1.
        rng = np.random.default_rng(7)
        nframes, natoms, dt = 400, 500, 0.005
        velocities = rng.normal(0.0, 1.0, (natoms, 3))
        # Force the sample velocity variance to exactly kT/m = 1 per component
        # so the analytic coefficient is 3, independent of the draw.
        velocities = (velocities - velocities.mean(axis=0)) / velocities.std(axis=0)
        times = np.arange(nframes) * dt
        frames = times[:, None, None] * velocities[None, :, :]
        curve = msd.msd_all_origins(frames)
        fit = msd.fit_diffusion(curve["lags"], curve["msd"], times,
                                window=(40, 200))
        self.assertAlmostEqual(fit["loglog_slope"], 2.0, delta=0.03)
        # MSD/tau^2 = 3 kT/m; the Einstein fit has no plateau and would be
        # misread as a lag-dependent D.
        ratio = curve["msd"] / (times[1:] ** 2)
        self.assertAlmostEqual(float(np.median(ratio[40:200])), 3.0, delta=0.02)
        self.assertGreater(fit["diffusion"], 0.0)
        short = msd.fit_diffusion(curve["lags"], curve["msd"], times, window=(40, 80))
        long = msd.fit_diffusion(curve["lags"], curve["msd"], times, window=(160, 200))
        self.assertGreater(long["diffusion"], short["diffusion"])


class UnwrapTests(unittest.TestCase):
    def test_periodic_crossings_are_removed(self):
        box = np.array([10.0, 10.0, 10.0])
        truth = _brownian(800, 150, 0.01, 1.0, seed=11, start=[9.8, 0.1, 5.0])
        wrapped = truth % box
        unwrapped, info = msd.unwrap_trajectory(wrapped, box)
        np.testing.assert_allclose(unwrapped, truth, atol=1e-9)
        self.assertGreater(info["crossing_events"], 20)
        self.assertEqual(info["ambiguous_steps"], 0)
        self.assertLess(info["max_step_fraction_of_half_box"], 0.2)

        wrapped_msd = msd.msd_all_origins(wrapped)["msd"]
        true_msd = msd.msd_all_origins(unwrapped)["msd"]
        self.assertGreater(wrapped_msd[0], 5.0 * true_msd[0])
        self.assertLess(abs(true_msd[399] / (6.0 * 1.0 * 4.0) - 1.0), 0.05)

    def test_per_frame_varying_box(self):
        box = np.array([[10.0, 10.0, 10.0], [10.2, 10.2, 10.2]])
        frames = np.array([[[9.9, 5.0, 5.0]], [[0.2, 5.0, 5.0]]])
        unwrapped, info = msd.unwrap_trajectory(frames, box)
        self.assertTrue(info["box_varying"])
        self.assertAlmostEqual(unwrapped[1, 0, 0], 10.4, places=12)

    def test_half_box_step_is_reported_as_ambiguous(self):
        box = np.array([10.0, 10.0, 10.0])
        frames = np.array([[[0.0, 5.0, 5.0]], [[5.0, 5.0, 5.0]]])
        _, info = msd.unwrap_trajectory(frames, box)
        self.assertEqual(info["ambiguous_steps"], 1)
        self.assertAlmostEqual(info["max_step_fraction_of_half_box"], 1.0)

    def test_half_box_flag_propagates_to_summary(self):
        box = np.array([10.0, 10.0, 10.0])
        rng = np.random.default_rng(5)
        frames = rng.uniform(0.0, 10.0, (6, 4, 3))
        frames[1, 0, 0] = (frames[0, 0, 0] + 5.0) % 10.0
        report = msd.diffusion_summary(frames, np.arange(6) * 0.02,
                                       box_lengths=box)
        codes = [flag["code"] for flag in report["sampling"]["flags"]]
        self.assertIn("ambiguous_unwrap_step", codes)


class SamplingTests(unittest.TestCase):
    def test_short_trajectory_is_flagged_without_a_physical_claim(self):
        xyz = _brownian(10, 258, 0.02, 0.5, seed=2)
        report = msd.diffusion_summary(xyz, np.arange(10) * 0.02,
                                       box_lengths=[20.0, 20.0, 20.0])
        codes = [flag["code"] for flag in report["sampling"]["flags"]]
        self.assertIn("time_span_below_10ps", codes)
        self.assertIn("few_frames", codes)
        self.assertIn("no absolute D claim", report["sampling"]["claim_limit"])
        self.assertIsNone(report["diffusion_internal"]["stderr_blocks"])
        self.assertGreaterEqual(len(report["fit"]["window_scan_internal"]), 2)
        self.assertTrue(any("10 ps is too short" in cue for cue in report["caveats"]))

    def test_legacy_estimator_agrees_on_a_diffusive_series(self):
        import analyze  # same directory; imported only for reconciliation
        diffusion = 0.35
        xyz = _brownian(1200, 120, 0.01, diffusion, seed=99)
        times = np.arange(1200) * 0.01
        report = msd.diffusion_summary(xyz, times)
        legacy = analyze.msd_diffusion(xyz, times)
        self.assertLess(abs(legacy["internal_diffusion"]
                            - report["diffusion_internal"]["value"]) / diffusion, 0.12)


if __name__ == "__main__":
    unittest.main()
