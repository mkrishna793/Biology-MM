#!/usr/bin/env python3
"""Small-sample trajectory diagnostics; not a force-field validation suite."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np


def rdf(coordinates, box, selection_a=None, selection_b=None, bins=100, r_max=None):
    """Compute frame-averaged g(r), using unordered pairs for auto-RDFs."""
    xyz = np.asarray(coordinates, dtype=float)
    lengths = np.asarray(box, dtype=float)[:3]
    if xyz.ndim != 3 or xyz.shape[2] != 3 or np.any(lengths <= 0):
        raise ValueError("coordinates must be frames x atoms x 3 and box lengths positive")
    if r_max is None:
        r_max = float(np.min(lengths) / 2)
    if not 0 < r_max <= np.min(lengths) / 2 + 1e-12:
        raise ValueError("r_max must not exceed half the shortest box length")
    edges = np.linspace(0.0, r_max, int(bins) + 1)
    counts = np.zeros(int(bins), dtype=float)
    norm = np.zeros(int(bins), dtype=float)
    n_atoms = xyz.shape[1]
    ia = np.arange(n_atoms) if selection_a is None else np.asarray(selection_a, dtype=int)
    auto = selection_b is None
    ib = ia if auto else np.asarray(selection_b, dtype=int)
    if auto and len(ia) < 2 or not auto and (len(ia) == 0 or len(ib) == 0):
        raise ValueError("RDF selections need at least two auto atoms or nonempty cross selections")
    volume = float(np.prod(lengths))
    shell_vol = (4.0 * np.pi / 3.0) * (edges[1:] ** 3 - edges[:-1] ** 3)
    for frame in xyz:
        delta = frame[ia, None, :] - frame[None, ib, :]
        delta -= lengths * np.round(delta / lengths)
        distance = np.sqrt(np.sum(delta * delta, axis=-1))
        if auto:
            distance = distance[np.triu_indices(len(ia), 1)]
            pairs = len(ia) * (len(ia) - 1) / 2
        else:
            pairs = len(ia) * len(ib)
            if np.array_equal(ia, ib):
                distance = distance[~np.eye(len(ia), dtype=bool)]
                pairs = len(ia) * (len(ia) - 1)
        counts += np.histogram(distance, bins=edges)[0]
        norm += pairs * shell_vol / volume
    return (edges[:-1] + edges[1:]) / 2, np.divide(counts, norm, out=np.zeros_like(counts), where=norm > 0)


def rdf_diagnostics(coordinates, box, bins=100, r_max=None, block_size=None):
    """Frame-block uncertainty for an auto-RDF and its first resolved peak."""
    xyz = np.asarray(coordinates, dtype=float)
    r, mean = rdf(xyz, box, bins=bins, r_max=r_max)
    size = int(block_size or max(1, int(math.sqrt(len(xyz)))))
    if size < 1:
        raise ValueError("block_size must be positive")
    blocks = [rdf(xyz[i:i + size], box, bins=bins, r_max=r_max)[1]
              for i in range(0, len(xyz) - size + 1, size)]
    errors = np.std(blocks, axis=0, ddof=1) / math.sqrt(len(blocks)) if len(blocks) > 1 else None
    peak = int(np.argmax(mean))
    peak_r = [float(r[int(np.argmax(block))]) for block in blocks]
    peak_h = [float(np.max(block)) for block in blocks]
    peak_r_se = float(np.std(peak_r, ddof=1) / math.sqrt(len(peak_r))) if len(peak_r) > 1 else None
    peak_h_se = float(np.std(peak_h, ddof=1) / math.sqrt(len(peak_h))) if len(peak_h) > 1 else None
    return {"r": r.tolist(), "g_r": mean.tolist(),
            "g_r_stderr": errors.tolist() if errors is not None else [None] * len(r),
            "block_size_frames": size, "n_blocks": len(blocks),
            "peak_r": float(r[peak]), "peak_r_stderr": peak_r_se,
            "peak_height": float(mean[peak]), "peak_height_stderr": peak_h_se}


def block_error(values, block_size=None):
    """Mean and standard error of block means (non-overlapping, remainder dropped)."""
    x = np.asarray(values, dtype=float).reshape(-1)
    if len(x) < 2:
        return {"mean": float(np.mean(x)) if len(x) else None, "stderr": None,
                "block_size": block_size or 1, "n_blocks": 0}
    size = int(block_size or max(1, int(math.sqrt(len(x)))))
    if size < 1:
        raise ValueError("block_size must be positive")
    means = x[:len(x) // size * size].reshape(-1, size).mean(axis=1)
    err = float(means.std(ddof=1) / math.sqrt(len(means))) if len(means) > 1 else None
    return {"mean": float(x.mean()), "stderr": err, "block_size": size, "n_blocks": len(means)}


def autocorrelation(values, max_lag=None):
    x = np.asarray(values, dtype=float).reshape(-1)
    if len(x) == 0:
        return []
    x = x - x.mean()
    var = float(np.dot(x, x))
    if var == 0:
        return [1.0] + [0.0] * min(len(x) - 1, max_lag or len(x) - 1)
    limit = min(len(x) - 1, max_lag if max_lag is not None else len(x) - 1)
    return [float(np.dot(x[:len(x)-lag], x[lag:]) / var) for lag in range(limit + 1)]


def scalar_diagnostics(values, block_size=None):
    values = np.asarray(values, dtype=float).reshape(-1)
    acf = autocorrelation(values)
    tau = 0.5
    for corr in acf[1:]:
        if corr <= 0:
            break
        tau += corr
    result = block_error(values, block_size)
    result.update({"tau_int_frames": float(tau), "effective_samples_estimate":
                   float(len(values) / (2 * tau)) if tau > 0 else None,
                   "autocorrelation_stderr": float(np.std(values, ddof=1) * math.sqrt(2 * tau / len(values)))
                   if len(values) > 1 else None})
    if len(values) >= 4:
        split = len(values) // 2
        a, b = values[:split], values[split:]
        se = result["autocorrelation_stderr"]
        result["equilibration_heuristic"] = {
            "status": "consistent" if se is not None and abs(float(a.mean() - b.mean())) <= 2 * se else "possible_drift",
            "method": "first-half versus second-half means within two full-series autocorrelation-corrected standard errors",
            "risk": "heuristic only; can miss slow modes and is not evidence of equilibration"}
    return result


def _linear_slope(values):
    x = np.arange(len(values), dtype=float)
    if len(x) < 3:
        return None, None
    slope, intercept = np.polyfit(x, values, 1)
    residual = values - (slope * x + intercept)
    slope_se = math.sqrt(float(np.sum(residual ** 2)) / (len(x) - 2) /
                         float(np.sum((x - x.mean()) ** 2)))
    return float(slope), float(slope_se)


def msd_diffusion(coordinates, times, masses=None, remove_com=True, unwrap=False):
    """All-origin MSD, reporting lab and linear-group-drift-corrected diffusion."""
    xyz = np.asarray(coordinates, dtype=float).copy()
    times = np.asarray(times, dtype=float)
    if xyz.ndim != 3 or xyz.shape[2] != 3 or len(times) != len(xyz) or len(times) < 3:
        raise ValueError("MSD needs >=3 coordinate frames and one time per frame")
    if unwrap:
        raise ValueError("periodic unwrapping is performed by the trajectory loader, not this function")
    weight = np.ones(xyz.shape[1]) if masses is None else np.asarray(masses, dtype=float)
    if len(weight) != xyz.shape[1] or np.any(weight <= 0):
        raise ValueError("COM masses must be positive and match selected atoms")
    com = np.average(xyz, axis=1, weights=weight)
    if len(times) < 3 or np.any(np.diff(times) <= 0):
        raise ValueError("MSD times must be strictly increasing")
    drift = np.polyfit(times, com, 1)[0] if remove_com else np.zeros(3)
    internal = xyz - (times - times[0])[:, None, None] * drift[None, None, :]
    lags = np.arange(1, len(xyz))
    def calculate(data):
        return np.array([np.mean(np.sum((data[lag:] - data[:-lag]) ** 2, axis=2)) for lag in lags])
    lab_msd, internal_msd = calculate(xyz), calculate(internal)
    lag_times = np.array([np.mean(times[lag:] - times[:-lag]) for lag in lags])
    use = (lags >= max(1, len(lags) // 3)) & (lags <= max(2, 2 * len(lags) // 3))
    def fit(data):
        slope, intercept = np.polyfit(lag_times[use], data[use], 1)
        return float(slope), float(intercept)
    lab_slope, lab_intercept = fit(lab_msd)
    int_slope, int_intercept = fit(internal_msd)
    # Independent trajectory thirds provide a conservative replicate-scale error estimate.
    replicate_lab, replicate_internal = [], []
    for lo, hi in ((0, len(xyz) // 3), (len(xyz) // 3, 2 * len(xyz) // 3), (2 * len(xyz) // 3, len(xyz))):
        if hi - lo >= 12:
            clags = np.arange(1, hi - lo)
            ct = np.array([np.mean(times[lo + k:hi] - times[lo:hi-k]) for k in clags])
            mask = (clags >= max(1, len(clags) // 3)) & (clags <= max(2, 2 * len(clags) // 3))
            for data, estimates in ((xyz, replicate_lab), (internal, replicate_internal)):
                chunk = data[lo:hi]
                cmsd = np.array([np.mean(np.sum((chunk[k:] - chunk[:-k]) ** 2, axis=2)) for k in clags])
                estimates.append(float(np.polyfit(ct[mask], cmsd[mask], 1)[0] / 6))
    def replicate_se(estimates):
        return float(np.std(estimates, ddof=1) / math.sqrt(len(estimates))) if len(estimates) > 1 else None
    return {"lag_time": lag_times.tolist(), "msd_lab": lab_msd.tolist(), "msd_internal": internal_msd.tolist(),
            "lab_diffusion": lab_slope / 6, "lab_diffusion_stderr": replicate_se(replicate_lab),
            "internal_diffusion": int_slope / 6, "internal_diffusion_stderr": replicate_se(replicate_internal),
            "fit_slope": int_slope, "fit_intercept": int_intercept,
            "drift_velocity": drift.tolist(), "drift_removed": bool(remove_com),
            "units": "coordinate_units^2/time_units",
            "fit_lag_fraction": [float(lag_times[use][0]), float(lag_times[use][-1])]}


def _load_trajectory(topology, trajectory, selection, unwrap):
    try:
        import MDAnalysis as mda
        from MDAnalysis.lib.distances import distance_array
    except ImportError as exc:
        raise RuntimeError("MDAnalysis is required for topology/trajectory input; install requirements.txt") from exc
    universe = mda.Universe(str(topology), str(trajectory))
    group = universe.select_atoms(selection)
    if not len(group):
        raise ValueError("atom selection matched no atoms")
    coords, boxes, times = [], [], []
    previous = None
    for ts in universe.trajectory:
        current = group.positions.astype(float).copy()
        box = np.asarray(ts.dimensions[:3], dtype=float)
        if unwrap:
            if np.any(box <= 0):
                raise ValueError("--unwrap requires periodic box lengths in every frame")
            if previous is not None:
                step = current - previous[0]
                current = previous[1] + step - box * np.round(step / box)
            previous = (group.positions.astype(float).copy(), current.copy())
        coords.append(current)
        boxes.append(box)
        times.append(float(ts.time))
    return universe, group, np.asarray(coords), np.asarray(boxes), np.asarray(times)


def analyze(args):
    universe, group, coords, boxes, times = _load_trajectory(args.topology, args.trajectory, args.selection, args.unwrap)
    report = {"inputs": {"topology": str(args.topology), "trajectory": str(args.trajectory),
                         "selection": args.selection, "frames": len(coords), "atoms": len(group)},
              "caveats": ["No physical acceptance ranges are asserted.",
                          "XYZ has no periodic metadata; RDF requires usable box dimensions.",
                          "Unwrapping assumes frame-to-frame displacement below half a box length; it is not bond-aware.",
                          "Group-drift correction removes fitted selected-group COM velocity and can remove real drift; compare lab and internal diffusion.",
                          "Equilibration check is a half-series heuristic and can miss slow modes."]}
    if args.rdf:
        report["rdf"] = rdf_diagnostics(coords, boxes[0], bins=args.bins, r_max=args.r_max,
                                        block_size=args.block_size)
        report["rdf"].update({"box_lengths": boxes[0].tolist(),
                              "normalization": "ideal-gas shell counts for selected auto pairs"})
    if args.msd:
        if not args.unwrap:
            report["msd"] = {"status": "not computed: pass --unwrap for continuous periodic displacements"}
        else:
            report["msd"] = msd_diffusion(coords, times, getattr(group, "masses", None), args.remove_com)
    if args.energy_column:
        with open(args.energy_csv, newline="", encoding="utf8") as stream:
            vals = [float(row[args.energy_column]) for row in csv.DictReader(stream)]
        energy = scalar_diagnostics(vals, args.block_size)
        slope, slope_se = _linear_slope(np.asarray(vals))
        energy["raw_mean"] = float(np.mean(vals))
        energy["raw_values"] = vals
        energy["linear_slope_per_sample"] = slope
        energy["linear_slope_stderr_per_sample"] = slope_se
        if len(vals) > 1:
            trend = np.polyval(np.polyfit(np.arange(len(vals)), vals, 1), np.arange(len(vals)))
            detrended = np.asarray(vals) - trend + np.mean(vals)
            energy["linear_detrended_values"] = detrended.tolist()
            energy["linear_detrended"] = scalar_diagnostics(detrended, args.block_size)
        else:
            energy["linear_detrended_values"] = None
            energy["linear_detrended_mean"] = None
        report["energy_trend"] = energy
    if args.scalar_csv:
        with open(args.scalar_csv, newline="", encoding="utf8") as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
            report["scalar_diagnostics"] = {
                name: scalar_diagnostics([float(row[name]) for row in rows], args.block_size)
                for name in args.scalar_column
            }
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", required=True, help="PDB or other MDAnalysis-readable topology")
    parser.add_argument("--trajectory", required=True, help="XYZ, DCD, or other MDAnalysis-readable trajectory")
    parser.add_argument("--selection", default="all", help="MDAnalysis atom selection")
    parser.add_argument("--rdf", action="store_true")
    parser.add_argument("--msd", action="store_true")
    parser.add_argument("--unwrap", action="store_true", help="minimum-image frame-to-frame unwrapping; see caveats")
    parser.add_argument("--remove-com", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bins", type=int, default=100)
    parser.add_argument("--r-max", type=float)
    parser.add_argument("--block-size", type=int)
    parser.add_argument("--energy-csv", help="CSV file with scalar energies")
    parser.add_argument("--energy-column")
    parser.add_argument("--scalar-csv")
    parser.add_argument("--scalar-column", action="append", default=[])
    parser.add_argument("--json", action="store_true", help="emit JSON instead of concise text")
    args = parser.parse_args(argv)
    if args.energy_column and not args.energy_csv:
        parser.error("--energy-column requires --energy-csv")
    if args.scalar_csv and not args.scalar_column:
        parser.error("--scalar-csv requires at least one --scalar-column")
    try:
        result = analyze(args)
    except (RuntimeError, ValueError, OSError) as exc:
        parser.exit(2, "error: {}\n".format(exc))
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print("Trajectory diagnostics (not a scientific acceptance test)")
        print("Frames: {}; selected atoms: {}".format(result["inputs"]["frames"], result["inputs"]["atoms"]))
        for name in ("msd", "energy_trend", "scalar_diagnostics"):
            if name in result:
                print("{}: {}".format(name, json.dumps(result[name], sort_keys=True)))
        for caveat in result["caveats"]:
            print("Caveat: {}".format(caveat))
    return 0


if __name__ == "__main__":
    sys.exit(main())
