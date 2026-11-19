#!/usr/bin/env python3
"""Audit the *old* WaterBox.production! molecular-COM denominator.

The old `julia/src/WaterBox.jl` built the molecular centre of mass as

    com = (mO*rO + mH*rH1 + mH*rH2) / sum(mO over all molecules)

The denominator should be the single-molecule mass M_mol = mO + 2*mH, so the old
engine stream is scaled by M_mol/(N*mO) = 1/f, with f = N*mO/M_mol, and every
MSD/diffusion number derived from it is divided by f^2. The current
`production!` now uses the per-molecule mass (`WaterBox.jl:1183-1190`); this
script audits the old behaviour on stored engine streams.

This script reads an engine `com_unwrapped.csv` (read-only; the trajectory lane
owns it), fits the all-origin COM MSD with the lane-2 pipeline, and then rescales
the same stream by f and 1/f.  The ratio of fitted D values demonstrates the
f^2 scaling on a real engine trajectory.  It is a scaling audit only: the
trajectory lane marks w200 as a superseded diagnostic, so its absolute D is not
used as a physical measurement.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

import msd_pipeline as msd

M_O_DEFAULT = 15.9994
M_H_DEFAULT = 1.008


def load_com_csv(path):
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    if data.ndim != 2 or data.shape[1] != 5:
        raise ValueError("expected columns time_ps,molecule,x_nm,y_nm,z_nm")
    times = np.unique(data[:, 0])
    molecules = np.unique(data[:, 1].astype(int))
    nframes, nmol = len(times), len(molecules)
    if data.shape[0] != nframes * nmol:
        raise ValueError("CSV does not contain a complete frames x molecules grid")
    com = data[:, 2:5].reshape(nframes, nmol, 3)
    if np.any(np.diff(times) <= 0):
        raise ValueError("times must be strictly increasing")
    return times, com, molecules


def to_units(d_nm2_ps):
    return {
        "nm2_per_ps": float(d_nm2_ps),
        "angstrom2_per_ps": float(d_nm2_ps * 100.0),
        "m2_per_s": float(d_nm2_ps * 1e-6),
    }


def fit_stream(times, com):
    curve = msd.msd_all_origins(com)
    fit = msd.fit_diffusion(curve["lags"], curve["msd"], times)
    blocks = msd.blocked_diffusion(com, times, n_blocks=4, remove_drift=False)
    return fit, blocks, curve


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--com-csv", required=True)
    parser.add_argument("--label", default="engine")
    parser.add_argument("--assume", choices=("correct", "buggy"), required=True,
                        help="whether the supplied COM stream uses the correct denominator")
    parser.add_argument("--m-o", type=float, default=M_O_DEFAULT)
    parser.add_argument("--m-h", type=float, default=M_H_DEFAULT)
    parser.add_argument("--output")
    args = parser.parse_args(argv)

    times, com, molecules = load_com_csv(args.com_csv)
    nmol = com.shape[1]
    m_mol = args.m_o + 2.0 * args.m_h
    factor = nmol * args.m_o / m_mol          # f
    factor2 = factor * factor

    fit_supplied, blocks_supplied, curve = fit_stream(times, com)
    fit_times_f, _, _ = fit_stream(times, com * factor)
    fit_div_f, _, _ = fit_stream(times, com / factor)
    if args.assume == "buggy":
        fit_correct, blocks_correct = fit_times_f, msd.blocked_diffusion(
            com * factor, times, n_blocks=4, remove_drift=False)
    else:
        fit_correct, blocks_correct = fit_supplied, blocks_supplied
    fit_buggy = fit_div_f if args.assume == "correct" else fit_supplied

    ratio = fit_correct["diffusion"] / fit_buggy["diffusion"]
    local_slopes = []
    lag_time = msd.lag_times(curve["lags"], times)
    for k in range(1, len(curve["msd"])):
        if curve["msd"][k - 1] > 0:
            local_slopes.append(float(
                math.log(curve["msd"][k] / curve["msd"][k - 1])
                / math.log(lag_time[k] / lag_time[k - 1])))

    report = {
        "label": args.label,
        "source_csv": str(Path(args.com_csv).resolve()),
        "assumed_state": args.assume,
        "frames": int(len(times)),
        "molecules": int(nmol),
        "time_span_ps": float(times[-1] - times[0]),
        "frame_interval_ps": float(np.median(np.diff(times))),
        "molecule_mass_amu": m_mol,
        "denominator_factor_f": float(factor),
        "expected_suppression_factor_f2": float(factor2),
        "expected_log10_suppression": float(math.log10(factor2)),
        "fit_window_lag_index": fit_correct["fit_lag_index"],
        "fit_window_time_ps": fit_correct["fit_time_range"],
        "loglog_slope_in_window": fit_correct["loglog_slope"],
        "local_loglog_slope_range": [float(np.min(local_slopes)),
                                     float(np.max(local_slopes))],
        "diffusion_correct_denominator": to_units(fit_correct["diffusion"]),
        "diffusion_correct_block_stderr": blocks_correct["stderr"],
        "diffusion_supplied_stream": to_units(fit_supplied["diffusion"]),
        "diffusion_emulated_buggy_denominator": to_units(fit_buggy["diffusion"]),
        "diffusion_stream_multiplied_by_f": to_units(fit_times_f["diffusion"]),
        "diffusion_stream_divided_by_f": to_units(fit_div_f["diffusion"]),
        "observed_correct_over_buggy": float(ratio),
        "observed_log10_suppression": float(math.log10(ratio)),
        "observed_over_expected": float(ratio / factor2),
        "caveats": [
            "Scaling audit only; w200 is a superseded diagnostic under integrator audit.",
            "The f^2 identity is exact for any linear MSD fit because the supplied COM stream is uniformly scaled.",
            "Absolute D values are not physical measurements of TIP3P or of any engine.",
        ],
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf8")
        print(f"wrote {args.output}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
