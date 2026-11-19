#!/usr/bin/env python3
"""Self-contained MSD / self-diffusion estimator for trajectory diagnostics.

This is the reference MSD implementation for the biomolecular-validation
experiment.  It is deliberately separate from ``analyze.py`` so that the other
lanes can keep their own loaders/RDF code untouched.

Estimator
---------
For unwrapped coordinates ``r_i(t)`` the mean-squared displacement at lag
``k`` averages all available time origins and all selected atoms::

    MSD(k) = mean_{i, t < N-k} |r_i(t+k) - r_i(t)|^2

with ``N-k`` origins per atom (no wrap-around, no per-lag renormalisation).
In ``d`` dimensions the Einstein relation is ``MSD(tau) = 2 d D tau``, so
``D = slope/6`` for ``d = 3``.

Periodic crossings are removed before the MSD is formed by integrating
frame-to-frame minimum-image displacements.  The unwrap assumes every selected
atom moves less than half a box edge between *saved* frames; the returned info
block reports the largest observed step as a fraction of the half box so this
assumption can be audited.  A step of exactly half a box is ambiguous and is
counted rather than silently guessed.

What this module does **not** do
-------------------------------
It does not decide whether a diffusive plateau exists, does not apply
finite-size hydrodynamic corrections, and does not supply an accepted TIP3P
diffusion value.  The fit window is an explicit, documented heuristic; the
log-log local slope and the window-sensitivity numbers are reported so the
user can see whether the fit is inside a diffusive regime.  An absolute ``D``
from a short or non-equilibrated trajectory is a diagnostic, not a physical
result.
"""

from __future__ import annotations

import argparse
import json
import math
import sys

import numpy as np

DEFAULT_FIT_FRACTION = (0.10, 0.30)
ASSUMED_TIME_UNITS = "ps"


# ---------------------------------------------------------------------------
# Unwrapping
# ---------------------------------------------------------------------------

def unwrap_trajectory(frames, box_lengths=None):
    """Remove periodic jumps by integrating minimum-image displacements.

    Parameters
    ----------
    frames : (nframes, natoms, 3) array
        Wrapped coordinates, one row per saved frame.
    box_lengths : (3,) or (nframes, 3) or None
        Orthorhombic box edge lengths.  ``None`` means the input is already
        continuous; a copy is returned with ``unwrapped=False``.

    Returns
    -------
    (unwrapped, info)
        ``info`` records the largest minimum-image step relative to the half
        box, the number of atoms/frames that needed an image correction, and
        how many steps were numerically ambiguous (>= half a box).
    """
    xyz = np.asarray(frames, dtype=float)
    if xyz.ndim != 3 or xyz.shape[2] != 3 or len(xyz) < 2:
        raise ValueError("frames must be nframes x natoms x 3 with nframes >= 2")
    if box_lengths is None:
        return xyz.copy(), {
            "unwrapped": False,
            "reason": "no box supplied; coordinates assumed already continuous",
            "crossing_events": 0,
            "ambiguous_steps": 0,
            "max_step_fraction_of_half_box": None,
        }

    boxes = np.asarray(box_lengths, dtype=float)
    if boxes.ndim == 1:
        if boxes.shape != (3,):
            raise ValueError("box_lengths must have three values")
        boxes = np.broadcast_to(boxes, (len(xyz), 3)).copy()
    elif boxes.shape != (len(xyz), 3):
        raise ValueError("box_lengths must be (3,) or (nframes, 3)")
    if np.any(boxes <= 0):
        raise ValueError("box lengths must be positive")

    raw = np.diff(xyz, axis=0)
    step = raw - boxes[1:, None, :] * np.round(raw / boxes[1:, None, :])
    unwrapped = np.empty_like(xyz)
    unwrapped[0] = xyz[0]
    unwrapped[1:] = xyz[0] + np.cumsum(step, axis=0)

    half = boxes[1:] / 2.0
    fraction = np.abs(step) / half[:, None, :]
    image_shift = np.round(raw / boxes[1:, None, :])
    return unwrapped, {
        "unwrapped": True,
        "box_varying": bool(np.any(np.ptp(boxes, axis=0) > 0)),
        "crossing_events": int(np.count_nonzero(np.any(image_shift != 0, axis=2))),
        "ambiguous_steps": int(np.count_nonzero(np.abs(step) >= half[:, None, :])),
        "max_step_fraction_of_half_box": float(np.max(fraction)),
        "assumption": "each atom moves less than half a box edge between saved frames",
    }


# ---------------------------------------------------------------------------
# MSD and fitting
# ---------------------------------------------------------------------------

def msd_all_origins(frames):
    """All-origin, all-atom MSD for every lag from 1 to nframes-1.

    ``n_samples`` is the number of (atom, time-origin) displacement vectors
    averaged at each lag.  This is the correct denominator; it decreases with
    lag and must not be replaced by a constant.
    """
    xyz = np.asarray(frames, dtype=float)
    if xyz.ndim != 3 or xyz.shape[2] != 3:
        raise ValueError("frames must be nframes x natoms x 3")
    nframes = len(xyz)
    if nframes < 3:
        raise ValueError("MSD needs at least 3 frames")
    lags = np.arange(1, nframes)
    msd = np.empty(len(lags), dtype=float)
    for j, lag in enumerate(lags):
        delta = xyz[lag:] - xyz[:-lag]
        msd[j] = float(np.mean(np.sum(delta * delta, axis=2)))
    samples = (nframes - lags) * xyz.shape[1]
    return {"lags": lags, "msd": msd, "n_samples": samples}


def lag_times(lags, times):
    """Mean elapsed time for each lag (handles non-uniform saved frames)."""
    t = np.asarray(times, dtype=float)
    lags = np.asarray(lags, dtype=int)
    if len(t) < 2 or np.any(np.diff(t) <= 0):
        raise ValueError("times must be strictly increasing")
    if lags[-1] >= len(t):
        raise ValueError("lag exceeds number of frames")
    return np.array([float(np.mean(t[lag:] - t[:-lag])) for lag in lags])


def _fraction_window(n_lags, fraction):
    lo_f, hi_f = fraction
    if not 0 < lo_f < hi_f <= 1:
        raise ValueError("fit fraction must satisfy 0 < start < end <= 1")
    if n_lags < 4:
        raise ValueError("MSD fitting needs at least four lags (five frames)")
    # Starting at lag 2 skips the single-step (integrator) transient; ending at
    # a fraction of the trace keeps several independent origins at the largest
    # fitted lag, where all-origin MSD has the most variance.
    lo = max(2, int(math.ceil(lo_f * n_lags)))
    hi = min(n_lags, max(lo + 2, int(math.ceil(hi_f * n_lags))))
    if hi <= lo:
        raise ValueError("lag window contains fewer than three lags")
    return lo, hi


def fit_diffusion(lags, msd, times, dim=3, window=None):
    """Least-squares slope of MSD(tau) and D = slope/(2*dim).

    ``window`` is an inclusive ``(lag_min, lag_max)`` index window.  The
    default is the documented 10%-30% fraction of the lag axis.  The slope
    standard error is the ordinary least-squares value; because MSD points at
    neighbouring lags share trajectories, that error is optimistically small
    and is not a substitute for independent replicates.
    """
    lags = np.asarray(lags, dtype=int)
    msd = np.asarray(msd, dtype=float)
    t = lag_times(lags, times)
    if window is None:
        window = _fraction_window(len(lags), DEFAULT_FIT_FRACTION)
    lo, hi = int(window[0]), int(window[1])
    mask = (lags >= lo) & (lags <= hi)
    x, y = t[mask], msd[mask]
    if len(x) < 3:
        raise ValueError("fit window needs at least three MSD points")
    if np.any(y < 0):
        raise ValueError("MSD values must be nonnegative")

    design = np.vstack([x, np.ones_like(x)]).T
    (slope, intercept), *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ np.array([slope, intercept])
    dof = len(x) - 2
    ss_res = float(residual @ residual)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    sigma2 = ss_res / dof
    cov = sigma2 * np.linalg.inv(design.T @ design)
    slope_stderr = float(math.sqrt(max(cov[0, 0], 0.0)))

    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "slope_stderr_ols": slope_stderr,
        "diffusion": float(slope) / (2.0 * dim),
        "diffusion_stderr_ols": slope_stderr / (2.0 * dim),
        "r2": r2,
        "dimensions": int(dim),
        "fit_lag_index": [lo, hi],
        "fit_lag_count": int(np.count_nonzero(mask)),
        "fit_time_range": [float(x[0]), float(x[-1])],
        "loglog_slope": _loglog_slope(x, y),
        "notes": [
            "OLS slope error ignores strong correlation between lagged MSD points.",
            "A linear fit is descriptive unless a diffusive plateau is demonstrated.",
        ],
    }


def _loglog_slope(x, y):
    positive = (x > 0) & (y > 0)
    if np.count_nonzero(positive) < 2:
        return None
    return float(np.polyfit(np.log(x[positive]), np.log(y[positive]), 1)[0])


# ---------------------------------------------------------------------------
# Uncertainty and sampling diagnostics
# ---------------------------------------------------------------------------

def _com_drift(frames, times, masses):
    weight = np.ones(frames.shape[1]) if masses is None else np.asarray(masses, dtype=float)
    com = np.average(frames, axis=1, weights=weight)
    return np.polyfit(times, com, 1)[0]


def blocked_diffusion(frames, times, masses=None, n_blocks=4, dim=3,
                       fit_fraction=DEFAULT_FIT_FRACTION, min_frames=12,
                       remove_drift=True):
    """Diffusion from contiguous trajectory blocks, reported as mean +/- SE.

    Blocks are not independent replicas: they are contiguous pieces of one
    trajectory and share slow modes.  With few blocks the standard error is a
    weak, usually optimistic, spread indicator; it is ``None`` when fewer than
    two blocks contain enough frames.
    """
    xyz = np.asarray(frames, dtype=float)
    t = np.asarray(times, dtype=float)
    n = len(xyz)
    if n_blocks < 2:
        raise ValueError("n_blocks must be at least 2")
    edges = np.linspace(0, n, int(n_blocks) + 1).astype(int)
    estimates = []
    used = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi - lo < max(4, int(min_frames)):
            continue
        block = xyz[lo:hi]
        block_t = t[lo:hi]
        if remove_drift:
            drift = _com_drift(block, block_t, masses)
            block = block - (block_t - block_t[0])[:, None, None] * drift[None, None, :]
        curve = msd_all_origins(block)
        window = _fraction_window(len(curve["lags"]), fit_fraction)
        try:
            fit = fit_diffusion(curve["lags"], curve["msd"], block_t,
                                dim=dim, window=window)
        except ValueError:
            continue
        estimates.append(fit["diffusion"])
        used.append([int(lo), int(hi)])

    result = {
        "n_blocks_requested": int(n_blocks),
        "blocks_used": used,
        "diffusion_per_block": [float(v) for v in estimates],
        "mean": float(np.mean(estimates)) if estimates else None,
        "stderr": (float(np.std(estimates, ddof=1) / math.sqrt(len(estimates)))
                   if len(estimates) > 1 else None),
        "note": "contiguous blocks are correlated; SE is weak and usually optimistic",
    }
    return result


def assess_sampling(times, fit_time_range=None, frame_interval=None):
    """Descriptive sampling flags; no threshold here is an acceptance criterion."""
    span = float(times[-1] - times[0])
    dt = float(frame_interval if frame_interval is not None
               else np.median(np.diff(times)))
    flags = []
    if len(times) < 100:
        flags.append({
            "code": "few_frames",
            "message": f"{len(times)} saved frames; lag averaging and blocking are weak",
        })
    if span < 10.0:
        flags.append({
            "code": "time_span_below_10ps",
            "message": (f"time span {span:.4g} {ASSUMED_TIME_UNITS} is below 10 "
                        f"{ASSUMED_TIME_UNITS}; 10 {ASSUMED_TIME_UNITS} is itself far "
                        "too short for a converged bulk self-diffusion coefficient"),
        })
    elif span < 100.0:
        flags.append({
            "code": "time_span_below_100ps",
            "message": (f"time span {span:.4g} {ASSUMED_TIME_UNITS} is below 100 "
                        f"{ASSUMED_TIME_UNITS}; a converged bulk D is not expected"),
        })
    if fit_time_range is not None and span > 0 and fit_time_range[1] < 0.1 * span:
        flags.append({
            "code": "fit_only_early_lags",
            "message": "fit uses only the earliest lags; no long-lag plateau is sampled",
        })
    if len(times) > 2 and np.ptp(np.diff(times)) > 0.01 * dt:
        flags.append({
            "code": "nonuniform_frame_spacing",
            "message": ("frame spacing varies; lag times are averaged, so the "
                        "linear MSD fit is approximate"),
        })
    return {
        "frames": int(len(times)),
        "time_span": span,
        "time_units": ASSUMED_TIME_UNITS,
        "frame_interval": dt,
        "fit_time_range": list(fit_time_range) if fit_time_range else None,
        "flags": flags,
        "claim_limit": ("no absolute D claim: a diffusive plateau, equilibration, "
                        "replicates, and finite-size treatment are all required"),
    }


# ---------------------------------------------------------------------------
# End-to-end summary
# ---------------------------------------------------------------------------

def diffusion_summary(frames, times, masses=None, remove_drift=True,
                      fit_fraction=DEFAULT_FIT_FRACTION, dim=3, n_blocks=4,
                      box_lengths=None, unwrap=True):
    """Unwrap, compute lab/internal MSD, fit D, and attach diagnostics.

    ``box_lengths`` may be None for already-continuous coordinates.  When
    ``unwrap`` is false the coordinates are used as supplied (only valid when
    the input is known to be continuous).
    """
    xyz = np.asarray(frames, dtype=float)
    t = np.asarray(times, dtype=float)
    if xyz.ndim != 3 or xyz.shape[2] != 3:
        raise ValueError("frames must be nframes x natoms x 3")
    if len(t) != len(xyz):
        raise ValueError("one time per frame is required")
    if len(t) < 5:
        raise ValueError("MSD fitting needs at least five frames")
    if np.any(np.diff(t) <= 0):
        raise ValueError("times must be strictly increasing")

    if unwrap:
        continuous, unwrap_info = unwrap_trajectory(xyz, box_lengths)
    else:
        continuous, unwrap_info = xyz.copy(), {
            "unwrapped": False,
            "reason": "caller disabled unwrapping",
            "crossing_events": 0,
            "ambiguous_steps": 0,
            "max_step_fraction_of_half_box": None,
        }

    if masses is not None:
        masses = np.asarray(masses, dtype=float)
        if len(masses) != xyz.shape[1] or np.any(masses <= 0):
            raise ValueError("masses must be positive and match the selected atoms")
        mass_note = "mass-weighted COM drift removal"
    else:
        mass_note = "uniform-weight COM drift removal (masses unavailable or not supplied)"

    drift = _com_drift(continuous, t, masses) if remove_drift else np.zeros(3)
    internal = continuous - (t - t[0])[:, None, None] * drift[None, None, :]

    lab_curve = msd_all_origins(continuous)
    int_curve = msd_all_origins(internal)
    window = _fraction_window(len(lab_curve["lags"]), fit_fraction)
    lab_fit = fit_diffusion(lab_curve["lags"], lab_curve["msd"], t, dim=dim, window=window)
    int_fit = fit_diffusion(int_curve["lags"], int_curve["msd"], t, dim=dim, window=window)

    lab_blocks = blocked_diffusion(continuous, t, masses=masses, n_blocks=n_blocks,
                                   dim=dim, fit_fraction=fit_fraction,
                                   remove_drift=False)
    int_blocks = blocked_diffusion(continuous, t, masses=masses, n_blocks=n_blocks,
                                   dim=dim, fit_fraction=fit_fraction,
                                   remove_drift=True)

    window_scan = []
    for fraction in ((0.05, 0.10), (0.10, 0.20), (0.10, 0.30),
                     (0.10, 0.50), (0.30, 0.70)):
        try:
            scan_window = _fraction_window(len(int_curve["lags"]), fraction)
            scan_fit = fit_diffusion(int_curve["lags"], int_curve["msd"], t,
                                     dim=dim, window=scan_window)
        except ValueError:
            continue
        window_scan.append({
            "fraction_of_lags": [float(fraction[0]), float(fraction[1])],
            "lag_index": list(scan_window),
            "diffusion_internal": scan_fit["diffusion"],
            "loglog_slope": scan_fit["loglog_slope"],
            "r2": scan_fit["r2"],
        })

    sampling = assess_sampling(t, int_fit["fit_time_range"])
    sampling["mass_weighting"] = mass_note
    if unwrap_info.get("ambiguous_steps", 0):
        sampling["flags"].append({
            "code": "ambiguous_unwrap_step",
            "message": ("at least one frame-to-frame step was >= half a box; "
                        "unwrapping and MSD are unreliable"),
        })

    return {
        "estimator": "all-origin, all-atom MSD; D = slope/(2*dim)",
        "units": {
            "msd": "coordinate_units^2",
            "diffusion": "coordinate_units^2/time_units",
            "time": ASSUMED_TIME_UNITS + " (assumed)",
        },
        "input": {
            "frames": int(len(xyz)),
            "atoms": int(xyz.shape[1]),
            "time_span": float(t[-1] - t[0]),
        },
        "unwrap": unwrap_info,
        "drift": {
            "removed": bool(remove_drift),
            "velocity": [float(v) for v in drift],
            "mass_weighting": mass_note,
        },
        "msd": {
            "lag_frames": lab_curve["lags"].tolist(),
            "lag_time": lag_times(lab_curve["lags"], t).tolist(),
            "lab": lab_curve["msd"].tolist(),
            "internal": int_curve["msd"].tolist(),
            "n_samples_per_lag": lab_curve["n_samples"].tolist(),
        },
        "fit": {
            "fraction_of_lags": [float(fit_fraction[0]), float(fit_fraction[1])],
            "lag_index": list(window),
            "lab": lab_fit,
            "internal": int_fit,
            "lab_blocked": lab_blocks,
            "internal_blocked": int_blocks,
            "window_scan_internal": window_scan,
        },
        "diffusion_lab": {
            "value": lab_fit["diffusion"],
            "stderr_ols": lab_fit["diffusion_stderr_ols"],
            "stderr_blocks": lab_blocks["stderr"],
        },
        "diffusion_internal": {
            "value": int_fit["diffusion"],
            "stderr_ols": int_fit["diffusion_stderr_ols"],
            "stderr_blocks": int_blocks["stderr"],
        },
        "sampling": sampling,
        "caveats": [
            "D is a descriptive fit, not a physical claim; a diffusive plateau is not established by this function.",
            "Unwrapping is not bond-aware and fails if an atom moves >= half a box between saved frames.",
            "Block and OLS errors are optimistic; independent replicated trajectories are required for a defensible uncertainty.",
            "No finite-size hydrodynamic correction is applied.",
            "Lab-frame D includes COM drift; internal D removes a fitted linear COM trend and can remove real drift.",
            "Even 10 ps is too short for a converged bulk-water D; only a demonstrated plateau over a much longer equilibrated trajectory supports a physical D claim.",
        ],
    }


# ---------------------------------------------------------------------------
# MDAnalysis loader and CLI
# ---------------------------------------------------------------------------

def load_trajectory(topology, trajectory, selection="all", dt=None):
    """Read wrapped coordinates, boxes, times, and masses via MDAnalysis."""
    try:
        import MDAnalysis as mda
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("MDAnalysis is required; install requirements.txt") from exc

    universe = mda.Universe(str(topology), str(trajectory))
    group = universe.select_atoms(selection)
    if not len(group):
        raise ValueError(f"selection {selection!r} matched no atoms")

    frames, boxes, times = [], [], []
    tilted = False
    for ts in universe.trajectory:
        frames.append(group.positions.astype(float).copy())
        dim = ts.dimensions
        if dim is None or np.any(np.asarray(dim[:3], dtype=float) <= 0):
            boxes.append(None)
        else:
            boxes.append(np.asarray(dim[:3], dtype=float))
        times.append(float(ts.time))
        if dim is not None and np.any(np.abs(np.asarray(dim[3:], dtype=float) - 90.0) > 1e-6):
            tilted = True

    frames = np.asarray(frames, dtype=float)
    times = np.asarray(times, dtype=float)
    time_source = "MDAnalysis ts.time"
    if dt is not None:
        if dt <= 0:
            raise ValueError("--dt must be positive")
        times = np.arange(len(frames), dtype=float) * float(dt)
        time_source = "user-supplied dt"
    if len(times) > 1 and np.any(np.diff(times) <= 0):
        raise ValueError("non-increasing frame times; pass --dt to define the frame spacing")

    box_array = None
    if all(b is not None for b in boxes):
        box_array = np.asarray(boxes, dtype=float)
    elif any(b is not None for b in boxes):
        raise ValueError("box metadata is present for only some frames")

    masses = None
    mass_note = "masses unavailable; uniform weights"
    raw_masses = getattr(group, "masses", None)
    if raw_masses is not None:
        raw_masses = np.asarray(raw_masses, dtype=float)
        if len(raw_masses) == len(group) and np.all(raw_masses > 0):
            masses = raw_masses
            mass_note = "MDAnalysis atom masses"

    return {
        "frames": frames,
        "boxes": box_array,
        "times": times,
        "masses": masses,
        "atoms": int(len(group)),
        "selection": selection,
        "tilted_box": bool(tilted),
        "time_source": time_source,
        "mass_note": mass_note,
    }


def _reconcile_legacy(unwrapped, times, masses, remove_drift):
    """Run the pre-existing analyze.msd_diffusion estimator for comparison."""
    try:
        import analyze
    except ImportError:
        return {"status": "unavailable: analyze.py not importable"}
    try:
        legacy = analyze.msd_diffusion(
            unwrapped, times,
            masses=masses if masses is not None else np.ones(unwrapped.shape[1]),
            remove_com=remove_drift,
        )
    except (ValueError, RuntimeError) as exc:
        return {"status": f"failed: {exc}"}
    return {
        "status": "computed",
        "estimator": "analyze.msd_diffusion (legacy middle-third lag window)",
        "internal_diffusion": legacy["internal_diffusion"],
        "internal_diffusion_stderr": legacy["internal_diffusion_stderr"],
        "lab_diffusion": legacy["lab_diffusion"],
        "fit_lag_fraction": legacy["fit_lag_fraction"],
        "note": ("Reconciliation only: the legacy window is also a heuristic. "
                 "Agreement on a diffusive series is expected; disagreement on a "
                 "short non-diffusive series exposes window sensitivity, not error "
                 "in either arithmetic."),
    }


def _json_default(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    raise TypeError(f"not JSON serialisable: {type(value)!r}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--topology", required=True)
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--selection", default="all")
    parser.add_argument("--dt", type=float,
                        help="override frame spacing (time units of the input)")
    parser.add_argument("--no-unwrap", action="store_true",
                        help="treat coordinates as already continuous (no periodic box)")
    parser.add_argument("--no-remove-drift", action="store_true",
                        help="keep the selected-group COM trend in the internal MSD")
    parser.add_argument("--dim", type=int, default=3,
                        help="spatial dimension for D = slope/(2*dim); default 3")
    parser.add_argument("--fit-start", type=float, default=DEFAULT_FIT_FRACTION[0],
                        help="start of fit window as fraction of lags")
    parser.add_argument("--fit-end", type=float, default=DEFAULT_FIT_FRACTION[1],
                        help="end of fit window as fraction of lags")
    parser.add_argument("--blocks", type=int, default=4,
                        help="contiguous blocks for the weak block-SE estimate")
    parser.add_argument("--reconcile-analyze", action="store_true",
                        help="also run analyze.msd_diffusion for comparison")
    parser.add_argument("--output", help="write JSON report to this path")
    args = parser.parse_args(argv)

    try:
        loaded = load_trajectory(args.topology, args.trajectory,
                                 args.selection, dt=args.dt)
        if loaded["tilted_box"] and not args.no_unwrap:
            raise ValueError("triclinic box detected; per-axis minimum-image "
                             "unwrapping is invalid for this input")
        report = diffusion_summary(
            loaded["frames"], loaded["times"], masses=loaded["masses"],
            remove_drift=not args.no_remove_drift,
            fit_fraction=(args.fit_start, args.fit_end),
            dim=args.dim, n_blocks=args.blocks,
            box_lengths=loaded["boxes"], unwrap=not args.no_unwrap,
        )
        report["input"].update({
            "topology": str(args.topology),
            "trajectory": str(args.trajectory),
            "selection": loaded["selection"],
            "time_source": loaded["time_source"],
            "mass_source": loaded["mass_note"],
        })
        if args.reconcile_analyze:
            continuous = (loaded["frames"] if args.no_unwrap
                          else unwrap_trajectory(loaded["frames"], loaded["boxes"])[0])
            report["legacy_analyze_estimator"] = _reconcile_legacy(
                continuous, loaded["times"], loaded["masses"],
                not args.no_remove_drift,
            )
        text = json.dumps(report, indent=2, sort_keys=True, default=_json_default)
        if args.output:
            with open(args.output, "w", encoding="utf8") as stream:
                stream.write(text + "\n")
            print(f"wrote {args.output}")
        else:
            print(text)
    except (RuntimeError, ValueError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
