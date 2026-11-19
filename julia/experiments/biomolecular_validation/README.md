# Biomolecular trajectory diagnostics

This is a small exploratory analysis harness, not a validated force field, simulator, or scientific acceptance suite. It reads trajectories through MDAnalysis and computes selected-atom auto-RDF with frame-block uncertainty and peak estimates, scalar autocorrelation/block estimates, an optional CSV energy trend, and optional MSD/diffusion estimates. See `BIOMOLECULAR_VALIDATION.md` for formulas, limits, and measured self-test evidence.

## Run

Use a Python environment that has NumPy and MDAnalysis installed (`python -m pip install -r julia/experiments/biomolecular_validation/requirements.txt`). Inputs are a topology and a trajectory, for example a PDB with DCD or PDB with XYZ when supported by the installed MDAnalysis readers:

```sh
python julia/experiments/biomolecular_validation/analyze.py \
  --topology system.pdb --trajectory production.dcd --rdf --msd --unwrap --json
```

Optional scalar series use `--scalar-csv values.csv --scalar-column temperature` (repeat `--scalar-column` for density, pressure, or any numeric column); energies use `--energy-csv values.csv --energy-column potential`. CSV columns are numeric. Synthetic checks cover ideal-gas RDF normalization, a simple cubic lattice peak, diffusion, and scalar diagnostics. Run them with `python -m unittest discover -s julia/experiments/biomolecular_validation -v`.

## Corrected MSD pipeline (`msd_pipeline.py`)

`msd_pipeline.py` is a standalone, single-purpose MSD/self-diffusion estimator. It
does not import `analyze.py` except when `--reconcile-analyze` is requested, and
it does not change the loader, RDF, or bonded code. Frame times come from
MDAnalysis and are treated as ps; if a reader reports frame indices instead of
physical times, pass `--dt` (ps) and the pipeline rebuilds the time axis as
`frame * dt`. Use it when the MSD number matters more than the convenience of the
one-shot CLI:

```sh
python julia/experiments/biomolecular_validation/msd_pipeline.py \
  --topology system.pdb --trajectory production.dcd --selection 'name O' \
  --reconcile-analyze --output msd.json
```

Estimator definition:

- Unwrap by integrating frame-to-frame minimum-image displacements (per-frame
  boxes supported). Every selected atom must move less than half a box edge
  between saved frames; the report gives the largest **inferred** minimum-image
  step as a fraction of the half box and counts ambiguous (exactly half-box)
  steps. Zero ambiguous steps only rules out exact-half ties; the wrapped frames
  cannot prove the true unobserved motion stayed below half a box.
- MSD at lag `k` averages **all time origins and all atoms**, with `N-k` origins
  per atom: `MSD(k) = mean_{i,t<N-k} |r_i(t+k)-r_i(t)|²`.
- Fit `MSD(τ)` linearly and report `D = slope/(2·dim)` (dim = 3 by default).
  The default window is the 10–30 % fraction of lags; this is an explicit
  heuristic, and the report includes a five-window sweep so window sensitivity is
  visible instead of hidden.
- Report lab-frame and COM-drift-corrected ("internal") estimates, OLS diffusion
  error (optimistic: lagged MSD points are correlated), and a contiguous-block
  standard error when enough frames exist (`null` otherwise, never zero).
- Report local log-log MSD slopes; a diffusive plateau requires a slope near 1,
  not just a large `R²` from a linear fit.

Tests (`test_msd_pipeline.py`) cover the exact MSD definition and sample counts,
recovery of a known Brownian `D` within the empirical standard error of eight
independent replicas (not a hard-coded percentage), the ideal-gas ballistic
`MSD ∝ τ²` slope and coefficient, lab-versus-internal drift behavior, periodic
crossing removal with a strongly wrapped trajectory, varying per-frame boxes,
half-box ambiguity reporting, short-trajectory flags, and agreement with the
legacy estimator on a genuinely diffusive synthetic series. Measured values are
recorded in `results/msd_synthetic.md`.

Long reference (100 ps equilibration + 200 ps production, OpenMM 8.6.1 OpenCL,
2 nm TIP3P/PME) and its diffusive-slope analysis are in
`results/msd_reference_long.md` (+ `.json`) with generation settings in
`reference_long/metadata.json`. The raw measured model value is about
0.44–0.50 Å²/ps (0.48 central), which is roughly 2× the experimental water value
of 2.3 × 10⁻³ nm²/ps = 0.23 Å²/ps. Finite-size corrections are illustrative
arithmetic examples conditional on an assumed viscosity, not a measured
corrected range, and are not folded into the headline. The older
0.2 ps `reference/` remains only as a smoke test.

The COM-denominator audit for the engine's four-order diffusion suppression is
in `results/engine_com_audit.md` (+ two JSON files). It demonstrates the exact
`f² = (N·m_O/M_mol)²` amplification (`D_correct/D_buggy`) on the trajectory
lane's `w200` and `w216` COM streams: log₁₀ ratios 4.499 and 4.566,
observed/expected 1.000000. The stored `w216` stream is from the old bookkeeping
path; `w200` is already corrected by the runner workaround. The current
`WaterBox.production!` now uses the per-molecule mass, so the defect itself is no
longer present.

This module is standalone Python analysis. It needs **no Julia registry entry**
and no `FoundationCore.jl` edit; it reads MDAnalysis trajectories or the engine
lane's `com_unwrapped.csv` directly. The engine **does** have a persistent frame
writer (`Dynamics.FrameWriter`, `src/Dynamics.jl:328`, writing XYZ + JSONL via
`write_frame!`) and the trajectory lane already consumes it; an earlier statement
here that no writer existed was wrong. Applying this pipeline to engine output is
an analysis/reader convenience, not a package-integration requirement; any
additional adapter should be requested separately rather than assumed.

## Interpretation limits

- RDF uses spherical-shell ideal-gas normalization and minimum-image distances. The cutoff must be no larger than half the shortest orthorhombic box edge. The first-frame box is used; changing-box trajectories are not supported. Auto RDF counts unordered pairs; cross RDF counts ordered pairs.
- This bullet describes the legacy `analyze.py` MSD; for new work use `msd_pipeline.py` above. MSD requires `--unwrap`. The loader unwraps selected coordinates by frame-to-frame minimum image and assumes no atom moves more than half a box edge between saved frames. This is not bond-aware and can fail for sparse output, large displacements, ambiguous cells, or molecules crossing inconsistently. XYZ generally does not carry periodic box metadata. Reports include lab-frame diffusion and internal diffusion after subtracting the linear selected-group COM displacement trend; this avoids removing instantaneous COM fluctuations, but can remove genuine constant drift. Internal diffusion gets a replicate-third standard error when enough frames exist. Both estimates use a heuristic middle/long-lag linear fit.
- RDF output includes per-bin frame-block standard errors and block peak-position/height estimates with standard errors. The default block size is approximately sqrt(number of frames), not an automatically justified decorrelation time. Scalar outputs include block and integrated-autocorrelation-corrected standard errors. Integrated autocorrelation truncates at the first nonpositive lag and remains a diagnostic estimate.
- Energy trend reads externally supplied scalar CSV; it reports raw and linear-detrended means and a fitted slope with ordinary regression standard error. It does not recompute energies or infer ensemble correctness. A slope is descriptive, not a pass/fail threshold.
- The first-half versus second-half scalar comparison is an equilibration-warning heuristic, not a convergence test; it can miss slow modes and nonstationarity.
- No accepted scientific ranges are encoded. Verify units, selections, equilibration, sampling, force field, electrostatics, box handling, and uncertainty independently.

## OpenMM reference generation

With OpenMM installed, `python julia/experiments/biomolecular_validation/generate_tip3p_pme.py --steps 100 --output-dir /path/to/output` creates a 2 nm TIP3P water box, PME system, minimized starting state, short Reference-platform DCD, topology PDB, sampled energy CSV, and metadata. This is a reproducible small fixture, not an equilibrated reference or accepted scientific range. Generation choices (box size, seed, cutoff, timestep, minimization, and Reference platform) are recorded in the script/metadata; they are not claimed to be validated production settings. No bulk TIP3P diffusion/reference RDF is inferred from this short run.

For the longer diffusion reference, use the separate generator (it refuses to overwrite an existing output directory):

```sh
python julia/experiments/biomolecular_validation/generate_tip3p_long.py \
  --output-dir julia/experiments/biomolecular_validation/reference_long \
  --platform OpenCL --equil-ps 100 --prod-ps 200 --frame-ps 1.0 --seed 2027
```

It writes `trajectory.dcd`, `topology.pdb`, `energy.csv`, `equil_energy.csv`, and
`metadata.json`. On this machine the 100 + 200 ps OpenCL run took 38 s wall; the
script falls back to CPU then Reference if OpenCL is unavailable. The output is a
single 2 nm box and still not a converged bulk-water reference.
