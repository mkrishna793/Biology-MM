# Biomolecular trajectory validation harness

## Status and scope

This is an analysis harness, not a simulator validator or a force-field certification. It can report trajectory diagnostics with explicit sampling limitations. It cannot prove that a protein folded, that kinetics are correct, or that an ensemble is equilibrated. It does not currently read a native trajectory written by the Foundation Julia engine: `Dynamics.run_trajectory` is not documented as a persistent trajectory writer, and the core does not provide production TIP3P PME dynamics. Supported input is any topology/trajectory pair readable by MDAnalysis (tested here with PDB + DCD); XYZ may work with an MDAnalysis topology but usually has no periodic cell and cannot support periodic RDF or unwrapping. No invented engine-native dump format is claimed.

**Evidence labels:** numbers below marked **MEASURED** were produced by the commands in this document. Choices such as the first/second-half stationarity test and heuristic MSD fit interval are **ASSUMED heuristics**, not validated scientific criteria. No universal acceptance thresholds are encoded.

## Methods

### RDF

For a fixed orthorhombic cell of volume `V`, with `N` selected atoms and an auto-RDF, the estimator in each bin is

`g_k = H_k / (F * N(N-1)/2 * ΔV_k/V)`,

where `H_k` is the count of unordered minimum-image atom pairs accumulated over `F` frames and `ΔV_k = 4π/3 (r_{k+1}^3-r_k^3)`. The equivalent ordered-pair form uses `N(N-1)` counts. This excludes self-pairs. Distances use minimum image, and the radius is restricted to half the shortest orthorhombic edge. `--r-max` defaults to that half-edge. Thus the RDF is explicitly **truncated** there; no long-range tail correction or extrapolation is applied. The current CLI computes selected-group auto-RDFs (e.g. `name O`); O-H partial RDF is not yet exposed as a CLI option.

`rdf_diagnostics` computes whole-trajectory g(r), then non-overlapping frame-block g(r) curves and the standard error across those block curves. The reported peak is the largest sampled bin, not a fitted continuous maximum. Peak position and height standard errors are computed across block maxima. With fewer than two complete blocks, uncertainty is unavailable. Correlated blocks make these errors optimistic; choose block size based on correlation analysis rather than treating the default `sqrt(frames)` as a guarantee.

### MSD and self diffusion

The estimator averages all time origins at each lag, `MSD(τ)=<|r_i(t+τ)-r_i(t)|²>`, and fits the middle/long portion of the available lag range (one third through two thirds of lag indices). The 3-D Einstein relation is `D = slope(MSD)/6`. Periodic trajectories require `--unwrap`; this uses frame-to-frame minimum-image displacement and assumes each selected atom moves less than half a box edge between saved frames. It is not bond-aware and may fail for sparse output, changing cells, or inconsistent molecular imaging. The default COM correction subtracts the fitted linear velocity of the selected-group COM from coordinates, preserving instantaneous COM fluctuations. Both lab-frame and drift-corrected estimates are reported. Diffusion uncertainty is estimated from the standard error among up to three contiguous trajectory-third estimates, only if each has enough frames; with short trajectories it is `null`, not zero. This is a weak uncertainty estimate, not independent replicas or a proven Einstein region.

Finite-size hydrodynamic corrections to D are not applied. Comparisons must match water model, temperature, density, electrostatics, constraints, system size, and correction convention. No TIP3P bulk diffusion value/range is asserted here because a traceable reference source and converged production run were not established for this harness execution.

### Scalar diagnostics, energy, and equilibration

CSV scalar columns (including user-supplied density, temperature, pressure, or energies) are summarized with mean, standard deviation of block means divided by square-root block count, and an integrated-autocorrelation diagnostic. The autocorrelation estimate uses normalized lag correlations summed from `τ_int=1/2+Σρ(k)` until the first nonpositive lag; the corresponding standard error is `s sqrt(2τ_int/N)`. This truncation rule can be noisy and the output is diagnostic, especially for short series. The default non-overlapping block length is `sqrt(N)`. Values and units are supplied by the caller; pressure is not recomputed from virials, and density/temperature/pressure are not inferred from coordinates.

Energy CSV output reports the raw scalar diagnostics, a least-squares linear slope and ordinary regression standard error, and a linearly detrended series with diagnostics. Detrending is descriptive and does not replace reporting raw energy. There is no integrator-specific expected drift band because this harness does not know which force model/integrator produced the input.

For scalar columns, the optional equilibration warning compares first-half and second-half means against two full-series autocorrelation-corrected standard errors. This simple **ASSUMED heuristic** can miss slow modes, transitions, and common nonstationary behavior; it does not automatically select a production window. Users must select and document the production range independently. Pressure virial decomposition and dielectric fluctuation analysis are not implemented.

## Self-tests and measured fixture

Run the focused tests:

```sh
$HOME/.cache/biology-mm/.venv/bin/python -m unittest discover -s julia/experiments/biomolecular_validation -v
```

**MEASURED self-test result:** six tests passed. A seeded uniform ideal-gas configuration has mean bulk RDF within 0.12 of 1; a simple cubic lattice produces its expected nearest-neighbor peak bin; periodic minimum-image cross-pair distance is checked; Brownian increments generated with `D=0.35` recover D within 10%; drift removal preserves the known internal diffusion; and scalar blocking/constant-series diagnostics are checked. These are implementation checks on synthetic data, not a molecular validation.

Generate a small independent OpenMM fixture:

```sh
$HOME/.cache/biology-mm/.venv/bin/python \
  julia/experiments/biomolecular_validation/generate_tip3p_pme.py \
  --steps 100 --output-dir julia/experiments/biomolecular_validation/reference
```

**MEASURED generation settings:** OpenMM 8.6.1, Reference platform, `tip3p.xml`, 2.0 nm cubic box, PME, 0.9 nm cutoff, Ewald error tolerance `5e-4`, rigid water/H-bond constraints, 300 K LangevinMiddle thermostat, friction 1/ps, 2 fs timestep, seed 2026, 100 steps, coordinate minimization capped at 100 iterations. It contains 774 atoms / 258 waters and saves 10 DCD frames (every 10 steps). Initial and minimized potential energies were -1071.216 and -11374.614 kJ/mol respectively. This is a minimization followed by a **0.2 ps short fixture**, not equilibrated at a validated density or temperature.

Analyze its oxygen subset (MDAnalysis reads the PDB/DCD pair):

```sh
$HOME/.cache/biology-mm/.venv/bin/python \
  julia/experiments/biomolecular_validation/analyze.py \
  --topology julia/experiments/biomolecular_validation/reference/topology.pdb \
  --trajectory julia/experiments/biomolecular_validation/reference/trajectory.dcd \
  --selection 'name O' --rdf --msd --unwrap --json
```

**MEASURED diagnostic only:** selected O-O RDF's largest sampled peak is at 2.75 Å with height 3.016 ± 0.130 (frame-block SE; three blocks, 3 frames each). The peak position is 2.750 ± 0.033 Å (block-maximum SE). This coarse-bin, three-block result is not a trustworthy bulk TIP3P reference peak. The short-run apparent internal diffusion is 0.532 Å²/ps; uncertainty is unavailable (`null`) because fewer than three sufficiently long trajectory thirds exist. The fit uses only 0.06–0.12 ps lags, so this is not a converged self-diffusion coefficient. No accepted TIP3P range comparison is made. The input lacks O-H analysis in CLI and no equilibration has been demonstrated. Do not use these values as engine pass/fail gates.

Environment versions recorded for this analysis: Python 3.12; NumPy 2.5.4; SciPy 1.18.1; MDAnalysis 2.11.0; MDTraj 1.11.2; OpenMM 8.6.1; Julia 1.13.3 is available but is not used by this Python reader. Execution was single-process/single-threaded by the harness; no performance benchmark was run.

## CLI and report

The CLI emits JSON with `--json`; omit it for a concise human summary. Supply topology and trajectory paths, `--selection`, and any combination of `--rdf`, `--msd --unwrap`, `--scalar-csv` + repeated `--scalar-column`, or `--energy-csv --energy-column`. `--block-size` overrides the heuristic block length. Full formulas, assumptions, and caveats are embedded in the JSON output/report alongside values. The current implementation reports RDF/MSD and externally supplied scalar columns; it does not calculate volume-derived density, kinetic temperature, virial pressure, O-H RDF, dielectric constant, or engine energy itself.

## What this establishes

The synthetic checks establish limited estimator behavior for ideal-gas/lattice RDF normalization, synthetic Brownian diffusion scale, COM drift handling, and scalar diagnostics. The OpenMM fixture establishes that the supplied generator can produce a PME-configured TIP3P short trajectory which MDAnalysis can read and the harness can process. It does **not** establish accurate PME dynamics, an equilibrated TIP3P ensemble, a converged RDF or diffusion coefficient, an integrator drift band, or compatibility with a Foundation engine trajectory writer. It cannot demonstrate folding, native structure, or kinetics. Those claims require longer, equilibrated, replicated, independently cross-validated molecular trajectories and suitable scientific references.
