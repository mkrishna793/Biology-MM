# PME scaling evidence

Executable evidence for the claim "PME is the right electrostatics backend at
scale": measured cost versus particle count, the crossover against direct
Ewald, the allocation/per-call profile of a reused plan, and an error-versus-N
table at fixed grid spacing.

Everything here is produced by the benchmark in
`julia/bench/pme_scaling.jl`; this directory only drives it and
stores the output. No numbers are hand-edited.

## Reproduce

```sh
export JULIA_DEPOT_PATH=~/.cache/biology-mm/julia-depot
~/.cache/biology-mm/julia-1.13.3/bin/julia \
    --project=julia julia/experiments/pme_scaling/run.jl
```

The driver runs three modes and writes `results/`:

| file | content |
| --- | --- |
| `results/levels.json`, `results/sweep.txt` | per-N timings/allocations/RSS and the fitted exponent |
| `results/crossover.txt` | PME vs the shipped direct Ewald and vs an allocation-free control Ewald |
| `results/crossover_small.txt` | the same at N < 500 with a scale-consistent cutoff |
| `results/accuracy.txt` | error versus N at fixed grid spacing (grid-refined and Ewald references) |

Individual modes can be run directly:

```sh
julia --project=julia julia/bench/pme_scaling.jl
julia --project=julia julia/bench/pme_scaling.jl --crossover
julia --project=julia julia/bench/pme_scaling.jl --accuracy
julia --project=julia julia/bench/pme_scaling.jl --level=10000
```

`--level=N` measures one size in a fresh process and prints one JSON line; the
sweep mode uses it so that peak RSS is attributable to a single N. The 10⁶
level is attempted only if the built-in preflight (estimated footprint < 10% of
RAM, < half of reclaimable memory, 1-min load < CPU count) passes.

## State point

Fixed for the whole timing series, so that only N changes:

* number density 0.1 sites/Å³ (all-atom-water-like), cubic box, uniform random
  positions, charges uniform on ±0.5 shifted to be exactly neutral;
* `alpha = 0.35 Å⁻¹` (`1/alpha = 2.86 Å`), `rcut = 8.0 Å` (`alpha*rcut = 2.8`,
  real-space truncation `erfc(2.8) = 7.5e-5`), `order = 4`;
* grid spacing 0.8 Å held fixed, odd grid dimensions (no Nyquist plane, so the
  influence function has no ambiguous half-mode);
* one energy+force evaluation with a plan built once, unit prefactor.

## What the numbers establish

See `results/` for the run used in the report. In summary (medians of repeated
calls, fresh process per level, machine under load ~5-6 of 10 CPUs):

* shorter repeat sweeps have medians 6.5 ms at 10³, 82.6 ms at 10⁴ and 656 ms
  at 10⁵. The four-size sweep measured 8.44 ms / 105.4 ms / 723.3 ms / 12.915 s
  through 10⁶ (269³ grid). Its fitted exponent is `1.039 ± 0.054`, R²=0.9947;
  the shorter three-point fits are `0.99–1.03`. Raw peak RSS at 10⁶ is
  1,320,681,472 bytes = 1259.5 MiB = 1.23 GiB (earlier prose rounded it as 1.26 GiB);
* a warm energy/force evaluation allocates **zero** bytes at every size, with
  and without pair corrections, for order 4 and 6;
* no crossover is observed: in the stored fixed-cutoff family PME beats the
  pre-scalarization direct Ewald by 45–462× and an allocation-free scalar control
  by 11–101× over N=500..10,000. In the scale-consistent `rcut = 0.45 L` family
  it beats that control by about 13× even at N=25. A universal crossover location
  is not established. Two reversed English summary lines in `crossover.txt`
  are a reporting typo; the numeric columns show PME winning;
* at fixed grid spacing the relative energy error stays at
  2×10⁻⁵–8×10⁻⁵ and the RMS force error at 7×10⁻⁵–1.3×10⁻⁴ from N = 10³ to
  10⁵, with no trend in N (the force error decreases slightly); the
  grid-refined error and the independent direct-Ewald error agree at that
  level, so the grid — not the particle count — controls accuracy.

Three PME defects were corrected before these measurements: a 3.2 MB per-call
allocation at N=10⁵ (now zero), `ceil(L/rc)` cell counts causing excessive search
(using `floor` reduced candidate pairs from 184M to 78M), and three divisions
per candidate pair (replaced by inverse-box multiplication). Energy-only,
energy+force and charge-gradient calls reuse the plan and allocate 0 B, including
corrections. Even grid dimensions omit Nyquist planes and can change coarse-grid
error by about 5×; odd grids or default power-of-two dimensions with order 6 are
smoother. The later direct-Ewald scalarization in `Electrostatics.jl` is audited
separately in `validation/STATUS.md`; the historical timing columns remain intact.

## What they do NOT establish

* not a comparison with OpenMM or any other engine, and not a GPU result;
* one density, one cutoff, one spline order, one grid spacing, one cell shape;
* timings are single-threaded wall clock on one laptop while other work was
  running: the 1-min load average is printed per level (5–6 of 10 CPUs for the
  stored run) and back-to-back repeats of the same level vary by 1.07–1.26×
  (`results/runs.txt`); medians of a handful of calls, not a statistically
  rigorous benchmark. Absolute times measured earlier under heavier load were
  up to 2× slower, so compare ratios, not absolute seconds;
* the fitted exponent is a local fit over the measured sizes; nothing here says
  it extrapolates to 10⁷ particles or predicts memory-bandwidth behaviour. The
  10⁶ point uses a 269³ (prime-length) FFT grid, which is slower per grid point
  than a smooth length, so that segment's exponent (1.25) is an upper bound;
* `results/accuracy.txt` measures the *difference from a finer grid* and from
  direct Ewald at a 10⁻⁴ per-mode reciprocal truncation; it is not an absolute
  accuracy certificate for arbitrary systems.
