# Rigid TIP3P water box: NVT trajectories with FoundationCore

Standalone research composition layer on top of the engine factors
(`StreamedLJ` for O–O Lennard-Jones, `PME` for electrostatics, `Constraints` for
rigid water). It is **demonstration evidence, not validated molecular
dynamics**. The only current result is a bounded 20 ps equilibration + 20 ps
production diagnostic at 200 molecules with the corrected principal-axis
integrator; it is not equilibrated bulk water.

## Current model

TIP3P (Jorgensen 1983) in `(nm, kJ/mol, amu, e, ps)`, where
`1 amu nm²/ps² = 1 kJ/mol` exactly, Coulomb prefactor `138.935456`,
`k_B = 0.008314462618 kJ/(mol K)`:

| quantity | value |
| --- | --- |
| `q_O`, `q_H` | -0.834 e, +0.417 e |
| `sigma_O`, `eps_O` | 0.315061 nm, 0.636386 kJ/mol |
| H sites | `eps_H = 0` (no LJ), so all H-involving LJ pairs are exactly zero |
| `r_OH`, `angle HOH` | 0.09572 nm, 104.52° |
| `m_O`, `m_H` | 15.9994, 1.008 amu |
| constraints | O–H, O–H, H–H per molecule, projected every step |

Intramolecular pairs are excluded in both backends (`StreamedLJ.PairTopology` and
`PME.PairCorrections`). LJ is plain-cutoff (`:lfs`, C¹, no dispersion tail
correction). PME real-space now applies the minimum image **per pair**, so any
number of cells per axis is valid; `pme_rcut` only needs
`0 < rcut < min(L)/2` and `alpha*pme_rcut = 4.5` is chosen so the hard real-space
cutoff is negligible (`erfc(4.5) = 1.97e-10`).

Integrator: constrained BAOAB with the **corrected rigid-body drift** — a
symmetric principal-axis splitting `H₁(dt/2) H₂(dt/2) H₃(dt) H₂(dt/2) H₁(dt/2)`
of the free asymmetric-top Hamiltonian. It is reversible and preserves both
kinetic energy and **laboratory angular momentum**. (The previous map rotated
positions and momenta together by `exp(ω dt)`; that conserves K but rotates
`L_lab`, which is wrong for an asymmetric top. All numbers predating that fix are
superseded.) The thermostat projects its noise in the mass metric and onto the
constraint tangent space, so the stationary temperature does not depend on
`gamma`.

## Files

| path | what |
| --- | --- |
| `../../src/WaterBox.jl` | the module: builder, composed field, integrators, analysis |
| `../../test/test_water_box.jl` | geometry, exclusions, finite-difference forces, constraints, PME/Ewald parity, thermostat |
| `run_tip3p.jl` | one run per process: build → relax → equilibrate → **temperature gate** → produce → replay check → NVE + dt ladder → report |
| `analyze_tip3p.jl` | independent re-analysis from the saved artefacts only (one-origin D; diagnostic) |
| `runs/w200_corrected/` | **current** bounded diagnostic: 200 molecules, 20 ps + 20 ps, 2 fs, γ = 1/ps |
| `runs/w200_corrected/msd_all_origin.py` | **selected** diffusion estimator (all-origin) driver over the COM stream |
| `runs/w200/` | aborted run of the pre-fix drift; `STATUS.md` says superseded |
| `runs/w216/` | legacy: old drift and the `production!` COM-denominator bug (D suppressed by the squared scale), so its D is **not usable** |
| `runs/w1728/` | completed legacy pre-fix run: 5,184 atoms, 15 ps production + 6 ps equilibration, 10,218.5 s production; NVE band 5,083.47 kJ/mol and broken diffusion diagnostic; invalid as physics evidence |

Those historical bands do not reproduce in current bounded probes; their exact
source/state provenance is unavailable. Current 216/1728-water equal-duration
probes show dt² energy bands after unconverged relaxation. See
[`../nve_diagnosis/README.md`](../nve_diagnosis/README.md) for conditions, COM-drift
limitations and the all-step diagnostic correction. No bulk qualification follows.

Per run directory: `traj.xyz` + `traj.xyz.jsonl` (molecules kept whole),
`com_unwrapped.csv` (exact unwrapped mass-weighted COM), `series.csv` /
`equil_series.csv` (`step,time_ps,U,K,E,T`), `msd_oo.csv`, `rdf_oo.csv`,
`nve_ladder.csv`, `report.txt`, `report.json`, and chunk checkpoints
`checkpoint.bin` / `checkpoint.bin.prev`. The current run also carries
`msd_all_origin.json` / `msd_all_origin.md` (selected all-origin D) and
`rdf_oo_recheck.csv` (RDF clamped to `min(L)/2`).

## Exact commands

```bash
cd julia
export JULIA_DEPOT_PATH=~/.cache/biology-mm/julia-depot
JULIA=~/.cache/biology-mm/julia-1.13.3/bin/julia

# current bounded run (2 fs, gamma = 1/ps, 20 + 20 ps, 5 ps checkpoint chunks)
$JULIA --project=. experiments/tip3p_trajectory/run_tip3p.jl \
    --waters 200 --equil-ps 20 --prod-ps 20 --dt-fs 2.0 --gamma 1.0 \
    --seed 11 --relax-steps 3000 --frame-ps 0.5 --rdf-frames 40 \
    --sample-every 10 --nve-steps 1000 --nve-dt-fs 2.0 \
    --nve-ladder 0.5,1.0,2.0 --nve-ladder-ps 1.0 --chunk-ps 5 \
    --outdir experiments/tip3p_trajectory/runs/w200_corrected \
    --label w200_corrected

# independent re-analysis (RDF, density, molecule integrity; one-origin D is a diagnostic)
$JULIA --project=. experiments/tip3p_trajectory/analyze_tip3p.jl \
    experiments/tip3p_trajectory/runs/w200_corrected

# SELECTED diffusion estimator: all-origin, all-molecule MSD
# (writes runs/w200_corrected/msd_all_origin.json and .md)
python3 experiments/tip3p_trajectory/runs/w200_corrected/msd_all_origin.py

# RDF / diffusion with MDAnalysis (XYZ + JSONL box sidecar)
python3 - <<'PY'
import json, numpy as np, MDAnalysis as mda
u = mda.Universe("experiments/tip3p_trajectory/runs/w200_corrected/traj.xyz",
                 topology_format="XYZ")
box = np.array([json.loads(l)["box"] for l in
                open("experiments/tip3p_trajectory/runs/w200_corrected/traj.xyz.jsonl")])
ow = u.select_atoms("name O")
from MDAnalysis.analysis.rdf import InterRDF
rdf = InterRDF(ow, ow, nbins=200, range=(0.0, 1.0)).run()
np.savetxt("rdf_mdanalysis.csv", np.c_[rdf.bins, rdf.rdf], delimiter=",")
PY

# quick ONE-ORIGIN diffusion diagnostic from the COM stream (not the selected
# estimator; the selected all-origin estimator is msd_all_origin.py above)
python3 - <<'PY'
import numpy as np
d = np.loadtxt("experiments/tip3p_trajectory/runs/w200_corrected/com_unwrapped.csv",
               delimiter=",", skiprows=1)
t = np.unique(d[:, 0]); n = int(d[:, 1].max())
com = d[:, 2:5].reshape(len(t), n, 3)
msd = ((com - com[0]) ** 2).sum(-1).mean(1)
lo = len(t) // 2
D = np.polyfit(t[lo:] - t[lo], msd[lo:], 1)[0] / 6      # nm^2/ps
print("D =", D * 1000, "nm^2/ns")
PY
```

The runner defaults (`--dt-fs 1.0`, `--gamma 5.0`) are the historical settings;
the corrected production protocol passes `2 fs` and `1/ps` explicitly. Use
`--temp-gate abort` to refuse production when the equilibration tail misses the
target at 3σ (default `warn` records `target_temp_reached` and
`rdf_diffusion_trusted` in `report.json`).

## Measured results — `runs/w200_corrected` (the only current run)

| | value |
| --- | --- |
| molecules / atoms | 200 / 600 |
| box / density | 1.8154³ nm³ / 1.0000 g/cm³ (**NVT input**) |
| force field | LJ `:lfs` 0.817 nm + PME 0.890 nm, α = 5.06 1/nm, 32³ order 6 |
| relaxation | 1666 steps, U 770.0 → −9651.3 kJ/mol, tangent \|F\|max 9.8 |
| equilibration | 20.0 ps; tail (last 10 ps) T = 296.43 ± 1.67 K (IAT g = 10.56) |
| production | 10000 steps = **0.0200 ns**, wall 137.6 s, **12.56 ns/day** |
| production T | **301.99 ± 1.35 K** (IAT-corrected SE, g = 13.72, 1000 samples) |
| energy drift | 0.5264 ± 1.3973 kJ/mol/ps (HAC; thermostat exchange, not conserved) |
| NVE, 2 ps at 2 fs | band 1.26 kJ/mol, slope −0.0396 ± 0.0314 kJ/mol/ps |
| NVE ladder, 1 ps each | band 0.0797 / 0.2941 / 1.221 kJ/mol at 0.5 / 1 / 2 fs |
| ⟨U⟩ | −7795.7 kJ/mol (−38.98 per molecule); mean still trending +0.24 kJ/mol/ps (non-stationary) |
| O–O RDF | first peak 0.278–0.279 nm (g ≈ 2.65), first min 0.326 nm; valid range 0–0.908 nm (= L/2) |
| **diffusion (selected, all-origin)** | **4.369 nm²/ns**; OLS SE 0.018, block SE 0.080 (3 blocks); window scan 4.31–4.56 |
| diffusion (one-origin diagnostic) | 4.5255 ± 0.1829 nm²/ns (Julia `diffusion_from_msd`; not the reported estimator) |
| memory | 625 MiB max RSS (521 MiB peak footprint) |
| throughput | 13.76 ms/step, 72.7 steps/s (production incl. frame/COM/RDF I/O) |
| restart | final 2500-step chunk replayed from checkpoint: bit-identical |

The stored temperature error uses a paired autocorrelation sum capped at 30
lags. Extending it to its first nonpositive pair gives g=29.63 and SE=1.98 K,
with the same 301.99 K mean. The stored 1.35 K error can understate uncertainty.

**Diffusion — selected estimator.** `runs/w200_corrected/msd_all_origin.py` runs
`biomolecular_validation/msd_pipeline.py` (all-origin, all-molecule MSD,
D = slope/6) on `com_unwrapped.csv` and writes
`msd_all_origin.json` / `msd_all_origin.md`:

| quantity | value |
| --- | --- |
| D internal (default 10–30 % lag window = 2.0–6.0 ps) | **4.369 nm²/ns** |
| OLS SE (optimistic; lagged points correlated) | 0.018 nm²/ns |
| block SE (3 contiguous blocks 13/13/14 frames) | 0.080 nm²/ns |
| lab frame (no drift removal) | 4.369 nm²/ns (no measurable COM drift) |
| window scan (0.05–0.10, 0.10–0.20, 0.10–0.30, 0.10–0.50, 0.30–0.70) | 4.561, 4.415, 4.369, 4.309, 4.475 nm²/ns |
| linearity | R² = 0.9999, log-log slope 0.94 |

The Julia one-origin estimate (4.5255) is quoted only as a diagnostic; the
pipeline's own flags (`few_frames`, `time_span_below_100ps`) and its caveats
apply.

Independent re-analysis (`analyze_tip3p.jl`, saved artefacts only): 40 frames,
**O,H,H labels and 600 atoms verified on every frame**, constant box verified
(NVT), frame spacing exactly 0.5 ps, O–H integrity
`max|r_OH − 0.09572| = 1.5e-13 nm`, density 1.00000 g/cm³, RDF clamped to
`r ≤ 0.9077 nm` with peak `g = 2.656 at 0.279 nm`, and the one-origin
`D = 4.5255 ± 0.1323 nm²/ns` (diagnostic only; the reported all-origin value is
4.369 nm²/ns).

> **Historical RDF tail:** the in-run `rdf_oo.csv` of this run was sampled to
> rmax = 1.2 nm, beyond `min(L)/2 = 0.9077 nm`, so its bins above 0.9077 nm are
> **invalid** (minimum image plus spherical-shell normalisation do not hold
> there). The first peak (0.278–0.279 nm) and first minimum (0.326 nm) are
> unaffected; use `rdf_oo_recheck.csv` (clamped, valid to 0.9077 nm) for the
> RDF. The runner and analysis script now clamp automatically and flag the
> invalid tail.

## How to read these numbers (and what they do not establish)

* **Temperature is gated before production.** The gate is the last half of
  equilibration with an IAT-corrected SE, 3σ (or 3 K) tolerance, and it only
  ever sets `temperature_trusted`; `rdf_diffusion_trusted` and
  `equilibrium_claim` remain false because no independent equilibration
  criteria are implemented. For `w200_corrected` the gate was computed
  post-hoc from `equil_series.csv` (disclosed in its `report.txt`/`RUN.md`;
  `report.json` records `temp_gate_stage = "post-production"`); the runner now
  evaluates it before production. The production window itself is at
  301.99 ± 1.35 K.
* **20 ps is not enough for a converged diffusion coefficient.** The selected
  all-origin estimator (default 2.0–6.0 ps window) gives 4.369 nm²/ns, moving
  between 4.31 and 4.56 nm²/ns across its documented window scan; block SE
  (0.080) and OLS SE (0.018) are optimistic, and the pipeline itself flags
  `few_frames` and `time_span_below_100ps`. The Julia one-origin value (4.5255)
  is a diagnostic, not the reported estimator. The legacy 100 ps `w216` value is
  **not** a usable comparison: that run used the old drift and a COM stream
  divided by `N*m_O` (D suppressed by the squared scale). A converged D would
  need nanosecond runs, multiple seeds, and a finite-size study.
* **The RDF is first-shell evidence only** at 40 frames, and it is valid only
  up to `min(L)/2 = 0.9077 nm`; the second shell is partially sampled.
* **Density is an input** (NVT), not a measured equilibrium density. There is
  no pressure, no NPT, no dispersion correction, and no experimental or
  OpenMM comparison in this directory.
* TIP3P is a non-polarizable three-site model that overestimates real-water
  self-diffusion; the value above is a model number, not a physical claim.

## Engine state relevant to this directory

1. **`production!` COM denominator** (fixed in `src/WaterBox.jl` at 20:43,
   2026-11-12): it divided the molecular COM by `sum(m_O over molecules)`
   instead of the molecule mass. The stored COM is scaled by
   `c = M_mol/(N*m_O)`, so **MSD and D are suppressed by `c²`** (≈2.72e-5 for
   N=216), not by `c`. The runner never relied on it — production bookkeeping
   here computes the mass-weighted COM from `WB.integrate!` — so
   `w200_corrected` is unaffected, while `runs/w216`'s D is unusable.
2. **Rigid-body drift** (replaced at 20:33): the old single-step rotation
   `exp(ω dt)` conserved K but rotated lab angular momentum for an asymmetric
   top. The current principal-axis symmetric splitting is reversible and
   preserves `L_lab`; the NVE band/ladder above are the measured check. The
   earlier "exact rigid-body drift" wording in this README was wrong: it was
   orthogonal (K-preserving), not the torque-free flow.
3. **PME real-space cell shift** (fixed): the old plan applied a fixed
   per-cell-pair shift that was only valid for ≤ 2 cells per axis and
   `water_field` refused larger boxes. `PME._cell_pairs` now uses the per-pair
   minimum image; the 1728-water box (4 cells/axis) builds and steps.
4. **`Dynamics.velocity_verlet_step!`** still does not accept a
   `CoupledConstraintSet`; the module keeps its own NVE step for the rigid-water
   band check.

## Unit conventions

Positions nm, energies kJ/mol, masses amu, charges e, time ps. Because
`1 amu nm²/ps² = 1 kJ/mol`, `p` is in amu·nm/ps, `dt = 0.002` is 2 fs and
`gamma = 1` is 1/ps.

## What is not supported

Triclinic cells (the box is orthorhombic), NPT/pressure, long-range dispersion,
polarization, SETTLE/SHAKE/LINCS kernels (the module projects coupled rigid
triangles), and any claim of validated bulk-water thermodynamics. The 1728-water
tier is not runnable in a 10-minute budget: a measured feasibility probe gives
912.7 ms/step (5184 atoms, 64³ PME grid), i.e. ~114–152 min for 15–20 ps.
