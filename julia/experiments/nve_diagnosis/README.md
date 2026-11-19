# Current-code NVE diagnosis

The stored `tip3p_trajectory/runs/w216` and `w1728` results are historical
artifacts, not reproductions of the current integrator: the existing
`tip3p_trajectory/README.md` identifies only `w200_corrected` as current.
Their JSON reports do not identify a source hash, and their starting phase-space states are not
available for an exact replay. The current source already contains the
principal-axis drift and molecular-COM normalization corrections. No new
integrator correction is justified solely by those old energy bands.

## Current reproductions

Run from `julia` with the Julia executable and depot documented in
`README.md`:

```sh
julia --project=. experiments/nve_diagnosis/probe_band.jl
julia --project=. experiments/nve_diagnosis/component_probe.jl
julia --project=. test/test_nve_sampling.jl
julia --project=. test/test_water_box.jl
```

`probe_band.jl` creates a fresh lattice, relaxes at most 3000 steps, redraws
300 K momenta, and runs 1000 NVE steps without NVT equilibration. Thus these
numbers are short numerical checks, not matched equilibrated production runs.
In particular, the historical corrected 200-water band of 1.255 kJ/mol was
measured after equilibration; comparing it directly with the fresh 200-water
9.303 kJ/mol band below confounds the starting state and observation window.

| Molecules | Seed | alpha (nm^-1) | dt (fs) | max abs(E-E0) (kJ/mol) | HAC slope (kJ/mol/ps) |
| --- | --- | --- | --- | --- | --- |
| 200 | 11 | default 5.059 | 2 | 9.3029 | -0.02 +/- 0.03 |
| 216 | 11 | 3.5 | 1 | 2.2754 | 0.07 +/- 0.03 |
| 216 | 11 | default 5.0 | 1 | 2.2838 | -0.03 +/- 0.03 |
| 216 | 5 | 3.5 | 1 | 2.4832 | 0.10 +/- 0.03 |

After concurrent nonbonded-owner updates, a final rerun
(`a-current-probe-final.log`) measured the same four bands as
9.3009, 2.2718, 2.2776 and 2.4816 kJ/mol. These runs are not a frozen-source
comparison of the other owners' changes; both independently exclude the
historical 609 kJ/mol behavior on the tested fresh states.

The w216 historical band of 609 kJ/mol does not reproduce. The similar bands
for alpha 3.5 and 5.0 do not support attributing the historical large band to
the Ewald split alone. Explicit alpha 3.5 still has finite real-space cutoff
jumps; this is a separate force-model limitation, not evidence of constraint
dissipation. The probe now labels the LJ cutoff and starting potential explicitly,
and reports band per molecule rather than dividing by the number of steps.

The existing full `test_water_box.jl` passed. It independently measured rigid
free-rotor energy-band ratios 4.01 and 4.00 on timestep halving, lab angular
momentum error 3.20e-14, reversal errors below 2.6e-12, and interacting-water
NVE bands 0.961, 0.1871, 0.05362 kJ/mol at successively halved timesteps over
the same 0.5 ps. The live PME momentum drift of 0.2019 in that suite remains
a measured limitation; the zero-force control drifts only 7.92e-12.

## Size and component isolation

`component_probe.jl` runs 0.1 ps from identical starting states at each
timestep. Relaxation is bounded at 100 steps and did not converge; there is
no thermostat. It logs K, LJ/PME potentials, max force, constraint residual,
COM momentum and accumulated final velocity-projection work each step.
It duplicates the production kick/drift/force/kick/projection sequence to
measure energy immediately before and after projection. It uses current
size-dependent cutoff/grid defaults, not the historical settings.

| Waters | Band at 1 fs | Band at 0.5 fs | Ratio |
| --- | --- | --- | --- |
| 27 | 0.231458780 | 0.057722844 | 4.01 |
| 64 | 0.673516176 | 0.168155215 | 4.01 |
| 216 | 1.803184440 | 0.449552305 | 4.01 |
| 1728 | 14.425519500 | 3.597461250 | 4.01 |

Bands are kJ/mol for the entire box. All-step position residuals remained
below 9.9e-13 nm^2. Final velocity-projection work was zero at the logged
precision. The 1728-water band scales as dt^2 in this short test; an exact
replay of the historical 5083 kJ/mol trajectory remains unavailable.

At 27/64 waters, constraint-only bands were 0.0001563/0.0002612 kJ/mol,
and LJ-only bands were 0.02449/0.05926 kJ/mol over 0.1 ps. PME-only controls
ran only 0.02 ps (without LJ repulsion) and had bands 0.22936/0.67040;
their different duration prevents directly ranking band contributions.
Constraint/LJ controls kept COM momentum below 4e-12; full-field COM momentum
reached 0.0555/0.0321 for those sizes and 0.1015 for 1728 waters. This
isolates the previously measured translation-invariance limitation to the
mesh electrostatics path and supplies no evidence of dissipative constraint
projection in these runs. There are no bonded energies in this rigid model.
Thermostat work is identically zero because no O stage is executed.

The full external per-step log is
`$HOME/.cache/biology-mm/nve_component_probe.log`. Its initial SETTINGS
line describes only the 27-water case (alpha 9.86122, PME cutoff 0.456333 nm,
LJ cutoff 0.419081 nm). The script now prints each size's actual settings.

## Confirmed diagnostic defect and correction

`run_nve!` previously evaluated its energy band, HAC fits and maximum residuals
only at saved frames. Increasing `sample_every` hid excursions. On the
8-water seed-11 regression, the same 100-step trajectory reported bands
0.0771153846 and 0.0412118867 kJ/mol with sampling intervals 1 and 100.
The sparse report understated the observed maximum by about 47%.

Diagnostics now evaluate every integration step; `sample_every` controls only
the returned frames. The initial bond error is also included in its maximum.
The baseline failed 11 of 20 regression assertions; the corrected code passes
all 20, including equality of every diagnostics field and unchanged input
state. This corrects reporting and does not alter the trajectory.

## Evidence and limits

The external content/hash baseline is
`/private/var/folders/cg/h5t4h00n5p51252tdk6zlj0c0000gn/T/biology-mm/a-physics-baseline-1794570815326620000`.
External logs are `a-current-probe.log` and `a-water-tests.log` in the same
temporary directory. Baseline regression failure is in Prime vault
`verify_62450`. The constraint worker's
`julia --project=. test/test_constraints.jl` run passed (Prime vault
`verify_70111`). It found no confirmed defect in `Dynamics.jl` or
`Constraints.jl`; neither file was changed by this task. Independent review
also ran the sampling regression and confirmed 20/20 assertions pass.
A subsequent standalone constraint rerun also passed (`verify_30310`), with
200 perturbed rigid waters below 9.86e-13 squared-distance residual and
8.89e-16 momentum error.
The initially blocked `julia --project=. test/test_dynamics.jl` subsequently
passed once the concurrent module dependency was available (Prime vault
`verify_43922`): Verlet order 2.000, constrained RATTLE order 2.005,
momentum error 2.73e-12, and bitwise checkpoint replay.

The historical Langevin production-energy slopes are thermostat exchange and
cannot be interpreted as NVE drift. A 1728-water current short-run diagnostic
and component isolation are recorded separately by `component_probe.jl`.
Neither short-run test establishes bulk equilibration or diffusion, and no
claim that all physics defects are fixed is made. Global verification is left
to the parent integration session while other owners are editing.
