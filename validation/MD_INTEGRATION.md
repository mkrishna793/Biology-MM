# MD integration evidence — 2026-11-13

The integration adds explicit `WaterBox.water_field` options
`neighbor_cache=false` and `neighbor_cache_skin=nothing`, forwarded to
StreamedLJ. Defaults remain unchanged. The fixed orthorhombic WaterBox context
does not gain changing-box dynamics. A 12-step composed-water comparison checks
trajectory and force parity, fresh-cache controls, reuse and forced rebuilds.

`WorkflowConfig(algorithm=:respa_nve, fast_factors=[1], respa_m=3,
temperature=0, gamma=nothing)` selects an unconstrained NVE splitting. Fast
factors are explicit indices, unique type names, or indexed type names such as
`1:HarmonicBond`; every remaining factor is slow. Ambiguous names, duplicates,
invalid indices, constraints and unsupported force models are refused. The
resolved partition and substep count enter the manifest digest. This is not a
speedup or resonance-safety claim. The current Workflow adapter reconstructs
its RESPA buffers each outer step. The standalone module retains reusable
buffers. SETTLE remains a separate analytic module: its projection is not a
validated complete WaterBox integrator or thermostat.

The JSON CLI accepts `integrator: "respa_nve"` with workflow fields
`fast_factors` and `respa_m`, for ensemble `nve`. Prepared AMBER execution uses
`normalized_amber` (a normalized JSON file path relative to the specification),
explicit N×3 `coordinates`, `units: "nm,kJ/mol,amu,ps"`, and
`protocol: "direct-nocutoff-flexible"`. This route builds its own forces through
Chemistry and ResolvedSystem. It refuses periodic imported models, constraints
and virtual sites. It does not parameterize raw chemistry. The normalized
record and protocol enter the manifest. CLI restart execution is still absent;
checkpoint serialization alone does not provide a CLI resume operation.

The real repository `DNA_mbondi3` AMBER fixture completed all six Workflow stages
through the CLI with **628 atoms, 20 steps, dt=0.0001 ps, 0.002 ps simulated**.
It used flexible, unconstrained, boxless direct interactions and zero initial
momenta. Whole-process elapsed time was **16.73 s**, including startup and
compilation, with **1,002,438,656 B peak RSS**. The production loop reported
0.9723 s including diagnostics and output. Sampled total energies ranged over
**9.0154 kJ/mol**, from 18227.3553 at the first recorded step to 18221.7825
kJ/mol at the last. Constraints are absent, so constraint residual is not
applicable. The initial energy is not included in that sampled band.

A separate warmed full Velocity-Verlet measurement on the same molecular
fixture took **0.0038342 s per full step**, allocating **13,250,240 B over 20
steps**, with **648,200,192 B process peak RSS**. These are actual force-and-update
steps, not force-only kernel timings. This one short smoke run establishes
execution only. It does not establish equilibrium, diffusion, energy accuracy
at production timesteps, or large-system throughput. The CLI's generic MSD
number is not a physical diffusion claim for this smoke trajectory.

Integration suites are `test_waterbox_cache`, `test_workflow_respa`, and
`test_cli_imported`; the runner contains 45 suites. Provider wallet exhaustion
prevented all requested independent review/verifier workers from completing.
The integration therefore remains partial until independent audit and the
frozen-tree verification outcome are reviewed. External evidence and the
pre-edit content baseline are recorded in the parent session's vault.
