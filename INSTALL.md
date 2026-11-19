# Installation and environment

The Julia core package declares its dependencies in `julia/Project.toml`
(`FFTW` and `JSON` plus Julia stdlibs) with exact versions pinned in
`julia/Manifest.toml`. Everything optional is recorded with its exact
status in `validation/STATUS.md`.

## Required

| Component | Version used here | Purpose |
| --- | --- | --- |
| Julia | 1.13.3 (arm64-apple-darwin) | the core package (`FFTW 1.10.0`, `JSON 1.13.4`, plus stdlib) |
| Lean 4 | 4.35.0 (elan-managed) | `lean/` proofs |
| Rocq | 9.3.0 (opam switch `rocq`) | `rocq/Core.v` |
| Python | 3.9+ (`python3`, `pytest`, `numpy`, `parmed`) | `experiment/` demonstrator and Phase-2 chemistry tests |

In `julia/Project.toml`, the core depends on two registered packages —
`FFTW = "1.10.0"` (used by `PME.jl`) and `JSON = "1.13.4"` (used by `Chemistry`
and `Workflow`) — alongside Julia stdlibs `LinearAlgebra`, `Printf`, `Random`,
`Statistics`, and `Test`. Exact transitive revisions are pinned in
`julia/Manifest.toml` (`julia --project=. -e 'using Pkg; Pkg.instantiate()'`).

## Reproducing this environment

The exact commands used to build the environment in this repository, all outside
the working tree:

```sh
# Julia 1.13.3, no system-wide install
TOOLS="$HOME/.cache/biology-mm"
mkdir -p "$TOOLS" && cd "$TOOLS"
curl -sSL -o julia.tar.gz \
  https://julialang-s3.julialang.org/bin/mac/aarch64/1.13/julia-1.13.3-macaarch64.tar.gz
tar xzf julia.tar.gz && rm julia.tar.gz
export JULIA_DEPOT_PATH="$TOOLS/julia-depot"

# Python 3.12 environment for the optional adapters
uv venv --python 3.12 "$TOOLS/.venv"
uv pip install --python "$TOOLS/.venv/bin/python" \
  numpy scipy pytest MDAnalysis mdtraj pandas physical_validation openmm
```

If Julia lives elsewhere, `verify.sh` takes `JULIA_BIN` and `JULIA_DEPOT_PATH`:

```sh
JULIA_BIN=/usr/local/bin/julia sh verify.sh
```

## NIST reference data (needed for the validation step)

The Lennard-Jones reference configurations are **not** vendored into this
repository — they are NIST material and the archive is fetched at setup time:

```sh
mkdir -p "$HOME/.cache/biology-mm/nist"
curl -sSL -A "Mozilla/5.0" -o /tmp/lj.tar.gz \
  https://www.nist.gov/document/ljsampleconfigurations-targz
tar xzf /tmp/lj.tar.gz -C "$HOME/.cache/biology-mm/nist"
# produces lj_sample_config_periodic{1..4}.txt + metadata.README
```

Source: NIST Standard Reference Simulation Website, *Lennard-Jones Fluid Reference
Calculations: Cuboid Cell*,
<https://www.nist.gov/chemical-informatics-research-group/lennard-jones-fluid-reference-calculations>.
Retrieved 2026-11-11; the bundled `metadata.README` is dated 2013-01-08.

`verify.sh` reads `NIST_DIR` and reports `BLOCKED` with these instructions if the
data is absent, rather than skipping the step silently.

## Optional third-party validation environment

The Molly.jl cross-check needs a separate environment so that Molly's dependency
solve can never perturb the core:

```sh
cd julia
export JULIA_DEPOT_PATH="$HOME/.cache/biology-mm/julia-depot"
julia --project=validation-env -e 'using Pkg; Pkg.develop(path="."); Pkg.add(["Molly","Unitful"])'
```

`julia/validation-env/Project.toml` is checked in with the exact UUIDs.

## What is *not* installed, and why

| Package | Status | Reason |
| --- | --- | --- |
| OpenFF Toolkit | not installed | heavy dependency chain (RDKit, AmberTools via conda); not needed for any claim made here |
| OpenFF Evaluator | not installed | **requires conda** (AmberTools, OpenEye); cannot be installed with `pip`/`uv` on this machine |
| GROMACS/Amber engines | not installed | out of scope; this core is not a replacement backend |

Exact error messages and the reproducible install command for each are in
`validation/STATUS.md`, so a future session can pick any of them up without
rediscovering the blocker.

## Running the Julia test suites directly

```sh
cd julia
export JULIA_DEPOT_PATH="$HOME/.cache/biology-mm/julia-depot"
JULIA="$HOME/.cache/biology-mm/julia-1.13.3/bin/julia"

$JULIA --project=. -e 'using Pkg; Pkg.instantiate()'
$JULIA --project=. test/run_all.jl            # all 34 registered test suites
$JULIA --project=. test/test_factors.jl
$JULIA --project=. test/test_neighbors.jl
$JULIA --project=. test/test_dynamics.jl
$JULIA --project=. test/test_sampling.jl
$JULIA --project=. test/test_polarizable.jl
$JULIA --project=. test/test_prepare.jl
$JULIA --project=. test/test_structure.jl
$JULIA --project=. test/test_shadow.jl          # Lean theorem vs implementation
$JULIA --project=. bench/nist_validate.jl       # note: needs NIST_DIR, see below
$JULIA --project=. bench/benchmark.jl
```

The NIST driver is loaded as a module, not run as a script:

```sh
NIST_DIR="$HOME/.cache/biology-mm/nist" $JULIA --project=. -e '
  include("bench/nist_validate.jl"); using .NISTValidate
  r = NISTValidate.run_nist_validation(ENV["NIST_DIR"]); exit(r.allpass ? 0 : 1)'
```

## Running one explicit simulation specification

From the repository root, after `Pkg.instantiate()`:

```sh
julia --project=julia julia/bin/md.jl --help
julia --project=julia julia/bin/md.jl /absolute/path/run.json
```

The CLI accepts explicit reduced-unit coordinates (one `[x,y,z]` row per atom),
positive masses, three orthorhombic box lengths, and harmonic-bond / LJ-pair
factors. Ensembles are `nve` and `nvt`. Parameter-file chemistry, inferred ligand
parameters, triclinic input, constraints and NPT are refused by this CLI. A minimal
diagnostic specification is:

```json
{
  "schema_version": 1,
  "coordinates": [[1, 1, 1], [2.3, 1, 1]],
  "masses": [1, 1], "box": [20, 20, 20],
  "ensemble": "nve", "workdir": "run",
  "factors": [{"type": "harmonic_bond", "i": 1, "j": 2, "k": 1, "r0": 1.3}],
  "workflow": {
    "run_id": "diagnostic", "dt": 0.001, "temperature": 0,
    "minimize_max_steps": 0, "equilibrate_steps": 0,
    "produce_steps": 100, "frame_every": 5, "checkpoint_every": 0
  },
  "analysis": {
    "min_frames": 10, "fit_window": [0.005, 0.02],
    "min_fit_points": 5, "unwrap": false
  }
}
```

`workdir` is relative to the specification file. This stationary pair checks the
six-stage pipeline; its MSD is not a liquid diffusion measurement. Positive
equilibration and minimization counts can be supplied for dynamical diagnostics.

`trajectory.xyz` is extended XYZ: `Properties=species:S:1:pos:R:3`, column-vector
`Lattice`, periodic flags, `Time`, and explicit length/time labels occur per frame.
Values retain the simulation unit scale; third-party readers often assume
angstrom/fs and require explicit conversion. Read the supported subset with
`FoundationCore.Workflow.TrajectoryFormat.read_extxyz_frames(io)` inside a Julia
`open(path) do io ... end` block. Independent format-reader validation remains
open: ASE is absent and MDAnalysis's plain XYZ reader ignores box/time metadata.
`trajectory.xyz.jsonl` preserves workflow metadata; `trajectory.jsonl` contains
sampled energies/components, temperature, volume and performance diagnostics.
Unavailable pressure/constraint diagnostics are JSON `null`.

Checkpoints are versioned, SHA-256 verified and atomically replaced with a
previous valid `.prev` copy. Corruption or a missing primary can recover from
`.prev`; version/manifest mismatches are refused. Restart through the library
`resume_workflow` API; the CLI has no restart flag. Bitwise continuation requires
the same runtime, algorithm and thread configuration; cross-version migration is
not promised. Statistical-only checkpoints are labelled and rejected when
bitwise recovery is required.

Additional focused suites for primary-session registration in `run_all.jl`:
`test_workflow_recovery.jl`, `test_cli_md.jl`, `test_trajectory_format_extxyz.jl`.
