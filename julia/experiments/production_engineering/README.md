# Production engineering primitives

Standalone, stdlib-only demonstrations of restart state, provenance, error
classification, and limited deterministic computation. This is not an OpenMM
checkpoint format, a complete molecular-engine state adapter, or a claim of
bitwise-identical simulation trajectories across Julia versions/platforms.

Run from the repository root with the documented Julia installation and depot:

```sh
export JULIA_DEPOT_PATH="$HOME/.cache/biology-mm/julia-depot"
JULIA="$HOME/.cache/biology-mm/julia-1.13.3/bin/julia"
"$JULIA" --project=julia julia/experiments/production_engineering/test/runtests.jl
"$JULIA" --project=julia julia/experiments/production_engineering/run.jl
```

## Contracts and limits

- `Checkpoint.SimulationSnapshot` records positions `q`, momenta `p`, velocities,
  box, topology and parameter identifiers, RNG state, integrator state, neighbor
  list/rebuild state, step, named seeds, and engine version. The caller is
  responsible for including every engine-specific state variable in the supplied
  integrator/neighbor/RNG values. The checkpoint uses Julia `Serialization`, a
  versioned envelope, payload length, and SHA-256. It is for trusted local files:
  Julia Serialization is not a safe untrusted-input format and is not a portable
  interchange format. Checksum/format and requested compatibility checks run
  before a snapshot is returned.
- `Provenance.provenance_manifest` records named fields for the seed, package
  versions, box, cutoff conventions, thread count, and actual SHA-256 model/
  parameter hash. `provenance_digest` hashes those named fields using canonical
  typed serialization. Dict insertion order does not affect the digest. Record
  exact package versions/manifest data; omitted data cannot be identified by
  the hash.
- `CounterRNG` produces a SHA-256-derived value at an explicit nonnegative index;
  this does not use thread IDs or mutable task-local streams. `canonical_sum`
  uses a fixed balanced pairwise tree over the input order. Thus the random value
  at each index and the reduction result are independent of thread count when
  indices, values, order, and supported runtime behavior are identical. This
  does **not** guarantee bitwise-identical full trajectories: callers must
  assign stable indices, avoid race-dependent updates, preserve canonical input
  order, and account for platform/compiler/math-library differences.
- Structured exceptions include distinct `InvalidInput`, `SingularGeometry`,
  `UnstableOperator`, `NonConvergence`, `UnsupportedFeature`, and
  `ResourceRefusal` types, plus checkpoint corruption/incompatibility/version
  errors and determinism-contract violations.

The test writes its checkpoint to a thread-count-specific file under `test/`
inside this directory and leaves it there. This lets separate Julia processes
run the lightweight thread-count checks without sharing a file. The demo writes
and removes its checkpoint inside this directory.
