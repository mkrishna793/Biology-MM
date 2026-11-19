# Production Engineering Design and Correctness Audit

This document records a sandbox implementation and a read-only audit. The new
primitives are prototypes, not yet part of `FoundationCore`, and no production
readiness claim follows from their tests.

## Audit Findings

Ranked by impact. Paths and line numbers refer to the current source tree audited
on 2026-11-12. Proposed changes below are intentionally not applied: source and
shared tests are outside this lane's write boundary.

| Rank | Severity | Location | Finding and minimal failure |
|---|---|---|---|
| 1 | correctness | `julia/src/Prepare.jl:333-334, 380-443` | `rep.close_contact_pairs` indexes the original parsed list, but altloc/occupancy/hydrogen filters change `atoms` before the pairs are consumed. A zero-occupancy atom at index 1 and a close pair `(2,3)`, followed by filtering, can index past the filtered array or repair the wrong atoms. Recompute close contacts after all filters using the filtered coordinates. |
| 2 | correctness | `julia/src/PolarizationOperator.jl:8-9, 111-118`; `julia/src/Polarizable.jl:123-124, 166-172` | Two modules implement opposite signs for the dipole interaction tensor. For two x-separated sites, the x-x off-diagonal element has opposite signs, changing stability and induced dipoles. `test_polarization_operator.jl` validates its documented convention while `test_polarizable.jl` validates the other; these are not a matched-model comparison. Pick one convention from an independent physical/reference derivation and align both code paths and fixtures. |
| 3 | correctness | `julia/src/PolarizationOperator.jl:142,154-167` | The initial residual threshold is `tol*max(norm(E),1)`, but the reported residual is relative to `norm(E)`. A nonzero field with norm `<= tol` returns zero dipoles, status `:converged`, residual 1; `require_converged` accepts them. Reproducer: scale any nonzero charge/field by `1e-12` at default tolerance. Use a scale-consistent relative stopping rule and explicit zero-field handling. |
| 4 | correctness | `julia/src/Dynamics.jl:478-499` | `select_dt` accepts zero or non-finite candidates. With `dts=[0.0]`, the stationary trajectory has zero defect/drift and can yield `accepted=true, dt=0.0`; production integration then never advances. Reject non-positive/non-finite candidate steps. |
| 5 | correctness | `julia/src/Prepare.jl:282-294, 326-329, 431-442` | Default `nudge_distance=1e-3` is much smaller than default contact tolerance `0.5`; a successful `:nudge` can return a still-invalid geometry despite the postcondition claim. Recompute and validate contacts after all nudges, failing if unresolved. |
| 6 | correctness | `julia/src/StreamedLJ.jl:794-795, 908-909` | Accepted finite parameters can overflow in `epsilon[i]*epsilon[j]` or `sigma[i]+sigma[j]` before a representable geometric mean/arithmetic mean is formed. Example epsilons `(1e308,1e308)` yield an infinite product although the mixed epsilon `1e308` is finite. Use `sqrt(epsilon[i])*sqrt(epsilon[j])` and `sigma[i]/2+sigma[j]/2`; add a finite-result regression. |
| 7 | correctness | `julia/src/Sampling.jl:250-258, 262-274` | Composite Simpson weights are used for arbitrary `n`, but require odd sample count and at least 3 points. `n=4` returns a number with invalid quadrature weights. Validate `isodd(n) && n>=3`, and validate finite positive beta and ordered finite limits. |
| 8 | validation-integrity | `julia/test/test_parameter_gradients.jl:242-255` | The test labeled `override` checks `:bond_k_1` against `base_parameters(factors)` rather than `params`, so it finite-differences the stored value, not the `4.2` override. Pass `params` to that check. |
| 9 | correctness | `julia/src/EquilibriumSensitivities.jl:246-291` | Finite sample values can overflow mean/variance arithmetic, with no finite result validation; constant input returns zero SE and can be reported as ordinary success even when it represents a stuck chain. Add arithmetic finite checks and a distinct degenerate-series status; do not interpret constant observations as convergence evidence. |
| 10 | correctness | `julia/src/ResourceProbe.jl:17-21, 23-31` | Commas are stripped before parsing but grouping syntax is not checked; malformed `1,,234` is accepted as 1234. Parse strictly grouped thousands (or no commas) and fail closed. |
| 11 | documentation | `julia/src/Prepare.jl:30-37, 168-189` | Module docs claim MDAnalysis/MDTraj parsing is selected when available, but Julia always calls `read_pdb_fallback`; the implementation comments at 184-189 admit this. Correct the docs to describe the narrow PDB reader and adapter cross-check, or implement parser dispatch. |

The following suspected issues were checked and are not findings: `leapfrog` at
`L=0` applies a gradient-dependent momentum reflection that is still an
involution and has unit absolute Jacobian (though it leaves positions unchanged);
the replica
exchange log-acceptance sign and full-H HMC correction in `Sampling.jl` are
consistent with the stated target; the BAOAB OU noise scale is consistent with
momentum variance `m/beta`; Ewald pair corrections consistently apply
`(scale-1)*prefactor*q_i*q_j/r`; and the streamed-LJ small-cell deduplication
logic is exercised against brute force. These checks are source/test inspection,
not a fresh full test run.

### Proposed Source Diffs (Not Applied)

```diff
--- a/julia/src/Prepare.jl
+++ b/julia/src/Prepare.jl
@@
+# Recompute contacts in the post-filter atom ordering (or refactor the existing
+# diagnosis helper to accept `atoms` directly).
+contacts = Tuple{Int,Int,Float64}[]
+for i in 1:length(atoms)-1, j in i+1:length(atoms)
+    d = norm([atoms[j].x-atoms[i].x, atoms[j].y-atoms[i].y, atoms[j].z-atoms[i].z])
+    d < policy.close_contact_tol && push!(contacts, (i, j, d))
+end
-    if rep.n_close_contacts > 0
+    if !isempty(contacts)
-    if rep.n_close_contacts > 0
--- a/julia/src/PolarizationOperator.jl
+++ b/julia/src/PolarizationOperator.jl
@@
-    normE=norm(E); threshold=tol64*max(normE,1.0)
+    normE=norm(E)
+    threshold=tol64*normE
@@
-    if norm(r)<=threshold
+    if normE == 0.0
+        return DipoleSolveResult(x, 0.0, 0, :converged, spd, mineig)
+    elseif norm(r)<=threshold
--- a/julia/src/Dynamics.jl
+++ b/julia/src/Dynamics.jl
@@
     check_force_model(s)
+    all(dt -> isfinite(dt) && dt > 0, dts) ||
+        throw(ArgumentError("candidate timesteps must be positive and finite"))
--- a/julia/src/StreamedLJ.jl
+++ b/julia/src/StreamedLJ.jl
@@
-    eij = sqrt(Float64(epsilon[i]) * Float64(epsilon[j]))
-    sij = 0.5 * (Float64(sigma[i]) + Float64(sigma[j]))
+    eij = sqrt(Float64(epsilon[i])) * sqrt(Float64(epsilon[j]))
+    sij = 0.5 * Float64(sigma[i]) + 0.5 * Float64(sigma[j])
--- a/julia/src/Prepare.jl
+++ b/julia/src/Prepare.jl
@@ after configured nudges
+            remaining = close_contacts(atoms, policy.close_contact_tol)
+            isempty(remaining) || throw(ArgumentError("nudge left unresolved close contacts"))
--- a/julia/src/EquilibriumSensitivities.jl
+++ b/julia/src/EquilibriumSensitivities.jl
@@ after mean and variance calculations
+    isfinite(μ) && isfinite(γ0) || throw(ArgumentError("sample statistics overflowed"))
@@ constant series
-        return (mean = μ, tau_int = 0.5, ess = Float64(n), se = 0.0)
+        return (mean = μ, tau_int = NaN, ess = NaN, se = NaN, status = :degenerate)
```

```diff
--- a/julia/src/Sampling.jl
+++ b/julia/src/Sampling.jl
@@
 function exact_doublewell_z(d::DoubleWell, beta::Float64; lo = -6.0, hi = 6.0, n = 400_001)
+    isfinite(beta) && beta > 0 || throw(ArgumentError("beta must be positive and finite"))
+    isfinite(lo) && isfinite(hi) && hi > lo || throw(ArgumentError("bounds must be finite and ordered"))
+    n >= 3 && isodd(n) || throw(ArgumentError("Simpson quadrature requires odd n >= 3"))
@@
 function doublewell_moments_exact(d::DoubleWell, beta::Float64;
                                   lo = -6.0, hi = 6.0, n = 400_001)
+    isfinite(beta) && beta > 0 || throw(ArgumentError("beta must be positive and finite"))
+    isfinite(lo) && isfinite(hi) && hi > lo || throw(ArgumentError("bounds must be finite and ordered"))
+    n >= 3 && isodd(n) || throw(ArgumentError("Simpson quadrature requires odd n >= 3"))
--- a/julia/test/test_parameter_gradients.jl
+++ b/julia/test/test_parameter_gradients.jl
@@
-    check_parameter("override", factors, q, ctx, :bond_k_1, base_parameters(factors))
+    check_parameter("override", factors, q, ctx, :bond_k_1, params)
--- a/julia/src/ResourceProbe.jl
+++ b/julia/src/ResourceProbe.jl
@@
-        header = match(r"\bpage\s+size\s+(?:of|:)\s*([0-9][0-9,]*)\s+bytes\b"i, line)
+        header = match(r"\bpage\s+size\s+(?:of|:)\s*((?:[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+))\s+bytes\b"i, line)
@@
-        value = match(r"^\s*Pages\s+(free|inactive|speculative)\s*:\s*([0-9][0-9,]*)\.?\s*$"i, line)
+        value = match(r"^\s*Pages\s+(free|inactive|speculative)\s*:\s*((?:[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+))\.?\s*$"i, line)
--- a/julia/src/Prepare.jl
+++ b/julia/src/Prepare.jl
@@
-The heavy lifting of *syntax* is done by mature packages where they are
-installed — MDAnalysis or MDTraj for PDB/mmCIF — rather than by a hand-rolled
-parser that would fail on the first exotic record. When neither is available the
-module falls back to a deliberately narrow PDB `ATOM`/`HETATM` reader that
-handles only the fixed-column records it claims to handle, and *says so* in its
-report. The fallback is never presented as equivalent to a full parser.
+The Julia core always uses a deliberately narrow fixed-column PDB `ATOM`/`HETATM`
+reader. MDAnalysis cross-checks are available only through the external validation
+adapter; this reader does not parse mmCIF or claim equivalence to a mature parser.
```

## Checkpoint Contract

The prototype in `julia/experiments/production_engineering/Checkpoint.jl`
uses a magic header, format version, network-order payload length, SHA-256, and a
serialized snapshot. The snapshot records `q`, `p`, velocities, box, topology and
parameter IDs, RNG value, integrator state, neighbor/rebuild state, step counter,
named seeds, and engine version. Readers verify framing and checksum, reject an
unsupported version, and check requested topology/parameter/engine/box
compatibility before returning a deep copy. Bump `CHECKPOINT_VERSION` for
incompatible schema changes; retain explicit migrations only when their state
semantics are specified and tested. Serialization is trusted-local Julia data,
not a safe untrusted input format or cross-version interchange format. Atomic
replacement, durability/fsync, and compatibility with actual FoundationCore RNG
and integrator types remain integration work.

`test/runtests.jl` performs a deterministic toy continuation: uninterrupted ten
steps versus save after four, load, and continue six. It compares all state and
bit patterns for q/p/velocities. This is **test evidence for the toy transition
function**, not proof of bit-identical `FoundationCore` trajectories. Existing
stochastic integrator/restart continuation is not yet validated.

## Provenance

`Provenance.provenance_manifest` names seed, package versions, box, cutoff
conventions, thread count, and `model_parameter_hash`. The latter is
`hex(SHA256(canonical_bytes(model_parameters)))`; the manifest digest hashes all
named fields with canonical typed encoding and dictionary keys sorted by name.
Example (generated by `run.jl`):

```text
 Julia 1.13.3 | checkpoint SHA-256 f38a38647bacda44772414df94ffff655c1b1aa015a55cdba7bc7c1bdf4e8235 | provenance SHA-256 69bd7fab7337455effaede89ee7e713f331a26d6ce896ca0d8e523a87259dc5e | restored step 42
```

The digest values depend on exact supplied metadata and model inputs. The test
asserts model hash construction and that changed seed/model parameter changes the
manifest digest. Package metadata must be supplied by the caller; the prototype
does not discover the active environment automatically.

## Error Taxonomy

`ProductionErrors` has typed `InvalidInput`, `SingularGeometry`,
`UnstableOperator`, `NonConvergence`, `UnsupportedFeature`, and
`ResourceRefusal` types. It also separates checkpoint corruption,
incompatibility, unsupported format version, and determinism-contract failures.
The lightweight test constructs each type and verifies its common base type.
These errors are not yet emitted by existing source APIs; wiring and testing
translation boundaries is a required integration gate. No source-path NaN
behavior is changed by this prototype.

## Determinism Guarantees and Limits

`CounterRNG` maps `(seed, stream, explicit index)` to a SHA-256-derived 64-bit
value. `canonical_sum` uses a fixed balanced pairwise tree in input order. The
test runs under Julia `1.13.3` with `--threads=1,2,4,8`, each with a lightweight
fixture (101 indexed draws and 1,001 reduction elements), and all 24 assertions
pass at each setting. Tests establish repeatability and thread-count-independent
indexed values/reduction for that fixture. They do not parallelize or validate a
production force loop. Full-trajectory determinism requires stable event/pair
indices, race-free state updates, canonical accumulation order, and identical
floating-point runtime/platform behavior; it is not claimed across thread counts
for existing threaded LJ, nor across Julia/platform/compiler versions.

## Staged Integration Plan and Acceptance Gates

1. **Contract/API landing:** choose checkpoint schema and actual engine state
   ownership; enumerate every integrator/RNG/neighbor/cache field. Gate: schema
   review and round-trip for each supported integrator, box and topology form.
2. **Safety and recovery:** add atomic write/replace and checksum/version handling
   in the production API. Gate: truncation, bit corruption, wrong IDs, unsupported
   version, interrupted write, and invalid-state tests all fail before state
   mutation; trusted-input boundary is explicit.
3. **Provenance integration:** collect active Julia/package/manifest versions,
   parameters, seed, units, box, cutoff, and thread settings from the run itself.
   Gate: changing each scientific input changes the digest; identical metadata
   regenerates it.
4. **Typed failures:** map domain validation, singular geometry, solver
   instability/non-convergence, unsupported features, and preflight refusals to
   typed errors at public boundaries. Gate: each failure path has a typed test;
   no failed solve can return a usable NaN state or false success flag.
5. **Determinism policy:** choose stable pair/random indices and fixed reduction
   semantics for the actual force/integrator kernels. Gate: same small seeded
   trajectory at 1/2/4/8 threads matches the declared tolerance/bitwise policy;
   separate tests cover reductions, stochastic streams, restart continuation,
   and repeatability.
6. **Release evidence:** run shared tests only after integration owners finish,
   then measure resource/performance separately. Gate: correctness suite passes;
   benchmarks are separately attributed and do not substitute for physics
   validation.

## Evidence Ledger

- **Proven by tests in this sandbox:** checkpoint envelope round-trip, corrupted
  payload refusal, requested-ID mismatch refusal, deterministic toy restart
  continuation bitwise equality, canonical hash behavior, taxonomy construction,
  and lightweight indexed RNG/reduction fixtures at 1/2/4/8 configured threads.
- **Measured:** Julia `1.13.3`; current shell default was one thread; each explicit
  thread run reports its configured count. The sandbox test commands and demo
  were run with:

  ```sh
  export JULIA_DEPOT_PATH="$HOME/.cache/biology-mm/julia-depot"
  JULIA="$HOME/.cache/biology-mm/julia-1.13.3/bin/julia"
  "$JULIA" --threads=1 --project=julia julia/experiments/production_engineering/test/runtests.jl
  "$JULIA" --threads=2 --project=julia julia/experiments/production_engineering/test/runtests.jl
  "$JULIA" --threads=4 --project=julia julia/experiments/production_engineering/test/runtests.jl
  "$JULIA" --threads=8 --project=julia julia/experiments/production_engineering/test/runtests.jl
  "$JULIA" --project=julia julia/experiments/production_engineering/run.jl
  ```
- **Not measured:** heavy benchmarks, RSS/scaling, production threaded force
  determinism, actual FoundationCore restart equivalence, statistical sampling
  quality, and performance impact.
- **Assumed:** callers include all scientifically relevant state and provenance
  inputs; explicit indices and canonical order are stable; checkpoint files are
  trusted; all serialized engine-specific values faithfully represent restart
  state.

## Main-Line Blockers

Resolve the ranked correctness findings, especially filtered contact indices and
the dipole-sign conflict, before broadening production claims. Then decide the
actual engine checkpoint schema and wire typed errors and provenance through the
production APIs. The audit did not apply any suggested source diff and did not
run the shared suite or `verify.sh`.
