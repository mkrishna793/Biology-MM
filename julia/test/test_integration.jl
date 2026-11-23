# Cross-module integration tests.
#
# Every existing suite tests its own module in isolation. This file tests the
# *seams*: structure file -> typed state -> factor assembly -> energy/forces ->
# integrator -> sampler. Each test measures a number and asserts against a
# tolerance justified by the mathematics of the seam, not by convenience.
#
# Test policy (same as the other suites): a test that cannot justify its
# tolerance scientifically was not written.

using FoundationCore
using FoundationCore.Types
using FoundationCore.Factors
using FoundationCore.Neighbors
using FoundationCore.Dynamics
using FoundationCore.Sampling
using FoundationCore.Prepare
using Random, Printf, LinearAlgebra, Statistics

const T = FoundationCore.Types
const FA = FoundationCore.Factors
const NB = FoundationCore.Neighbors
const DY = FoundationCore.Dynamics
const SA = FoundationCore.Sampling
const PR = FoundationCore.Prepare
const RS = FoundationCore.ResolvedSystem
const SC = FoundationCore.SystemContract
const SL = FoundationCore.StreamedLJ
const PM = FoundationCore.PME
const EW = FoundationCore.Electrostatics
const CN = FoundationCore.Constraints
const MB = FoundationCore.MonteCarloBarostat
const PO = FoundationCore.Polarizable
const POP = FoundationCore.PolarizationOperator

# The real protein fixture already used by test_structure.jl (OpenMM corpus,
# BPTI, 892 atoms). Resolved relative to this file so the suite is location
# independent.
const BPTI_PATH = normpath(joinpath(@__DIR__, "..", "..",
                                    "wrappers", "python", "tests", "systems", "bpti.pdb"))

# ---------------------------------------------------------------------------
# 1. structure -> state -> energy -> forces -> dynamics on a real PDB
# ---------------------------------------------------------------------------
"""
The full pipeline on bpti.pdb: diagnose (892 atoms, no close contacts), build
an AtomSet in a periodic box, assemble LJ factors from a neighbour list,
evaluate energy + forces, and run velocity Verlet without producing a
non-finite number.

Tolerances are justified as follows:
* atom/contact counts are properties of the file (fixture, 892 heavy+light
  atoms from the OpenMM corpus);
* Newton's third law: every pair force is scattered as +g/-g, so the total
  force must cancel to accumulation roundoff (~eps*sqrt(npairs)*max|F|);
* the dynamics assertion is finiteness — the configuration is a crystal with
  neutral charges and generic LJ parameters, so it is NOT expected to be
  physically meaningful, only numerically valid. dt is chosen small enough
  (dt*max|F|/m << typical sigma over 20 steps) that the short-time energy
  change is a shadow-Hamiltonian oscillation.
"""
function test_structure_to_dynamics_pipeline()
    @assert isfile(BPTI_PATH) "real structure fixture missing: $BPTI_PATH"
    rep = PR.diagnose_structure(BPTI_PATH; contact_tol = 0.5)
    @printf("  diagnose: n_atoms=%d close_contacts=%d min_dist=%.4f nonfinite=%d dups=%d altlocs=%d\n",
            rep.n_atoms, rep.n_close_contacts, rep.min_pair_distance,
            rep.n_nonfinite, rep.n_duplicate_identities, rep.n_altloc)
    @assert rep.n_atoms == 892 "expected 892 atoms in bpti.pdb, got $(rep.n_atoms)"
    @assert rep.n_close_contacts == 0 "close contacts found: $(rep.close_contact_pairs[1:min(3, end)])"
    @assert rep.n_nonfinite == 0 && rep.n_duplicate_identities == 0
    @assert rep.min_pair_distance > 0.9 "min distance $(rep.min_pair_distance) too short for force evaluation"

    atoms, _, _ = PR.read_pdb_fallback(BPTI_PATH)
    N = length(atoms)
    q = zeros(3, N)
    for (i, a) in enumerate(atoms)
        q[1, i] = a.x; q[2, i] = a.y; q[3, i] = a.z
    end
    # put the molecule into a box comfortably larger than its span so the
    # minimum-image convention (|d| < L/2) is valid for every pair
    span = vec(maximum(q, dims = 2) .- minimum(q, dims = 2))
    L = maximum(span) + 20.0
    q .-= minimum(q, dims = 2)
    @assert all(0.0 .<= q .< L) "positions must lie inside the primary cell"
    box = T.OrthorhombicBox((L, L, L))

    # typed state: unit masses (Prepare.element_mass in reduced units), zero
    # charges, real atom names — the AtomSet constructor itself re-validates
    as = T.AtomSet(q, zeros(3, N), ones(N), zeros(N), [a.name for a in atoms])
    sys = T.System(as, box)

    rc = 8.0
    nl = NB.build_neighbors(q, box, rc, 0.4)
    pairs = NB.neighbor_pairs(nl)
    @assert !NB.needs_rebuild!(nl, q) "fresh list must not need rebuild"
    ctx = FA.Context(box, T.REDUCED, N)
    facs = FA.AbstractFactor[
        FA.LennardJonesPair(i, j, 0.05, 2.0; mode = :lfs, rc = rc) for (i, j) in pairs]
    U0, F0 = FA.total_energy_forces(facs, q, ctx)
    scaleF = maximum(abs, F0)
    net = norm(sum(F0, dims = 2)) / scaleF
    @printf("  %d atoms in %.3f box, %d LJ pairs within rc=%.1f: U=%.6f finite=%s max|F|=%.4f\n",
            N, L, length(facs), rc, U0, isfinite(U0), scaleF)
    @printf("  net force / max force = %.3e (Newton's third law, ~eps accumulation)\n", net)
    @assert isfinite(U0) "energy must be finite for a diagnosed-clean structure"
    @assert all(isfinite, F0) "forces must be finite"
    @assert net < 1e-12 "action-reaction cancellation failed: $net"

    # dynamics seam: 20 velocity-Verlet steps, every energy must stay finite
    hs = DY.HarmonicSystem(facs, ones(N), ctx)
    st = DY.State(copy(q), zeros(3, N))
    dt = 2.5e-4
    Es = Float64[]
    for _ in 1:20
        DY.velocity_verlet_step!(st, hs, dt)
        push!(Es, DY.total_energy(hs, st.q, st.p))
    end
    @assert all(isfinite, Es) "a non-finite energy appeared during Verlet: $Es"
    rel_change = abs(Es[end] - Es[1]) / abs(Es[1])
    @printf("  20 Verlet steps (dt=%.1e): E_1=%.6f E_20=%.6f rel|ΔE|=%.3e all_finite=true\n",
            dt, Es[1], Es[end], rel_change)
    # short-time change of a symplectic run is an O(dt^2) oscillation, not a
    # leak; 2% over 20 steps is far above the expected value yet far below
    # anything a broken pipeline (first-order scheme, stale force) would give
    @assert rel_change < 0.02 "energy moved $(100*rel_change)% in 20 steps: pipeline not behaving symplectically"
    return true
end

# ---------------------------------------------------------------------------
# 2. neighbour staleness within the skin must not change the energy
# ---------------------------------------------------------------------------
"""
The `Neighbors`<->`Factors` seam. If every atom moves less than skin/2, every
pair inside `rc` at the new positions was already inside `rc + skin` at the
old ones, so a stale list and a rebuilt list contain the same *energetically
active* pairs; all extra pairs contribute exactly 0.0 under the C^1 `:lfs`
cutoff. The energies must therefore agree to accumulation roundoff only.

The lists must actually differ, otherwise the test asserts nothing about
staleness — the set-difference counts are part of the assertion.
"""
function test_stale_neighbor_list_energy_invariance()
    rng = MersenneTwister(404)
    L, rc, skin = 10.0, 2.5, 0.4
    box = T.OrthorhombicBox((L, L, L))
    # jittered lattice so the reference configuration is well separated
    n = 100
    q0 = zeros(3, n)
    c = 0
    for k in 1:4, j in 1:5, i in 1:5
        c += 1
        q0[1, c] = (i - 0.5) * 2.0 + (rand(rng) - 0.5) * 0.7
        q0[2, c] = (j - 0.5) * 2.0 + (rand(rng) - 0.5) * 0.7
        q0[3, c] = (k - 0.5) * 2.5 + (rand(rng) - 0.5) * 0.7
    end
    nl_stale = NB.build_neighbors(q0, box, rc, skin)
    pairs_stale = NB.neighbor_pairs(nl_stale)

    # move every atom by strictly less than skin/2
    disp = randn(rng, 3, n)
    disp ./= sqrt.(sum(abs2, disp, dims = 1))
    mag = 0.49 * skin * rand(rng, 1, n)
    q1 = q0 .+ disp .* mag
    maxdisp = maximum(mag)
    @printf("  max displacement = %.4f < skin/2 = %.4f\n", maxdisp, skin / 2)
    @assert maxdisp < skin / 2
    @assert !NB.needs_rebuild!(nl_stale, q1) "displacements below skin/2 must not trigger the rebuild criterion"

    nl_fresh = NB.build_neighbors(q1, box, rc, skin)
    pairs_fresh = NB.neighbor_pairs(nl_fresh)
    only_stale = length(setdiff(pairs_stale, pairs_fresh))
    only_fresh = length(setdiff(pairs_fresh, pairs_stale))
    @printf("  stale pairs=%d fresh pairs=%d (only_stale=%d only_fresh=%d -> the lists really differ)\n",
            length(pairs_stale), length(pairs_fresh), only_stale, only_fresh)
    @assert only_stale > 0 && only_fresh > 0 "stale and fresh lists are identical: test would be vacuous"

    ctx = FA.Context(box, T.REDUCED, n)
    make_facs(pairs) = FA.AbstractFactor[
        FA.LennardJonesPair(i, j, 1.0, 1.0; mode = :lfs, rc = rc) for (i, j) in pairs]
    Us, Fs = FA.total_energy_forces(make_facs(pairs_stale), q1, ctx)
    Uf, Ff = FA.total_energy_forces(make_facs(pairs_fresh), q1, ctx)
    relU = abs(Us - Uf) / max(abs(Uf), 1e-30)
    relF = maximum(abs, Fs - Ff) / max(maximum(abs, Ff), 1e-30)
    @printf("  U_stale=%.13f  U_fresh=%.13f  rel|ΔU|=%.3e  rel|ΔF|=%.3e\n", Us, Uf, relU, relF)
    # only the summation order of the *nonzero* terms can differ; the differing
    # pairs are exactly zero under :lfs, so agreement is accumulation roundoff
    @assert relU < 1e-14 "stale list changed the energy: rel |ΔU| = $relU"
    @assert relF < 1e-14 "stale list changed the forces: rel |ΔF| = $relF"
    return true
end

# ---------------------------------------------------------------------------
# 3. exactly-summarised bipartite group vs explicit pair enumeration at n=m=40
# ---------------------------------------------------------------------------
"""
The closed-form (centroid/residual) summary must reproduce the explicit
n*m-pair model — not just the energy but *every per-atom force*, at a size
where O(nm)=1600 pair evaluations is the honest reference and the summary's
80-coordinate read actually saves something. Tolerance 1e-12 relative: the
two paths are algebraically identical and differ only by float64 summation
roundoff (~sqrt(40)*eps per atom).
"""
function test_bipartite_summary_vs_explicit_pairs()
    rng = MersenneTwister(77)
    na = 40; nb = 40
    A = randn(rng, 3, na)
    B = 1.7 .* randn(rng, 3, nb)
    q = [A B]
    k = 0.37
    ctx = FA.Context(T.REDUCED, na + nb)
    bp = FA.BipartiteHarmonicGroup(collect(1:na), collect(na+1:na+nb), k)
    @assert !FA.is_local(bp) && FA.factor_support_size(bp) == na + nb
    Usum, Fsum = FA.total_energy_forces([bp], q, ctx)

    # explicit enumeration of exactly the same model
    Udir = 0.0
    Fdir = zeros(3, na + nb)
    for i in 1:na, j in 1:nb
        d = q[:, i] .- q[:, na + j]
        Udir += 0.5 * k * sum(abs2, d)
        Fdir[:, i] += -k * d          # F = -dU/dq_i for this pair
        Fdir[:, na + j] += k * d
    end
    relU = abs(Usum - Udir) / abs(Udir)
    relF = maximum(abs, Fsum - Fdir) / max(maximum(abs, Fdir), 1e-30)
    worst_atom = 0.0
    for j in 1:na+nb
        sc = max(maximum(abs, Fdir[:, j]), 1e-12)
        worst_atom = max(worst_atom, maximum(abs, Fsum[:, j] - Fdir[:, j]) / sc)
    end
    @printf("  n=m=40, k=%.2f: U_summary=%.10f U_explicit=%.10f (1600 pair evals)\n", k, Usum, Udir)
    @printf("  rel|ΔU|=%.3e  rel|ΔF|_max=%.3e  worst per-atom rel|ΔF|=%.3e\n", relU, relF, worst_atom)
    @assert relU < 1e-12 "summary energy deviates from explicit enumeration: $relU"
    @assert relF < 1e-12 "summary forces deviate from explicit enumeration: $relF"
    @assert worst_atom < 1e-12 "a per-atom force differs from the explicit model: $worst_atom"
    return true
end

# ---------------------------------------------------------------------------
# shared mixed-factor system (used by tests 4 and 5)
# ---------------------------------------------------------------------------
"""
A deliberately mixed small system: 2 harmonic bonds, 1 harmonic angle,
2 periodic dihedrals, 3 cutoff Lennard-Jones pairs (`:lfs`, so force
consistency holds at the cutoff too), 3 Coulomb pairs — all on the same 8
atoms. The geometry is seeded, well separated, and verified non-degenerate
(no singular or cusp configurations) at construction.
"""
function mixed_system()
    rng = MersenneTwister(4711)
    n = 8
    q = randn(rng, 3, n) * 1.3
    for d in 1:3, j in 1:n
        q[d, j] += (j - 1) * 1.05
    end
    ctx = FA.Context(T.REDUCED, n)
    rc_m = 3.5
    facs = FA.AbstractFactor[
        FA.HarmonicBond(1, 2, 2.0, 1.0),
        FA.HarmonicBond(2, 3, 1.8, 1.05),
        FA.HarmonicAngle(1, 2, 3, 1.5, 1.9),
        FA.PeriodicDihedral(1, 2, 3, 4, 0.8, 2, 0.3),
        FA.PeriodicDihedral(2, 3, 4, 5, 0.5, 1, 1.0),
        FA.LennardJonesPair(5, 6, 0.6, 1.0; mode = :lfs, rc = rc_m),
        FA.LennardJonesPair(2, 4, 0.6, 1.0; mode = :lfs, rc = rc_m),
        FA.LennardJonesPair(7, 8, 0.6, 1.0; mode = :lfs, rc = rc_m),
        FA.CoulombPair(1, 4, 0.7, -0.7),
        FA.CoulombPair(3, 8, 0.5, -0.5),
        FA.CoulombPair(5, 6, -0.4, 0.4),
    ]
    # the configuration must actually exercise the mixture
    F = zeros(3, n)
    for (c, f) in enumerate(facs)
        U = FA.evaluate!(F, f, q, ctx)
        @assert isfinite(U) "factor $c is singular at the reference configuration"
    end
    lj_energies = [FA.evaluate!(F, f, q, ctx) for f in facs if f isa FA.LennardJonesPair]
    @assert count(x -> x != 0.0, lj_energies) >= 2 "LJ pairs are inert at rc: the set is not a real mix"
    return q, facs, ctx
end

# ---------------------------------------------------------------------------
# 4. F = -grad U for the MIXED factor set (composition/adjoint seam)
# ---------------------------------------------------------------------------
"""
Central finite differences over every one of the 24 coordinates, comparing
`-grad_fd U` with the forces scattered by the composed factor set. This is
the property no single-module suite checks: adjoint scatter and FD pass per
factor, but the *composition* (shared accumulator, overlapping supports —
atom 2, 3, 4 appear in bond, angle, dihedral and LJ simultaneously) could
double-count or mis-sign a contribution and every isolated test would miss it.

Tolerance 1e-5 on the max relative error: central differences with h=1e-6
have truncation O(h^2 U''') and roundoff O(eps U / h) — measured agreement is
~1e-9 (see printed value), so 1e-5 is 3+ orders of margin while still failing
loudly for any structural error (a sign flip or double count is O(1)).
"""
function test_mixed_set_force_is_gradient()
    q, facs, ctx = mixed_system()
    U0, F0 = FA.total_energy_forces(facs, q, ctx)
    @assert isfinite(U0)
    h = 1e-6
    G = zeros(size(q))
    for j in 1:size(q, 2), d in 1:3
        qp = copy(q); qp[d, j] += h
        qm = copy(q); qm[d, j] -= h
        G[d, j] = (FA.total_energy_forces(facs, qp, ctx)[1] -
                   FA.total_energy_forces(facs, qm, ctx)[1]) / (2h)
    end
    scale = max(maximum(abs, G), 1e-12)
    relerr = maximum(abs, F0 .+ G) / scale
    abserr = maximum(abs, F0 .+ G)
    @printf("  mixed set (%d factors: bond+angle+dihedral+LJ+Coulomb), U=%.6f\n", length(facs), U0)
    @printf("  all 24 coordinates: max|F + grad_fd U| = %.3e   max relative = %.3e (tol 1e-5)\n", abserr, relerr)
    @assert relerr < 1e-5 "force is not the FD gradient of the composed energy: rel $relerr"
    return true
end

# ---------------------------------------------------------------------------
# 5. energy conservation of the mixed system under velocity Verlet
# ---------------------------------------------------------------------------
"""
2000 steps at dt=0.002 of the full mixed system through the real integrator,
judged by the package's own drift statistic (fitted slope extrapolated over
the run, divided by the oscillation band). A symplectic run scores ~0; a
linear leak scores 2.0; the built-in verdict threshold is 0.5. The additional
`|slope|*T < max|ΔE|` assertion requires the extrapolated drift to stay below
the observed excursion — the same statistic style as test_dynamics.jl.
"""
function test_mixed_system_energy_conservation()
    q, facs, ctx = mixed_system()
    n = size(q, 2)
    hs = DY.HarmonicSystem(facs, ones(n), ctx)
    p = randn(MersenneTwister(5), 3, n) * 0.1
    p .-= sum(p, dims = 2) / n          # COM at rest: a pure internal motion
    st = DY.State(copy(q), p)
    recs, es = DY.run_trajectory(st, hs, 0.002, 2000; record_every = 10)
    @assert all(isfinite, es.compensated) "non-finite energy during the mixed run"
    rep = DY.energy_drift_report(recs, es)
    T_run = recs[end].t - recs[1].t
    @printf("  E0=%.6f Efinal=%.6f max|ΔE|=%.3e slope=%.3e  slope*T=%.3e\n",
            rep.E0, rep.Efinal, rep.max_abs_dev, rep.slope, abs(rep.slope) * T_run)
    @printf("  predicted_drift/band = %.4f (threshold %.1f) -> bounded=%s\n",
            rep.drift_fraction, DY.DRIFT_FRACTION_MAX, rep.bounded)
    @assert rep.drift_fraction < 0.5 "mixed-system Verlet drifts: fraction $(rep.drift_fraction)"
    @assert rep.bounded
    @assert abs(rep.slope) * T_run < rep.max_abs_dev "drift extrapolation exceeds the oscillation band"
    return true
end

# ---------------------------------------------------------------------------
# 6. HMC whose potential is produced by the Factors machinery
# ---------------------------------------------------------------------------
"""
`Sampling.hmc_trajectory` is driven by `U(q)` and `gradU(q)` that both call
`Factors.total_energy_forces` on a real 2-atom bond system (q is the 1-D
COM-projected bond coordinate: atom 1 at +v, atom 2 at -v). The gradient is
the *scattered force*, mapped through the projection — the sampler therefore
consumes the factor adjoint, not a hand-written closure.

Target check: U(v) = (k/2)(2v)^2 = 2k v^2, so the Boltzmann marginal is
Gaussian with variance 1/(4k beta), exactly. Assertions:
* acceptance > 0.3 (dt*omega = 0.4*sqrt(4k) < 2 keeps the leapfrog stable);
* sampled mean and variance within 5 autocorrelation-corrected standard
  errors of the analytic values (the style of test_sampling.jl / BAOAB);
* Metropolis-criterion audit: with p_i = min(1, exp(-beta dH_i)) from the
  recorded dH series, the observed acceptance fraction is a Bernoulli sum
  conditional on p_i, so |mean(p) - observed| has standard error
  sqrt(mean(p(1-p))/n). Agreement to within 3 SE verifies that the accepted/
  rejected decision used the FULL Hamiltonian dH (energy + kinetic). If the
  integrator-only or potential-only error leaked into the criterion, the two
  numbers would separate far beyond 3 SE.
"""
function test_hmc_on_factor_potential()
    k, beta = 0.5, 1.5
    facs = FA.AbstractFactor[FA.HarmonicBond(1, 2, k, 0.0)]
    ctx = FA.Context(T.NoBox(), T.REDUCED, 2)
    qmat = zeros(3, 2)
    U(v::Vector{Float64}) = begin
        qmat[1, 1] = v[1]; qmat[1, 2] = -v[1]
        FA.total_energy_forces(facs, qmat, ctx)[1]
    end
    # dU/dv = (dU/dq1)(dq1/dv) + (dU/dq2)(dq2/dv) = -F1 + F2 (x components)
    gradU(v::Vector{Float64}) = begin
        qmat[1, 1] = v[1]; qmat[1, 2] = -v[1]
        _, F = FA.total_energy_forces(facs, qmat, ctx)
        [F[1, 2] - F[1, 1]]
    end
    # the projected gradient must itself be FD-consistent (projection seam)
    vtest = 0.83
    gfd = (U([vtest + 1e-6]) - U([vtest - 1e-6])) / 2e-6
    relg = abs(gradU([vtest])[1] - gfd) / abs(gfd)
    @printf("  factor-supplied potential: U(2)=%.6f (exact v^2 form %.6f), grad vs FD rel=%.3e\n",
            U([2.0]), 2k * 4, relg)
    @assert relg < 1e-8 "projected factor gradient is not FD-consistent: $relg"

    dt, L, nsteps = 0.4, 20, 60_000
    rng = MersenneTwister(607)
    res = SA.hmc_trajectory([0.9], U, gradU, [1.0], dt, L, nsteps, beta, rng)
    xs = vec(res.samples)
    @assert all(isfinite, xs) && all(isfinite, res.dH) "non-finite HMC trajectory data"
    var_exp = 1.0 / (4 * k * beta)
    tau = SA.integrated_autocorrelation(xs)
    n_eff = length(xs) / tau
    se_mu = sqrt(var_exp / n_eff)
    se_v = sqrt(2 * var_exp^2 / n_eff)
    μ, v = mean(xs), var(xs)
    @printf("  HMC acc=%.4f  mean=%.4f ± %.4f (exact 0)  var=%.4f ± %.4f (exact %.4f)  tau=%.2f\n",
            res.acceptance_rate, μ, se_mu, v, se_v, var_exp, tau)
    @assert res.acceptance_rate > 0.3 "acceptance $(res.acceptance_rate) too low to sample"
    @assert abs(μ - 0.0) < 5 * se_mu + 1e-3 "HMC mean inconsistent with Boltzmann target"
    @assert abs(v - var_exp) < 5 * se_v + 1e-3 "HMC variance inconsistent with Boltzmann target"

    pvec = @. min(1.0, exp(-beta * res.dH))
    pred = mean(pvec)
    se_diff = sqrt(mean(pvec .* (1 .- pvec)) / length(pvec))
    diff = abs(pred - res.acceptance_rate)
    @printf("  Metropolis audit: mean[min(1,e^-b dH)]=%.5f  observed=%.5f  |diff|=%.5f  SE=%.5f  (%.2f SE)\n",
            pred, res.acceptance_rate, diff, se_diff, diff / se_diff)
    @assert diff < 3 * se_diff + 1e-9 "acceptance not consistent with the full-Hamiltonian criterion"
    return true
end

# ---------------------------------------------------------------------------
# 7. ResolvedSystem + StreamedLJ + Constraints + Box Derivative + Restart seam
# ---------------------------------------------------------------------------
"""
Cross-phase seam: `ResolvedSystem` + `StreamedLJFactor` (`:lfs`) + `HarmonicBond`
+ `CoupledConstraintSet` -> `SystemContract` (`energy_forces`, `box_derivative`,
`virial`, `pressure` with static `constraint_virial`) -> `restart_system` box
change + `validate_restart_state` constraint refusal and re-projection.
"""
function test_resolved_streamed_constraints_pressure_and_restart_seam()
    box = T.OrthorhombicBox((10.0, 10.0, 10.0))
    N = 4
    masses = [12.0, 12.0, 16.0, 1.0]
    charges = [0.0, 0.0, 0.0, 0.0]
    q = [1.0 2.1 5.0 6.1;
         1.0 1.2 5.0 5.4;
         1.0 1.1 5.0 5.1]
    p = [0.2 -0.2 0.1 -0.1;
         -0.1 0.1 0.2 -0.2;
         0.05 -0.05 0.0 0.0]
    d0 = 1.2
    cs = CN.CoupledConstraintSet(N, [(3, 4, d0)])
    CN.project!(q, p, masses, cs)

    bond = FA.HarmonicBond(1, 2, 50.0, 1.0)
    topo = SL.PairTopology(excluded = [(1, 2), (3, 4)])
    chem_topo = T.ChemicalTopology(; exclusions = Set([(1, 2), (3, 4)]))
    lj = SL.StreamedLJFactor(q, fill(0.4, N), fill(1.0, N), box;
                             rcut = 3.5, mode = :lfs, topology = topo)
    rs = RS.ResolvedSystem(;
        masses = masses, charges = charges, names = [:C1, :C2, :O, :H],
        box = box, topology = chem_topo,
        factors = FA.AbstractFactor[bond, lj],
        component_names = [:bond, :lj], constraints = cs,
        provenance = RS.Provenance(; source = "integration", force_field = "toy"),
        declared = [:energy, :cartesian_forces, :component_energies, :box_derivative],
    )
    @assert RS.validate_chemistry(rs) === rs
    @assert RS.validate_geometry(rs, q)
    @assert SC.check_numerical_stability(rs, q)

    U, F = SC.energy_forces(rs, q)
    ce = SC.component_energies(rs, q)
    @assert abs(U - (ce.bond + ce.lj)) < 1e-13 "component sum mismatch: $U vs $(ce.bond + ce.lj)"

    # Fixed-fractional box derivative vs central FD across all 3 orthorhombic axes
    G = SC.box_derivative(rs, q)
    s0 = SC.fractional_coordinates(rs, q)
    eps_h = 1e-5
    G_fd = zeros(3, 3)
    for a in 1:3
        Lp = collect(box.L); Lp[a] += eps_h
        Lm = collect(box.L); Lm[a] -= eps_h
        bp = T.OrthorhombicBox((Lp[1], Lp[2], Lp[3]))
        bm = T.OrthorhombicBox((Lm[1], Lm[2], Lm[3]))
        rs_p = RS.restart_system(rs; box = bp)
        rs_m = RS.restart_system(rs; box = bm)
        qp = [s0[d, j] * Lp[d] for d in 1:3, j in 1:N]
        qm = [s0[d, j] * Lm[d] for d in 1:3, j in 1:N]
        G_fd[a, a] = (SC.energy(rs_p, qp) - SC.energy(rs_m, qm)) / (2 * eps_h)
    end
    g_err = maximum(abs(G[a, a] - G_fd[a, a]) for a in 1:3)
    @assert g_err < 1e-8 "composed StreamedLJ+bond box derivative FD error: $g_err"

    # Constraint virial + SystemContract pressure seam
    λ = CN.constraint_multipliers(q, masses, cs, F)
    W_c = CN.constraint_virial(q, masses, cs, λ)
    P_unconstrained = SC.pressure(rs, q, p)
    P_constrained = SC.pressure(rs, q, p; extra_virial = W_c)
    @assert isfinite(W_c) && isfinite(P_constrained)
    @assert abs(P_constrained - P_unconstrained - W_c / (3 * T.box_volume(box))) < 1e-14

    # Restart seam: box dilation breaks rigid bond length until re-projected
    new_box = T.OrthorhombicBox((10.5, 10.5, 10.5))
    rs2 = RS.restart_system(rs; box = new_box, provenance_notes = "volume move")
    @assert rs2.box == new_box
    @assert rs2.atoms.masses == rs.atoms.masses
    @assert occursin("volume move", rs2.provenance.notes)
    q_scaled = q .* 1.05
    threw_constraint = false
    try
        RS.validate_restart_state(rs2; q = q_scaled, p = p)
    catch e
        threw_constraint = e isa T.InvalidGeometry
    end
    @assert threw_constraint "validate_restart_state must refuse dilated constrained bond"

    q_reproj = copy(q_scaled)
    p_reproj = copy(p)
    CN.project!(q_reproj, p_reproj, rs2.atoms.masses, rs2.constraints)
    @assert RS.validate_restart_state(rs2; q = q_reproj, p = p_reproj)
    @printf("  resolved+streamed+constraints+restart: U=%.6f G_fd_err=%.2e W_c=%.4f\n",
            U, g_err, W_c)
    return true
end

# ---------------------------------------------------------------------------
# 8. PME + StreamedLJ + Dynamics + Stress seam
# ---------------------------------------------------------------------------
"""
Cross-phase seam: `StreamedLJFactor(:lfs)` + `PMEFactor` inside `ResolvedSystem`
and `Dynamics.HarmonicSystem`. Energy/forces match bit-for-bit, 10 Verlet steps
run cleanly, `:box_derivative` capability and `box_derivative`/`virial`/
`affine_pressure`/`pressure` work end-to-end, while `Dynamics.force_model_supported`
still refuses `:shift` and `:trunc`.
"""
function test_pme_streamed_lj_dynamics_and_stress_refusal_seam()
    box = T.OrthorhombicBox((12.0, 12.0, 12.0))
    N = 6
    rng = MersenneTwister(108)
    q = 2.0 .+ 8.0 .* rand(rng, 3, N)
    charges = [0.6, -0.6, 0.4, -0.4, 0.25, -0.25]
    masses = ones(N)
    lj = SL.StreamedLJFactor(q, fill(0.2, N), fill(0.9, N), box; rcut = 4.0, mode = :lfs)
    pme = PM.PMEFactor(q, charges, box; alpha = 0.35, rcut = 4.5, grid = (16, 16, 16), order = 4)

    rs = RS.ResolvedSystem(; masses = masses, charges = charges, box = box,
        factors = FA.AbstractFactor[lj, pme], component_names = [:lj, :pme],
        declared = [:energy, :cartesian_forces, :component_energies, :box_derivative],
        provenance = RS.Provenance(; source = "integration", force_field = "pme+lj"))

    U_sc, F_sc = SC.energy_forces(rs, q)
    ctx = SC.resolved_context(rs)
    hs = DY.HarmonicSystem(rs.factors, rs.atoms.masses, ctx)
    @assert DY.force_model_supported(hs)
    st = DY.State(copy(q), zeros(3, N))
    U_dy = DY.forces!(hs, st.q, st.F)
    @assert U_sc == U_dy && st.F == F_sc "SystemContract and Dynamics factor sweeps must agree bit-for-bit"

    recs, es = DY.run_trajectory(st, hs, 1e-3, 10; record_every = 1)
    @assert length(recs) == 10 && all(isfinite, es.compensated)

    # Full stress & component box derivatives match
    G_tot = SC.box_derivative(rs, q)
    G_lj = SC.box_derivative_component(rs, q, :lj)
    G_pme = SC.box_derivative_component(rs, q, :pme)
    G_lj_streamed = SL.streamed_box_derivative(lj, q, ctx)
    G_pme_direct = PM.pme_factor_box_derivative(pme, q, ctx)
    @assert isapprox(G_lj, G_lj_streamed; atol = 1e-12, rtol = 1e-12)
    @assert isapprox(G_pme, G_pme_direct; atol = 1e-12, rtol = 1e-12)
    @assert isapprox(G_tot, G_lj .+ G_pme; atol = 1e-12, rtol = 1e-12)
    @assert all(isfinite, SC.virial(rs, q))
    @assert isfinite(SC.affine_pressure(rs, q))
    @assert isfinite(SC.pressure(rs, q, zeros(3, N)))

    # Non-C^1 LJ modes must be refused by Dynamics integrator gate
    for bad_mode in (:shift, :trunc)
        lj_bad = SL.StreamedLJFactor(q, fill(0.2, N), fill(0.9, N), box; rcut = 4.0, mode = bad_mode)
        hs_bad = DY.HarmonicSystem(FA.AbstractFactor[lj_bad], masses, ctx)
        @assert !DY.force_model_supported(hs_bad) "Dynamics must refuse StreamedLJ mode $bad_mode"
    end
    @printf("  PME+StreamedLJ+Dynamics: U=%.6f 10 steps finite, total+component G verified\n", U_sc)
    return true
end

# ---------------------------------------------------------------------------
# 9. TriclinicBox supported vs refused cross-module matrix
# ---------------------------------------------------------------------------
"""
Verifies the live triclinic boundary across all modules:
* Supported: `Neighbors` triclinic MIC/cell list, local `Factors` + `StreamedLJFactor(:lfs)`
  + `PMEFactor` inside `ResolvedSystem` and `SystemContract` analytic `box_derivative` /
  `virial` / `affine_pressure` / `pressure` (9-entry FD verified).
* Refused: `EwaldFactor` on `TriclinicBox` (`UnsupportedGeometry`), `Polarizable` /
  `PolarizationOperator` on `TriclinicBox`, and `MonteCarloBarostat` on `TriclinicBox`.
"""
function test_triclinic_supported_vs_refused_cross_module_matrix()
    H = [10.0 1.5 0.6;
         0.0 10.0 1.0;
         0.0  0.0 10.0]
    tbox = T.TriclinicBox(H)
    N = 4
    S = [0.2 0.35 0.6 0.8;
         0.2 0.25 0.6 0.7;
         0.2 0.20 0.5 0.8]
    q = H * S
    charges = [0.5, -0.5, 0.3, -0.3]

    # Supported 1: Neighbors + local Factors + SystemContract box_derivative
    nl = NB.build_neighbors(q, tbox, 3.0, 0.3)
    @assert !NB.needs_rebuild!(nl, q, tbox)
    bond = FA.HarmonicBond(1, 2, 20.0, 1.2)
    ljp = FA.LennardJonesPair(3, 4, 0.3, 1.0; mode = :lfs, rc = 3.5)
    rs_local = RS.ResolvedSystem(;
        masses = ones(N), charges = charges, box = tbox,
        factors = FA.AbstractFactor[bond, ljp], component_names = [:bond, :ljp],
        declared = [:energy, :cartesian_forces, :component_energies, :box_derivative],
        provenance = RS.Provenance(; source = "integration", force_field = "triclinic local"),
    )
    G_local = SC.box_derivative(rs_local, q)
    @assert all(isfinite, G_local) && isfinite(SC.affine_pressure(rs_local, q))

    # Supported 2: StreamedLJFactor(:lfs) + PMEFactor inside ResolvedSystem on TriclinicBox
    slj_tri = SL.StreamedLJFactor(q, fill(0.3, N), fill(1.0, N), tbox; rcut = 3.5, mode = :lfs)
    G_slj = SL.streamed_box_derivative(slj_tri, q, FA.Context(tbox, T.REDUCED, N))
    @assert all(isfinite, G_slj)
    pme_tri = PM.PMEFactor(q, charges, tbox; alpha = 0.35, rcut = 3.5, grid = (24, 24, 24), order = 4)
    rs_tri = RS.ResolvedSystem(;
        masses = ones(N), charges = charges, box = tbox,
        factors = FA.AbstractFactor[bond, slj_tri, pme_tri],
        component_names = [:bond, :slj, :pme],
        declared = [:energy, :cartesian_forces, :component_energies, :box_derivative],
        provenance = RS.Provenance(; source = "integration", force_field = "tri bond+slj+pme"),
    )
    U_tri, F_tri = SC.energy_forces(rs_tri, q)
    G_tri = SC.box_derivative(rs_tri, q)
    @assert isfinite(U_tri) && all(isfinite, F_tri) && all(isfinite, G_tri)
    eps_fd = 1e-5
    max_fd_err = 0.0
    for a in 1:3, b in 1:3
        Hp = copy(H); Hp[a, b] += eps_fd
        Hm = copy(H); Hm[a, b] -= eps_fd
        rsp = RS.restart_system(rs_tri; box = T.TriclinicBox(Hp))
        rsm = RS.restart_system(rs_tri; box = T.TriclinicBox(Hm))
        fd = (SC.energy(rsp, Hp * S) - SC.energy(rsm, Hm * S)) / (2.0 * eps_fd)
        max_fd_err = max(max_fd_err, abs(G_tri[a, b] - fd))
    end
    @assert max_fd_err < 1e-7 "triclinic bond+StreamedLJ+PME 9-entry FD err=$(max_fd_err)"

    # Refused: EwaldFactor, Polarizable, PolarizationOperator, MonteCarloBarostat
    ew_refused = false
    try
        EW.EwaldFactor(q, charges, tbox; alpha = 0.35, rcut = 3.5, kmax = 4)
    catch e
        ew_refused = e isa T.UnsupportedGeometry
    end
    @assert ew_refused "EwaldFactor must refuse TriclinicBox with UnsupportedGeometry"

    pol_refused = false
    try
        PO.InducedDipoleSystem(q, charges, fill(0.1, N), tbox, T.REDUCED)
    catch e
        pol_refused = e isa T.UnsupportedGeometry
    end
    @assert pol_refused "Polarizable must refuse TriclinicBox with UnsupportedGeometry"

    pop_refused = false
    try
        POP.build_operator(q, charges, fill(0.1, N), tbox)
    catch e
        pop_refused = e isa ArgumentError
    end
    @assert pop_refused "PolarizationOperator must refuse TriclinicBox with ArgumentError"

    baro = MB.VolumeMoveBarostat(1.0, 0.5, 0.1)
    baro_refused = false
    try
        MB.step!(baro, q, tbox, (x, b) -> 0.0, MersenneTwister(1))
    catch e
        baro_refused = e isa T.UnsupportedGeometry
    end
    @assert baro_refused "MonteCarloBarostat must refuse TriclinicBox with UnsupportedGeometry"

    println("  triclinic matrix: local+StreamedLJ+PMEFactor pass (FD err=$(max_fd_err)); EwaldFactor/Polarizable/Barostat refuse")
    return true
end

# ---------------------------------------------------------------------------
# 10. Barostat & Virtual-Site Refusal Seams
# ---------------------------------------------------------------------------
"""
Verifies that typed refusal gates hold across module boundaries:
* `MonteCarloBarostat` refuses constraints (`ConstraintIncompatibilityError`),
  missing virials (`MissingVirialError`), and `NoBox` (`UnsupportedGeometry`),
  while succeeding on an ideal-gas orthorhombic system.
* `ResolvedSystem` accepts `VirtualSite` representation (mass 0) while
  `SystemContract` refuses execution (`UnsupportedCapability(:virtual_sites)`).
"""
function test_barostat_and_virtual_site_refusal_seams()
    # Barostat refusals
    baro_c = false
    try
        MB.VolumeMoveBarostat(1.0, 0.5, 0.1; constraints = [(1, 2, 1.0)])
    catch e
        baro_c = e isa MB.ConstraintIncompatibilityError
    end
    @assert baro_c "VolumeMoveBarostat must refuse non-empty constraints"

    baro_v = false
    try
        MB.NPTBarostat(1.0, 0.5, 0.1, MB.NPTModel((x, b) -> 0.0; masses = 1.0))
    catch e
        baro_v = e isa MB.MissingVirialError
    end
    @assert baro_v "NPTBarostat must refuse model without virial provider"

    baro_nb = false
    try
        MB.step!(MB.VolumeMoveBarostat(1.0, 0.5, 0.1), zeros(3, 2), T.NoBox(), (x, b) -> 0.0)
    catch e
        baro_nb = e isa T.UnsupportedGeometry
    end
    @assert baro_nb "MonteCarloBarostat must refuse NoBox"

    # Ideal-gas VolumeMoveBarostat succeeds
    q_ig = [1.0 3.0; 1.0 2.0; 1.0 2.5]
    res_ig = MB.step!(MB.VolumeMoveBarostat(1.0, 0.5, 0.05), q_ig,
                      T.OrthorhombicBox((5.0, 5.0, 5.0)), (x, b) -> 0.0, MersenneTwister(9))
    @assert isfinite(res_ig.new_volume) && res_ig.new_volume > 0

    # VirtualSite representation accepted, SystemContract execution refused
    vs = RS.VirtualSite(:linear, 3, [1, 2], [0.5, 0.5])
    rs_vs = RS.ResolvedSystem(;
        masses = [12.0, 12.0, 0.0], charges = [0.1, 0.1, -0.2],
        box = T.OrthorhombicBox((10.0, 10.0, 10.0)),
        factors = FA.AbstractFactor[FA.HarmonicBond(1, 2, 100.0, 1.0)],
        virtual_sites = [vs],
        provenance = RS.Provenance(; source = "integration", force_field = "vs toy"),
    )
    @assert RS.virtual_site_indices(rs_vs) == Set([3])
    q_vs = [0.0 1.0 0.5; 0.0 0.0 0.0; 0.0 0.0 0.0]
    p_vs = zeros(3, 3)
    for fn in (
        () -> SC.energy(rs_vs, q_vs),
        () -> SC.forces(rs_vs, q_vs),
        () -> SC.energy_forces(rs_vs, q_vs),
        () -> SC.box_derivative(rs_vs, q_vs),
        () -> SC.kinetic_energy(rs_vs, p_vs),
        () -> SC.to_atomset(rs_vs, q_vs, p_vs),
    )
        vs_refused = false
        try
            fn()
        catch e
            vs_refused = e isa RS.UnsupportedCapability && e.capability === :virtual_sites
        end
        @assert vs_refused "SystemContract must refuse virtual-site execution"
    end
    println("  barostat and virtual-site refusal gates verified")
    return true
end

# ---------------------------------------------------------------------------
# 11. PolarizationOperator <-> Polarizable cross-check & periodic refusal
# ---------------------------------------------------------------------------
"""
Cross-checks the matrix-free `PolarizationOperator` against the dense
`Polarizable` solver on a Thole-damped 3-site `NoBox` configuration, and
verifies periodic boxes are refused by both.
"""
function test_polarization_modules_cross_check_and_refusal()
    q = [0.0 2.9 0.5;
         0.0 0.0 2.3;
         0.0 0.0 0.0]
    z = [1.0, -0.5, -0.5]
    α = [0.5, 0.3, 0.3]

    sys = PO.InducedDipoleSystem(q, z, α, T.NoBox(), T.REDUCED; thole = true, a_thole = 1.2)
    r_dense = PO.solve_dipoles!(sys; tol = 1e-13, method = :cg)
    @assert r_dense.converged && r_dense.spd

    op = POP.build_operator(q, z, α; thole = true, a_thole = 1.2)
    r_mf = POP.solve_dipoles!(op; tol = 1e-13)
    @assert r_mf.status === :converged && r_mf.spd === :certified

    mu_diff = maximum(abs, r_dense.μ .- POP.require_converged(r_mf))
    @assert mu_diff < 1e-10 "Polarizable and PolarizationOperator dipoles disagree: $mu_diff"

    U_perm, F_perm = PO.permanent_energy_forces(sys)
    U_tot_dense = PO.induced_energy(sys, r_dense)
    U_tot_mf = U_perm + POP.induced_energy(op, r_mf)
    @assert abs(U_tot_dense - U_tot_mf) < 1e-12 "polarization energy mismatch"

    F_tot_dense = PO.induced_forces(sys, r_dense; h = 1e-6)
    F_tot_mf = F_perm .+ POP.induced_forces(op, r_mf; h = 1e-6)
    f_diff = maximum(abs, F_tot_dense .- F_tot_mf)
    @assert f_diff < 1e-8 "polarization nuclear forces mismatch: $f_diff"

    ortho = T.OrthorhombicBox((10.0, 10.0, 10.0))
    p_ref = false
    try
        PO.InducedDipoleSystem(q, z, α, ortho, T.REDUCED)
    catch e
        p_ref = e isa T.UnsupportedGeometry
    end
    @assert p_ref
    @printf("  Polarizable <-> PolarizationOperator: max|dmu|=%.2e max|dF|=%.2e, periodic refused\n",
            mu_diff, f_diff)
    return true
end

function _run_all()
    tests = [
        ("structure_to_dynamics_pipeline", test_structure_to_dynamics_pipeline),
        ("stale_neighbor_list_energy_invariance", test_stale_neighbor_list_energy_invariance),
        ("bipartite_summary_vs_explicit_pairs", test_bipartite_summary_vs_explicit_pairs),
        ("mixed_set_force_is_gradient", test_mixed_set_force_is_gradient),
        ("mixed_system_energy_conservation", test_mixed_system_energy_conservation),
        ("hmc_on_factor_potential", test_hmc_on_factor_potential),
        ("resolved_streamed_constraints_pressure_and_restart_seam", test_resolved_streamed_constraints_pressure_and_restart_seam),
        ("pme_streamed_lj_dynamics_and_stress_refusal_seam", test_pme_streamed_lj_dynamics_and_stress_refusal_seam),
        ("triclinic_supported_vs_refused_cross_module_matrix", test_triclinic_supported_vs_refused_cross_module_matrix),
        ("barostat_and_virtual_site_refusal_seams", test_barostat_and_virtual_site_refusal_seams),
        ("polarization_modules_cross_check_and_refusal", test_polarization_modules_cross_check_and_refusal),
    ]
    for (name, fn) in tests
        println("TEST ", name)
        fn()
    end
    println("")
    println("ALL TESTS PASSED")
end

_run_all()
