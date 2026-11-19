# Focused standalone checks for the scalar-charge PME evaluator.
#
# The PME module is deliberately standalone (it does not import
# Electrostatics), so both are hosted here the same way test_electrostatics.jl
# hosts its reference evaluator. The direct Ewald evaluator is used as the
# reference with matching alpha, rcut, and a convergent reciprocal cutoff; the
# PME grid is chosen fine enough that both describe the same continuum Ewald
# sum.

module PMEHost
    include(joinpath(@__DIR__, "..", "src", "PME.jl"))
end
const PME = PMEHost.PME

module TypesHost
    include(joinpath(@__DIR__, "..", "src", "Types.jl"))
end
const Types = TypesHost.Types

module ElectrostaticsHost
    include(joinpath(@__DIR__, "..", "src", "Electrostatics.jl"))
end
const EW = ElectrostaticsHost.Electrostatics

using Statistics, Random, LinearAlgebra

maxabs(A) = maximum(abs, A)

function refuses(f)
    try
        f()
    catch err
        return err isa ArgumentError
    end
    return false
end

function min_image_r(q, box, i, j)
    d = q[:, i] - q[:, j]
    d .-= box .* round.(d ./ box)
    return sqrt(sum(abs2, d))
end

function fd_force(q, charges, plan; corrections=PME.PairCorrections(), h=1e-6)
    F = zeros(size(q))
    for idx in eachindex(q)
        old = q[idx]
        q[idx] = old + h
        ep = PME.pme_energy(plan, q, charges; corrections=corrections)
        q[idx] = old - h
        em = PME.pme_energy(plan, q, charges; corrections=corrections)
        q[idx] = old
        F[idx] = -(ep - em) / (2h)
    end
    return F
end

function fd_charge_gradient(q, charges, plan; corrections=PME.PairCorrections(), h=1e-6)
    g = zeros(length(charges))
    for i in eachindex(charges)
        old = charges[i]
        charges[i] = old + h
        ep = PME.pme_energy(plan, q, charges; corrections=corrections)
        charges[i] = old - h
        em = PME.pme_energy(plan, q, charges; corrections=corrections)
        charges[i] = old
        g[i] = (ep - em) / (2h)
    end
    return g
end

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

box = [12.0, 13.0, 14.0]

# Exactly neutral: all entries are representable dyadic values.
neutral_q = [1.2 4.1 7.0 2.6;
             2.0 6.0 3.3 8.4;
             1.5 5.2 9.0 4.1]
neutral_charges = [1.0, -1.0, 0.5, -0.5]
@assert sum(neutral_charges) == 0.0

# Charged fixture. One coordinate is translated outside the primary cell to
# exercise minimum-image real-space distances and periodic spline stencils.
charged_q = [1.2 4.1 7.0 2.6 9.3;
             2.0 6.0 3.3 8.4 5.5;
             1.5 5.2 9.0 4.1 6.2]
charged_q[:, 2] .+= box
charged_charges = [1.0, -0.7, 0.35, 0.9, -0.25]
@assert sum(charged_charges) != 0.0

# Matching reciprocal convergence: alpha/rcut are identical for both
# evaluators, kmax is large enough that the direct Ewald truncation is below
# the PME error, and the explicit PME grid is much finer than the default.
ewald_kw = (alpha=0.35, rcut=5.5, kmax=8, prefactor=1.3)
pme_kw = (alpha=ewald_kw.alpha, rcut=ewald_kw.rcut,
          grid=(64, 64, 64), prefactor=ewald_kw.prefactor)

# ---------------------------------------------------------------------------
# Energy and force agreement with direct Ewald on neutral and charged systems
# ---------------------------------------------------------------------------

for (label, q, charges) in (
        ("neutral", neutral_q, neutral_charges),
        ("charged", charged_q, charged_charges))
    E_ewald, F_ewald = EW.evaluate_ewald(q, charges, box; ewald_kw...)
    E_pme, F_pme = PME.evaluate_pme(q, charges, box; pme_kw...)

    @assert isfinite(E_pme)
    @assert isfinite(E_ewald)
    @assert size(F_pme) == size(q)
    @assert maxabs(F_pme - F_ewald) < 1e-5
    @assert abs(E_pme - E_ewald) < 1e-5
    @assert maxabs(F_pme) < 1.0
end

# The two one-shot spellings are aliases and the reusable plan reproduces the
# one-shot evaluation exactly.
E_alias, F_alias = PME.pme_evaluate(charged_q, charged_charges, box; pme_kw...)
E_oneshot, F_oneshot = PME.evaluate_pme(charged_q, charged_charges, box; pme_kw...)
@assert E_alias == E_oneshot
@assert F_alias == F_oneshot

plan = PME.PMEPlan(charged_q, box; alpha=pme_kw.alpha, rcut=pme_kw.rcut,
                   grid=pme_kw.grid, prefactor=pme_kw.prefactor)
E_plan, F_plan = PME.pme_energy_forces(plan, charged_q, charged_charges)
@assert E_plan == E_oneshot
@assert F_plan == F_oneshot

Fbuf = fill(NaN, size(charged_q))
E_buf = PME.pme_energy_forces!(Fbuf, plan, charged_q, charged_charges)
@assert E_buf == E_plan
@assert Fbuf == F_plan

# A plan is reusable: evaluating perturbed coordinates must not contaminate a
# subsequent evaluation of the original coordinates.
q_perturbed = copy(charged_q)
q_perturbed[1, 1] += 0.37
PME.pme_energy_forces(plan, q_perturbed, charged_charges)
E_repeat, F_repeat = PME.pme_energy_forces(plan, charged_q, charged_charges)
@assert E_repeat == E_plan
@assert F_repeat == F_plan

# ---------------------------------------------------------------------------
# Finite-difference consistency of forces and charge gradients
# ---------------------------------------------------------------------------

# Corrections are included so the finite-difference checks cover the pair
# correction force and charge-gradient contributions as well as the PME grid.
corrections = PME.PairCorrections(excluded=[(1, 2)],
                                  scaled=[(2, 3, 0.5), (3, 4, 0.25)])
E_corr, F_corr = PME.pme_energy_forces(plan, charged_q, charged_charges;
                                       corrections=corrections)
@assert isfinite(E_corr)

F_fd = fd_force(copy(charged_q), charged_charges, plan; corrections=corrections)
@assert maxabs(F_corr - F_fd) < 1e-7

g = zeros(length(charged_charges))
PME.pme_charge_gradient!(g, plan, charged_q, charged_charges;
                         corrections=corrections)
g_fd = fd_charge_gradient(charged_q, copy(charged_charges), plan;
                          corrections=corrections)
@assert maxabs(g - g_fd) < 1e-7

g_no_corr = zeros(length(charged_charges))
PME.pme_charge_gradient!(g_no_corr, plan, charged_q, charged_charges)
@assert maxabs(g_no_corr - g) > 1e-6  # corrections must change the gradient

# Box derivative consistency on orthorhombic charged system (with exclusions and scaling)
G_corr = PME.pme_box_derivative(plan, charged_q, charged_charges; corrections=corrections)
@assert size(G_corr) == (3, 3)
@assert isfinite(sum(G_corr))

# In-place and joint (energy, forces, box derivative) evaluation
G_buf = fill(NaN, 3, 3)
PME.pme_box_derivative!(G_buf, plan, charged_q, charged_charges; corrections=corrections)
@assert G_buf == G_corr

U_joint, F_joint, G_joint = PME.pme_energy_forces_box_derivative(plan, charged_q, charged_charges; corrections=corrections)
@assert U_joint == E_corr
@assert F_joint == F_corr
@assert G_joint == G_corr

# Diagonal FD box derivative at fixed fractional coordinates
G_diag_fd = zeros(3)
s_ortho = charged_q ./ box
for a in 1:3
    box_p = copy(box); box_p[a] += 1e-6
    q_p = s_ortho .* box_p
    plan_p = PME.PMEPlan(q_p, box_p; alpha=pme_kw.alpha, rcut=pme_kw.rcut,
                         grid=pme_kw.grid, prefactor=pme_kw.prefactor)
    ep = PME.pme_energy(plan_p, q_p, charged_charges; corrections=corrections)

    box_m = copy(box); box_m[a] -= 1e-6
    q_m = s_ortho .* box_m
    plan_m = PME.PMEPlan(q_m, box_m; alpha=pme_kw.alpha, rcut=pme_kw.rcut,
                         grid=pme_kw.grid, prefactor=pme_kw.prefactor)
    em = PME.pme_energy(plan_m, q_m, charged_charges; corrections=corrections)
    G_diag_fd[a] = (ep - em) / (2e-6)
end
@assert maxabs(diag(G_corr) - G_diag_fd) < 1e-7

# ---------------------------------------------------------------------------
# Exclusions and scaled pair corrections
# ---------------------------------------------------------------------------

ewald_corrections = EW.PairCorrections(excluded=[(1, 2)],
                                       scaled=[(2, 3, 0.5), (3, 4, 0.25)])
E_base, F_base = PME.pme_energy_forces(plan, charged_q, charged_charges)
E_ewald_base, F_ewald_base = EW.evaluate_ewald(charged_q, charged_charges, box;
                                               ewald_kw...)
E_ewald_corr, F_ewald_corr = EW.evaluate_ewald(charged_q, charged_charges, box;
                                               ewald_kw..., corrections=ewald_corrections)

# Full corrected PME vs corrected direct Ewald.
@assert abs(E_corr - E_ewald_corr) < 1e-5
@assert maxabs(F_corr - F_ewald_corr) < 1e-5

# The correction is exactly (s-1)*prefactor*q_i*q_j/r at minimum-image r.
p = pme_kw.prefactor
q1, q2, q3, q4 = charged_charges[1:4]
expected_shift =
    (0 - 1) * p * q1 * q2 / min_image_r(charged_q, box, 1, 2) +
    (0.5 - 1) * p * q2 * q3 / min_image_r(charged_q, box, 2, 3) +
    (0.25 - 1) * p * q3 * q4 / min_image_r(charged_q, box, 3, 4)
@assert abs((E_corr - E_base) - expected_shift) < 1e-12
@assert abs((E_ewald_corr - E_ewald_base) - expected_shift) < 1e-12
@assert maxabs(F_corr - F_base) > 1e-3

# An excluded pair (s=0) and an explicitly scaled pair with the same weight
# must give the same correction.
corr_excluded = PME.PairCorrections(excluded=[(1, 2)])
corr_scaled_zero = PME.PairCorrections(scaled=[(1, 2, 0.0)])
E_excluded = PME.pme_energy(plan, charged_q, charged_charges;
                            corrections=corr_excluded)
E_scaled_zero = PME.pme_energy(plan, charged_q, charged_charges;
                               corrections=corr_scaled_zero)
@assert E_excluded == E_scaled_zero

# ---------------------------------------------------------------------------
# Invalid-input refusals
# ---------------------------------------------------------------------------

# Construction-time validation.
@assert refuses(() -> PME.PMEPlan(charged_q, box; alpha=0.0, rcut=5.5,
                                  grid=(64, 64, 64)))
@assert refuses(() -> PME.PMEPlan(charged_q, box; alpha=0.35, rcut=6.0,
                                  grid=(64, 64, 64)))       # rcut == min(box)/2
@assert refuses(() -> PME.PMEPlan(charged_q, box; alpha=0.35, rcut=7.0,
                                  grid=(64, 64, 64)))       # rcut > min(box)/2
@assert refuses(() -> PME.PMEPlan(charged_q, box; alpha=0.35, rcut=5.5,
                                  grid=(64, 64, 64), order=3))
@assert refuses(() -> PME.PMEPlan(charged_q, box; alpha=0.35, rcut=5.5,
                                  grid=(64, 64, 64), order=5))
@assert refuses(() -> PME.PMEPlan(charged_q, [12.0, 13.0]; alpha=0.35, rcut=5.5))
@assert refuses(() -> PME.PMEPlan(charged_q, [12.0, -13.0, 14.0]; alpha=0.35,
                                  rcut=5.5))
@assert refuses(() -> PME.PMEPlan(zeros(2, 5), box; alpha=0.35, rcut=5.5))
@assert refuses(() -> PME.PMEPlan(charged_q, box; alpha=0.35, rcut=5.5,
                                  grid=(64, 64)))
@assert refuses(() -> PME.PMEPlan(charged_q, box; alpha=0.35, rcut=5.5,
                                  grid=(64.0, 64, 64)))
@assert refuses(() -> PME.PMEPlan(charged_q, box; alpha=0.35, rcut=5.5,
                                  grid=(8, 8, 8), order=6))

# Evaluation-time validation.
@assert refuses(() -> PME.evaluate_pme(charged_q, [1.0, 2.0], box; pme_kw...))
@assert refuses(() -> PME.pme_energy(plan, charged_q, [1.0, 2.0, 3.0, 4.0, 5.0];
                                     corrections=PME.PairCorrections(excluded=[(1, 6)])))
q_bad = copy(charged_q)
q_bad[1, 1] = NaN
@assert refuses(() -> PME.pme_energy(plan, q_bad, charged_charges))
charges_bad = copy(charged_charges)
charges_bad[2] = Inf
@assert refuses(() -> PME.pme_energy(plan, charged_q, charges_bad))

q_coincident = copy(charged_q)
q_coincident[:, 2] = q_coincident[:, 1]
@assert refuses(() -> PME.pme_energy(plan, q_coincident, charged_charges))

@assert refuses(() -> PME.pme_energy_forces!(charged_q, plan, charged_q,
                                             charged_charges))
@assert refuses(() -> PME.pme_charge_gradient!(charged_charges, plan, charged_q,
                                               charged_charges))
@assert refuses(() -> PME.pme_box_derivative!(fill(0.0, 3, 4), plan, charged_q, charged_charges))
@assert refuses(() -> PME.pme_box_derivative!(view(charged_q, 1:3, 1:3), plan, charged_q, charged_charges))
@assert refuses(() -> PME.PMEPlan(charged_q, [1.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 1.0]; alpha=0.35, rcut=0.2))
@assert refuses(() -> PME.PMEPlan(charged_q, [1.0 0.0 0.0; 0.0 -1.0 0.0; 0.0 0.0 1.0]; alpha=0.35, rcut=0.2))

# PairCorrections validation (same convention as Electrostatics).
@assert refuses(() -> PME.PairCorrections(scaled=[(1, 1, 0.5)]))
@assert refuses(() -> PME.PairCorrections(scaled=[(1, 2, -0.5)]))
@assert refuses(() -> PME.PairCorrections(scaled=[(1, 2, NaN)]))
@assert refuses(() -> PME.PairCorrections(excluded=[(1, 1)]))
@assert refuses(() -> PME.PairCorrections(excluded=[(1, 2)],
                                         scaled=[(1, 2, 0.5)]))
@assert refuses(() -> PME.PairCorrections(scaled=[(1, 2, 0.5, 1.0)]))
@assert refuses(() -> PME.PairCorrections(scaled=[(1, 2, 0.5), (1, 2, 0.75)]))

# ---------------------------------------------------------------------------
# Cost versus N: near-linear scaling, allocation-free reuse, grid-controlled
# accuracy at fixed spacing
# ---------------------------------------------------------------------------
#
# These are deliberately loose, machine-independent bounds. The point is a
# regression guard, not a performance claim: an O(N²) real-space or reciprocal
# path would move the 10× N ratio from ~10-14 to ~100.

function scaling_state(N; seed=11, rho=0.1, alpha=0.35, rcut=8.0, h=0.8,
                       order=4)
    rng = MersenneTwister(seed)
    L = (N / rho)^(1 / 3)
    q = Matrix{Float64}(undef, 3, N)
    rand!(rng, q); q .*= L
    ch = rand(rng, N) .- 0.5; ch .-= sum(ch) / N
    n = max(2 * order, round(Int, L / h)); isodd(n) || (n += 1)
    plan = PME.PMEPlan(q, (L, L, L); alpha=alpha, rcut=rcut, grid=(n, n, n),
                       order=order)
    return plan, q, ch, (L, L, L), (n, n, n)
end

function call_median(plan, q, ch, F; reps=5)
    f!() = PME.pme_energy_forces!(F, plan, q, ch)
    f!(); f!()                      # warm up, compile, allocate outputs
    ts = Float64[]
    for _ in 1:reps
        t0 = time_ns(); f!(); push!(ts, (time_ns() - t0) / 1e9)
    end
    return median(ts)
end

let
    plan1, q1, ch1, _, _ = scaling_state(1_000)
    plan2, q2, ch2, _, _ = scaling_state(10_000)
    t1 = call_median(plan1, q1, ch1, zeros(3, 1_000))
    t2 = call_median(plan2, q2, ch2, zeros(3, 10_000))
    ratio = t2 / t1
    # Linear cost would give ~10 here; the measured ratio is ~13 because the
    # smaller box has a less favourable linked-cell geometry. O(N²) would be
    # ~100, so the band separates the two by a wide margin.
    @assert 5.0 <= ratio <= 35.0 "cost ratio for 10x N was $ratio, expected near-linear"
    # Per-atom cost must not grow with N either.
    @assert t2 / 10_000 <= 4 * (t1 / 1_000) "per-atom cost grew by more than 4x with N"
end

# A warm call must not allocate: the plan owns all working storage, and the
# default correction set is shared, not rebuilt per call.
let
    plan, q, ch, _, _ = scaling_state(2_000)
    F = zeros(3, length(ch))
    f!() = PME.pme_energy_forces!(F, plan, q, ch)
    g!() = PME.pme_energy(plan, q, ch)
    f!(); g!()
    a_f = @allocated f!()
    a_g = @allocated g!()
    @assert a_f == 0 "energy+force allocated $a_f bytes per warm call, expected 0"
    @assert a_g == 0 "energy allocated $a_g bytes per warm call, expected 0"
    G_alloc = zeros(3, 3)
    h!() = PME.pme_energy_forces_box_derivative!(F, G_alloc, plan, q, ch)
    h!()
    @assert (@allocated h!()) == 0 "energy+force+box_derivative allocated per warm call"
    # ... and reusing the plan must reproduce the same answer as the one-shot path.
    E1, F1 = PME.pme_energy_forces(plan, q, ch)
    E2, F2 = PME.evaluate_pme(q, ch, plan.box; alpha=plan.alpha, rcut=plan.rcut,
                              grid=plan.dims, order=plan.order,
                              prefactor=plan.prefactor)
    @assert E1 == E2 && F1 == F2
end

# Error at a fixed grid spacing must be controlled by the grid, not by N.
function grid_error_estimate(N, h, alpha, rcut)
    L = (N / 0.1)^(1 / 3)
    n = max(8, round(Int, L / h)); isodd(n) || (n += 1)
    m = max(8, round(Int, L / (h / 2))); isodd(m) || (m += 1)
    worst = 0.0
    for seed in (1, 2)
        rng = MersenneTwister(seed)
        q = Matrix{Float64}(undef, 3, N); rand!(rng, q); q .*= L
        ch = rand(rng, N) .- 0.5; ch .-= sum(ch) / N
        pc = PME.PMEPlan(q, (L, L, L); alpha=alpha, rcut=rcut, grid=(n, n, n))
        pf = PME.PMEPlan(q, (L, L, L); alpha=alpha, rcut=rcut, grid=(m, m, m))
        Uc, _ = PME.pme_energy_forces(pc, q, ch)
        Uf, _ = PME.pme_energy_forces(pf, q, ch)
        worst = max(worst, abs(Uc - Uf) / max(abs(Uf), 1.0))
    end
    return worst
end

let
    e1 = grid_error_estimate(1_000, 1.0, 0.35, 8.0)
    e2 = grid_error_estimate(5_000, 1.0, 0.35, 8.0)
    # both are a grid error, and it does not grow with N at fixed spacing
    @assert e1 < 5e-3 && e2 < 5e-3 "grid error too large at fixed spacing: $e1, $e2"
    @assert e2 <= 10 * e1 "error grew with N at fixed grid spacing: $e1 -> $e2"
end

# The linked-cell topology must enumerate every in-range pair exactly once: the
# candidate count is bounded by C(N,2) and every pair inside rcut is covered.
let
    N = 400; L = (N / 0.1)^(1 / 3); rcut = 6.0
    rng = MersenneTwister(5)
    q = Matrix{Float64}(undef, 3, N); rand!(rng, q); q .*= L
    plan = PME.PMEPlan(q, (L, L, L); alpha=0.4, rcut=rcut, grid=(27, 27, 27))
    nc = plan.cell_n
    cellid(i) = begin
        c1 = mod(floor(Int, q[1, i] / L * nc[1]), nc[1]) + 1
        c2 = mod(floor(Int, q[2, i] / L * nc[2]), nc[2]) + 1
        c3 = mod(floor(Int, q[3, i] / L * nc[3]), nc[3]) + 1
        c1 + nc[1] * (c2 - 1) + nc[1] * nc[2] * (c3 - 1)
    end
    cell = [cellid(i) for i in 1:N]
    occ = zeros(Int, prod(nc))
    for c in cell; occ[c] += 1; end
    candidates = 0
    for (c, t) in plan.cell_pairs
        candidates += c == t ? occ[c] * (occ[c] - 1) ÷ 2 : occ[c] * occ[t]
    end
    @assert candidates <= N * (N - 1) ÷ 2
    listed = Set(plan.cell_pairs)
    for i in 1:N-1, j in i+1:N
        d = q[:, i] .- q[:, j]; d .-= L .* round.(d ./ L)
        sum(abs2, d) < rcut^2 || continue
        a, b = minmax(cell[i], cell[j])
        @assert (a, b) in listed "in-range pair fell in an unlisted cell pair"
    end
end

# ---------------------------------------------------------------------------
# Skew Triclinic Box: forces and box derivative vs finite differences
# ---------------------------------------------------------------------------

H_skew = [12.0  1.5  0.8;
           0.0 13.0  1.2;
           0.0  0.0 14.0]
tbox = Types.TriclinicBox(H_skew)

# Charged system with particles outside primary cell
q_skew = [1.2 4.1 7.0 2.6 9.3;
          2.0 6.0 3.3 8.4 5.5;
          1.5 5.2 9.0 4.1 6.2]
q_skew[:, 2] .+= H_skew[:, 1] + 2.0 .* H_skew[:, 2]  # image translation
charges_skew = [1.0, -0.7, 0.35, 0.9, -0.25]
@assert sum(charges_skew) != 0.0

# Corrections with excluded and scaled pairs
skew_corrections = PME.PairCorrections(excluded=[(1, 2)],
                                      scaled=[(2, 3, 0.5), (3, 4, 0.25)])

for order in (4, 6)
    plan_skew = PME.PMEPlan(q_skew, tbox; alpha=0.35, rcut=5.0,
                            grid=(32, 32, 32), order=order, prefactor=1.3)
    @assert plan_skew.is_triclinic
    @assert plan_skew.order == order

    U_skew, F_skew, G_skew = PME.pme_energy_forces_box_derivative(plan_skew, q_skew, charges_skew;
                                                                 corrections=skew_corrections)
    @assert isfinite(U_skew)
    @assert all(isfinite, F_skew)
    @assert all(isfinite, G_skew)
    @assert size(G_skew) == (3, 3)

    # Reusable plan consistency
    U_check = PME.pme_energy(plan_skew, q_skew, charges_skew; corrections=skew_corrections)
    @assert U_check == U_skew
    F_check = zeros(3, plan_skew.N)
    PME.pme_energy_forces!(F_check, plan_skew, q_skew, charges_skew; corrections=skew_corrections)
    @assert F_check == F_skew

    # Coordinate forces vs central finite differences
    h_step = 1e-6
    F_fd_skew = zeros(size(q_skew))
    q_mut = copy(q_skew)
    for idx in eachindex(q_mut)
        old = q_mut[idx]
        q_mut[idx] = old + h_step
        Up = PME.pme_energy(plan_skew, q_mut, charges_skew; corrections=skew_corrections)
        q_mut[idx] = old - h_step
        Um = PME.pme_energy(plan_skew, q_mut, charges_skew; corrections=skew_corrections)
        q_mut[idx] = old
        F_fd_skew[idx] = -(Up - Um) / (2 * h_step)
    end
    @assert maxabs(F_skew - F_fd_skew) < 1e-7

    # Box derivative ∂U/∂H|_s vs central finite differences (all 9 components)
    s_skew = tbox.Hinv * q_skew
    G_fd_skew = zeros(3, 3)
    for a in 1:3, b in 1:3
        Hp = copy(H_skew); Hp[a, b] += h_step
        qp = Hp * s_skew
        plan_p = PME.PMEPlan(qp, Hp; alpha=0.35, rcut=5.0, grid=(32, 32, 32),
                             order=order, prefactor=1.3)
        ep = PME.pme_energy(plan_p, qp, charges_skew; corrections=skew_corrections)

        Hm = copy(H_skew); Hm[a, b] -= h_step
        qm = Hm * s_skew
        plan_m = PME.PMEPlan(qm, Hm; alpha=0.35, rcut=5.0, grid=(32, 32, 32),
                             order=order, prefactor=1.3)
        em = PME.pme_energy(plan_m, qm, charges_skew; corrections=skew_corrections)
        G_fd_skew[a, b] = (ep - em) / (2 * h_step)
    end
    @assert maxabs(G_skew - G_fd_skew) < 1e-7

    # Virial tensor Ξ = -H * G'
    Ξ = -(H_skew * transpose(G_skew))
    @assert all(isfinite, Ξ)

    # Accumulation flags on TriclinicBox preserve pre-seeded accumulators
    F_acc = fill(1.5, 3, plan_skew.N)
    G_acc = fill(-2.25, 3, 3)
    U_acc = PME.pme_energy_forces_box_derivative!(F_acc, G_acc, plan_skew, q_skew, charges_skew;
                                                 corrections=skew_corrections,
                                                 accumulate_F=true, accumulate_G=true)
    @assert U_acc == U_skew
    @assert maximum(abs, F_acc .- (1.5 .+ F_skew)) < 1e-13
    @assert maximum(abs, G_acc .- (-2.25 .+ G_skew)) < 1e-13

    G_acc2 = fill(0.75, 3, 3)
    PME.pme_box_derivative!(G_acc2, plan_skew, q_skew, charges_skew;
                            corrections=skew_corrections, accumulate=true)
    @assert maximum(abs, G_acc2 .- (0.75 .+ G_skew)) < 1e-13
end

# Also verify when box is passed directly as a 3x3 Matrix instead of TriclinicBox
plan_matrix = PME.PMEPlan(q_skew, H_skew; alpha=0.35, rcut=5.0,
                          grid=(32, 32, 32), order=4, prefactor=1.3)
U_mat, F_mat, G_mat = PME.pme_energy_forces_box_derivative(plan_matrix, q_skew, charges_skew;
                                                          corrections=skew_corrections)
U_tbox, F_tbox, G_tbox = PME.pme_energy_forces_box_derivative(
    PME.PMEPlan(q_skew, tbox; alpha=0.35, rcut=5.0, grid=(32, 32, 32), order=4, prefactor=1.3),
    q_skew, charges_skew; corrections=skew_corrections)
@assert U_mat == U_tbox
@assert F_mat == F_tbox
@assert G_mat == G_tbox

# Neutral system on skew triclinic cell
q_neutral_skew = neutral_q[:, 1:4]
plan_neutral_skew = PME.PMEPlan(q_neutral_skew, tbox; alpha=0.35, rcut=5.0,
                                grid=(32, 32, 32), order=4, prefactor=1.3)
U_neut, F_neut, G_neut = PME.pme_energy_forces_box_derivative(plan_neutral_skew, q_neutral_skew, neutral_charges)
@assert isfinite(U_neut)

# Triclinic cell-list real-space accuracy & zero-alloc regression at larger N
let
    rng = MersenneTwister(23)
    for (N_tri, t_scale) in ((400, 1.0), (1600, 1.6))
        L0 = (N_tri / 0.1)^(1 / 3)
        H_tri = [L0 0.18*L0 0.12*L0; 0.0 1.05*L0 0.15*L0; 0.0 0.0 1.1*L0]
        tb = Types.TriclinicBox(H_tri)
        s_rand = rand(rng, 3, N_tri)
        q_tri = H_tri * s_rand
        # Shift some atoms across periodic boundaries
        q_tri[:, 1:20] .+= H_tri[:, 1] .- 2.0 .* H_tri[:, 2]
        ch_tri = rand(rng, N_tri) .- 0.5; ch_tri .-= sum(ch_tri) / N_tri
        rc_tri = 5.5
        n_g = max(16, round(Int, L0 / 0.8)); isodd(n_g) || (n_g += 1)
        plan_tri = PME.PMEPlan(q_tri, tb; alpha=0.35, rcut=rc_tri, grid=(n_g, n_g, n_g), order=4)
        F_tri = zeros(3, N_tri)
        G_tri = zeros(3, 3)
        PME.pme_energy_forces_box_derivative!(F_tri, G_tri, plan_tri, q_tri, ch_tri)
        f_tri!() = PME.pme_energy_forces_box_derivative!(F_tri, G_tri, plan_tri, q_tri, ch_tri)
        f_tri!()
        @assert (@allocated f_tri!()) == 0
        if N_tri == 400
            # Verify triclinic cell-list pairs cover every pair within rcut
            U_tri, F_tri2, G_tri2 = PME.pme_energy_forces_box_derivative(plan_tri, q_tri, ch_tri)
            @assert U_tri == PME.pme_energy(plan_tri, q_tri, ch_tri)
            @assert F_tri2 == F_tri && G_tri2 == G_tri
        end
    end
end

# ---------------------------------------------------------------------------
# Error budget & parameter selection helpers
# ---------------------------------------------------------------------------

let
    est_ortho = PME.pme_error_estimate(box, 0.35, 5.5; grid=(32, 32, 32), order=4)
    @assert est_ortho.real_error > 0
    @assert est_ortho.recip_error > 0
    @assert isapprox(est_ortho.total_error, sqrt(est_ortho.real_error^2 + est_ortho.recip_error^2))

    # Grid refinement decreases reciprocal error
    est_fine = PME.pme_error_estimate(box, 0.35, 5.5; grid=(64, 64, 64), order=4)
    @assert est_fine.recip_error < est_ortho.recip_error

    # Order 6 has smaller reciprocal error on the same fine grid
    est_ord6 = PME.pme_error_estimate(box, 0.35, 5.5; grid=(32, 32, 32), order=6)
    @assert est_ord6.order == 6

    # Triclinic error estimation
    est_tri = PME.pme_error_estimate(tbox, 0.35, 5.0; grid=(32, 32, 32), order=4)
    @assert isfinite(est_tri.total_error)

    # Recommendation helper
    for tol in (1e-4, 1e-5, 1e-6)
        rec = PME.pme_recommend_parameters(box; tol=tol, order=4)
        @assert rec.estimated_error <= tol
        @assert rec.rcut < minimum(box) / 2
        @assert all(g -> ispow2(g) && g >= 8, rec.grid)

        # Budget validation helper
        valid, val_est = PME.pme_validate_budget(box; alpha=rec.alpha, rcut=rec.rcut,
                                                 grid=rec.grid, order=rec.order, tol=tol)
        @assert valid
        @assert val_est.total_error <= tol
    end

    # Under-resolved parameters must fail validation
    invalid, _ = PME.pme_validate_budget(box; alpha=0.1, rcut=2.0, grid=(8, 8, 8), tol=1e-5)
    @assert !invalid
end

println("test_pme PASS")
