"""
Lane B — interval arithmetic.

IntervalArithmetic is used as a *containment oracle*: evaluate a potential law
on the interval hull of a small position neighbourhood, then assert that the
engine's `Float64` evaluations at points inside that neighbourhood lie in the
enclosure. A wrong sign, a missing term, or a wrong geometry convention makes
the enclosure miss the engine value.

Checks:

1. Full bonded-chain energy enclosure over a `1e-6` ball in every coordinate,
   against 200 sampled engine energies.
2. Componentwise interval enclosure of the engine's `HarmonicBond` force.
3. Rigid-water constraint geometry after `project_positions!`: the O-H
   distance interval must contain the constrained distance and the squared
   residual interval must contain zero.
4. `MonteCarloBarostat.log_acceptance` enclosure over a box of `(dU, V, V')`,
   plus the exact antisymmetry `l(dU,V,V') = -l(-dU,V',V)` that a reversible
   volume move requires, and per-step consistency of `step!`'s returned fields.
"""

using IntervalArithmetic
using Random

struct IntervalMetrics
    chain_energy_width::Float64
    chain_energy_contained::Int
    chain_energy_samples::Int
    bond_force_contained::Bool
    bond_force_width::Float64
    constraint_contained::Bool
    constraint_residual_contained::Bool
    decorations_ok::Bool
    barostat_contained::Bool
    barostat_symmetry_err::Float64
    barostat_step_consistency_err::Float64
end

"Interval hull `[x - d, x + d]` of every entry."
function interval_hull(q::AbstractMatrix{Float64}, delta::Float64)
    qi = similar(q, Interval{Float64})
    for k in eachindex(q)
        qi[k] = interval(q[k] - delta, q[k] + delta)
    end
    return qi
end

contained(x::Float64, I::Interval) = inf(I) <= x <= sup(I)
width_of(I::Interval) = sup(I) - inf(I)

"Bond force on atom `i` of `f`: `-k (r - r0) / r * (q_i - q_j)` in intervals."
function bond_force_interval(q::AbstractMatrix{<:Real}, i::Int, j::Int,
                             k::Float64, r0::Float64)
    d = _vec(q, i) .- _vec(q, j)
    r = sqrt(_dot3(d, d))
    c = -k * (r - r0) / r
    return [c * d[1], c * d[2], c * d[3]]
end

function chain_energy_containment(nsample::Int = 200)
    q0 = chain_positions()
    delta = 1e-6
    qi = interval_hull(q0, delta)
    U_interval = chain_energy(qi)

    engine_energies = Float64[]
    rng = MersenneTwister(20261112)
    for _ in 1:nsample
        qs = copy(q0)
        for k in eachindex(qs)
            qs[k] += delta * (2rand(rng) - 1)
        end
        push!(engine_energies, chain_energy_forces(qs)[1])
    end
    n_in = count(u -> contained(u, U_interval), engine_energies)
    return width_of(U_interval), n_in, nsample, decoration(U_interval) == com
end

function bond_force_containment()
    q0 = chain_positions()
    delta = 1e-6
    qi = interval_hull(q0, delta)
    p = CHAIN
    Fi = bond_force_interval(qi, 1, 2, p.bond_k, p.bond_r0)
    bond = FactorLib.HarmonicBond(1, 2, p.bond_k, p.bond_r0)
    _, F = FactorLib.total_energy_forces([bond], q0, chain_context(size(q0, 2)))
    ok = contained(F[1, 1], Fi[1]) && contained(F[2, 1], Fi[2]) && contained(F[3, 1], Fi[3])
    dec_ok = all(I -> decoration(I) == com, Fi)
    return ok, maximum(width_of, Fi), dec_ok
end

function water_constraint_containment()
    cs, q0 = water_constraints()
    m = [16.0, 1.0, 1.0]
    qd = copy(q0)
    displacement = [0.051 * sin(1.3 * i + 0.7 * j) for i in 1:3, j in 1:3]
    qd .+= displacement
    ConstraintLib.project_positions!(qd, m, cs; tolerance = 1e-14)

    qi = interval_hull(qd, 1e-13)
    ok_dist = true
    ok_res2 = true
    dec_ok = true
    for (i, j) in ((1, 2), (1, 3), (2, 3))
        d = norm(q0[:, j] - q0[:, i])
        delta = _vec(qi, j) .- _vec(qi, i)
        d_interval = sqrt(_dot3(delta, delta))
        ok_dist &= contained(d, d_interval)
        dec_ok &= decoration(d_interval) == com
        # squared-distance residual |x_j - x_i|^2 - d^2 must enclose 0
        r2 = _dot3(delta, delta)
        residual = r2 - interval(d * d, d * d)
        ok_res2 &= contained(0.0, residual)
        dec_ok &= decoration(residual) == com
    end
    return ok_dist, ok_res2, dec_ok
end

function barostat_interval_and_stepping()
    beta, pressure, natom = 1.0, 0.35, 2
    dU = interval(-0.02, 0.05)
    V = interval(26.9, 27.1)
    Vn = interval(27.0, 27.2)
    log_ratio_interval = -beta * (dU + pressure * (Vn - V)) +
                         (natom + 1) * log(Vn / V)
    log_ratio_point = BarostatLib.log_acceptance(beta, pressure, 0.015, 27.0, 27.1, natom)
    ok = contained(log_ratio_point, log_ratio_interval)

    # exact reversibility of the exponent under swapping (dU, V) <-> (-dU, V')
    sym_err = 0.0
    for (du, v, vn) in ((0.015, 27.0, 27.1), (-0.4, 30.0, 28.5), (0.9, 25.0, 25.0 * exp(0.02)))
        fwd = BarostatLib.log_acceptance(beta, pressure, du, v, vn, natom)
        rev = BarostatLib.log_acceptance(beta, pressure, -du, vn, v, natom)
        sym_err = max(sym_err, abs(fwd + rev))
    end

    # step! field consistency over a long enough run to see both outcomes
    barostat = BarostatLib.VolumeMoveBarostat(beta, pressure, 0.08)
    box = TypesLib.OrthorhombicBox((3.0, 3.0, 3.0))
    q = [0.1 1.0; -0.2 0.7; 0.3 -0.4]
    rng = MersenneTwister(7)
    step_err = 0.0
    for _ in 1:300
        q_before = copy(q)
        box_before = box
        old_volume = TypesLib.box_volume(box_before)
        r = BarostatLib.step!(barostat, q, box, harmonic_trap_energy, rng)
        scale = exp(r.delta_log_volume / 3)
        candidate = q_before .* scale
        candidate_box = TypesLib.OrthorhombicBox(ntuple(i -> box_before.L[i] * scale, 3))
        step_err = max(step_err, abs(r.old_volume - old_volume) / old_volume)
        step_err = max(step_err, abs(r.new_volume - old_volume * exp(r.delta_log_volume)) /
                                 r.new_volume)
        step_err = max(step_err, abs(r.old_energy - harmonic_trap_energy(q_before, box_before)))
        step_err = max(step_err, abs(r.new_energy - harmonic_trap_energy(candidate, candidate_box)))
        step_err = max(step_err, abs(TypesLib.box_volume(r.box) -
                                     (r.accepted ? r.new_volume : old_volume)) / old_volume)
        expected = BarostatLib.log_acceptance(beta, pressure, r.new_energy - r.old_energy,
                                              r.old_volume, r.new_volume, size(q, 2))
        step_err = max(step_err, abs(r.log_acceptance - expected))
        q = r.accepted ? q : q_before
        box = r.box
    end
    dec_ok = decoration(log_ratio_interval) == com
    return ok, sym_err, step_err, dec_ok
end

function run_intervals()
    ew, nin, ntot, edec = chain_energy_containment()
    fok, fwidth, fdec = bond_force_containment()
    cdist, cres, cdec = water_constraint_containment()
    bok, sym, step, bdec = barostat_interval_and_stepping()
    return IntervalMetrics(ew, nin, ntot, fok, fwidth, cdist, cres,
                           edec && fdec && cdec && bdec, bok, sym, step)
end
