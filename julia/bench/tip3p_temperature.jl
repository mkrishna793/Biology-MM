#!/usr/bin/env julia
# TIP3P temperature/energy-leak diagnostics for the WaterBox module.
#
# This is the reproducible probe behind the fixes in `src/WaterBox.jl`:
#   * the reversible principal-axis (symmetric, second-order) rigid-body drift
#     replaces the dissipative SHAKE half-drift;
#   * the PME Ewald split is scaled so `alpha*rcut = 4.5`, which removes the
#     hard-cutoff energy jump that broke force/energy consistency.
#
# It prints, in order:
#   1. the PME split and the measured composed-field force finite difference;
#   2. NVE term-by-term drift (LJ vs PME vs kinetic) and the total band;
#   3. the free-rotor kinetic-energy check for the drift map;
#   4. the Langevin friction ladder with *all forces switched off* (this isolates
#      the thermostat/constraint composition) and with the full field.
#
# Usage:
#     cd julia
#     JULIA_DEPOT_PATH=... julia --project=. bench/tip3p_temperature.jl
#
# This is a diagnostic, not a validated physical trajectory.

using FoundationCore
using LinearAlgebra
using Random
using Statistics

const ROOT = normpath(joinpath(@__DIR__, ".."))
isdefined(FoundationCore, :WaterBox) ||
    Base.include(FoundationCore, joinpath(ROOT, "src", "WaterBox.jl"))
const WB = FoundationCore.WaterBox
const F = FoundationCore.Factors
const D = FoundationCore.Dynamics
const C = FoundationCore.Constraints

"Term energies (LJ, PME) and the composed force at `q`."
function term_state(field, q)
    N = size(q, 2)
    Flj = zeros(3, N)
    Ulj = F.evaluate!(Flj, field.lj, q, field.ctx)
    Fp = zeros(3, N)
    Up = F.evaluate!(Fp, field.pme, q, field.ctx)
    return Ulj, Up, Flj .+ Fp
end

"Central finite difference of the composed energy, worst component."
function composed_fd(field, q; h = 1e-5)
    N = size(q, 2)
    _, _, F0 = term_state(field, q)
    worst = 0.0
    worst_at = (0, 0)
    for j in 1:N, d in 1:3
        qp = copy(q); qm = copy(q)
        qp[d, j] += h; qm[d, j] -= h
        fd = -(WB.potential(field, qp) - WB.potential(field, qm)) / (2h)
        e = abs(fd - F0[d, j])
        if e > worst
            worst = e
            worst_at = (d, j)
        end
    end
    return worst, worst_at, maximum(abs, F0)
end

"Same box with charges and LJ epsilon zeroed, to isolate the constraint layer."
function zero_force_water(w)
    return WB.TIP3PWater(w.positions, w.masses, zero(w.charges), zero(w.epsilon),
                         w.sigma, w.molecules, w.constraints, w.lj_topology,
                         w.pme_corrections, w.box, w.density, w.seed)
end

"Laboratory angular momentum about the molecular centres of mass."
function lab_angular_momentum(q, p, w)
    L = zeros(3)
    for (o, h1, h2) in w.molecules
        idx = (o, h1, h2)
        M = sum(w.masses[i] for i in idx)
        xc = [sum(w.masses[i] * q[d, i] for i in idx) / M for d in 1:3]
        vc = [sum(p[d, i] for i in idx) / M for d in 1:3]
        for i in idx
            r = [q[d, i] - xc[d] for d in 1:3]
            pr = [p[d, i] - w.masses[i] * vc[d] for d in 1:3]
            L .+= cross(r, pr)
        end
    end
    return L
end

function friction_ladder(w, field; label, gamma, nsteps = 2000, init = nothing)
    tr = WB.WaterTrajectory(w, field; temperature = 300.0, seed = 11)
    if init !== nothing
        tr.st.q .= init[1]
        tr.st.p .= init[2]
        WB.forces(field, tr.st.q, tr.st.F)
        WB.tangent_force!(tr.st.F, tr.st.F, tr.st.q, w.masses, w.constraints)
        tr.st.force_valid = true
    end
    smp = WB.integrate!(tr, nsteps; dt = 0.002, temperature = 300.0,
                        gamma = gamma, sample_every = 10)
    tail = smp[(length(smp) ÷ 2 + 1):end]
    Tt = [s.T_inst for s in tail]
    T, Tse, Tiat = WB.mean_with_error(Tt)
    E0, E1 = smp[1].E, smp[end].E
    println("  ", label, " gamma=", gamma, ": T_tail=", round(T, digits = 2),
            " +/- ", round(Tse, digits = 2), " K (autocorr, g=",
            round(Tiat, digits = 1), "), E first/last=",
            round(E0, digits = 1), "/", round(E1, digits = 1))
    return T
end

function main()
    w = WB.tip3p_water(27; seed = 3)
    field = WB.water_field(w)
    println("27-water box: L=", w.box.L, " rcut=", field.rcut,
            " pme_rcut=", field.pme_rcut)
    println("PME split: alpha=", field.alpha, " alpha*rcut=",
            field.alpha * field.pme_rcut, " grid=", field.grid)
    tr = WB.WaterTrajectory(w, field; temperature = 0.0, seed = 1)
    rel = WB.relax!(tr; max_steps = 1200)
    q = copy(tr.st.q)
    println("relax: converged=", rel.converged, " U=", round(rel.u_final, digits = 2),
            " |F|max=", round(rel.max_force_final, digits = 2))

    worst, at, fmax = composed_fd(field, q)
    println("composed FD: |F|max=", round(fmax, digits = 3),
            " worst|F-FD|=", worst, " at ", at)

    # NVE term accounting.
    m = w.masses
    p = zeros(3, size(q, 2))
    rng = MersenneTwister(6)
    for j in 1:size(q, 2), d in 1:3
        p[d, j] = sqrt(m[j] * WB.KB * 300.0) * randn(rng)
    end
    C.project_velocities!(q, p, m, w.constraints; tolerance = WB.CONSTRAINT_TOL)
    Fbuf = zeros(3, size(q, 2))
    WB.forces(field, q, Fbuf)
    WB.tangent_force!(Fbuf, Fbuf, q, m, w.constraints)
    Ftan = similar(q)
    dt = 0.001
    E = Float64[]
    Ulj = Float64[]
    Up = Float64[]
    Ks = Float64[]
    for _ in 1:1000
        p .+= 0.5 * dt .* Fbuf
        WB._drift!(q, p, m, w.molecules, dt)
        WB._tangent_forces!(Fbuf, Ftan, q, m, w.constraints, field)
        p .+= 0.5 * dt .* Fbuf
        C.project_velocities!(q, p, m, w.constraints; tolerance = WB.CONSTRAINT_TOL)
        a, b, _ = term_state(field, q)
        push!(Ulj, a); push!(Up, b)
        push!(Ks, D.kinetic_energy(field.system, p))
        push!(E, a + b + Ks[end])
    end
    tp = 1000 * dt
    println("NVE 1 ps: band=", round(maximum(abs.(E .- E[1])), digits = 3),
            " kJ/mol, slope=", round((E[end] - E[1]) / tp, digits = 3),
            " kJ/mol/ps")
    println("  dU_LJ/ps=", round((Ulj[end] - Ulj[1]) / tp, digits = 2),
            "  dU_PME/ps=", round((Up[end] - Up[1]) / tp, digits = 2),
            "  dK/ps=", round((Ks[end] - Ks[1]) / tp, digits = 2))

    # Free-rotor drift check.
    w0 = zero_force_water(w)
    f0 = WB.water_field(w0)
    t0 = WB.WaterTrajectory(w0, f0; temperature = 300.0, seed = 5)
    q0 = t0.st.q
    p0 = t0.st.p
    K0 = D.kinetic_energy(f0.system, p0)
    L0 = lab_angular_momentum(q0, p0, w0)
    for _ in 1:2000
        WB._drift!(q0, p0, w0.masses, w0.molecules, 0.002)
    end
    K1 = D.kinetic_energy(f0.system, p0)
    L1 = lab_angular_momentum(q0, p0, w0)
    println("rigid free drift 4 ps: K ", round(K0, digits = 6), " -> ",
            round(K1, digits = 6), " (rel ", round(abs(K1 - K0) / K0, sigdigits = 3),
            "), max|ΔL_lab| = ", maximum(abs.(L1 .- L0)),
            " (|L| = ", round(norm(L0), digits = 6), ")")

    println("friction ladder, zero-force box (isolates constraints + thermostat):")
    for g in (1.0, 5.0, 100.0)
        friction_ladder(w0, f0; label = "zero", gamma = g)
    end
    # The full-field ladder needs a common starting point; a directly relaxed
    # lattice still converts potential energy and would be read as a
    # friction-dependent temperature. Two ps at strong coupling settles it.
    tr_eq = WB.WaterTrajectory(w, field; temperature = 300.0, seed = 11)
    eq = WB.equilibrate!(tr_eq; ps = 4.0, dt = 0.002, temperature = 300.0,
                         gamma = 5.0, sample_every = 10)
    println("full-field start: 4 ps gamma=5 -> T=", round(eq.temp_mean, digits = 2),
            " +/- ", round(eq.temp_se, digits = 2), " K, U=",
            round(WB.potential(field, tr_eq.st.q), digits = 1), " kJ/mol")
    init = (copy(tr_eq.st.q), copy(tr_eq.st.p))
    println("friction ladder, full field (27 waters, 4 ps from the same state):")
    for g in (1.0, 5.0, 100.0)
        friction_ladder(w, field; label = "full", gamma = g, init = init)
    end
end

main()
