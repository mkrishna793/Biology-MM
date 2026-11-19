# Rigid TIP3P water box: builder, composed field, constraints, exclusions, NVT.
#
#     cd julia
#     JULIA_DEPOT_PATH=... julia --project=. test/test_water_box.jl
#
# Assertion-driven like the rest of the suite (an uncaught throw is a failure).
# Every tolerance below is the measured value with an order of magnitude of
# headroom, and the measured number is printed so a regression is visible even
# when it still passes.

using FoundationCore
using LinearAlgebra
using Random
using Statistics

const WB_ROOT = normpath(joinpath(@__DIR__, ".."))
isdefined(FoundationCore, :WaterBox) ||
    Base.include(FoundationCore, joinpath(WB_ROOT, "src", "WaterBox.jl"))
const WB = FoundationCore.WaterBox
const T = FoundationCore.Types
const F = FoundationCore.Factors
const C = FoundationCore.Constraints
const D = FoundationCore.Dynamics
const SLJ = FoundationCore.StreamedLJ
const PME = FoundationCore.PME
const EW = FoundationCore.Electrostatics

const R_OH = WB.TIP3P_PARAMETERS.r_OH
const THETA = WB.TIP3P_PARAMETERS.theta_HOH
const R_HH = WB.R_HH

"Distance between two atoms (no periodic image: intramolecular geometry)."
bond(q, i, j) = sqrt(sum(abs2, view(q, :, j) .- view(q, :, i)))

"H-O-H angle in degrees for molecule `(o, h1, h2)`."
function hoh_angle(q, (o, h1, h2))
    u = q[:, h1] .- q[:, o]
    v = q[:, h2] .- q[:, o]
    return rad2deg(acos(clamp(dot(u, v) / (norm(u) * norm(v)), -1.0, 1.0)))
end

"Minimum O-O distance over all molecule pairs under the orthorhombic minimum image."
function min_oo(q, w)
    L = w.box.L
    best = Inf
    for b in 2:length(w.molecules), a in 1:(b - 1)
        oa, ob = 3a - 2, 3b - 2
        d = [q[1, oa] - q[1, ob], q[2, oa] - q[2, ob], q[3, oa] - q[3, ob]]
        for k in 1:3
            d[k] -= L[k] * round(d[k] / L[k])
        end
        best = min(best, sum(abs2, d))
    end
    return sqrt(best)
end

"Same box with every charge and LJ epsilon set to zero (constraint-only probe)."
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

"""
One-water audit of the rigid-body drift: lab angular momentum, kinetic energy,
exact reversibility, and second-order convergence. The old
rotate-(r,p)-together map failed the lab-L check by construction; the old SHAKE
half-drift failed the K/reversibility checks.
"""
function test_rigid_drift_audit()
    w = WB.tip3p_water(1; box = (2.0, 2.0, 2.0), seed = 3)
    w0 = zero_force_water(w)
    field = WB.water_field(w0)
    tr = WB.WaterTrajectory(w0, field; temperature = 300.0, seed = 5)
    q = tr.st.q
    p = tr.st.p
    m = w0.masses
    K0 = D.kinetic_energy(field.system, p)
    L0 = lab_angular_momentum(q, p, w0)
    band = 0.0
    Ldrift = 0.0
    for _ in 1:2000
        WB._drift!(q, p, m, w0.molecules, 0.002)
        band = max(band, abs(D.kinetic_energy(field.system, p) - K0))
        Ldrift = max(Ldrift, maximum(abs.(lab_angular_momentum(q, p, w0) .- L0)))
    end
    K1 = D.kinetic_energy(field.system, p)
    @assert band < 2e-5 * K0                     # measured 7.2e-6 relative
    @assert Ldrift < 1e-10 * max(norm(L0), 1.0)  # measured 4.2e-14 absolute
    println("  rigid free drift (1 water, 4 ps): K band ", band / K0,
            " relative, max |ΔL_lab| = ", Ldrift, " (|L| = ", round(norm(L0), digits = 6), ")")

    # Exact time reversibility of the drift map.
    q0 = copy(q)
    p0 = copy(p)
    for _ in 1:500
        WB._drift!(q, p, m, w0.molecules, 0.002)
    end
    p .*= -1
    for _ in 1:500
        WB._drift!(q, p, m, w0.molecules, 0.002)
    end
    p .*= -1
    dq = maximum(abs.(q .- q0))
    dp = maximum(abs.(p .- p0))
    @assert dq < 1e-9 && dp < 1e-9             # measured ~1e-13 / 1e-12
    println("  rigid drift reversibility (500+500 steps): max|Δq| = ", dq,
            ", max|Δp| = ", dp)

    # Second-order convergence of the symmetric principal-axis splitting.
    qref = copy(q0)
    pref = copy(p0)
    for _ in 1:10_000
        WB._drift!(qref, pref, m, w0.molecules, 1e-4)
    end
    errs = Float64[]
    for dt in (0.002, 0.001, 0.0005)
        qq = copy(q0)
        pp = copy(p0)
        for _ in 1:round(Int, 1.0 / dt)
            WB._drift!(qq, pp, m, w0.molecules, dt)
        end
        push!(errs, maximum(abs.(qq .- qref)))
    end
    r1 = errs[1] / errs[2]
    r2 = errs[2] / errs[3]
    @assert 3.0 < r1 < 5.0 && 3.0 < r2 < 5.0  # measured 4.03, 4.12
    println("  splitting order: dt errors ", errs[1], ", ", errs[2], ", ", errs[3],
            " (ratios ", round(r1, digits = 2), ", ", round(r2, digits = 2), ")")

    # The kinetic-energy band must also converge at second order.
    kbands = Float64[]
    Kref = D.kinetic_energy(field.system, p0)
    for dt in (0.002, 0.001, 0.0005)
        qq = copy(q0)
        pp = copy(p0)
        kb = 0.0
        for _ in 1:round(Int, 0.5 / dt)
            WB._drift!(qq, pp, m, w0.molecules, dt)
            kb = max(kb, abs(D.kinetic_energy(field.system, pp) - Kref))
        end
        push!(kbands, kb)
    end
    rk1 = kbands[1] / kbands[2]
    rk2 = kbands[2] / kbands[3]
    @assert 3.0 < rk1 < 5.0 && 3.0 < rk2 < 5.0  # measured 4.01, 4.00
    println("  K-band order: ", kbands[1], ", ", kbands[2], ", ", kbands[3],
            " kJ/mol (ratios ", round(rk1, digits = 2), ", ", round(rk2, digits = 2), ")")
    return (band, Ldrift, dq, dp)
end

"Forward/reverse NVE error for a box with the full composed field."
function nve_reversal_error(w; dt::Real = 0.001, nsteps::Integer = 500, seed = 5)
    field = WB.water_field(w)
    tr = WB.WaterTrajectory(w, field; temperature = 300.0, seed = seed)
    q = copy(tr.st.q)
    p = copy(tr.st.p)
    m = w.masses
    cs = w.constraints
    N = size(q, 2)
    F = zeros(3, N)
    Ftan = similar(F)
    WB.forces(field, q, F)
    WB.tangent_force!(F, F, q, m, cs)
    force_max = maximum(abs, F)
    q0 = copy(q)
    p0 = copy(p)
    for _ in 1:nsteps
        WB._nve_step!(q, p, F, Ftan, field, m, cs, w.molecules, Float64(dt))
    end
    p .*= -1
    for _ in 1:nsteps
        WB._nve_step!(q, p, F, Ftan, field, m, cs, w.molecules, Float64(dt))
    end
    p .*= -1
    return maximum(abs.(q .- q0)), maximum(abs.(p .- p0)), force_max
end

"""
Time reversal of the full velocity-Verlet + rigid-drift step. The one-water case
is labeled *isolated*: the only nonbonded pairs are excluded intramolecular
pairs, so it exercises the drift/kick/projection algebra, not an intermolecular
force. The two-water case adds a live O–O LJ/PME interaction.
"""
function test_nve_reversibility()
    w1 = WB.tip3p_water(1; box = (2.0, 2.0, 2.0), seed = 3)
    dq1, dp1, f1 = nve_reversal_error(w1; dt = 0.001, nsteps = 500, seed = 5)
    @assert dq1 < 1e-9 && dp1 < 1e-9           # measured 5.0e-14 / 5.5e-13
    println("  NVE reversibility, isolated water (intramolecular exclusions only, ",
            "|F|max=", round(f1, sigdigits = 3), "): max|Δq| = ", dq1,
            ", max|Δp| = ", dp1)

    w2 = WB.tip3p_water(2; box = (1.6, 1.6, 1.6), seed = 4)
    dq2, dp2, f2 = nve_reversal_error(w2; dt = 0.0005, nsteps = 200, seed = 5)
    @assert f2 > 1.0                            # the O–O interaction is live
    @assert dq2 < 1e-8 && dp2 < 1e-8
    println("  NVE reversibility, interacting two-water (|F|max=",
            round(f2, sigdigits = 3), ", dt=0.5 fs): max|Δq| = ", dq2,
            ", max|Δp| = ", dp2)
    return (dq1, dp1, dq2, dp2, f2)
end

"""
A conservative drift plus a correct thermostat makes the stationary temperature
independent of the friction. The old SHAKE half-drift failed this with every
force switched off: 259 K at gamma=1/ps and 304 K at gamma=100/ps.
"""
function test_constraint_thermostat_gamma_independence()
    w = WB.tip3p_water(64; seed = 6)
    w0 = zero_force_water(w)
    field = WB.water_field(w0)
    means = Float64[]
    ses = Float64[]
    iats = Float64[]
    for g in (1.0, 100.0)
        tr = WB.WaterTrajectory(w0, field; temperature = 300.0, seed = 6)
        smp = WB.integrate!(tr, 2000; dt = 0.002, temperature = 300.0,
                            gamma = g, sample_every = 10)
        tail = smp[(length(smp) ÷ 2 + 1):end]
        Tt = [s.T_inst for s in tail]
        tm, tse, iat = WB.mean_with_error(Tt)
        push!(means, tm)
        push!(ses, tse)
        push!(iats, iat)
    end
    @assert abs(means[1] - 300.0) < 35.0
    @assert abs(means[2] - 300.0) < 35.0
    @assert abs(means[1] - means[2]) < 20.0
    # The tail is only 2 ps (100 saved samples); the error bars are
    # autocorrelation-corrected, not independent-sample sqrt(n) bars.
    println("  constraint-only thermostat (autocorr SE, ~2 ps tail, 100 samples): ",
            "gamma=1 -> ", round(means[1], digits = 1), " +/- ",
            round(ses[1], digits = 1), " K (g=", round(iats[1], digits = 1),
            "), gamma=100 -> ", round(means[2], digits = 1), " +/- ",
            round(ses[2], digits = 1), " K (g=", round(iats[2], digits = 1),
            "), |Δ| = ", round(abs(means[1] - means[2]), digits = 2), " K")
    return means
end

"""
Deterministic uniform translation on a zero-force box: the unwrapped molecular
COM stream must move by `v t` per molecule. The old `production!` divided every
COM by the sum of all oxygen masses, which shrinks the stream by
`M_water/(N m_O)` and makes the MSD/D estimate scale like `N^-2`.
"""
function test_production_uniform_translation()
    v = (0.10, -0.05, 0.02)                 # nm/ps
    msd = Dict{Int,Float64}()
    Ds = Dict{Int,Float64}()
    for n in (2, 4)
        w = WB.tip3p_water(n; box = (2.0, 2.0, 2.0), seed = 7)
        w0 = zero_force_water(w)
        field = WB.water_field(w0)
        tr = WB.WaterTrajectory(w0, field; temperature = 300.0, seed = 1)
        for j in 1:size(tr.st.p, 2), d in 1:3
            tr.st.p[d, j] = w0.masses[j] * v[d]
        end
        io = IOBuffer()
        res = WB.production!(tr; ps = 0.05, dt = 0.001, temperature = 300.0,
                             gamma = 0.0, sample_every = 1, frame_every = 2,
                             rdf_frames = 0, com_io = io)
        expected = (v[1]^2 + v[2]^2 + v[3]^2) .*
                   (res.msd_time_ps .- res.msd_time_ps[1]).^2
        @assert maximum(abs.(res.msd_nm2 .- expected)) < 1e-12
        @assert all(isfinite, res.msd_nm2)

        # The COM stream itself must be the true translation, for every molecule.
        first = Dict{Int,NTuple{4,Float64}}()
        for line in split(String(take!(io)), '\n')
            isempty(line) && continue
            t, i, x, y, z = split(line, ',')
            key = parse(Int, i)
            vals = (parse(Float64, t), parse(Float64, x), parse(Float64, y),
                    parse(Float64, z))
            haskey(first, key) || (first[key] = vals; continue)
            t0, x0, y0, z0 = first[key]
            dtp = vals[1] - t0
            # com_io is written with %.10f, so 1e-8 is the parse-safe bound.
            @assert abs(vals[2] - x0 - v[1] * dtp) < 1e-8
            @assert abs(vals[3] - y0 - v[2] * dtp) < 1e-8
            @assert abs(vals[4] - z0 - v[3] * dtp) < 1e-8
        end
        msd[n] = res.msd_nm2[end]
        Ds[n] = res.diffusion
        println("  uniform translation n=", n, ": COM/MSD exact, last MSD = ",
                round(msd[n], sigdigits = 6), " nm^2, D = ",
                round(Ds[n], sigdigits = 6), " nm^2/ps")
    end
    @assert abs(msd[2] - msd[4]) < 1e-14
    @assert abs(Ds[2] / Ds[4] - 1.0) < 1e-6   # not the old N^-2 scaling
    return msd
end

function test_build_counts()
    w = WB.tip3p_water(27; seed = 3)
    @assert size(w.positions, 2) == 81
    @assert length(w.molecules) == 27
    @assert length(w.masses) == 81 && length(w.charges) == 81
    @assert length(w.epsilon) == 81 && length(w.sigma) == 81
    @assert abs(sum(w.charges)) < 1e-12          # neutral box
    @assert sum(w.masses) ≈ 27 * 18.0154 rtol = 1e-12
    @assert length(w.constraints.components) == 27
    for comp in w.constraints.components
        @assert length(comp.atoms) == 3
        @assert length(comp.edges) == 3
    end
    @assert length(w.lj_topology.excluded) == 3 * 27
    # O site carries the LJ parameters, H sites carry eps = 0.
    for (o, h1, h2) in w.molecules
        @assert w.epsilon[o] ≈ 0.636386
        @assert w.sigma[o] ≈ 0.315061
        @assert w.epsilon[h1] == 0.0 && w.epsilon[h2] == 0.0
        @assert w.charges[o] == -0.834
        @assert w.charges[h1] == 0.417 && w.charges[h2] == 0.417
    end
    # Cube lattice, density exactly 1.0 g/cm^3.
    @assert w.box.L[1] ≈ w.box.L[2] ≈ w.box.L[3]
    @assert WB.box_density(w) ≈ 1.0 rtol = 1e-12
    println("  counts: 27 molecules / 81 atoms, 27 rigid components, density ",
            round(WB.box_density(w), digits = 9), " g/cm^3, L=",
            round.(w.box.L, digits = 5))
    return w
end

function test_rigid_geometry()
    w = WB.tip3p_water(64; seed = 8)
    worst_oh = 0.0
    worst_hh = 0.0
    worst_ang = 0.0
    for m in w.molecules
        o, h1, h2 = m
        worst_oh = max(worst_oh, abs(bond(w.positions, o, h1) - R_OH),
                       abs(bond(w.positions, o, h2) - R_OH))
        worst_hh = max(worst_hh, abs(bond(w.positions, h1, h2) - R_HH))
        worst_ang = max(worst_ang, abs(hoh_angle(w.positions, m) - THETA))
    end
    @assert worst_oh < 1e-12
    @assert worst_hh < 1e-12
    @assert worst_ang < 1e-10
    println("  geometry: max|O-H - $(R_OH)| = ", worst_oh,
            ", max|H-H - $(round(R_HH, digits = 6))| = ", worst_hh,
            ", max|angle - $(THETA)| deg = ", worst_ang)
    return w
end

function test_no_overlap(w)
    d = min_oo(w.positions, w)
    @assert d >= 0.25 - 1e-12
    # 0.310 nm is the 1 g/cm^3 lattice constant; anything much below it means
    # the builder compressed the cell.
    @assert d > 0.25
    println("  contact check: min O-O = ", round(d, digits = 6), " nm >= 0.25 nm")
end

"Brute-force intramolecular point-Coulomb energy of every excluded pair."
function intramol_coulomb(w)
    U = 0.0
    for (o, h1, h2) in w.molecules
        for (i, j) in ((o, h1), (o, h2), (h1, h2))
            r = bond(w.positions, i, j)
            U += WB.COULOMB_PREFACTOR * w.charges[i] * w.charges[j] / r
        end
    end
    return U
end

function test_exclusions_present(w)
    for (o, h1, h2) in w.molecules
        for (i, j) in ((o, h1), (o, h2), (h1, h2))
            @assert SLJ.pair_weight(w.lj_topology, i, j) == 0.0
            @assert haskey(w.pme_corrections.weights, (min(i, j), max(i, j)))
            @assert w.pme_corrections.weights[(min(i, j), max(i, j))] == 0.0
        end
    end
    # An intermolecular pair must not be excluded.
    @assert SLJ.pair_weight(w.lj_topology, 1, 6) == 1.0
    println("  exclusions: ", length(w.lj_topology.excluded),
            " LJ pairs and ", length(w.pme_corrections.weights),
            " PME pair corrections, all weight 0")
end

"LJ exclusion is *measurable*: same topology, all atoms given real LJ parameters."
function test_lj_exclusion_energy(w)
    N = size(w.positions, 2)
    ctx = F.Context(w.box, WB.TIP3P_UNITS, N)
    eps = fill(1.0, N)
    sig = fill(0.3, N)
    with = SLJ.StreamedLJFactor(w.positions, eps, sig, w.box; rcut = 0.45,
                                mode = :lfs, topology = w.lj_topology)
    without = SLJ.StreamedLJFactor(w.positions, eps, sig, w.box; rcut = 0.45,
                                   mode = :lfs, topology = nothing)
    uw = F.evaluate!(zeros(3, N), with, w.positions, ctx)
    un = F.evaluate!(zeros(3, N), without, w.positions, ctx)
    # Exact excluded-pair sum at the actual (unwrapped) geometry, with the
    # same :lfs shift and Lorentz-Berthelot mixing.
    rc = 0.45
    expected = 0.0
    for (o, h1, h2) in w.molecules
        for (i, j) in ((o, h1), (o, h2), (h1, h2))
            r = bond(w.positions, i, j)
            eij = sqrt(eps[i] * eps[j])
            sij = 0.5 * (sig[i] + sig[j])
            s6 = (sij^2 / r^2)^3
            s12 = s6 * s6
            s6c = (sij^2 / rc^2)^3
            s12c = s6c * s6c
            duc = 24 * eij * (s6c - 2 * s12c) / rc
            expected += 4 * eij * ((s12 - s6) - (s12c - s6c)) - duc * (r - rc)
        end
    end
    @assert un - uw ≈ expected rtol = 1e-10
    @assert abs(un - uw) > 1e-6           # the exclusion is not a no-op
    # (the magnitude is huge because sigma=0.3 nm at a 0.0957 nm O-H bond is a
    # deliberately unphysical probe of the exclusion path, not a water model)
    println("  LJ exclusions zero ", round(un - uw, digits = 6),
            " kJ/mol (vs exact excluded-pair sum ", round(expected, digits = 6), ")")
end

"Coulomb exclusion is *measurable*: drop the PME pair corrections and compare."
function test_coulomb_exclusion_energy(w)
    N = size(w.positions, 2)
    ctx = F.Context(w.box, WB.TIP3P_UNITS, N)
    L = w.box.L
    rc = WB.default_pme_rcut(L)
    with = PME.PMEFactor(w.positions, w.charges, w.box; alpha = 3.5, rcut = rc,
                         grid = WB.default_grid(L, 6), order = 6,
                         prefactor = WB.COULOMB_PREFACTOR,
                         corrections = w.pme_corrections)
    without = PME.PMEFactor(w.positions, w.charges, w.box; alpha = 3.5, rcut = rc,
                            grid = WB.default_grid(L, 6), order = 6,
                            prefactor = WB.COULOMB_PREFACTOR)
    uw = F.evaluate!(zeros(3, N), with, w.positions, ctx)
    un = F.evaluate!(zeros(3, N), without, w.positions, ctx)
    expected = intramol_coulomb(w)
    @assert abs((un - uw) - expected) < 1e-3 * abs(expected) + 1e-4
    @assert abs(un - uw) > 100.0          # ~ -500 kJ/mol per molecule
    println("  Coulomb exclusions remove ", round(un - uw, digits = 3),
            " kJ/mol (analytic ", round(expected, digits = 3), ", PME-mesh error ",
            round(abs((un - uw) - expected), digits = 5), ")")
end

"100 BAOAB steps: the rigid geometry must hold to <1e-10 nm and velocities stay tangent."
function test_constrained_steps(w)
    field = WB.water_field(w)
    tr = WB.WaterTrajectory(w, field; temperature = 300.0, seed = 4)
    WB.rethermalize!(tr; temperature = 300.0)
    n0 = 100
    worst_oh = 0.0
    worst_hh = 0.0
    worst_tan = 0.0
    for _ in 1:n0
        WB.integrate!(tr, 1; dt = 0.002, temperature = 300.0, gamma = 1.0,
                      sample_every = 1)
        for m in w.molecules
            o, h1, h2 = m
            worst_oh = max(worst_oh, abs(bond(tr.st.q, o, h1) - R_OH),
                           abs(bond(tr.st.q, o, h2) - R_OH))
            worst_hh = max(worst_hh, abs(bond(tr.st.q, h1, h2) - R_HH))
        end
        worst_tan = max(worst_tan,
                        maximum(abs.(C.tangent_residuals(w.constraints, tr.st.q,
                                                         tr.st.p, w.masses))))
    end
    @assert worst_oh < 1e-10
    @assert worst_hh < 1e-10
    @assert worst_tan < 1e-9
    @assert all(isfinite, tr.st.q) && all(isfinite, tr.st.p)
    println("  ", n0, " constrained steps: max|O-H - r0| = ", worst_oh,
            ", max|H-H - R_HH| = ", worst_hh, ", max|r.v_rel| = ", worst_tan)
end

"Composed field: analytic force == -dU/dq on a small LJ+PME box."
function test_force_vs_finite_difference()
    w = WB.tip3p_water(2; box = (2.0, 2.0, 2.0), seed = 12)
    # Pull the molecules onto one 0.42 nm O-O contact. On the build lattice the
    # spacing exceeds the default 0.9 nm LJ cutoff, so every O-O pair is outside
    # the cutoff and the LJ force is identically zero: a dropped LJ term would
    # then be invisible to this test. The compressed pair makes it live.
    o1 = w.molecules[1][1]
    o2 = w.molecules[2][1]
    delta = (w.positions[:, o1] .+ [0.42, 0.0, 0.0]) .- w.positions[:, o2]
    for atom in w.molecules[2]
        w.positions[:, atom] .+= delta
    end
    @assert abs(min_oo(w.positions, w) - 0.42) < 1e-12
    field = WB.water_field(w)
    q = copy(w.positions)
    q0 = copy(q)
    N = size(q, 2)
    Fbuf = zeros(3, N)
    U = WB.forces(field, q, Fbuf)
    @assert q == q0                        # forces must not mutate positions

    # (a) The composition itself: the total force must be the sum of the parts.
    #     This is the regression test for PME.evaluate! zeroing its accumulator,
    #     which silently dropped every other factor's force.
    Flj = zeros(3, N); Ulj = F.evaluate!(Flj, field.lj, q, field.ctx)
    Fpme = zeros(3, N); Upme = F.evaluate!(Fpme, field.pme, q, field.ctx)
    @assert maximum(abs, Fbuf .- Flj .- Fpme) < 1e-10
    @assert U ≈ Ulj + Upme rtol = 1e-12
    @assert abs(Ulj) > 1e-3 && maximum(abs, Flj) > 1e-2   # LJ is actually live

    # (b) Central differences of the *composed* energy.
    h = 1e-6
    worst = 0.0
    for j in 1:N, d in 1:3
        qp = copy(q); qm = copy(q)
        qp[d, j] += h
        qm[d, j] -= h
        fd = -(WB.potential(field, qp) - WB.potential(field, qm)) / (2h)
        worst = max(worst, abs(fd - Fbuf[d, j]))
    end
    @assert worst < 5e-4
    @assert worst < 0.5 * maximum(abs, Flj)   # not just the PME force again
    println("  force check: |F|max = ", round(maximum(abs, Fbuf), digits = 3),
            " kJ/mol/nm (|F_LJ|max = ", round(maximum(abs, Flj), digits = 3),
            "), max|F_analytic - F_fd| = ", worst,
            " over ", 3N, " components (tol 5e-4)")
    return field
end

"PME at production settings must agree with the independent direct Ewald sum."
function test_pme_vs_ewald()
    w = WB.tip3p_water(4; box = (2.0, 2.0, 2.0), seed = 21)
    field = WB.water_field(w)
    q = copy(w.positions)
    N = size(q, 2)
    Upme, Fpme = PME.pme_energy_forces(field.pme.plan, q, w.charges;
                                       corrections = field.pme.corrections)
    cew = EW.PairCorrections(excluded = collect(keys(w.pme_corrections.weights)))
    Uew, Few = EW.evaluate_ewald(q, w.charges, collect(w.box.L);
                                 alpha = field.alpha, rcut = field.pme_rcut,
                                 kmax = 12, prefactor = WB.COULOMB_PREFACTOR,
                                 corrections = cew)
    dU = abs(Upme - Uew)
    dF = maximum(abs, Fpme .- Few)
    @assert field.alpha * field.pme_rcut ≈ WB.PME_ALPHA_TARGET rtol = 1e-12
    @assert dU < 5e-3
    @assert dF < 5e-2
    println("  PME vs direct Ewald (grid ", field.grid[1], "^3, order ", field.order,
            ", alpha ", field.alpha, "): dU = ", round(dU, digits = 6),
            " kJ/mol, max dF = ", round(dF, digits = 6), " kJ/mol/nm")
end

"""
PME real-space with 3+ cells per axis must agree with direct Ewald. The old
`nc <= 2` gate was based on a fixed per-cell-pair shift that the current
`PME.PMEPlan` loop no longer uses; its real-space loop applies the minimum
image per pair.
"""
function test_pme_multicell_parity()
    w = WB.tip3p_water(4; box = (2.0, 2.0, 2.0), seed = 21)
    field = WB.water_field(w; pme_rcut = 0.6)
    nc = floor(Int, minimum(w.box.L) / field.pme_rcut)
    @assert nc >= 3
    @assert field.alpha * field.pme_rcut ≈ WB.PME_ALPHA_TARGET rtol = 1e-12
    q = copy(w.positions)
    Upme, Fpme = PME.pme_energy_forces(field.pme.plan, q, w.charges;
                                       corrections = field.pme.corrections)
    cew = EW.PairCorrections(excluded = collect(keys(w.pme_corrections.weights)))
    Uew, Few = EW.evaluate_ewald(q, w.charges, collect(w.box.L);
                                 alpha = field.alpha, rcut = field.pme_rcut,
                                 kmax = 24, prefactor = WB.COULOMB_PREFACTOR,
                                 corrections = cew)
    dU = abs(Upme - Uew)
    dF = maximum(abs, Fpme .- Few)
    @assert dU < 5e-3
    @assert dF < 5e-2
    @assert WB.default_pme_rcut((2.0, 2.0, 2.0)) == 0.9
    small = WB.default_pme_rcut((0.93, 0.93, 0.93))
    @assert small < 0.5 * 0.93
    println("  PME multicell parity: ", nc, " cells/axis at rcut=",
            field.pme_rcut, " nm, alpha=", round(field.alpha, digits = 3),
            ", grid=", field.grid[1], "^3: dU = ", round(dU, digits = 6),
            " kJ/mol, max dF = ", round(dF, digits = 6), " kJ/mol/nm")
end

"NVE shadow energy over 1000 steps must stay bounded (band, slope, HAC error)."
function test_energy_bounded()
    # 64 molecules: a 27-molecule cell (L = 0.93 nm) has only ~5 oxygens inside
    # its 0.42 nm cutoff and is too poorly coordinated for a meaningful
    # energy-conservation statement.
    w = WB.tip3p_water(64; seed = 6)
    field = WB.water_field(w)
    tr = WB.WaterTrajectory(w, field; temperature = 300.0, seed = 6)
    rel = WB.relax!(tr; max_steps = 1200)
    WB.rethermalize!(tr; temperature = 300.0)
    band, slope, se, ns = WB.nve_energy_band(tr; nsteps = 1000, dt = 0.001)
    U0 = WB.potential(field, tr.st.q)
    @assert ns == 1000
    @assert isfinite(band)
    @assert rel.converged
    @assert U0 < -40.0 * length(w.molecules)      # a real hydrogen-bond network
    @assert band < 0.01 * abs(U0)
    println("  1000 NVE steps (1 fs): band = ", round(band, digits = 3),
            " kJ/mol (", round(100 * band / abs(U0), digits = 3), "% of |U| = ",
            round(abs(U0), digits = 1), "), slope = ", round(slope, digits = 2),
            " +/- ", round(se, digits = 2), " kJ/mol/ps")
    # A thermostatted run must also stay in a sane window: no blow-up, no NaN.
    # The first ~1 ps converts the minimiser's potential energy back into
    # kinetic energy, so the *tail* is the meaningful stationary estimate.
    smp = WB.integrate!(tr, 2000; dt = 0.002, temperature = 300.0, gamma = 1.0,
                        sample_every = 10)
    Tt = [s.T_inst for s in smp]
    Ttail = Tt[(length(Tt) ÷ 2):end]
    @assert all(isfinite, Tt)
    @assert 200.0 < mean(Ttail) < 400.0
    @assert all(abs.(Tt .- 300.0) .< 400.0)
    println("  2000 Langevin steps (2 fs, gamma=1/ps): T tail mean ",
            round(mean(Ttail), digits = 1), " K, min ", round(minimum(Tt), digits = 1),
            ", max ", round(maximum(Tt), digits = 1), ", no NaN")
    return band
end

"Langevin at a very strong coupling must reproduce the target temperature."
function test_thermostat_target(w)
    field = WB.water_field(w)
    tr = WB.WaterTrajectory(w, field; temperature = 300.0, seed = 9)
    WB.rethermalize!(tr; temperature = 300.0)
    samples = WB.integrate!(tr, 1200; dt = 0.002, temperature = 300.0,
                            gamma = 1000.0, sample_every = 2)
    tail = samples[(length(samples) ÷ 2 + 1):end]
    tm, tse, g = WB.mean_with_error([s.T_inst for s in tail])
    @assert abs(tm - 300.0) < 15.0
    println("  thermostat: gamma=1000/ps gives T = ", round(tm, digits = 2), " +/- ",
            round(tse, digits = 2), " K for a 300 K target (g = ", round(g, digits = 2), ")")
    return tm
end

"Kinetic DOF, Maxwell-Boltzmann initialization and the (nm, kJ/mol, amu, ps) units."
function test_kinetic_dof_and_temperature()
    w = WB.tip3p_water(8; seed = 5)
    field = WB.water_field(w)
    nmol = length(w.molecules)

    tr = WB.WaterTrajectory(w, field; temperature = 300.0, seed = 1)
    @assert tr.dof == 6nmol - 3               # 3N atoms - 3 rigid constraints/N - 3 COM
    for d in 1:3
        @assert abs(sum(tr.st.p[d, :])) < 1e-12   # COM momentum removed
    end

    # Equipartition over independent momentum draws: <T> recovers the target
    # within the 2/dof relative fluctuation (dof = 45, so a 40-draw mean has a
    # ~3% standard error; the window is deliberately wide).
    temps = Float64[]
    for seed in 1:40
        trs = WB.WaterTrajectory(w, field; temperature = 300.0, seed = seed)
        push!(temps, WB.temperature_of(trs, trs.st.p))
    end
    @assert all(isfinite, temps) && all(temps .> 0.0)
    @assert 240.0 <= mean(temps) <= 360.0

    # Zero temperature: exactly zero momenta and zero kinetic temperature.
    tr0 = WB.WaterTrajectory(w, field; temperature = 0.0, seed = 1)
    @assert all(tr0.st.p .== 0.0)
    @assert WB.temperature_of(tr0, tr0.st.p) == 0.0

    # Unit convention: p_j = m_j * (1 nm/ps) gives K = 0.5 sum(m) kJ/mol, since
    # 1 amu nm^2/ps^2 = 1 kJ/mol exactly (no conversion factor in Dynamics).
    trd = WB.WaterTrajectory(w, field; temperature = 300.0, seed = 2)
    fill!(trd.st.p, 0.0)
    for j in axes(trd.st.p, 2)
        trd.st.p[1, j] = w.masses[j]
    end
    K = D.kinetic_energy(field.system, trd.st.p)
    @assert abs(K - 0.5 * sum(w.masses)) < 1e-12
    # beta = 1/(k_B T): the temperature estimator is 2K/(dof k_B), with the same
    # k_B declared in TIP3P_UNITS.
    @assert abs(WB.temperature_of(trd, trd.st.p) - 2K / (trd.dof * WB.KB)) < 1e-12
    T1 = WB.temperature_of(trd, trd.st.p)
    trd.st.p .*= 2.0
    @assert abs(WB.temperature_of(trd, trd.st.p) - 4T1) < 1e-10

    u = WB.TIP3P_UNITS
    @assert u.boltzmann == WB.KB
    @assert abs(u.mass * (u.length / 1.0e-12)^2 / u.energy - 1.0) < 1e-9

    println("  kinetic: dof = ", tr.dof, ", <T> = ", round(mean(temps), digits = 2),
            " K (target 300), K = 0.5 p^2/m in kJ/mol, k_B = ", WB.KB)
end

"Analysis helpers: RDF normalisation, MSD-based diffusion, IAT/HAC error bars."
function test_analysis_helpers(w)
    r, g = WB.radial_distribution(w.positions, w; rmax = 0.8, nbins = 100)
    # g(r < 0.25) must be zero (the builder guarantees it) and the normalisation
    # must be finite and positive at the first shell.
    @assert all(g[r .< 0.24] .== 0.0)
    @assert any(gi -> gi > 0.5, g)

    # A synthetic diffusive series: MSD = 6Dt with D = 0.002 nm^2/ps.
    t = collect(0.0:1.0:100.0)
    msd = 6 * 0.002 .* t
    Dfit, Dse, win = WB.diffusion_from_msd(t, msd)
    @assert abs(Dfit - 0.002) < 1e-12
    @assert win == (t[50], t[end])

    # Deterministic 3D Brownian walkers: each component gains variance 2 D dt
    # per step, so the ensemble MSD is 6 D t. Seeded, hence reproducible.
    Db = 0.002
    dtb = 0.05
    nb = 160
    nwalk = 3000
    rb = MersenneTwister(2024)
    pos = zeros(3, nwalk)
    msdb = zeros(Float64, nb + 1)
    sig = sqrt(2Db * dtb)
    for k in 1:nb
        for walker in 1:nwalk, d in 1:3
            pos[d, walker] += sig * randn(rb)
        end
        msdb[k + 1] = sum(abs2, pos) / nwalk
    end
    tb = collect(0.0:dtb:nb * dtb)
    Dbt, sebt, winb = WB.diffusion_from_msd(tb, msdb)
    @assert isfinite(Dbt) && isfinite(sebt)
    @assert abs(Dbt - Db) <= 0.05 * Db
    @assert winb[1] > 0.0 && winb[2] == tb[end]

    # White noise has g ~ 1; a strongly correlated AR(1) series must give g > 1.
    rng = MersenneTwister(1)
    white = randn(rng, 4000)
    gwhite, _ = WB.autocorrelation_time(white)
    x = zeros(4000)
    for i in 2:4000
        x[i] = 0.9 * x[i - 1] + randn(rng)
    end
    gcorr, _ = WB.autocorrelation_time(x)
    @assert 0.8 < gwhite < 1.4
    @assert gcorr > 5.0

    # HAC slope: a deterministic line is recovered exactly, and the error bar is
    # finite for correlated residuals.
    y = 2.5 .* t .+ 1.0
    sl, se, _ = WB.ols_slope_hac(t, y)
    @assert abs(sl - 2.5) < 1e-9
    @assert se < 1e-6 || isnan(se)
    println("  analysis: g(r) normalised, D(synthetic) = ", Dfit,
            " nm^2/ps, D(Brownian) = ", round(Dbt, sigdigits = 4),
            " nm^2/ps, IAT g_white = ", round(gwhite, digits = 2),
            ", g_correlated = ", round(gcorr, digits = 2))
end

"State machine stage verification and reversible/stochastic classification for rigid water."
function test_water_integrator_state_machine()
    nve_spec = WB.water_integrator_spec(:rigid_water_nve)
    baoab_spec = WB.water_integrator_spec(:rigid_water_baoab)
    @assert D.is_reversible(nve_spec) && !D.is_stochastic(nve_spec)
    @assert !D.is_reversible(baoab_spec) && D.is_stochastic(baoab_spec)
    @assert count(D.is_stochastic, baoab_spec.stages) == 1

    w = WB.tip3p_water(2; box = (1.6, 1.6, 1.6), seed = 4)
    field = WB.water_field(w)
    tr = WB.WaterTrajectory(w, field; temperature = 300.0, seed = 5)
    nve_trace = Symbol[]
    WB.run_nve!(tr, 1; dt = 0.001, stage_hook = s -> push!(nve_trace, s))
    @assert nve_trace == [:half_kick, :rigid_body_drift, :force_evaluation, :half_kick, :velocity_constraints]
    @assert all(s -> D.stage_classification(s) === :reversible, nve_trace)

    baoab_trace = Symbol[]
    WB._baoab_step!(tr, 0.002, 1.0, 1.0 / (WB.KB * 300.0);
                    stage_hook = s -> push!(baoab_trace, s))
    @assert baoab_trace == [:force_evaluation, :half_kick, :velocity_constraints,
                            :rigid_body_drift, :thermostat_ou, :rigid_body_drift,
                            :force_evaluation, :half_kick, :velocity_constraints]
    @assert count(s -> D.stage_classification(s) === :stochastic, baoab_trace) == 1
    println("  water state machine: NVE (reversible) and BAOAB (stochastic) traces match specs")
end

"""
Phase 4 NVE audit on rigid TIP3P water (64 molecules, NO thermostat):
1. Timestep-halving O(dt^2) convergence of the NVE energy-error band.
2. Explicit diagnosis of water potential trend:
   - Un-equilibrated post-relax NVE shows initial equipartition/structural relaxation
     (U <-> K exchange with conserved E = U + K).
   - Post-equilibration NVE shows stationary U, K, and E (slopes within SE of zero)
     and O(dt^2) energy-error bands.
3. Constraint residuals and linear/angular momentum behavior:
   - Zero-force rigid water conserves both linear momentum P and molecular lab
     angular momentum L_lab to ~1e-13.
   - Live PME+LJ conserves E to O(dt^2), while B-spline PME mesh breaks continuous
     translation invariance (measurable non-zero net PME force -> ΔP != 0).
"""
function test_water_nve_convergence_and_potential_trend()
    w = WB.tip3p_water(64; seed = 6)
    field = WB.water_field(w)
    tr = WB.WaterTrajectory(w, field; temperature = 300.0, seed = 6)
    rel = WB.relax!(tr; max_steps = 1200)
    @assert rel.converged

    # Equilibrate briefly (3 ps NVT) then switch thermostat OFF for pure NVE tests.
    tr_eq = WB.WaterTrajectory(w, field; temperature = 300.0, seed = 6)
    tr_eq.st.q .= tr.st.q
    WB.equilibrate!(tr_eq; ps = 3.0, dt = 0.002, temperature = 300.0, gamma = 5.0)

    # (a) Timestep-halving O(dt^2) NVE energy-band convergence over 0.5 ps (same trajectory regime).
    bands = Float64[]
    dts = (0.002, 0.001, 0.0005)
    for dt in dts
        nsteps = round(Int, 0.5 / dt)
        res = WB.run_nve!(tr_eq, nsteps; dt = dt, sample_every = max(1, round(Int, 0.01 / dt)))
        push!(bands, res.diagnostics.energy_band)
        @assert res.diagnostics.max_pos_residual < 1e-12
        @assert res.diagnostics.max_tan_residual < 1e-12
        @assert res.diagnostics.max_bond_error < 1e-12
    end
    r1 = bands[1] / bands[2]
    r2 = bands[2] / bands[3]
    @assert 3.0 < r1 < 5.5 && 3.0 < r2 < 5.5
    println("  water NVE dt-halving bands (0.5 ps): ", round.(bands, sigdigits = 4),
            " kJ/mol (ratios ", round(r1, digits = 2), ", ", round(r2, digits = 2), ")")

    # (b) Explicit potential-trend comparison: post-relax (no NVT) vs post-equilibrated NVE (4 ps).
    WB.rethermalize!(tr; temperature = 300.0)
    U0_unequil = WB.potential(field, tr.st.q)
    res_unequil = WB.run_nve!(tr, 4000; dt = 0.001, sample_every = 50)
    res_equil = WB.run_nve!(tr_eq, 4000; dt = 0.001, sample_every = 50)

    # Post-relax NVE: large initial equipartition jump in U (U rises ~200 kJ/mol from 0 K minimum)
    # followed by annealing, yet E = U + K stays conserved!
    dU_jump = res_unequil.samples[min(10, end)].U - U0_unequil
    @assert abs(dU_jump) > 50.0
    @assert abs(res_unequil.diagnostics.potential_slope + res_unequil.diagnostics.kinetic_slope) <
            max(0.2, 3 * res_unequil.diagnostics.energy_slope_se + 0.1)
    @assert res_unequil.diagnostics.energy_band < 1.5

    # Post-equilibration NVE: stationary U, K, and E (slopes within 3 SE of 0, tiny band).
    @assert abs(res_equil.diagnostics.energy_slope) < 3 * res_equil.diagnostics.energy_slope_se + 0.05
    @assert abs(res_equil.diagnostics.potential_slope) < 3 * res_equil.diagnostics.potential_slope_se + 1.0
    @assert res_equil.diagnostics.energy_band < 0.5
    println("  water NVE trend (4 ps, dt=1 fs):")
    println("    post-relax (no NVT): ΔU_0.5ps=", round(dU_jump, digits = 1),
            " kJ/mol, E_band=", round(res_unequil.diagnostics.energy_band, digits = 3),
            " kJ/mol, slopeE=", round(res_unequil.diagnostics.energy_slope, digits = 4),
            " ± ", round(res_unequil.diagnostics.energy_slope_se, digits = 4))
    println("    post-equil (3 ps NVT): E_band=", round(res_equil.diagnostics.energy_band, digits = 3),
            " kJ/mol, slopeU=", round(res_equil.diagnostics.potential_slope, digits = 2),
            " ± ", round(res_equil.diagnostics.potential_slope_se, digits = 2),
            ", slopeE=", round(res_equil.diagnostics.energy_slope, digits = 4),
            " ± ", round(res_equil.diagnostics.energy_slope_se, digits = 4))

    # (c) Momentum audit: zero-force box conserves P and L_lab to ~1e-13; live PME breaks P invariance.
    w0 = zero_force_water(w)
    f0 = WB.water_field(w0)
    tr0 = WB.WaterTrajectory(w0, f0; temperature = 300.0, seed = 6)
    res0 = WB.run_nve!(tr0, 500; dt = 0.001, sample_every = 25)
    @assert res0.diagnostics.max_momentum_drift < 1e-11
    @assert res0.diagnostics.max_angmom_drift < 1e-12
    @assert res0.diagnostics.max_pos_residual < 1e-12
    @assert res0.diagnostics.max_tan_residual < 1e-12
    @assert res_equil.diagnostics.max_momentum_drift > 1e-3  # PME grid breaks continuous translation invariance
    println("  momentum audit: zero-force max|ΔP|=", res0.diagnostics.max_momentum_drift,
            ", max|ΔL_lab|=", res0.diagnostics.max_angmom_drift,
            "; live PME max|ΔP|=", round(res_equil.diagnostics.max_momentum_drift, digits = 4))
end

"Exact deterministic checkpoint/state replay for WaterTrajectory (NVE and stochastic BAOAB)."
function test_water_checkpoint_replay()
    w = WB.tip3p_water(8; seed = 5)
    field = WB.water_field(w)
    tr = WB.WaterTrajectory(w, field; temperature = 300.0, seed = 17)
    WB.integrate!(tr, 10; dt = 0.002, temperature = 300.0, gamma = 1.0, sample_every = 5)

    # Checkpoint and run 20 more BAOAB steps.
    cp = D.checkpoint(tr)
    smp_cont = WB.integrate!(tr, 20; dt = 0.002, temperature = 300.0, gamma = 1.0, sample_every = 5)
    q_cont = copy(tr.st.q); p_cont = copy(tr.st.p); F_cont = copy(tr.st.F)
    step_cont = tr.step; time_cont = tr.time_ps

    # Perturb trajectory, restore checkpoint, and replay: must be bit-identical!
    WB.integrate!(tr, 5; dt = 0.002, temperature = 300.0, gamma = 1.0, sample_every = 5)
    D.restore!(tr, cp)
    smp_rep = WB.integrate!(tr, 20; dt = 0.002, temperature = 300.0, gamma = 1.0, sample_every = 5)
    @assert tr.st.q == q_cont && tr.st.p == p_cont && tr.st.F == F_cont
    @assert tr.step == step_cont && tr.time_ps == time_cont
    @assert smp_rep == smp_cont

    # Also verify NVE replay from the same checkpoint and water_residuals helper.
    cp_nve = D.checkpoint(tr)
    r1 = WB.run_nve!(tr, 25; dt = 0.001, sample_every = 5, mutate = true)
    q_nve = copy(tr.st.q); p_nve = copy(tr.st.p)
    D.restore!(tr, cp_nve)
    r2 = WB.run_nve!(tr, 25; dt = 0.001, sample_every = 5, mutate = true)
    @assert tr.st.q == q_nve && tr.st.p == p_nve && r1.samples == r2.samples
    res = WB.water_residuals(tr)
    @assert res.pos_residual < 1e-12 && res.tan_residual < 1e-12 && res.max_bond_error < 1e-12
    println("  water checkpoint replay: bit-identical across stochastic BAOAB and NVE")
end

function main()
    println("test_water_box: rigid TIP3P water box")
    w27 = test_build_counts()
    w64 = test_rigid_geometry()
    test_no_overlap(w64)
    test_exclusions_present(w27)
    test_lj_exclusion_energy(w27)
    test_coulomb_exclusion_energy(w27)
    test_analysis_helpers(w27)
    test_production_uniform_translation()
    test_constrained_steps(w27)
    test_rigid_drift_audit()
    test_nve_reversibility()
    test_constraint_thermostat_gamma_independence()
    test_force_vs_finite_difference()
    test_pme_vs_ewald()
    test_pme_multicell_parity()
    test_kinetic_dof_and_temperature()
    test_thermostat_target(w27)
    test_energy_bounded()
    test_water_integrator_state_machine()
    test_water_nve_convergence_and_potential_trend()
    test_water_checkpoint_replay()
    println("test_water_box PASS")
end

main()
