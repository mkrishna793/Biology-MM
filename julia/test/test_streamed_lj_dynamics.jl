# End-to-end streamed Lennard-Jones dynamics through the factor protocol.
#
# Run directly:
#
#     julia --project=julia julia/test/test_streamed_lj_dynamics.jl
#
# Coverage, in order:
#
# * a small periodic 3x3x3 LJ system integrated with Dynamics.velocity_verlet_step!
#   for 160 steps, with periodic wrapping and forced out-of-cell coordinates at
#   several checkpoints (the streamed kernel rebuilds its cell list on every
#   call, so every step exercises a rebuild);
# * energy-drift verdict from Dynamics.energy_drift_report (bounded, not a leak);
# * time reversibility (negate momenta, integrate back);
# * every checkpoint is compared against an independent O(N^2) LFS pair sum that
#   uses a plain dictionary topology, for both energy and forces, including the
#   excluded/1-4-scaled pair set;
# * additive composition with a bonded factor: total U/F equal the sum of the
#   contributions, so the streamed factor neither loses nor double counts;
# * finite-difference forces and eps/sigma parameter gradients against the
#   dedicated evaluator, which uses the *same* kernel, cutoff, and topology;
# * cutoff-crossing continuity for :lfs;
# * constructor/state validation and structural refusal of :shift integration.
#
# The suite uses only `@assert`, matching the rest of FoundationCore. It is also
# listed in run_all.jl, which executes every suite inside a fresh isolated module
# and calls `main()` here at include time.

using FoundationCore
using FoundationCore.Types
using FoundationCore.Factors
using Random, Printf

const T = FoundationCore.Types
const FA = FoundationCore.Factors
const DY = FoundationCore.Dynamics
const SL = FoundationCore.StreamedLJ
const PG = FoundationCore.ParameterGradients

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

function _caught(f)
    try
        f()
    catch e
        return e
    end
    return nothing
end

function expect_argument_error(f::Function, label::String)
    e = _caught(f)
    @assert e isa ArgumentError "$label should throw ArgumentError, got $(typeof(e))"
    return e
end

function check_array(a::AbstractArray, b::AbstractArray, tol::Float64, label::String)
    err = maximum(abs, a .- b)
    scale = max(maximum(abs, b), 1.0)
    @assert err <= tol * scale "$label: max|Δ|=$err exceeds $(tol)×scale=$scale"
    return err
end

"Wrap every coordinate into [0, L) in place (a display operation)."
function wrap_positions!(q::Matrix{Float64}, L::NTuple{3,Float64})
    @inbounds for j in axes(q, 2), d in 1:3
        q[d, j] -= L[d] * floor(q[d, j] / L[d])
    end
    return q
end

"""
Independent O(N^2) linear-force-shifted Lennard-Jones reference.

`weights` is a plain `Dict{Tuple{Int,Int},Float64}` on canonical pairs; a missing
pair means weight 1, and weight 0 is an exclusion. This deliberately shares no
topology or kernel code with StreamedLJ.
"""
function ref_lj(q::AbstractMatrix{Float64}, epsilon::Vector{Float64},
                sigma::Vector{Float64}, box::T.OrthorhombicBox, rc::Float64;
                weights = nothing)
    N = size(q, 2)
    F = zeros(Float64, 3, N)
    U = 0.0
    npairs = 0
    rc2 = rc * rc
    @inbounds for i in 1:N-1
        for j in i+1:N
            w = weights === nothing ? 1.0 : get(weights, (i, j), 1.0)
            w == 0.0 && continue
            d = T.unwrap_displacement(box, q, i, j)
            r2 = T.norm2(d)
            r2 > rc2 && continue
            r = sqrt(r2)
            eij = sqrt(epsilon[i] * epsilon[j])
            sij = 0.5 * (sigma[i] + sigma[j])
            s6 = (sij / r)^6
            s12 = s6 * s6
            s6c = (sij / rc)^6
            s12c = s6c * s6c
            uc = 4.0 * eij * (s12c - s6c)
            duc = 24.0 * eij * (s6c - 2.0 * s12c) / rc
            u = 4.0 * eij * (s12 - s6)
            dudr = 24.0 * eij * (s6 - 2.0 * s12) / r
            u = u - uc - duc * (r - rc)
            dudr = dudr - duc
            u *= w
            c = -w * dudr / r
            F[1, i] += c * d.x
            F[2, i] += c * d.y
            F[3, i] += c * d.z
            F[1, j] -= c * d.x
            F[2, j] -= c * d.y
            F[3, j] -= c * d.z
            U += u
            npairs += 1
        end
    end
    return U, F, npairs
end

# 3x3x3 simple-cubic lattice at spacing 2.0 in a 6x6x6 box, with a small seeded
# jitter so the cell traversal is not accidentally symmetric.
function small_system(; topology = true)
    L = 6.0
    box = T.OrthorhombicBox((L, L, L))
    rc = 2.2
    skin = 0.2
    N = 27
    q = zeros(Float64, 3, N)
    k = 0
    for z in 0:2, y in 0:2, x in 0:2
        k += 1
        q[1, k] = 2.0 * x
        q[2, k] = 2.0 * y
        q[3, k] = 2.0 * z
    end
    rng = MersenneTwister(20260412)
    q .+= 0.01 .* (rand(rng, 3, N) .- 0.5)
    epsilon = 0.8 .+ 0.2 .* rand(rng, N)
    sigma = 0.95 .+ 0.1 .* rand(rng, N)
    p = 0.15 .* (rand(rng, 3, N) .- 0.5)

    topo = nothing
    weights = nothing
    if topology
        pairs = Tuple{Int,Int}[]
        for i in 1:N-1, j in i+1:N
            push!(pairs, (i, j))
        end
        sort!(pairs; by = pr -> T.norm2(T.unwrap_displacement(box, q, pr[1], pr[2])))
        excluded = pairs[1:4]
        scaled = [(pairs[4 + s][1], pairs[4 + s][2], 0.5) for s in 1:3]
        topo = SL.PairTopology(excluded = excluded, scaled = scaled)
        weights = Dict{Tuple{Int,Int},Float64}()
        for (i, j) in excluded
            weights[(i, j)] = 0.0
        end
        for (i, j, w) in scaled
            weights[(i, j)] = w
        end
    end

    fac = SL.StreamedLJFactor(q, epsilon, sigma, box;
                              rcut = rc, skin = skin, mode = :lfs,
                              topology = topo)
    ctx = FA.Context(box, T.REDUCED, N)
    sys = DY.HarmonicSystem([fac], ones(Float64, N), ctx)
    st = DY.State(q, p)
    return (fac = fac, ctx = ctx, sys = sys, st = st, box = box, rc = rc,
            weights = weights, topo = topo, q0 = copy(q), p0 = copy(p))
end

function check_reference(fac, q, ctx, weights, rc, label::String)
    N = fac.N
    Ff = zeros(Float64, 3, N)
    Uf = FA.evaluate!(Ff, fac, q, ctx)
    Ur, Fr, npr = ref_lj(q, fac.epsilon, fac.sigma, fac.box, rc; weights = weights)
    eu = check_array([Uf], [Ur], 1e-11, "$label energy vs O(N^2)")
    ef = check_array(Ff, Fr, 1e-11, "$label forces vs O(N^2)")

    # The same call through the public workspace API must have the same energy,
    # forces, and pair count; the factor returns no pair count itself.
    Uw, Fw, npw = SL.evaluate!(fac.ws, q, fac.epsilon, fac.sigma, ctx.box;
                                rcut = fac.rcut, skin = fac.skin, mode = fac.mode,
                                topology = fac.topology)
    @assert Uw == Uf "factor and workspace energy disagree"
    @assert Fw == Ff "factor and workspace forces disagree"
    @assert npw == npr "pair count $npw != independent $npr"
    return eu, ef, npw
end

# ---------------------------------------------------------------------------
# trajectory: drift, wrapping, cell rebuilds, independent parity
# ---------------------------------------------------------------------------

function test_trajectory_drift_wrapping_and_parity()
    s = small_system()
    fac, ctx, sys, st, box = s.fac, s.ctx, s.sys, s.st, s.box
    @assert FA.is_local(fac) == false
    @assert FA.factor_support_size(fac) == fac.N
    @assert FA.read_support(fac) == Tuple(1:fac.N)
    @assert FA.force_consistent(fac) == true
    @assert DY.force_model_supported(sys)

    # The explicit PairTopology must actually change the potential; nothing is
    # inferred from a ChemicalTopology.
    fac_plain = SL.StreamedLJFactor(s.q0, fac.epsilon, fac.sigma, box;
                                    rcut = s.rc, skin = 0.2)
    @assert SL.streamed_energy(fac, s.q0, ctx) != SL.streamed_energy(fac_plain, s.q0, ctx)

    nsteps = 800
    dt = 2e-3
    times = Float64[]
    energies = Float64[]
    parity_pairs = Int[]

    for step in 1:nsteps
        DY.velocity_verlet_step!(st, sys, dt)

        if step % 50 == 0
            # Force coordinates out of the primary cell, then wrap them. Both
            # operations are exact periodic translations, so U must not move;
            # the streamed kernel has to rebuild the cell list at each call.
            E_before = DY.total_energy(sys, st.q, st.p)
            @inbounds for j in 1:fac.N
                if (j + step) % 3 == 0
                    st.q[1, j] += 2.0 * box.L[1]
                end
            end
            E_shift = DY.total_energy(sys, st.q, st.p)
            wrap_positions!(st.q, box.L)
            DY.invalidate!(st)
            E_after = DY.total_energy(sys, st.q, st.p)
            scale = max(abs(E_before), 1.0)
            @assert abs(E_shift - E_before) <= 1e-12 * scale "out-of-cell translation changed energy"
            @assert abs(E_after - E_before) <= 1e-12 * scale "wrapping changed energy"
            @inbounds for j in 1:fac.N, d in 1:3
                @assert 0.0 <= st.q[d, j] < box.L[d] "wrapped coordinate outside the box"
            end
        end

        if step % 100 == 0
            eu, ef, np = check_reference(fac, st.q, ctx, s.weights, s.rc,
                                         "trajectory step $step")
            push!(parity_pairs, np)
            @assert np > 0 "the trajectory fixture must have interacting pairs"
            println("    step $step: pairs=$np  energy_err=$eu  force_err=$ef")
        end

        push!(times, step * dt)
        push!(energies, DY.total_energy(sys, st.q, st.p))
    end

    @assert length(energies) == nsteps
    @assert all(isfinite, energies)
    rep = DY.energy_drift_report(times, energies)
    @printf("  trajectory: N=%d steps=%d dt=%.4g  E0=%.12f  max|ΔE|=%.3e  drift_frac=%.4f\n",
            fac.N, nsteps, dt, rep.E0, rep.max_abs_dev, rep.drift_fraction)
    @assert rep.bounded "energy drift fraction $(rep.drift_fraction) is not bounded"
    @assert rep.max_abs_dev < 1e-4 "energy excursion $(rep.max_abs_dev) is implausibly large"
    return true
end

function test_time_reversibility()
    s = small_system()
    sys, st = s.sys, s.st
    dt = 1e-3
    nsteps = 150
    q0 = copy(st.q)
    p0 = copy(st.p)

    for _ in 1:nsteps
        DY.velocity_verlet_step!(st, sys, dt)
    end
    st.p .= -st.p
    for _ in 1:nsteps
        DY.velocity_verlet_step!(st, sys, dt)
    end
    st.p .= -st.p

    relq = maximum(abs, st.q .- q0) / max(maximum(abs, q0), 1.0)
    relp = maximum(abs, st.p .- p0) / max(maximum(abs, p0), 1.0)
    @printf("  reversibility over %d+%d steps: rel|Δq|=%.3e  rel|Δp|=%.3e\n",
            nsteps, nsteps, relq, relp)
    @assert relq < 1e-8 && relp < 1e-8 "velocity Verlet + StreamedLJFactor is not reversible"
    return true
end

# ---------------------------------------------------------------------------
# additive composition with a bonded factor
# ---------------------------------------------------------------------------

function test_additive_bonded_composition()
    s = small_system()
    fac, ctx = s.fac, s.ctx
    bond = FA.HarmonicBond(1, 2, 2.0, 1.5)
    N = fac.N
    sys = DY.HarmonicSystem([fac, bond], ones(Float64, N), ctx)
    @assert DY.force_model_supported(sys)

    U_total, F_total = FA.total_energy_forces(sys.factors, s.st.q, ctx)

    F_lj = zeros(Float64, 3, N)
    U_lj = FA.evaluate!(F_lj, fac, s.st.q, ctx)
    F_b = zeros(Float64, 3, N)
    U_b = FA.evaluate!(F_b, bond, s.st.q, ctx)

    @assert abs(U_b) > 1e-6 "the bonded factor must make a nonzero contribution"
    @assert maximum(abs, F_b) > 1e-8 "the bonded factor must scatter a force"
    uerr = check_array([U_total], [U_lj + U_b], 1e-12,
                       "additive energy (LJ + bond)")
    ferr = check_array(F_total, F_lj .+ F_b, 1e-12,
                       "additive force (LJ + bond)")
    @assert U_total != U_lj "total energy is just the LJ energy: the bond was dropped"
    @printf("  additive composition: U_LJ=%.9f U_bond=%.9f  energy_err=%.3e force_err=%.3e\n",
            U_lj, U_b, uerr, ferr)

    # Repeated factor evaluation is deterministic and does not accumulate into
    # the previous result; this is the no-double-counting check.
    F_again = zeros(Float64, 3, N)
    U_again = FA.evaluate!(F_again, fac, s.st.q, ctx)
    @assert U_again == U_lj && F_again == F_lj

    # additivity is literal: a prefilled accumulator keeps its prior content.
    sentinel = reshape(collect(1.0:(3 * N)), 3, N)
    F_prefilled = copy(sentinel)
    U_prefilled = FA.evaluate!(F_prefilled, fac, s.st.q, ctx)
    @assert U_prefilled == U_lj
    check_array(F_prefilled, sentinel .+ F_lj, 1e-12,
                "force scatter onto a prefilled accumulator")
    return true
end

# ---------------------------------------------------------------------------
# finite differences: forces and parameter gradients
# ---------------------------------------------------------------------------

function test_finite_difference_forces_and_parameters()
    s = small_system()
    fac, ctx = s.fac, s.ctx
    N = fac.N
    q = s.st.q

    F = zeros(Float64, 3, N)
    U = FA.evaluate!(F, fac, q, ctx)
    h = 1e-6
    G = zeros(Float64, 3, N)
    qp = copy(q)
    qm = copy(q)
    for j in 1:N, d in 1:3
        qp[d, j] += h
        qm[d, j] -= h
        Up = SL.streamed_energy(fac, qp, ctx)
        Um = SL.streamed_energy(fac, qm, ctx)
        G[d, j] = (Up - Um) / (2h)
        qp[d, j] -= h
        qm[d, j] += h
    end
    ferr = maximum(abs, F .+ G) / max(maximum(abs, F), 1.0)
    @assert ferr < 1e-5 "finite-difference force error $ferr"
    println("  finite-difference forces: normalized_err=$ferr")

    U0, ge, gs = SL.streamed_parameter_gradients!(fac, q, ctx)
    @assert U0 == U "parameter-gradient evaluator changed the energy"
    @assert ge !== fac.ws.grad_epsilon && gs !== fac.ws.grad_sigma
    @assert length(ge) == N && length(gs) == N

    eps_fd = zeros(Float64, N)
    sig_fd = zeros(Float64, N)
    for k in 1:N
        ep = copy(fac.epsilon); ep[k] += h
        em = copy(fac.epsilon); em[k] -= h
        eps_fd[k] = (SL.streamed_energy(fac, q, ctx; epsilon = ep) -
                     SL.streamed_energy(fac, q, ctx; epsilon = em)) / (2h)

        sp = copy(fac.sigma); sp[k] += h
        sm = copy(fac.sigma); sm[k] -= h
        sig_fd[k] = (SL.streamed_energy(fac, q, ctx; sigma = sp) -
                     SL.streamed_energy(fac, q, ctx; sigma = sm)) / (2h)
    end
    eerr = check_array(ge, eps_fd, 1e-5, "dU/depsilon")
    serr = check_array(gs, sig_fd, 1e-5, "dU/dsigma")
    @printf("  parameter gradients: eps_err=%.3e sigma_err=%.3e\n", eerr, serr)

    ge2 = zeros(Float64, N)
    gs2 = zeros(Float64, N)
    U2 = SL.streamed_parameter_gradients!(ge2, gs2, fac, q, ctx)
    @assert U2 == U0 && ge2 == ge && gs2 == gs

    # Convenience forwarding through the module-local evaluator names.
    F_fwd = zeros(Float64, 3, N)
    @assert SL.evaluate!(F_fwd, fac, q, ctx) == U
    @assert F_fwd == F
    U_fwd, ge_fwd, gs_fwd = SL.evaluate_with_gradients!(fac, q, ctx)
    @assert U_fwd == U0 && ge_fwd == ge && gs_fwd == gs

    # ParameterGradients only knows the five local factor types. It must refuse
    # this factor rather than silently dropping it from a joint gradient; the
    # dedicated evaluator above is the supported route.
    grads = Dict{Symbol,Float64}()
    Fscratch = zeros(Float64, 3, N)
    e = _caught(() -> PG.evaluate_parameter_gradients!(Fscratch, grads, [fac], q, ctx))
    @assert e isa ArgumentError "ParameterGradients must refuse StreamedLJFactor explicitly"
    return true
end

# ---------------------------------------------------------------------------
# cutoff continuity
# ---------------------------------------------------------------------------

function test_cutoff_crossing_continuity()
    L = 20.0
    box = T.OrthorhombicBox((L, L, L))
    rc = 2.5
    epsilon = [1.2, 0.8]
    sigma = [1.0, 1.1]
    q = zeros(Float64, 3, 2)
    fac = SL.StreamedLJFactor(q, epsilon, sigma, box; rcut = rc, mode = :lfs)
    ctx = FA.Context(box, T.REDUCED, 2)

    @assert FA.force_consistent(fac)
    @assert !FA.is_local(fac)

    deltas = (-1e-3, -1e-5, -1e-7, 0.0, 1e-7, 1e-5, 1e-3)
    Us = Float64[]
    Fmax = Float64[]
    for delta in deltas
        q[1, 2] = rc + delta
        F = zeros(Float64, 3, 2)
        push!(Us, FA.evaluate!(F, fac, q, ctx))
        push!(Fmax, maximum(abs, F))
    end
    @assert maximum(abs, Us) < 1e-6 "LFS energy jumps at the cutoff: $Us"
    @assert maximum(Fmax) < 1e-2 "LFS force jumps at the cutoff: $Fmax"
    @printf("  LFS cutoff crossing: max|U|=%.3e max|F|=%.3e over ±1e-3\n",
            maximum(abs, Us), maximum(Fmax))

    # The independent reference agrees on both sides of the cutoff.
    for delta in (-1e-3, 0.0, 1e-3)
        q[1, 2] = rc + delta
        F = zeros(Float64, 3, 2)
        U = FA.evaluate!(F, fac, q, ctx)
        Ur, Fr, npr = ref_lj(q, epsilon, sigma, box, rc)
        check_array([U], [Ur], 1e-12, "near-cutoff energy (δ=$delta)")
        check_array(F, Fr, 1e-12, "near-cutoff forces (δ=$delta)")
        @assert npr == (delta <= 0 ? 1 : 0)
    end

    # A :shift factor is allowed for energy evaluation but is structurally
    # refused by the integrator gate.
    fac_shift = SL.StreamedLJFactor(q, epsilon, sigma, box; rcut = rc, mode = :shift)
    @assert !FA.force_consistent(fac_shift)
    @assert isfinite(SL.streamed_energy(fac_shift, q, ctx))
    return true
end

# ---------------------------------------------------------------------------
# shift refusal and validation
# ---------------------------------------------------------------------------

function test_shift_is_refused_for_integration()
    s = small_system()
    q = s.q0
    N = s.fac.N
    shift_fac = SL.StreamedLJFactor(q, s.fac.epsilon, s.fac.sigma, s.box;
                                    rcut = s.rc, skin = 0.2, mode = :shift)
    ctx = s.ctx
    sys = DY.HarmonicSystem([shift_fac], ones(Float64, N), ctx)
    @assert !DY.force_model_supported(sys)

    st = DY.State(q, s.p0)
    q_before = copy(st.q)
    p_before = copy(st.p)
    err = _caught(() -> DY.velocity_verlet_step!(st, sys, 1e-3))
    @assert err isa ErrorException "a :shift StreamedLJFactor was integrated"
    msg = sprint(showerror, err)
    @assert occursin("non-force-consistent", msg) &&
            occursin("StreamedLJFactor", msg) "refusal is not actionable: $msg"
    @assert st.q == q_before && st.p == p_before "state moved before the refusal"
    @assert isfinite(DY.potential_energy(sys, st.q)) "energy evaluation must remain available"

    # The same construction in :lfs is accepted.
    lfs_fac = SL.StreamedLJFactor(q, s.fac.epsilon, s.fac.sigma, s.box;
                                  rcut = s.rc, skin = 0.2, mode = :lfs)
    @assert DY.force_model_supported(DY.HarmonicSystem([lfs_fac], ones(Float64, N), ctx))
    println("  :shift refused structurally; :lfs accepted")
    return true
end

function test_validation_and_aliasing()
    L = 8.0
    box = T.OrthorhombicBox((L, L, L))
    box2 = T.OrthorhombicBox((L + 0.5, L, L))
    rc = 2.0
    rng = MersenneTwister(77)
    N = 8
    q = rand(rng, 3, N) .* L
    epsilon = 0.8 .+ 0.1 .* rand(rng, N)
    sigma = 0.9 .+ 0.1 .* rand(rng, N)

    # Constructor validation.
    expect_argument_error(() -> SL.StreamedLJFactor(q, epsilon[1:N-1], sigma, box;
                                                    rcut = rc),
                          "short epsilon")
    expect_argument_error(() -> SL.StreamedLJFactor(q, epsilon, sigma[1:N-1], box;
                                                    rcut = rc),
                          "short sigma")
    bad_eps = copy(epsilon); bad_eps[1] = -1.0
    expect_argument_error(() -> SL.StreamedLJFactor(q, bad_eps, sigma, box; rcut = rc),
                          "negative epsilon")
    bad_sig = copy(sigma); bad_sig[1] = 0.0
    expect_argument_error(() -> SL.StreamedLJFactor(q, epsilon, bad_sig, box; rcut = rc),
                          "zero sigma")
    bad_q = copy(q); bad_q[2, 3] = NaN
    expect_argument_error(() -> SL.StreamedLJFactor(bad_q, epsilon, sigma, box; rcut = rc),
                          "non-finite coordinate")
    for mode in (:none, :bogus, "lfs")
        expect_argument_error(() -> SL.StreamedLJFactor(q, epsilon, sigma, box;
                                                        rcut = rc, mode = mode),
                              "unsupported mode $(repr(mode))")
    end
    expect_argument_error(() -> SL.StreamedLJFactor(q, epsilon, sigma, T.NoBox();
                                                    rcut = rc),
                          "NoBox")
    expect_argument_error(() -> SL.StreamedLJFactor(q, epsilon, sigma, box;
                                                    rcut = 0.5 * L),
                          "rcut at min(L)/2")
    expect_argument_error(() -> SL.StreamedLJFactor(q, epsilon, sigma, box;
                                                    rcut = rc, skin = -1.0),
                          "negative skin")
    expect_argument_error(() -> SL.StreamedLJFactor(q, epsilon, sigma, box;
                                                    rcut = rc,
                                                    topology = Dict((1, 2) => 0.5)),
                          "non-PairTopology topology")
    expect_argument_error(() -> SL.StreamedLJFactor(q, epsilon, sigma, box;
                                                    rcut = rc,
                                                    topology = T.ChemicalTopology()),
                          "ChemicalTopology is not inferred")
    expect_argument_error(() -> SL.StreamedLJFactor(
                              q, epsilon, sigma, box; rcut = rc,
                              topology = SL.PairTopology(excluded = [(1, N + 1)])),
                          "topology atom index beyond N")

    fac = SL.StreamedLJFactor(q, epsilon, sigma, box; rcut = rc)
    ctx = FA.Context(box, T.REDUCED, N)

    # Atom-count constructor allocates the same workspace shape and is usable
    # before a configuration exists.
    fac_n = SL.StreamedLJFactor(N, epsilon, sigma, box; rcut = rc)
    @assert fac_n.N == N && size(fac_n.ws.forces) == (3, N)
    @assert isfinite(FA.evaluate!(zeros(Float64, 3, N), fac_n, q, ctx))

    # Context/state validation.
    bad_ctx = FA.Context(box, T.REDUCED, N + 1)
    expect_argument_error(() -> FA.evaluate!(zeros(Float64, 3, N + 1), fac, q, bad_ctx),
                          "context N mismatch")
    ctx_wrong_box = FA.Context(box2, T.REDUCED, N)
    expect_argument_error(() -> FA.evaluate!(zeros(Float64, 3, N), fac, q, ctx_wrong_box),
                          "context box mismatch")
    ctx_nobox = FA.Context(T.NoBox(), T.REDUCED, N)
    expect_argument_error(() -> FA.evaluate!(zeros(Float64, 3, N), fac, q, ctx_nobox),
                          "non-periodic context box")
    expect_argument_error(() -> FA.evaluate!(zeros(Float64, 3, N - 1), fac, q, ctx),
                          "wrong force matrix shape")
    q_short = copy(q)[:, 1:N-1]
    expect_argument_error(() -> FA.evaluate!(zeros(Float64, 3, N), fac, q_short, ctx),
                          "coordinate count mismatch")

    # The constructor copies epsilon/sigma; mutating the caller's vectors later
    # cannot silently change an existing factor's physics.
    eps_alias = copy(epsilon)
    sig_alias = copy(sigma)
    fac_alias = SL.StreamedLJFactor(q, eps_alias, sig_alias, box; rcut = rc)
    eps_alias[1] = 123.0
    sig_alias[1] = 456.0
    @assert fac_alias.epsilon[1] != 123.0 && fac_alias.sigma[1] != 456.0
    @assert occursin("StreamedLJFactor", sprint(show, fac_alias))

    # The force accumulator must not alias the factor's own workspace.
    before = copy(fac.ws.forces)
    expect_argument_error(() -> FA.evaluate!(fac.ws.forces, fac, q, ctx),
                          "force accumulator aliasing the workspace")
    expect_argument_error(() -> FA.evaluate!(view(fac.ws.forces, :, :), fac, q, ctx),
                          "force accumulator aliasing the workspace through a view")
    expect_argument_error(() -> FA.evaluate!(zeros(Float64, 3, N), fac,
                                             fac.ws.forces, ctx),
                          "q aliasing the workspace")
    expect_argument_error(() -> SL.streamed_energy(fac, fac.ws.forces, ctx),
                          "q aliasing the workspace in streamed_energy")
    q_as_F = copy(q)
    expect_argument_error(() -> FA.evaluate!(q_as_F, fac, q_as_F, ctx),
                          "force accumulator aliasing q")
    @assert fac.ws.forces == before "aliasing rejection mutated the workspace"

    # Mutated factor state is revalidated on every call.
    fac_bad_mode = SL.StreamedLJFactor(q, epsilon, sigma, box; rcut = rc)
    fac_bad_mode.mode = :bogus
    expect_argument_error(() -> FA.evaluate!(zeros(Float64, 3, N), fac_bad_mode, q, ctx),
                          "mutated unsupported mode")
    fac_bad_topo = SL.StreamedLJFactor(q, epsilon, sigma, box; rcut = rc)
    fac_bad_topo.topology = SL.PairTopology(excluded = [(1, N + 1)])
    expect_argument_error(() -> FA.evaluate!(zeros(Float64, 3, N), fac_bad_topo, q, ctx),
                          "mutated topology beyond N")
    fac_bad_len = SL.StreamedLJFactor(q, epsilon, sigma, box; rcut = rc)
    push!(fac_bad_len.epsilon, 1.0)
    expect_argument_error(() -> FA.evaluate!(zeros(Float64, 3, N), fac_bad_len, q, ctx),
                          "mutated epsilon length")
    fac_bad_rc = SL.StreamedLJFactor(q, epsilon, sigma, box; rcut = rc)
    fac_bad_rc.rcut = 0.5 * L
    expect_argument_error(() -> FA.evaluate!(zeros(Float64, 3, N), fac_bad_rc, q, ctx),
                          "mutated invalid cutoff")

    println("  constructor/state/box/topology validation and aliasing refusal verified")
    return true
end

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

function main()
    println("=" ^ 72)
    println("StreamedLJ factor / Dynamics tests")
    println("=" ^ 72)
    for (name, fn) in [
        ("trajectory_drift_wrapping_parity", test_trajectory_drift_wrapping_and_parity),
        ("time_reversibility", test_time_reversibility),
        ("additive_bonded_composition", test_additive_bonded_composition),
        ("finite_difference_forces_and_parameters",
         test_finite_difference_forces_and_parameters),
        ("cutoff_crossing_continuity", test_cutoff_crossing_continuity),
        ("shift_refused_for_integration", test_shift_is_refused_for_integration),
        ("validation_and_aliasing", test_validation_and_aliasing),
    ]
        println("TEST ", name)
        fn()
    end
    println("")
    println("STREAMED LJ DYNAMICS TESTS PASSED")
    return true
end

main()
