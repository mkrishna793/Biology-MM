using FoundationCore
using FoundationCore.Types
using FoundationCore.Factors
using FoundationCore.Dynamics

const SL = FoundationCore.StreamedLJ
const NB = FoundationCore.Neighbors

function test_threaded_factor_parity()
    N = 256
    box = Types.OrthorhombicBox((16.0, 16.0, 16.0))
    q = Matrix{Float64}(undef, 3, N)
    for i in 1:N
        q[1, i] = mod(0.73 * i, box.L[1])
        q[2, i] = mod(1.17 * i, box.L[2])
        q[3, i] = mod(1.91 * i, box.L[3])
    end
    epsilon = 0.7 .+ 0.2 .* sin.(1:N)
    sigma = 0.9 .+ 0.1 .* cos.(1:N)
    rc = 2.3
    ctx = Factors.Context(box, Types.REDUCED, N)
    serial = SL.StreamedLJFactor(q, epsilon, sigma, box; rcut=rc,
                                 cache=false, threaded=false)
    threaded = SL.StreamedLJFactor(q, epsilon, sigma, box; rcut=rc,
                                   cache=true, cache_skin=0.3, threaded=true)
    Fs = zeros(3, N); Ft = zeros(3, N)
    Us = Factors.evaluate!(Fs, serial, q, ctx)
    Ut = Factors.evaluate!(Ft, threaded, q, ctx)
    scaleU = max(abs(Us), 1.0)
    scaleF = max(maximum(abs, Fs), 1.0)
    du = abs(Ut - Us) / scaleU
    df = maximum(abs, Ft .- Fs) / scaleF
    @assert du < 2e-12 "threaded energy relative error $du"
    @assert df < 2e-11 "threaded force relative error $df"
    Ft2 = zeros(3, N)
    Ut2 = Factors.evaluate!(Ft2, threaded, q, ctx)
    @assert Ut2 == Ut && Ft2 == Ft "fixed-thread-count threaded path must repeat"
    @assert threaded.pair_cache.n_rebuilds == 1
    @assert threaded.pair_cache.n_reuses == 2
    println("  threads=$(Threads.nthreads()) N=$N pairs=$(FoundationCore.Neighbors.num_pairs(threaded.pair_cache)) rel_energy=$du rel_force=$df rebuilds=$(threaded.pair_cache.n_rebuilds) reuses=$(threaded.pair_cache.n_reuses)")
end

function test_cached_dynamics_trajectory_parity()
    N = 64
    box = Types.OrthorhombicBox((10.0, 10.0, 10.0))
    q0 = Matrix{Float64}(undef, 3, N)
    for k in 0:N-1
        q0[:, k+1] .= ((k % 4 + 0.5) * 2.5,
                       ((k ÷ 4) % 4 + 0.5) * 2.5,
                       (k ÷ 16 + 0.5) * 2.5)
    end
    q0 .+= 0.01 .* reshape(sin.(1:3N), 3, N)
    p0 = 0.002 .* reshape(cos.(1:3N), 3, N)
    eps = fill(0.2, N); sig = fill(0.8, N)
    rc = 3.0
    cached_factor = SL.StreamedLJFactor(q0, eps, sig, box; rcut=rc,
                                        cache=true, cache_skin=0.4)
    sorted_factor = SL.StreamedLJFactor(q0, eps, sig, box; rcut=rc,
                                        cache=true, cache_skin=0.4)
    every_factor = SL.StreamedLJFactor(q0, eps, sig, box; rcut=rc,
                                       cache=false)
    sys_cached = HarmonicSystem([cached_factor], ones(N),
                                Factors.Context(box, Types.REDUCED, N))
    sys_every = HarmonicSystem([every_factor], ones(N),
                               Factors.Context(box, Types.REDUCED, N))
    st_cached = State(q0, p0)
    st_every = State(q0, p0)
    dt = 1e-4
    initial_F = zeros(3, N)
    initial_U = Dynamics.forces!(sys_cached, st_cached.q, initial_F)
    @assert isfinite(initial_U) && any(!iszero, initial_F)

    # Compare every cached force against an independent cache rebuilt at the
    # same geometry. Both PairCaches enumerate the same sorted pair order, so
    # this is a bitwise force/energy check rather than a tolerance comparison.
    exact_state = State(q0, p0)
    Fcached = zeros(3, N)
    Fsorted = zeros(3, N)
    for _ in 1:20
        NB.fill_pair_cache!(sorted_factor.pair_cache, exact_state.q, box)
        fill!(Fcached, 0.0)
        fill!(Fsorted, 0.0)
        Ucached = Factors.evaluate!(Fcached, cached_factor, exact_state.q,
                                    sys_cached.ctx)
        Usorted = Factors.evaluate!(Fsorted, sorted_factor, exact_state.q,
                                    sys_cached.ctx)
        @assert (Ucached == Usorted && Fcached == Fsorted) "cached force differs from freshly rebuilt sorted cache"
        velocity_verlet_step!(exact_state, sys_cached, dt)
    end
    @assert sorted_factor.pair_cache.n_rebuilds == 21

    # Warm both trajectories before measuring steady-state calls.
    velocity_verlet_step!(st_cached, sys_cached, dt)
    velocity_verlet_step!(st_every, sys_every, dt)
    st_cached = State(q0, p0)
    st_every = State(q0, p0)
    cached_time = @elapsed for _ in 1:20
        velocity_verlet_step!(st_cached, sys_cached, dt)
    end
    every_time = @elapsed for _ in 1:20
        velocity_verlet_step!(st_every, sys_every, dt)
    end
    cached_step_alloc = @allocated velocity_verlet_step!(st_cached, sys_cached, dt)
    every_step_alloc = @allocated velocity_verlet_step!(st_every, sys_every, dt)
    for _ in 1:20
        velocity_verlet_step!(st_cached, sys_cached, dt)
        velocity_verlet_step!(st_every, sys_every, dt)
    end
    qerr = maximum(abs, st_cached.q .- st_every.q)
    perr = maximum(abs, st_cached.p .- st_every.p)
    ferr = maximum(abs, st_cached.F .- st_every.F)
    @assert max(qerr, perr, ferr) < 1e-11 "cache changed Dynamics trajectory: $((qerr, perr, ferr))"
    @assert cached_factor.pair_cache.n_rebuilds >= 1
    @assert cached_factor.pair_cache.n_reuses >= 1
    println("  Dynamics N=$N steps=20 cached rebuilds=$(cached_factor.pair_cache.n_rebuilds) reuses=$(cached_factor.pair_cache.n_reuses) " *
            "cached_20steps=$(cached_time)s every_20steps=$(every_time)s " *
            "cached_per_step=$(cached_step_alloc)B every_per_step=$(every_step_alloc)B " *
            "trajectory_max_diff=$((qerr, perr, ferr))")
end

test_threaded_factor_parity()
test_cached_dynamics_trajectory_parity()
println("ALL THREADING TESTS PASSED")
