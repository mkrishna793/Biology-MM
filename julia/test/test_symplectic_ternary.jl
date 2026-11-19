using Test
using FoundationCore
if !isdefined(FoundationCore, :SymplecticTernary)
    @eval FoundationCore begin
        include(joinpath(@__DIR__, "..", "src", "SymplecticTernary.jl"))
    end
end
const SYMT = FoundationCore.SymplecticTernary

using FoundationCore.Types
using FoundationCore.Factors
using FoundationCore.Dynamics
using LinearAlgebra
using Statistics
using Random

@testset "Symplectic Ternary Time-Reversible Integrator & Adjoint Engine" begin

    # =========================================================================
    # Test 1: Symmetric Rounding Property
    # =========================================================================
    @testset "1. Symmetric Rounding on Balanced Ternary Lattice" begin
        # Zero
        @test SYMT.round_symmetric(0.0) == 0
        @test SYMT.round_symmetric(-0.0) == 0

        # Exact half-integer ties: must round away from zero with strict sign symmetry
        @test SYMT.round_symmetric(0.5) == 1
        @test SYMT.round_symmetric(-0.5) == -1
        @test SYMT.round_symmetric(1.5) == 2
        @test SYMT.round_symmetric(-1.5) == -2
        @test SYMT.round_symmetric(2.5) == 3
        @test SYMT.round_symmetric(-2.5) == -3
        @test SYMT.round_symmetric(10.5) == 11
        @test SYMT.round_symmetric(-10.5) == -11

        # Comprehensive lattice scan: round_symmetric(-x) == -round_symmetric(x)
        for x in -50.0:0.25:50.0
            @test SYMT.round_symmetric(-x) == -SYMT.round_symmetric(x)
        end

        # Random float fuzzing across wide dynamic range
        rng = MersenneTwister(12345)
        for _ in 1:1000
            val = (rand(rng) - 0.5) * 1e8
            @test SYMT.round_symmetric(-val) == -SYMT.round_symmetric(val)
        end

        # Int128 type preservation
        @test SYMT.round_symmetric(3.14159) isa Int128
        @test SYMT.round_symmetric(100) isa Int128
    end

    # =========================================================================
    # Test 2: State Construction, Round-Trip & Invariants
    # =========================================================================
    @testset "2. State Construction, Invariants & Conversions" begin
        box = OrthorhombicBox((2.5, 2.5, 2.5))
        q = [0.1 0.5; 0.2 0.6; 0.3 0.7]
        p = [0.01 -0.01; 0.02 -0.02; 0.03 -0.03]
        m = [12.0, 14.0]
        z = [-0.2, 0.2]
        names = [:C1, :N1]

        sys = Types.System(Types.AtomSet(q, p, m, z, names), box)
        st = SYMT.SymplecticTernaryState(sys)

        @test size(st.Q) == (3, 2)
        @test size(st.P) == (3, 2)
        @test st.masses == m
        @test st.box == box
        @test st.step_index == 0

        # Round trip to SystemState
        sys_recovered = SYMT.to_system_state(st)
        @test isapprox(sys_recovered.atoms.q, q; atol = 1e-11)
        @test isapprox(sys_recovered.atoms.p, p; atol = 1e-10)
        @test sys_recovered.atoms.m == m
        @test sys_recovered.atoms.z == z
        @test sys_recovered.atoms.names == names

        # Conversion to Dynamics.State
        st_dyn = SYMT.to_dynamics_state(st)
        @test isapprox(st_dyn.q, q; atol = 1e-11)
        @test isapprox(st_dyn.p, p; atol = 1e-10)

        # Invariant checks: shape mismatches and unphysical arguments
        @test_throws ArgumentError SYMT.SymplecticTernaryState(zeros(Int128, 3, 2), zeros(Int128, 3, 3), 3.0^26, 3.0^24, m)
        @test_throws ArgumentError SYMT.SymplecticTernaryState(zeros(Int128, 2, 2), zeros(Int128, 2, 2), 3.0^26, 3.0^24, m)
        @test_throws ArgumentError SYMT.SymplecticTernaryState(zeros(Int128, 3, 2), zeros(Int128, 3, 2), -1.0, 3.0^24, m)
        @test_throws ArgumentError SYMT.SymplecticTernaryState(zeros(Int128, 3, 2), zeros(Int128, 3, 2), 3.0^26, 3.0^24, [-1.0, 1.0])
    end

    # =========================================================================
    # Test 3: 1,000-Step 3D Nonlinear MD Exact Bit-Reversibility vs Float64 Verlet
    # =========================================================================
    @testset "3. 1,000-Step 3D Nonlinear MD Exact Bit-Reversibility" begin
        # 12-atom 3D periodic coupled system with HarmonicBond + HarmonicAngle + LennardJones
        box = OrthorhombicBox((4.0, 4.0, 4.0))
        rng = MersenneTwister(999)

        N = 12
        q0 = zeros(3, N)
        for mol in 1:4
            bx = 0.8 * mol
            by = 1.0 + 0.2 * (mol % 2)
            bz = 1.0 + 0.2 * (mol % 3)
            q0[:, 3*(mol-1) + 1] = [bx, by, bz]
            q0[:, 3*(mol-1) + 2] = [bx + 0.15, by, bz]
            q0[:, 3*(mol-1) + 3] = [bx + 0.15 + 0.15*cos(1.9), by + 0.15*sin(1.9), bz]
        end

        p0 = randn(rng, 3, N) .* 0.04
        m = fill(12.0, N)

        factors = AbstractFactor[]
        # Bonded potentials
        for mol in 1:4
            i1 = 3*(mol-1) + 1
            i2 = 3*(mol-1) + 2
            i3 = 3*(mol-1) + 3
            push!(factors, HarmonicBond(i1, i2, 2500.0, 0.15))
            push!(factors, HarmonicBond(i2, i3, 2500.0, 0.15))
            push!(factors, HarmonicAngle(i1, i2, i3, 600.0, 1.9))
        end

        # Nonbonded Lennard-Jones between molecules (lfs C^1 conservative cutoff)
        for mol1 in 1:4, mol2 in (mol1+1):4
            for a1 in 1:3, a2 in 1:3
                i = 3*(mol1-1) + a1
                j = 3*(mol2-1) + a2
                push!(factors, LennardJonesPair(i, j, 0.8, 0.28; mode = :lfs, rc = 1.5))
            end
        end

        sys = Types.System(Types.AtomSet(copy(q0), copy(p0), m, zeros(N), fill(:C, N)), box)
        st0 = SYMT.SymplecticTernaryState(sys)
        st = copy(st0)

        dt = 0.001 # 1 fs time step

        # Run 1,000 steps forward
        SYMT.simulate_symplectic_ternary!(st, factors, dt, 1000; direction = +1)
        @test st.step_index == 1000
        @test st.Q != st0.Q
        @test st.P != st0.P

        # Run 1,000 steps backward
        SYMT.simulate_symplectic_ternary!(st, factors, dt, 1000; direction = -1)
        @test st.step_index == 0

        # Assert EXACT bit/trit match: ZERO error in all 36 coordinates and 36 momenta!
        @test all(st.Q .== st0.Q)
        @test all(st.P .== st0.P)
        @test maximum(abs.(st.Q .- st0.Q)) == 0
        @test maximum(abs.(st.P .- st0.P)) == 0

        # Test time-step sign equivalence: dt < 0 inverts direction
        st_test_sign = copy(st0)
        SYMT.symplectic_ternary_step!(st_test_sign, factors, dt; direction = +1)
        @test st_test_sign.Q != st0.Q
        SYMT.symplectic_ternary_step!(st_test_sign, factors, -dt; direction = +1)
        @test st_test_sign.Q == st0.Q
        @test st_test_sign.P == st0.P

        # Compare with standard IEEE-754 Float64 Velocity Verlet on bit-identical problem
        hs = Dynamics.HarmonicSystem(sys, factors)
        st_vv = Dynamics.State(copy(q0), copy(p0))
        for _ in 1:1000
            Dynamics.velocity_verlet_step!(st_vv, hs, dt)
        end
        for _ in 1:1000
            Dynamics.velocity_verlet_step!(st_vv, hs, -dt)
        end
        err_q = norm(st_vv.q - q0)
        err_p = norm(st_vv.p - p0)

        # Standard Float64 Velocity Verlet fails exact reversibility (error > 0)
        @test (err_q + err_p) > 0.0
        # While SymplecticTernary achieves exact 0
        ternary_err_q = norm(Float64.(st.Q .- st0.Q))
        ternary_err_p = norm(Float64.(st.P .- st0.P))
        @test ternary_err_q == 0.0
        @test ternary_err_p == 0.0
    end

    # =========================================================================
    # Test 4: 10,000-Step Zero-Secular-Drift Energy Stability
    # =========================================================================
    @testset "4. 10,000-Step Zero-Secular-Drift Energy Stability" begin
        # 4-atom 3D coupled nonlinear oscillator
        N = 4
        q0 = [0.0 0.4 0.8 1.2;
              0.0 0.1 0.0 0.1;
              0.0 0.0 0.2 0.0]
        p0 = [0.05 -0.04 0.02 -0.03;
              0.01 0.02 -0.01 -0.02;
              -0.02 0.01 0.03 -0.02]
        m = [12.0, 14.0, 16.0, 12.0]

        factors = AbstractFactor[
            HarmonicBond(1, 2, 2000.0, 0.4),
            HarmonicBond(2, 3, 2000.0, 0.4),
            HarmonicBond(3, 4, 2000.0, 0.4),
            HarmonicAngle(1, 2, 3, 400.0, 2.5),
            HarmonicAngle(2, 3, 4, 400.0, 2.5)
        ]

        sys = Types.System(Types.AtomSet(copy(q0), copy(p0), m, zeros(N), fill(:C, N)), Types.NoBox())
        st = SYMT.SymplecticTernaryState(sys)

        dt = 0.0005
        n_steps = 10000
        energies = zeros(n_steps)

        for step in 1:n_steps
            SYMT.symplectic_ternary_step!(st, factors, dt; direction = +1)
            energies[step] = SYMT.total_energy(st, factors)
        end

        e_first_half = energies[1:5000]
        e_second_half = energies[5001:10000]
        m1 = mean(e_first_half)
        m2 = mean(e_second_half)

        # Zero secular drift: relative difference between first and second half means is < 1e-5
        relative_drift = abs(m2 - m1) / abs(m1)
        @test relative_drift < 1e-5

        # Energy remains bounded throughout the entire 10,000 steps (no divergence)
        max_rel_deviation = maximum(abs.(energies .- energies[1])) / abs(energies[1])
        @test max_rel_deviation < 1e-3

        # Kinetic and potential energies are mutually consistent
        @test SYMT.kinetic_energy(st) > 0.0
        @test SYMT.potential_energy(st, factors) > 0.0
        @test isapprox(SYMT.total_energy(st, factors), SYMT.kinetic_energy(st) + SYMT.potential_energy(st, factors); atol = 1e-12)
    end

    # =========================================================================
    # Test 5: O(1)-Memory Backward Trajectory Replay vs Stored Frames
    # =========================================================================
    @testset "5. O(1)-Memory Backward Trajectory Replay vs Stored Frames" begin
        N = 4
        q0 = [0.0 0.4 0.8 1.2;
              0.0 0.1 0.0 0.1;
              0.0 0.0 0.2 0.0]
        p0 = [0.05 -0.04 0.02 -0.03;
              0.01 0.02 -0.01 -0.02;
              -0.02 0.01 0.03 -0.02]
        m = [12.0, 14.0, 16.0, 12.0]

        factors = AbstractFactor[
            HarmonicBond(1, 2, 1500.0, 0.4),
            HarmonicBond(2, 3, 1500.0, 0.4),
            HarmonicBond(3, 4, 1500.0, 0.4),
            HarmonicAngle(1, 2, 3, 300.0, 2.5),
            HarmonicAngle(2, 3, 4, 300.0, 2.5)
        ]

        sys = Types.System(Types.AtomSet(copy(q0), copy(p0), m, zeros(N), fill(:C, N)), Types.NoBox())
        st0 = SYMT.SymplecticTernaryState(sys)

        dt = 0.001
        n_steps = 250

        # Reference: forward execution saving ALL 250 frames in memory (O(N_steps * N_atoms) memory)
        stored_frames = SYMT.SymplecticTernaryState[]
        st_fwd = copy(st0)
        for t in 0:(n_steps - 1)
            push!(stored_frames, copy(st_fwd))
            SYMT.symplectic_ternary_step!(st_fwd, factors, dt; direction = +1)
        end

        # Test A: Integer phase-space observable: Σ_t [ (t+1)*Σ Q_t - (t+2)*Σ P_t ]
        obs_int(step_idx, s) = Int128(step_idx + 1) * sum(s.Q) - Int128(step_idx + 2) * sum(s.P)
        ref_int_sum = sum(obs_int(t, stored_frames[t + 1]) for t in 0:(n_steps - 1))

        # Replay accumulator with O(1) memory: zero intermediate frames stored!
        replay_int_sum = SYMT.reversible_trajectory_replay_accum(st0, factors, dt, n_steps, obs_int)

        @test ref_int_sum == replay_int_sum
        @test (ref_int_sum - replay_int_sum) == 0

        # Test B: Float64 dynamic observable: Σ_t t * [ E_kin(t) + 0.5 * E_pot(t) ]
        obs_float(step_idx, s) = Float64(step_idx) * (SYMT.kinetic_energy(s) + 0.5 * SYMT.potential_energy(s, factors))
        ref_float_sum = sum(obs_float(t, stored_frames[t + 1]) for t in (n_steps - 1):-1:0)

        replay_float_sum = SYMT.reversible_trajectory_replay_accum(st0, factors, dt, n_steps, obs_float)

        @test ref_float_sum == replay_float_sum
        @test (ref_float_sum - replay_float_sum) == 0.0
    end

    # =========================================================================
    # Test 6: Flexibility with Single Factors and Callables
    # =========================================================================
    @testset "6. Single Factor and Callable Force Sources" begin
        q0 = [0.0 0.5; 0.0 0.0; 0.0 0.0]
        p0 = [0.1 -0.1; 0.0 0.0; 0.0 0.0]
        m = [10.0, 10.0]

        bond = HarmonicBond(1, 2, 1000.0, 0.5)

        # Single factor overload
        st1 = SYMT.SymplecticTernaryState(q0, p0, m)
        SYMT.symplectic_ternary_step!(st1, bond, 0.001; direction = +1)
        SYMT.symplectic_ternary_step!(st1, bond, 0.001; direction = -1)
        @test st1.Q == SYMT.round_symmetric.(q0 .* 3.0^26)
        @test st1.P == SYMT.round_symmetric.(p0 .* 3.0^24)

        # Callable force function overload: (q) -> (U, F)
        spring_fn = function(q)
            dx = q[1, 1] - q[1, 2]
            U = 0.5 * 1000.0 * dx^2
            F = zeros(3, 2)
            F[1, 1] = -1000.0 * dx
            F[1, 2] = 1000.0 * dx
            return (U, F)
        end

        st2 = SYMT.SymplecticTernaryState(q0, p0, m)
        SYMT.simulate_symplectic_ternary!(st2, spring_fn, 0.001, 100; direction = +1)
        SYMT.simulate_symplectic_ternary!(st2, spring_fn, 0.001, 100; direction = -1)
        @test st2.Q == SYMT.round_symmetric.(q0 .* 3.0^26)
        @test st2.P == SYMT.round_symmetric.(p0 .* 3.0^24)
    end
end
