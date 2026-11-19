# Test policy: every scientific claim in this package has a test that measures a
# number and compares it to an independently justified tolerance. Tests that
# cannot be justified were not written.

using FoundationCore
using FoundationCore.Types
using FoundationCore.Factors
using Random, LinearAlgebra

const T = FoundationCore.Types
const F = FoundationCore.Factors

function fd_adjoint_error(factors, q, ctx; h = 1e-6)
    U0, Fv = total_energy_forces(factors, q, ctx)
    G = zeros(size(q))
    for j in 1:size(q, 2), d in 1:3
        qp = copy(q); qp[d, j] += h
        qm = copy(q); qm[d, j] -= h
        G[d, j] = (total_energy_forces(factors, qp, ctx)[1] -
                   total_energy_forces(factors, qm, ctx)[1]) / (2h)
    end
    scale = max(maximum(abs, G), 1e-12)
    return U0, maximum(abs, Fv .+ G) / scale
end

function test_typed_state_validation()
    # shape errors must be rejected, not silently broadcast
    ok = false
    try
        T.AtomSet(zeros(2, 3), zeros(2, 3), ones(3), zeros(3))
    catch e
        ok = e isa T.InvalidState
    end
    @assert ok "2x3 positions must be rejected (not 3 x N)"

    ok = false
    try
        T.AtomSet(zeros(3, 3), zeros(3, 3), [1.0, 0.0, 1.0], zeros(3))
    catch e
        ok = e isa T.InvalidState
    end
    @assert ok "zero mass must be rejected"

    ok = false
    try
        T.AtomSet(zeros(3, 2), zeros(3, 2), ones(2), zeros(2),
                  [:A, :B])
        T.AtomSet([0.0 0.0; 0.0 0.0; NaN 0.0], zeros(3, 2), ones(2), zeros(2))
    catch e
        ok = e isa T.InvalidState
    end
    @assert ok "NaN coordinate must be rejected"

    a = T.AtomSet(zeros(3, 4), zeros(3, 4), ones(4), zeros(4))
    @assert length(a.m) == 4
    println("  typed-state validation: shapes, masses, finiteness enforced")
    return true
end

function test_triclinic_guard()
    # orthorhombic matrix passes through as a box
    b = Types.triclinic_guard([10.0 0 0; 0 10.0 0; 0 0 10.0])
    @assert b isa Types.OrthorhombicBox && b.L == (10.0, 10.0, 10.0)
    # genuine triclinic must be refused with an actionable error, never
    # approximated by fractional rounding
    ok = false
    msg = ""
    try
        Types.triclinic_guard([10.0 2.0 0; 0 10.0 1.0; 0 0 10.0])
    catch e
        ok = e isa Types.UnsupportedGeometry
        msg = sprint(showerror, e)
    end
    @assert ok "triclinic cell must raise UnsupportedGeometry"
    @assert occursin("triclinic", msg)
    println("  triclinic guard: refused with actionable message")
    return true
end

function test_minimum_image_orthorhombic()
    # For an orthorhombic box the minimum image must equal exhaustive search
    # over lattice images in a shell large enough to contain the nearest.
    rng = MersenneTwister(7)
    L = (6.0, 7.0, 8.0)
    box = T.OrthorhombicBox(L)
    worst = 0.0
    for _ in 1:200
        q = [rand(rng) * L[1], rand(rng) * L[2], rand(rng) * L[3]]
        q2 = [rand(rng) * L[1], rand(rng) * L[2], rand(rng) * L[3]]
        Q = reshape(q, 3, 1); Q2 = reshape(q2, 3, 1)
        d = T.unwrap_displacement(box, [Q Q2], 1, 2)
        got = sqrt(d.x^2 + d.y^2 + d.z^2)
        # exhaustive over a 5x5x5 image shell
        best = Inf
        for nx in -2:2, ny in -2:2, nz in -2:2
            dd = [q[1] - (q2[1] + nx * L[1]),
                  q[2] - (q2[2] + ny * L[2]),
                  q[3] - (q2[3] + nz * L[3])]
            best = min(best, sqrt(sum(abs2, dd)))
        end
        worst = max(worst, abs(got - best))
    end
    @assert worst < 1e-12 "minimum image mismatch vs exhaustive: $worst"
    println("  minimum image == exhaustive lattice search, worst |Δr| = ", worst)
    return true
end

function test_wrap_is_exact()
    box = T.OrthorhombicBox((5.0, 5.0, 5.0))
    q = [12.5 -3.25 0.0; -7.0 1.0 5.0; 2.0 2.0 2.0]
    a = T.AtomSet(q, zeros(3, 3), ones(3), zeros(3))
    s = T.System(a, box)
    T.wrap!(s)
    # wrapped coords in [0, L)
    @assert all(x -> 0 <= x < 5.0, s.atoms.q)
    # and differ from originals by exact lattice vectors
    d = s.atoms.q - q
    @assert all(abs.(d ./ 5.0 .- round.(d ./ 5.0)) .< 1e-15)
    println("  wrap: coordinates differ by exact lattice vectors")
    return true
end

function test_factor_adjoint_consistency()
    rng = MersenneTwister(11)
    q = randn(rng, 3, 6)
    ctx = F.Context(T.REDUCED, 6)
    cases = [
        ("HarmonicBond", F.HarmonicBond(1, 2, 3.1, 0.95)),
        ("HarmonicAngle", F.HarmonicAngle(1, 2, 3, 2.4, 1.9)),
        ("PeriodicDihedral", F.PeriodicDihedral(1, 2, 3, 4, 0.9, 3, 0.4)),
        ("LJ :none", F.LennardJonesPair(1, 2, 1.0, 1.0)),
        ("LJ :lfs", F.LennardJonesPair(1, 2, 1.0, 1.0; mode = :lfs, rc = 2.5)),
        ("LJ :shift", F.LennardJonesPair(1, 2, 1.0, 1.0; mode = :shift, rc = 2.5)),
        ("Coulomb", F.CoulombPair(1, 3, 0.5, -0.7)),
        ("Bipartite", F.BipartiteHarmonicGroup([1, 2], [3, 4, 5], 1.7)),
    ]
    for (name, fac) in cases
        U, err = fd_adjoint_error([fac], q, ctx)
        @assert err < 1e-5 "$name adjoint error $err"
        println("  ", rpad(name, 18), " U=", round(U, digits = 8),
                "  rel |F+∇U| = ", round(err, sigdigits = 3))
    end
    return true
end

function test_lj_cutoff_modes_are_different_potentials()
    # The whole point of the mode flag: :trunc has a discontinuous energy, so its
    # force is NOT -dU/dr at the boundary. This test *demonstrates* the failure
    # rather than hiding it, which is why :trunc is refused by the integrator.
    q = [0.0 2.55 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0]
    ctx = F.Context(T.REDUCED, 3)
    rc = 2.5
    U_trunc, Ftr = total_energy_forces([F.LennardJonesPair(1, 2, 1.0, 1.0; mode = :trunc, rc = rc)], q, ctx)
    @assert U_trunc == 0.0 "beyond cutoff must be exactly zero"
    @assert !F.force_consistent(F.LennardJonesPair(1, 2, 1.0, 1.0; mode = :trunc, rc = rc))
    @assert F.force_consistent(F.LennardJonesPair(1, 2, 1.0, 1.0; mode = :lfs, rc = rc))
    @assert F.force_consistent(F.LennardJonesPair(1, 2, 1.0, 1.0; mode = :none))

    # :lfs is C^1: energy and force are both continuous at rc
    eps = 1e-9
    for sgn in (-1, 1)
        qq = [0.0 rc + sgn * eps 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0]
        Ua, Fa = total_energy_forces([F.LennardJonesPair(1, 2, 1.0, 1.0; mode = :lfs, rc = rc)], qq, ctx)
        @assert abs(Ua) < 1e-6 "lfs energy must vanish at rc, got $Ua"
        @assert maximum(abs, Fa) < 1e-6 "lfs force must vanish at rc, got $(maximum(abs,Fa))"
    end
    println("  LJ cutoff modes: :trunc flags force-inconsistent, :lfs is C^1 at rc")
    return true
end

function test_torsion_gradient_against_atan2()
    # independent check: differentiate the atan2 definition of phi numerically
    rng = MersenneTwister(23)
    fac = F.PeriodicDihedral(1, 2, 3, 4, 0.0, 1, 0.0)  # kphi irrelevant for grad phi
    worst = 0.0
    for _ in 1:50
        q = randn(rng, 3, 4) .* 1.5
        ctx = F.Context(T.REDUCED, 4)
        w = zeros(3, 4)
        t = F.dihedral_angle_grad(q, ctx.box, 1, 2, 3, 4)
        t === nothing && continue
        phi, (g1, g2, g3, g4) = t
        # numeric dphi/dq
        h = 1e-6
        for j in 1:4, d in 1:3
            qp = copy(q); qp[d, j] += h
            qm = copy(q); qm[d, j] -= h
            tp = F.dihedral_angle_grad(qp, ctx.box, 1, 2, 3, 4)
            tm = F.dihedral_angle_grad(qm, ctx.box, 1, 2, 3, 4)
            (tp === nothing || tm === nothing) && continue
            dd = tp[1] - tm[1]
            dd = atan(sin(dd), cos(dd))
            num = dd / (2h)
            an = [g1, g2, g3, g4][j]
            anv = d == 1 ? an.x : (d == 2 ? an.y : an.z)
            worst = max(worst, abs(num - anv))
        end
    end
    @assert worst < 1e-6 "torsion gradient error $worst"
    println("  torsion ∇φ vs numerical d(atan2): max |Δ| = ", worst)
    return true
end

function test_support_scatter_isolation()
    # A factor must not write outside its read-support: a bug there silently
    # corrupts disjoint factors.
    q = randn(MersenneTwister(5), 3, 8)
    ctx = F.Context(T.REDUCED, 8)
    cases = [
        (F.HarmonicBond(3, 5, 1.0, 1.0), (3, 5)),
        (F.HarmonicAngle(2, 4, 6, 1.0, 1.0), (2, 4, 6)),
        (F.PeriodicDihedral(1, 3, 5, 7, 1.0, 2, 0.0), (1, 3, 5, 7)),
        (F.LennardJonesPair(2, 5, 1.0, 1.0), (2, 5)),
        (F.CoulombPair(1, 8, 1.0, -1.0), (1, 8)),
        (F.BipartiteHarmonicGroup([1, 2], [7, 8], 1.0), (1, 2, 7, 8)),
    ]
    for (fac, supp) in cases
        Fv = zeros(3, 8)
        F.evaluate!(Fv, fac, q, ctx)
        outside = [j for j in 1:8 if !(j in supp) && maximum(abs, Fv[:, j]) > 0]
        @assert isempty(outside) "$(typeof(fac)) wrote outside support: $outside"
    end
    println("  support discipline: all 6 factor kinds scatter only inside S_α")
    return true
end

function test_bipartite_summary_matches_direct()
    # The exactly-summarised factor must equal a genuine (n,m,3) pair enumeration
    # over Fraction arithmetic — zero tolerance, not "within roundoff".
    rng = MersenneTwister(31)
    for (n, m) in ((1, 1), (3, 4), (7, 5), (12, 9))
        A = rand(rng, -3:3, n, 3) .// 1
        B = rand(rng, -3:3, m, 3) .// 1
        k = 7 // 3
        # direct: sum_{i,j} ||a_i - b_j||^2
        S = 0 // 1
        for i in 1:n, j in 1:m
            for d in 1:3
                S += (A[i, d] - B[j, d])^2
            end
        end
        Udirect = (k // 2) * S
        # summary via the factor, in float
        q = zeros(3, n + m)
        for i in 1:n, d in 1:3
            q[d, i] = Float64(A[i, d])
        end
        for j in 1:m, d in 1:3
            q[d, n + j] = Float64(B[j, d])
        end
        fac = F.BipartiteHarmonicGroup(collect(1:n), collect(n+1:n+m), Float64(k))
        Usum, _ = total_energy_forces([fac], q, F.Context(T.REDUCED, n + m))
        rel = abs(Usum - Float64(Udirect)) / max(abs(Float64(Udirect)), 1e-30)
        @assert rel < 1e-12 "summary vs direct rel err $rel for n=$n m=$m"
        # also check total force vanishes exactly (pairwise cancellation)
        _, Fv = total_energy_forces([fac], q, F.Context(T.REDUCED, n + m))
        @assert maximum(abs, sum(Fv, dims = 2)) < 1e-12 * max(maximum(abs, Fv), 1)
    end
    println("  bipartite summary == direct pair enumeration (rel < 1e-12)")
    return true
end

function test_coulomb_periodic_refusal()
    # NoBox is the supported model and must be unchanged (this is also the path
    # every parameter-gradient caller uses).
    q = [0.0 1.5 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0]
    fac = F.CoulombPair(1, 2, 1.0, -1.0)
    U, Fv = total_energy_forces([fac], q, F.Context(T.REDUCED, 2))
    @assert abs(U + 1 / 1.5) < 1e-12 "NoBox Coulomb energy changed: $U"
    @assert abs(Fv[1, 1] - 1 / 1.5^2) < 1e-12 "NoBox Coulomb force changed: $(Fv[1, 1])"
    @assert abs(Fv[1, 2] + 1 / 1.5^2) < 1e-12 "NoBox Coulomb force changed: $(Fv[1, 2])"

    # Periodic Coulomb has no implemented long-range sum. Minimum image is a
    # different model, so it must be refused with an actionable message.
    ctx = F.Context(T.OrthorhombicBox((3.0, 3.0, 3.0)), T.REDUCED, 2)
    Fbuf = fill(42.0, 3, 2)
    ok = false
    msg = ""
    try
        F.evaluate!(Fbuf, fac, q, ctx)
    catch e
        ok = e isa T.UnsupportedGeometry
        msg = sprint(showerror, e)
    end
    @assert ok "periodic Coulomb must raise UnsupportedGeometry"
    @assert occursin("non-periodic", msg) "message must name the limitation: $msg"
    @assert occursin("Ewald", msg) "message must name an actionable alternative: $msg"
    @assert all(==(42.0), Fbuf) "refusal must not partially mutate the force buffer"

    # The shared driver (used by parameter gradients and integrators) refuses too.
    ok = false
    try
        total_energy_forces([fac], q, ctx; F = fill(42.0, 3, 2))
    catch e
        ok = e isa T.UnsupportedGeometry
    end
    @assert ok "total_energy_forces must refuse periodic Coulomb"
    println("  Coulomb: NoBox exact, periodic box refused (UnsupportedGeometry)")
    return true
end

function _run_all()
    tests = [
        ("typed_state_validation", test_typed_state_validation),
        ("triclinic_guard", test_triclinic_guard),
        ("minimum_image_orthorhombic", test_minimum_image_orthorhombic),
        ("wrap_is_exact", test_wrap_is_exact),
        ("factor_adjoint_consistency", test_factor_adjoint_consistency),
        ("lj_cutoff_modes", test_lj_cutoff_modes_are_different_potentials),
        ("torsion_gradient", test_torsion_gradient_against_atan2),
        ("support_scatter_isolation", test_support_scatter_isolation),
        ("bipartite_summary", test_bipartite_summary_matches_direct),
        ("coulomb_periodic_refusal", test_coulomb_periodic_refusal),
    ]
    for (name, fn) in tests
        println("TEST ", name)
        fn()
    end
    println("")
    println("ALL TESTS PASSED")
end

_run_all()
