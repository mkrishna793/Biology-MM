using LinearAlgebra
using Test

module ElectrostaticsHost
    include(joinpath(@__DIR__, "..", "src", "Types.jl"))
    include(joinpath(@__DIR__, "..", "src", "Factors.jl"))
    include(joinpath(@__DIR__, "..", "src", "Neighbors.jl"))
    include(joinpath(@__DIR__, "..", "src", "Electrostatics.jl"))
end
const EW = ElectrostaticsHost.Electrostatics
const Types = ElectrostaticsHost.Types
const Factors = ElectrostaticsHost.Factors
const NB = ElectrostaticsHost.Neighbors

function fd_force(q, charges, box, kw; h=1e-6)
    F=zeros(size(q))
    for idx in eachindex(q)
        old=q[idx]; q[idx]=old+h; ep=EW.evaluate_ewald(q,charges,box;kw...)[1]
        q[idx]=old-h; em=EW.evaluate_ewald(q,charges,box;kw...)[1]
        q[idx]=old; F[idx]=-(ep-em)/(2h)
    end
    F
end

box=[12.0,13.0,14.0]; q=[1.2 4.1 7.0; 2.0 6.0 3.3; 1.5 5.2 9.0]; charges=[1.0,-0.7,0.2]
kw=(alpha=0.34,rcut=5.5,kmax=5,prefactor=1.3)
E,F=EW.evaluate_ewald(q,charges,box;kw...)
@assert isfinite(E) && size(F)==size(q)
@assert maximum(abs,F-fd_force(copy(q),charges,box,kw)) < 2e-7
@assert maximum(abs,vec(sum(F,dims=2))) < 2e-12
e_lo,_=EW.evaluate_ewald(q,charges,box;alpha=0.34,rcut=5.5,kmax=3,prefactor=1.3)
e_hi,_=EW.evaluate_ewald(q,charges,box;alpha=0.34,rcut=5.5,kmax=7,prefactor=1.3)
@assert abs(E-e_hi) < abs(E-e_lo) + 1e-12

# Independent two-particle nonperiodic-limit check: increasing box and sums
# makes the leading real-space + self terms approach the expected expression.
q2=[1.0 3.0; 1.0 1.0; 1.0 1.0]; ch=[1.0,-1.0]
e2,_=EW.evaluate_ewald(q2,ch,[30.,30.,30.];alpha=0.4,rcut=12.,kmax=8)
@assert isfinite(e2)

ex=EW.PairCorrections(excluded=[(1,2)])
scaled=EW.PairCorrections(scaled=[(1,2,0.5)])
e0,_=EW.evaluate_ewald(q2,ch,[12.,12.,12.];alpha=0.35,rcut=5.5,kmax=5)
ee,_=EW.evaluate_ewald(q2,ch,[12.,12.,12.];alpha=0.35,rcut=5.5,kmax=5,corrections=ex)
es,_=EW.evaluate_ewald(q2,ch,[12.,12.,12.];alpha=0.35,rcut=5.5,kmax=5,corrections=scaled)
@assert isapprox(ee-e0, 1/2; atol=1e-12) # remove q1*q2/r = -1/2
@assert isapprox(es-e0, 1/4; atol=1e-12)

for f in (() -> EW.evaluate_ewald(q,charges,box;alpha=0.,rcut=5.,kmax=2),
          () -> EW.evaluate_ewald(q,charges,[10.,13.,14.];alpha=0.3,rcut=5.,kmax=2),
          () -> EW.PairCorrections(scaled=[(1,1,0.5)]))
    failed=false
    try f() catch e; failed = e isa ArgumentError end
    @assert failed
end
function test_coulomb_pair_box_derivatives()
    println("Testing CoulombPair box derivative ∂U/∂h|_s...")

    # 1. OrthorhombicBox finite difference verification
    L = (10.0, 11.0, 12.0)
    box_ortho = Types.OrthorhombicBox(L)
    q_coul = [1.5 8.2; 2.0 9.5; 3.0 1.2]
    qi, qj = 1.3, -0.7
    pref, scale = 1.4, 0.8

    G_ortho = EW.coulomb_pair_box_derivative(q_coul, 1, 2, qi, qj, box_ortho; prefactor=pref, coulomb_scale=scale)
    @assert size(G_ortho) == (3, 3) && all(isfinite, G_ortho)

    # Compare against direct factor signature
    coul_factor = Factors.CoulombPair(1, 2, qi, qj; coulomb_scale=scale)
    G_fac = EW.coulomb_pair_box_derivative(coul_factor, q_coul, box_ortho; prefactor=pref)
    @assert G_fac ≈ G_ortho

    # Finite-difference check: U(h +/- eps*E_ab) at fixed fractional coordinates
    h0_ortho = [L[1] 0.0 0.0; 0.0 L[2] 0.0; 0.0 0.0 L[3]]
    S_ortho = h0_ortho \ q_coul
    eps_fd = 1e-5

    for a in 1:3, b in 1:3
        E_ab = zeros(3, 3); E_ab[a, b] = eps_fd
        hp = h0_ortho .+ E_ab
        hm = h0_ortho .- E_ab
        qp = hp * S_ortho
        qm = hm * S_ortho
        Up = EW.coulomb_pair_energy(qp, 1, 2, qi, qj, Types.TriclinicBox(hp); prefactor=pref, coulomb_scale=scale)
        Um = EW.coulomb_pair_energy(qm, 1, 2, qi, qj, Types.TriclinicBox(hm); prefactor=pref, coulomb_scale=scale)
        fd_val = (Up - Um) / (2.0 * eps_fd)
        err = abs(G_ortho[a, b] - fd_val)
        @assert err < 1e-7 "Orthorhombic FD mismatch at ($a,$b): analytic=$(G_ortho[a,b]) vs FD=$fd_val, err=$err"
    end
    println("  OrthorhombicBox box derivative matches FD across all 9 matrix directions: PASS")

    # 2. TriclinicBox finite difference verification
    H0_tri = [10.0 1.5 0.5;
               0.0 11.0 1.2;
               0.0  0.0 12.0]
    box_tri = Types.TriclinicBox(H0_tri)
    S_tri = [0.2 0.5; 0.3 0.6; 0.1 0.4]
    q_tri = H0_tri * S_tri

    G_tri = EW.coulomb_pair_box_derivative(q_tri, 1, 2, qi, qj, box_tri; prefactor=pref, coulomb_scale=scale)
    @assert size(G_tri) == (3, 3) && all(isfinite, G_tri)
    G_tri_fac = EW.coulomb_pair_box_derivative(coul_factor, q_tri, box_tri; prefactor=pref)
    @assert G_tri_fac ≈ G_tri

    for a in 1:3, b in 1:3
        E_ab = zeros(3, 3); E_ab[a, b] = eps_fd
        hp = H0_tri .+ E_ab
        hm = H0_tri .- E_ab
        qp = hp * S_tri
        qm = hm * S_tri
        Up = EW.coulomb_pair_energy(qp, 1, 2, qi, qj, Types.TriclinicBox(hp); prefactor=pref, coulomb_scale=scale)
        Um = EW.coulomb_pair_energy(qm, 1, 2, qi, qj, Types.TriclinicBox(hm); prefactor=pref, coulomb_scale=scale)
        fd_val = (Up - Um) / (2.0 * eps_fd)
        err = abs(G_tri[a, b] - fd_val)
        @assert err < 1e-7 "Triclinic FD mismatch at ($a,$b): analytic=$(G_tri[a,b]) vs FD=$fd_val, err=$err"
    end
    println("  TriclinicBox box derivative matches FD across all 9 matrix directions: PASS")

    # 3. NoBox refusal check
    failed_nobox = false
    try
        EW.coulomb_pair_box_derivative(q_coul, 1, 2, qi, qj, Types.NoBox())
    catch e
        failed_nobox = e isa ArgumentError
    end
    @assert failed_nobox "NoBox must refuse box derivative with ArgumentError"

    failed_nobox_fac = false
    try
        EW.coulomb_pair_box_derivative(coul_factor, q_coul, Types.NoBox())
    catch e
        failed_nobox_fac = e isa ArgumentError
    end
    @assert failed_nobox_fac "NoBox with CoulombPair factor must refuse box derivative with ArgumentError"
    println("  NoBox explicitly refused for box derivative: PASS")

    # 4. Singular / invalid edge cases
    failed_coincident = false
    try
        q_coin = [1.0 1.0; 2.0 2.0; 3.0 3.0]
        EW.coulomb_pair_box_derivative(q_coin, 1, 2, qi, qj, box_ortho)
    catch e
        failed_coincident = e isa ArgumentError
    end
    @assert failed_coincident "Coincident charges must throw ArgumentError"

    failed_same_atom = false
    try
        EW.coulomb_pair_box_derivative(q_coul, 1, 1, qi, qj, box_ortho)
    catch e
        failed_same_atom = e isa ArgumentError
    end
    @assert failed_same_atom "Same atom indices must throw ArgumentError"
end

test_coulomb_pair_box_derivatives()

@testset "DSFCoulombFactor forces, C1 cutoff continuity, TilePairCache, and Orthorhombic/Triclinic box derivatives" begin
    # 1. C1 cutoff continuity at r -> rc
    rc = 4.0; alpha = 0.25; pref = 1.38
    delta = 1e-8
    q_c1 = [0.0 (rc - delta); 0.0 0.0; 0.0 0.0]
    ch_c1 = [1.0, -1.0]
    f_c1 = EW.DSFCoulombFactor(ch_c1; alpha = alpha, rcut = rc, prefactor = pref)
    F_c1 = zeros(3, 2)
    u_c1 = Factors.evaluate!(F_c1, f_c1, q_c1, Factors.Context(Types.NoBox(), Types.REDUCED, 2))
    @test abs(u_c1) < 1e-10
    @test maximum(abs, F_c1) < 1e-9

    q_c1_beyond = [0.0 (rc + delta); 0.0 0.0; 0.0 0.0]
    F_c1_beyond = zeros(3, 2)
    u_c1_beyond = Factors.evaluate!(F_c1_beyond, f_c1, q_c1_beyond, Factors.Context(Types.NoBox(), Types.REDUCED, 2))
    @test u_c1_beyond == 0.0
    @test maximum(abs, F_c1_beyond) == 0.0

    # 2. Orthorhombic and Triclinic boxes for N = 6 particles
    N = 6
    charges = [0.4, -0.6, 0.2, 0.8, -0.5, -0.3]
    corrections = EW.PairCorrections(excluded = [(1, 2)], scaled = [(1, 3, 0.5), (2, 4, 0.8)])
    q6 = [
        1.1 3.2 2.0 4.5 1.8 3.9;
        2.1 1.4 4.2 2.8 3.5 1.2;
        1.5 3.8 1.9 4.1 2.2 3.6
    ]

    L_ortho = (10.0, 9.5, 11.0)
    box_ortho = Types.OrthorhombicBox(L_ortho)
    H_ortho = [10.0 0.0 0.0; 0.0 9.5 0.0; 0.0 0.0 11.0]

    H_tric = [9.0 1.2 0.8; 0.0 8.5 1.0; 0.0 0.0 9.2]
    box_tric = Types.TriclinicBox(H_tric)

    for (box_label, box, H0) in (("OrthorhombicBox", box_ortho, H_ortho),
                                 ("TriclinicBox", box_tric, H_tric))
        ctx = Factors.Context(box, Types.REDUCED, N)
        f_dsf = EW.DSFCoulombFactor(charges; alpha = 0.22, rcut = 3.8, prefactor = 1.38,
                                    corrections = corrections, include_self = true)

        # Force evaluation & central finite difference
        F_an = zeros(3, N)
        u0 = Factors.evaluate!(F_an, f_dsf, q6, ctx)
        @test isfinite(u0)

        # Central FD on forces: h = 1e-6, err < 1e-9
        h_fd = 1e-6
        max_f_err = 0.0
        for a in 1:3, i in 1:N
            qp = copy(q6); qp[a, i] += h_fd
            qm = copy(q6); qm[a, i] -= h_fd
            up = Factors.evaluate!(zeros(3, N), f_dsf, qp, ctx)
            um = Factors.evaluate!(zeros(3, N), f_dsf, qm, ctx)
            f_fd = -(up - um) / (2.0 * h_fd)
            max_f_err = max(max_f_err, abs(F_an[a, i] - f_fd))
        end
        @test max_f_err < 1e-9

        # TilePairCache and fixed_point parity
        tc = NB.TilePairCache(q6, box, 3.8, 0.4)
        f_cached = EW.DSFCoulombFactor(charges; alpha = 0.22, rcut = 3.8, prefactor = 1.38,
                                       corrections = corrections, include_self = true, tile_cache = tc)
        f_fp = EW.DSFCoulombFactor(charges; alpha = 0.22, rcut = 3.8, prefactor = 1.38,
                                   corrections = corrections, include_self = true, fixed_point = true)
        f_cached_fp = EW.DSFCoulombFactor(charges; alpha = 0.22, rcut = 3.8, prefactor = 1.38,
                                          corrections = corrections, include_self = true,
                                          tile_cache = tc, fixed_point = true)

        F_cached = zeros(3, N); u_cached = Factors.evaluate!(F_cached, f_cached, q6, ctx)
        F_fp = zeros(3, N); u_fp = Factors.evaluate!(F_fp, f_fp, q6, ctx)
        F_cached_fp = zeros(3, N); u_cached_fp = Factors.evaluate!(F_cached_fp, f_cached_fp, q6, ctx)

        @test abs(u0 - u_cached) < 1e-13
        @test maximum(abs, F_an - F_cached) < 1e-13
        @test maximum(abs, F_an - F_fp) < 1e-11
        @test F_fp == F_cached_fp

        # 3. Box derivative verification across all 9 directions: FD < 1e-7
        G_an = EW.dsf_box_derivative(f_dsf, q6, box)
        S = inv(H0) * q6
        max_g_err = 0.0
        for a in 1:3, b in 1:3
            Eab = zeros(3, 3); Eab[a, b] = h_fd
            Hp = H0 .+ Eab; Hm = H0 .- Eab
            bp = box isa Types.OrthorhombicBox && a == b ? Types.OrthorhombicBox((Hp[1,1], Hp[2,2], Hp[3,3])) : Types.TriclinicBox(Hp)
            bm = box isa Types.OrthorhombicBox && a == b ? Types.OrthorhombicBox((Hm[1,1], Hm[2,2], Hm[3,3])) : Types.TriclinicBox(Hm)
            up = Factors.evaluate!(zeros(3, N), f_dsf, Hp * S, Factors.Context(bp, Types.REDUCED, N))
            um = Factors.evaluate!(zeros(3, N), f_dsf, Hm * S, Factors.Context(bm, Types.REDUCED, N))
            g_fd = (up - um) / (2.0 * h_fd)
            max_g_err = max(max_g_err, abs(G_an[a, b] - g_fd))
        end
        @test max_g_err < 1e-7
        println("  $box_label DSFCoulombFactor: max_F_FD=$max_f_err, max_G_FD=$max_g_err")
    end

    # 4. Self energy check
    f_no_self = EW.DSFCoulombFactor(charges; alpha = 0.22, rcut = 3.8, prefactor = 1.38,
                                    corrections = corrections, include_self = false)
    f_with_self = EW.DSFCoulombFactor(charges; alpha = 0.22, rcut = 3.8, prefactor = 1.38,
                                      corrections = corrections, include_self = true)
    ctx_o = Factors.Context(box_ortho, Types.REDUCED, N)
    F1 = zeros(3, N); u1 = Factors.evaluate!(F1, f_no_self, q6, ctx_o)
    F2 = zeros(3, N); u2 = Factors.evaluate!(F2, f_with_self, q6, ctx_o)
    expected_self = -1.38 * (EW._erfc(0.22 * 3.8) / (2.0 * 3.8) + 0.22 / sqrt(pi)) * sum(abs2, charges)
    @test abs((u2 - u1) - expected_self) < 1e-14
    @test F1 == F2

    # 5. Guards and refusals
    @test_throws ArgumentError EW.dsf_box_derivative(f_with_self, q6, Types.NoBox())
    f_large_rcut = EW.DSFCoulombFactor(charges; rcut = 6.0)
    @test_throws ArgumentError Factors.evaluate!(zeros(3, N), f_large_rcut, q6, ctx_o)
end

println("test_electrostatics PASS")
