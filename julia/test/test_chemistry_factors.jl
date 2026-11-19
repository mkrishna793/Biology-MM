# Finite-difference adjoint tests for the Phase-2 factor additions.
#
# Each factor must return U and scatter F = -dU/dq using the Phase-1 protocol.
# These are mathematical checks of the implemented potentials, not force-field
# validation.

using FoundationCore
using Test
const T = FoundationCore.Types
const F = FoundationCore.Factors
const CF = FoundationCore.ChemistryFactors
const RS = FoundationCore.ResolvedSystem
const SC = FoundationCore.SystemContract
const NB = FoundationCore.Neighbors

function factor_energy_forces(f, q, ctx)
    acc = zeros(3, size(q, 2))
    u = F.evaluate!(acc, f, q, ctx)
    return u, acc
end

function fd_check(f, q, ctx; h = 1e-6, rtol = 2e-6, label = "")
    acc = zeros(3, size(q, 2))
    u0 = F.evaluate!(acc, f, q, ctx)
    qp = copy(q); qm = copy(q)
    err = 0.0
    scale = 1.0
    for j in 1:size(q, 2), d in 1:3
        qp .= q; qm .= q
        qp[d, j] += h; qm[d, j] -= h
        up = F.evaluate!(zeros(3, size(q, 2)), f, qp, ctx)
        um = F.evaluate!(zeros(3, size(q, 2)), f, qm, ctx)
        fdc = -(up - um) / (2h)
        err = max(err, abs(fdc - acc[d, j]))
        scale = max(scale, abs(fdc), abs(acc[d, j]))
    end
    err <= rtol * scale || error("$label: FD force error $err exceeds $(rtol)×$scale")
    return u0, acc, err
end

function main()
    println("="^72)
    println("Chemistry factor tests")
    println("="^72)

    ctx = F.Context(T.NoBox(), T.REDUCED, 5)
    q = [0.0 1.7 2.9 1.1 2.2;
         0.0 0.4 1.9 2.4 0.2;
         0.0 0.9 0.7 1.6 2.8]

    # Multi-term proper torsion: both terms and both multiplicities must act.
    md = CF.MultiPeriodicDihedral(1, 2, 3, 4, [1.2, 0.7], [2, 3], [0.3, 2.0])
    U, Fm, e = fd_check(md, q, ctx; label = "multi torsion")
    # Collapsing to the first term is a detectable physics error.
    U1, _ = factor_energy_forces(F.PeriodicDihedral(1, 2, 3, 4, 1.2, 2, 0.3), q, ctx)
    @assert abs(U - U1) > 1e-6 "multi-term torsion fixture had no measurable second term"
    println("  multi torsion: U=$U FD_err=$e")

    # Harmonic improper: FD, periodic wrap across ±pi, and negative k (CHARMM).
    hi = CF.HarmonicImproper(1, 2, 3, 4, 3.0, 0.4)
    U, _, e = fd_check(hi, q, ctx; label = "harmonic improper")
    dmin = CF._wrap_pi(-pi + 0.05 - 0.4)
    @assert abs(dmin - (pi + 0.05 - 0.4)) < 1e-12
    @assert isapprox(CF._wrap_pi(pi + 0.1), -pi + 0.1; atol = 1e-12)
    neg = CF.HarmonicImproper(1, 2, 3, 4, -2.5, 0.0)
    _, _, e2 = fd_check(neg, q, ctx; label = "negative-k improper")
    println("  harmonic improper: U=$U FD_err=$e negative_k_FD_err=$e2")

    # Urey-Bradley: U = k(r-r0)^2 (no 1/2), analytic and FD.
    ub = CF.UreyBradley(1, 3, 500.0, 0.3)
    U, Fub, e = fd_check(ub, q, ctx; label = "Urey-Bradley")
    d = T.unwrap_displacement(T.NoBox(), q, 1, 3)
    r = sqrt(T.norm2(d))
    @assert isapprox(U, 500.0 * (r - 0.3)^2; rtol = 1e-14)
    println("  Urey-Bradley: U=$U FD_err=$e")

    # Explicit non-periodic pair table: must equal the sum of the existing
    # LennardJonesPair and CoulombPair factors, including 1-4 scale factors.
    sigma = [0.30, 0.24, 0.32, 0.28, 0.26]
    eps = [0.6, 0.1, 0.4, 0.2, 0.05]
    z = [0.4, -0.2, 0.1, -0.3, 0.0]
    pairs = [(1, 2, 0.5, 0.8), (1, 3, 1.0, 1.0), (2, 4, 1.0, 1.0), (3, 5, 0.5, 0.6)]
    ti = [p[1] for p in pairs]; tj = [p[2] for p in pairs]
    tsig = [0.5 * (sigma[p[1]] + sigma[p[2]]) for p in pairs]
    teps = [sqrt(eps[p[1]] * eps[p[2]]) for p in pairs]
    tqq = [z[p[1]] * z[p[2]] for p in pairs]
    tlj = [p[3] for p in pairs]; tc = [p[4] for p in pairs]
    pt = CF.PairTableNonbonded(ti, tj, tsig, teps, tqq, tlj, tc; mode = :none)
    U, Fpt, e = fd_check(pt, q, ctx; label = "pair table")
    direct = F.AbstractFactor[]
    for (p, s, epsy, qq, lj, cc) in zip(pairs, tsig, teps, tqq, tlj, tc)
        if lj != 0 && epsy != 0
            push!(direct, F.LennardJonesPair(p[1], p[2], epsy, s; mode = :none, lj_scale = lj))
        end
        if cc != 0 && qq != 0
            push!(direct, F.CoulombPair(p[1], p[2], z[p[1]], z[p[2]]; coulomb_scale = cc))
        end
    end
    Ud, Fd = F.total_energy_forces(direct, q, ctx)
    @assert abs(U - Ud) <= 1e-12 * max(1.0, abs(Ud))
    @assert maximum(abs, Fpt .- Fd) <= 1e-12 * max(1.0, maximum(abs, Fd))
    println("  pair table: U=$U direct_err=$(abs(U - Ud)) FD_err=$e")

    # AMBER 1-6-4: U = -C/r^4, analytic and FD-checked.
    c64 = CF.PairC64Table([1], [3], [0.075]; mode = :trunc, rcut = 10.0)
    Uc, Fc, ec = fd_check(c64, q, ctx; label = "1-6-4")   # ctx NoBox, mode :trunc
    rc = sqrt(T.norm2(T.unwrap_displacement(T.NoBox(), q, 1, 3)))
    @assert isapprox(Uc, -0.075 / rc^4; rtol = 1e-14)
    println("  1-6-4: U=$Uc FD_err=$ec")

    # Support discipline: a factor writes only into its own read-support.
    for (f, supp) in ((md, (1, 2, 3, 4)), (hi, (1, 2, 3, 4)),
                      (ub, (1, 3)), (c64, (1, 3)),
                      (pt, Tuple(sort!(unique!(vcat(ti, tj))))))
        acc = zeros(3, 5)
        F.evaluate!(acc, f, q, ctx)
        @assert F.read_support(f) == supp
        @assert all(acc[d, j] == 0.0 for d in 1:3, j in 1:5 if !(j in supp))
    end
    @assert CF.MultiPeriodicDihedral(1, 2, 3, 4, [1.0], [1], [0.0]) isa F.AbstractFactor
    @assert CF.HarmonicImproper(1, 2, 3, 4, 1.0, 0.0) isa F.AbstractFactor
    @assert !F.force_consistent(CF.PairTableNonbonded(ti, tj, tsig, teps, tqq, tlj, tc;
                                                      mode = :trunc, rcut = 0.5))
    @assert F.force_consistent(pt)
    @assert !F.force_consistent(c64)
    @assert F.force_consistent(CF.PairC64Table([1], [3], [0.075]; mode = :none))
    dup_threw = false
    try
        CF.PairTableNonbonded([1, 2], [2, 1], [0.3, 0.3], [0.1, 0.1], [0.0, 0.0],
                              [1.0, 1.0], [1.0, 1.0])
    catch e
        dup_threw = e isa ArgumentError
    end
    @assert dup_threw "PairTableNonbonded must reject duplicate canonical pairs"
    println("  support discipline, duplicate guard, and force_consistent flags: OK")

    # Degenerate geometries return Inf rather than a fabricated finite force.
    qbad = copy(q); qbad[:, 2] .= qbad[:, 1]
    @assert F.evaluate!(zeros(3, 5), md, qbad, ctx) == Inf
    qbad2 = copy(q); qbad2[:, 3] .= qbad2[:, 1]
    @assert F.evaluate!(zeros(3, 5), ub, qbad2, ctx) == Inf
    println("  degenerate geometry: Inf")

    # Phase 1-3 seam: analytic box derivatives (orthorhombic + skewed triclinic)
    # and triclinic MIC agreement with Neighbors.triclinic_mic.
    H_skew = [5.0 1.8 0.4;
              0.0 4.6 1.1;
              0.0 0.0 5.2]
    ortho_box = T.OrthorhombicBox((5.0, 4.6, 5.2))
    tric_box = T.TriclinicBox(H_skew)

    # Skewed PBC coordinates: atoms 2 and 4 shifted across periodic boundaries.
    q_pbc = copy(q)
    q_pbc[:, 2] .+= H_skew[:, 1]
    q_pbc[:, 4] .-= H_skew[:, 2]
    for i in 1:4, j in (i + 1):5
        d_types = T.unwrap_displacement(tric_box, q_pbc, i, j)
        d_nb = NB.triclinic_mic(tric_box, q_pbc[1, i] - q_pbc[1, j],
                                q_pbc[2, i] - q_pbc[2, j],
                                q_pbc[3, i] - q_pbc[3, j])
        @assert isapprox(d_types.x, d_nb.x; atol = 1e-13) &&
                isapprox(d_types.y, d_nb.y; atol = 1e-13) &&
                isapprox(d_types.z, d_nb.z; atol = 1e-13) "triclinic MIC mismatch ($i,$j)"
    end

    c64_none = CF.PairC64Table([1, 2], [3, 5], [0.075, -0.02]; mode = :none)
    bd_factors = F.AbstractFactor[md, hi, ub, c64_none]
    bd_names = [:multi_dihedral, :improper, :urey_bradley, :lj146]
    for (box_label, box, H0) in (("orthorhombic", ortho_box, [5.0 0.0 0.0; 0.0 4.6 0.0; 0.0 0.0 5.2]),
                                  ("triclinic", tric_box, H_skew))
        rs = RS.ResolvedSystem(masses = ones(5), charges = zeros(5), box = box,
                               factors = bd_factors, component_names = bd_names,
                               declared = [:energy, :cartesian_forces, :component_energies, :box_derivative])
        @assert :box_derivative in RS.engine_capabilities(rs)
        for f in bd_factors
            @assert SC.supports_box_derivative(f, box)
        end
        G = SC.box_derivative(rs, q_pbc)
        G_sum = zeros(3, 3)
        for nm in bd_names
            G_sum .+= SC.box_derivative_component(rs, q_pbc, nm)
        end
        @assert maximum(abs, G .- G_sum) <= 1e-12 * max(1.0, maximum(abs, G))

        # Check all 9 entries of G[a,b] = ∂U/∂h[a,b] at fixed fractional s = H0^-1 q_pbc.
        S = SC.fractional_coordinates(rs, q_pbc)
        eps = 1e-6
        max_fd_err = 0.0
        for a in 1:3, b in 1:3
            Eab = zeros(3, 3)
            Eab[a, b] = eps
            Hp = H0 .+ Eab
            Hm = H0 .- Eab
            bp = box isa T.OrthorhombicBox && a == b ? T.OrthorhombicBox((Hp[1,1], Hp[2,2], Hp[3,3])) : T.TriclinicBox(Hp)
            bm = box isa T.OrthorhombicBox && a == b ? T.OrthorhombicBox((Hm[1,1], Hm[2,2], Hm[3,3])) : T.TriclinicBox(Hm)
            up = F.total_energy_forces(bd_factors, Hp * S, F.Context(bp, T.REDUCED, 5))[1]
            um = F.total_energy_forces(bd_factors, Hm * S, F.Context(bm, T.REDUCED, 5))[1]
            g_fd = (up - um) / (2.0 * eps)
            max_fd_err = max(max_fd_err, abs(G[a, b] - g_fd))
        end
        scale = max(1.0, maximum(abs, G))
        @assert max_fd_err <= 1e-7 * scale "$box_label box_derivative FD err $max_fd_err exceeds 1e-7 * $scale"

        # Check virial and affine_pressure identities.
        Xi = SC.virial(rs, q_pbc)
        @assert maximum(abs, Xi .+ H0 * transpose(G)) <= 1e-13 * max(1.0, maximum(abs, Xi))
        V = T.box_volume(box)
        p_aff = SC.affine_pressure(rs, q_pbc)
        @assert isapprox(p_aff, -sum(G[i, j] * H0[i, j] for i in 1:3, j in 1:3) / (3.0 * V); rtol = 1e-13)
        println("  $box_label ChemistryFactors box_derivative: max_FD_err=$max_fd_err, p_aff=$p_aff")
    end

    # Discontinuous :trunc PairC64Table and non-periodic PairTableNonbonded must refuse box_derivative.
    @assert !SC.supports_box_derivative(c64, ortho_box)
    @assert !SC.supports_box_derivative(pt, ortho_box)
    rs_c64_trunc = RS.ResolvedSystem(masses = ones(5), charges = zeros(5), box = ortho_box,
                                     factors = [c64], component_names = [:lj146])
    @assert !(:box_derivative in RS.engine_capabilities(rs_c64_trunc))
    threw_trunc = false
    try
        SC.box_derivative(rs_c64_trunc, q)
    catch e
        threw_trunc = e isa SC.MissingBoxDerivative
    end
    @assert threw_trunc "PairC64Table(mode=:trunc) must refuse box_derivative"

    # Linear-force-shifted (:lfs) PairC64Table and with_lfs_c64 upgrade:
    # force-consistent, zero energy & force at r == cutoff, exact FD forces and box derivative.
    c64_lfs = CF.with_lfs_c64(c64)
    @assert c64_lfs.mode === :lfs
    @assert F.force_consistent(c64_lfs)
    @assert SC.supports_box_derivative(c64_lfs, ortho_box)
    @assert SC.supports_box_derivative(c64_lfs, tric_box)

    # Check exact FD forces and box derivative for PairC64Table(mode=:lfs) on orthorhombic & triclinic boxes
    for (box_label, box, H0) in (("orthorhombic", ortho_box, [5.0 0.0 0.0; 0.0 4.6 0.0; 0.0 0.0 5.2]),
                                  ("triclinic", tric_box, H_skew))
        ctx_b = F.Context(box, T.REDUCED, 5)
        F_an = zeros(3, 5)
        u0 = F.evaluate!(F_an, c64_lfs, q_pbc, ctx_b)
        @assert isfinite(u0)
        h_fd = 1e-6
        max_f_err = 0.0
        for a in 1:3, i in 1:5
            qp = copy(q_pbc); qp[a, i] += h_fd
            qm = copy(q_pbc); qm[a, i] -= h_fd
            up = F.evaluate!(zeros(3, 5), c64_lfs, qp, ctx_b)
            um = F.evaluate!(zeros(3, 5), c64_lfs, qm, ctx_b)
            f_fd = -(up - um) / (2.0 * h_fd)
            max_f_err = max(max_f_err, abs(F_an[a, i] - f_fd))
        end
        @assert max_f_err <= 1e-8 "$box_label PairC64Table(mode=:lfs) force FD err $max_f_err > 1e-8"

        rs_lfs = CF.with_lfs_c64(rs_c64_trunc)
        @assert rs_lfs.factors[1].mode === :lfs
        rs_b = RS.ResolvedSystem(masses = ones(5), charges = zeros(5), box = box,
                                 factors = F.AbstractFactor[c64_lfs], component_names = [:lj146],
                                 declared = [:energy, :cartesian_forces, :component_energies, :box_derivative])
        G_lfs = SC.box_derivative(rs_b, q_pbc)
        S_lfs = SC.fractional_coordinates(rs_b, q_pbc)
        max_g_err = 0.0
        for a in 1:3, b in 1:3
            Eab = zeros(3, 3); Eab[a, b] = h_fd
            Hp = H0 .+ Eab; Hm = H0 .- Eab
            bp = box isa T.OrthorhombicBox && a == b ? T.OrthorhombicBox((Hp[1,1], Hp[2,2], Hp[3,3])) : T.TriclinicBox(Hp)
            bm = box isa T.OrthorhombicBox && a == b ? T.OrthorhombicBox((Hm[1,1], Hm[2,2], Hm[3,3])) : T.TriclinicBox(Hm)
            up = F.evaluate!(zeros(3, 5), c64_lfs, Hp * S_lfs, F.Context(bp, T.REDUCED, 5))
            um = F.evaluate!(zeros(3, 5), c64_lfs, Hm * S_lfs, F.Context(bm, T.REDUCED, 5))
            g_fd = (up - um) / (2.0 * h_fd)
            max_g_err = max(max_g_err, abs(G_lfs[a, b] - g_fd))
        end
        @assert max_g_err <= 1e-7 "$box_label PairC64Table(mode=:lfs) box_derivative FD err $max_g_err > 1e-7"
        println("  $box_label PairC64Table(mode=:lfs): max_F_FD=$max_f_err, max_G_FD=$max_g_err")
    end

    @testset "PairC64Table fixed_point and threaded determinism" begin
        i_idx = [1, 2, 1, 3]
        j_idx = [3, 4, 5, 5]
        c_vals = [0.08, 0.05, -0.02, 0.06]
        c64_std = CF.PairC64Table(i_idx, j_idx, c_vals; mode = :lfs, rcut = 4.0, threaded = false, fixed_point = false)
        c64_fp = CF.PairC64Table(i_idx, j_idx, c_vals; mode = :lfs, rcut = 4.0, threaded = false, fixed_point = true)
        c64_th = CF.PairC64Table(i_idx, j_idx, c_vals; mode = :lfs, rcut = 4.0, threaded = true, fixed_point = false)
        c64_th_fp = CF.PairC64Table(i_idx, j_idx, c_vals; mode = :lfs, rcut = 4.0, threaded = true, fixed_point = true)

        ctx_o = F.Context(ortho_box, T.REDUCED, 5)
        F_std = zeros(3, 5); u_std = F.evaluate!(F_std, c64_std, q_pbc, ctx_o)
        F_fp = zeros(3, 5); u_fp = F.evaluate!(F_fp, c64_fp, q_pbc, ctx_o)
        F_th = zeros(3, 5); u_th = F.evaluate!(F_th, c64_th, q_pbc, ctx_o)
        F_th_fp = zeros(3, 5); u_th_fp = F.evaluate!(F_th_fp, c64_th_fp, q_pbc, ctx_o)

        @test abs(u_std - u_fp) < 1e-13
        @test abs(u_std - u_th) < 1e-13
        @test abs(u_std - u_th_fp) < 1e-13
        @test maximum(abs, F_std - F_fp) < 1e-11
        @test maximum(abs, F_std - F_th) < 1e-11
        @test F_fp == F_th_fp

        h_fd = 1e-6
        max_f_err = 0.0
        for a in 1:3, i in 1:5
            qp = copy(q_pbc); qp[a, i] += h_fd
            qm = copy(q_pbc); qm[a, i] -= h_fd
            up = F.evaluate!(zeros(3, 5), c64_fp, qp, ctx_o)
            um = F.evaluate!(zeros(3, 5), c64_fp, qm, ctx_o)
            f_fd = -(up - um) / (2.0 * h_fd)
            max_f_err = max(max_f_err, abs(F_fp[a, i] - f_fd))
        end
        @test max_f_err < 1e-8
    end

    @testset "NBFIXPairTable forces and box derivatives (Orthorhombic & Triclinic)" begin
        i_nbfix = [1, 2, 3, 1]
        j_nbfix = [2, 3, 4, 4]
        sigma = [0.32, 0.28, 0.35, 0.30]
        epsilon = [1.2, 0.9, 1.5, 0.8]
        scale = [1.0, 0.5, 1.0, 0.8]

        nbfix_none = CF.NBFIXPairTable(i_nbfix, j_nbfix, sigma, epsilon; scale = scale, mode = :none)
        nbfix_lfs = CF.NBFIXPairTable(i_nbfix, j_nbfix, sigma, epsilon; scale = scale, mode = :lfs, rcut = 3.2)
        nbfix_trunc = CF.NBFIXPairTable(i_nbfix, j_nbfix, sigma, epsilon; scale = scale, mode = :trunc, rcut = 3.2)

        @test !F.is_local(nbfix_none)
        @test F.force_consistent(nbfix_none)
        @test F.force_consistent(nbfix_lfs)
        @test !F.force_consistent(nbfix_trunc)

        nbfix_fp = CF.NBFIXPairTable(i_nbfix, j_nbfix, sigma, epsilon; scale = scale, mode = :lfs, rcut = 3.2, fixed_point = true)
        nbfix_th = CF.NBFIXPairTable(i_nbfix, j_nbfix, sigma, epsilon; scale = scale, mode = :lfs, rcut = 3.2, threaded = true)
        nbfix_th_fp = CF.NBFIXPairTable(i_nbfix, j_nbfix, sigma, epsilon; scale = scale, mode = :lfs, rcut = 3.2, threaded = true, fixed_point = true)

        ctx_o = F.Context(ortho_box, T.REDUCED, 5)
        F_std = zeros(3, 5); u_std = F.evaluate!(F_std, nbfix_lfs, q_pbc, ctx_o)
        F_fp = zeros(3, 5); u_fp = F.evaluate!(F_fp, nbfix_fp, q_pbc, ctx_o)
        F_th = zeros(3, 5); u_th = F.evaluate!(F_th, nbfix_th, q_pbc, ctx_o)
        F_th_fp = zeros(3, 5); u_th_fp = F.evaluate!(F_th_fp, nbfix_th_fp, q_pbc, ctx_o)

        @test abs(u_std - u_fp) < 1e-13
        @test abs(u_std - u_th) < 1e-13
        @test maximum(abs, F_std - F_fp) < 1e-11
        @test maximum(abs, F_std - F_th) < 1e-11
        @test F_fp == F_th_fp

        for (box_label, box, H0) in (("orthorhombic", ortho_box, [5.0 0.0 0.0; 0.0 4.6 0.0; 0.0 0.0 5.2]),
                                      ("triclinic", tric_box, H_skew))
            ctx_b = F.Context(box, T.REDUCED, 5)

            for (m_label, factor) in ((":none", nbfix_none), (":lfs", nbfix_lfs))
                F_an = zeros(3, 5)
                u0 = F.evaluate!(F_an, factor, q_pbc, ctx_b)
                @test isfinite(u0)
                h_fd = 1e-6
                max_f_err = 0.0
                for a in 1:3, i in 1:5
                    qp = copy(q_pbc); qp[a, i] += h_fd
                    qm = copy(q_pbc); qm[a, i] -= h_fd
                    up = F.evaluate!(zeros(3, 5), factor, qp, ctx_b)
                    um = F.evaluate!(zeros(3, 5), factor, qm, ctx_b)
                    f_fd = -(up - um) / (2.0 * h_fd)
                    max_f_err = max(max_f_err, abs(F_an[a, i] - f_fd))
                end
                @test max_f_err < 1e-8
            end

            G_an = CF.accumulate_box_derivative(nbfix_lfs, q_pbc, box)
            S_frac = inv(H0) * q_pbc
            h_fd = 1e-6
            max_g_err = 0.0
            for a in 1:3, b in 1:3
                Eab = zeros(3, 3); Eab[a, b] = h_fd
                Hp = H0 .+ Eab; Hm = H0 .- Eab
                bp = box isa T.OrthorhombicBox && a == b ? T.OrthorhombicBox((Hp[1,1], Hp[2,2], Hp[3,3])) : T.TriclinicBox(Hp)
                bm = box isa T.OrthorhombicBox && a == b ? T.OrthorhombicBox((Hm[1,1], Hm[2,2], Hm[3,3])) : T.TriclinicBox(Hm)
                up = F.evaluate!(zeros(3, 5), nbfix_lfs, Hp * S_frac, F.Context(bp, T.REDUCED, 5))
                um = F.evaluate!(zeros(3, 5), nbfix_lfs, Hm * S_frac, F.Context(bm, T.REDUCED, 5))
                g_fd = (up - um) / (2.0 * h_fd)
                max_g_err = max(max_g_err, abs(G_an[a, b] - g_fd))
            end
            @test max_g_err < 1e-7
            println("  $box_label NBFIXPairTable(mode=:lfs): max_G_FD=$max_g_err")

            @test_throws ArgumentError CF.accumulate_box_derivative(nbfix_trunc, q_pbc, box)
        end

        @test_throws ArgumentError CF.accumulate_box_derivative(nbfix_lfs, q_pbc, T.NoBox())
    end

    println("CHEMISTRY FACTOR TESTS PASSED")
    return true
end

main()
