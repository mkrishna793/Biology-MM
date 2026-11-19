using LinearAlgebra
using Random
using Statistics
using Test

module _POHost
    include(joinpath(@__DIR__, "..", "src", "Types.jl"))
    include(joinpath(@__DIR__, "..", "src", "Factors.jl"))
    include(joinpath(@__DIR__, "..", "src", "Polarizable.jl"))
    include(joinpath(@__DIR__, "..", "src", "PolarizationOperator.jl"))
end
const PO = _POHost.PolarizationOperator
const P = _POHost.Polarizable
const T = _POHost.Types

function independent_dense(q, alpha)
    active=findall(>(0.0),alpha); m=length(active); A=zeros(3m,3m)
    for a in 1:m, d in 1:3; A[3(a-1)+d,3(a-1)+d]=1/alpha[active[a]]; end
    for a in 1:m-1, b in a+1:m
        i,j=active[a],active[b]
        dx=q[1,i]-q[1,j]; dy=q[2,i]-q[2,j]; dz=q[3,i]-q[3,j]
        r2=dx^2+dy^2+dz^2; r=sqrt(r2)
        for u in 1:3, v in 1:3
            du=(u==1 ? dx : u==2 ? dy : dz)
            dv=(v==1 ? dx : v==2 ? dy : dz)
            t=3du*dv/r^5-(u==v ? 1/r^3 : 0.0)
            A[3(a-1)+u,3(b-1)+v]=-t
            A[3(b-1)+v,3(a-1)+u]=-t
        end
    end
    return A
end

function test_two_site_sign_and_dense_match()
    q=[0.0 2.0; 0 0; 0 0]
    A=PO.build_operator(q,[1.0,-1.0],[0.1,0.1])
    @assert isapprox(A.field, [0.25,0,0,0.25,0,0]; atol=1e-14)
    dense=independent_dense(q,[0.1,0.1])
    @assert isapprox(dense,PO.dense_reference(A);atol=1e-14)
    @assert isapprox(dense, transpose(dense); atol=1e-14)
    @assert isapprox(dense[1,4], -0.25; atol=1e-14)
    r=PO.solve_dipoles!(A;tol=1e-12)
    @assert r.status===:converged && r.spd===:certified
    @assert r.iterations>0 && r.residual<1e-12
    @assert isapprox(PO.require_converged(r), dense\A.field; atol=1e-13)
    @assert isfinite(PO.induced_energy(A,r))
    println("  two-site field sign, tensor sign, dense solve match: residual=",r.residual," iter=",r.iterations)
end

function test_active_sites_and_matrix_free_storage()
    q=[0.0 1.0 3.0;0 0 0;0 0 0]
    A=PO.build_operator(q,[0.0,1.0,-0.5],[0.0,0.2,0.1])
    @assert A.active==[2,3] && length(A.field)==6
    y=zeros(6); mul!(y,A,ones(6))
    @assert length(y)==6 && size(A)==(6,6)
    @assert !hasproperty(A,:matrix)
    println("  alpha=0 eliminated; active dimension=",size(A,1),"; O(N²) direct matvec")
end

function test_failures_are_not_usable()
    q=[0.0 0.01;0 0;0 0]
    unstable=PO.build_operator(q,[0.0,0.0],[1.0,1.0])
    r=PO.solve_dipoles!(unstable)
    @assert r.status===:indefinite && r.mu===nothing
    try PO.require_converged(r); error("accepted indefinite solve") catch e; @assert e isa ErrorException end
    safe=PO.build_operator([0.0 2.0;0 0;0 0],[1.0,-1.0],[0.1,0.1])
    nr=PO.solve_dipoles!(safe;tol=1e-14,maxiter=0)
    @assert nr.status===:not_converged && nr.mu===nothing && nr.iterations==0
    @assert isfinite(nr.residual)
    for (qq,aa) in ((zeros(3,2),[0.1,0.1]),([0.0 2.0;0 0;0 0],[-1.0,0.1]))
        failed=false
        try PO.build_operator(qq,[1.0,-1.0],aa) catch e; failed=e isa ArgumentError end
        @assert failed
    end
    failed=false
    try PO.build_operator([0.0 2.0;0 0;0 0],[1.0,-1.0],[0.1,0.1],T.OrthorhombicBox((10.0,10.0,10.0))) catch e; failed=e isa ArgumentError end
    @assert failed
    println("  indefinite, nonconverged, coincident, invalid alpha and periodic physics refused")
end

function test_numeric_edge_guards()
    failed=false
    try PO.build_operator([0.0 2.0;0 0;0 0],[0.0,0.0],[nextfloat(0.0),0.1]) catch e; failed=e isa ArgumentError end
    @assert failed
    failed=false
    try PO.build_operator([0.0 2.0;0 0;0 0],[0.0,0.0],BigFloat[big"1e-10000",0.1]) catch e; failed=e isa ArgumentError end
    @assert failed

    failed=false
    try PO.build_operator([0.0 1e-4;0 0;0 0],[0.0,1e308],[0.1,0.1]) catch e; failed=e isa ArgumentError end
    @assert failed

    failed=false
    try PO.build_operator([-1e308 1e308;0 0;0 0],[0.0,0.0],[0.1,0.1]) catch e; failed=e isa ArgumentError end
    @assert failed

    A=PO.build_operator([0.0 2.0;0 0;0 0],[1.0,-1.0],[0.1,0.1])
    for tol in (0.0,-1.0,Inf,NaN,BigFloat("1e10000"),BigFloat("1e-10000"))
        failed=false
        try PO.solve_dipoles!(A;tol=tol) catch e; failed=e isa ArgumentError end
        @assert failed
    end
    A.field[1]=Inf
    failed=false
    try PO.solve_dipoles!(A) catch e; failed=e isa ArgumentError end
    @assert failed
    println("  non-finite inverse alpha, geometry/field overflow, RHS and tolerances refused")
end

function test_convergence_status_and_residual()
    q = [0.0 2.0; 0 0; 0 0]
    z = [1.0, -1.0]
    α = [0.1, 0.1]
    # Convergence on stable system
    A = PO.build_operator(q, z, α)
    r = PO.solve_dipoles!(A; tol=1e-12)
    @assert r.status === :converged
    @assert r.residual < 1e-12
    @assert r.iterations > 0
    @assert r.spd === :certified

    # Polarizable module convergence and status reporting
    s = P.InducedDipoleSystem(q, z, α, T.NoBox(), T.REDUCED; thole = true, a_thole = 1.0)
    r_p = P.solve_dipoles!(s; tol=1e-12)
    @assert r_p.status === :converged
    @assert r_p.converged == true
    @assert r_p.residual < 1e-12
    @assert r_p.n_iter > 0

    # SCF solve method
    r_scf = P.scf_solve_dipoles!(s; tol=1e-12, maxiter=200)
    @assert r_scf.status === :converged
    @assert r_scf.converged == true
    @assert r_scf.residual < 1e-12
    @assert r_scf.n_iter > 0
    @assert norm(r_scf.μ - r_p.μ) < 1e-10

    # Non-convergence with maxiter=0
    nr = PO.solve_dipoles!(A; tol=1e-14, maxiter=0)
    @assert nr.status === :not_converged && nr.mu === nothing && nr.iterations == 0

    nr_p = P.solve_dipoles!(s; method = :scf, tol=1e-14, maxiter=0)
    @assert nr_p.status === :not_converged && nr_p.converged == false && nr_p.n_iter == 0

    # Indefinite / catastrophe detection
    q_close = [0.0 0.01; 0 0; 0 0]
    unstable = PO.build_operator(q_close, [0.0, 0.0], [1.0, 1.0])
    r_indef = PO.solve_dipoles!(unstable)
    @assert r_indef.status === :indefinite && r_indef.spd === :indefinite

    s_indef = P.InducedDipoleSystem(q_close, [0.0, 0.0], [1.0, 1.0], T.NoBox(), T.REDUCED; thole = false)
    r_p_indef = P.solve_dipoles!(s_indef)
    @assert r_p_indef.status === :indefinite && r_p_indef.spd == false

    println("  convergence status and honest residual reporting: PASS")
end

function test_force_consistency_against_fd_with_damping()
    # Water-like 3-site system with Thole damping enabled
    q = [0.0 2.9 0.5;
         0.0 0.0 2.3;
         0.0 0.0 0.0]
    z = [1.0, -0.5, -0.5]
    α = [0.5, 0.3, 0.3]
    h = 1e-6

    # 1. Test via Polarizable module (damping active)
    s = P.InducedDipoleSystem(q, z, α, T.NoBox(), T.REDUCED; thole = true, a_thole = 1.2)
    r = P.solve_dipoles!(s; tol = 1e-13)
    @assert r.converged && r.status === :converged
    F_stat = P.induced_forces(s, r; h = h)

    N = size(q, 2)
    F_relaxed = zeros(3, N)
    for j in 1:N, d in 1:3
        qp = copy(q); qp[d, j] += h
        qm = copy(q); qm[d, j] -= h
        sp = P.InducedDipoleSystem(qp, z, α, T.NoBox(), T.REDUCED; thole = true, a_thole = 1.2)
        sm = P.InducedDipoleSystem(qm, z, α, T.NoBox(), T.REDUCED; thole = true, a_thole = 1.2)
        rp = P.solve_dipoles!(sp; tol = 1e-13)
        rm = P.solve_dipoles!(sm; tol = 1e-13)
        Up = P.induced_energy(sp, rp)
        Um = P.induced_energy(sm, rm)
        F_relaxed[d, j] = -(Up - Um) / (2.0 * h)
    end
    err = maximum(abs, F_stat .- F_relaxed)
    @assert err < 1e-5 "Polarizable envelope force consistency failed: max |F_stat - F_relaxed| = $err"
    println("  Polarizable force consistency with damping vs relaxed FD: error=$err < 1e-5: PASS")

    # 2. Test via PolarizationOperator directly (with damping)
    A = PO.build_operator(q, z, α; thole = true, a_thole = 1.2)
    r_po = PO.solve_dipoles!(A; tol = 1e-13)
    @assert r_po.status === :converged
    F_po_stat = PO.induced_forces(A, r_po; h = h)

    F_po_relaxed = zeros(3, N)
    for j in 1:N, d in 1:3
        qp = copy(q); qp[d, j] += h
        qm = copy(q); qm[d, j] -= h
        Ap = PO.build_operator(qp, z, α; thole = true, a_thole = 1.2)
        Am = PO.build_operator(qm, z, α; thole = true, a_thole = 1.2)
        rp = PO.solve_dipoles!(Ap; tol = 1e-13)
        rm = PO.solve_dipoles!(Am; tol = 1e-13)
        Up = PO.induced_energy(Ap, rp)
        Um = PO.induced_energy(Am, rm)
        F_po_relaxed[d, j] = -(Up - Um) / (2.0 * h)
    end
    err_po = maximum(abs, F_po_stat .- F_po_relaxed)
    @assert err_po < 1e-5 "PolarizationOperator force consistency failed: max |F_po_stat - F_po_relaxed| = $err_po"
    println("  PolarizationOperator force consistency with damping vs relaxed FD: error=$err_po < 1e-5: PASS")
end

function test_periodic_polarization_modes_gated()
    q = [0.0 2.0; 0 0; 0 0]
    z = [1.0, -1.0]
    α = [0.1, 0.1]
    ortho_box = T.OrthorhombicBox((10.0, 10.0, 10.0))
    tri_box = T.TriclinicBox([10.0 0 0; 0 10.0 0; 0 0 10.0])

    # PolarizationOperator gates
    for box in (ortho_box, tri_box)
        failed_po = false
        try
            PO.build_operator(q, z, α, box)
        catch e
            failed_po = e isa ArgumentError
        end
        @assert failed_po "PolarizationOperator must refuse periodic box $(typeof(box)) with ArgumentError"
    end

    # Polarizable InducedDipoleSystem gates
    for box in (ortho_box, tri_box)
        failed_p = false
        try
            P.InducedDipoleSystem(q, z, α, box, T.REDUCED)
        catch e
            failed_p = e isa T.UnsupportedGeometry
        end
        @assert failed_p "Polarizable must refuse periodic box $(typeof(box)) with UnsupportedGeometry"
    end
    println("  Unsupported periodic polarization modes explicitly gated/refused: PASS")
end

function test_analytic_and_periodic_polarization_operator_forces()
    q = [0.2  1.8 -0.5  0.3;
         0.1 -0.4  1.5 -1.2;
        -0.3  0.7  0.2  1.1]
    z = [0.8, -0.4, -0.6, 0.2]
    α = [0.4, 0.3, 0.0, 0.5] # site 3 has alpha=0

    boxes = [
        ("NoBox", T.NoBox(), false),
        ("OrthorhombicBox", T.OrthorhombicBox((10.0, 10.0, 10.0)), true),
        ("TriclinicBox", T.TriclinicBox([10.0 1.0 0.5; 0.0 10.0 0.8; 0.0 0.0 10.0]), true)
    ]

    for (bname, box, periodic) in boxes
        for thole in [false, true]
            A = PO.build_operator(q, z, α, box; thole = thole, a_thole = 1.2, periodic = periodic)
            r = PO.solve_dipoles!(A; tol = 1e-13)
            @assert r.status === :converged
            F_ana = PO.analytic_induced_forces(A, r)
            F_fd = PO.induced_forces(A, r; h = 1e-6, analytic = false)
            err = maximum(abs, F_ana .- F_fd)
            @assert err < 1e-8 "Analytic forces differ from FD on $bname (thole=$thole): $err >= 1e-8"
            # Test analytic=true kwarg in induced_forces
            F_kw = PO.induced_forces(A, r; analytic = true)
            @assert F_kw == F_ana
        end
    end
    println("  PolarizationOperator analytic vs FD forces across boxes (active+inactive, thole on/off): PASS")
end

function test_phase5_dual_rail_ternary_preconditioner()
    @testset "Phase 5: Dual-Rail Ternary Preconditioned Mutual Polarization" begin
        # 1. Compression ratio > 10x for N >= 150 polarizable sites (M = 450 DOFs) vs dense Float64
        N = 150
        M = 3 * N
        q_150 = rand(3, N) .* 15.0
        z_150 = randn(N)
        α_150 = fill(0.4, N)
        op_150 = PO.build_operator(q_150, z_150, α_150; thole=true, a_thole=1.2)

        prec_2p = PO.build_ternary_preconditioner(op_150; num_planes=2)
        ratio_2p = PO.ternary_compression_ratio(prec_2p)
        @test ratio_2p > 10.0 # ~12.86x compression

        prec_3p = PO.build_ternary_preconditioner(op_150; num_planes=3)
        ratio_3p = PO.ternary_compression_ratio(prec_3p)
        @test ratio_3p > 8.0 # ~8.65x total including scales & inv_diag

        bitplane_bytes = sizeof(prec_3p.W_P) + sizeof(prec_3p.W_N)
        dense_bytes = M * M * 8
        @test dense_bytes / bitplane_bytes >= 9.0

        # Larger N = 250 (M = 750 DOFs) with 3 planes
        N_250 = 250
        M_250 = 3 * N_250
        q_250 = rand(3, N_250) .* 25.0
        z_250 = randn(N_250)
        α_250 = fill(0.4, N_250)
        op_250 = PO.build_operator(q_250, z_250, α_250; thole=true, a_thole=1.2)
        prec_250 = PO.build_ternary_preconditioner(op_250; num_planes=3)
        @test (M_250 * M_250 * 8) / (sizeof(prec_250.W_P) + sizeof(prec_250.W_N)) > 10.0

        # 2. Parity (< 1e-10) between solve_induced_dipoles_ternary_pcg and solve_induced_dipoles
        q_np = zeros(3, 30)
        for i in 1:30
            q_np[1, i] = 2.5 * (i % 5)
            q_np[2, i] = 2.5 * ((i ÷ 5) % 3)
            q_np[3, i] = 2.5 * (i ÷ 15)
        end
        z_np = (i -> isodd(i) ? 0.7 : -0.7).(1:30)
        α_np = fill(0.35, 30)
        op_np = PO.build_operator(q_np, z_np, α_np; thole=true, a_thole=1.2)

        r_std_np = PO.solve_induced_dipoles(op_np; tol=1e-11)
        r_ter_np = PO.solve_induced_dipoles_ternary_pcg(op_np; tol=1e-11)
        @test r_std_np.status === :converged
        @test r_ter_np.status === :converged
        @test norm(r_ter_np.mu .- r_std_np.mu) < 1e-10

        # Periodic boxes: Orthorhombic & Triclinic
        box_ortho = T.OrthorhombicBox((15.0, 15.0, 15.0))
        op_p_ortho = PO.build_operator(q_np, z_np, α_np, box_ortho; thole=true, a_thole=1.2, periodic=true)
        r_std_po = PO.solve_induced_dipoles(op_p_ortho; tol=1e-11)
        r_ter_po = PO.solve_induced_dipoles_ternary_pcg(op_p_ortho; tol=1e-11)
        @test r_std_po.status === :converged
        @test r_ter_po.status === :converged
        @test norm(r_ter_po.mu .- r_std_po.mu) < 1e-10

        box_tri = T.TriclinicBox([15.0 1.0 0.5; 0.0 15.0 0.8; 0.0 0.0 15.0])
        op_p_tri = PO.build_operator(q_np, z_np, α_np, box_tri; thole=true, a_thole=1.2, periodic=true)
        r_std_pt = PO.solve_induced_dipoles(op_p_tri; tol=1e-11)
        r_ter_pt = PO.solve_induced_dipoles_ternary_pcg(op_p_tri; tol=1e-11)
        @test r_std_pt.status === :converged
        @test r_ter_pt.status === :converged
        @test norm(r_ter_pt.mu .- r_std_pt.mu) < 1e-10

        # External field: matrix & vector forms
        E_mat = randn(3, 30)
        r_std_mat = PO.solve_induced_dipoles(op_np, E_mat; tol=1e-11)
        r_ter_mat = PO.solve_induced_dipoles_ternary_pcg(op_np, E_mat; tol=1e-11)
        @test norm(r_ter_mat.mu .- r_std_mat.mu) < 1e-10

        E_vec = randn(3 * 30)
        r_std_vec = PO.solve_induced_dipoles(op_np, E_vec; tol=1e-11)
        r_ter_vec = PO.solve_induced_dipoles_ternary_pcg(op_np, E_vec; tol=1e-11)
        @test norm(r_ter_vec.mu .- r_std_vec.mu) < 1e-10

        # Preconditioner reuse
        prec_reuse = PO.build_ternary_preconditioner(op_np; num_planes=3)
        r_reuse = PO.solve_induced_dipoles_ternary_pcg(op_np, E_vec; tol=1e-11, prec=prec_reuse)
        @test norm(r_reuse.mu .- r_ter_vec.mu) < 1e-14

        # Destructuring support
        (mu_destruct, stats_destruct) = r_reuse
        @test mu_destruct == r_reuse.mu
        @test stats_destruct.iterations == r_reuse.iterations

        # 3. Strongly coupled polarizable cluster: fewer or equal iterations and analytic force match (< 1e-8)
        q_tight = zeros(3, 50)
        for i in 1:50
            q_tight[1, i] = 2.4 * (i % 5)
            q_tight[2, i] = 2.4 * ((i ÷ 5) % 5)
            q_tight[3, i] = 2.4 * (i ÷ 25)
        end
        z_tight = (i -> isodd(i) ? 0.8 : -0.8).(1:50)
        α_tight = fill(0.4, 50)
        op_tight = PO.build_operator(q_tight, z_tight, α_tight; thole=true, a_thole=1.2)

        r_std_tight = PO.solve_induced_dipoles(op_tight; tol=1e-11)
        r_ter_tight = PO.solve_induced_dipoles_ternary_pcg(op_tight; tol=1e-11, num_planes=3)
        r_b243_tight = PO.solve_induced_dipoles_ternary_pcg(op_tight; tol=1e-11, num_planes=5) # Base-243

        println("  Tight cluster iterations: Jacobi=", r_std_tight.iterations, " vs Ternary(3-plane)=", r_ter_tight.iterations, " vs Base-243(5-plane)=", r_b243_tight.iterations)
        @test r_ter_tight.iterations <= r_std_tight.iterations
        @test r_b243_tight.iterations <= r_std_tight.iterations
        @test norm(r_ter_tight.mu .- r_std_tight.mu) < 1e-10
        @test norm(r_b243_tight.mu .- r_std_tight.mu) < 1e-10

        # Analytic forces match central FD (< 1e-8)
        F_ana_t = PO.analytic_induced_forces(op_tight, r_ter_tight)
        F_fd_t = PO.induced_forces(op_tight, r_ter_tight; h=1e-6, analytic=false)
        f_err = maximum(abs, F_ana_t .- F_fd_t)
        println("  Analytic vs FD induced force error on Ternary PCG solution: ", f_err)
        @test f_err < 1e-8
    end
    println("  Phase 5: Dual-Rail Ternary Preconditioned Mutual Polarization: PASS")
end

function test_phase8_qc_gray_4word_lt_spd()
    @testset "Phase 8: QC Block Preconditioner, 4-Word Matvec, Gray-Code Sweep & Bloch LT SPD" begin
        # 1. N=512 O(M^2) preconditioner build, 0-alloc 4-word matvec, 7-arg compat, LT SPD cert
        q_512 = zeros(3, 512)
        idx = 0
        for iz in 0:7, iy in 0:7, ix in 0:7
            idx += 1
            q_512[1, idx] = 0.28 * ix
            q_512[2, idx] = 0.28 * iy
            q_512[3, idx] = 0.28 * iz
        end
        z_512 = [isodd(i) ? 0.6 : -0.6 for i in 1:512]
        α_512 = fill(0.001, 512)
        op_512 = PO.build_operator(q_512, z_512, α_512; thole=true, a_thole=1.2)

        PO.build_ternary_preconditioner(op_512; num_planes=3) # warmup
        t0 = time_ns()
        prec_512 = PO.build_ternary_preconditioner(op_512; num_planes=3)
        build_ms = (time_ns() - t0) / 1e6
        @test prec_512.n_atoms == 512
        @test prec_512.n_words == cld(3 * 512, 64)
        @test build_ms < 500.0

        # 7-arg backward-compatible constructor
        prec_compat = PO.TernaryDipolePreconditioner(
            prec_512.n_atoms, prec_512.num_planes, prec_512.n_words,
            prec_512.inv_diag, prec_512.W_P, prec_512.W_N, prec_512.row_scales
        )
        x_512 = sin.(collect(1:(3 * 512)) .* 0.17)
        y_512 = zeros(3 * 512)
        y_compat = zeros(3 * 512)
        PO.ternary_bitplane_matvec!(y_512, prec_512, x_512; Kx=3)
        PO.ternary_bitplane_matvec!(y_compat, prec_compat, x_512; Kx=3)
        @test y_512 == y_compat

        # 0-allocation warmed matvec (both default Kx=3 and resized Kx=10)
        alloc_kx3 = @allocated PO.ternary_bitplane_matvec!(y_512, prec_512, x_512; Kx=3)
        @test alloc_kx3 == 0
        PO.ternary_bitplane_matvec!(y_512, prec_512, x_512; Kx=10)
        alloc_kx10 = @allocated PO.ternary_bitplane_matvec!(y_512, prec_512, x_512; Kx=10)
        @test alloc_kx10 == 0
        println("  N=512 build=", round(build_ms; digits=2), " ms, ternary_bitplane_matvec! alloc=", alloc_kx3, " B")

        # Exact O(N^2) Bloch Luttinger-Tisza SPD certificate on N=512
        lt_spd, lt_bound = PO.certify_dipole_spd_luttinger_tisza(op_512)
        @test lt_spd === :certified
        @test isfinite(lt_bound) && lt_bound > 0.0

        r_512_ter = PO.solve_induced_dipoles_ternary_pcg(op_512; tol=1e-11)
        r_512_jac = PO.solve_dipoles!(op_512; tol=1e-11)
        @test r_512_ter.status === :converged && r_512_ter.spd === :certified
        @test r_512_jac.status === :converged && r_512_jac.spd === :certified
        @test isapprox(r_512_ter.min_eigenvalue, lt_bound; rtol=1e-14)
        @test norm(r_512_ter.mu .- r_512_jac.mu) < 1e-10

        # Lower-bound soundness on small system & conservative :not_certified on strong/indefinite coupling
        q_sm = [0.0 2.5 5.0; 0.0 0.0 0.0; 0.0 0.0 0.0]
        op_sm = PO.build_operator(q_sm, [1.0, -1.0, 0.5], [0.2, 0.2, 0.2])
        sm_spd, sm_bound = PO.certify_dipole_spd_luttinger_tisza(op_sm)
        @test sm_spd === :certified
        @test sm_bound <= minimum(eigvals(Symmetric(PO.dense_reference(op_sm)))) + 1e-12

        op_indef = PO.build_operator([0.0 0.01; 0.0 0.0; 0.0 0.0], [0.0, 0.0], [1.0, 1.0])
        indef_spd, indef_bound = PO.certify_dipole_spd_luttinger_tisza(op_indef)
        @test indef_spd === :not_certified
        @test indef_bound <= 0.0

        # 2. Gray-code incremental bitplane sweep vs exact direct bitplane matvec at every state
        q_g = [0.0 1.9 3.8 0.5 2.4;
               0.0 0.2 -0.3 2.1 2.0;
               0.0 0.1 0.4 -0.2 0.3]
        op_g = PO.build_operator(q_g, [0.5, -0.5, 0.4, -0.4, 0.2], fill(0.3, 5); thole=true, a_thole=1.2)
        prec_g = PO.build_ternary_preconditioner(op_g; num_planes=3)
        M_g = 3 * prec_g.n_atoms
        base_P = zeros(UInt64, prec_g.n_words)
        base_N = zeros(UInt64, prec_g.n_words)
        # Set signed pattern on cols 1:10 and leave cols 11:15 zero to test both flip modes
        for col in 1:10
            if isodd(col)
                base_P[1] |= UInt64(1) << (col - 1)
            else
                base_N[1] |= UInt64(1) << (col - 1)
            end
        end
        flip_sites = [1, 4, 7, 11, 14] # 2^5 = 32 Gray states
        max_step_err = 0.0
        direct_norms = Float64[]
        direct_grays = UInt64[]

        function check_gray_step(step_i, gray, y_inc, cur_P, cur_N)
            # Direct full matvec from (cur_P, cur_N)
            y_dir = zeros(Float64, M_g)
            for row in 1:M_g, p in 1:prec_g.num_planes
                sp = prec_g.row_scales[row, p]
                sp == 0.0 && continue
                db = 0
                for w in 1:prec_g.n_words
                    wp = prec_g.W_P[row, w, p]; wn = prec_g.W_N[row, w, p]
                    db += Int(count_ones((wp & cur_P[w]) | (wn & cur_N[w]))) -
                          Int(count_ones((wp & cur_N[w]) | (wn & cur_P[w])))
                end
                y_dir[row] += sp * db
            end
            err = maximum(abs, y_inc .- y_dir)
            err > max_step_err && (max_step_err = err)
            push!(direct_norms, norm(y_dir))
            push!(direct_grays, gray)
        end

        best_min_n, best_min_g, n_evals_min = PO.gray_code_bitplane_sweep(
            prec_g, base_P, base_N, flip_sites; objective=:min, on_step=check_gray_step
        )
        @test max_step_err < 1e-12
        @test n_evals_min == 32
        min_idx = argmin(direct_norms)
        @test isapprox(best_min_n, direct_norms[min_idx]; atol=1e-12)
        @test best_min_g == direct_grays[min_idx]

        best_max_n, best_max_g, n_evals_max = PO.gray_code_bitplane_sweep(
            prec_g, base_P, base_N, flip_sites; objective=:max
        )
        max_idx = argmax(direct_norms)
        @test n_evals_max == 32
        @test isapprox(best_max_n, direct_norms[max_idx]; atol=1e-12)
        @test best_max_g == direct_grays[max_idx]

        # 3. QuasiCyclicBlockPreconditioner: iteration reduction & parity < 1e-10
        q_cl = zeros(3, 30)
        for b in 0:9
            cx = 4.0 * (b % 3)
            cy = 4.0 * ((b ÷ 3) % 3)
            cz = 4.0 * (b ÷ 9)
            q_cl[:, 3b + 1] = [cx, cy, cz]
            q_cl[:, 3b + 2] = [cx + 0.95, cy, cz]
            q_cl[:, 3b + 3] = [cx, cy + 0.95, cz]
        end
        z_cl = [i % 3 == 1 ? -0.8 : 0.4 for i in 1:30]
        α_cl = fill(0.18, 30)
        op_cl = PO.build_operator(q_cl, z_cl, α_cl; thole=true, a_thole=1.2)

        r_jac_cl = PO.solve_induced_dipoles(op_cl; tol=1e-11)
        r_ter_cl = PO.solve_induced_dipoles_ternary_pcg(op_cl; tol=1e-11, precond_mode=:ternary)
        r_qc_cl = PO.solve_induced_dipoles_ternary_pcg(op_cl; tol=1e-11, precond_mode=:qc_hybrid, block_sites=3)

        println("  QC cluster iterations: Jacobi=", r_jac_cl.iterations,
                " Ternary=", r_ter_cl.iterations, " QC_Hybrid=", r_qc_cl.iterations)
        @test r_qc_cl.status === :converged
        @test r_qc_cl.iterations < r_jac_cl.iterations
        @test r_qc_cl.iterations < r_ter_cl.iterations
        @test norm(r_qc_cl.mu .- r_jac_cl.mu) < 1e-10

        # Preconditioner reuse + 0-alloc apply_qc_ternary_precond!
        qc_prec = PO.build_qc_block_preconditioner(op_cl; block_sites=3, num_planes=3)
        r_qc_reuse = PO.solve_induced_dipoles_ternary_pcg(op_cl; tol=1e-11, prec=qc_prec)
        @test norm(r_qc_reuse.mu .- r_qc_cl.mu) < 1e-14

        z_tmp = zeros(3 * 30); r_vec = randn(3 * 30); scratch_tmp = zeros(3 * 30)
        PO.apply_qc_ternary_precond!(z_tmp, qc_prec, r_vec, scratch_tmp)
        alloc_qc = @allocated PO.apply_qc_ternary_precond!(z_tmp, qc_prec, r_vec, scratch_tmp)
        @test alloc_qc == 0

        # Periodic OrthorhombicBox QC hybrid parity
        box_ortho = T.OrthorhombicBox((15.0, 15.0, 15.0))
        op_cl_p = PO.build_operator(q_cl, z_cl, α_cl, box_ortho; thole=true, a_thole=1.2, periodic=true)
        r_jac_p = PO.solve_induced_dipoles(op_cl_p; tol=1e-11)
        r_qc_p = PO.solve_induced_dipoles_ternary_pcg(op_cl_p; tol=1e-11, precond_mode=:qc_hybrid, block_sites=3)
        @test r_qc_p.status === :converged
        @test norm(r_qc_p.mu .- r_jac_p.mu) < 1e-10
    end
    println("  Phase 8: QC Block Preconditioner, 4-Word Matvec, Gray-Code Sweep & Bloch LT SPD: PASS")
end

function test_cw_orthogonal_noise_and_block_filter()
    @testset "CW(n, k) Orthogonal Noise & Multiplication-Free Block Filter" begin
        rng = MersenneTwister(20261118)

        for (n_blk, k_blk) in ((13, 9), (7, 4))
            expected_mag = sqrt(Float64(n_blk) / Float64(k_blk))
            Z1 = PO.cw_orthogonal_noise(rng, n_blk; n_block=n_blk)
            @test size(Z1) == (n_blk, n_blk)
            # Strictly ternary {-1, 0, +1} weights scaled by sqrt(n_block / k)
            @test all(v -> v == 0.0 || isapprox(abs(v), expected_mag; atol=1e-14), Z1)
            # Exact row orthogonality: Z1 * Z1' == n_blk * I (equiv. W * W' / k == I)
            @test norm((Z1 * Z1') ./ n_blk - I) < 1e-14
            @test norm((Z1' * Z1) ./ n_blk - I) < 1e-14

            # Multiple blocks (n_cols = 3 * n_blk)
            Z3 = PO.cw_orthogonal_noise(rng, 3 * n_blk; n_block=n_blk)
            @test norm((Z3 * Z3') ./ (3 * n_blk) - I) < 1e-14

            # Empirical isotropic covariance E[z * z'] == I across independent draws
            cov_est = zeros(n_blk, n_blk)
            n_trials = 2500
            for _ in 1:n_trials
                z = PO.cw_orthogonal_noise(rng, 1; n_block=n_blk)[:, 1]
                cov_est .+= z * z'
            end
            cov_est ./= n_trials
            @test maximum(abs, cov_est - I) < 0.08

            # Empirical trace-estimation variance reduction vs independent standard normal
            A_raw = randn(rng, n_blk, n_blk)
            A_sym = 0.5 .* (A_raw .+ A_raw')
            true_tr = tr(A_sym)

            # Full-block Hutchinson trace has zero variance (< 1e-26)
            tr_full = sum(dot(Z1[:, c], A_sym * Z1[:, c]) for c in 1:n_blk) / n_blk
            @test abs(tr_full - true_tr) < 1e-12

            # Partial-block Hutchinson (n_cols < n_blk) still beats standard normal variance
            m_partial = max(3, n_blk ÷ 2)
            est_cw = Float64[]
            est_gauss = Float64[]
            for _ in 1:400
                Zc = PO.cw_orthogonal_noise(rng, m_partial; n_block=n_blk)
                Zg = randn(rng, n_blk, m_partial)
                push!(est_cw, sum(dot(Zc[:, c], A_sym * Zc[:, c]) for c in 1:m_partial) / m_partial)
                push!(est_gauss, sum(dot(Zg[:, c], A_sym * Zg[:, c]) for c in 1:m_partial) / m_partial)
            end
            @test var(est_cw) < 0.75 * var(est_gauss)
            println("  CW(", n_blk, ",", k_blk, ") trace var (m=", m_partial, "): cw=",
                    round(var(est_cw); digits=3), " < gauss=", round(var(est_gauss); digits=3))
        end

        # apply_cw_block_filter! inversion parity < 1e-14 for exact & non-multiple lengths
        for n_blk in (13, 7)
            for len in (n_blk, 2 * n_blk, n_blk + 5, 3 * n_blk + 2, 1)
                x = randn(rng, len)
                y = zeros(Float64, len)
                PO.apply_cw_block_filter!(y, x; n_block=n_blk)
                @test maximum(abs, y .- x) < 1e-14
            end

            # Separate :forward then :inverse on full-block length
            x_full = randn(rng, 2 * n_blk)
            z_fwd = zeros(Float64, 2 * n_blk)
            y_inv = zeros(Float64, 2 * n_blk)
            PO.apply_cw_block_filter!(z_fwd, x_full; n_block=n_blk, mode=:forward)
            PO.apply_cw_block_filter!(y_inv, z_fwd; n_block=n_blk, mode=:inverse)
            @test maximum(abs, y_inv .- x_full) < 1e-14
        end

        # Guardrails
        @test_throws ArgumentError PO.cw_orthogonal_noise(rng, 0; n_block=13)
        @test_throws ArgumentError PO.cw_orthogonal_noise(rng, 5; n_block=5)
        buf = zeros(13)
        @test_throws ArgumentError PO.apply_cw_block_filter!(buf, buf; n_block=13)
        @test_throws ArgumentError PO.apply_cw_block_filter!(zeros(12), zeros(13); n_block=13)
        @test_throws ArgumentError PO.apply_cw_block_filter!(zeros(13), zeros(13); n_block=5)
        @test_throws ArgumentError PO.apply_cw_block_filter!(zeros(13), zeros(13); n_block=13, mode=:bogus)
    end
    println("  CW(n, k) Orthogonal Noise & Multiplication-Free Block Filter: PASS")
end

test_two_site_sign_and_dense_match()
test_active_sites_and_matrix_free_storage()
test_failures_are_not_usable()
test_numeric_edge_guards()
test_convergence_status_and_residual()
test_force_consistency_against_fd_with_damping()
test_periodic_polarization_modes_gated()
test_analytic_and_periodic_polarization_operator_forces()
test_phase5_dual_rail_ternary_preconditioner()
test_phase8_qc_gray_4word_lt_spd()
test_cw_orthogonal_noise_and_block_filter()
println("POLARIZATION OPERATOR TESTS PASSED")
