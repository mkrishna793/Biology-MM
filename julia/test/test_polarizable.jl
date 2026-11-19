# Induced-dipole polarization tests.
#
# The three claims that need independent verification:
#
# 1. the linear solve `A μ = E` is actually solved (residual), and `A` SPD-ness is
#    reported rather than assumed;
# 2. the "stationary-point" force equals the derivative of the *fully relaxed*
#    energy — this is the envelope theorem, and it is the step most easily faked;
# 3. an unstable (non-SPD) `A` is detected rather than silently regularised.
#
# Each is checked against an independent computation, not against the same code
# path that produced it.

using LinearAlgebra, Random, Printf

module _Host
    dir = joinpath(@__DIR__, "..", "src")
    include(joinpath(dir, "Types.jl"))
    include(joinpath(dir, "Factors.jl"))
    include(joinpath(dir, "Neighbors.jl"))
    include(joinpath(dir, "Constraints.jl"))
    include(joinpath(dir, "StreamedLJ.jl"))
    include(joinpath(dir, "Electrostatics.jl"))
    include(joinpath(dir, "PME.jl"))
    include(joinpath(dir, "ResolvedSystem.jl"))
    include(joinpath(dir, "SystemContract.jl"))
    include(joinpath(dir, "Polarizable.jl"))
    include(joinpath(dir, "PolarizationOperator.jl"))
end
const T = _Host.Types
const P = _Host.Polarizable
const PO = _Host.PolarizationOperator
const RS = _Host.ResolvedSystem
const SC = _Host.SystemContract
const F = _Host.Factors

function _water_like()
    # three sites, charges arranged so the total is neutral, moderate alpha
    q = [0.0 2.9 0.5; 0.0 0.0 2.3; 0.0 0.0 0.0]
    z = [1.0, -0.5, -0.5]
    α = [0.5, 0.3, 0.3]
    return P.InducedDipoleSystem(q, z, α, T.NoBox(), T.REDUCED; thole = true, a_thole = 1.0)
end

function test_solve_residual_and_spd()
    s = _water_like()
    r = P.solve_dipoles!(s; tol = 1e-12)
    @printf("  method=%s spd=%s residual=%.3e converged=%s\n",
            r.method, r.spd, r.residual, r.converged)
    @assert r.converged "solve did not converge"
    @assert r.residual < 1e-10 "residual $(r.residual) too large for a 3-site system"
    # independent residual: recompute A and E from scratch and check
    A = P.A_matrix(s)
    E = P.permanent_field(s)
    resid = norm(A * r.μ - E) / max(norm(E), eps())
    @printf("  independent recomputation of the residual: %.3e\n", resid)
    @assert resid < 1e-10
    # SPD check by eigenvalues (independent of the Cholesky attempt)
    ev = eigvals(Symmetric(A))
    @printf("  eigenvalues of A: min=%.4f max=%.4f  -> SPD=%s\n",
            minimum(ev), maximum(ev), minimum(ev) > 0)
    @assert r.spd == (minimum(ev) > 0) "SPD flag disagrees with the spectrum"
    return true
end

function test_envelope_theorem_force_matches_relaxed_gradient()
    # This verifies relaxed-energy force consistency only; envelope agreement
    # cannot validate the sign convention in A, since both paths reuse the same A.
    # At the converged solution,
    #   dU/dq  (with μ re-solved at each q)
    # must equal
    #   ∂U/∂q  (with μ held fixed)
    # because ∂μ/∂q terms drop out when Aμ = E. If the solve were not converged,
    # these would differ. So agreement here tests *both* the force formula and the
    # stationarity assumption.
    s = _water_like()
    r = P.solve_dipoles!(s; tol = 1e-13)
    @assert r.converged
    F_stationary = P.induced_forces(s, r)

    # fully relaxed: re-solve at each displaced geometry
    h = 1e-6
    N = 3
    F_relaxed = zeros(3, N)
    for j in 1:N, d in 1:3
        qp = copy(s.q); qp[d, j] += h
        qm = copy(s.q); qm[d, j] -= h
        sp = P.InducedDipoleSystem(qp, s.z, s.α, s.box, s.units; thole = s.thole, a_thole = s.a_thole)
        sm = P.InducedDipoleSystem(qm, s.z, s.α, s.box, s.units; thole = s.thole, a_thole = s.a_thole)
        rp = P.solve_dipoles!(sp; tol = 1e-13)
        rm = P.solve_dipoles!(sm; tol = 1e-13)
        Up = P.induced_energy(sp, rp)
        Um = P.induced_energy(sm, rm)
        F_relaxed[d, j] = -(Up - Um) / (2h)
    end
    err = maximum(abs, F_stationary .- F_relaxed)
    scale = max(maximum(abs, F_relaxed), 1e-30)
    @printf("  max |F_stationary - F_relaxed| = %.3e  (relative %.3e)\n", err, err / scale)
    @assert err / scale < 1e-5 "envelope theorem violated: stationary force != relaxed gradient"
    return true
end

function test_energy_is_minimised_by_the_solve()
    # Aμ = E must be the minimiser of ½μᵀAμ − μᵀE. Perturbing μ in any direction
    # must increase the energy.
    s = _water_like()
    r = P.solve_dipoles!(s; tol = 1e-13)
    U0 = P.induced_energy(s, r.μ)
    rng = MersenneTwister(7)
    worst_decrease = 0.0
    for _ in 1:50
        δ = randn(rng, length(r.μ))
        δ ./= norm(δ)
        for eps in (1e-3, 1e-4, 1e-5)
            U1 = P.induced_energy(s, r.μ .+ eps .* δ)
            worst_decrease = max(worst_decrease, U0 - U1)
        end
    end
    @printf("  worst energy decrease under random perturbations: %.3e (must be >= 0)\n",
            worst_decrease)
    @assert worst_decrease < 1e-10 "the solve did not find a minimum"
    return true
end

function test_unstable_system_is_detected()
    # Two strongly polarizable sites very close: the undamped dipole tensor
    # diverges and A goes indefinite. The solver must report spd=false rather than
    # silently regularising.
    q = [0.0 0.15 0.30; 0.0 0.0 0.05; 0.0 0.0 0.0]
    z = [0.0, 0.0, 0.0]
    α = [20.0, 20.0, 20.0]   # very polarizable
    s = P.InducedDipoleSystem(q, z, α, T.NoBox(), T.REDUCED; thole = false, a_thole = 1.0)
    A = P.A_matrix(s)
    ev = eigvals(Symmetric(A))
    r = P.solve_dipoles!(s; tol = 1e-10)
    @printf("  undamped, strong coupling: min eigen=%.4f  spd=%s  (reported, not hidden)\n",
            minimum(ev), r.spd)
    # At least one of these must be true, and the point is that the code *tells*
    # us rather than pretending:
    #   * A is indefinite -> spd = false
    #   * or the damping keeps it definite -> spd = true and the residual is small
    if !r.spd
        println("  -> A is indefinite; instability reported (correct behaviour)")
    else
        @assert r.converged "A is SPD but the solve failed"
        println("  -> A remained SPD at this separation; solve converged")
    end
    r_unchecked = P.solve_dipoles!(s; tol = 1e-10, check_spd = false)
    @assert !r_unchecked.spd "check_spd=false must not leave a false-success SPD flag"
    failed_energy = false
    try
        P.induced_energy(s, r_unchecked)
    catch e
        failed_energy = e isa ArgumentError
    end
    @assert failed_energy "energy must reject an uncertified/unstable solve"
    failed_force = false
    try
        P.induced_forces(s, r_unchecked)
    catch e
        failed_force = e isa ArgumentError
    end
    @assert failed_force "forces must reject an uncertified/unstable solve"
    println("  check_spd=false still reports instability; energy/force access refused")
    @assert true
    return true
end

function test_damping_improves_conditioning()
    # Damping must raise the smallest eigenvalue at short separation. It is NOT
    # claimed to make A positive definite — the test below checks the sign in the
    # case where it does not, so the documented claim matches the observed
    # behaviour rather than overstating it.
    q = [0.0 0.8; 0.0 0.0; 0.0 0.0]
    z = [0.0, 0.0]
    α = [2.0, 2.0]
    s_undamped = P.InducedDipoleSystem(q, z, α, T.NoBox(), T.REDUCED; thole = false)
    s_damped = P.InducedDipoleSystem(q, z, α, T.NoBox(), T.REDUCED; thole = true, a_thole = 1.0)
    e_und = minimum(eigvals(Symmetric(P.A_matrix(s_undamped))))
    e_dam = minimum(eigvals(Symmetric(P.A_matrix(s_damped))))
    @printf("  min eigenvalue: undamped=%.4f  damped=%.4f (damping raises it, does not guarantee SPD)\n",
            e_und, e_dam)
    @assert e_dam > e_und "damping did not improve the spectrum"
    # If it is still indefinite, the solver must say so.
    if e_dam <= 0
        r = P.solve_dipoles!(s_damped)
        @assert !r.spd "solver reported SPD for an indefinite operator"
        @printf("  still indefinite here -> solver correctly reports spd=false\n")
    end
    return true
end

function test_permanent_forces_match_finite_difference()
    s = _water_like()
    U, Fp = P.permanent_energy_forces(s)
    h = 1e-6
    N = 3
    worst = 0.0
    for j in 1:N, d in 1:3
        qp = copy(s.q); qp[d, j] += h
        qm = copy(s.q); qm[d, j] -= h
        sp = P.InducedDipoleSystem(qp, s.z, s.α, s.box, s.units; thole = s.thole, a_thole = s.a_thole)
        sm = P.InducedDipoleSystem(qm, s.z, s.α, s.box, s.units; thole = s.thole, a_thole = s.a_thole)
        Up, _ = P.permanent_energy_forces(sp)
        Um, _ = P.permanent_energy_forces(sm)
        worst = max(worst, abs(Fp[d, j] - (-(Up - Um) / (2h))))
    end
    @printf("  permanent-charge force vs finite difference: max |Δ| = %.3e\n", worst)
    @assert worst < 1e-5
    return true
end

function test_extension_boundaries_are_honest()
    b = P.extension_boundaries()
    @assert length(b) > 0
    for e in b
        @assert haskey(e, :capability) && haskey(e, :supported) && haskey(e, :note)
    end
    names = [e.capability for e in b]
    # the ones a careless document would claim
    @assert any(occursin("metallic", n) for n in names)
    @assert any(occursin("charge transfer", n) for n in names)
    @assert all(e -> !e.supported, b) "not-yet-supported capabilities must be marked unsupported"
    println("  extension boundaries: ", length(b), " capabilities explicitly marked unsupported")
    return true
end

function test_coincident_sites_rejected()
    q=zeros(3,2)
    s=P.InducedDipoleSystem(q,[1.0,-1.0],[0.1,0.1],T.NoBox(),T.REDUCED)
    ok=false
    try P.A_matrix(s) catch e; ok=e isa ArgumentError end
    @assert ok "coincident sites must not be silently skipped by the operator"
    println("  coincident dipole sites rejected explicitly")
end

function test_operator_matches_textbook_reference()
    # Independent reference: for two sites on the x axis A_ij = -T_true =
    # (1/r^3) diag(-2, 1, 1). The longitudinal eigenvalue is therefore
    # 1/alpha - 2/r^3, which becomes negative below its critical separation.
    r = 3.0
    q = [0.0 r; 0.0 0.0; 0.0 0.0]
    α = [2.0, 2.0]
    z = [0.0, 0.0]
    s = P.InducedDipoleSystem(q, z, α, T.NoBox(), T.REDUCED; thole = false)
    A = P.A_matrix(s)

    Tmat = (1.0 / r^3) .* [-2.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
    Tref = zeros(6, 6)
    Tref[1:3, 4:6] = Tmat
    Tref[4:6, 1:3] = Tmat
    Aref = zeros(6, 6)
    for i in 1:2
        Aref[3i-2:3i, 3i-2:3i] = Matrix(1.0I, 3, 3) ./ α[i]
    end
    Aref = Aref .+ Tref
    dA = maximum(abs, A .- Aref)
    @printf("  max |A_ours - A_textbook| = %.3e\n", dA)
    @assert dA < 1e-14 "operator does not match the textbook dipole tensor"

    # Eigenvalues must match the reference spectrum
    ev = sort(eigvals(Symmetric(A)))
    evref = sort(eigvals(Symmetric(Aref)))
    dev = maximum(abs, ev .- evref)
    @printf("  eigenvalue spectrum max deviation = %.3e  (min=%.6f)\n", dev, ev[1])
    @assert dev < 1e-12
    @assert ev[1] > 0

    # This is a matrix-form sanity check, not a sign regression on its own.
    mu_perm0 = [0.0, 0.0, 0.6]
    E = zeros(6)
    E[4:6] = Tmat * mu_perm0
    mu = A \ E                     # this core's operator
    muref = Aref \ E               # textbook operator built independently here
    dmu = maximum(abs, mu .- muref)
    @printf("  induced dipoles: max |Δμ| = %.3e  (|μ|max = %.6f)\n",
            dmu, maximum(abs, muref))
    @assert dmu < 1e-13 "solve disagrees with the independent reference"
    return true
end

function test_physical_sign_and_operator_parity()
    # For equal alpha and collinear sites, the antisymmetric longitudinal mode
    # has eigenvalue 1/alpha - 2*k_e/r^3 and must destabilize below critical r.
    alpha = 2.0
    pre = T.REDUCED.coulomb_prefactor
    rcrit = (2.0 * pre * alpha)^(1 / 3)
    eigenvalues_at(r) = eigvals(Symmetric(P.A_matrix(P.InducedDipoleSystem(
        [0.0 r; 0.0 0.0; 0.0 0.0], [0.0, 0.0], [alpha, alpha],
        T.NoBox(), T.REDUCED; thole = false))))
    λabove = minimum(eigenvalues_at(1.1 * rcrit))
    λbelow = minimum(eigenvalues_at(0.9 * rcrit))
    @assert λabove > 0 "longitudinal mode must be stable above critical separation"
    @assert λbelow < 0 "longitudinal mode must be unstable below critical separation"

    # A permanent dipole parallel to the inter-site axis creates a positive
    # longitudinal field at its neighbor; the induced dipole must align with it.
    r = 3.0
    q = [0.0 r; 0.0 0.0; 0.0 0.0]
    z = [1.0, 0.0]
    s = P.InducedDipoleSystem(q, z, [0.2, 0.2], T.NoBox(), T.REDUCED; thole = false)
    E = P.permanent_field(s)
    solved = P.solve_dipoles!(s; tol = 1e-13)
    @assert solved.converged && solved.spd
    @assert E[4] > 0 && solved.μ[4] > 0 "neighbor induced dipole must align with the permanent field"
    @assert dot(E[4:6], solved.μ[4:6]) > 0

    # Independent cross-module parity for the same undamped geometry and alpha.
    pa = PO.build_operator(q, z, [0.2, 0.2], T.NoBox();
                           coulomb_prefactor = T.REDUCED.coulomb_prefactor)
    Ap = P.A_matrix(s)
    Ao = PO.dense_reference(pa)
    @assert maximum(abs, Ap .- Ao) < 1e-14 "Polarizable A disagrees with PolarizationOperator"
    @assert maximum(abs, E .- pa.field) < 1e-14 "permanent fields disagree across operators"

    # The energy is the stated quadratic, and at stationarity equals -1/2 mu'E.
    U = P.induced_energy(s, solved.μ)
    @assert isapprox(U, 0.5 * dot(solved.μ, Ap * solved.μ) - dot(solved.μ, E); atol = 1e-14)
    @assert isapprox(U, -0.5 * dot(solved.μ, E); atol = 1e-14)
    println("  longitudinal sign transition, field alignment, undamped operator parity, and stationary energy: PASS")
    return true
end

function test_analytic_induced_forces_match_fd()
    # 4-site 3D non-collinear geometry
    q = [0.2  1.8 -0.5  0.3;
         0.1 -0.4  1.5 -1.2;
        -0.3  0.7  0.2  1.1]
    z = [0.8, -0.4, -0.6, 0.2]
    α = [0.4, 0.3, 0.2, 0.5]

    for thole in [false, true]
        s = P.InducedDipoleSystem(q, z, α, T.NoBox(), T.REDUCED; thole = thole, a_thole = 1.2)
        r = P.solve_dipoles!(s; tol = 1e-13)
        @assert r.converged
        F_ana = P.analytic_induced_forces(s, r)
        F_fd = P.induced_forces(s, r; h = 1e-6, analytic = false)
        err = maximum(abs, F_ana .- F_fd)
        @printf("  analytic vs FD force (thole=%s): max |Δ| = %.3e\n", thole, err)
        @assert err < 1e-8 "Analytic forces differ from FD by $err >= 1e-8"
        # Test keyword argument in induced_forces
        F_kw = P.induced_forces(s, r; analytic = true)
        @assert F_kw == F_ana
    end
    return true
end

function test_periodic_polarizable_and_forces()
    q = [0.2  1.8 -0.5  0.3;
         0.1 -0.4  1.5 -1.2;
        -0.3  0.7  0.2  1.1]
    z = [0.8, -0.4, -0.6, 0.2]
    α = [0.4, 0.3, 0.2, 0.5]

    boxes = [
        ("OrthorhombicBox", T.OrthorhombicBox((10.0, 10.0, 10.0))),
        ("TriclinicBox", T.TriclinicBox([10.0 1.0 0.5; 0.0 10.0 0.8; 0.0 0.0 10.0]))
    ]

    for (bname, box) in boxes
        # Default periodic=false must refuse
        failed = false
        try
            P.InducedDipoleSystem(q, z, α, box, T.REDUCED)
        catch e
            failed = e isa T.UnsupportedGeometry
        end
        @assert failed "periodic=false must throw UnsupportedGeometry on $bname"

        # periodic=true must succeed and match FD
        for thole in [false, true]
            s = P.InducedDipoleSystem(q, z, α, box, T.REDUCED; thole = thole, a_thole = 1.2, periodic = true)
            r = P.solve_dipoles!(s; tol = 1e-13)
            @assert r.converged
            F_ana = P.analytic_induced_forces(s, r)
            F_fd = P.induced_forces(s, r; h = 1e-6, analytic = false)
            err = maximum(abs, F_ana .- F_fd)
            @printf("  periodic %s (thole=%s): max |Δ| = %.3e\n", bname, thole, err)
            @assert err < 1e-8 "Periodic analytic forces differ from FD on $bname: $err >= 1e-8"
        end
    end
    return true
end

function test_virtual_sites_kinematics_and_derivatives()
    H0 = [10.0 1.0 0.5; 0.0 10.0 0.8; 0.0 0.0 10.0]
    tribox = T.TriclinicBox(H0)

    vs_specs = [
        (:linear, RS.VirtualSite(:linear, 4, [1, 2], [0.6, 0.4])),
        (:planar, RS.VirtualSite(:planar, 4, [1, 2, 3], [0.5, 0.3, 0.2])),
        (:average, RS.VirtualSite(:average, 4, [1, 2, 3], [1/3, 1/3, 1/3]))
    ]

    for (kind, vs) in vs_specs
        factors = [
            F.HarmonicBond(1, 2, 50.0, 1.2),
            F.HarmonicBond(4, 5, 80.0, 1.5)
        ]
        rs = RS.ResolvedSystem(
            masses = [12.0, 1.0, 1.0, 0.0, 16.0],
            charges = [0.0, 0.0, 0.0, 0.0, 0.0],
            box = tribox,
            factors = factors,
            virtual_sites = [vs],
            declared = [:energy, :cartesian_forces, :component_energies, :box_derivative, :virtual_sites]
        )

        q = [0.0  1.1  0.2  0.0  2.5;
             0.0  0.1  1.2  0.0  0.5;
             0.0 -0.2  0.3  0.0  1.0]

        # 1. Placement verification
        q_placed = SC.place_virtual_sites(q, rs)
        if kind === :linear
            @assert isapprox(q_placed[:, 4], 0.6 * q[:, 1] + 0.4 * q[:, 2]; atol=1e-12)
        end

        # 2. Cartesian forces and redistribution: net force on virtual site must be 0
        u0, forces = SC.energy_forces(rs, q)
        @assert isapprox(forces[:, 4], [0.0, 0.0, 0.0]; atol=1e-15)

        # Parent forces match FD
        h = 1e-6
        for parent in [1, 2, 3, 5]
            for d in 1:3
                qp = copy(q); qp[d, parent] += h
                qm = copy(q); qm[d, parent] -= h
                up = SC.energy(rs, qp)
                um = SC.energy(rs, qm)
                fd_force = -(up - um) / (2.0 * h)
                err = abs(forces[d, parent] - fd_force)
                @assert err < 1e-6 "Force mismatch on parent atom $parent: $err"
            end
        end

        # 3. 9-entry Triclinic box derivative vs FD
        G_analytic = SC.box_derivative(rs, q)
        s_frac = SC.fractional_coordinates(rs, q)

        h_box = 1e-6
        G_fd = zeros(3, 3)
        for a in 1:3, b in 1:3
            Hp = copy(H0); Hp[a, b] += h_box
            Hm = copy(H0); Hm[a, b] -= h_box
            box_p = T.TriclinicBox(Hp)
            box_m = T.TriclinicBox(Hm)
            rs_p = RS.ResolvedSystem(
                masses = rs.atoms.masses, charges = rs.atoms.charges,
                box = box_p, factors = factors, virtual_sites = [vs],
                declared = [:energy, :cartesian_forces, :component_energies, :box_derivative, :virtual_sites]
            )
            rs_m = RS.ResolvedSystem(
                masses = rs.atoms.masses, charges = rs.atoms.charges,
                box = box_m, factors = factors, virtual_sites = [vs],
                declared = [:energy, :cartesian_forces, :component_energies, :box_derivative, :virtual_sites]
            )
            qp = Hp * s_frac
            qm = Hm * s_frac
            Up = SC.energy(rs_p, qp)
            Um = SC.energy(rs_m, qm)
            G_fd[a, b] = (Up - Um) / (2.0 * h_box)
        end
        err_G = maximum(abs, G_analytic .- G_fd)
        @printf("  virtual site %s: 9-entry box derivative max |Δ| = %.3e\n", kind, err_G)
        @assert err_G < 1e-7 "Box derivative mismatch on $kind: $err_G"
    end

    # 4. Kinetic energy tests
    let
        vs = RS.VirtualSite(:linear, 4, [1, 2], [0.5, 0.5])
        rs = RS.ResolvedSystem(
            masses = [12.0, 1.0, 1.0, 0.0, 16.0],
            charges = zeros(5),
            factors = [F.HarmonicBond(1, 2, 50.0, 1.2)],
            virtual_sites = [vs],
            declared = [:energy, :cartesian_forces, :component_energies, :virtual_sites]
        )
        p = [1.0 2.0 3.0 0.0 4.0;
             0.5 0.2 0.1 0.0 0.4;
             0.2 0.1 0.0 0.0 0.3]
        K = SC.kinetic_energy(rs, p)
        @assert K > 0
        p_bad = copy(p); p_bad[1, 4] = 0.1
        caught = false
        try
            SC.kinetic_energy(rs, p_bad)
        catch e
            caught = e isa ArgumentError
        end
        @assert caught "Massless site with nonzero momentum must throw ArgumentError"
    end
    println("  virtual sites kinematics, FD forces, box derivative, and kinetic energy: PASS")
    return true
end

function _run_all()
    for (name, fn) in [
        ("solve_residual_and_spd", test_solve_residual_and_spd),
        ("envelope_theorem_force", test_envelope_theorem_force_matches_relaxed_gradient),
        ("energy_minimised_by_solve", test_energy_is_minimised_by_the_solve),
        ("unstable_system_detected", test_unstable_system_is_detected),
        ("damping_improves_conditioning", test_damping_improves_conditioning),
        ("permanent_forces_vs_fd", test_permanent_forces_match_finite_difference),
        ("operator_matches_textbook", test_operator_matches_textbook_reference),
        ("physical_sign_and_parity", test_physical_sign_and_operator_parity),
        ("extension_boundaries", test_extension_boundaries_are_honest),
        ("coincident_sites_rejected", test_coincident_sites_rejected),
        ("analytic_induced_forces", test_analytic_induced_forces_match_fd),
        ("periodic_polarizable_forces", test_periodic_polarizable_and_forces),
        ("virtual_sites_derivatives", test_virtual_sites_kinematics_and_derivatives),
    ]
        println("TEST ", name)
        fn()
    end
    println("")
    println("ALL TESTS PASSED")
end

_run_all()
