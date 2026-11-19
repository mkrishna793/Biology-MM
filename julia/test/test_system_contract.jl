# Execution-contract tests: `SystemContract` against independent references.
#
# Run directly:
#
#     julia --project=julia julia/test/test_system_contract.jl
#
# Everything here is an assertion suite executed at include time, like the rest of
# the FoundationCore suites. No check compares the implementation to itself as its
# *only* evidence: forces, box derivatives and pressures are measured against
# central finite differences of `SystemContract.energy` (and of the per-component
# energies), and the one closed-form case (a harmonic bond) is compared to the
# analytic expression written out by hand below.
#
# Two formula errata are encoded here, because the interface text had typos and the
# derivatives are only meaningful in the corrected form:
#
#   1. `PeriodicDihedral` box derivative. With `w1, w2, w3` the fractional
#      displacements of `unwrap(j,i)`, `unwrap(k,j)`, `unwrap(l,k)`,
#      `G = -mult * (g_j w1' + g_k (w1+w2)' + g_l (w1+w2+w3)')`. The cumulative
#      sums are not optional: `w1, w2, w3` alone is not d(phi)/dh.
#   2. `affine_pressure = -tr(G h')/(3V) = +tr(Xi)/(3V)`, NOT `-tr(Xi)/(3V)`.
#      The check below is an isotropic-scaling finite difference of the energy,
#      which fixes the sign independently of the analytic formula.
#
# The suite also states the conventions it relies on:
#
#   * `F = -dU/dq`; `box_derivative` is `G[a,b] = dU/dh[a,b]` at fixed fractional
#     coordinates `s = h^-1 q`; `Xi = -h G'`, so `Xi[a,b] = sum_i q_i[a] F_i[b]`.
#   * PME declares a **uniform neutralizing background plus tin-foil** boundary
#     conditions (PME.jl always includes the background term), so a net-charged
#     fixture is well posed and the direct `PME.pme_energy_forces` reference uses
#     exactly the same convention.
#
# Finite-difference policy. Central differences have truncation error O(h^2) and a
# cancellation floor O(eps * |U| / h). The assertions therefore demand:
#   * second-order convergence in the truncation regime (ratio >= 10 per 10x step),
#   * an absolute accuracy target at a step where truncation dominates,
#   * at the smallest step, only that the measured error stays under the predicted
#     cancellation floor. Demanding a *strict* decrease through 1e-8 would encode a
#     numerical artifact, not a property of the derivative, and is called out where
#     it applies.
#
# Measured errors are printed so the orchestrator can cite them.

using FoundationCore
const T  = FoundationCore.Types
const F  = FoundationCore.Factors
const RS = FoundationCore.ResolvedSystem
const SC = FoundationCore.SystemContract
const SL = FoundationCore.StreamedLJ
const PM = FoundationCore.PME
const EL = FoundationCore.Electrostatics
const DY = FoundationCore.Dynamics

using LinearAlgebra
using Random
using Printf

const LA = LinearAlgebra

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

"Run `f`, returning the exception it threw (or `nothing`)."
function _caught(f)
    try
        f()
    catch e
        return e
    end
    return nothing
end

"Maximum absolute entry of a scalar or array."
_maxabs(x) = x isa Number ? abs(x) : maximum(abs, x)

"Scale-aware maximum absolute difference: `max|A-B| / max(1, max|B|)`."
function _relmax(a, b)
    return _maxabs(a isa Number ? a - b : a .- b) / max(1.0, _maxabs(b))
end

"Lattice matrix (columns are cell vectors), built independently of SystemContract."
_hmat(b::T.OrthorhombicBox) = [b.L[1] 0.0 0.0; 0.0 b.L[2] 0.0; 0.0 0.0 b.L[3]]
_hmat(b::T.TriclinicBox) = copy(b.H)

# Six independent matrix directions: three diagonal (volume-changing) and three
# off-diagonal (shear) entries. A full 3x3 FD would count each symmetric pair twice
# and says nothing extra; these six span the symmetric 3x3 space.
const _SIX_DIRECTIONS = ((1, 1), (2, 2), (3, 3), (1, 2), (1, 3), (2, 3))

"""
Central difference of `U(h)` with respect to `h[a,b]` at fixed fractional
coordinates: perturb `h -> h +/- eps*E_ab`, rebuild `q = h*s` from the same `s`,
and evaluate the caller's energy function.
"""
function _fd_h(ufun::Function, h::Matrix{Float64}, S::Matrix{Float64},
               a::Int, b::Int, eps::Float64)
    E = zeros(3, 3)
    E[a, b] = eps
    return (ufun(h .+ E) - ufun(h .- E)) / (2.0 * eps)
end

"Cartesian-force finite differences of `energy_fun(q)` at step `h`."
function _fd_forces(energy_fun::Function, q::Matrix{Float64}, h::Float64)
    g = zeros(3, size(q, 2))
    qp = copy(q)
    qm = copy(q)
    for j in axes(q, 2), a in 1:3
        qp[a, j] += h
        qm[a, j] -= h
        g[a, j] = (energy_fun(qp) - energy_fun(qm)) / (2.0 * h)
        qp[a, j] -= h
        qm[a, j] += h
    end
    return -g
end

"Report a convergence table: step, max error, and ratio to the previous step."
function _print_convergence(label::String, steps, errs)
    @printf("  %s\n", label)
    for k in eachindex(steps)
        ratio = k == 1 ? NaN : errs[k - 1] / errs[k]
        @printf("    eps=%.0e  max|err|=%.6e  ratio=%.3g\n", steps[k], errs[k], ratio)
    end
    return nothing
end

# ---------------------------------------------------------------------------
# 1. NoBox local factor set: forces against central differences
# ---------------------------------------------------------------------------

"Six-atom non-periodic fixture with one of each local factor type."
function _nobox_fixture()
    q = [0.0 1.1 1.6 2.2 4.0 5.2;
         0.0 0.3 -0.4 1.0 0.2 -0.9;
         0.0 0.2 0.6 0.4 -0.3 0.8]
    charges = zeros(6)
    charges[5] = 0.7
    charges[6] = -0.4
    factors = F.AbstractFactor[
        F.HarmonicBond(1, 2, 40.0, 1.05),
        F.HarmonicAngle(2, 3, 4, 30.0, 1.9),
        F.PeriodicDihedral(1, 2, 3, 4, 1.5, 3, 0.3),
        F.LennardJonesPair(3, 4, 0.5, 1.0; mode = :lfs, rc = 2.5),
        F.CoulombPair(5, 6, 0.7, -0.4),
    ]
    rs = RS.ResolvedSystem(masses = ones(6), charges = charges, factors = factors,
                           component_names = [:bond, :angle, :dihedral, :lj, :coulomb])
    return rs, q
end

function test_nobox_local_factors()
    rs, q = _nobox_fixture()
    @assert !(:box_derivative in RS.engine_capabilities(rs))
    @assert RS.n_atoms(rs) == 6

    # The contract delegates to the factor protocol. Passing a poisoned buffer also
    # checks that `energy_forces` zeroes it rather than adding to prior content.
    ctx = SC.resolved_context(rs)
    Uref, Fref = F.total_energy_forces(rs.factors, q, ctx)
    Fbuf = fill(NaN, 3, 6)
    U, Fm = SC.energy_forces(rs, q; F = Fbuf)
    @assert all(isfinite, Fm) "energy_forces did not zero the supplied accumulator"
    @assert Fbuf === Fm
    uerr = _relmax(U, Uref)
    ferr = _relmax(Fm, Fref)
    @printf("  NoBox delegation: max|U-Uref|=%.3e  max|F-Fref|=%.3e\n", uerr, ferr)
    @assert uerr <= 1e-14 && ferr <= 1e-14
    @assert SC.energy(rs, q) == U
    @assert SC.forces(rs, q) == Fm
    U2, F2 = SC.evaluate_system(rs, q)
    @assert U2 == U && F2 == Fm

    steps = (1e-4, 1e-6, 1e-8)
    errs = Float64[]
    for h in steps
        F_fd = _fd_forces(x -> SC.energy(rs, x), q, h)
        push!(errs, maximum(abs, Fm .- F_fd))
    end
    _print_convergence("NoBox Cartesian force FD", steps, errs)
    # `U` is O(20) here, so the cancellation floor at 1e-8 is ~eps*|U|/(2h) ~ 1e-7.
    # Truncation error must fall by ~1e4 per 100x step for a smooth potential.
    @assert errs[2] < errs[1] / 10 "force FD is not converging: $(errs)"
    @assert errs[2] < 1e-7 "max force FD error at 1e-6 is $(errs[2])"
    @assert errs[3] < 1e-6 "1e-8 step should stay under the roundoff floor; got $(errs[3])"

    # The same central differences through `fractional_derivative` are meaningless
    # without a box, which is asserted separately in test_nobox_refusals.
    return true
end

# ---------------------------------------------------------------------------
# 2. Periodic image crossing: F = -h^-T dU/ds
# ---------------------------------------------------------------------------

function test_fractional_chain_rule()
    L = (4.0, 5.0, 6.0)
    box = T.OrthorhombicBox(L)
    # d = q1 - q2 = (-3.8, -0.2, -1.0) -> minimum image (0.2, -0.2, -1.0).
    q = [0.1 3.9; 1.0 1.2; 1.0 2.0]
    bond = F.HarmonicBond(1, 2, 100.0, 1.0)
    rs = RS.ResolvedSystem(masses = ones(2), charges = zeros(2), box = box,
                           factors = [bond], component_names = [:bond])

    r_mic = sqrt(0.2^2 + 0.2^2 + 1.0^2)
    U_mic = 0.5 * 100.0 * (r_mic - 1.0)^2
    U = SC.energy(rs, q)
    @assert abs(U - U_mic) <= 1e-12 * max(1.0, abs(U_mic))
    # The unwrapped separation is 3.98; if MIC were skipped the energy would be
    # enormous, so this pins the image choice, not just the arithmetic.
    r_direct = sqrt(3.8^2 + 0.2^2 + 1.0^2)
    U_direct = 0.5 * 100.0 * (r_direct - 1.0)^2
    @assert abs(U - U_direct) > 1.0
    @printf("  image crossing: U_MIC=%.12f  U_unwrapped=%.6f  r_MIC=%.9f\n",
            U, U_direct, r_mic)

    h = _hmat(box)
    F_analytic = SC.forces(rs, q)
    steps = (1e-5, 1e-6, 1e-7)
    errs = Float64[]
    for step in steps
        g = SC.fractional_derivative(rs, q; h = step)
        F_fd = -(transpose(h) \ g)          # F = -h^-T dU/ds
        push!(errs, maximum(abs, F_analytic .- F_fd))
    end
    _print_convergence("fractional chain rule (F vs -h^-T dU/ds)", steps, errs)
    # |F| ~ 3.8 and the bond curvature is k=100, so truncation O(k h^2) dominates
    # down to 1e-6; 1e-7 is at the cancellation floor eps*|U|/(2h) ~ 1e-10.
    @assert errs[2] < errs[1] / 10 "chain-rule FD is not converging: $(errs)"
    @assert errs[1] < 1e-7 && errs[2] < 1e-9 "chain-rule error is $(errs)"
    @assert errs[3] < 1e-8 "chain-rule error at 1e-7 is $(errs[3])"
    return true
end

# ---------------------------------------------------------------------------
# 3. Exact analytic box derivative and six-direction FD (harmonic bond)
# ---------------------------------------------------------------------------

function test_exact_bond_box_derivative()
    h = _hmat(T.OrthorhombicBox((4.0, 5.0, 6.0)))
    s = [0.1, 0.2, -0.15]                   # fractional displacement atom2 - atom1
    S = hcat(zeros(3), s)                   # atom 1 at the origin
    q = h * S
    k = 120.0
    r0 = 1.05
    bond = F.HarmonicBond(1, 2, k, r0)
    rs = RS.ResolvedSystem(masses = ones(2), charges = zeros(2),
                           box = T.OrthorhombicBox((4.0, 5.0, 6.0)),
                           factors = [bond], component_names = [:bond])
    @assert :box_derivative in RS.engine_capabilities(rs)
    @assert SC.supports_box_derivative(bond, rs.box)

    rvec = h * s                            # = h*s = atom2 position
    r = sqrt(sum(abs2, rvec))
    G_exact = (k * (r - r0) / r) .* (rvec * transpose(s))
    G = SC.box_derivative_component(rs, q, :bond)
    exact_err = maximum(abs, G .- G_exact)
    @printf("  exact analytic bond G: max|G-G_exact|=%.3e  (|G|max=%.3e)\n",
            exact_err, maximum(abs, G_exact))
    @assert exact_err <= 1e-12 * max(1.0, maximum(abs, G_exact))

    steps = (1e-3, 1e-4, 1e-5, 1e-6)
    errs = Float64[]
    ufun = hh -> SC.energy(rs, hh * S)
    for eps in steps
        err = 0.0
        for (a, b) in _SIX_DIRECTIONS
            g_fd = _fd_h(ufun, h, S, a, b, eps)
            err = max(err, abs(G[a, b] - g_fd))
        end
        push!(errs, err)
    end
    _print_convergence("bond box derivative FD (six directions)", steps, errs)
    # Truncation error is O(eps^2) on |G| ~ 4; 1e-8 absolute at 1e-5 is >100x margin.
    @assert errs[2] < errs[1] "error did not shrink from 1e-3 to 1e-4: $(errs)"
    @assert errs[3] < errs[2] "error did not shrink from 1e-4 to 1e-5: $(errs)"
    @assert errs[3] < 1e-8 "box-derivative FD error at 1e-5 is $(errs[3])"
    # At 1e-6 cancellation (eps*|U|/(2eps) ~ 1e-10) starts to dominate; the bound is
    # an order-of-magnitude guard, not a demand for further decrease.
    @assert errs[4] < 1e-8 "1e-6 step left the roundoff floor: $(errs[4])"
    return true
end

# ---------------------------------------------------------------------------
# periodic local fixture used by tests 4, 5 and the dihedral errata check
# ---------------------------------------------------------------------------

function _periodic_local_fixture()
    box = T.OrthorhombicBox((6.0, 6.0, 6.0))
    q = [0.0 1.1 1.6 2.2;
         0.0 0.3 -0.4 1.0;
         0.0 0.2 0.6 0.4]
    factors = F.AbstractFactor[
        F.HarmonicBond(1, 2, 40.0, 1.05),
        F.HarmonicAngle(2, 3, 4, 30.0, 1.9),
        F.PeriodicDihedral(1, 2, 3, 4, 1.5, 3, 0.3),
        F.LennardJonesPair(3, 4, 0.5, 1.0; mode = :lfs, rc = 2.5),
    ]
    rs = RS.ResolvedSystem(masses = ones(4), charges = zeros(4), box = box,
                           factors = factors,
                           component_names = [:bond, :angle, :dihedral, :lj])
    return rs, q, box
end

# ---------------------------------------------------------------------------
# 4. affine pressure against isotropic scaling, and the sign of tr(Xi)
# ---------------------------------------------------------------------------

function test_affine_pressure_isotropic_scaling()
    rs, q, box = _periodic_local_fixture()
    G = SC.box_derivative(rs, q)
    h = _hmat(box)
    V = T.box_volume(box)
    tr_Gh = sum(G .* transpose(h))
    p_affine = SC.affine_pressure(rs, q)
    p_formula = -tr_Gh / (3.0 * V)
    @assert abs(p_affine - p_formula) <= 1e-12 * max(1.0, abs(p_formula))

    Xi = SC.virial(rs, q)
    @assert abs(p_affine - tr(Xi) / (3.0 * V)) <= 1e-12 * max(1.0, abs(p_affine))
    # The interface typo had -tr(Xi)/(3V). tr(Xi) is nonzero here, so the wrong sign
    # would be a measurable, not a rounding, difference.
    @assert abs(tr(Xi)) > 1e-6
    @assert abs(p_affine + tr(Xi) / (3.0 * V)) > 1e-3 *
            max(1.0, abs(p_affine)) "pressure sign is not what the errata says"

    # Isotropic scaling: h -> lambda*h, s fixed, so q -> lambda*q. Then
    # dU/dlambda = tr(G h'), hence p_affine = -(dU/dlambda)/(3V). Every displacement
    # in this fixture is shorter than L/2, so no pair uses a lattice image and
    # "scale q at fixed box" is exactly "scale h at fixed s" (for a pair with an
    # image n != 0 the two differ by an eps*n term, which is why the streamed test
    # rebuilds its kernel on the perturbed box).
    steps = (1e-3, 1e-4, 1e-5)
    errs = Float64[]
    for delta in steps
        dU = (SC.energy(rs, (1.0 + delta) * q) - SC.energy(rs, (1.0 - delta) * q)) /
             (2.0 * delta)
        p_fd = -dU / (3.0 * V)
        push!(errs, abs(p_affine - p_fd))
    end
    @printf("  affine pressure: p=%.12f  tr(Xi)=%.9f  V=%.1f\n",
            p_affine, tr(Xi), V)
    _print_convergence("affine pressure vs isotropic-scaling FD", steps, errs)
    @assert errs[2] < errs[1] && errs[3] < errs[2] "isotropic FD is not converging: $(errs)"
    @assert errs[3] < 1e-8 "affine-pressure FD error at 1e-5 is $(errs[3])"
    return true
end

# ---------------------------------------------------------------------------
# 5. virial identity Xi = -h G' = sum_i r_i F_i'
# ---------------------------------------------------------------------------

function test_virial_identity()
    rs, q, box = _periodic_local_fixture()
    G = SC.box_derivative(rs, q)
    h = _hmat(box)
    Xi_h = -(h * transpose(G))
    Xi = SC.virial(rs, q)
    err_formula = maximum(abs, Xi .- Xi_h)

    Fm = SC.forces(rs, q)
    Xi_direct = q * transpose(Fm)           # Xi[a,b] = sum_i q_i[a] F_i[b]
    err_direct = maximum(abs, Xi .- Xi_direct)
    scale = max(1.0, maximum(abs, Xi))
    @printf("  virial: max|Xi-(-hG')|=%.3e  max|Xi-rF'|=%.3e  |Xi|max=%.3e\n",
            err_formula, err_direct, maximum(abs, Xi))
    @assert err_formula <= 1e-12 * scale
    @assert err_direct <= 1e-12 * scale
    @assert maximum(abs, Xi) > 1e-3 "the fixture has no virial content"

    # Total pressure is (2K + tr(Xi))/(3V); K is checked through kinetic_energy.
    p = zeros(3, 4)
    p[1, 1] = 0.3
    K = SC.kinetic_energy(rs, p)
    @assert K == 0.5 * 0.3^2
    V = T.box_volume(box)
    @assert abs(SC.pressure(rs, q, p) - (2.0 * K + tr(Xi)) / (3.0 * V)) <= 1e-12
    @assert SC.pressure(rs, q, p; extra_virial = 2.0) ==
            (2.0 * K + tr(Xi) + 2.0) / (3.0 * V)
    return true
end

# ---------------------------------------------------------------------------
# 5b. per-component box derivatives, including the dihedral errata
# ---------------------------------------------------------------------------

function test_periodic_component_box_derivatives()
    rs, q, box = _periodic_local_fixture()
    S = q ./ box.L
    h = _hmat(box)
    steps = (1e-3, 1e-4, 1e-5)
    Gsum = zeros(3, 3)
    for name in (:bond, :angle, :dihedral, :lj)
        G = SC.box_derivative_component(rs, q, name)
        Gsum .+= G
        ufun = hh -> SC.component_energy(rs, hh * S, name)
        errs = Float64[]
        for eps in steps
            err = 0.0
            for (a, b) in _SIX_DIRECTIONS
                err = max(err, abs(G[a, b] - _fd_h(ufun, h, S, a, b, eps)))
            end
            push!(errs, err)
        end
        _print_convergence("box derivative FD, component :$name", steps, errs)
        @assert maximum(abs, G) > 1e-8 "component :$name has no box derivative content"
        # 1e-3 -> 1e-4 is the truncation regime (expected ratio ~100); by 1e-5 the
        # difference quotient is limited by cancellation (|U| ~ 10 => ~1e-10 floor),
        # so it is required to be accurate in absolute terms, not to keep shrinking.
        @assert errs[2] < errs[1] / 10 "component :$name FD is not converging: $(errs)"
        @assert errs[2] < 1e-8 && errs[3] < 1e-7 "component :$name FD error is $(errs)"

        if name === :dihedral
            # Errata check: the derivative needs the *cumulative* fractional
            # offsets. Writing w1, w2, w3 instead of w1, w1+w2, w1+w2+w3 is a
            # different matrix, and the FD says so.
            t = F.dihedral_angle_grad(q, box, 1, 2, 3, 4)
            @assert t !== nothing
            phi, (gi, gj, gk, gl) = t
            arg = 3 * phi - 0.3
            mult = 1.5 * 3 * sin(arg)
            w1 = (q[:, 2] .- q[:, 1]) ./ box.L
            w2 = (q[:, 3] .- q[:, 2]) ./ box.L
            w3 = (q[:, 4] .- q[:, 3]) ./ box.L
            gv(v) = [v.x, v.y, v.z]
            G_wrong = -mult .* (gv(gj) * w1' + gv(gk) * w2' + gv(gl) * w3')
            fd = [_fd_h(ufun, h, S, a, b, 1e-5) for a in 1:3, b in 1:3]
            err_wrong = maximum(abs, G_wrong .- fd)
            @printf("  dihedral errata: correct-form err=%.3e  non-cumulative err=%.3e\n",
                    maximum(abs, G .- fd), err_wrong)
            @assert err_wrong > 1e-3 "the non-cumulative formula was not distinguished"
        end
    end
    Gtot = SC.box_derivative(rs, q)
    @assert maximum(abs, Gsum .- Gtot) <= 1e-12 * max(1.0, maximum(abs, Gtot))
    return true
end



function test_named_components()
    rs, q = _nobox_fixture()
    props = SC.component_energies(rs, q)
    @assert keys(props) == (:bond, :angle, :dihedral, :lj, :coulomb)
    total = SC.energy(rs, q)
    summ = sum(values(props))
    @printf("  components: sum=%.12f  energy=%.12f  err=%.3e\n",
            summ, total, abs(summ - total))
    @assert abs(summ - total) <= 1e-12 * max(1.0, abs(total))
    for name in keys(props)
        @assert SC.component_energy(rs, q, name) == props[name]
    end

    # Superposition: each factor in its own resolved system, same box/atoms/charges.
    charges = zeros(6)
    charges[5] = 0.7
    charges[6] = -0.4
    partial = 0.0
    for (k, fac) in enumerate(rs.factors)
        one = RS.ResolvedSystem(masses = ones(6), charges = charges,
                                factors = F.AbstractFactor[fac],
                                component_names = [rs.component_names[k]])
        partial += SC.energy(one, q)
    end
    @assert abs(partial - total) <= 1e-12 * max(1.0, abs(total))
    @printf("  superposition: sum of single-factor energies=%.12f  err=%.3e\n",
            partial, abs(partial - total))

    e = _caught(() -> SC.component_energy(rs, q, :nosuch))
    @assert e isa RS.ResolvedSystemError && e.code === :component_name
    return true
end

# ---------------------------------------------------------------------------
# 7. StreamedLJ: resolved parity, box derivative, pair weights, cutoff, :shift
# ---------------------------------------------------------------------------

"3x3x3 lattice fixture (spacing 2.0 in L=6), same seed as the dynamics suite."
function _streamed_fixture()
    L = 6.0
    box = T.OrthorhombicBox((L, L, L))
    rc = 2.2
    skin = 0.2
    N = 27
    q = zeros(3, N)
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

    pairs = Tuple{Int,Int}[]
    for i in 1:N-1, j in i+1:N
        push!(pairs, (i, j))
    end
    sort!(pairs; by = pr -> T.norm2(T.unwrap_displacement(box, q, pr[1], pr[2])))
    excluded = pairs[1:4]
    scaled = [(pairs[4 + s][1], pairs[4 + s][2], 0.5) for s in 1:3]
    topo = SL.PairTopology(excluded = excluded, scaled = scaled)
    fac = SL.StreamedLJFactor(q, epsilon, sigma, box;
                              rcut = rc, skin = skin, mode = :lfs, topology = topo)
    return (fac = fac, q = q, p = p, box = box, rc = rc, N = N,
            excluded = excluded, scaled = scaled)
end

"ChemicalTopology spelling of the same pair rules, required by resolution."
function _matching_topology(excluded, scaled)
    ex = Set{Tuple{Int,Int}}((min(i, j), max(i, j)) for (i, j) in excluded)
    sc = Dict{Tuple{Int,Int},NTuple{2,Float64}}(
        (min(i, j), max(i, j)) => (w, w) for (i, j, w) in scaled)
    return T.ChemicalTopology(exclusions = ex, scales = sc)
end

function test_streamed_lj_contract()
    s = _streamed_fixture()
    fac = s.fac
    rs = RS.ResolvedSystem(masses = ones(s.N), charges = zeros(s.N), box = s.box,
                           topology = _matching_topology(s.excluded, s.scaled),
                           factors = [fac], component_names = [:lj])
    @assert :box_derivative in RS.engine_capabilities(rs)

    ctx = SC.resolved_context(rs)
    Fref = zeros(3, s.N)
    Uref = F.evaluate!(Fref, fac, s.q, ctx)
    U, Fm = SC.energy_forces(rs, s.q)
    uerr = abs(U - Uref) / max(1.0, abs(Uref))
    ferr = maximum(abs, Fm .- Fref) / max(1.0, maximum(abs, Fref))
    @printf("  streamed parity: max|dU|=%.3e  max|dF|=%.3e  U=%.9f\n", uerr, ferr, U)
    @assert uerr <= 1e-12 && ferr <= 1e-12
    @assert U < 0.0 "the LJ fixture should be net attractive"

    # Box derivative, diagonal directions only (the backend is orthorhombic-only).
    #
    # The FD evaluates the same cell at fixed fractional coordinates: `q = h'*s`
    # *and* the kernel is rebuilt on the perturbed box `h'`. Using the original box
    # for the perturbed positions is wrong for a periodic kernel whenever a pair
    # uses a lattice image `n != 0`: the wrapped displacement would be
    # `h'*w - h*n` instead of `h'*(w - n)`, adding a spurious `eps*n` term (it shows
    # up as an eps-independent offset that never converges). The two-atom bond
    # tests elsewhere are insensitive to this only because their pairs have n = 0.
    S = s.q ./ s.box.L
    G = SC.box_derivative(rs, s.q)
    @assert maximum(abs, G) > 1e-6
    function eval_at(dir::Int, eps::Float64)
        Lv = collect(s.box.L)
        Lv[dir] += eps
        boxp = T.OrthorhombicBox((Lv[1], Lv[2], Lv[3]))
        qp = _hmat(boxp) * S
        facp = SL.StreamedLJFactor(qp, s.fac.epsilon, s.fac.sigma, boxp;
                                   rcut = s.fac.rcut, skin = s.fac.skin,
                                   mode = s.fac.mode, topology = s.fac.topology)
        return F.evaluate!(zeros(3, s.N), facp, qp,
                           F.Context(boxp, T.REDUCED, s.N))
    end
    steps = (1e-3, 1e-4, 1e-5, 1e-6)
    errs = Float64[]
    for eps in steps
        err = 0.0
        for a in 1:3
            fd = (eval_at(a, eps) - eval_at(a, -eps)) / (2.0 * eps)
            err = max(err, abs(G[a, a] - fd))
        end
        push!(errs, err)
    end
    _print_convergence("streamed box derivative FD (diagonal h)", steps, errs)
    @assert errs[2] < errs[1] && errs[3] < errs[2] "streamed FD is not converging: $(errs)"
    @assert errs[3] < 1e-7 "streamed box-derivative error at 1e-5 is $(errs[3])"
    @assert errs[4] < 1e-6 "1e-6 step left the roundoff floor: $(errs[4])"

    return true
end

function test_streamed_pair_weights()
    L = 20.0
    box = T.OrthorhombicBox((L, L, L))
    rc = 5.0
    q = [0.0 0.0; 0.0 0.0; 0.0 1.4]
    eps = [1.0, 1.0]
    sig = [1.0, 1.0]

    function resolved(; excluded = Tuple{Int,Int}[], scaled = Tuple{Int,Int,Float64}[])
        fac = SL.StreamedLJFactor(q, eps, sig, box; rcut = rc, mode = :lfs,
                                  topology = SL.PairTopology(excluded = excluded,
                                                             scaled = scaled))
        topo = _matching_topology(excluded, scaled)
        return RS.ResolvedSystem(masses = ones(2), charges = zeros(2), box = box,
                                 topology = topo, factors = [fac],
                                 component_names = [:lj]), fac
    end

    rs_plain, fac_plain = resolved()
    rs_half, _ = resolved(scaled = [(1, 2, 0.5)])
    rs_none, _ = resolved(excluded = [(1, 2)])

    U_plain = SC.energy(rs_plain, q)
    U_half = SC.energy(rs_half, q)
    U_none = SC.energy(rs_none, q)
    @assert abs(U_plain) > 1e-2 "the two-atom pair must contribute"
    @printf("  pair weights: U=%.12f  U/2=%.12f  excluded=%.3e\n",
            U_plain, U_half, U_none)
    # The pair weight is a linear multiplier of the pair energy, so these are exact
    # up to one rounding of the product, not approximate physics.
    @assert abs(U_half - 0.5 * U_plain) <= 1e-12
    @assert abs(U_none) <= 1e-12
    @assert SL.pair_weight(fac_plain.topology, 2, 1) == 1.0
    return true
end

function test_streamed_cutoff_and_shift_refusal()
    L = 20.0
    box = T.OrthorhombicBox((L, L, L))
    rc = 2.5
    eps = [1.0, 1.0]
    sig = [1.5, 1.5]                       # sigma/r = 0.6 at rc, so the lfs tail is visible

    q = zeros(3, 2)
    fac = SL.StreamedLJFactor(q, eps, sig, box; rcut = rc, mode = :lfs)
    ctx = F.Context(box, T.REDUCED, 2)

    function energy_at(r)
        q[1, 2] = r
        return F.evaluate!(zeros(3, 2), fac, q, ctx)
    end

    U_plus = energy_at(rc + 1e-6)
    U_minus = energy_at(rc - 1e-6)
    @printf("  cutoff probe: U(rc+1e-6)=%.3e  U(rc-1e-6)=%.3e\n", U_plus, U_minus)
    @assert U_plus == 0.0                     # beyond the cutoff the lfs pair vanishes
    @assert abs(U_minus) < 1e-5 "lfs energy is not continuous at rc: $(U_minus)"

    r0 = rc - 1e-6
    q[1, 2] = r0
    Fana = zeros(3, 2)
    F.evaluate!(Fana, fac, q, ctx)
    # Force by central difference of the *energy* at a step far below the 1e-6
    # offset from the cutoff, so both samples stay inside the same branch.
    step = 1e-8
    F_fd = -(energy_at(r0 + step) - energy_at(r0 - step)) / (2.0 * step)
    ferr = abs(Fana[1, 2] - F_fd)
    @printf("  cutoff force: F_analytic=%.6e  F_fd=%.6e  |dF|=%.3e\n",
            Fana[1, 2], F_fd, ferr)
    @assert abs(F_fd) > 1e-8 "the force probe has no signal"
    @assert ferr < 2e-8 "force and FD disagree at rc-1e-6: $(ferr)"

    # A :shift backend is energy-only, so every stress entry point must refuse.
    shift_fac = SL.StreamedLJFactor(q, eps, sig, box; rcut = rc, mode = :shift)
    rs_shift = RS.ResolvedSystem(masses = ones(2), charges = zeros(2), box = box,
                                 factors = [shift_fac], component_names = [:lj])
    @assert RS.box_derivative_support(shift_fac) !== nothing
    @assert !(:box_derivative in RS.engine_capabilities(rs_shift))
    for (label, thunk) in (
            ("box_derivative", () -> SC.box_derivative(rs_shift, q)),
            ("virial", () -> SC.virial(rs_shift, q)),
            ("pressure", () -> SC.pressure(rs_shift, q, zeros(3, 2))),
            ("affine_pressure", () -> SC.affine_pressure(rs_shift, q)))
        e = _caught(thunk)
        @assert e isa SC.MissingBoxDerivative ":$label did not refuse a :shift backend"
        @assert e.component === :lj ":$label named $(e.component), expected :lj"
    end
    return true
end

# ---------------------------------------------------------------------------
# 8. PME: resolved parity, refusals, and a working local component
# ---------------------------------------------------------------------------

function test_pme_contract()
    # Net charge +1. PME includes the uniform neutralizing background and uses
    # tin-foil boundaries, which is exactly the reference below, so the fixture is
    # well posed rather than conditionally convergent.
    box = T.OrthorhombicBox((6.0, 6.0, 6.0))
    q = [0.7 2.1 4.6;
         1.3 3.4 0.9;
         2.5 1.1 3.8]
    charges = [0.75, 0.5, -0.25]            # dyadic, so sum(charges) == 1.0 exactly
    @assert sum(charges) == 1.0
    # PME requires rcut < min(L)/2 strictly; the requested 3.0 == 6.0/2 is refused by
    # PMEPlan, so the fixture uses 2.9 (same physics, valid cutoff).
    alpha = 0.35
    rcut = 2.9
    pme = PM.PMEFactor(q, charges, box; alpha = alpha, rcut = rcut,
                       grid = (8, 8, 8), order = 4)
    bond = F.HarmonicBond(1, 2, 50.0, 1.2)
    rs = RS.ResolvedSystem(masses = ones(3), charges = charges, box = box,
                           factors = F.AbstractFactor[pme, bond],
                           component_names = [:pme, :bond],
                           declared = [:energy, :cartesian_forces, :component_energies, :box_derivative])
    @assert :box_derivative in RS.engine_capabilities(rs)
    @assert SC.supports_box_derivative(pme, box)

    U_pme_ref, F_pme_ref = PM.pme_energy_forces(pme.plan, q, charges)
    F_bond_ref = zeros(3, 3)
    U_bond_ref = F.evaluate!(F_bond_ref, bond, q, SC.resolved_context(rs))
    U, Fm = SC.energy_forces(rs, q)
    uerr = abs(U - (U_pme_ref + U_bond_ref))
    ferr = maximum(abs, Fm .- (F_pme_ref .+ F_bond_ref))
    @printf("  PME parity: |dU|=%.3e  max|dF|=%.3e  U_pme=%.9f\n",
            uerr, ferr, U_pme_ref)
    @assert uerr <= 1e-12 && ferr <= 1e-12
    @assert abs(SC.component_energy(rs, q, :pme) - U_pme_ref) <= 1e-12
    @assert abs(U_pme_ref) > 1e-3

    G_tot = SC.box_derivative(rs, q)
    G_pme = SC.box_derivative_component(rs, q, :pme)
    G_bond = SC.box_derivative_component(rs, q, :bond)
    @assert maximum(abs, G_tot .- (G_pme .+ G_bond)) <= 1e-13
    @assert maximum(abs, G_pme .- PM.pme_box_derivative(pme.plan, q, charges)) <= 1e-13
    @assert all(isfinite, SC.virial(rs, q))
    @assert isfinite(SC.affine_pressure(rs, q))
    @assert isfinite(SC.pressure(rs, q, zeros(3, 3)))

    # 9-entry central FD check of total PME+bond box derivative
    S = q ./ box.L
    h = _hmat(box)
    err = 0.0
    for a in 1:3, b in 1:3
        hp = copy(h); hp[a, b] += 1e-5
        hm = copy(h); hm[a, b] -= 1e-5
        bp = T.TriclinicBox(hp); bm = T.TriclinicBox(hm)
        rsp = RS.restart_system(rs; box = bp)
        rsm = RS.restart_system(rs; box = bm)
        fd = (SC.energy(rsp, hp * S) - SC.energy(rsm, hm * S)) / (2e-5)
        err = max(err, abs(G_tot[a, b] - fd))
    end
    @printf("  PME+bond ortho 9-entry box-derivative FD error=%.3e\n", err)
    @assert err < 1e-7 "PME+bond FD error is $(err)"

    # EwaldFactor still refuses box_derivative
    ew = FoundationCore.Electrostatics.EwaldFactor(q, charges, box; alpha = alpha, rcut = rcut, kmax = 4)
    rs_ew = RS.ResolvedSystem(masses = ones(3), charges = charges, box = box,
                              factors = F.AbstractFactor[ew, bond],
                              component_names = [:ewald, :bond])
    for (label, thunk) in (
            ("box_derivative", () -> SC.box_derivative(rs_ew, q)),
            ("virial", () -> SC.virial(rs_ew, q)),
            ("pressure", () -> SC.pressure(rs_ew, q, zeros(3, 3))),
            ("affine_pressure", () -> SC.affine_pressure(rs_ew, q)))
        e = _caught(thunk)
        @assert e isa SC.MissingBoxDerivative ":$label did not refuse the Ewald system"
        @assert e.component === :ewald ":$label named $(e.component), expected :ewald"
    end
    return true
end

# ---------------------------------------------------------------------------
# 9. Triclinic local + StreamedLJ + PMEFactor box derivatives & refusals
# ---------------------------------------------------------------------------

function test_triclinic_local_and_refusal()
    H = [4.0 0.8 0.0;
         0.0 5.0 0.5;
         0.0 0.0 6.0]
    box = T.TriclinicBox(H)
    s1 = [0.05, 0.05, 0.05]
    w = [0.3, 0.2, -0.1]
    S = hcat(s1, s1 .+ w)
    q = H * S
    bond = F.HarmonicBond(1, 2, 80.0, 1.5)
    rs = RS.ResolvedSystem(masses = ones(2), charges = zeros(2), box = box,
                           factors = [bond], component_names = [:bond])
    @assert :box_derivative in RS.engine_capabilities(rs)
    @assert abs(SC.energy(rs, q) - 0.5 * 80.0 * (sqrt(sum(abs2, H * w)) - 1.5)^2) <= 1e-12

    G = SC.box_derivative(rs, q)
    @assert maximum(abs, G) > 1e-6
    h = copy(H)
    ufun = hh -> SC.energy(rs, hh * S)
    steps = (1e-3, 1e-4, 1e-5, 1e-6)
    errs = Float64[]
    for eps in steps
        err = 0.0
        for (a, b) in _SIX_DIRECTIONS
            err = max(err, abs(G[a, b] - _fd_h(ufun, h, S, a, b, eps)))
        end
        push!(errs, err)
    end
    _print_convergence("triclinic bond box-derivative FD (six directions)", steps, errs)
    @assert errs[2] < errs[1] && errs[3] < errs[2] "triclinic FD is not converging: $(errs)"
    @assert errs[3] < 1e-8 "triclinic box-derivative FD error at 1e-5 is $(errs[3])"

    # Orthorhombic-built StreamedLJFactor resolved against TriclinicBox refuses with :box_mismatch
    ortho = T.OrthorhombicBox((4.0, 5.0, 6.0))
    lj = SL.StreamedLJFactor(q, [1.0, 1.0], [1.0, 1.0], ortho; rcut = 1.0)
    e = _caught(() -> RS.ResolvedSystem(masses = ones(2), charges = zeros(2),
                                        box = box, factors = [lj],
                                        component_names = [:lj]))
    @assert e isa RS.ResolvedSystemError "mismatched ortho StreamedLJ on triclinic was not refused"
    @assert e.code === :box_mismatch "refusal code is $(e.code), expected :box_mismatch"

    # Triclinic StreamedLJFactor(:lfs) + PMEFactor + HarmonicBond end-to-end 9-entry FD check
    H_tri = [10.0 1.4 0.5;
              0.0 10.5 0.9;
              0.0 0.0 11.0]
    tbox = T.TriclinicBox(H_tri)
    S_tri = [0.12 0.28 0.55 0.78;
             0.18 0.31 0.62 0.81;
             0.15 0.25 0.48 0.72]
    q_tri = H_tri * S_tri
    ch_tri = [0.8, -0.5, 0.4, -0.7]
    topo_tri = T.ChemicalTopology(exclusions = Set([(1, 2)]),
                                  scales = Dict((2, 3) => (0.5, 0.5)))
    slj_tri = SL.StreamedLJFactor(q_tri, fill(0.25, 4), fill(1.1, 4), tbox;
                                  rcut = 3.5, mode = :lfs,
                                  topology = SL.PairTopology(excluded = [(1, 2)],
                                                             scaled = [(2, 3, 0.5)]))
    pme_tri = PM.PMEFactor(q_tri, ch_tri, tbox; alpha = 0.35, rcut = 3.5,
                           grid = (24, 24, 24), order = 4,
                           corrections = PM.PairCorrections(excluded = [(1, 2)],
                                                            scaled = [(2, 3, 0.5)]))
    bond_tri = F.HarmonicBond(1, 2, 40.0, 1.8)
    rs_tri = RS.ResolvedSystem(masses = ones(4), charges = ch_tri, box = tbox,
                               topology = topo_tri,
                               factors = F.AbstractFactor[bond_tri, slj_tri, pme_tri],
                               component_names = [:bond, :lj, :pme],
                               declared = [:energy, :cartesian_forces, :component_energies, :box_derivative])
    @assert :box_derivative in RS.engine_capabilities(rs_tri)
    @assert SC.supports_box_derivative(slj_tri, tbox)
    @assert SC.supports_box_derivative(pme_tri, tbox)

    G_tri = SC.box_derivative(rs_tri, q_tri)
    G_b = SC.box_derivative_component(rs_tri, q_tri, :bond)
    G_l = SC.box_derivative_component(rs_tri, q_tri, :lj)
    G_p = SC.box_derivative_component(rs_tri, q_tri, :pme)
    @assert maximum(abs, G_tri .- (G_b .+ G_l .+ G_p)) <= 1e-13

    # Warm PMEFactor box_derivative! and energy_forces_box_derivative! are 0-allocation
    ctx_tri = SC.resolved_context(rs_tri)
    F_buf = zeros(3, 4)
    G_buf = zeros(3, 3)
    PM.pme_factor_box_derivative!(G_buf, pme_tri, q_tri, ctx_tri)
    @assert (@allocated PM.pme_factor_box_derivative!(G_buf, pme_tri, q_tri, ctx_tri)) == 0
    PM.pme_factor_energy_forces_box_derivative!(F_buf, G_buf, pme_tri, q_tri, ctx_tri)
    @assert (@allocated PM.pme_factor_energy_forces_box_derivative!(F_buf, G_buf, pme_tri, q_tri, ctx_tri)) == 0

    # 9-entry central FD check
    eps_tri = 1e-5
    max_tri_err = 0.0
    for a in 1:3, b in 1:3
        Hp = copy(H_tri); Hp[a, b] += eps_tri
        Hm = copy(H_tri); Hm[a, b] -= eps_tri
        rsp = RS.restart_system(rs_tri; box = T.TriclinicBox(Hp))
        rsm = RS.restart_system(rs_tri; box = T.TriclinicBox(Hm))
        fd = (SC.energy(rsp, Hp * S_tri) - SC.energy(rsm, Hm * S_tri)) / (2.0 * eps_tri)
        max_tri_err = max(max_tri_err, abs(G_tri[a, b] - fd))
    end
    @printf("  triclinic bond+StreamedLJ+PME 9-entry FD error=%.3e\n", max_tri_err)
    @assert max_tri_err < 1e-7 "triclinic full system 9-entry FD error=$(max_tri_err)"
    @assert isfinite(SC.affine_pressure(rs_tri, q_tri))
    @assert isfinite(SC.pressure(rs_tri, q_tri, zeros(3, 4)))
    return true
end

# ---------------------------------------------------------------------------
# 10. adapter preservation: resolved_from_system and to_atomset
# ---------------------------------------------------------------------------

function test_adapter_preservation()
    s = _streamed_fixture()
    fac = s.fac
    names = [Symbol("A", i) for i in 1:s.N]
    atoms = T.AtomSet(s.q, s.p, ones(s.N), zeros(s.N), names)
    sys = T.System(atoms, s.box, _matching_topology(s.excluded, s.scaled))
    rs = SC.resolved_from_system(sys, [fac])
    @assert RS.n_atoms(rs) == s.N
    @assert rs.factors[1] === fac "the resolved record must share the factor object"
    @assert rs.atoms.masses == atoms.m
    @assert rs.atoms.charges == atoms.z
    @assert rs.atoms.names == names
    @assert SC.resolved_context(rs).box === s.box
    @assert rs.box === s.box && rs.topology === sys.topology && rs.units == sys.units

    ctx_raw = F.Context(s.box, T.REDUCED, s.N)
    sys_raw = DY.HarmonicSystem([fac], ones(s.N), ctx_raw)
    sys_res = DY.HarmonicSystem(rs.factors, rs.atoms.masses, SC.resolved_context(rs))

    F_raw = zeros(3, s.N)
    F_res = zeros(3, s.N)
    U_raw = F.evaluate!(F_raw, fac, s.q, ctx_raw)
    U_res = F.evaluate!(F_res, fac, s.q, ctx_raw)
    @assert U_raw == U_res && F_raw == F_res "adapter changed the force field"

    st_raw = DY.State(s.q, s.p)
    st_res = DY.State(s.q, s.p)
    DY.velocity_verlet_step!(st_raw, sys_raw, 1e-3)
    DY.velocity_verlet_step!(st_res, sys_res, 1e-3)
    dq = maximum(abs, st_raw.q .- st_res.q)
    dp = maximum(abs, st_raw.p .- st_res.p)
    dF = maximum(abs, st_raw.F .- st_res.F)
    @printf("  adapter: max|dq|=%.1f  max|dp|=%.1f  max|dF|=%.1f (bit-identical)\n",
            dq, dp, dF)
    @assert st_raw.q == st_res.q
    @assert st_raw.p == st_res.p
    @assert st_raw.F == st_res.F
    @assert st_raw.q != s.q "the step must actually move the state"

    aset = SC.to_atomset(rs, s.q, s.p)
    @assert aset.m == rs.atoms.masses
    @assert aset.z == rs.atoms.charges
    @assert aset.names == rs.atoms.names
    @assert aset.q == s.q && aset.p == s.p
    return true
end

# ---------------------------------------------------------------------------
# 11. NoBox refusals and virtual-site execution refusal
# ---------------------------------------------------------------------------

function test_nobox_and_virtual_site_refusals()
    rs, q = _nobox_fixture()
    e = _caught(() -> SC.fractional_coordinates(rs, q))
    @assert e isa RS.UnsupportedCapability
    @assert e.capability === :fractional_coordinates
    e2 = _caught(() -> SC.cartesian_coordinates(rs, q))
    @assert e2 isa RS.UnsupportedCapability && e2.capability === :cartesian_coordinates
    e3 = _caught(() -> SC.box_derivative(rs, q))
    @assert e3 isa SC.MissingBoxDerivative && e3.component === :system

    # A massless declared site is a valid specification but not executable.
    vs = RS.VirtualSite(:linear, 3, [1, 2], [0.5, 0.5])
    rs_vs = RS.ResolvedSystem(masses = [1.0, 1.0, 0.0], charges = zeros(3),
                              factors = [F.HarmonicBond(1, 2, 20.0, 1.0)],
                              virtual_sites = [vs], component_names = [:bond])
    @assert RS.virtual_site_indices(rs_vs) == Set([3])
    q_vs = [0.0 1.0 0.5; 0.0 0.0 0.0; 0.0 0.0 0.0]
    for (label, thunk) in (
            ("energy", () -> SC.energy(rs_vs, q_vs)),
            ("forces", () -> SC.forces(rs_vs, q_vs)),
            ("component_energies", () -> SC.component_energies(rs_vs, q_vs)),
            ("to_atomset", () -> SC.to_atomset(rs_vs, q_vs, zeros(3, 3))))
        err = _caught(thunk)
        @assert err isa RS.UnsupportedCapability ":$label executed a virtual-site system"
        @assert err.capability === :virtual_sites ":$label named $(err.capability)"
    end
    return true
end

# ---------------------------------------------------------------------------
# delegated factor for extensible box derivative & contract errors
# ---------------------------------------------------------------------------

struct DelegatedContractFactor <: F.AbstractFactor
    i::Int
    j::Int
    k::Float64
    r0::Float64
end
F.read_support(f::DelegatedContractFactor) = (f.i, f.j)
F.factor_support_size(::DelegatedContractFactor) = 2

function F.evaluate!(Fout::AbstractMatrix{Float64}, f::DelegatedContractFactor,
                     q::AbstractMatrix{Float64}, ctx::F.Context)
    d = T.unwrap_displacement(ctx.box, q, f.i, f.j)
    r = sqrt(T.norm2(d))
    dr = r - f.r0
    u = 0.5 * f.k * dr * dr
    c = -f.k * dr / r
    for m in 1:3
        g = c * (m == 1 ? d.x : m == 2 ? d.y : d.z)
        Fout[m, f.i] += g
        Fout[m, f.j] -= g
    end
    return u
end

function RS.accumulate_box_derivative!(G::Matrix{Float64}, f::DelegatedContractFactor,
                                      q::AbstractMatrix{Float64}, ctx::F.Context)
    d = T.unwrap_displacement(ctx.box, q, f.i, f.j)
    r = sqrt(T.norm2(d))
    dr = r - f.r0
    c = f.k * dr / r
    w = SC._frac_disp(ctx.box, d)
    SC._outer_add!(G, c, d, w)
    return G
end

function test_contract_errors_and_box_derivative_delegation()
    box = T.OrthorhombicBox((8.0, 8.0, 8.0))
    f_del = DelegatedContractFactor(1, 2, 350.0, 1.2)
    rs = RS.ResolvedSystem(;
        masses = [14.0, 14.0],
        charges = [0.0, 0.0],
        box = box,
        factors = [f_del],
        declared = [:energy, :cartesian_forces, :component_energies, :box_derivative]
    )

    # 1. Error checks
    q = [1.0 2.2; 1.0 1.0; 1.0 1.0]
    p = [0.5 -0.5; 0.0 0.0; 0.0 0.0]

    # Non-finite coordinates throw InvalidGeometry
    q_bad = copy(q); q_bad[1, 1] = NaN
    err = _caught(() -> SC.energy(rs, q_bad))
    @assert err isa RS.InvalidGeometry "expected InvalidGeometry on NaN position"

    # Non-finite momenta throw InvalidGeometry
    p_bad = copy(p); p_bad[1, 1] = Inf
    err = _caught(() -> SC.kinetic_energy(rs, p_bad))
    @assert err isa RS.InvalidGeometry "expected InvalidGeometry on Inf momentum"

    # NumericalFailure on check_numerical_stability
    err = _caught(() -> SC.check_numerical_stability(NaN, zeros(3, 2)))
    @assert err isa RS.NumericalFailure "expected NumericalFailure on NaN energy"

    # 2. Delegation checks
    @assert SC.supports_box_derivative(f_del, box) "DelegatedContractFactor should be supported"
    @assert :box_derivative in RS.engine_capabilities(rs)

    G = SC.box_derivative(rs, q)
    @assert all(isfinite, G)
    G_comp = SC.box_derivative_component(rs, q, :component_1)
    @assert G == G_comp "component and total box derivative must match"

    # Virial identity
    h = SC._hmat(box)
    Xi = SC.virial(rs, q)
    @assert _relmax(Xi, -(h * transpose(G))) == 0.0 "virial identity Xi = -h G' must hold"

    # Affine pressure
    V = T.box_volume(box)
    p_aff = SC.affine_pressure(rs, q)
    @assert isapprox(p_aff, -tr(G * transpose(h)) / (3.0 * V); atol = 1e-14)

    # FD verification against energy
    s = SC.fractional_coordinates(rs, q)
    eps = 1e-4
    for a in 1:3
        L_p = collect(box.L); L_p[a] += eps
        b_p = T.OrthorhombicBox((L_p[1], L_p[2], L_p[3]))
        q_p = SC.cartesian_coordinates(RS.restart_system(rs; box = b_p), s)
        u_p = SC.energy(RS.restart_system(rs; box = b_p), q_p)

        L_m = collect(box.L); L_m[a] -= eps
        b_m = T.OrthorhombicBox((L_m[1], L_m[2], L_m[3]))
        q_m = SC.cartesian_coordinates(RS.restart_system(rs; box = b_m), s)
        u_m = SC.energy(RS.restart_system(rs; box = b_m), q_m)

        g_fd = (u_p - u_m) / (2.0 * eps)
        @assert abs(G[a, a] - g_fd) < 1e-8 "FD mismatch in delegated factor box derivative"
    end
    println("  contract errors and delegated box derivative: verified")
    return true
end

# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

println("=" ^ 72)
println("SystemContract tests")
println("=" ^ 72)
for (name, fn) in (
        ("nobox_local_factors", test_nobox_local_factors),
        ("fractional_chain_rule", test_fractional_chain_rule),
        ("exact_bond_box_derivative", test_exact_bond_box_derivative),
        ("affine_pressure_isotropic_scaling", test_affine_pressure_isotropic_scaling),
        ("virial_identity", test_virial_identity),
        ("periodic_component_box_derivatives", test_periodic_component_box_derivatives),
        ("named_components", test_named_components),
        ("streamed_lj_contract", test_streamed_lj_contract),
        ("streamed_pair_weights", test_streamed_pair_weights),
        ("streamed_cutoff_and_shift_refusal", test_streamed_cutoff_and_shift_refusal),
        ("pme_contract", test_pme_contract),
        ("triclinic_local_and_refusal", test_triclinic_local_and_refusal),
        ("adapter_preservation", test_adapter_preservation),
        ("nobox_and_virtual_site_refusals", test_nobox_and_virtual_site_refusals),
        ("contract_errors_and_box_derivative_delegation", test_contract_errors_and_box_derivative_delegation))
    println("TEST ", name)
    fn()
end
println("")
println("SYSTEM CONTRACT TESTS PASSED")
