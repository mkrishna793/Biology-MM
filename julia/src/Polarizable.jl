"""
Extensible polarizable energy: a validated induced-dipole (Drude/Applequist) model.

## The model, and what is and is not being claimed

Permanent + induced dipole energy for `N` polarizable sites with induced dipoles
`μ ∈ R^{3N}`:

    U(q, μ) = U_perm(q) + ½ μᵀ A(q) μ − μᵀ E(q)

with `A` the (symmetric) dipole interaction / inverse-polarizability operator and
`E` the field from permanent charges and dipoles. At fixed `q` the minimisation
over `μ` gives the linear system

    A(q) μ = E(q)

whose solution is the self-consistent induced dipole. Two facts about this are
used carefully:

1. **Stationary-point forces.** The force on the nuclei is `−dU/dq` *at the
   minimiser*. By the envelope theorem, the `∂μ/∂q` term drops out **only when
   `A μ = E` holds to the stated tolerance**. If the solve is not converged, the
   analytic "stationary" force is *not* the gradient of the energy actually being
   evaluated, and the two disagree. `InducedDipoleResult.residual` records the
   actual residual so this is never assumed.
2. **Stability.** `A` must be symmetric positive definite for the minimisation to
   be a minimum. If it is not (strong coupling, short distances, no damping), the
   fixed point is unstable and the induced-dipole model has no solution. That is a
   real physical/model limitation, detected and reported, not hidden.

## Damping

A bare point-dipole `A` is `1/r³`-coupled and diverges at short range; real
models use Thole damping or a smeared charge distribution. A simple exponential
Thole-style damping is provided (`thole_damping = true`) because without it `A`
is frequently indefinite at contact distances. This is *one* damping family, not
the only or the best one.

## Standard, not universal

AMOEBA and Drude models already exist in mature engines (OpenMM has both). This
module is a small, independently tested induced-dipole core for the
*extensibility interface*, not a replacement for those parameterisations, and it
does not model charge transfer, metallic bonding, spin states, or coordination
chemistry. See `extension_boundaries()`.
"""
module Polarizable

using ..Types
using ..Factors
using LinearAlgebra

export InducedDipoleSystem, InducedDipoleResult, solve_dipoles!, scf_solve_dipoles!,
       induced_energy, induced_forces, analytic_induced_forces, dipole_residual, A_matrix, thole_damping_factor,
       extension_boundaries, permanent_energy_forces

# ---------------------------------------------------------------------------
# system
# ---------------------------------------------------------------------------

"""
Polarizable system: point charges `z` plus isotropic polarizabilities `α` at
`N` sites, in an orthorhombic (or non-periodic) box.

`α` is in volume units (reduced units: `α` has units of `σ³`). A site with
`α = 0` is non-polarizable and still contributes its charge field.
"""
struct InducedDipoleSystem
    q::Matrix{Float64}      # 3 x N permanent site positions
    z::Vector{Float64}      # permanent charges
    α::Vector{Float64}      # polarizabilities (>= 0)
    box::Types.AbstractBox
    units::Types.Units
    thole::Bool
    a_thole::Float64        # Thole damping radius, in length units
    periodic::Bool
    function InducedDipoleSystem(q, z, α, box, units; thole::Bool = true, a_thole::Real = 1.0, periodic::Bool = false)
        if !periodic && !(box isa Types.NoBox)
            throw(Types.UnsupportedGeometry(
                "Polarizable does not implement periodic polarization (minimum-image dipoles " *
                "without Ewald/PME is an unsupported periodic mode; got $(typeof(box)))"))
        end
        if periodic && !(box isa Types.NoBox || box isa Types.OrthorhombicBox || box isa Types.TriclinicBox)
            throw(Types.UnsupportedGeometry("Unsupported periodic box type $(typeof(box))"))
        end
        q = Matrix{Float64}(q); z = Vector{Float64}(z); α = Vector{Float64}(α)
        N = size(q, 2)
        length(z) == N || error("charge count mismatch")
        length(α) == N || error("polarizability count mismatch")
        all(x -> x >= 0, α) || error("polarizabilities must be non-negative")
        a_thole > 0 || error("Thole damping radius must be positive")
        new(q, z, α, box, units, thole, Float64(a_thole), periodic)
    end
    function InducedDipoleSystem(q, z, α, box, units, thole::Bool, a_thole::Real, periodic::Bool = false)
        InducedDipoleSystem(q, z, α, box, units; thole = thole, a_thole = a_thole, periodic = periodic)
    end
end

Nsites(s::InducedDipoleSystem) = size(s.q, 2)

# ---------------------------------------------------------------------------
# dipole interaction operator
# ---------------------------------------------------------------------------

"""
Thole-style exponential damping factor for a screened dipole interaction at
separation `u = r/a`.

Returns `(f_T, f_E)` used to scale the dipole–dipole tensor and the field. The
functional form is the simple exponential screening family; it is a *model
choice*, documented as such, not derived.

**What the test actually establishes.** An earlier version of this docstring
claimed that damping makes `A` positive definite at contact distances. The test
falsified that: at `r = 0.8` with `α = 2` the damped `A` still has a negative
eigenvalue. What damping does is *shift the spectrum upward* (measured:
-5.02 → -1.82 for that case), which widens the stable range but is not a
guarantee. Positive definiteness is therefore **checked and reported**, never
assumed, and an indefinite `A` is surfaced to the caller as `spd = false` rather
than being regularised away. Real parameterisations tune `a_thole` per site pair;
this module exposes the knob without pretending one default is universally stable.
"""
function thole_damping_factor(u::Float64)
    u <= 0 && return (0.0, 0.0)
    # exponential damping: the 1/r^3 tensor is multiplied by (1 - exp(-a u^3)),
    # which vanishes as u -> 0 and tends to 1 for large u.
    a = 2.1304  # standard exponential-Thole coefficient used with u = r/(α_i α_j)^(1/6)
    g = 1.0 - exp(-a * u^3)
    return (g, g)
end

"""
Build `A`, the `3N x 3N` dipole operator:

    A_{ij} = δ_{ij} / α_i  −  T_{ij}
    T_{ij} = k_e(3 r̂ r̂ᵀ − I)/r³ · g(r)

`T` is the standard dipole–dipole tensor scaled by the damping factor `g`.

For a non-periodic or orthorhombic system the pair sum is over minimum images;
this is a **direct** operator, `O(N²)` in pairs — no fast multipole for
polarizability is implemented, and that is a stated limitation.

**`A` is not guaranteed positive definite.** Strong coupling, short separations,
or large polarizabilities make it indefinite, in which case there is no stable
induced-dipole solution. `solve_dipoles!` reports this via `spd` and the smallest
eigenvalue instead of regularising it.
"""
function A_matrix(s::InducedDipoleSystem; include_self::Bool = true)
    N = Nsites(s)
    A = zeros(3N, 3N)
    if include_self
        for i in 1:N
            if s.α[i] > 0
                for d in 1:3
                    A[3(i-1)+d, 3(i-1)+d] = 1.0 / s.α[i]
                end
            else
                # a non-polarizable site is not part of the induced system; make
                # its block a large penalty so the solve returns μ ≈ 0 there and
                # the matrix stays definite.
                for d in 1:3
                    A[3(i-1)+d, 3(i-1)+d] = 1e12
                end
            end
        end
    end
    for i in 1:N-1
        for j in i+1:N
            dd = Types.unwrap_displacement(s.box, s.q, i, j)
            r2 = dd.x^2 + dd.y^2 + dd.z^2
            r = sqrt(r2)
            r < 1e-12 && throw(ArgumentError("coincident sites $i and $j are singular in the point-dipole model"))
            aij = (max(s.α[i], 1e-30) * max(s.α[j], 1e-30))^(1 / 6)
            u = r / (s.a_thole > 0 ? max(s.a_thole, aij) : aij)
            g = s.thole ? thole_damping_factor(u)[1] : 1.0
            rhat = [dd.x / r, dd.y / r, dd.z / r]
            T = zeros(3, 3)
            pre = s.units.coulomb_prefactor
            for a in 1:3, b in 1:3
                # T_true = k_e (3 rhat*rhat' - I) g(r) / r^3; A_ij = -T_true.
                T[a, b] = (pre * g / r^3) * (3.0 * rhat[a] * rhat[b] - (a == b ? 1.0 : 0.0))
            end
            for a in 1:3, b in 1:3
                A[3(i-1)+a, 3(j-1)+b] -= T[a, b]
                A[3(j-1)+b, 3(i-1)+a] -= T[a, b]
            end
        end
    end
    return A
end

# ---------------------------------------------------------------------------
# field from permanent charges
# ---------------------------------------------------------------------------

"""
Field `E_i` at each site from all *permanent* charges: `E_i = Σ_{j≠i} z_j r̂ / r²`.

Sign convention: `E` is the field that polarises `μ`, so the energy is `−μ·E`.
No external field term is included; adding one is a one-line extension and is
listed in `extension_boundaries()`.
"""
function permanent_field(s::InducedDipoleSystem)
    N = Nsites(s)
    E = zeros(3N)
    for i in 1:N
        for j in 1:N
            i == j && continue
            dd = Types.unwrap_displacement(s.box, s.q, i, j)
            r2 = dd.x^2 + dd.y^2 + dd.z^2
            r2 < 1e-24 && throw(ArgumentError("coincident sites $i and $j are singular in the permanent-charge field"))
            r = sqrt(r2)
            c = s.units.coulomb_prefactor * s.z[j] / (r2 * r)
            E[3(i-1)+1] += c * dd.x
            E[3(i-1)+2] += c * dd.y
            E[3(i-1)+3] += c * dd.z
        end
    end
    return E
end

# ---------------------------------------------------------------------------
# solve
# ---------------------------------------------------------------------------

struct InducedDipoleResult
    μ::Vector{Float64}         # induced dipoles, 3N
    residual::Float64          # ‖Aμ − E‖ / max(‖E‖, eps)
    n_iter::Int
    method::Symbol
    spd::Bool
    min_eig::Float64
    converged::Bool
    status::Symbol
end

InducedDipoleResult(μ, residual, n_iter, method, spd, min_eig, converged) =
    InducedDipoleResult(μ, residual, n_iter, method, spd, min_eig, converged,
                        !spd ? :indefinite : (converged ? :converged : :not_converged))

"""
Solve `A μ = E` for the induced dipoles.

Methods supported:
- `:cholesky`: Direct Cholesky decomposition when `A` is SPD, followed by iterative refinement.
- `:scf`: Self-Consistent Field (fixed-point / Jacobi) iteration: `μ_{k+1} = D^-1 (E + T μ_k)`.
- `:cg`: Preconditioned Conjugate Gradients.

Reports `InducedDipoleResult` with convergence status (`:converged`, `:not_converged`, `:indefinite`),
scaled residual `‖Aμ − E‖ / max(‖E‖, eps)`, iteration count, SPD status, and minimum eigenvalue.
"""
function solve_dipoles!(s::InducedDipoleSystem; tol::Float64 = 1e-10,
                        check_spd::Bool = true, maxiter::Int = 200,
                        method::Symbol = :cholesky)
    A = A_matrix(s)
    E = permanent_field(s)
    n = length(E)
    if n == 0
        return InducedDipoleResult(Float64[], 0.0, 0, :empty, true, NaN, true, :converged)
    end
    normE = max(norm(E), eps(Float64))
    mineig = NaN

    F = cholesky(Symmetric(A); check = false)
    spd = issuccess(F)
    if check_spd || !spd
        ev = eigvals(Symmetric(A))
        mineig = minimum(ev)
        spd = mineig > 0
    end

    if !spd
        sol_method = :lstsq
        μ = A \ E
        resid = norm(A * μ - E) / normE
        return InducedDipoleResult(μ, resid, 0, sol_method, false, mineig, false, :indefinite)
    end

    if method === :scf
        μ = zeros(n)
        actual_iter = 0
        for it in 1:maxiter
            actual_iter = it
            res = norm(A * μ - E) / normE
            if res <= tol
                return InducedDipoleResult(μ, res, actual_iter, :scf, true, mineig, true, :converged)
            end
            r_vec = E - A * μ
            for i in 1:n
                μ[i] += r_vec[i] / A[i, i]
            end
        end
        resid = norm(A * μ - E) / normE
        conv = resid <= tol
        status = conv ? :converged : :not_converged
        return InducedDipoleResult(μ, resid, actual_iter, :scf, true, mineig, conv, status)
    elseif method === :cg
        μ = zeros(n)
        r_vec = copy(E)
        d_inv = [1.0 / A[i, i] for i in 1:n]
        z = r_vec .* d_inv
        p = copy(z)
        rz = dot(r_vec, z)
        actual_iter = 0
        if norm(r_vec) / normE <= tol
            return InducedDipoleResult(μ, norm(r_vec) / normE, 0, :cg, true, mineig, true, :converged)
        end
        for it in 1:maxiter
            actual_iter = it
            Ap = A * p
            curv = dot(p, Ap)
            if curv <= 0
                return InducedDipoleResult(μ, Inf, it, :cg, false, mineig, false, :indefinite)
            end
            alpha_step = rz / curv
            μ .+= alpha_step .* p
            r_vec .-= alpha_step .* Ap
            rel_res = norm(r_vec) / normE
            if rel_res <= tol
                return InducedDipoleResult(μ, rel_res, actual_iter, :cg, true, mineig, true, :converged)
            end
            z = r_vec .* d_inv
            rz_new = dot(r_vec, z)
            beta = rz_new / rz
            p .= z .+ beta .* p
            rz = rz_new
        end
        resid = norm(A * μ - E) / normE
        conv = resid <= tol
        status = conv ? :converged : :not_converged
        return InducedDipoleResult(μ, resid, actual_iter, :cg, true, mineig, conv, status)
    else # :cholesky
        sol_method = :cholesky
        μ = F \ E
        actual_iter = 1
        for it in 1:maxiter
            res = norm(A * μ - E) / normE
            res <= tol && break
            δ = A \ (E - A * μ)
            μ .+= δ
            actual_iter += 1
        end
        resid = norm(A * μ - E) / normE
        conv = resid <= tol
        status = conv ? :converged : :not_converged
        return InducedDipoleResult(μ, resid, actual_iter, sol_method, true, mineig, conv, status)
    end
end

"""
    scf_solve_dipoles!(s::InducedDipoleSystem; tol=1e-10, maxiter=200)

Solve for induced dipoles using Self-Consistent Field (SCF) iterations, reporting
convergence status, residual, iterations, and SPD verification.
"""
scf_solve_dipoles!(s::InducedDipoleSystem; kwargs...) = solve_dipoles!(s; method = :scf, kwargs...)

dipole_residual(r::InducedDipoleResult) = r.residual

# ---------------------------------------------------------------------------
# energies
# ---------------------------------------------------------------------------

"""
Permanent-charge Coulomb energy (the `U_perm` term), plus its forces.

Kept as its own function because it is the part that exists whether or not the
polarization solve succeeds, and because it lets the tests separate the two
contributions.
"""
function permanent_energy_forces(s::InducedDipoleSystem)
    N = Nsites(s)
    F = zeros(3, N)
    U = 0.0
    pre = s.units.coulomb_prefactor
    for i in 1:N-1
        for j in i+1:N
            dd = Types.unwrap_displacement(s.box, s.q, i, j)
            r2 = dd.x^2 + dd.y^2 + dd.z^2
            r2 < 1e-24 && (U = Inf; continue)
            r = sqrt(r2)
            U += pre * s.z[i] * s.z[j] / r
            c = pre * s.z[i] * s.z[j] / (r2 * r)
            F[1, i] += c * dd.x; F[2, i] += c * dd.y; F[3, i] += c * dd.z
            F[1, j] -= c * dd.x; F[2, j] -= c * dd.y; F[3, j] -= c * dd.z
        end
    end
    return U, F
end

"""
Induced-dipole energy at a *given* `μ`: `½ μᵀAμ − μᵀE`.

Evaluated with the same `A` and `E` used in the solve, so no inconsistency can
creep in between the energy and the stationarity condition.
"""
function induced_energy(s::InducedDipoleSystem, μ::AbstractVector{Float64};
                        A = nothing, E = nothing)
    Amat = A === nothing ? A_matrix(s) : A
    Evec = E === nothing ? permanent_field(s) : E
    return 0.5 * dot(μ, Amat * μ) - dot(μ, Evec)
end

"""
Total polarizable energy at the converged (or not) solution:

    U = U_perm + ½μᵀAμ − μᵀE

The result carries the residual, so a caller can see whether the stationary-point
force is valid.
"""
function induced_energy(s::InducedDipoleSystem, r::InducedDipoleResult)
    r.spd || throw(ArgumentError("induced energy unavailable: dipole operator is not certified positive definite"))
    r.converged || throw(ArgumentError("induced energy unavailable: dipole solve did not converge (residual=$(r.residual))"))
    Uperm, _ = permanent_energy_forces(s)
    Uind = induced_energy(s, r.μ)
    return Uperm + Uind
end

# ---------------------------------------------------------------------------
# forces
# ---------------------------------------------------------------------------

"""
    analytic_induced_forces(s::InducedDipoleSystem, r::InducedDipoleResult) -> Matrix{Float64}

Nuclear forces `-dU/dq` at the stationary induced dipoles `μ`, including both the
permanent charge forces and the analytic Hellmann-Feynman gradient of the induced
dipole energy at fixed `μ`:
    U = U_perm + ½ μᵀ A(q) μ − μᵀ E(q)
Runs in O(N²) time without requiring finite-difference evaluations.
"""
function analytic_induced_forces(s::InducedDipoleSystem, r::InducedDipoleResult)
    r.spd || throw(ArgumentError("induced forces unavailable: dipole operator is not certified positive definite"))
    r.converged || throw(ArgumentError("induced forces unavailable: dipole solve did not converge (residual=$(r.residual))"))
    N = Nsites(s)
    F = zeros(Float64, 3, N)
    pre = s.units.coulomb_prefactor

    # Permanent charge force contribution
    _, Fperm = permanent_energy_forces(s)
    F .+= Fperm

    for i in 1:N-1
        mi = (r.μ[3(i-1)+1], r.μ[3(i-1)+2], r.μ[3(i-1)+3])
        for j in i+1:N
            mj = (r.μ[3(j-1)+1], r.μ[3(j-1)+2], r.μ[3(j-1)+3])
            dd = Types.unwrap_displacement(s.box, s.q, i, j)
            r2 = dd.x^2 + dd.y^2 + dd.z^2
            r2 < 1e-24 && continue
            r_dist = sqrt(r2)

            # 1. Field term contribution: -μᵀ E(q)
            # Pair interaction: -ke * (z_j * μ_i - z_i * μ_j) · r / r^3
            # With p = z_j * μ_i - z_i * μ_j:
            # F_field = ke * [ p / r^3 - 3 (p · r) r / r^5 ]
            px = s.z[j] * mi[1] - s.z[i] * mj[1]
            py = s.z[j] * mi[2] - s.z[i] * mj[2]
            pz = s.z[j] * mi[3] - s.z[i] * mj[3]
            p_dot_r = px * dd.x + py * dd.y + pz * dd.z

            c_field = pre / (r2 * r_dist)
            t_field = 3.0 * pre / (r2 * r2 * r_dist)

            fx_field = c_field * px - t_field * p_dot_r * dd.x
            fy_field = c_field * py - t_field * p_dot_r * dd.y
            fz_field = c_field * pz - t_field * p_dot_r * dd.z

            # 2. Dipole-dipole term contribution: ½ μᵀ A(q) μ
            # Pair interaction: -μ_iᵀ T_ij μ_j
            g = 1.0
            gprime = 0.0
            if s.thole
                aij = (max(s.α[i], 1e-30) * max(s.α[j], 1e-30))^(1 / 6)
                a_eff = s.a_thole > 0 ? max(s.a_thole, aij) : aij
                u = r_dist / a_eff
                s_val = 2.1304 * u^3
                g = 1.0 - exp(-s_val)
                gprime = (3.0 * s_val / r_dist) * exp(-s_val)
            end

            d1 = mi[1] * dd.x + mi[2] * dd.y + mi[3] * dd.z
            d2 = mj[1] * dd.x + mj[2] * dd.y + mj[3] * dd.z
            d12 = mi[1] * mj[1] + mi[2] * mj[2] + mi[3] * mj[3]

            C1 = -3.0 * pre * g / (r2 * r2 * r_dist)
            C1prime = 3.0 * pre * (5.0 * g - r_dist * gprime) / (r2 * r2 * r2)
            C2prime = -pre * (3.0 * g - r_dist * gprime) / (r2 * r2)

            scalar_factor = (C1prime * d1 * d2 + C2prime * d12) / r_dist

            fx_dip = -(scalar_factor * dd.x + C1 * (d2 * mi[1] + d1 * mj[1]))
            fy_dip = -(scalar_factor * dd.y + C1 * (d2 * mi[2] + d1 * mj[2]))
            fz_dip = -(scalar_factor * dd.z + C1 * (d2 * mi[3] + d1 * mj[3]))

            f_tot_x = fx_field + fx_dip
            f_tot_y = fy_field + fy_dip
            f_tot_z = fz_field + fz_dip

            F[1, i] += f_tot_x; F[2, i] += f_tot_y; F[3, i] += f_tot_z
            F[1, j] -= f_tot_x; F[2, j] -= f_tot_y; F[3, j] -= f_tot_z
        end
    end
    return F
end

"""
Forces via the stationary-point (envelope) derivative.

At `Aμ = E`, `dU/dq = ∂U/∂q|_{μ fixed}`, so the force is `−∂U/∂q` holding `μ`
constant. When `analytic = true`, computes exact forces via `analytic_induced_forces`.
Otherwise evaluates via central finite differences of the quadratic form at fixed `μ`.
"""
function induced_forces(s::InducedDipoleSystem, r::InducedDipoleResult;
                        h::Float64 = 1e-6, analytic::Bool = false)
    if analytic
        return analytic_induced_forces(s, r)
    end
    r.spd || throw(ArgumentError("induced forces unavailable: dipole operator is not certified positive definite"))
    r.converged || throw(ArgumentError("induced forces unavailable: dipole solve did not converge (residual=$(r.residual))"))
    N = Nsites(s)
    F = zeros(3, N)
    A = A_matrix(s)
    E = permanent_field(s)
    # d/dq of ½μᵀAμ − μᵀE at fixed μ, by central differences of the quadratic form
    base = induced_energy(s, r.μ; A = A, E = E)
    for j in 1:N, d in 1:3
        qp = copy(s.q); qp[d, j] += h
        qm = copy(s.q); qm[d, j] -= h
        sp = InducedDipoleSystem(qp, s.z, s.α, s.box, s.units;
                                 thole = s.thole, a_thole = s.a_thole, periodic = s.periodic)
        sm = InducedDipoleSystem(qm, s.z, s.α, s.box, s.units;
                                 thole = s.thole, a_thole = s.a_thole, periodic = s.periodic)
        Up = induced_energy(sp, r.μ)
        Um = induced_energy(sm, r.μ)
        F[d, j] = -(Up - Um) / (2h)
    end
    # add the permanent-charge force
    _, Fperm = permanent_energy_forces(s)
    return F .+ Fperm
end

# ---------------------------------------------------------------------------
# boundaries
# ---------------------------------------------------------------------------

"""
What this model does **not** cover. Returned as data so it can be printed in a
report or a status matrix rather than living only in prose.

Polarization is not a theory of metals. Real metallic systems need band
structure, charge transfer, spin, and screening; those require QM/MM or a
parameterised many-body potential, not an induced-dipole correction. Saying
otherwise would be a category error.
"""
extension_boundaries() = [
    (capability = "charge transfer between sites", supported = false,
     note = "requires a fluctuating-charge / QEq model, not fixed polarizability"),
    (capability = "metallic bonding / band structure", supported = false,
     note = "needs QM/MM or a many-body potential; induced dipoles are the wrong model"),
    (capability = "spin states and magnetism", supported = false,
     note = "not representable in a classical dipole model"),
    (capability = "Thole damping families other than exponential", supported = false,
     note = "interface allows replacement; only one family implemented and tested"),
    (capability = "periodic Ewald sum for dipoles", supported = false,
     note = "operator is direct O(N^2) minimum image; Ewald dipoles are a separate model"),
    (capability = "external electric field", supported = false,
     note = "a one-line addition to E; not implemented here"),
    (capability = "analytic tensor derivatives for O(N) force", supported = false,
     note = "forces use finite differences of the quadratic form; correct but not fast"),
    (capability = "AMOEBA/Drude parameterisation", supported = false,
     note = "use OpenMM, which supports both; this is an interface, not a force field"),
]

end # module Polarizable
