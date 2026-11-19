"""
Lane A — derivatives.

Cross-checks the engine's exact bonded force kernels against three independent
derivative sources:

1. **TaylorSeries** — first-order Taylor coefficients of the *same* potential
   law, an independent implementation pinned to the engine by an energy-parity
   assertion before any derivative is compared.
2. **Enzyme** — reverse-mode autodiff of the engine's own
   `total_energy_forces` code path (runtime activity + relaxed aliasing are
   required: the engine holds an `AbstractBox` field inside `Context`).
3. **Central finite differences** of the engine energy along one direction,
   used only as a low-order sanity check.

It also uses TaylorSeries to differentiate the rigid-water constraint distance
along a *projected* velocity, i.e. it checks the tangency condition
`r · (v_j - v_i) = 0` of `Constraints.project_velocities!` at first order.
"""

using TaylorSeries
using Enzyme

struct DerivativeMetrics
    energy_parity::Float64
    taylor_force_err::Float64
    enzyme_force_err::Float64
    fd_direction_err::Float64
    tangent_derivative::Float64
end

"First-order Taylor coefficients of `chain_energy` (i.e. `dU/dq_k`), exactly."
function taylor_gradient(q::AbstractMatrix{Float64}, p = CHAIN)
    g = similar(q)
    t = Taylor1(Float64, 1)
    for k in eachindex(q)
        qt = q .+ zero(t)
        qt[k] += t
        g[k] = chain_energy(qt, p)[1]
    end
    return g
end

"Reverse-mode gradient of the engine's own energy path."
function enzyme_gradient(q::AbstractMatrix{Float64})
    energy = x -> chain_energy_forces(x)[1]
    Enzyme.API.strictAliasing!(false)
    grad = Enzyme.gradient(Enzyme.set_runtime_activity(Enzyme.Reverse), energy, q)
    return grad[1]
end

"Deterministic, non-degenerate perturbations of the chain fixture."
function chain_configurations()
    q = chain_positions()
    na = size(q, 2)
    bump1 = [0.03 * sin(1.7 * i + 0.4 * j) for i in 1:3, j in 1:na]
    bump2 = [-0.02 * cos(0.9 * i - 1.1 * j) for i in 1:3, j in 1:na]
    return [q, q .+ bump1, q .+ bump2]
end

"Forces are the negative gradient; compare componentwise on a common scale."
force_scale(F) = max(1.0, maximum(abs, F))

function taylor_force_error(q::AbstractMatrix{Float64}, F::AbstractMatrix{Float64})
    g = taylor_gradient(q)
    return maximum(abs, g .+ F) / force_scale(F)
end

function enzyme_force_error(q::AbstractMatrix{Float64}, F::AbstractMatrix{Float64})
    g = enzyme_gradient(q)
    return maximum(abs, g .+ F) / force_scale(F)
end

"`-F · d` against a central difference of the engine energy along `d`."
function fd_direction_error(q::AbstractMatrix{Float64}, F::AbstractMatrix{Float64})
    d = [sin(0.7 * i + 0.3 * j) for i in 1:3, j in 1:size(q, 2)]
    d ./= norm(d)
    h = 1e-6
    up, _ = chain_energy_forces(q .+ h .* d)
    down, _ = chain_energy_forces(q .- h .* d)
    fd = (up - down) / (2h)
    analytic = sum(F .* d)          # F·d = -dU/dq·d
    return abs(fd + analytic) / max(1.0, abs(analytic), abs(fd))
end

"""
First-order change of a rigid-water O-H distance along a projected velocity.

`Constraints.project_velocities!` should leave `r · (v_j - v_i) = 0`, so the
Taylor coefficient of `|x_j(t) - x_i(t)|` at `t = 0` must vanish.
"""
function constraint_tangent_derivative()
    cs, q = water_constraints()
    m = [16.0, 1.0, 1.0]
    p = [0.4 -0.2 0.7; -0.3 0.5 -0.1; 0.2 0.1 -0.6]   # deliberately non-tangent
    qs = copy(q)
    ConstraintLib.project_positions!(qs, m, cs; tolerance = 1e-14)
    ConstraintLib.project_velocities!(qs, p, m, cs; tolerance = 1e-14)

    v = p ./ reshape(m, 1, :)          # v[:, j] = p[:, j] / m[j]
    t = Taylor1(Float64, 1)
    worst = 0.0
    for (i, j, _) in ((1, 2, 0.0), (1, 3, 0.0))
        ri = [qs[1, j] + t * v[1, j] - qs[1, i] - t * v[1, i],
              qs[2, j] + t * v[2, j] - qs[2, i] - t * v[2, i],
              qs[3, j] + t * v[3, j] - qs[3, i] - t * v[3, i]]
        r2 = ri[1]^2 + ri[2]^2 + ri[3]^2
        r = sqrt(r2)
        worst = max(worst, abs(r[1]))
    end
    return worst
end

function run_derivatives()
    parity = 0.0
    taylor_err = 0.0
    enzyme_err = 0.0
    fd_err = 0.0
    for q in chain_configurations()
        U, F = chain_energy_forces(q)
        isfinite(U) || error("chain fixture produced non-finite energy")
        parity = max(parity, abs(chain_energy(q) - U) / max(1.0, abs(U)))
        taylor_err = max(taylor_err, taylor_force_error(q, F))
        enzyme_err = max(enzyme_err, enzyme_force_error(q, F))
        fd_err = max(fd_err, fd_direction_error(q, F))
    end
    tangent = constraint_tangent_derivative()
    return DerivativeMetrics(parity, taylor_err, enzyme_err, fd_err, tangent)
end
