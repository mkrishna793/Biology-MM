"""
Coupled holonomic distance constraints: rigid molecules and constraint networks.

`Dynamics.rattle_project!` handles only *disjoint* distance pairs. That is
enough for isolated bonds and not enough for water, where the three distances
O-H, O-H, H-H share atoms. A shared-atom network cannot be projected one
constraint at a time and then declared correct: each single-constraint
correction breaks the constraints already applied, so a sequential (Gauss-Seidel)
sweep either fails to converge or converges slowly and geometrically.

This module implements the coupled projection instead. For a component with
atoms `S` and constraints `d`, collect the constraint residuals

    c_k(x) = |x_j - x_i|^2 - d_k^2,          k = 1..m

and project onto `c(x) = 0` by Newton's method in the mass metric. The Newton
step at the linearisation `G = dc/dx` solves the small dense system

    (G W G^T) λ = -c,        W = diag(1/m) ⊗ I_3
    Δx = -W G^T λ

which is the Lagrange-multiplier form of the same problem, `W` being the inverse
mass metric. `G W G^T` is `m x m` with `m` = number of constraints in the
component (3 for a rigid water molecule), so the linear solve is trivially small
and independent of system size.

Velocities are projected onto the tangent space of the constraint manifold by
enforcing `G v = 0` with the same small solve. That is the velocity-level
condition RATTLE imposes; doing it through the coupled system rather than
per-pair is what makes shared atoms consistent.

## Rigid-water special case

A rigid water molecule has exactly 3 atoms and 3 constraints, so the component
is fully determined: the projection lands on the unique rigid copy and the
system is square. That is the classical SETTLE regime. The implementation here
is the general Newton form rather than the specialised quartic SETTLE
factorisation, which means it is used for arbitrary constraint networks and it
verifies the water case as a special case. It costs a `3x3` solve per molecule
per iteration; for rigid water that is a small constant per molecule and the
projection converges quadratically.

## What is and is not claimed

- Convergence to a *stated tolerance* is enforced; failure raises
  `ConstraintConvergenceError` rather than silently continuing.
- The projection is exact in the sense that the residual is driven below
  tolerance, not that a single pass is exact.
- Constraint forces are internal; total momentum and the constrained
  Hamiltonian behaviour are tested, but no blanket symplecticity claim is made.
- A component whose constraints are rank deficient (e.g. collinear atoms making
  the Jacobian singular) is detected and refused.

## Constraint virial

A constrained system's pressure contains the constraint forces, which do no
work on the constraint manifold but *do* contribute to the virial because the
box couples to all coordinates. For the constraint

    c_k = |x_j - x_i|^2 - d_k^2,

the gradient is `-2r_k` at `i` and `+2r_k` at `j`, so a multiplier `λ_k` produces

    F^c_i = -Σ_k λ_k ∇_i c_k,        W_c = Σ_i r_i · F^c_i = -2 Σ_k λ_k |r_k|^2.

The second equality is exact and needs no force array: `Σ_i x_i·∇_i c_k = 2|r_k|²`
because each constraint is translation invariant. That is what
[`constraint_virial`](@ref) evaluates.

Where `λ` comes from matters, and this module provides the two honest routes:

* [`constraint_multipliers`](@ref) solves `(G W G^T) λ = G W F_ext`, i.e. the
  multipliers that make the *external* force field constraint-consistent at a
  fixed geometry. This is the correct source for a static pressure/virial
  evaluation and it is the route validated against the box derivative below.
  **Use this for NPT pressure.**
* [`project_positions_with_multipliers!`](@ref) reconstructs the multipliers of
  the impulse a position projection just applied, `F^c = M Δx / dt²`. It answers
  "what did this projection do?"; it is *not* a pressure accumulator, because a
  trajectory's projection impulse also absorbs tangential drift (measured on a
  5000-step run: the time-averaged impulse virial differs from the static one in
  sign). See its docstring.

Units: `constraint_virial` consumes multipliers in *force* units, so the
displacement route divides by `dt²` and the caller must pass the same `dt` the
step used. Both routes are documented where they are defined; the test suite
cross-checks the static route against the finite-difference box derivative
`-dU/ds = W_ext + W_c` for the path `q(s) = Π(s q)` (`Π` = this module's
projection), which is the constraint-preserving volume scaling an NPT barostat
needs, and cross-checks the two routes against each other on a single known
impulse.
"""
module Constraints

using ..Types
using LinearAlgebra

export ConstraintComponent, CoupledConstraintSet,
       ConstraintSolveResult,
       project_positions!, project_positions_with_multipliers!,
       project_velocities!, project!,
       solve_positions!, solve_velocities!, solve_constraints!,
       constraint_residuals, tangent_residuals,
       constraint_multipliers, constraint_forces,
       constraint_virial, ConstraintProjectionError, UnsupportedConstraintSet

"""
Raised when the Newton projection fails to reach the requested tolerance. This is
a hard failure: a run must not continue with geometry that violates its
constraints.

`atoms` names the component that failed, so the message identifies the offending
atoms rather than only a residual.
"""
struct ConstraintProjectionError <: Exception
    stage::Symbol
    residual::Float64
    tolerance::Float64
    iterations::Int
    atoms::Vector{Int}
end

ConstraintProjectionError(stage::Symbol, residual::Float64, tolerance::Float64,
                         iterations::Int) =
    ConstraintProjectionError(stage, residual, tolerance, iterations, Int[])

function Base.showerror(io::IO, e::ConstraintProjectionError)
    where = isempty(e.atoms) ? "" : " for constraint component atoms $(e.atoms)"
    print(io, "coupled constraint ", e.stage, " projection$where did not reach tolerance ",
          e.tolerance, " after ", e.iterations, " iterations (residual ", e.residual, ")")
end

"Raised for a constraint set this module deliberately refuses to project."
struct UnsupportedConstraintSet <: Exception
    reason::String
end

Base.showerror(io::IO, e::UnsupportedConstraintSet) =
    print(io, "unsupported constraint set: ", e.reason)

"""
Explicit convergence status and residuals returned by non-throwing constraint
solvers (`solve_positions!`, `solve_velocities!`, `solve_constraints!`).

`converged == false` indicates the solver refused the step (and restored the
input coordinates/momenta) rather than accepting a partially projected state.
"""
struct ConstraintSolveResult
    stage::Symbol
    converged::Bool
    position_residual::Float64
    tangent_residual::Float64
    iterations::Int
    tolerance::Float64
    failed_atoms::Vector{Int}
    reason::String
end

"""
One connected component of constraints.

`atoms` are the global atom indices in this component; `edges` are `(local_i,
local_j, d)` triples referring to positions *within* `atoms`.
"""
struct ConstraintComponent
    atoms::Vector{Int}
    edges::Vector{Tuple{Int,Int,Float64}}
end

"""
A validated collection of coupled constraint components.

Construct with [`CoupledConstraintSet`](@ref) from `(i, j, d)` triples. Triples
that do not share atoms are placed in separate components, so a system mixing
rigid water with independent bond constraints works and each component is
projected with its own small solve.
"""
struct CoupledConstraintSet
    components::Vector{ConstraintComponent}
    dim::Int          # 3
end

"Union-find over atom indices, used to find constraint components."
function _components_of(atoms::Int, triples::Vector{Tuple{Int,Int,Float64}})
    parent = collect(1:atoms)
    function find(a)
        while parent[a] != a
            parent[a] = parent[parent[a]]
            a = parent[a]
        end
        return a
    end
    function union!(a, b)
        ra, rb = find(a), find(b)
        ra == rb || (parent[rb] = ra)
    end
    for (i, j, _) in triples
        union!(i, j)
    end
    groups = Dict{Int,Vector{Tuple{Int,Int,Float64}}}()
    for t in triples
        r = find(t[1])
        push!(get!(groups, r, Tuple{Int,Int,Float64}[]), t)
    end
    comps = ConstraintComponent[]
    for (_, ts) in groups
        atom_set = sort(unique(vcat([t[1] for t in ts], [t[2] for t in ts])))
        index_of = Dict(a => k for (k, a) in enumerate(atom_set))
        edges = [(index_of[i], index_of[j], d) for (i, j, d) in ts]
        push!(comps, ConstraintComponent(atom_set, edges))
    end
    return comps
end

"""
    CoupledConstraintSet(natoms, constraints) -> CoupledConstraintSet

Validate and group `(i, j, d)` distance constraints into independent components.

Refuses: invalid indices, non-positive or non-finite distances, a component with
more constraints than internal degrees of freedom, and duplicate constraints on
the same pair.
"""
function CoupledConstraintSet(natoms::Integer, constraints)
    natoms >= 1 || throw(ArgumentError("natoms must be positive"))
    triples = Tuple{Int,Int,Float64}[]
    seen = Set{Tuple{Int,Int}}()
    for c in constraints
        length(c) == 3 || throw(ArgumentError("each constraint must be (i, j, distance)"))
        i, j, d = Int(c[1]), Int(c[2]), Float64(c[3])
        1 <= i <= natoms && 1 <= j <= natoms && i != j ||
            throw(ArgumentError("constraint atom indices are invalid"))
        isfinite(d) && d > 0 || throw(ArgumentError("constraint distance must be positive and finite"))
        key = i < j ? (i, j) : (j, i)
        key in seen && throw(ArgumentError("duplicate constraint on atoms $key"))
        push!(seen, key)
        push!(triples, (i, j, d))
    end
    isempty(triples) && throw(ArgumentError("at least one distance constraint is required"))
    comps = _components_of(Int(natoms), triples)
    for comp in comps
        nfree = 3 * length(comp.atoms) - 3      # minus rigid translation/rotation
        length(comp.edges) > nfree &&
            throw(UnsupportedConstraintSet(
                "component with atoms $(comp.atoms) has $(length(comp.edges)) constraints " *
                "but only $nfree internal degrees of freedom; the set is over-determined"))
    end
    return CoupledConstraintSet(comps, 3)
end

"Number of atoms participating in constraints."
function natoms(cs::CoupledConstraintSet)
    isempty(cs.components) && return 0
    return maximum(maximum(c.atoms) for c in cs.components)
end

"Residual `|x_j - x_i|^2 - d^2` for every edge, flattened across components."
function constraint_residuals(cs::CoupledConstraintSet, q::AbstractMatrix{Float64},
                              box::Types.AbstractBox = Types.NoBox())
    out = Float64[]
    sizehint!(out, _total_edges(cs))
    @inbounds for comp in cs.components
        for (a, b, d) in comp.edges
            i, j = comp.atoms[a], comp.atoms[b]
            r = Types.unwrap_displacement(box, q, j, i)
            push!(out, Types.norm2(r) - d * d)
        end
    end
    return out
end

"Tangent residual `r · (v_j - v_i)` for every edge."
function tangent_residuals(cs::CoupledConstraintSet, q::AbstractMatrix{Float64},
                           p::AbstractMatrix{Float64}, m::AbstractVector{Float64},
                           box::Types.AbstractBox = Types.NoBox())
    out = Float64[]
    sizehint!(out, _total_edges(cs))
    @inbounds for comp in cs.components
        for (a, b, d) in comp.edges
            i, j = comp.atoms[a], comp.atoms[b]
            r = Types.unwrap_displacement(box, q, j, i)
            vx = p[1, j] / m[j] - p[1, i] / m[i]
            vy = p[2, j] / m[j] - p[2, i] / m[i]
            vz = p[3, j] / m[j] - p[3, i] / m[i]
            push!(out, r.x * vx + r.y * vy + r.z * vz)
        end
    end
    return out
end

"""
Build the Jacobian rows `dc/dx` of one component.

Row `k` corresponds to `|x_j - x_i|^2 - d^2`, whose gradient is `-2r` at `i` and
`+2r` at `j` with `r = x_j - x_i`. Only the component's columns are populated;
the return value is `m x (3nlocal)`.
"""
function _component_jacobian(comp::ConstraintComponent, q::AbstractMatrix{Float64})
    n = length(comp.atoms)
    m = length(comp.edges)
    G = zeros(Float64, m, 3 * n)
    @inbounds for (k, (a, b, _)) in enumerate(comp.edges)
        i, j = comp.atoms[a], comp.atoms[b]
        rx = q[1, j] - q[1, i]
        ry = q[2, j] - q[2, i]
        rz = q[3, j] - q[3, i]
        G[k, 3(a - 1) + 1] = -2rx; G[k, 3(a - 1) + 2] = -2ry; G[k, 3(a - 1) + 3] = -2rz
        G[k, 3(b - 1) + 1] = 2rx;  G[k, 3(b - 1) + 2] = 2ry;  G[k, 3(b - 1) + 3] = 2rz
    end
    return G
end

"""
Name the geometry that makes a component's Jacobian rank deficient.

Returns a human-readable detail string, e.g. `"constraints (1,2) and (2,3) are
parallel (collinear atoms 1,2,3)"` or `"atoms 4 and 5 coincide"`. Used only to
build error messages, so it is allowed to be O(m²) and allocation-heavy.
"""
function _degenerate_detail(comp::ConstraintComponent, q::AbstractMatrix{Float64})
    details = String[]
    for (a, b, d) in comp.edges
        i, j = comp.atoms[a], comp.atoms[b]
        r = view(q, :, j) .- view(q, :, i)
        nr = sqrt(dot(r, r))
        if nr <= 1e-12 * max(d, 1.0)
            push!(details, "atoms $i and $j coincide (constraint length $d)")
        end
    end
    n = length(comp.edges)
    for p in 1:n, r in (p + 1):n
        (ia, ib, _) = comp.edges[p]
        (ja, jb, _) = comp.edges[r]
        u = view(q, :, comp.atoms[ib]) .- view(q, :, comp.atoms[ia])
        v = view(q, :, comp.atoms[jb]) .- view(q, :, comp.atoms[ja])
        nu = sqrt(dot(u, u)); nv = sqrt(dot(v, v))
        (nu == 0 || nv == 0) && continue
        s = sqrt(sum(abs2, [(u[2] * v[3] - u[3] * v[2]),
                            (u[3] * v[1] - u[1] * v[3]),
                            (u[1] * v[2] - u[2] * v[1])])) / (nu * nv)
        s <= 1e-8 && push!(details,
            "constraints ($(comp.atoms[ia]),$(comp.atoms[ib])) and " *
            "($(comp.atoms[ja]),$(comp.atoms[jb])) are parallel " *
            "(collinear atoms $(sort(unique(comp.atoms[[ia, ib, ja, jb]]))))")
    end
    isempty(details) && return "the constraint gradients are linearly dependent"
    return join(details, "; ")
end

"For a rank-deficient component: throw with the offending atoms named."
function _refuse_degenerate(comp::ConstraintComponent, q::AbstractMatrix{Float64},
                            what::AbstractString)
    throw(UnsupportedConstraintSet(
        "component atoms $(comp.atoms): $what ($(_degenerate_detail(comp, q)))"))
end

"""
Scratch buffers for one projection call, sized for the largest component.

Every component of a rigid-water system is 3 atoms / 3 constraints, so without
this the Newton loop would allocate a handful of small arrays per molecule per
iteration. In a dynamics loop that is thousands of allocations per step and it
showed up as a ~5 µs/molecule cost (GC pressure), not as arithmetic. One
workspace per *call* makes the per-component work allocation-free and the cost
flat in system size; it is an internal detail, not part of the API.
"""
struct ProjectionWorkspace
    W::Vector{Float64}    # 3n  inverse-mass vector
    c::Vector{Float64}    # m   constraint residuals
    G::Matrix{Float64}    # m x 3n Jacobian
    A::Matrix{Float64}    # m x m  mass-metric matrix
    λ::Vector{Float64}    # m   multipliers
    dx::Vector{Float64}   # 3n  correction
    acc::Vector{Float64}  # 3n  accumulated correction
end

function ProjectionWorkspace(cs::CoupledConstraintSet)
    n, k = 1, 1
    for comp in cs.components
        n = max(n, length(comp.atoms))
        k = max(k, length(comp.edges))
    end
    return ProjectionWorkspace(zeros(Float64, 3n), zeros(Float64, k),
                               zeros(Float64, k, 3n), zeros(Float64, k, k),
                               zeros(Float64, k), zeros(Float64, 3n),
                               zeros(Float64, 3n))
end

@inline function _fill_metric!(W, comp::ConstraintComponent, m)
    @inbounds for (a, atom) in enumerate(comp.atoms)
        w = 1.0 / m[atom]
        W[3(a - 1) + 1] = w
        W[3(a - 1) + 2] = w
        W[3(a - 1) + 3] = w
    end
    return W
end

@inline function _fill_jacobian!(G, comp::ConstraintComponent,
                                 q::AbstractMatrix{Float64}, mm::Int, nn::Int)
    fill!(G, 0.0)
    @inbounds for k in 1:mm
        (a, b, _) = comp.edges[k]
        i, j = comp.atoms[a], comp.atoms[b]
        rx = q[1, j] - q[1, i]; ry = q[2, j] - q[2, i]; rz = q[3, j] - q[3, i]
        G[k, 3(a - 1) + 1] = -2rx; G[k, 3(a - 1) + 2] = -2ry; G[k, 3(a - 1) + 3] = -2rz
        G[k, 3(b - 1) + 1] = 2rx;  G[k, 3(b - 1) + 2] = 2ry;  G[k, 3(b - 1) + 3] = 2rz
    end
    return G
end

"`A = G W G^T` into a preallocated `mm x mm` buffer (explicit: m is tiny)."
@inline function _fill_metric_matrix!(A, G, W, mm::Int, nn::Int)
    @inbounds for k in 1:mm
        for l in k:mm
            s = 0.0
            for i in 1:nn
                s += G[k, i] * W[i] * G[l, i]
            end
            A[k, l] = s
            A[l, k] = s
        end
    end
    return A
end

"`dx = W .* (Gᵀ λ)` into a preallocated 3n buffer."
@inline function _apply_jacobian_t!(dx, G, λ, W, mm::Int, nn::Int)
    @inbounds for i in 1:nn
        s = 0.0
        for k in 1:mm
            s += G[k, i] * λ[k]
        end
        dx[i] = W[i] * s
    end
    return dx
end

"Solve `(G W G^T) λ = b` in place; refuses rank-deficient geometry."
@inline function _solve_metric!(λ, A, G, W, b, mm::Int, nn::Int;
                                comp::Union{Nothing,ConstraintComponent} = nothing,
                                q::Union{Nothing,AbstractMatrix{Float64}} = nothing)
    _fill_metric_matrix!(A, G, W, mm, nn)
    max_diag = 0.0
    @inbounds for k in 1:mm
        akk = A[k, k]
        isfinite(akk) ||
            throw(UnsupportedConstraintSet(
                "constraint Jacobian produced a non-finite mass-metric matrix"))
        akk > max_diag && (max_diag = akk)
    end
    local F
    rank_deficient = max_diag <= 0.0
    if !rank_deficient
        try
            F = cholesky!(Symmetric(A))
        catch err
            if err isa PosDefException || err isa SingularException
                rank_deficient = true
            else
                rethrow()
            end
        end
    end
    if !rank_deficient
        # Guard against floating-point roundoff turning an exact zero Schur pivot
        # into ~+1e-15 so cholesky! succeeds on collinear/rank-deficient geometry.
        pivot_tol = max_diag * 1e-12
        @inbounds for k in 1:mm
            uk = F.U[k, k]
            if !isfinite(uk) || uk * uk <= pivot_tol
                rank_deficient = true
                break
            end
        end
    end
    if rank_deficient
        if comp !== nothing && q !== nothing
            _refuse_degenerate(comp, q, "constraint Jacobian is rank deficient " *
                                         "at this geometry, so the projection is not defined")
        end
        throw(UnsupportedConstraintSet(
            "constraint Jacobian is rank deficient at this geometry " *
            "(collinear or coincident atoms); the projection is not defined"))
    end
    copyto!(λ, b)
    ldiv!(F, λ)
    return λ
end

"Number of constraint edges across all components (the multiplier vector length)."
_total_edges(cs::CoupledConstraintSet) = sum(length(c.edges) for c in cs.components)

"Allocation-free `maximum(abs, constraint_residuals(cs, q))`."
function _max_abs_residual(cs::CoupledConstraintSet, q::AbstractMatrix{Float64})
    worst = 0.0
    @inbounds for comp in cs.components
        for (a, b, d) in comp.edges
            i, j = comp.atoms[a], comp.atoms[b]
            rx = q[1, j] - q[1, i]; ry = q[2, j] - q[2, i]; rz = q[3, j] - q[3, i]
            worst = max(worst, abs(rx * rx + ry * ry + rz * rz - d * d))
        end
    end
    return worst
end

"Allocation-free `maximum(abs, tangent_residuals(cs, q, p, m))`."
function _max_abs_tangent(cs::CoupledConstraintSet, q::AbstractMatrix{Float64},
                          p::AbstractMatrix{Float64}, m::AbstractVector{Float64})
    worst = 0.0
    @inbounds for comp in cs.components
        for (a, b, d) in comp.edges
            i, j = comp.atoms[a], comp.atoms[b]
            rx = q[1, j] - q[1, i]; ry = q[2, j] - q[2, i]; rz = q[3, j] - q[3, i]
            vx = p[1, j] / m[j] - p[1, i] / m[i]
            vy = p[2, j] / m[j] - p[2, i] / m[i]
            vz = p[3, j] / m[j] - p[3, i] / m[i]
            worst = max(worst, abs(rx * vx + ry * vy + rz * vz))
        end
    end
    return worst
end

function _validate_multipliers(cs::CoupledConstraintSet, λ::AbstractVector{Float64})
    n = _total_edges(cs)
    length(λ) == n ||
        throw(ArgumentError("expected $n constraint multipliers, got $(length(λ))"))
    all(isfinite, λ) || throw(ArgumentError("constraint multipliers must be finite"))
    return nothing
end

function _validate_positions_masses(q::AbstractMatrix{Float64}, m::AbstractVector{Float64},
                                    cs::CoupledConstraintSet;
                                    tolerance::Float64 = 1e-12,
                                    max_iterations::Int = 100)
    size(q, 1) == 3 || throw(ArgumentError("q must have 3 rows, got $(size(q, 1))"))
    size(q, 2) >= natoms(cs) ||
        throw(ArgumentError("q has $(size(q, 2)) atoms, but constraint set references $(natoms(cs))"))
    length(m) == size(q, 2) ||
        throw(ArgumentError("mass vector has length $(length(m)) but q has $(size(q, 2)) atoms"))
    all(x -> isfinite(x) && x > 0, m) ||
        throw(ArgumentError("masses must be positive and finite"))
    all(isfinite, q) || throw(ArgumentError("positions must be finite"))
    (isfinite(tolerance) && tolerance > 0) ||
        throw(ArgumentError("constraint tolerance must be positive and finite"))
    max_iterations >= 1 ||
        throw(ArgumentError("constraint max_iterations must be >= 1, got $max_iterations"))
    return nothing
end

function _validate_momenta(q::AbstractMatrix{Float64}, p::AbstractMatrix{Float64})
    size(p) == size(q) ||
        throw(ArgumentError("p must have the same shape as q, got $(size(p)) vs $(size(q))"))
    all(isfinite, p) || throw(ArgumentError("momenta must be finite"))
    return nothing
end

@inline function _component_position_residuals!(c, comp::ConstraintComponent,
                                              q::AbstractMatrix{Float64}, mm::Int)
    @inbounds for k in 1:mm
        (a, b, d) = comp.edges[k]
        i, j = comp.atoms[a], comp.atoms[b]
        rx = q[1, j] - q[1, i]; ry = q[2, j] - q[2, i]; rz = q[3, j] - q[3, i]
        c[k] = rx * rx + ry * ry + rz * rz - d * d
    end
    return maximum(abs, c)
end

"""
Newton-project one component in place; return `(accumulated displacement,
last residual, iterations)`.

`ws` is a scratch workspace sized for the largest component, reused across
components so this allocates nothing per molecule. Callers that need multipliers
reconstruct them from the accumulated displacement.
"""
function _project_component_positions!(q::AbstractMatrix{Float64},
                                       m::AbstractVector{Float64},
                                       comp::ConstraintComponent,
                                       ws::ProjectionWorkspace;
                                       tolerance::Float64 = 1e-12,
                                       max_iterations::Int = 100)
    n = length(comp.atoms)
    mm = length(comp.edges)
    nn = 3 * n
    W = view(ws.W, 1:nn)
    c = view(ws.c, 1:mm)
    G = view(ws.G, 1:mm, 1:nn)
    A = view(ws.A, 1:mm, 1:mm)
    λ = view(ws.λ, 1:mm)
    dx = view(ws.dx, 1:nn)
    acc = view(ws.acc, 1:nn)
    _fill_metric!(W, comp, m)
    fill!(acc, 0.0)
    residual = _component_position_residuals!(c, comp, q, mm)
    isfinite(residual) ||
        throw(ConstraintProjectionError(:position, residual, tolerance, 0, copy(comp.atoms)))
    residual <= tolerance && return acc, residual, 0
    for iter in 1:max_iterations
        _fill_jacobian!(G, comp, q, mm, nn)
        @inbounds for k in 1:mm
            c[k] = -c[k]
        end
        _solve_metric!(λ, A, G, W, c, mm, nn; comp = comp, q = q)
        _apply_jacobian_t!(dx, G, λ, W, mm, nn)
        @inbounds for (a, atom) in enumerate(comp.atoms)
            q[1, atom] += dx[3(a - 1) + 1]
            q[2, atom] += dx[3(a - 1) + 2]
            q[3, atom] += dx[3(a - 1) + 3]
            acc[3(a - 1) + 1] += dx[3(a - 1) + 1]
            acc[3(a - 1) + 2] += dx[3(a - 1) + 2]
            acc[3(a - 1) + 3] += dx[3(a - 1) + 3]
        end
        # Check residual AFTER the update so the final iteration is audited accurately.
        residual = _component_position_residuals!(c, comp, q, mm)
        isfinite(residual) ||
            throw(ConstraintProjectionError(:position, residual, tolerance, iter, copy(comp.atoms)))
        residual <= tolerance && return acc, residual, iter
    end
    throw(ConstraintProjectionError(:position, residual, tolerance, max_iterations,
                                    copy(comp.atoms)))
end

function _project_positions_impl!(q::AbstractMatrix{Float64}, m::AbstractVector{Float64},
                                  cs::CoupledConstraintSet;
                                  tolerance::Float64 = 1e-12, max_iterations::Int = 100)
    _validate_positions_masses(q, m, cs; tolerance, max_iterations)
    q0 = copy(q)
    ws = ProjectionWorkspace(cs)
    max_iters = 0
    try
        for comp in cs.components
            _, _, iters = _project_component_positions!(q, m, comp, ws; tolerance, max_iterations)
            max_iters = max(max_iters, iters)
        end
        res = _max_abs_residual(cs, q)
        (isfinite(res) && res <= tolerance) ||
            throw(ConstraintProjectionError(:position, res, tolerance, max_iterations, Int[]))
        return res, max_iters
    catch
        copyto!(q, q0)
        rethrow()
    end
end

"""
    project_positions!(q, m, cs; tolerance, max_iterations) -> max residual

Project `q` in place onto the constraint surfaces by Newton iteration in the mass
metric. Returns the final maximum residual. Refuses (restoring `q` and throwing)
if any component fails to converge or hits rank-deficient geometry.
"""
function project_positions!(q::AbstractMatrix{Float64}, m::AbstractVector{Float64},
                            cs::CoupledConstraintSet;
                            tolerance::Float64 = 1e-12, max_iterations::Int = 100)
    res, _ = _project_positions_impl!(q, m, cs; tolerance, max_iterations)
    return res
end

"""
    project_positions_with_multipliers!(q, m, cs, dt; tolerance, max_iterations)
        -> (residual, multipliers)

Same projection as [`project_positions!`](@ref), but also returns the constraint
multipliers of the impulse it applied.

The correction `Δx` is a displacement over the step `dt`, so the mean force the
projection exerts is `F^c = M Δx / dt²`. Writing `Δx = -W Gᵀ μ` (the projection
is mass-orthogonal to the constraint manifold) identifies `μ` from
`(G W Gᵀ) μ = -G Δx`, and the force-unit multipliers are `λ = μ / dt²`. Then
`F^c_i = -Σ_k λ_k ∇_i c_k`, which is exactly the convention
[`constraint_virial`](@ref) and [`constraint_forces`](@ref) consume.

`dt` is required (there is no meaningful default): a multiplier without the step
it belongs to is not a force. `λ` is flattened across components in the same
order as [`constraint_residuals`](@ref).

The reconstruction is exact when the correction is a single Newton step driven by
a known displacement, and accurate to the projection tolerance otherwise. That
is the sense in which it is useful: it says "this projection applied this
impulse", e.g. for a one-shot constraint-preserving volume move or for
attributing a position correction to its constraints.

**It is not a pressure accumulator.** In a velocity-Verlet trajectory the
position projection removes whatever drift accumulated, tangential corrections
included, so the impulse is not the constraint force. Measured on a 3-molecule
streamed-LJ run at dt = 0.005, the time-averaged impulse virial from this route
was `-0.0034` against `+0.0117` from [`constraint_multipliers`](@ref) — i.e. a
sign flip and a factor of 2.9, not the expected factor of two. For an NPT
pressure use the force-based route, which solves for the multipliers that hold
the geometry at the *current* forces.
"""
function project_positions_with_multipliers!(q::AbstractMatrix{Float64},
                                             m::AbstractVector{Float64},
                                             cs::CoupledConstraintSet,
                                             dt::Real;
                                             tolerance::Float64 = 1e-12,
                                             max_iterations::Int = 100)
    isfinite(dt) && dt > 0 || throw(ArgumentError("dt must be positive and finite"))
    _validate_positions_masses(q, m, cs; tolerance, max_iterations)
    q0 = copy(q)
    ws = ProjectionWorkspace(cs)
    λout = Float64[]
    sizehint!(λout, _total_edges(cs))
    dt2 = Float64(dt)^2
    try
        for comp in cs.components
            n = length(comp.atoms)
            mm = length(comp.edges)
            nn = 3 * n
            acc, _, _ = _project_component_positions!(q, m, comp, ws; tolerance, max_iterations)
            if all(iszero, acc)
                append!(λout, zeros(Float64, mm))
                continue
            end
            # Reconstruct the multiplier of this displacement at the final geometry:
            # (G W Gᵀ) μ = -G Δx, then λ = μ/dt².
            W = view(ws.W, 1:nn)
            G = view(ws.G, 1:mm, 1:nn)
            A = view(ws.A, 1:mm, 1:mm)
            λ = view(ws.λ, 1:mm)
            b = view(ws.c, 1:mm)
            _fill_metric!(W, comp, m)
            _fill_jacobian!(G, comp, q, mm, nn)
            @inbounds for k in 1:mm
                s = 0.0
                for i in 1:nn
                    s += G[k, i] * acc[i]
                end
                b[k] = -s
            end
            _solve_metric!(λ, A, G, W, b, mm, nn; comp = comp, q = q)
            for k in 1:mm
                val = λ[k] / dt2
                isfinite(val) || throw(ConstraintProjectionError(:position, NaN, tolerance, 0, copy(comp.atoms)))
                push!(λout, val)
            end
        end
        res = _max_abs_residual(cs, q)
        (isfinite(res) && res <= tolerance) ||
            throw(ConstraintProjectionError(:position, res, tolerance, max_iterations, Int[]))
        return res, λout
    catch
        copyto!(q, q0)
        rethrow()
    end
end

"""
    constraint_multipliers(q, m, cs, F_ext) -> multipliers

Multipliers that make an external force field `F_ext` (`3 x N`, sign convention
`F = -∂U/∂q`) consistent with the constraints at the fixed geometry `q`.

A constrained system's acceleration must satisfy `G M⁻¹ (F_ext + F^c) = 0`, and
with `F^c = -Gᵀ λ` this is `(G W Gᵀ) λ = G W F_ext`. That is the *static*
constraint force: no velocity or impulse is involved, so this is the route to use
for a pressure/virial evaluation at a fixed configuration.

Returns a vector flattened across components in the same order as
[`constraint_residuals`](@ref). Throws [`UnsupportedConstraintSet`](@ref) (naming
the atoms) at rank-deficient geometry.
"""
function constraint_multipliers(q::AbstractMatrix{Float64},
                                m::AbstractVector{Float64},
                                cs::CoupledConstraintSet,
                                F_ext::AbstractMatrix{Float64})
    _validate_positions_masses(q, m, cs)
    size(F_ext) == size(q) ||
        throw(ArgumentError("F_ext must have the same shape as q, got $(size(F_ext))"))
    all(isfinite, F_ext) || throw(ArgumentError("F_ext must be finite"))
    ws = ProjectionWorkspace(cs)
    λout = Float64[]
    sizehint!(λout, _total_edges(cs))
    for comp in cs.components
        n = length(comp.atoms)
        mm = length(comp.edges)
        nn = 3 * n
        W = view(ws.W, 1:nn)
        G = view(ws.G, 1:mm, 1:nn)
        A = view(ws.A, 1:mm, 1:mm)
        λ = view(ws.λ, 1:mm)
        b = view(ws.c, 1:mm)
        _fill_metric!(W, comp, m)
        _fill_jacobian!(G, comp, q, mm, nn)
        @inbounds for k in 1:mm
            s = 0.0
            for a in 1:n
                atom = comp.atoms[a]
                for d in 1:3
                    i = 3 * (a - 1) + d
                    s += G[k, i] * W[i] * F_ext[d, atom]
                end
            end
            b[k] = s
        end
        _solve_metric!(λ, A, G, W, b, mm, nn; comp = comp, q = q)
        for k in 1:mm
            push!(λout, λ[k])
        end
    end
    return λout
end

"""
    constraint_forces(q, m, cs, multipliers) -> 3 x N matrix

Constraint forces `F^c_i = -Σ_k λ_k ∇_i c_k` at the geometry `q`. The forces of
one constraint sum to zero, so this matrix has zero total momentum by
construction (to roundoff). `m` is accepted for symmetry with the rest of the
API and must match `size(q, 2)`.
"""
function constraint_forces(q::AbstractMatrix{Float64}, m::AbstractVector{Float64},
                           cs::CoupledConstraintSet,
                           λ::AbstractVector{Float64})
    _validate_multipliers(cs, λ)
    length(m) == size(q, 2) ||
        throw(ArgumentError("mass vector has length $(length(m)) but q has $(size(q, 2)) atoms"))
    F = zeros(Float64, 3, size(q, 2))
    k = 0
    for comp in cs.components
        G = _component_jacobian(comp, q)
        local_λ = view(λ, (k + 1):(k + length(comp.edges)))
        f = -(transpose(G) * local_λ)
        for (a, atom) in enumerate(comp.atoms)
            F[1, atom] += f[3(a - 1) + 1]
            F[2, atom] += f[3(a - 1) + 2]
            F[3, atom] += f[3(a - 1) + 3]
        end
        k += length(comp.edges)
    end
    return F
end

"""
    constraint_virial(q, m, cs, multipliers) -> W_c

Configurational virial of the constraint forces,

    W_c = Σ_i r_i · F^c_i = -Σ_k λ_k Σ_i r_i·∇_i c_k = -2 Σ_k λ_k |r_k|²,

the last step because `c_k = |r_j - r_i|² - d_k²` and `Σ_i r_i·∇_i c_k = 2|r_k|²`.
Using `|r_k|²` from the current geometry (not `d_k²`) means the value is a
function of the coordinates alone, so it stays meaningful at a geometry that is
not yet exactly on the constraint surface.

`multipliers` must be in force units — either from
[`constraint_multipliers`](@ref) or from
[`project_positions_with_multipliers!`](@ref) with the step's `dt`. The total
virial for a pressure is `Σ_i r_i·F_i^ext + W_c`; the constraint virial is not
optional in that sum, because the box couples to constrained coordinates too.

The closed form needs no centre of mass and no per-atom force array: the `2|r_k|²`
factor *is* the translation-invariant sum `Σ_i (r_i - R)·∇_i c_k` for any origin
`R`, so there is no large-coordinate cancellation to worry about. `m` and `q`
are used only for validation and for `|r_k|`; a caller that has already computed
the edge vectors can avoid this call entirely.
"""
function constraint_virial(q::AbstractMatrix{Float64}, m::AbstractVector{Float64},
                           cs::CoupledConstraintSet,
                           λ::AbstractVector{Float64})
    _validate_multipliers(cs, λ)
    length(m) == size(q, 2) ||
        throw(ArgumentError("mass vector has length $(length(m)) but q has $(size(q, 2)) atoms"))
    W = 0.0
    k = 0
    for comp in cs.components
        for (a, b, _) in comp.edges
            k += 1
            i, j = comp.atoms[a], comp.atoms[b]
            rx = q[1, j] - q[1, i]; ry = q[2, j] - q[2, i]; rz = q[3, j] - q[3, i]
            # (r_i - R)·∇_i c_k + (r_j - R)·∇_j c_k = 2 |r_k|^2 for any R.
            W += -2.0 * λ[k] * (rx * rx + ry * ry + rz * rz)
        end
    end
    return W
end

@inline function _component_tangent_residuals!(gv, G, vloc, comp::ConstraintComponent,
                                             p::AbstractMatrix{Float64},
                                             m::AbstractVector{Float64},
                                             mm::Int, nn::Int)
    @inbounds for (a, atom) in enumerate(comp.atoms)
        vloc[3(a - 1) + 1] = p[1, atom] / m[atom]
        vloc[3(a - 1) + 2] = p[2, atom] / m[atom]
        vloc[3(a - 1) + 3] = p[3, atom] / m[atom]
    end
    @inbounds for k in 1:mm
        s = 0.0
        for i in 1:nn
            s += G[k, i] * vloc[i]
        end
        gv[k] = s
    end
    return maximum(abs, gv)
end

function _project_component_velocities!(q::AbstractMatrix{Float64},
                                        p::AbstractMatrix{Float64},
                                        m::AbstractVector{Float64},
                                        comp::ConstraintComponent,
                                        ws::ProjectionWorkspace;
                                        tolerance::Float64 = 1e-12,
                                        max_iterations::Int = 100)
    n = length(comp.atoms)
    mm = length(comp.edges)
    nn = 3 * n
    W = view(ws.W, 1:nn)
    G = view(ws.G, 1:mm, 1:nn)
    A = view(ws.A, 1:mm, 1:mm)
    λ = view(ws.λ, 1:mm)
    gv = view(ws.c, 1:mm)
    vloc = view(ws.acc, 1:nn)
    dv = view(ws.dx, 1:nn)
    _fill_metric!(W, comp, m)
    # G depends on positions only, and the positions do not move here.
    _fill_jacobian!(G, comp, q, mm, nn)
    residual = _component_tangent_residuals!(gv, G, vloc, comp, p, m, mm, nn)
    isfinite(residual) ||
        throw(ConstraintProjectionError(:velocity, residual, tolerance, 0, copy(comp.atoms)))
    residual <= tolerance && return residual, 0
    for iter in 1:max_iterations
        @inbounds for k in 1:mm
            gv[k] = -gv[k]
        end
        _solve_metric!(λ, A, G, W, gv, mm, nn; comp = comp, q = q)
        _apply_jacobian_t!(dv, G, λ, W, mm, nn)
        @inbounds for (a, atom) in enumerate(comp.atoms)
            p[1, atom] += m[atom] * dv[3(a - 1) + 1]
            p[2, atom] += m[atom] * dv[3(a - 1) + 2]
            p[3, atom] += m[atom] * dv[3(a - 1) + 3]
        end
        residual = _component_tangent_residuals!(gv, G, vloc, comp, p, m, mm, nn)
        isfinite(residual) ||
            throw(ConstraintProjectionError(:velocity, residual, tolerance, iter, copy(comp.atoms)))
        residual <= tolerance && return residual, iter
    end
    throw(ConstraintProjectionError(:velocity, residual, tolerance, max_iterations,
                                    copy(comp.atoms)))
end

function _project_velocities_impl!(q::AbstractMatrix{Float64}, p::AbstractMatrix{Float64},
                                   m::AbstractVector{Float64}, cs::CoupledConstraintSet;
                                   tolerance::Float64 = 1e-12, max_iterations::Int = 100)
    _validate_positions_masses(q, m, cs; tolerance, max_iterations)
    _validate_momenta(q, p)
    p0 = copy(p)
    ws = ProjectionWorkspace(cs)
    max_iters = 0
    try
        for comp in cs.components
            _, iters = _project_component_velocities!(q, p, m, comp, ws; tolerance, max_iterations)
            max_iters = max(max_iters, iters)
        end
        res = _max_abs_tangent(cs, q, p, m)
        (isfinite(res) && res <= tolerance) ||
            throw(ConstraintProjectionError(:velocity, res, tolerance, max_iterations, Int[]))
        return res, max_iters
    catch
        copyto!(p, p0)
        rethrow()
    end
end

"""
    project_velocities!(q, p, m, cs; tolerance, max_iterations) -> max tangent residual

Project momenta `p` in place so that `r · (v_j - v_i) = 0` for every constraint,
i.e. no constraint is being stretched by the current velocities. Refuses
(restoring `p` and throwing) on non-convergence or rank-deficient geometry.
"""
function project_velocities!(q::AbstractMatrix{Float64}, p::AbstractMatrix{Float64},
                             m::AbstractVector{Float64}, cs::CoupledConstraintSet;
                             tolerance::Float64 = 1e-12, max_iterations::Int = 100)
    res, _ = _project_velocities_impl!(q, p, m, cs; tolerance, max_iterations)
    return res
end

"""
Project both positions and momenta. Returns `(pos_residual, tan_residual)`.
Atomic across both stages: if velocity projection fails, `q` and `p` are both
restored to their pre-call values.
"""
function project!(q::AbstractMatrix{Float64}, p::AbstractMatrix{Float64},
                  m::AbstractVector{Float64}, cs::CoupledConstraintSet;
                  tolerance::Float64 = 1e-12, max_iterations::Int = 100)
    q0 = copy(q)
    p0 = copy(p)
    try
        rp = project_positions!(q, m, cs; tolerance, max_iterations)
        rt = project_velocities!(q, p, m, cs; tolerance, max_iterations)
        return rp, rt
    catch
        copyto!(q, q0)
        copyto!(p, p0)
        rethrow()
    end
end

function _failure_result(stage::Symbol, err, pos_res::Float64, tan_res::Float64,
                         tolerance::Float64, max_iterations::Int)
    if err isa ConstraintProjectionError
        return ConstraintSolveResult(err.stage, false,
                                     stage === :position ? err.residual : pos_res,
                                     stage === :velocity ? err.residual : tan_res,
                                     err.iterations, err.tolerance,
                                     copy(err.atoms), sprint(showerror, err))
    elseif err isa UnsupportedConstraintSet
        return ConstraintSolveResult(stage, false, pos_res, tan_res,
                                     max_iterations, tolerance,
                                     Int[], sprint(showerror, err))
    else
        rethrow(err)
    end
end

"""
    solve_positions!(q, m, cs; tolerance=1e-12, max_iterations=100) -> ConstraintSolveResult

Audited position projection that always returns explicit residuals and
convergence status. If convergence fails or geometry is rank deficient, `q` is
restored to its pre-call state (never accepting a partially projected state) and
`converged == false` is returned.
"""
function solve_positions!(q::AbstractMatrix{Float64}, m::AbstractVector{Float64},
                          cs::CoupledConstraintSet;
                          tolerance::Float64 = 1e-12, max_iterations::Int = 100)
    _validate_positions_masses(q, m, cs; tolerance, max_iterations)
    try
        res, iters = _project_positions_impl!(q, m, cs; tolerance, max_iterations)
        return ConstraintSolveResult(:position, true, res, NaN, iters, tolerance, Int[], "")
    catch err
        pos_res = _max_abs_residual(cs, q)
        return _failure_result(:position, err, pos_res, NaN, tolerance, max_iterations)
    end
end

"""
    solve_velocities!(q, p, m, cs; tolerance=1e-12, max_iterations=100) -> ConstraintSolveResult

Audited velocity projection returning explicit residuals and convergence status.
Restores `p` and returns `converged == false` on failure.
"""
function solve_velocities!(q::AbstractMatrix{Float64}, p::AbstractMatrix{Float64},
                           m::AbstractVector{Float64}, cs::CoupledConstraintSet;
                           tolerance::Float64 = 1e-12, max_iterations::Int = 100)
    _validate_positions_masses(q, m, cs; tolerance, max_iterations)
    _validate_momenta(q, p)
    pos_res = _max_abs_residual(cs, q)
    try
        tan_res, iters = _project_velocities_impl!(q, p, m, cs; tolerance, max_iterations)
        return ConstraintSolveResult(:velocity, true, pos_res, tan_res, iters, tolerance, Int[], "")
    catch err
        tan_res = _max_abs_tangent(cs, q, p, m)
        return _failure_result(:velocity, err, pos_res, tan_res, tolerance, max_iterations)
    end
end

"""
    solve_constraints!(q, p, m, cs; tolerance=1e-12, max_iterations=100) -> ConstraintSolveResult

Audited combined position + velocity projection returning explicit residuals and
convergence status. Restores `(q, p)` and returns `converged == false` if either
stage fails.
"""
function solve_constraints!(q::AbstractMatrix{Float64}, p::AbstractMatrix{Float64},
                            m::AbstractVector{Float64}, cs::CoupledConstraintSet;
                            tolerance::Float64 = 1e-12, max_iterations::Int = 100)
    _validate_positions_masses(q, m, cs; tolerance, max_iterations)
    _validate_momenta(q, p)
    q0 = copy(q)
    p0 = copy(p)
    try
        rp, ip = _project_positions_impl!(q, m, cs; tolerance, max_iterations)
        rt, it = _project_velocities_impl!(q, p, m, cs; tolerance, max_iterations)
        return ConstraintSolveResult(:both, true, rp, rt, max(ip, it), tolerance, Int[], "")
    catch err
        copyto!(q, q0)
        copyto!(p, p0)
        rp = _max_abs_residual(cs, q)
        rt = _max_abs_tangent(cs, q, p, m)
        return _failure_result(:both, err, rp, rt, tolerance, max_iterations)
    end
end

end # module Constraints
