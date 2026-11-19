"""Configurational isotropic NPT Monte-Carlo barostat (orthorhombic boxes).

## What this module is

A single, self-contained transition kernel and pressure estimator for the
isothermal–isobaric (configurational) ensemble on orthorhombic periodic boxes.
It separates:

1. **Energy-based MC volume moves**:
   - Potential energy recomputation `U(s, h)` at the trial state,
   - `P * ΔV` work,
   - Proposal density ratio (for symmetric proposals in `log(V)` or linear `V`),
   - Exact translational coordinate-measure Jacobian:
     - `N * log(V1 / V0)` when scaling all `N` unconstrained atoms affinely, or
     - `N_mol * log(V1 / V0)` when scaling `N_mol` molecular centers of mass
       (`R_α -> a R_α`, internal displacements `q_i - R_α` fixed) for rigid or
       constrained molecules. Constrained bond atoms are **never** scaled
       independently.

2. **Decomposed configurational pressure estimator** (`pressure_components`):
   Separates instantaneous mechanical pressure `P = P_kin + P_int + P_con` into:
   - `kinetic`: `2 K / (3 V)` (atomic kinetic energy, or COM translational kinetic
     energy in `:molecular_com` mode),
   - `interaction`: `W_int / (3 V)` with `W_int = -tr((∂U/∂h|_s) hᵀ)` from an
     analytic box-derivative provider (or a consistent interaction virial),
   - `constraint`: `W_con / (3 V)` from holonomic constraint forces via
     `Constraints.constraint_virial` using dynamical multipliers
     `(G W Gᵀ) λ = G W F_ext + Ġ v` (or intramolecular external-force virial in
     `:molecular_com` mode).

## The measure and the Jacobian (verified, not assumed)

For a system partitioned into `N_mol` translationally independent units
(individual atoms for an unconstrained fluid, or disjoint molecules with fixed
internal coordinates `δ_i = q_i - R_α` relative to their centers of mass `R_α`),
introduce scaled COM coordinates `S_α = h⁻¹ R_α`. Because internal displacements
`δ_i` are translation-invariant and unscaled, the internal/manifold measure is
volume-independent while each `dR_α = V dS_α`, giving

    π(S, δ, V) ∝ exp(-β (U(q(S, δ; h), h) + P V)) * V^(N_mol).

With a proposal `q(V -> V')` in volume space, the log Metropolis acceptance is

    log α = -β (ΔU + P ΔV) + N_mol log(V'/V) + log(q(V' -> V) / q(V -> V')).

* For `:log_volume` (uniform in `x = log V`), `q(V' -> V)/q(V -> V') = V'/V`, so
  `log_proposal_ratio = log(V'/V)` and the total volume-measure term is
  `(N_mol + 1) * log(V'/V)`.
* For `:linear_volume` (uniform in `V` with rejection of `V' <= 0`),
  `log_proposal_ratio = 0.0` (for `V' > 0`), leaving `N_mol * log(V'/V)`.

## What is deliberately refused

* **Atomic scaling with constraints** (`ConstraintIncompatibilityError`). Passing
  non-empty constraints without molecular COM grouping refuses instead of
  stretching rigid bonds.
* **Inter-molecular constraints** (`ConstraintIncompatibilityError`). Every
  constraint edge must lie strictly inside one molecular group; a constraint
  linking two different molecules or an ungrouped atom cannot be preserved by
  COM scaling.
* **Missing masses on multi-atom molecules** (`MissingMassError`). Mass-weighted
  COM scaling and kinetic pressure require per-atom masses.
* **Missing virial / box derivative or missing forces under constraints**
  (`MissingVirialError`). No ideal-gas or finite-difference surrogate is ever
  substituted.
* **Non-orthorhombic geometry** (`UnsupportedGeometry`). `TriclinicBox` and
  `NoBox` are explicitly refused by this isotropic orthorhombic kernel.
"""
module MonteCarloBarostat

using LinearAlgebra
using Random
using Statistics
using ..Types
using ..Types: OrthorhombicBox, TriclinicBox, NoBox, AbstractBox, Units,
               box_volume, UnsupportedGeometry, unwrap_displacement
using ..Constraints
using ..Constraints: CoupledConstraintSet, constraint_multipliers,
                     constraint_virial, UnsupportedConstraintSet
import ..Constraints: constraint_residuals

export VolumeMoveBarostat, NPTBarostat, NPTModel, PressureComponents,
       step!, npt_step!, npt_mc_step!, propose_volume_move, configurational_pressure, pressure_components,
       virial_from_box_derivative, log_acceptance, log_acceptance_components,
       translational_log_jacobian, log_proposal_ratio, propose_log_volume,
       scaled_coordinates, scale_coordinates!, scale_anisotropic_coordinates!,
       scale_molecular_coordinates!, molecular_centers_of_mass,
       wrap_molecules!, static_constraint_multipliers,
       dynamical_constraint_multipliers, validate_molecular_constraints,
       infer_molecules,
       internal_pressure_from_bar, bar_from_internal_pressure,
       ConstraintIncompatibilityError, MissingVirialError, MissingMassError

# ---------------------------------------------------------------------------
# typed refusals
# ---------------------------------------------------------------------------

"""
Raised when a volume move or barostat configuration is incompatible with the
system's holonomic constraints (e.g., atomic scaling on constrained bonds,
inter-molecular constraints, or broken molecular geometry across PBC).
"""
struct ConstraintIncompatibilityError <: Exception
    reason::String
end

Base.showerror(io::IO, e::ConstraintIncompatibilityError) =
    print(io, "constraint-incompatible barostat: ", e.reason)

"""
Raised when a pressure-dependent operation lacks a required virial / box
derivative provider (or external forces needed for constraint multipliers).
"""
struct MissingVirialError <: Exception
    reason::String
end

Base.showerror(io::IO, e::MissingVirialError) =
    print(io, "missing virial: ", e.reason)

"Raised when mass-weighted COM scaling or kinetic pressure is requested without masses."
struct MissingMassError <: Exception
    reason::String
end

Base.showerror(io::IO, e::MissingMassError) =
    print(io, "missing masses: ", e.reason)

# ---------------------------------------------------------------------------
# unit conversion helpers
# ---------------------------------------------------------------------------

"""
    internal_pressure_from_bar(p_bar, units::Types.Units) -> Float64

Convert pressure from bar (`1e5 Pa`) to the `units` energy/length³ unit.
For `WaterBox.TIP3P_UNITS` (`nm`, `kJ/mol`), `1 bar = 0.0602214076 kJ/(mol nm³)`.
"""
function internal_pressure_from_bar(p_bar::Real, units::Units)
    isfinite(p_bar) || throw(ArgumentError("pressure must be finite"))
    return Float64(p_bar) * 1.0e5 * (units.length^3) / units.energy
end

"""
    bar_from_internal_pressure(p_internal, units::Types.Units) -> Float64

Convert pressure from `units` (`energy / length³`) to bar (`1e5 Pa`).
"""
function bar_from_internal_pressure(p_internal::Real, units::Units)
    isfinite(p_internal) || throw(ArgumentError("pressure must be finite"))
    return Float64(p_internal) * units.energy / (1.0e5 * (units.length^3))
end

# ---------------------------------------------------------------------------
# validation & molecular-grouping helpers
# ---------------------------------------------------------------------------

function _check_scalars(beta, pressure, max_delta)
    isfinite(beta) && beta > 0 || throw(ArgumentError("beta must be finite and positive"))
    isfinite(pressure) || throw(ArgumentError("pressure must be finite"))
    isfinite(max_delta) && max_delta > 0 ||
        throw(ArgumentError("max_delta_log_volume must be finite and positive"))
    return nothing
end

function _check_proposal(proposal::Symbol)
    (proposal === :log_volume || proposal === :linear_volume) ||
        throw(ArgumentError("proposal must be :log_volume or :linear_volume, got $(repr(proposal))"))
    return proposal
end

function _check_box(box::AbstractBox; allow_triclinic::Bool = false)
    if box isa OrthorhombicBox
        return box
    elseif box isa TriclinicBox && allow_triclinic
        return box
    else
        throw(UnsupportedGeometry(
            "MonteCarloBarostat supports orthorhombic boxes only" *
            (allow_triclinic ? " (or triclinic with allow_triclinic=true)" : "") *
            "; an isotropic dilation of $(nameof(typeof(box))) is not supported by this kernel"))
    end
end

function _is_empty_constraints(constraints)
    constraints === nothing && return true
    constraints isa CoupledConstraintSet && return isempty(constraints.components)
    return isempty(constraints)
end

function _as_coupled_constraints(n_atoms::Int, constraints)
    if constraints === nothing
        return nothing
    elseif constraints isa CoupledConstraintSet
        for comp in constraints.components
            all(i -> 1 <= i <= n_atoms, comp.atoms) ||
                throw(ConstraintIncompatibilityError(
                    "constraint atom index out of bounds for N=$n_atoms"))
        end
        return constraints
    else
        isempty(constraints) && return nothing
        return CoupledConstraintSet(n_atoms, constraints)
    end
end

function _mass_vector(masses, n::Integer)
    if masses === nothing
        throw(MissingMassError(
            "operation needs per-atom masses; supply masses = ..."))
    elseif masses isa Real
        m = Float64(masses)
        isfinite(m) && m > 0 || throw(ArgumentError("mass must be finite and positive"))
        return fill(m, Int(n))
    else
        length(masses) == n || throw(ArgumentError("mass count $(length(masses)) != N=$n"))
        m = Float64.(masses)
        all(x -> isfinite(x) && x > 0, m) ||
            throw(ArgumentError("all masses must be finite and positive"))
        return m
    end
end

"""
    infer_molecules(n_atoms::Int; bonds = Tuple{Int,Int}[], constraints = nothing) -> Vector{Vector{Int}}

Infer disjoint molecular topology from bonded connectivity and holonomic distance
constraints using Disjoint-Set (Union-Find) over `1:n_atoms`. Returns sorted connected
components `Vector{Vector{Int}}`. Singletons are included for ungrouped atoms.
"""
function infer_molecules(n_atoms::Integer;
                         bonds::AbstractVector = Tuple{Int,Int}[],
                         constraints = nothing)::Vector{Vector{Int}}
    n = Int(n_atoms)
    n >= 1 || throw(ArgumentError("n_atoms must be positive"))
    parent = collect(1:n)
    function find_root(i::Int)
        root = i
        while parent[root] != root
            root = parent[root]
        end
        curr = i
        while curr != root
            nxt = parent[curr]
            parent[curr] = root
            curr = nxt
        end
        return root
    end
    function union_atoms!(i::Int, j::Int)
        1 <= i <= n && 1 <= j <= n || throw(ArgumentError("atom index out of bounds 1:$n: ($i, $j)"))
        ri = find_root(i)
        rj = find_root(j)
        if ri != rj
            parent[ri] = rj
        end
        return nothing
    end

    for b in bonds
        union_atoms!(Int(b[1]), Int(b[2]))
    end

    if constraints !== nothing
        if constraints isa CoupledConstraintSet
            for comp in constraints.components
                for (a, b, _) in comp.edges
                    union_atoms!(comp.atoms[a], comp.atoms[b])
                end
            end
        elseif isdefined(Constraints, :DistanceConstraintSet) && constraints isa getfield(Constraints, :DistanceConstraintSet)
            for edge in constraints.edges
                union_atoms!(Int(edge[1]), Int(edge[2]))
            end
        elseif constraints isa AbstractVector
            for item in constraints
                if (item isa Tuple || item isa AbstractVector) && length(item) >= 2
                    union_atoms!(Int(item[1]), Int(item[2]))
                end
            end
        end
    end

    groups_map = Dict{Int,Vector{Int}}()
    for i in 1:n
        r = find_root(i)
        if !haskey(groups_map, r)
            groups_map[r] = Int[]
        end
        push!(groups_map[r], i)
    end

    groups = Vector{Vector{Int}}()
    sizehint!(groups, length(groups_map))
    for g in values(groups_map)
        sort!(g)
        push!(groups, g)
    end
    sort!(groups, by = g -> g[1])
    return groups
end

"""
    validate_molecular_constraints(n_atoms, molecules, constraints) -> (groups, cs)

Partition `1:n_atoms` into disjoint translational groups (each supplied
molecule plus any remaining ungrouped atoms as singletons) and verify that every
constraint edge in `constraints` lies strictly inside a single group. Throws
`ConstraintIncompatibilityError` on inter-molecular constraints, overlapping
molecules, or atomic-only constraints.
"""
function validate_molecular_constraints(n_atoms::Integer, molecules, constraints)
    n = Int(n_atoms)
    n >= 1 || throw(ArgumentError("n_atoms must be positive"))
    has_c = !_is_empty_constraints(constraints)

    if molecules === nothing
        if has_c
            cnt = constraints isa CoupledConstraintSet ?
                  sum(length(c.edges) for c in constraints.components) : length(constraints)
            throw(ConstraintIncompatibilityError(
                "uniform atomic volume scaling stretches absolute bond lengths, so " *
                "fixed-length constraints require molecular COM scaling (`molecules = ...`). " *
                "Received $cnt constraint(s) without molecular grouping."))
        end
        groups = [[j] for j in 1:n]
        return groups, nothing
    end

    seen = falses(n)
    groups = Vector{Vector{Int}}()
    atom_to_group = zeros(Int, n)
    for raw_mol in molecules
        mol = collect(Int, raw_mol)
        isempty(mol) && throw(ConstraintIncompatibilityError("molecule group cannot be empty"))
        for idx in mol
            1 <= idx <= n || throw(ConstraintIncompatibilityError(
                "molecule atom index $idx out of bounds 1:$n"))
            seen[idx] && throw(ConstraintIncompatibilityError(
                "atom $idx appears in multiple molecules"))
            seen[idx] = true
        end
        push!(groups, mol)
        gid = length(groups)
        for idx in mol
            atom_to_group[idx] = gid
        end
    end
    # Remaining ungrouped atoms are free 1-atom translational units.
    for j in 1:n
        if !seen[j]
            push!(groups, [j])
            atom_to_group[j] = length(groups)
        end
    end

    cs = _as_coupled_constraints(n, constraints)
    if cs !== nothing
        for comp in cs.components
            for (a, b, d) in comp.edges
                i, j = comp.atoms[a], comp.atoms[b]
                if atom_to_group[i] != atom_to_group[j]
                    throw(ConstraintIncompatibilityError(
                        "constraint ($i, $j, d=$d) connects atoms in different " *
                        "molecular groups ($(atom_to_group[i]) vs $(atom_to_group[j])); " *
                        "molecular COM volume scaling preserves only intra-molecular constraints"))
                end
            end
        end
    end
    return groups, cs
end

function _prepare_scaling_groups(n_atoms::Int, molecules, masses, constraints,
                                 scale_mode::Symbol)
    (scale_mode === :auto || scale_mode === :atomic || scale_mode === :molecular) ||
        throw(ArgumentError("scale_mode must be :auto, :atomic, or :molecular"))
    has_c = !_is_empty_constraints(constraints)
    if scale_mode === :atomic && has_c
        throw(ConstraintIncompatibilityError(
            "scale_mode = :atomic is incompatible with holonomic constraints"))
    end
    if scale_mode === :molecular && molecules === nothing
        has_c && throw(ConstraintIncompatibilityError(
            "scale_mode = :molecular with constraints requires `molecules`"))
        throw(ArgumentError("scale_mode = :molecular requires `molecules`"))
    end
    eff_molecules = scale_mode === :atomic ? nothing : molecules
    groups, cs = validate_molecular_constraints(n_atoms, eff_molecules, constraints)
    has_multi = any(g -> length(g) > 1, groups)
    m = if masses !== nothing
        _mass_vector(masses, n_atoms)
    elseif has_multi
        throw(MissingMassError(
            "multi-atom molecular COM scaling requires per-atom `masses`"))
    else
        ones(Float64, n_atoms)
    end
    group_masses = Float64[sum(m[i] for i in g) for g in groups]
    return groups, m, group_masses, cs, has_multi
end

# ---------------------------------------------------------------------------
# coordinate scaling & molecular COM helpers
# ---------------------------------------------------------------------------

"Scaled (fractional) coordinates `q ./ L`, with one row per Cartesian axis."
function scaled_coordinates(q::AbstractMatrix{<:Real}, box::OrthorhombicBox)
    size(q, 1) == 3 || throw(ArgumentError("positions must be 3 x N"))
    return q ./ collect(box.L)
end

"""
    molecular_centers_of_mass(q, molecules, masses) -> 3 x N_mol Matrix{Float64}

Compute mass-weighted centers of mass for each atom group in `molecules`.
"""
function molecular_centers_of_mass(q::AbstractMatrix{<:Real}, molecules,
                                   masses::AbstractVector{<:Real})
    size(q, 1) == 3 || throw(ArgumentError("positions must be 3 x N"))
    n = size(q, 2)
    length(masses) == n || throw(ArgumentError("mass count != N=$n"))
    nmol = length(molecules)
    R = Matrix{Float64}(undef, 3, nmol)
    @inbounds for (alpha, mol) in enumerate(molecules)
        M = 0.0
        rx = 0.0; ry = 0.0; rz = 0.0
        for i in mol
            mi = Float64(masses[i])
            M += mi
            rx += mi * Float64(q[1, i])
            ry += mi * Float64(q[2, i])
            rz += mi * Float64(q[3, i])
        end
        invM = 1.0 / M
        R[1, alpha] = rx * invM
        R[2, alpha] = ry * invM
        R[3, alpha] = rz * invM
    end
    return R
end

"""
    wrap_molecules!(q, box::OrthorhombicBox, molecules, masses)

Wrap each molecule rigidly as a whole into the primary orthorhombic cell
`[0, L₁) × [0, L₂) × [0, L₃)` by shifting every atom in a molecule by the same
integer box vector determined by its center of mass. Intramolecular bond
vectors are preserved bit-for-bit (up to floating-point addition).
"""
function wrap_molecules!(q::AbstractMatrix{Float64}, box::OrthorhombicBox,
                         molecules, masses::AbstractVector{<:Real})
    size(q, 1) == 3 || throw(ArgumentError("positions must be 3 x N"))
    L = box.L
    @inbounds for mol in molecules
        M = 0.0
        rx = 0.0; ry = 0.0; rz = 0.0
        for i in mol
            mi = Float64(masses[i])
            M += mi
            rx += mi * q[1, i]
            ry += mi * q[2, i]
            rz += mi * q[3, i]
        end
        invM = 1.0 / M
        cx = rx * invM; cy = ry * invM; cz = rz * invM
        sx = L[1] * floor(cx / L[1])
        sy = L[2] * floor(cy / L[2])
        sz = L[3] * floor(cz / L[3])
        if sx != 0.0 || sy != 0.0 || sz != 0.0
            for i in mol
                q[1, i] -= sx
                q[2, i] -= sy
                q[3, i] -= sz
            end
        end
    end
    return q
end

"""
    scale_coordinates!(q_out, q_in, scale; groups = nothing, masses = nothing,
                       group_masses = nothing) -> Int

Apply isotropic dilation `scale = (V1/V0)^(1/3)` to `q_in` writing into `q_out`.
* When `groups === nothing`, scales all `N` atomic coordinates uniformly (`q_out .= q_in .* scale`)
  and returns `N` translational degrees-of-freedom units.
* When `groups` is provided, translates each molecule rigidly by `(scale - 1) * R_α`
  where `R_α` is its mass-weighted center of mass, leaving all intramolecular
  displacements `q_i - R_α` invariant, and returns `length(groups)` (`N_mol`).
"""
function scale_coordinates!(q_out::AbstractMatrix{Float64},
                            q_in::AbstractMatrix{<:Real},
                            scale::Real;
                            box::Union{Nothing,AbstractBox} = nothing,
                            groups = nothing,
                            masses::Union{Nothing,AbstractVector{Float64}} = nothing,
                            group_masses::Union{Nothing,AbstractVector{Float64}} = nothing)
    size(q_in, 1) == 3 && size(q_out) == size(q_in) ||
        throw(ArgumentError("q_out and q_in must both be 3 x N"))
    if box isa OrthorhombicBox && groups !== nothing
        s = Float64(scale)
        return scale_anisotropic_coordinates!(q_out, q_in, box, s, s, s;
                                              molecules = groups, masses = masses)
    elseif box isa TriclinicBox && groups !== nothing
        return scale_triclinic_coordinates!(q_out, q_in, box, Float64(scale);
                                            molecules = groups, masses = masses)
    end
    s = Float64(scale)
    n = size(q_in, 2)
    if groups === nothing
        @inbounds for j in 1:n, d in 1:3
            q_out[d, j] = Float64(q_in[d, j]) * s
        end
        return n
    end
    m = masses === nothing ? ones(Float64, n) : masses
    sm1 = s - 1.0
    @inbounds for (alpha, g) in enumerate(groups)
        if length(g) == 1
            i = g[1]
            q_out[1, i] = Float64(q_in[1, i]) * s
            q_out[2, i] = Float64(q_in[2, i]) * s
            q_out[3, i] = Float64(q_in[3, i]) * s
        else
            Mg = group_masses === nothing ? sum(m[i] for i in g) : group_masses[alpha]
            rx = 0.0; ry = 0.0; rz = 0.0
            for i in g
                mi = m[i]
                rx += mi * Float64(q_in[1, i])
                ry += mi * Float64(q_in[2, i])
                rz += mi * Float64(q_in[3, i])
            end
            invM = 1.0 / Mg
            dx = sm1 * (rx * invM)
            dy = sm1 * (ry * invM)
            dz = sm1 * (rz * invM)
            for i in g
                q_out[1, i] = Float64(q_in[1, i]) + dx
                q_out[2, i] = Float64(q_in[2, i]) + dy
                q_out[3, i] = Float64(q_in[3, i]) + dz
            end
        end
    end
    return length(groups)
end

"""
    scale_anisotropic_coordinates!(q_out, q_in, box::OrthorhombicBox,
                                   sx::Real, sy::Real, sz::Real;
                                   molecules = nothing, masses = nothing) -> Int

Apply anisotropic dilation factors `(sx, sy, sz)` to `q_in` writing into `q_out`.
When `molecules === nothing`, scales all `N` atomic coordinates affinely.
When `molecules !== nothing`, for each molecule unwraps coordinates relative to `mol[1]`
using minimum-image displacement on `box` before computing the molecular center of mass,
scales the COM by `(sx, sy, sz)`, and translates the intact molecule.
"""
function scale_anisotropic_coordinates!(q_out::AbstractMatrix{Float64},
                                        q_in::AbstractMatrix{<:Real},
                                        box::OrthorhombicBox,
                                        sx::Real, sy::Real, sz::Real;
                                        molecules = nothing,
                                        masses = nothing)
    size(q_in, 1) == 3 && size(q_out) == size(q_in) ||
        throw(ArgumentError("q_out and q_in must both be 3 x N"))
    s_x = Float64(sx)
    s_y = Float64(sy)
    s_z = Float64(sz)
    n = size(q_in, 2)
    if molecules === nothing
        @inbounds for j in 1:n
            q_out[1, j] = Float64(q_in[1, j]) * s_x
            q_out[2, j] = Float64(q_in[2, j]) * s_y
            q_out[3, j] = Float64(q_in[3, j]) * s_z
        end
        return n
    end
    m = masses === nothing ? ones(Float64, n) : masses
    q_mat = Matrix{Float64}(q_in)
    seen = falses(n)
    @inbounds for mol in molecules
        if length(mol) == 1
            i = mol[1]
            seen[i] = true
            q_out[1, i] = Float64(q_in[1, i]) * s_x
            q_out[2, i] = Float64(q_in[2, i]) * s_y
            q_out[3, i] = Float64(q_in[3, i]) * s_z
        else
            ref = mol[1]
            rx0 = Float64(q_mat[1, ref])
            ry0 = Float64(q_mat[2, ref])
            rz0 = Float64(q_mat[3, ref])
            M = 0.0
            cx = 0.0; cy = 0.0; cz = 0.0
            for i in mol
                seen[i] = true
                mi = Float64(m[i])
                disp = Types.unwrap_displacement(box, q_mat, i, ref)
                xi = rx0 + disp.x
                yi = ry0 + disp.y
                zi = rz0 + disp.z
                M += mi
                cx += mi * xi
                cy += mi * yi
                cz += mi * zi
            end
            invM = 1.0 / M
            cx *= invM; cy *= invM; cz *= invM
            cx_new = s_x * cx
            cy_new = s_y * cy
            cz_new = s_z * cz
            for i in mol
                disp = Types.unwrap_displacement(box, q_mat, i, ref)
                xi = rx0 + disp.x
                yi = ry0 + disp.y
                zi = rz0 + disp.z
                q_out[1, i] = cx_new + (xi - cx)
                q_out[2, i] = cy_new + (yi - cy)
                q_out[3, i] = cz_new + (zi - cz)
            end
        end
    end
    @inbounds for j in 1:n
        if !seen[j]
            q_out[1, j] = Float64(q_in[1, j]) * s_x
            q_out[2, j] = Float64(q_in[2, j]) * s_y
            q_out[3, j] = Float64(q_in[3, j]) * s_z
        end
    end
    return length(molecules)
end

"""
    scale_triclinic_coordinates!(q_out, q_in, box::TriclinicBox, scale;
                                 molecules = nothing, masses = nothing) -> Int

Apply isotropic dilation `scale` to `q_in` in a `TriclinicBox` with PBC-safe molecular unwrapping.
"""
function scale_triclinic_coordinates!(q_out::AbstractMatrix{Float64},
                                      q_in::AbstractMatrix{<:Real},
                                      box::TriclinicBox,
                                      scale::Real;
                                      molecules = nothing,
                                      masses = nothing)
    size(q_in, 1) == 3 && size(q_out) == size(q_in) ||
        throw(ArgumentError("q_out and q_in must both be 3 x N"))
    s = Float64(scale)
    n = size(q_in, 2)
    if molecules === nothing
        @inbounds for j in 1:n, d in 1:3
            q_out[d, j] = Float64(q_in[d, j]) * s
        end
        return n
    end
    m = masses === nothing ? ones(Float64, n) : masses
    q_mat = Matrix{Float64}(q_in)
    seen = falses(n)
    @inbounds for mol in molecules
        if length(mol) == 1
            i = mol[1]
            seen[i] = true
            q_out[1, i] = Float64(q_in[1, i]) * s
            q_out[2, i] = Float64(q_in[2, i]) * s
            q_out[3, i] = Float64(q_in[3, i]) * s
        else
            ref = mol[1]
            rx0 = Float64(q_mat[1, ref])
            ry0 = Float64(q_mat[2, ref])
            rz0 = Float64(q_mat[3, ref])
            M = 0.0
            cx = 0.0; cy = 0.0; cz = 0.0
            for i in mol
                seen[i] = true
                mi = Float64(m[i])
                disp = Types.unwrap_displacement(box, q_mat, i, ref)
                xi = rx0 + disp.x
                yi = ry0 + disp.y
                zi = rz0 + disp.z
                M += mi
                cx += mi * xi
                cy += mi * yi
                cz += mi * zi
            end
            invM = 1.0 / M
            cx *= invM; cy *= invM; cz *= invM
            cx_new = s * cx
            cy_new = s * cy
            cz_new = s * cz
            for i in mol
                disp = Types.unwrap_displacement(box, q_mat, i, ref)
                xi = rx0 + disp.x
                yi = ry0 + disp.y
                zi = rz0 + disp.z
                q_out[1, i] = cx_new + (xi - cx)
                q_out[2, i] = cy_new + (yi - cy)
                q_out[3, i] = cz_new + (zi - cz)
            end
        end
    end
    @inbounds for j in 1:n
        if !seen[j]
            q_out[1, j] = Float64(q_in[1, j]) * s
            q_out[2, j] = Float64(q_in[2, j]) * s
            q_out[3, j] = Float64(q_in[3, j]) * s
        end
    end
    return length(molecules)
end

"""
    scale_molecular_coordinates!(q_out, q_in, scale; box = nothing, molecules = nothing, masses = nothing) -> Int

Convenience helper to scale molecular coordinates with PBC-safe molecular unwrapping
relative to `mol[1]` when `box` is provided.
"""
function scale_molecular_coordinates!(q_out::AbstractMatrix{Float64},
                                      q_in::AbstractMatrix{<:Real},
                                      scale::Real;
                                      box::Union{Nothing,AbstractBox} = nothing,
                                      molecules = nothing,
                                      masses = nothing)
    s = Float64(scale)
    if box isa OrthorhombicBox
        return scale_anisotropic_coordinates!(q_out, q_in, box, s, s, s;
                                              molecules = molecules, masses = masses)
    elseif box isa TriclinicBox
        return scale_triclinic_coordinates!(q_out, q_in, box, s;
                                            molecules = molecules, masses = masses)
    else
        return scale_coordinates!(q_out, q_in, s; groups = molecules, masses = masses)
    end
end

scale_molecular_coordinates!(q_out::AbstractMatrix{Float64}, q_in::AbstractMatrix{<:Real},
                             box::AbstractBox, scale::Real; kwargs...) =
    scale_molecular_coordinates!(q_out, q_in, scale; box = box, kwargs...)

scale_molecular_coordinates!(q_out::AbstractMatrix{Float64}, q_in::AbstractMatrix{<:Real},
                             scale::Real, box::AbstractBox; kwargs...) =
    scale_molecular_coordinates!(q_out, q_in, scale; box = box, kwargs...)

# ---------------------------------------------------------------------------
# proposal & acceptance algebra (cleanly separated terms)
# ---------------------------------------------------------------------------

"Apply an additive log-volume displacement; its inverse is the negative displacement."
propose_log_volume(log_volume::Real, delta_log_volume::Real) =
    Float64(log_volume) + Float64(delta_log_volume)

"""
    translational_log_jacobian(n_translational_units, old_volume, new_volume) -> Float64

Exact translational coordinate-measure log Jacobian `N_mol * log(V1 / V0)`.
For unconstrained atomic scaling `N_mol == N_atoms`; for rigid/constrained
molecular COM scaling `N_mol` is the number of independent molecular centers
of mass (never the atom count).
"""
function translational_log_jacobian(n_translational_units::Integer,
                                    old_volume::Real, new_volume::Real)
    old_volume > 0 && new_volume > 0 || throw(ArgumentError("volumes must be positive"))
    n_translational_units >= 0 || throw(ArgumentError("n_translational_units must be nonnegative"))
    return Int(n_translational_units) * log(Float64(new_volume) / Float64(old_volume))
end

"""
    log_proposal_ratio(proposal::Symbol, old_volume, new_volume) -> Float64

Log Hastings proposal density ratio `log(q(V_new -> V_old) / q(V_old -> V_new))`
expressed in volume coordinates `V`:
* `:log_volume` (uniform in `log V`): `log(V_new / V_old)` (since `dV/d(log V) = V`).
* `:linear_volume` (uniform in `V`): `0.0` for `V_new > 0`, `-Inf` for `V_new <= 0`.
"""
function log_proposal_ratio(proposal::Symbol, old_volume::Real, new_volume::Real)
    _check_proposal(proposal)
    old_volume > 0 || throw(ArgumentError("old_volume must be positive"))
    if proposal === :log_volume
        new_volume > 0 || throw(ArgumentError("new_volume must be positive"))
        return log(Float64(new_volume) / Float64(old_volume))
    else
        return new_volume > 0 ? 0.0 : -Inf
    end
end

"""
    log_acceptance_components(beta, pressure, delta_energy, old_volume, new_volume,
                              n_translational_units; proposal = :log_volume,
                              surface_tension = 0.0, delta_area = 0.0)

Return the separated log-acceptance terms as a named tuple:
`(energy_term, pv_term, surface_tension_term, jacobian_term, proposal_term, total)`.
"""
function log_acceptance_components(beta::Real, pressure::Real, delta_energy::Real,
                                   old_volume::Real, new_volume::Real,
                                   n_translational_units::Integer;
                                   proposal::Symbol = :log_volume,
                                   surface_tension::Real = 0.0,
                                   delta_area::Real = 0.0)
    _check_proposal(proposal)
    isfinite(beta) && beta > 0 || throw(ArgumentError("beta must be finite and positive"))
    isfinite(pressure) || throw(ArgumentError("pressure must be finite"))
    old_volume > 0 || throw(ArgumentError("old_volume must be positive"))
    n_translational_units >= 0 || throw(ArgumentError("n_translational_units must be nonnegative"))

    if proposal === :linear_volume && new_volume <= 0
        return (energy_term = -Inf, pv_term = -Inf, surface_tension_term = -Inf,
                jacobian_term = -Inf, proposal_term = -Inf, total = -Inf)
    end
    new_volume > 0 || throw(ArgumentError("new_volume must be positive"))
    isfinite(delta_energy) || throw(ArgumentError("delta_energy must be finite"))

    b = Float64(beta)
    energy_term = -b * Float64(delta_energy)
    pv_term = -b * Float64(pressure) * (Float64(new_volume) - Float64(old_volume))
    st_term = b * Float64(surface_tension) * Float64(delta_area)
    jacobian_term = translational_log_jacobian(n_translational_units, old_volume, new_volume)
    proposal_term = log_proposal_ratio(proposal, old_volume, new_volume)
    total = energy_term + pv_term + st_term + jacobian_term + proposal_term
    return (energy_term = energy_term, pv_term = pv_term,
            surface_tension_term = st_term,
            jacobian_term = jacobian_term, proposal_term = proposal_term,
            total = total)
end

"""
    log_acceptance(beta, pressure, delta_energy, old_volume, new_volume,
                   n_translational_units; proposal = :log_volume,
                   surface_tension = 0.0, delta_area = 0.0)

Log Metropolis ratio for a volume move:

    -β * (ΔU + P * ΔV - γ * ΔA) + N_mol * log(V1 / V0) + log_proposal_ratio(proposal, V0, V1).
"""
function log_acceptance(beta::Real, pressure::Real, delta_energy::Real,
                        old_volume::Real, new_volume::Real,
                        n_translational_units::Integer;
                        proposal::Symbol = :log_volume,
                        surface_tension::Real = 0.0,
                        delta_area::Real = 0.0)
    return log_acceptance_components(beta, pressure, delta_energy, old_volume,
                                     new_volume, n_translational_units;
                                     proposal = proposal,
                                     surface_tension = surface_tension,
                                     delta_area = delta_area).total
end

# ---------------------------------------------------------------------------
# plain / molecular configurational volume moves
# ---------------------------------------------------------------------------

"""
    VolumeMoveBarostat(beta, pressure, max_delta;
                       constraints = nothing, molecules = nothing,
                       masses = nothing, proposal = :log_volume,
                       scale_mode = :auto, box = nothing,
                       constraint_tolerance = 1e-3)

Isotropic volume Monte-Carlo move kernel for an orthorhombic box.

* When `molecules === nothing`, scales all `N` atomic coordinates affinely with
  translational Jacobian `N * log(V1/V0)`. Any non-empty `constraints` raises
  `ConstraintIncompatibilityError`.
* When `molecules` is supplied (e.g. `w.molecules` from `WaterBox`), scales
  only molecular centers of mass with translational Jacobian `N_mol * log(V1/V0)`,
  keeping internal molecular geometry unchanged. Any inter-molecular constraint
  or missing mass on a multi-atom molecule is explicitly refused.
* Passing a `TriclinicBox` or `NoBox` via `box` (or later to `step!`) raises
  `UnsupportedGeometry`.
"""
struct VolumeMoveBarostat{M,C,W}
    beta::Float64
    pressure::Float64
    max_delta_log_volume::Float64
    proposal::Symbol
    scale_mode::Symbol
    molecules::M
    constraints::C
    masses::W
    constraint_tolerance::Float64
    coupling::Symbol
    surface_tension::Float64
    allow_triclinic::Bool
    function VolumeMoveBarostat(beta::Real, pressure::Real, max_delta::Real;
                                constraints = nothing,
                                molecules = nothing,
                                masses = nothing,
                                proposal::Symbol = :log_volume,
                                scale_mode::Symbol = :auto,
                                box::Union{Nothing,AbstractBox} = nothing,
                                constraint_tolerance::Real = 1e-3,
                                coupling::Symbol = :isotropic,
                                surface_tension::Real = 0.0,
                                allow_triclinic::Bool = false)
        _check_scalars(beta, pressure, max_delta)
        _check_proposal(proposal)
        (scale_mode === :auto || scale_mode === :atomic || scale_mode === :molecular) ||
            throw(ArgumentError("scale_mode must be :auto, :atomic, or :molecular"))
        (coupling === :isotropic || coupling === :semi_isotropic_xy || coupling === :anisotropic) ||
            throw(ArgumentError("coupling must be :isotropic, :semi_isotropic_xy, or :anisotropic, got $(repr(coupling))"))
        isfinite(surface_tension) || throw(ArgumentError("surface_tension must be finite"))
        isfinite(constraint_tolerance) && constraint_tolerance > 0 ||
            throw(ArgumentError("constraint_tolerance must be finite and positive"))
        box === nothing || _check_box(box; allow_triclinic = allow_triclinic)

        has_c = !_is_empty_constraints(constraints)
        if scale_mode === :atomic && has_c
            throw(ConstraintIncompatibilityError(
                "scale_mode = :atomic is incompatible with holonomic constraints"))
        end
        if scale_mode === :molecular && molecules === nothing
            has_c && throw(ConstraintIncompatibilityError(
                "scale_mode = :molecular with constraints requires `molecules`"))
            throw(ArgumentError("scale_mode = :molecular requires `molecules`"))
        end
        if molecules === nothing && has_c
            cnt = constraints isa CoupledConstraintSet ?
                  sum(length(c.edges) for c in constraints.components) : length(constraints)
            throw(ConstraintIncompatibilityError(
                "uniform volume scaling changes every absolute bond length by the " *
                "same factor, so fixed-length constraints would be violated without " *
                "molecular COM scaling (`molecules = ...`). Received $cnt constraint(s)."))
        end
        if molecules !== nothing
            # Pre-validate molecular structure and intra-molecular constraints if
            # atom count is inferable from masses or molecules.
            max_idx = 0
            has_multi = false
            for mol in molecules
                isempty(mol) && throw(ConstraintIncompatibilityError("molecule group cannot be empty"))
                length(mol) > 1 && (has_multi = true)
                for idx in mol
                    idx >= 1 || throw(ConstraintIncompatibilityError("invalid atom index $idx"))
                    max_idx = max(max_idx, Int(idx))
                end
            end
            if has_multi && masses === nothing
                throw(MissingMassError(
                    "multi-atom molecular COM scaling requires per-atom `masses`"))
            end
            if masses isa AbstractVector
                n_est = length(masses)
                max_idx <= n_est || throw(ConstraintIncompatibilityError(
                    "molecule atom index $max_idx exceeds length(masses)=$n_est"))
                validate_molecular_constraints(n_est, molecules, constraints)
            elseif max_idx > 0 && has_c
                # Validate at least up to max atom index seen in constraints/molecules
                c_max = if constraints isa CoupledConstraintSet
                    isempty(constraints.components) ? 0 : maximum(maximum(c.atoms) for c in constraints.components)
                else
                    maximum(max(Int(c[1]), Int(c[2])) for c in constraints)
                end
                validate_molecular_constraints(max(max_idx, c_max), molecules, constraints)
            end
        end
        eff_mol = scale_mode === :atomic ? nothing : molecules
        return new{typeof(eff_mol),typeof(constraints),typeof(masses)}(
            Float64(beta), Float64(pressure), Float64(max_delta),
            proposal, scale_mode, eff_mol, constraints, masses,
            Float64(constraint_tolerance), coupling, Float64(surface_tension),
            allow_triclinic)
    end
end

function _check_constraints_satisfied(cs::CoupledConstraintSet, q::Matrix{Float64}, tol::Float64,
                                      box::AbstractBox = NoBox())
    res = constraint_residuals(cs, q, box)
    if !isempty(res)
        maxres = maximum(abs, res)
        if !isfinite(maxres) || maxres > tol
            throw(ConstraintIncompatibilityError(
                "molecular constraint residual $maxres exceeds tolerance $tol before " *
                "volume move (check that constrained molecules are not split across PBC)"))
        end
    end
    return nothing
end

"""
    propose_volume_move(barostat::VolumeMoveBarostat, q, box, energy, rng = Random.default_rng())

Attempt one Monte-Carlo volume move (isotropic, semi-isotropic, or anisotropic)
under PBC-safe molecular unwrapping.
"""
function propose_volume_move(barostat::VolumeMoveBarostat, q::AbstractMatrix{<:Real},
                             box::AbstractBox, energy, rng::AbstractRNG = Random.default_rng())
    _check_box(box; allow_triclinic = barostat.allow_triclinic)
    size(q, 1) == 3 || throw(ArgumentError("positions must be 3 x N"))
    all(isfinite, q) || throw(ArgumentError("positions must be finite"))
    n_atoms = size(q, 2)

    groups, masses, group_masses, cs, has_multi =
        _prepare_scaling_groups(n_atoms, barostat.molecules, barostat.masses,
                                barostat.constraints, barostat.scale_mode)

    q_mat = Matrix{Float64}(q)
    if cs !== nothing && (barostat.molecules !== nothing || has_multi)
        _check_constraints_satisfied(cs, q_mat, barostat.constraint_tolerance, box)
    end

    old_volume = box_volume(box)
    old_energy = Float64(energy(q_mat, box))
    isfinite(old_energy) || throw(ArgumentError("energy callback returned non-finite energy"))

    n_units = barostat.molecules === nothing ? n_atoms : length(groups)

    delta_area = 0.0
    if box isa OrthorhombicBox
        if barostat.coupling === :isotropic
            if barostat.proposal === :log_volume
                delta_log_volume = (2 * rand(rng) - 1) * barostat.max_delta_log_volume
                scale = exp(delta_log_volume / 3)
                new_box = OrthorhombicBox(ntuple(i -> box.L[i] * scale, 3))
                new_volume = box_volume(new_box)
                sx = sy = sz = scale
            else
                dV = (2 * rand(rng) - 1) * barostat.max_delta_log_volume
                new_volume_target = old_volume + dV
                if !(isfinite(new_volume_target) && new_volume_target > 0)
                    return (accepted = false, box = box, old_volume = old_volume,
                            new_volume = new_volume_target, old_energy = old_energy,
                            new_energy = NaN, delta_log_volume = NaN,
                            log_acceptance = -Inf, n_translational_units = n_units,
                            log_jacobian = -Inf, log_proposal_ratio = -Inf,
                            delta_area = 0.0, surface_tension_work = 0.0)
                end
                scale = cbrt(new_volume_target / old_volume)
                new_box = OrthorhombicBox(ntuple(i -> box.L[i] * scale, 3))
                new_volume = box_volume(new_box)
                delta_log_volume = log(new_volume / old_volume)
                sx = sy = sz = scale
            end
        elseif barostat.coupling === :semi_isotropic_xy
            d_xy = (2 * rand(rng) - 1) * barostat.max_delta_log_volume
            d_z = (2 * rand(rng) - 1) * barostat.max_delta_log_volume
            sx = exp(d_xy)
            sy = exp(d_xy)
            sz = exp(d_z)
            new_box = OrthorhombicBox((box.L[1] * sx, box.L[2] * sy, box.L[3] * sz))
            new_volume = box_volume(new_box)
            delta_log_volume = 2.0 * d_xy + d_z
            delta_area = (new_box.L[1] * new_box.L[2]) - (box.L[1] * box.L[2])
        elseif barostat.coupling === :anisotropic
            dx = (2 * rand(rng) - 1) * barostat.max_delta_log_volume
            dy = (2 * rand(rng) - 1) * barostat.max_delta_log_volume
            dz = (2 * rand(rng) - 1) * barostat.max_delta_log_volume
            sx = exp(dx)
            sy = exp(dy)
            sz = exp(dz)
            new_box = OrthorhombicBox((box.L[1] * sx, box.L[2] * sy, box.L[3] * sz))
            new_volume = box_volume(new_box)
            delta_log_volume = dx + dy + dz
            delta_area = (new_box.L[1] * new_box.L[2]) - (box.L[1] * box.L[2])
        else
            throw(ArgumentError("unsupported coupling $(repr(barostat.coupling))"))
        end
    elseif box isa TriclinicBox && barostat.allow_triclinic
        if barostat.coupling !== :isotropic
            throw(UnsupportedGeometry("TriclinicBox only supports isotropic coupling"))
        end
        delta_log_volume = (2 * rand(rng) - 1) * barostat.max_delta_log_volume
        scale = exp(delta_log_volume / 3)
        new_box = TriclinicBox(box.H .* scale)
        new_volume = box_volume(new_box)
        sx = sy = sz = scale
        delta_area = 0.0
    else
        throw(UnsupportedGeometry("box type $(typeof(box)) not supported"))
    end

    candidate_q = Matrix{Float64}(undef, 3, n_atoms)
    if box isa OrthorhombicBox
        if barostat.molecules === nothing
            scale_anisotropic_coordinates!(candidate_q, q_mat, box, sx, sy, sz)
        else
            scale_anisotropic_coordinates!(candidate_q, q_mat, box, sx, sy, sz;
                                           molecules = groups, masses = masses)
        end
    else # TriclinicBox
        if barostat.molecules === nothing
            scale_triclinic_coordinates!(candidate_q, q_mat, box, sx)
        else
            scale_triclinic_coordinates!(candidate_q, q_mat, box, sx;
                                         molecules = groups, masses = masses)
        end
    end

    new_energy = Float64(energy(candidate_q, new_box))
    isfinite(new_energy) || throw(ArgumentError("energy callback returned non-finite energy"))

    acc = log_acceptance_components(barostat.beta, barostat.pressure,
                                    new_energy - old_energy, old_volume,
                                    new_volume, n_units;
                                    proposal = barostat.proposal,
                                    surface_tension = barostat.surface_tension,
                                    delta_area = delta_area)
    accepted = isfinite(acc.total) && (log(rand(rng)) < min(0.0, acc.total))
    st_work = barostat.beta * barostat.surface_tension * delta_area
    if accepted
        q .= candidate_q
        return (accepted = true, box = new_box, old_volume = old_volume,
                new_volume = new_volume, old_energy = old_energy,
                new_energy = new_energy, delta_log_volume = delta_log_volume,
                log_acceptance = acc.total, n_translational_units = n_units,
                log_jacobian = acc.jacobian_term,
                log_proposal_ratio = acc.proposal_term,
                delta_area = delta_area, surface_tension_work = st_work)
    end
    return (accepted = false, box = box, old_volume = old_volume,
            new_volume = new_volume, old_energy = old_energy,
            new_energy = new_energy, delta_log_volume = delta_log_volume,
            log_acceptance = acc.total, n_translational_units = n_units,
            log_jacobian = acc.jacobian_term,
            log_proposal_ratio = acc.proposal_term,
            delta_area = delta_area, surface_tension_work = st_work)
end

"""
    step!(barostat::VolumeMoveBarostat, q, box, energy, rng = Random.default_rng())

Attempt one volume move via [`propose_volume_move`](@ref).
"""
step!(barostat::VolumeMoveBarostat, q::AbstractMatrix{<:Real},
      box::AbstractBox, energy, rng::AbstractRNG = Random.default_rng()) =
    propose_volume_move(barostat, q, box, energy, rng)

# ---------------------------------------------------------------------------
# pressure estimator: kinetic + interaction (∂U/∂h|_s) + constraint
# ---------------------------------------------------------------------------

"""
    virial_from_box_derivative(G, box::OrthorhombicBox) -> Float64

Convert the 3×3 fixed-fractional box derivative `G[a,b] = ∂U/∂h[a,b]` to the
scalar interaction virial `W_int = -tr(G * hᵀ)`. Refuses non-orthorhombic boxes
with `UnsupportedGeometry`.
"""
function virial_from_box_derivative(G::AbstractMatrix{<:Real}, box::AbstractBox)
    _check_box(box)
    size(G) == (3, 3) || throw(ArgumentError("box derivative G must be 3 x 3, got $(size(G))"))
    all(isfinite, G) || throw(ArgumentError("box derivative G contains non-finite values"))
    L = (box::OrthorhombicBox).L
    return -(Float64(G[1, 1]) * L[1] + Float64(G[2, 2]) * L[2] + Float64(G[3, 3]) * L[3])
end

"""
    PressureComponents

Separated mechanical pressure contributions `(2K + W_int + W_con) / (3V)`:
* `kinetic = 2 * kinetic_energy / (3 * volume)`
* `interaction = interaction_virial / (3 * volume)`
* `constraint = constraint_virial / (3 * volume)`
* `total = kinetic + interaction + constraint`
"""
struct PressureComponents
    kinetic::Float64
    interaction::Float64
    constraint::Float64
    total::Float64
    volume::Float64
    kinetic_energy::Float64
    interaction_virial::Float64
    constraint_virial::Float64
end

"""
    NPTModel(energy; virial = nothing, box_derivative = nothing,
             forces = nothing, masses = nothing, constraints = nothing,
             molecules = nothing, pressure_mode = :auto)

A force model for the NPT barostat and separated pressure estimator.

* `energy(q, box)` → potential energy `U`.
* `box_derivative(q, box)` → `3×3` matrix `∂U/∂h|_s` at fixed fractional
  coordinates (preferred), OR `virial(q, box)` → scalar interaction virial
  `W_int = -tr((∂U/∂h|_s) hᵀ)`. Supplying neither makes `NPTBarostat` and
  `configurational_pressure` throw `MissingVirialError`; supplying both throws
  `ArgumentError`.
* `forces(q, box)` → `3×N` external forces `F_ext = -∇_q U`, required whenever
  constraints (or `:molecular_com` pressure) are active.
* `masses` → per-atom masses (`Vector`) or scalar uniform mass.
* `constraints` → optional `CoupledConstraintSet` or `(i, j, d)` list.
* `molecules` → optional molecular atom index groups for COM scaling/pressure.
* `pressure_mode` → `:auto` (default: atomic `K` + dynamical constraint virial
  when constrained), `:dynamical` (atomic `K` + dynamical constraint virial), or
  `:molecular_com` (molecular COM `K_COM` + intramolecular external-force virial).
"""
struct NPTModel{E,V,BD,F,M,C,MOL}
    energy::E
    virial::V
    box_derivative::BD
    forces::F
    masses::M
    constraints::C
    molecules::MOL
    pressure_mode::Symbol
    function NPTModel(energy::E;
                      virial::V = nothing,
                      box_derivative::BD = nothing,
                      forces::F = nothing,
                      masses::M = nothing,
                      constraints::C = nothing,
                      molecules::MOL = nothing,
                      pressure_mode::Symbol = :auto) where {E,V,BD,F,M,C,MOL}
        (virial !== nothing && box_derivative !== nothing) &&
            throw(ArgumentError("specify either `box_derivative` or `virial`, not both"))
        (pressure_mode === :auto || pressure_mode === :dynamical ||
         pressure_mode === :molecular_com) ||
            throw(ArgumentError("pressure_mode must be :auto, :dynamical, or :molecular_com"))
        return new{E,V,BD,F,M,C,MOL}(energy, virial, box_derivative, forces,
                                     masses, constraints, molecules, pressure_mode)
    end
end

function _kinetic_energy(p::AbstractMatrix{<:Real}, m::Vector{Float64})
    size(p, 1) == 3 || throw(ArgumentError("momenta must be 3 x N"))
    length(m) == size(p, 2) || throw(ArgumentError("mass count != momentum columns"))
    k = 0.0
    @inbounds for j in eachindex(m), d in 1:3
        k += Float64(p[d, j])^2 / m[j]
    end
    return 0.5 * k
end

function _com_kinetic_energy(p::AbstractMatrix{<:Real}, m::Vector{Float64},
                             groups::Vector{Vector{Int}}, group_masses::Vector{Float64})
    k = 0.0
    @inbounds for (alpha, g) in enumerate(groups)
        px = 0.0; py = 0.0; pz = 0.0
        for i in g
            px += Float64(p[1, i])
            py += Float64(p[2, i])
            pz += Float64(p[3, i])
        end
        k += (px * px + py * py + pz * pz) / group_masses[alpha]
    end
    return 0.5 * k
end

"""
    static_constraint_multipliers(q, m, cs::CoupledConstraintSet, F_ext) -> Vector{Float64}

Solve `(G W Gᵀ) λ = G W F_ext` for static constraint multipliers (force units).
"""
static_constraint_multipliers(q::AbstractMatrix{Float64}, m::AbstractVector{Float64},
                              cs::CoupledConstraintSet, F_ext::AbstractMatrix{Float64}) =
    constraint_multipliers(q, m, cs, F_ext)

"""
    dynamical_constraint_multipliers(q, p, m, cs::CoupledConstraintSet, F_ext) -> Vector{Float64}

Solve `(G W Gᵀ) λ = G W F_ext + Ġ v` for the exact dynamical constraint
multipliers (force units) at coordinates `q`, momenta `p`, and external forces
`F_ext`. Because `c_k = |q_j - q_i|² - d_k²`, `(Ġ v)_k = 2 |v_j - v_i|²`, which
includes the centrifugal constraint force that cancels rotational kinetic energy
in the atomic pressure estimator `(2K + W_int + W_c)/(3V)`.
"""
function dynamical_constraint_multipliers(q::AbstractMatrix{Float64},
                                          p::AbstractMatrix{Float64},
                                          m::AbstractVector{Float64},
                                          cs::CoupledConstraintSet,
                                          F_ext::AbstractMatrix{Float64})
    size(q, 1) == 3 && size(p) == size(q) && size(F_ext) == size(q) ||
        throw(ArgumentError("q, p, and F_ext must all be 3 x N"))
    length(m) == size(q, 2) || throw(ArgumentError("mass count != N"))

    total_edges = isempty(cs.components) ? 0 : sum(length(c.edges) for c in cs.components)
    λ_out = Float64[]
    sizehint!(λ_out, total_edges)

    for comp in cs.components
        n = length(comp.atoms)
        mm = length(comp.edges)
        nn = 3 * n
        G = zeros(Float64, mm, nn)
        W = Vector{Float64}(undef, nn)
        rhs = Vector{Float64}(undef, mm)
        @inbounds for (a, atom) in enumerate(comp.atoms)
            inv_m = 1.0 / m[atom]
            W[3 * (a - 1) + 1] = inv_m
            W[3 * (a - 1) + 2] = inv_m
            W[3 * (a - 1) + 3] = inv_m
        end
        @inbounds for (k, (a, b, _)) in enumerate(comp.edges)
            i, j = comp.atoms[a], comp.atoms[b]
            rx = q[1, j] - q[1, i]
            ry = q[2, j] - q[2, i]
            rz = q[3, j] - q[3, i]
            G[k, 3 * (a - 1) + 1] = -2.0 * rx
            G[k, 3 * (a - 1) + 2] = -2.0 * ry
            G[k, 3 * (a - 1) + 3] = -2.0 * rz
            G[k, 3 * (b - 1) + 1] = 2.0 * rx
            G[k, 3 * (b - 1) + 2] = 2.0 * ry
            G[k, 3 * (b - 1) + 3] = 2.0 * rz

            # G W F_ext contribution
            gw_f = 2.0 * (rx * (F_ext[1, j] / m[j] - F_ext[1, i] / m[i]) +
                          ry * (F_ext[2, j] / m[j] - F_ext[2, i] / m[i]) +
                          rz * (F_ext[3, j] / m[j] - F_ext[3, i] / m[i]))
            # Centrifugal term (G_dot v)_k = 2 |v_j - v_i|^2
            dvx = p[1, j] / m[j] - p[1, i] / m[i]
            dvy = p[2, j] / m[j] - p[2, i] / m[i]
            dvz = p[3, j] / m[j] - p[3, i] / m[i]
            gdot_v = 2.0 * (dvx * dvx + dvy * dvy + dvz * dvz)
            rhs[k] = gw_f + gdot_v
        end
        GW = G .* reshape(W, 1, :)
        A = Symmetric(GW * transpose(G))
        F_chol = cholesky(A; check = false)
        issuccess(F_chol) || throw(UnsupportedConstraintSet(
            "rank-deficient constraint Jacobian on component atoms $(comp.atoms)"))
        λ_loc = F_chol \ rhs
        for k in 1:mm
            push!(λ_out, λ_loc[k])
        end
    end
    return λ_out
end

function _intramolecular_force_virial(q::Matrix{Float64}, F_ext::Matrix{Float64},
                                      m::Vector{Float64}, groups::Vector{Vector{Int}},
                                      group_masses::Vector{Float64})
    # -sum_{alpha} sum_{i in alpha} (q_i - R_alpha) . F_ext_i
    w = 0.0
    @inbounds for (alpha, g) in enumerate(groups)
        length(g) == 1 && continue
        invM = 1.0 / group_masses[alpha]
        rx = 0.0; ry = 0.0; rz = 0.0
        for i in g
            mi = m[i]
            rx += mi * q[1, i]
            ry += mi * q[2, i]
            rz += mi * q[3, i]
        end
        rx *= invM; ry *= invM; rz *= invM
        for i in g
            w -= (q[1, i] - rx) * F_ext[1, i] +
                 (q[2, i] - ry) * F_ext[2, i] +
                 (q[3, i] - rz) * F_ext[3, i]
        end
    end
    return w
end

"""
    pressure_components(model::NPTModel, q, p, box; mode = model.pressure_mode) -> PressureComponents

Compute the separated kinetic, interaction (from `∂U/∂h|_s`), and constraint
pressure contributions `(2K + W_int + W_con) / (3V)`.

* When `constraints` are present and `mode` is `:auto` or `:dynamical`, uses
  atomic `K` and dynamical constraint virial `W_c` from
  `dynamical_constraint_multipliers`.
* When `mode == :molecular_com`, uses molecular COM kinetic energy `K_COM` and
  the intramolecular external-force constraint virial `-Σ (q_i - R_α) ⋅ F_ext_i`
  (identical to `W_c^static` for rigid molecules).
* Throws `UnsupportedGeometry` for non-orthorhombic boxes, `MissingVirialError`
  if neither `box_derivative` nor `virial` (or `forces` when constrained) is
  supplied, and `MissingMassError` if `masses === nothing`.
"""
function pressure_components(model::NPTModel, q::AbstractMatrix{<:Real},
                             p::AbstractMatrix{<:Real}, box::AbstractBox;
                             mode::Symbol = model.pressure_mode)
    _check_box(box)
    (mode === :auto || mode === :dynamical || mode === :molecular_com) ||
        throw(ArgumentError("mode must be :auto, :dynamical, or :molecular_com"))
    size(q, 1) == 3 || throw(ArgumentError("positions must be 3 x N"))
    size(p) == size(q) || throw(ArgumentError("momenta must match positions"))
    all(isfinite, q) || throw(ArgumentError("positions must be finite"))
    all(isfinite, p) || throw(ArgumentError("momenta must be finite"))

    (model.virial === nothing && model.box_derivative === nothing) &&
        throw(MissingVirialError(
            "this NPTModel has neither box_derivative nor virial. Configurational " *
            "pressure requires W = -tr((∂U/∂h|_s) hᵀ) from the same force field; " *
            "substituting NkT/V or finite differences is not permitted."))

    n = size(q, 2)
    m = _mass_vector(model.masses, n)
    q_mat = Matrix{Float64}(q)
    p_mat = Matrix{Float64}(p)
    volume = box_volume(box)

    # 1. Interaction virial from ∂U/∂h|_s or explicit virial callback
    W_int = if model.box_derivative !== nothing
        G = model.box_derivative(q_mat, box)
        virial_from_box_derivative(G, box)
    else
        v = model.virial(q_mat, box)
        val = v isa AbstractMatrix ? Float64(tr(v)) : Float64(v)
        isfinite(val) || throw(ArgumentError("virial callback returned a non-finite value"))
        val
    end

    # 2. Kinetic & constraint contributions
    has_c = !_is_empty_constraints(model.constraints)
    if mode === :molecular_com
        model.molecules === nothing &&
            throw(ArgumentError("mode = :molecular_com requires `molecules` in NPTModel"))
        groups, _ = validate_molecular_constraints(n, model.molecules, model.constraints)
        group_masses = Float64[sum(m[i] for i in g) for g in groups]
        K = _com_kinetic_energy(p_mat, m, groups, group_masses)
        if any(g -> length(g) > 1, groups)
            model.forces === nothing && throw(MissingVirialError(
                "molecular COM pressure on multi-atom molecules requires `forces(q, box)` " *
                "for the intramolecular constraint/internal virial"))
            F_ext = Matrix{Float64}(model.forces(q_mat, box))
            size(F_ext) == size(q_mat) ||
                throw(ArgumentError("forces callback returned wrong shape $(size(F_ext))"))
            all(isfinite, F_ext) || throw(ArgumentError("forces callback returned non-finite values"))
            W_con = _intramolecular_force_virial(q_mat, F_ext, m, groups, group_masses)
        else
            W_con = 0.0
        end
    elseif has_c
        model.forces === nothing && throw(MissingVirialError(
            "constrained pressure requires external `forces(q, box)` to solve " *
            "constraint multipliers (G W Gᵀ) λ = G W F_ext + Ġ v"))
        cs = _as_coupled_constraints(n, model.constraints)
        F_ext = Matrix{Float64}(model.forces(q_mat, box))
        size(F_ext) == size(q_mat) ||
            throw(ArgumentError("forces callback returned wrong shape $(size(F_ext))"))
        all(isfinite, F_ext) || throw(ArgumentError("forces callback returned non-finite values"))
        K = _kinetic_energy(p_mat, m)
        λ = dynamical_constraint_multipliers(q_mat, p_mat, m, cs, F_ext)
        W_con = constraint_virial(q_mat, m, cs, λ)
    else
        K = _kinetic_energy(p_mat, m)
        W_con = 0.0
    end

    inv_3V = 1.0 / (3.0 * volume)
    p_kin = 2.0 * K * inv_3V
    p_int = W_int * inv_3V
    p_con = W_con * inv_3V
    return PressureComponents(p_kin, p_int, p_con, p_kin + p_int + p_con,
                              volume, K, W_int, W_con)
end

"""
    configurational_pressure(model, q, p, box; mode = model.pressure_mode) -> Float64

Total configurational pressure `(2K + W_int + W_con) / (3V)` computed via
[`pressure_components`](@ref).
"""
configurational_pressure(model::NPTModel, q::AbstractMatrix{<:Real},
                         p::AbstractMatrix{<:Real}, box::AbstractBox;
                         mode::Symbol = model.pressure_mode) =
    pressure_components(model, q, p, box; mode = mode).total

# ---------------------------------------------------------------------------
# pressure-aware NPT driver
# ---------------------------------------------------------------------------

"""
    NPTBarostat(beta, pressure, max_delta, model;
                constraints = nothing, molecules = nothing,
                proposal = :log_volume, scale_mode = :auto,
                box = nothing, constraint_tolerance = 1e-3)

Validated NPT driver: one isotropic volume move (atomic or molecular-COM) plus
separated kinetic, interaction, and constraint pressure diagnostics per call.
"""
struct NPTBarostat{M,VB}
    beta::Float64
    pressure::Float64
    max_delta_log_volume::Float64
    model::M
    volume_barostat::VB
    function NPTBarostat(beta::Real, pressure::Real, max_delta::Real,
                         model::NPTModel;
                         constraints = nothing,
                         molecules = nothing,
                         proposal::Symbol = :log_volume,
                         scale_mode::Symbol = :auto,
                         box::Union{Nothing,AbstractBox} = nothing,
                         constraint_tolerance::Real = 1e-3,
                         coupling::Symbol = :isotropic,
                         surface_tension::Real = 0.0,
                         allow_triclinic::Bool = false)
        eff_constraints = constraints !== nothing ? constraints : model.constraints
        eff_molecules = molecules !== nothing ? molecules : model.molecules
        (model.virial === nothing && model.box_derivative === nothing) &&
            throw(MissingVirialError(
                "NPTBarostat requires a box_derivative or virial provider in NPTModel; " *
                "a model without one cannot report a physically meaningful pressure."))
        model.masses === nothing && throw(MissingMassError(
            "NPTBarostat requires masses in NPTModel for the kinetic pressure."))
        if !_is_empty_constraints(eff_constraints) && model.forces === nothing
            throw(MissingVirialError(
                "NPTBarostat on a constrained model requires `forces` in NPTModel " *
                "to compute constraint multipliers."))
        end
        vb = VolumeMoveBarostat(beta, pressure, max_delta;
                                constraints = eff_constraints,
                                molecules = eff_molecules,
                                masses = model.masses,
                                proposal = proposal,
                                scale_mode = scale_mode,
                                box = box,
                                constraint_tolerance = constraint_tolerance,
                                coupling = coupling,
                                surface_tension = surface_tension,
                                allow_triclinic = allow_triclinic)
        return new{typeof(model),typeof(vb)}(
            Float64(beta), Float64(pressure), Float64(max_delta), model, vb)
    end
end

function npt_step!(barostat::NPTBarostat, q::AbstractMatrix{<:Real},
                   p::AbstractMatrix{<:Real}, box::AbstractBox,
                   rng::AbstractRNG = Random.default_rng())
    p_old = pressure_components(barostat.model, q, p, box)
    move = step!(barostat.volume_barostat, q, box, barostat.model.energy, rng)
    p_new = move.accepted ?
        pressure_components(barostat.model, q, p, move.box) :
        p_old
    return merge(move, (pressure_old = p_old.total,
                        pressure_new = p_new.total,
                        pressure_components_old = p_old,
                        pressure_components_new = p_new))
end

const npt_mc_step! = npt_step!

end # module MonteCarloBarostat
