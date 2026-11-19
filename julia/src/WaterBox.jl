"""
Rigid TIP3P water: box builder, composed force field, NVT runner, analysis.

This module is the *composition* layer. It adds no new force law: it builds a
rigid TIP3P water box, wires the existing engine factors into one Hamiltonian,
projects the three rigid distance constraints per molecule every step, and
measures what came out. It is included and exported by `FoundationCore` and
has a registered suite (`test/test_water_box.jl`); its trajectory output is
demonstration evidence only, not validated molecular dynamics.

## Model

TIP3P (Jorgensen 1983) in the unit system `(nm, kJ/mol, amu, ps, e)`:

| quantity | value |
| --- | --- |
| `q_O`, `q_H` | -0.834 e, +0.417 e |
| `sigma_O`, `eps_O` | 0.315061 nm, 0.636386 kJ/mol |
| `sigma_H`, `eps_H` | no LJ site (0), implemented as `eps=0` with a placeholder `sigma` |
| `r_OH`, `angle HOH` | 0.09572 nm, 104.52 degrees |
| `m_O`, `m_H` | 15.9994, 1.008 amu |

The O-H and H-H nonbonded pairs inside a molecule are excluded twice, once per
backend: `PairTopology` (Lennard-Jones, `StreamedLJ`) and
`PME.PairCorrections` (electrostatics, `PMEFactor`). The geometry is held rigid
by `Constraints.CoupledConstraintSet`, which solves the coupled 3-constraint
system per molecule (the shared-atom case a per-pair RATTLE cannot do).

LJ is evaluated O-O only: `eps_H = 0` makes every H-involving Lorentz-Berthelot
pair have `eps_ij = 0`, hence exactly zero energy and force, while the pair is
still enumerated (so an accidentally missing exclusion is observable).

## Unit system

Masses in amu, lengths in nm, time in ps give energies in kJ/mol *exactly*
(`1 amu nm^2/ps^2 = 1.660539e-21 J = 1 kJ/mol`), so the Coulomb prefactor is
`138.935456` kJ nm / (mol e^2) and `k_B = 0.008314463` kJ/(mol K). `dt = 0.002`
is therefore 2 fs and `gamma = 1.0` is 1/ps.

## What is not claimed

* No long-range dispersion correction: the LJ tail beyond `rcut` is dropped, as
  in the usual plain-cutoff TIP3P setup. Energies are not directly comparable
  to a dispersion-corrected reference.
 * PME accuracy is mesh-dependent. The module's registered tests check composed
   force finite differences and small-system PME/Ewald parity, not bulk dynamics.
 * The historical 10 ps, 200-water numbers predate the principal-axis rigid-body
   drift and COM-denominator fixes and are not validation of the current
   integrator. The current bounded diagnostic is 20 ps equilibration + 20 ps
   production at 200 molecules in
   `julia/experiments/tip3p_trajectory/README.md` (`runs/w200_corrected/`); it
   is still not equilibrated bulk water.
* Pressure is not computed, so the runs are strictly NVT (the density is an
  input, not a result).
"""
module WaterBox

using Random
using Statistics
using Printf

using LinearAlgebra

using ..Types
using ..Factors
using ..Constraints
using ..Settle
using ..Dynamics
using ..StreamedLJ
using ..PME

export TIP3P_PARAMETERS, TIP3P_UNITS, KB, COULOMB_PREFACTOR, R_HH, MIN_OO,
       TIP3PWater, WaterField, WaterTrajectory, Sample, RelaxReport,
       EquilibrationResult, ProductionResult, RunReport, AccumulatingFactor,
       tip3p_water, water_field, relax!, rethermalize!, tangent_force!,
       integrate!, equilibrate!, production!, nve_energy_band,
       autocorrelation_time, mean_with_error, ols_slope_hac,
       diffusion_from_msd, radial_distribution, molecule_frame,
       box_density, summarize_run, print_report,
       default_rcut, default_pme_rcut, default_grid, molar_volume,
       minimum_oo_distance, rdf_extrema,
       water_integrator_spec, WaterCheckpoint, NVEDiagnostics, NVERunResult,
       run_nve!, water_residuals

# ---------------------------------------------------------------------------
# physical constants (nm, kJ/mol, amu, ps, e)
# ---------------------------------------------------------------------------

"TIP3P charges, geometry and LJ parameters (Jorgensen et al., JCP 79, 926, 1983)."
const TIP3P_PARAMETERS = (
    q_O = -0.834,
    q_H = 0.417,
    sigma_O = 0.315061,      # nm
    eps_O = 0.636386,        # kJ/mol
    sigma_H = 1.0,           # placeholder: eps_H = 0 makes every H pair exactly zero
    eps_H = 0.0,
    r_OH = 0.09572,          # nm
    theta_HOH = 104.52,      # degrees
    m_O = 15.9994,           # amu
    m_H = 1.008,             # amu
)

const COULOMB_PREFACTOR = 138.935456   # kJ nm / (mol e^2)
const KB = 0.008314462618             # kJ / (mol K)
const AVOGADRO = 6.02214076e23
const M_WATER = 2 * TIP3P_PARAMETERS.m_H + TIP3P_PARAMETERS.m_O   # amu = g/mol
const R_HH = 2 * TIP3P_PARAMETERS.r_OH * sind(TIP3P_PARAMETERS.theta_HOH / 2)
const THETA_HOH = deg2rad(TIP3P_PARAMETERS.theta_HOH)

"Minimum allowed O-O distance at build time (nm); below this the builder reseeds."
const MIN_OO = 0.25

"""
Physical unit system for this module: lengths in nm, energies in kJ/mol, masses
in amu, charges in elementary charge units, times in ps.
`1 amu nm^2/ps^2 = 1 kJ/mol` exactly, so no factor appears in the kinetic energy.
"""
const TIP3P_UNITS = Types.Units("TIP3P (nm, kJ/mol, amu, e, ps)",
                                1.0e-9,                       # m per length unit
                                1000.0 / AVOGADRO,            # J per energy unit
                                1.66053906660e-27,            # kg per mass unit
                                1.602176634e-19,              # C per charge unit
                                COULOMB_PREFACTOR,
                                KB)

"Constraint projection tolerance on squared distance residuals (nm^2)."
const CONSTRAINT_TOL = 1e-12

# ---------------------------------------------------------------------------
# the built box
# ---------------------------------------------------------------------------

"""
A rigid TIP3P water box.

Fields
* `positions` — `3 x 3N` coordinates in nm, molecule `i` owns atoms
  `(3i-2, 3i-1, 3i)` = `(O, H1, H2)`.
* `masses`, `charges` — length `3N`.
* `epsilon`, `sigma` — per-atom LJ parameters; H carries `eps = 0`.
* `molecules` — `Vector{NTuple{3,Int}}` atom index triples.
* `constraints` — `CoupledConstraintSet` with the three rigid distances per
  molecule (O-H1, O-H2, H1-H2), one independent component per molecule.
* `lj_topology` — `StreamedLJ.PairTopology` with the three intramolecular pairs
  excluded per molecule.
* `pme_corrections` — `PME.PairCorrections` with the same pairs excluded.
* `box` — orthorhombic periodic cell; `density` is the actual (g/cm^3).
"""
struct TIP3PWater
    positions::Matrix{Float64}
    masses::Vector{Float64}
    charges::Vector{Float64}
    epsilon::Vector{Float64}
    sigma::Vector{Float64}
    molecules::Vector{NTuple{3,Int}}
    constraints::Constraints.CoupledConstraintSet
    lj_topology::StreamedLJ.PairTopology
    pme_corrections::PME.PairCorrections
    box::Types.OrthorhombicBox
    density::Float64
    seed::Int
end

Base.show(io::IO, w::TIP3PWater) =
    print(io, "TIP3PWater(N=", length(w.molecules), " molecules, ",
          size(w.positions, 2), " atoms, L=", w.box.L,
          " nm, rho=", round(w.density, digits = 4), " g/cm^3)")

# ---------------------------------------------------------------------------
# builder
# ---------------------------------------------------------------------------

"Volume of one water molecule at the given mass density, in nm^3."
molar_volume(density::Real) = M_WATER / (density * AVOGADRO) * 1.0e21

"""
Choose per-axis lattice counts `(nx, ny, nz)` for `n` molecules.

Only near-cubic factorisations (`nz/nx <= 1.25`) are accepted, and the most
isotropic one wins. Anything else falls back to the `ceil(n^(1/3))` cube with a
few vacant sites: an anisotropic cell would force a long PME real-space cutoff
(an anisotropic cell needs a longer real-space cutoff and neighbour shell for
the same density; see `water_field`) and a
longer, more expensive neighbour shell for the same density.
"""
function _lattice_counts(n::Int)
    cube = ceil(Int, Float64(n)^(1 / 3))
    best = (cube, cube, cube)
    bestratio = Inf
    found = false
    for nx in 1:(cube + 1)
        n % nx == 0 || continue
        rest = n ÷ nx
        for ny in nx:rest
            rest % ny == 0 || continue
            nz = rest ÷ ny
            nz < ny && continue
            ratio = nz / nx
            ratio <= 1.25 || continue
            if ratio < bestratio - 1e-12
                bestratio = ratio
                best = (nx, ny, nz)
                found = true
            end
        end
    end
    return (best, found)
end

"Wrap one scalar coordinate into `[0, L)`."
@inline _wrap1(x::Float64, L::Float64) = x - L * floor(x / L)

"Rotation matrix from a uniformly random unit quaternion (seeded `rng`)."
function _random_rotation(rng::AbstractRNG)
    w, x, y, z = randn(rng, 4)
    n = sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return [1 - 2(y * y + z * z) 2(x * y - w * z)     2(x * z + w * y);
            2(x * y + w * z)     1 - 2(x * x + z * z) 2(y * z - w * x);
            2(x * z - w * y)     2(y * z + w * x)     1 - 2(x * x + y * y)]
end

"Minimum O-O distance over all molecule pairs under the orthorhombic minimum image."
function minimum_oo_distance(positions::AbstractMatrix, molecules, L)
    n = length(molecules)
    O = [3i - 2 for i in 1:n]
    best = Inf
    @inbounds for b in 2:n, a in 1:(b - 1)
        dx = positions[1, O[a]] - positions[1, O[b]]
        dy = positions[2, O[a]] - positions[2, O[b]]
        dz = positions[3, O[a]] - positions[3, O[b]]
        dx -= L[1] * round(dx / L[1])
        dy -= L[2] * round(dy / L[2])
        dz -= L[3] * round(dz / L[3])
        r2 = dx * dx + dy * dy + dz * dz
        r2 < best && (best = r2)
    end
    return sqrt(best)
end

"""
    tip3p_water(nmolecules; density=1.0, seed=1, box=nothing, min_oo=MIN_OO)

Build `nmolecules` rigid TIP3P waters on a near-cubic lattice with random
(seeded) orientations.

The cell is sized so that the mass density is exactly `density` g/cm^3:
`n * V_mol = V`, with `V_mol = M_water/(rho N_A)`. Molecules sit at lattice
sites offset by half a spacing, so the O-O ordering at build time is uniform and
no two oxygens are closer than the lattice constant (~0.31 nm at 1 g/cm^3).

`box` overrides the density-derived cell (a 3-tuple of side lengths in nm); this
is used by tests that need a small dilute cell, and `density` then reports the
actual value. If any O-O pair is closer than `min_oo` the whole configuration is
jittered with a fresh sub-seed (up to 20 attempts) before an error is raised —
the tolerance is never silently loosened.
"""
function tip3p_water(nmolecules::Integer; density::Real = 1.0, seed::Integer = 1,
                     box = nothing, min_oo::Real = MIN_OO)
    n = Int(nmolecules)
    n >= 1 || throw(ArgumentError("nmolecules must be >= 1, got $nmolecules"))
    (isfinite(density) && density > 0) ||
        throw(ArgumentError("density must be finite and positive, got $density"))
    n <= 2_000_000 || throw(ArgumentError("refusing to build $n molecules"))

    p = TIP3P_PARAMETERS
    counts, exact = _lattice_counts(n)
    prod_sites = prod(counts)
    vmol = molar_volume(density)

    positions = Matrix{Float64}(undef, 3, 3n)
    masses = Vector{Float64}(undef, 3n)
    charges = Vector{Float64}(undef, 3n)
    epsilon = Vector{Float64}(undef, 3n)
    sigma = Vector{Float64}(undef, 3n)
    molecules = Vector{NTuple{3,Int}}(undef, n)

    build_rng = MersenneTwister(seed)

    if box === nothing
        # Lattice constant from the target molecular volume. When the count is
        # not a perfect factorisation the cell is scaled down by (n/prod)^(1/3)
        # so the *reported* density is exactly the requested one.
        a0 = cbrt(vmol)
        s = (n / prod_sites)^(1 / 3)
        a = a0 * s
        L = (counts[1] * a, counts[2] * a, counts[3] * a)
    else
        L = (Float64(box[1]), Float64(box[2]), Float64(box[3]))
        # Grow the axis with the largest current spacing until there are enough
        # sites. This keeps the sites as evenly spread as the cell allows (a
        # dilute 2-molecule test cell gets a 2x1x1 lattice, not a single site).
        cts = [1, 1, 1]
        while prod(cts) < n
            k = argmax([L[1] / cts[1], L[2] / cts[2], L[3] / cts[3]])
            cts[k] += 1
        end
        counts = (cts[1], cts[2], cts[3])
        prod_sites = prod(counts)
    end

    sites = vec([(i, j, k) for k in 0:(counts[3] - 1),
                             j in 0:(counts[2] - 1),
                             i in 0:(counts[1] - 1)])
    length(sites) >= n || throw(ArgumentError(
        "internal lattice error: $(length(sites)) sites for $n molecules"))
    perm = randperm(build_rng, length(sites))
    chosen = sites[perm[1:n]]

    a_axis = (L[1] / counts[1], L[2] / counts[2], L[3] / counts[3])

    cosT, sinT = cos(THETA_HOH), sin(THETA_HOH)
    for m in 1:n
        i, j, k = chosen[m]
        O = [((i + 0.5) * a_axis[1]) % L[1],
             ((j + 0.5) * a_axis[2]) % L[2],
             ((k + 0.5) * a_axis[3]) % L[3]]
        R = _random_rotation(build_rng)
        H1 = O .+ R * [p.r_OH, 0.0, 0.0]
        H2 = O .+ R * [p.r_OH * cosT, p.r_OH * sinT, 0.0]
        o, h1, h2 = 3m - 2, 3m - 1, 3m
        positions[:, o] = O
        positions[:, h1] = H1
        positions[:, h2] = H2
        molecules[m] = (o, h1, h2)
        masses[o], masses[h1], masses[h2] = p.m_O, p.m_H, p.m_H
        charges[o], charges[h1], charges[h2] = p.q_O, p.q_H, p.q_H
        epsilon[o], epsilon[h1], epsilon[h2] = p.eps_O, p.eps_H, p.eps_H
        sigma[o], sigma[h1], sigma[h2] = p.sigma_O, p.sigma_H, p.sigma_H
    end

    # O-O contact check. The lattice guarantees it in the default path; a cell
    # override or a degenerate fallback can violate it, so it is always checked.
    jitter_rng = MersenneTwister(hash((seed, n, 0x5eed)))
    min_seen = minimum_oo_distance(positions, molecules, L)
    attempts = 0
    while min_seen < min_oo
        attempts += 1
        attempts > 20 && throw(Types.InvalidState(
            "could not build a water box with min O-O $(min_seen) nm >= $(min_oo) nm " *
            "for N=$n, L=$(L) after 20 jitter attempts"))
        delta = 0.05 * min_oo .* randn(jitter_rng, 3, n)
        for m in 1:n
            o, h1, h2 = molecules[m]
            positions[:, o] .+= delta[:, m]
            positions[:, h1] .+= delta[:, m]
            positions[:, h2] .+= delta[:, m]
        end
        for d in 1:3
            positions[d, :] .-= L[d] .* floor.(positions[d, :] ./ L[d])
        end
        min_seen = minimum_oo_distance(positions, molecules, L)
    end

    triples = Tuple{Int,Int,Float64}[]
    excluded = Tuple{Int,Int}[]
    for (o, h1, h2) in molecules
        push!(triples, (o, h1, p.r_OH))
        push!(triples, (o, h2, p.r_OH))
        push!(triples, (h1, h2, R_HH))
        push!(excluded, (o, h1))
        push!(excluded, (o, h2))
        push!(excluded, (h1, h2))
    end

    cs = Constraints.CoupledConstraintSet(3n, triples)
    topo = StreamedLJ.PairTopology(; excluded = excluded)
    corrections = PME.PairCorrections(; excluded = excluded)
    bx = Types.OrthorhombicBox(L)
    rho = box_density(positions[1, :], n, L)   # exact by construction; reported
    return TIP3PWater(positions, masses, charges, epsilon, sigma, molecules, cs,
                      topo, corrections, bx, rho, Int(seed))
end

"Mass density (g/cm^3) of `n` waters in a cell of side `L` (nm)."
box_density(::Any, n::Integer, L) =
    n * M_WATER / (AVOGADRO * _volume(L) * 1.0e-21)

box_density(w::TIP3PWater) = box_density(w.positions, length(w.molecules), w.box.L)

# ---------------------------------------------------------------------------
# composed force field
# ---------------------------------------------------------------------------

"Default real-space cutoff for the Lennard-Jones factor: 0.9 nm, capped at 45% of the smallest side."
default_rcut(L) = min(0.9, 0.45 * minimum(L))

"""
Default PME real-space cutoff.

The only requirement is `0 < rcut < min(L)/2`; the `PME.PMEPlan` real-space
loop applies the minimum image per pair, so any number of cells per axis is
valid. The default is 0.9 nm, capped below `min(L)/2` for small boxes. The
Ewald split is chosen in `water_field` so that `alpha*rcut =
PME_ALPHA_TARGET` (`erfc(4.5) = 1.97e-10`), rather than holding alpha fixed
and letting the real-space discontinuity grow as the box shrinks.
"""
function default_pme_rcut(L; target::Real = 0.9)
    t = Float64(target)
    (isfinite(t) && t > 0) || throw(ArgumentError("PME cutoff target must be > 0"))
    return min(t, 0.49 * minimum(L))
end

"""
Target for the Ewald splitting parameter: `alpha * rcut = 4.5`.

At this value `erfc(4.5) = 1.97e-10`, so the real-space interaction is already
below 1e-9 of its value at the cutoff. That matters for dynamics, not just
accuracy: a hard truncation is a C⁰ energy discontinuity, and a pair crossing
the cutoff injects/loses a finite energy jump every time. The old fixed
`alpha = 3.5` gave `alpha*rcut = 1.60` in the 27-water box (`L = 0.93 nm`),
where `erfc(1.60)/rcut = 2.6e-2` — a ~1 kJ/mol jump per O-H crossing. Scaling
alpha with the cutoff keeps the jump negligible for every box size the module
admits.
"""
const PME_ALPHA_TARGET = 4.5

"Legacy grid spacing at `alpha = 3.5`; the actual spacing is `0.42/alpha`."
const PME_GRID_SPACING = 0.12

"""
Default PME grid: about `0.42/alpha` nm spacing (so the reciprocal Gaussian is
resolved to ~1e-6 at the Nyquist mode), a power of two, and at least `2*order`
points per axis. At `alpha = 3.5` this is the historical 0.12 nm spacing.
"""
function default_grid(L, order::Int = 6; alpha::Real = 3.5)
    spacing = 0.42 / Float64(alpha)
    return ntuple(d -> max(2 * order,
                           nextpow(2, ceil(Int, L[d] / spacing))), 3)
end

"""
Adapter that makes *any* inner factor accumulate.

`Factors.total_energy_forces` calls `evaluate!(Fbuf, f, q, ctx)` on each factor
and relies on every factor *adding* to `Fbuf`. `PME.evaluate!` is called through
`_pme_core!(...; accumulate=true)`, so the factor protocol adds; its direct
`pme_energy_forces!` API keeps the zeroing default for callers that pass a fresh
buffer. The wrapper below remains useful for older/third-party factors that may
still overwrite: it evaluates into private scratch and adds explicitly. `test_water_box.jl` checks the
composed force against central finite differences of the composed energy, which
fails loudly if this ever regresses.
"""
struct AccumulatingFactor <: Factors.AbstractFactor
    inner::Factors.AbstractFactor
    scratch::Matrix{Float64}
end

AccumulatingFactor(inner::Factors.AbstractFactor, N::Integer) =
    AccumulatingFactor(inner, zeros(Float64, 3, Int(N)))

Factors.factor_support_size(w::AccumulatingFactor) = Factors.factor_support_size(w.inner)
Factors.read_support(w::AccumulatingFactor) = Factors.read_support(w.inner)
Factors.is_local(w::AccumulatingFactor) = Factors.is_local(w.inner)
Factors.force_consistent(w::AccumulatingFactor) = Factors.force_consistent(w.inner)

function Factors.evaluate!(F::AbstractMatrix{Float64}, w::AccumulatingFactor,
                           q::AbstractMatrix{Float64}, ctx::Factors.Context)
    size(F) == (3, ctx.N) || throw(ArgumentError("F must be 3 x $(ctx.N)"))
    fill!(w.scratch, 0.0)
    U = Factors.evaluate!(w.scratch, w.inner, q, ctx)
    @inbounds for j in 1:ctx.N, d in 1:3
        F[d, j] += w.scratch[d, j]
    end
    return U
end

"""
The composed Hamiltonian: streamed periodic Lennard-Jones plus PME electrostatics
plus the rigid water constraints (applied by the runner, not by a factor).

`rcut` must satisfy `0 < rcut < min(L)/2` (the LJ cell list requires it).
`pme_rcut` has the same requirement; it has **no** `max(L)/3` lower bound —
the PME real-space loop applies the minimum image per pair, so 3 or more cells
per axis is valid. `alpha` controls the Ewald split; the default is
`PME_ALPHA_TARGET / pme_rcut` with `PME_ALPHA_TARGET = 4.5`, so
`erfc(alpha*pme_rcut) = 1.97e-10` and the hard real-space cutoff is an
negligible energy discontinuity. Passing `alpha` explicitly reproduces the old
fixed-split behavior (and the corresponding cutoff jumps) for diagnostics.
"""
struct WaterField
    lj::StreamedLJ.StreamedLJFactor
    pme::PME.PMEFactor
    ctx::Factors.Context
    system::Dynamics.HarmonicSystem
    rcut::Float64
    pme_rcut::Float64
    alpha::Float64
    grid::NTuple{3,Int}
    order::Int
end

function water_field(w::TIP3PWater; rcut::Union{Nothing,Real} = nothing,
                     pme_rcut::Union{Nothing,Real} = nothing,
                     alpha::Union{Nothing,Real} = nothing, grid = nothing,
                     order::Integer = 6,
                     prefactor::Real = COULOMB_PREFACTOR,
                     threaded::Bool = false,
                     neighbor_cache::Bool = false,
                     neighbor_cache_skin::Union{Nothing,Real} = nothing)
    L = w.box.L
    rc = rcut === nothing ? default_rcut(L) : Float64(rcut)
    (isfinite(rc) && rc > 0 && rc < 0.5 * minimum(L)) || throw(ArgumentError(
        "LJ rcut=$rc must satisfy 0 < rcut < min(L)/2 = $(0.5 * minimum(L)) nm"))
    rc_pme = pme_rcut === nothing ? default_pme_rcut(L) : Float64(pme_rcut)
    (isfinite(rc_pme) && rc_pme > 0 && rc_pme < 0.5 * minimum(L)) ||
        throw(ArgumentError(
            "PME rcut=$rc_pme must satisfy 0 < rcut < min(L)/2 = $(0.5 * minimum(L)) nm"))
    a = alpha === nothing ? PME_ALPHA_TARGET / rc_pme : Float64(alpha)
    isfinite(a) && a > 0 || throw(ArgumentError("alpha must be > 0"))
    g = grid === nothing ? default_grid(L, Int(order); alpha = a) :
        ntuple(d -> Int(grid[d]), 3)

    q = w.positions
    lj = StreamedLJ.StreamedLJFactor(q, w.epsilon, w.sigma, w.box;
                                     rcut = rc, skin = 0.0, mode = :lfs,
                                     topology = w.lj_topology, threaded = threaded,
                                     cache = neighbor_cache, cache_skin = neighbor_cache_skin)
    pme = PME.PMEFactor(q, w.charges, w.box; alpha = a, rcut = rc_pme,
                        grid = g, order = Int(order), prefactor = prefactor,
                        corrections = w.pme_corrections)
    ctx = Factors.Context(w.box, TIP3P_UNITS, size(q, 2))
    # Wrap both factors: see AccumulatingFactor. Without the LJ wrapper the LJ
    # force is silently dropped when it precedes PME in the factor list.
    wlj = AccumulatingFactor(lj, size(q, 2))
    wpme = AccumulatingFactor(pme, size(q, 2))
    sys = Dynamics.HarmonicSystem(Factors.AbstractFactor[wlj, wpme], w.masses, ctx)
    return WaterField(lj, pme, ctx, sys, rc, rc_pme, a, g, Int(order))
end

"Total potential energy (kJ/mol) of the composed field at `q`."
potential(field::WaterField, q::AbstractMatrix) =
    Dynamics.potential_energy(field.system, q)

"Fill `F` with `-dU/dq` at `q` and return `U` (kJ/mol). `q` is not modified."
forces(field::WaterField, q::AbstractMatrix, F::Matrix{Float64}) =
    Dynamics.forces!(field.system, q, F)

# ---------------------------------------------------------------------------
# trajectory state
# ---------------------------------------------------------------------------

"One measurement from a run."
struct Sample
    step::Int
    time_ps::Float64
    U::Float64        # potential energy, kJ/mol (pre-projection, see integrate!)
    K::Float64        # kinetic energy, kJ/mol (post-projection)
    E::Float64        # U + K
    T_inst::Float64   # instantaneous temperature from the kinetic DOF, K
end

"""
Rigid-water NVT trajectory: built box + composed field + dynamic state + RNG.

Momenta are drawn from Maxwell-Boltzmann at `temperature` and the centre-of-mass
momentum is removed, so the reported temperature uses `dof = 6N - 3`
(3N atoms minus 3 constraints per molecule, minus 3 for the fixed COM).
"""
mutable struct WaterTrajectory
    water::TIP3PWater
    field::WaterField
    st::Dynamics.State
    rng::Random.MersenneTwister
    noise::Matrix{Float64}      # scratch for the projected Langevin noise
    Ftan::Matrix{Float64}       # scratch: tangent-projected force
    dof::Int
    step::Int
    time_ps::Float64
    temperature::Float64
end

function WaterTrajectory(w::TIP3PWater, field::WaterField;
                         temperature::Real = 300.0, seed::Integer = 1)
    T = Float64(temperature)
    T >= 0 || throw(ArgumentError("temperature must be >= 0"))
    rng = MersenneTwister(seed)
    N = size(w.positions, 2)
    p = zeros(Float64, 3, N)
    if T > 0
        for j in 1:N, d in 1:3
            p[d, j] = sqrt(w.masses[j] * KB * T) * randn(rng)
        end
    end
    # Remove COM momentum so the kinetic temperature is not contaminated by a
    # uniform drift (which the constraint projection would preserve anyway).
    M = sum(w.masses)
    for d in 1:3
        vcm = sum(p[d, j] for j in 1:N) / M
        for j in 1:N
            p[d, j] -= w.masses[j] * vcm
        end
    end
    dof = 3N - 3 * length(w.molecules) - 3
    dof > 0 || throw(ArgumentError("not enough degrees of freedom"))
    st = Dynamics.State(copy(w.positions), p)
    # Start on the constraint manifold: the 3N momenta just drawn include the
    # 3N_mol bond-stretch components that the rigid geometry cannot support, so
    # remove them once here rather than letting the first step discover it.
    Constraints.project_velocities!(st.q, st.p, w.masses, w.constraints;
                                    tolerance = CONSTRAINT_TOL)
    return WaterTrajectory(w, field, st, rng, zeros(3, N), zeros(3, N),
                          dof, 0, 0.0, T)
end

WaterTrajectory(w::TIP3PWater; kwargs...) = WaterTrajectory(w, water_field(w); kwargs...)

"Kinetic temperature (K) from the constrained DOF."
temperature_of(tr::WaterTrajectory, p::AbstractMatrix) =
    2 * Dynamics.kinetic_energy(tr.field.system, p) / (tr.dof * KB)

"""
Exact deterministic checkpoint of a `WaterTrajectory` (dynamic state, RNG
state, step counter, simulation time, target temperature) for bit-identical replay.
"""
struct WaterCheckpoint
    state::Dynamics.State
    rng::Random.MersenneTwister
    step::Int
    time_ps::Float64
    temperature::Float64
end

function Dynamics.checkpoint(tr::WaterTrajectory)
    return WaterCheckpoint(copy(tr.st), copy(tr.rng), tr.step, tr.time_ps, tr.temperature)
end

function Dynamics.restore!(tr::WaterTrajectory, cp::WaterCheckpoint)
    Dynamics.restore!(tr.st, Dynamics.DynamicsCheckpoint(cp.state, cp.step, cp.time_ps, nothing))
    copy!(tr.rng, cp.rng)
    tr.step = cp.step
    tr.time_ps = cp.time_ps
    tr.temperature = cp.temperature
    return tr
end

"""
    water_residuals(tr::WaterTrajectory) -> (pos_residual, tan_residual, max_bond_error)

Measure the current maximum squared-distance constraint residual, tangent-velocity
residual, and linear bond-length error across all rigid TIP3P molecules.
"""
function water_residuals(tr::WaterTrajectory)
    q = tr.st.q
    p = tr.st.p
    m = tr.water.masses
    cs = tr.water.constraints
    pos_res = maximum(abs, Constraints.constraint_residuals(cs, q))
    tan_res = maximum(abs, Constraints.tangent_residuals(cs, q, p, m))
    max_bond = 0.0
    r_oh = TIP3P_PARAMETERS.r_OH
    @inbounds for (o, h1, h2) in tr.water.molecules
        d1 = sqrt((q[1, h1] - q[1, o])^2 + (q[2, h1] - q[2, o])^2 + (q[3, h1] - q[3, o])^2)
        d2 = sqrt((q[1, h2] - q[1, o])^2 + (q[2, h2] - q[2, o])^2 + (q[3, h2] - q[3, o])^2)
        d3 = sqrt((q[1, h2] - q[1, h1])^2 + (q[2, h2] - q[2, h1])^2 + (q[3, h2] - q[3, h1])^2)
        max_bond = max(max_bond, abs(d1 - r_oh), abs(d2 - r_oh), abs(d3 - R_HH))
    end
    return (pos_residual = pos_res, tan_residual = tan_res, max_bond_error = max_bond)
end

"""
    water_integrator_spec(name::Symbol) -> Dynamics.IntegratorSpec

Return the explicit stage machine specification and reversible/stochastic
classification for rigid-water integrators (`:rigid_water_nve`, `:rigid_water_baoab`).
"""
function water_integrator_spec(name::Symbol)
    if name === :rigid_water_nve
        return Dynamics.IntegratorSpec(:rigid_water_nve, [
            Dynamics.IntegratorStage(:half_kick, "p ← p + (dt/2) P F"),
            Dynamics.IntegratorStage(:rigid_body_drift, "symmetric principal-axis rigid-body drift (dt)"),
            Dynamics.IntegratorStage(:force_evaluation, "composed LJ+PME tangent-projected force evaluation"),
            Dynamics.IntegratorStage(:half_kick, "p ← p + (dt/2) P F"),
            Dynamics.IntegratorStage(:velocity_constraints, "mass-metric tangent velocity projection"),
            Dynamics.IntegratorStage(:diagnostics, "optional energy/residual/momentum sampling"),
        ])
    elseif name === :rigid_water_baoab
        return Dynamics.IntegratorSpec(:rigid_water_baoab, [
            Dynamics.IntegratorStage(:half_kick, "B: p ← p + (dt/2) P F"),
            Dynamics.IntegratorStage(:velocity_constraints, "mass-metric tangent velocity projection"),
            Dynamics.IntegratorStage(:rigid_body_drift, "A: symmetric principal-axis half drift (dt/2)"),
            Dynamics.IntegratorStage(:thermostat_ou, "O: exact OU update with COM+tangent-projected noise"),
            Dynamics.IntegratorStage(:rigid_body_drift, "A: symmetric principal-axis half drift (dt/2)"),
            Dynamics.IntegratorStage(:force_evaluation, "composed LJ+PME tangent-projected force evaluation"),
            Dynamics.IntegratorStage(:half_kick, "B: p ← p + (dt/2) P F"),
            Dynamics.IntegratorStage(:velocity_constraints, "final mass-metric tangent velocity projection"),
            Dynamics.IntegratorStage(:diagnostics, "optional energy/residual/temperature sampling"),
        ])
    else
        throw(ArgumentError("unknown rigid-water integrator :$name; expected :rigid_water_nve or :rigid_water_baoab"))
    end
end

# ---------------------------------------------------------------------------
# relaxation
# ---------------------------------------------------------------------------

"Outcome of the contact-clearing relaxation."
struct RelaxReport
    steps::Int
    steps_accepted::Int
    u_initial::Float64
    u_final::Float64
    max_force_initial::Float64
    max_force_final::Float64
    max_tangent_force_final::Float64
    step_size::Float64
    converged::Bool
end

"""
Project the force onto the constraint tangent space, per molecule.

For a constraint component with constraints `c_k(q) = |q_j - q_i|^2 - d_k^2` and
Jacobian `G = dc/dq`, the component of the force that the rigid geometry cannot
support is removed by solving the same small Lagrange system that
`Constraints.project_velocities!` solves for velocities:

    (G W G^T) lambda = G W F,     F_tangent = F - G^T lambda,     W = diag(1/m)

Stepping along the *raw* force and then projecting positions is not equivalent,
and the difference is not small: measured on a 216-water box, a raw steepest
descent step lowered the energy by 143 kJ/mol and the subsequent position
projection raised it by 85 kJ/mol, because the raw step's constraint-violating
component dominates. Projecting the force first makes every step a genuine
descent direction on the constraint manifold.
"""
function tangent_force!(Fout::AbstractMatrix{Float64}, F::AbstractMatrix{Float64},
                        q::AbstractMatrix{Float64}, m::AbstractVector{Float64},
                        cs::Constraints.CoupledConstraintSet)
    copyto!(Fout, F)
    for comp in cs.components
        na = length(comp.atoms)
        nc = length(comp.edges)
        G = zeros(Float64, nc, 3na)
        W = zeros(Float64, 3na)
        for (a, atom) in enumerate(comp.atoms), d in 1:3
            W[3(a - 1) + d] = 1.0 / m[atom]
        end
        @inbounds for (k, (a, b, _)) in enumerate(comp.edges)
            i = comp.atoms[a]
            j = comp.atoms[b]
            for d in 1:3
                r = q[d, j] - q[d, i]
                G[k, 3(a - 1) + d] = -2r
                G[k, 3(b - 1) + d] = 2r
            end
        end
        floc = zeros(Float64, 3na)
        @inbounds for (a, atom) in enumerate(comp.atoms), d in 1:3
            floc[3(a - 1) + d] = F[d, atom]
        end
        A = G * LinearAlgebra.Diagonal(W) * transpose(G)
        lam = A \ (G * (W .* floc))
        fproj = floc - transpose(G) * lam
        @inbounds for (a, atom) in enumerate(comp.atoms), d in 1:3
            Fout[d, atom] = fproj[3(a - 1) + d]
        end
    end
    return Fout
end

"""
    relax!(tr; max_steps=600, step0=5e-4, max_step=5e-3, ...)

Clear the build-time contacts by steepest descent *on the constraint manifold*
(the direction is `tangent_force!`, so no step fights the rigid geometry).

A random-orientation lattice sits thousands of kJ/mol above the hydrogen-bond
network it is about to form. Running dynamics on it directly is not safe: the
release heats the box far above the target and one bad H-O contact (TIP3P has no
Lennard-Jones site on H) ends the trajectory. Descent is therefore done first,
with a trust region: the step grows 1.2x on an accepted (energy-decreasing) step
and halves on a rejected one, and every trial is re-projected onto the
constraints. This is a *preparation* step, not an equilibration claim.
"""
function relax!(tr::WaterTrajectory; max_steps::Integer = 3000, step0::Real = 5e-4,
                max_step::Real = 5e-3, min_step::Real = 1e-7,
                plateau::Real = 2e-5, plateau_window::Integer = 60,
                tangent_force_tol::Real = 10.0)
    w = tr.water
    q = tr.st.q
    F = tr.st.F
    Fp = similar(F)
    qtry = similar(q)
    Ftry = similar(F)
    U = forces(tr.field, q, F)
    tr.st.force_valid = true
    fmax0 = maximum(abs, F)
    U0 = U
    step = Float64(step0)
    accepted = 0
    converged = false
    u_hist = Float64[U]
    for it in 1:Int(max_steps)
        tangent_force!(Fp, F, q, w.masses, w.constraints)
        fp = maximum(abs, Fp)
        if fp <= Float64(tangent_force_tol) || step < Float64(min_step)
            converged = true
            break
        end
        # The descent direction is the *mass-weighted* projected force. Only
        # W*Fp is guaranteed downhill: Fp is W-orthogonal to the constraint
        # gradients (G W Fp = 0), so dU = -F.dq = -(step/fp) Fp'W Fp <= 0,
        # whereas the Euclidean F.Fp can be negative once the constraint part
        # of F is removed. Using Fp directly stalls the line search.
        vmax = 0.0
        @inbounds for j in axes(q, 2), d in 1:3
            v = Fp[d, j] / w.masses[j]
            abs(v) > vmax && (vmax = abs(v))
        end
        @inbounds for j in axes(q, 2), d in 1:3
            qtry[d, j] = q[d, j] + step * (Fp[d, j] / w.masses[j]) / vmax
        end
        Constraints.project_positions!(qtry, w.masses, w.constraints;
                                       tolerance = CONSTRAINT_TOL)
        Utry = forces(tr.field, qtry, Ftry)
        if isfinite(Utry) && Utry < U
            q .= qtry
            F .= Ftry
            U = Utry
            accepted += 1
            step = min(step * 1.2, Float64(max_step))
            push!(u_hist, U)
            if length(u_hist) > plateau_window
                prev = u_hist[end - plateau_window]
                if abs(U - prev) <= Float64(plateau) * max(abs(U), 1.0)
                    converged = true
                    break
                end
            end
        else
            step *= 0.5
        end
    end
    # A minimised configuration has no kinetic energy; equilibration re-draws it.
    fill!(tr.st.p, 0.0)
    tangent_force!(F, F, q, w.masses, w.constraints)   # force now tangent at q
    tr.st.force_valid = true
    return RelaxReport(length(u_hist) - 1, accepted, U0, U, fmax0, maximum(abs, F),
                       maximum(abs, F), step, converged)
end

# ---------------------------------------------------------------------------
# integration
# ---------------------------------------------------------------------------

"""
Evaluate the composed force and write its constraint-tangent projection back
into `F`.

At fixed `q`, let `P` be the mass-metric velocity projector onto the constraint
tangent space. For a tangent momentum `p` (`P p = p`), a half kick followed by
the projection is

    P (p + dt F/2) = p + dt (P F)/2.

So kicking with `P F` and then projecting gives the same tangent state as
kicking with the full `F` and projecting; projecting the force first is a
numerical convenience that avoids carrying a large constraint-normal component
through the kick, not a different force law.

The current composed field excludes the intramolecular O-H/H-H nonbonded pairs
(`PairTopology` and `PME.PairCorrections`), so there is no multi-10^3 kJ/mol/nm
intra-bond point force in the present model. Historical intermediate builds
without those exclusions measured ~5e3 kJ/mol/nm at 0.0957 nm, 0.78 kJ/mol per
step, and 300 K -> 144 K over 20 unthermostatted steps. Those numbers explain
why the projection is applied before the kick; they are not a property of the
current field and are not a current kinetic-energy-sink claim.
"""
function _tangent_forces!(F::Matrix{Float64}, Ftan::Matrix{Float64},
                          q::Matrix{Float64}, m::Vector{Float64},
                          cs::Constraints.CoupledConstraintSet, field::WaterField)
    U = forces(field, q, F)
    tangent_force!(Ftan, F, q, m, cs)
    copyto!(F, Ftan)
    return U
end

"""
    _axis_rotation(axis, θ) -> 3×3 Matrix

Right-handed rotation matrix about body axis `axis ∈ (1,2,3)` by `θ` radians.
"""
function _axis_rotation(axis::Int, θ::Float64)
    c, s = cos(θ), sin(θ)
    if axis == 1
        return [1.0 0.0 0.0; 0.0 c -s; 0.0 s c]
    elseif axis == 2
        return [c 0.0 s; 0.0 1.0 0.0; -s 0.0 c]
    else
        return [c -s 0.0; s c 0.0; 0.0 0.0 1.0]
    end
end

"""
    _body_frame(v1, v2) -> 3×3 Matrix

Right-handed frame with columns `(e1,e2,e3)`, `e1 ∥ v1`, `e2` in the
`(v1,v2)` plane, and `e3 = e1 × e2`. `v1` and `v2` must be non-collinear.
"""
function _body_frame(v1::NTuple{3,Float64}, v2::NTuple{3,Float64})
    n1 = sqrt(v1[1]^2 + v1[2]^2 + v1[3]^2)
    e1x, e1y, e1z = v1[1] / n1, v1[2] / n1, v1[3] / n1
    d = v2[1] * e1x + v2[2] * e1y + v2[3] * e1z
    wx, wy, wz = v2[1] - d * e1x, v2[2] - d * e1y, v2[3] - d * e1z
    n2 = sqrt(wx^2 + wy^2 + wz^2)
    e2x, e2y, e2z = wx / n2, wy / n2, wz / n2
    e3x = e1y * e2z - e1z * e2y
    e3y = e1z * e2x - e1x * e2z
    e3z = e1x * e2y - e1y * e2x
    return [e1x e2x e3x; e1y e2y e3y; e1z e2z e3z]
end

"""
Principal-axis body template shared by every rigid water molecule: `Iv` are
the principal moments, `Rb` the body coordinates in the principal frame, and
`P` the rotation from the two-vector body frame to the principal frame.
"""
struct RigidTemplate
    Iv::NTuple{3,Float64}
    Rb::Matrix{Float64}
    P::Matrix{Float64}
end

function _rigid_template(q::Matrix{Float64}, m::Vector{Float64}, molecules)
    o, h1, h2 = molecules[1]
    M = m[o] + m[h1] + m[h2]
    x1 = (m[o] * q[1, o] + m[h1] * q[1, h1] + m[h2] * q[1, h2]) / M
    x2 = (m[o] * q[2, o] + m[h1] * q[2, h1] + m[h2] * q[2, h2]) / M
    x3 = (m[o] * q[3, o] + m[h1] * q[3, h1] + m[h2] * q[3, h2]) / M
    R = zeros(3, 3)
    for (a, i) in enumerate((o, h1, h2))
        R[1, a] = q[1, i] - x1
        R[2, a] = q[2, i] - x2
        R[3, a] = q[3, i] - x3
    end
    Bt = _body_frame((R[1, 2] - R[1, 1], R[2, 2] - R[2, 1], R[3, 2] - R[3, 1]),
                     (R[1, 3] - R[1, 1], R[2, 3] - R[2, 1], R[3, 3] - R[3, 1]))
    b = Bt' * R
    It = zeros(3, 3)
    for (a, i) in enumerate((o, h1, h2))
        rx, ry, rz = R[1, a], R[2, a], R[3, a]
        mi = m[i]
        It[1, 1] += mi * (ry^2 + rz^2)
        It[2, 2] += mi * (rx^2 + rz^2)
        It[3, 3] += mi * (rx^2 + ry^2)
        It[1, 2] -= mi * rx * ry
        It[1, 3] -= mi * rx * rz
        It[2, 3] -= mi * ry * rz
    end
    It[2, 1] = It[1, 2]
    It[3, 1] = It[1, 3]
    It[3, 2] = It[2, 3]
    IB = Bt' * It * Bt
    E = eigen(Symmetric(IB))
    P = E.vectors
    if det(P) < 0
        P[:, 3] = -P[:, 3]
    end
    Rb = P' * b
    Iv = (E.values[1], E.values[2], E.values[3])
    return RigidTemplate(Iv, Rb, P)
end

"""
    _drift!(q, p, m, molecules, dt)

Reversible rigid-body drift by a symmetric principal-axis Hamiltonian
splitting.

For each molecule the inertia tensor is diagonalised in a body frame built from
the two O–H vectors.  In the principal frame the free Hamiltonian splits into
`H = L₁²/(2I₁) + L₂²/(2I₂) + L₃²/(2I₃)`.  The exact flow of each term rotates
the body and the body angular momentum about one principal axis; composing them
as `H₁(dt/2) H₂(dt/2) H₃(dt) H₂(dt/2) H₁(dt/2)` is a symmetric, second-order
splitting.  Each subflow preserves the *laboratory* angular momentum
`L_lab = Q L_body` exactly because `Q → Q R` and `L_body → Rᵀ L_body`; the
composition therefore also preserves `L_lab`, as required for a torque-free
asymmetric top.  The map is reversible: reversing the order with `-dt` is the
exact inverse.

The previous implementation rotated positions and momenta together by
`exp(ω dt)`.  That conserves kinetic energy but rotates `L_lab` by the same
matrix, which is wrong for an asymmetric top and fails a lab-angular-momentum
conservation test.  The earlier SHAKE half-drift was worse: it removed the part
of the momentum outside the rotated tangent plane, a kinetic-energy sink that
made the stationary Langevin temperature depend on friction even with zero
forces.
"""
function _drift!(q::Matrix{Float64}, p::Matrix{Float64}, m::Vector{Float64},
                 molecules, dt::Float64)
    tpl = _rigid_template(q, m, molecules)
    Iv, Rb, P = tpl.Iv, tpl.Rb, tpl.P
    @inbounds for (o, h1, h2) in molecules
        idx = (o, h1, h2)
        M = m[o] + m[h1] + m[h2]
        x1 = (m[o] * q[1, o] + m[h1] * q[1, h1] + m[h2] * q[1, h2]) / M
        x2 = (m[o] * q[2, o] + m[h1] * q[2, h1] + m[h2] * q[2, h2]) / M
        x3 = (m[o] * q[3, o] + m[h1] * q[3, h1] + m[h2] * q[3, h2]) / M
        v1 = (p[1, o] + p[1, h1] + p[1, h2]) / M
        v2 = (p[2, o] + p[2, h1] + p[2, h2]) / M
        v3 = (p[3, o] + p[3, h1] + p[3, h2]) / M
        R = zeros(3, 3)
        Pr = zeros(3, 3)
        for (a, i) in enumerate(idx)
            R[1, a] = q[1, i] - x1
            R[2, a] = q[2, i] - x2
            R[3, a] = q[3, i] - x3
            Pr[1, a] = p[1, i] - m[i] * v1
            Pr[2, a] = p[2, i] - m[i] * v2
            Pr[3, a] = p[3, i] - m[i] * v3
        end
        Lt = _body_frame((R[1, 2] - R[1, 1], R[2, 2] - R[2, 1], R[3, 2] - R[3, 1]),
                         (R[1, 3] - R[1, 1], R[2, 3] - R[2, 1], R[3, 3] - R[3, 1]))
        Q = Lt * P
        Lx = R[2, 1] * Pr[3, 1] - R[3, 1] * Pr[2, 1] +
             R[2, 2] * Pr[3, 2] - R[3, 2] * Pr[2, 2] +
             R[2, 3] * Pr[3, 3] - R[3, 3] * Pr[2, 3]
        Ly = R[3, 1] * Pr[1, 1] - R[1, 1] * Pr[3, 1] +
             R[3, 2] * Pr[1, 2] - R[1, 2] * Pr[3, 2] +
             R[3, 3] * Pr[1, 3] - R[1, 3] * Pr[3, 3]
        Lz = R[1, 1] * Pr[2, 1] - R[2, 1] * Pr[1, 1] +
             R[1, 2] * Pr[2, 2] - R[2, 2] * Pr[1, 2] +
             R[1, 3] * Pr[2, 3] - R[2, 3] * Pr[1, 3]
        Lb = Q' * [Lx, Ly, Lz]
        for (axis, h) in ((1, 0.5 * dt), (2, 0.5 * dt), (3, dt),
                          (2, 0.5 * dt), (1, 0.5 * dt))
            θ = Lb[axis] / Iv[axis] * h
            Q = Q * _axis_rotation(axis, θ)
            Lb = _axis_rotation(axis, -θ) * Lb
        end
        ω1 = Lb[1] / Iv[1]
        ω2 = Lb[2] / Iv[2]
        ω3 = Lb[3] / Iv[3]
        for a in 1:3
            i = idx[a]
            rb1, rb2, rb3 = Rb[1, a], Rb[2, a], Rb[3, a]
            q[1, i] = x1 + v1 * dt + Q[1, 1] * rb1 + Q[1, 2] * rb2 + Q[1, 3] * rb3
            q[2, i] = x2 + v2 * dt + Q[2, 1] * rb1 + Q[2, 2] * rb2 + Q[2, 3] * rb3
            q[3, i] = x3 + v3 * dt + Q[3, 1] * rb1 + Q[3, 2] * rb2 + Q[3, 3] * rb3
            c1 = ω2 * rb3 - ω3 * rb2
            c2 = ω3 * rb1 - ω1 * rb3
            c3 = ω1 * rb2 - ω2 * rb1
            p[1, i] = m[i] * (v1 + Q[1, 1] * c1 + Q[1, 2] * c2 + Q[1, 3] * c3)
            p[2, i] = m[i] * (v2 + Q[2, 1] * c1 + Q[2, 2] * c2 + Q[2, 3] * c3)
            p[3, i] = m[i] * (v3 + Q[3, 1] * c1 + Q[3, 2] * c2 + Q[3, 3] * c3)
        end
    end
    return nothing
end

"""
One BAOAB step: half kick (tangent-projected), reversible principal-axis half drift,
exact Ornstein-Uhlenbeck update with *tangent-projected* noise, reversible
principal-axis half drift, force, half kick.

Two points about the constrained composition:

* the O-step noise is projected onto the constraint tangent space in the mass
  metric before it is added. For a tangent OU process, projecting the noise (or
  post-projecting the updated momentum) is the correct constrained FDT update
  and does not by itself bias equipartition; the former gamma-dependent
  temperature came from the dissipative drift, not from the projection order.
* the drift must be the reversible principal-axis splitting, not a linear drift
  followed by a SHAKE projection. The latter removed the part of the momentum
  pointing out of the rotated tangent plane, a real kinetic-energy sink at 2 fs
  (see [`_drift!`](@ref)); it made the stationary temperature gamma-dependent
  even with all forces switched off.
"""
@inline _emit_water_stage(hook, stage::Symbol) = (hook === nothing || hook(stage); nothing)

function _baoab_step!(tr::WaterTrajectory, dt::Float64, gamma::Float64,
                      beta::Float64; stage_hook = nothing,
                      constraint_solver::Symbol = :principal_axis,
                      q_ref::Union{Nothing,Matrix{Float64}} = nothing)
    constraint_solver in (:principal_axis, :settle) ||
        throw(ArgumentError("unsupported constraint_solver: $constraint_solver (expected :principal_axis or :settle)"))
    q = tr.st.q
    p = tr.st.p
    F = tr.st.F
    m = tr.water.masses
    cs = tr.water.constraints
    box = tr.field.system.ctx.box
    n = size(q, 2)
    ξ = tr.noise
    if !tr.st.force_valid
        _emit_water_stage(stage_hook, :force_evaluation)
        _tangent_forces!(F, tr.Ftan, q, m, cs, tr.field)
        tr.st.force_valid = true
    end
    # B: half kick + velocity projection
    _emit_water_stage(stage_hook, :half_kick)
    @inbounds for j in 1:n, d in 1:3
        p[d, j] += 0.5 * dt * F[d, j]
    end
    _emit_water_stage(stage_hook, :velocity_constraints)
    if constraint_solver === :settle
        Settle.project_momenta!(q, p, m, cs; box = box)
    else
        Constraints.project_velocities!(q, p, m, cs; tolerance = CONSTRAINT_TOL)
    end
    # A: half drift
    _emit_water_stage(stage_hook, :rigid_body_drift)
    if constraint_solver === :settle
        qold = q_ref === nothing ? copy(q) : copyto!(q_ref, q)
        @inbounds for j in 1:n, d in 1:3
            q[d, j] += 0.5 * dt * (p[d, j] / m[j])
        end
        Settle.settle_step!(qold, q, p, m, cs, 0.5 * dt; box = box)
        Settle.project_momenta!(q, p, m, cs; box = box)
    else
        _drift!(q, p, m, tr.water.molecules, 0.5 * dt)
    end
    # O: exact OU update with tangent-projected noise (stochastic stage)
    _emit_water_stage(stage_hook, :thermostat_ou)
    @inbounds for j in 1:n, d in 1:3
        ξ[d, j] = sqrt(m[j]) * randn(tr.rng)
    end
    # The trajectory ensemble fixes total COM momentum to zero; project the OU
    # noise into that subspace as well as the rigid-constraint tangent space.
    total_mass = sum(m)
    for d in 1:3
        vcm = sum(ξ[d, j] for j in 1:n) / total_mass
        for j in 1:n
            ξ[d, j] -= m[j] * vcm
        end
    end
    if constraint_solver === :settle
        Settle.project_momenta!(q, ξ, m, cs; box = box)
    else
        Constraints.project_velocities!(q, ξ, m, cs; tolerance = CONSTRAINT_TOL)
    end
    c1 = exp(-gamma * dt)
    c2 = sqrt((1.0 - c1 * c1) / beta)
    @inbounds for j in 1:n, d in 1:3
        p[d, j] = c1 * p[d, j] + c2 * ξ[d, j]
    end
    # A: half drift
    _emit_water_stage(stage_hook, :rigid_body_drift)
    if constraint_solver === :settle
        qold = q_ref === nothing ? copy(q) : copyto!(q_ref, q)
        @inbounds for j in 1:n, d in 1:3
            q[d, j] += 0.5 * dt * (p[d, j] / m[j])
        end
        Settle.settle_step!(qold, q, p, m, cs, 0.5 * dt; box = box)
        Settle.project_momenta!(q, p, m, cs; box = box)
    else
        _drift!(q, p, m, tr.water.molecules, 0.5 * dt)
    end
    # force at the new position (tangent-projected)
    _emit_water_stage(stage_hook, :force_evaluation)
    U = _tangent_forces!(F, tr.Ftan, q, m, cs, tr.field)
    # B: half kick + velocity projection
    _emit_water_stage(stage_hook, :half_kick)
    @inbounds for j in 1:n, d in 1:3
        p[d, j] += 0.5 * dt * F[d, j]
    end
    _emit_water_stage(stage_hook, :velocity_constraints)
    if constraint_solver === :settle
        Settle.project_momenta!(q, p, m, cs; box = box)
    else
        Constraints.project_velocities!(q, p, m, cs; tolerance = CONSTRAINT_TOL)
    end
    tr.st.force_valid = true
    return U
end

"""
    integrate!(tr, nsteps; dt, temperature, gamma, sample_every, on_sample, constraint_solver=:principal_axis)

Run `nsteps` BAOAB Langevin steps at fixed `dt` (ps), projecting positions and
momenta onto the rigid-water manifold after every step.

Returns the `Vector{Sample}` collected every `sample_every` steps. The reported
potential energy is the value at the end of the BAOAB step, before the (tiny,
`O(1e-11)` nm) constraint projection; kinetic energy is measured after the
projection, so the total is consistent to roundoff.
"""
function integrate!(tr::WaterTrajectory, nsteps::Integer;
                    dt::Real = 0.002, temperature::Real = tr.temperature,
                    gamma::Real = 1.0, sample_every::Integer = 10,
                    on_sample = nothing,
                    constraint_solver::Symbol = :principal_axis)
    constraint_solver in (:principal_axis, :settle) ||
        throw(ArgumentError("unsupported constraint_solver: $constraint_solver (expected :principal_axis or :settle)"))
    nsteps >= 0 || throw(ArgumentError("nsteps must be >= 0"))
    sample_every >= 1 || throw(ArgumentError("sample_every must be >= 1"))
    dtf = Float64(dt)
    (isfinite(dtf) && dtf > 0) || throw(ArgumentError("dt must be positive"))
    beta = 1.0 / (KB * Float64(temperature))
    samples = Sample[]
    q_ref = constraint_solver === :settle ? similar(tr.st.q) : nothing
    for s in 1:nsteps
        U = _baoab_step!(tr, dtf, Float64(gamma), beta;
                         constraint_solver = constraint_solver, q_ref = q_ref)
        tr.step += 1
        tr.time_ps += dtf
        if s % sample_every == 0
            K = Dynamics.kinetic_energy(tr.field.system, tr.st.p)
            push!(samples, Sample(tr.step, tr.time_ps, U, K, U + K,
                                  temperature_of(tr, tr.st.p)))
            on_sample !== nothing && on_sample(tr, samples[end])
        end
    end
    return samples
end

"""
Re-draw Maxwell-Boltzmann momenta at `temperature` and project them onto the
constraint manifold. Called at the start of equilibration so the thermostat does
not have to heat a minimised (zero-momentum) state.
"""
function rethermalize!(tr::WaterTrajectory; temperature::Real = tr.temperature)
    T = Float64(temperature)
    w = tr.water
    p = tr.st.p
    N = size(p, 2)
    for j in 1:N, d in 1:3
        p[d, j] = sqrt(w.masses[j] * KB * T) * randn(tr.rng)
    end
    M = sum(w.masses)
    for d in 1:3
        vcm = sum(p[d, j] for j in 1:N) / M
        for j in 1:N
            p[d, j] -= w.masses[j] * vcm
        end
    end
    Constraints.project_velocities!(tr.st.q, p, w.masses, w.constraints;
                                    tolerance = CONSTRAINT_TOL)
    return tr
end

"Equilibration outcome: the temperature/energy series and wall time."
struct EquilibrationResult
    samples::Vector{Sample}
    steps::Int
    ps::Float64
    wall_s::Float64
    temp_mean::Float64
    temp_se::Float64
    temp_iat::Float64
end

"""
    equilibrate!(tr; ps=20.0, dt=0.002, temperature=300, gamma=5.0, ...)

BAOAB Langevin equilibration. This is *not* a claim of equilibration in the
statistical-mechanics sense; it is the thermostat settling into its stationary
window, reported with the measured temperature and its autocorrelation-corrected
standard error so the reader can judge.

The default friction is `5/ps` rather than the production `1/ps`: the structure
is still descending towards its hydrogen-bond network when relaxation ends, and
a weak thermostat lets that release overheat the box. Production uses `1/ps`.
"""
function equilibrate!(tr::WaterTrajectory; ps::Real = 20.0, dt::Real = 0.002,
                      temperature::Real = tr.temperature, gamma::Real = 5.0,
                      sample_every::Integer = 10)
    nsteps = max(1, round(Int, Float64(ps) / Float64(dt)))
    tr.temperature = Float64(temperature)
    rethermalize!(tr; temperature = temperature)
    t0 = time()
    samples = integrate!(tr, nsteps; dt = dt, temperature = temperature,
                         gamma = gamma, sample_every = sample_every)
    wall = time() - t0
    temps = [s.T_inst for s in samples]
    tm, tse, tiat = mean_with_error(temps)
    return EquilibrationResult(samples, nsteps, nsteps * Float64(dt), wall, tm, tse, tiat)
end

"Production outcome: sample series, MSD/RDF curves, diffusion estimate, wall time."
struct ProductionResult
    samples::Vector{Sample}
    steps::Int
    ps::Float64
    dt_ps::Float64
    wall_s::Float64
    msd_time_ps::Vector{Float64}
    msd_nm2::Vector{Float64}
    rdf_r::Vector{Float64}
    rdf_g::Vector{Float64}
    rdf_frames::Int
    diffusion::Float64        # nm^2/ps
    diffusion_se::Float64
    msd_fit_window_ps::NTuple{2,Float64}
end

"""
    production!(tr; ps, dt, temperature, gamma, sample_every, frame_writer,
                frame_every, rdf_frames, rdf_every, rmax, nbins)

Production window. Writes wrapped frames (each molecule kept whole: the O is
wrapped into the cell and the H atoms follow it) through `frame_writer` every
`frame_every` steps, accumulates the O-O radial distribution function over up to
`rdf_frames` frames, and tracks the molecular centre-of-mass mean squared
displacement from the *unwrapped* coordinates (which the dynamics keeps exact,
because positions are never wrapped mid-run).
"""
function production!(tr::WaterTrajectory; ps::Real = 100.0, dt::Real = 0.002,
                     temperature::Real = tr.temperature, gamma::Real = 1.0,
                     sample_every::Integer = 10,
                     frame_writer::Union{Nothing,Dynamics.FrameWriter} = nothing,
                     frame_every::Integer = 500,
                     rdf_frames::Integer = 40, rdf_every::Integer = 5,
                     rmax::Real = 1.2, nbins::Integer = 300,
                     progress_every::Integer = 0,
                     com_io::Union{Nothing,IO} = nothing)
    nsteps = max(1, round(Int, Float64(ps) / Float64(dt)))
    frame_every >= 1 || throw(ArgumentError("frame_every must be >= 1"))
    sample_every >= 1 || throw(ArgumentError("sample_every must be >= 1"))
    tr.temperature = Float64(temperature)

    w = tr.water
    m = w.masses
    n = length(w.molecules)
    com = Matrix{Float64}(undef, 3, n)
    com0 = Matrix{Float64}(undef, 3, n)
    filled0 = false
    msd_t = Float64[]
    msd_v = Float64[]
    counts = zeros(Int, Int(nbins))
    rdf_done = 0
    nframes = 0
    samples = Sample[]
    dr = Float64(rmax) / Int(nbins)

    function record_com!(q)
        @inbounds for i in 1:n
            o, h1, h2 = w.molecules[i]
            M = m[o] + m[h1] + m[h2]
            for d in 1:3
                com[d, i] = (m[o] * q[d, o] + m[h1] * q[d, h1] +
                             m[h2] * q[d, h2]) / M
            end
        end
    end

    t0 = time()
    beta = 1.0 / (KB * Float64(temperature))
    cs = w.constraints
    for s in 1:nsteps
        U = _baoab_step!(tr, Float64(dt), Float64(gamma), beta)
        tr.step += 1
        tr.time_ps += Float64(dt)
        if s % sample_every == 0
            K = Dynamics.kinetic_energy(tr.field.system, tr.st.p)
            push!(samples, Sample(tr.step, tr.time_ps, U, K, U + K,
                                  temperature_of(tr, tr.st.p)))
        end
        if s % frame_every == 0
            nframes += 1
            if frame_writer !== nothing
                Dynamics.write_frame!(frame_writer, molecule_frame(tr.st.q, w),
                                      w.box, tr.step, tr.time_ps)
            end
            record_com!(tr.st.q)
            if !filled0
                com0 .= com
                filled0 = true
            end
            msd = 0.0
            @inbounds for i in 1:n
                msd += (com[1, i] - com0[1, i])^2 + (com[2, i] - com0[2, i])^2 +
                       (com[3, i] - com0[3, i])^2
            end
            push!(msd_t, tr.time_ps)
            push!(msd_v, msd / n)
            # Exact unwrapped centre-of-mass stream: the dynamics never wraps
            # positions, so this is the true displacement even when the frame
            # spacing is far too coarse to unwrap a wrapped XYZ trajectory
            # (a molecule moves further than L/2 per frame at 1 ps).
            if com_io !== nothing
                for i in 1:n
                    @printf(com_io, "%.6f,%d,%.10f,%.10f,%.10f\n", tr.time_ps, i,
                            com[1, i], com[2, i], com[3, i])
                end
            end
            if rdf_done < rdf_frames && (nframes % max(1, Int(rdf_every)) == 0)
                _accumulate_oo_rdf!(counts, tr.st.q, w, Float64(rmax), dr)
                rdf_done += 1
            end
        end
        if progress_every > 0 && s % progress_every == 0
            K = Dynamics.kinetic_energy(tr.field.system, tr.st.p)
            @printf("  [production] step %d/%d  t=%.3f ps  T=%.1f K\n", s, nsteps,
                    tr.time_ps, 2K / (tr.dof * KB))
        end
    end
    wall = time() - t0

    r = _rdf_axis(Float64(rmax), Int(nbins))
    g = _rdf_g(counts, r, n, _volume(w.box.L), max(rdf_done, 1))
    D, Dse, win = diffusion_from_msd(msd_t, msd_v)
    return ProductionResult(samples, nsteps, nsteps * Float64(dt), Float64(dt),
                            wall, msd_t, msd_v, r, g, rdf_done, D, Dse, win)
end

# ---------------------------------------------------------------------------
# geometry / analysis helpers
# ---------------------------------------------------------------------------

"""
Wrapped frame for output: each molecule is kept whole. The O atom is wrapped
into the primary cell and both H atoms keep their offset from that O, so a
trajectory viewer never splits a rigid molecule across a periodic boundary.
"""
function molecule_frame(q::AbstractMatrix, w::TIP3PWater)
    L = w.box.L
    out = similar(q)
    @inbounds for (o, h1, h2) in w.molecules
        for d in 1:3
            xo = q[d, o] - L[d] * floor(q[d, o] / L[d])
            out[d, o] = xo
            out[d, h1] = xo + (q[d, h1] - q[d, o])
            out[d, h2] = xo + (q[d, h2] - q[d, o])
        end
    end
    return out
end

"RDF bin centres for `nbins` bins on `[0, rmax]`."
function _rdf_axis(rmax::Float64, nbins::Int)
    dr = rmax / nbins
    return [(k - 0.5) * dr for k in 1:nbins]
end

"Normalise pair counts into g(r) with the finite-N ideal-pair convention."
function _rdf_g(counts::Vector{Int}, r::Vector{Float64}, n::Int, volume::Float64,
                frames::Int)
    dr = length(r) >= 2 ? r[2] - r[1] : 1.0
    npairs_ideal = 0.5 * n * (n - 1)
    g = zeros(Float64, length(counts))
    for k in eachindex(counts)
        rin = r[k] - 0.5 * dr
        rout = r[k] + 0.5 * dr
        shell = (4 / 3) * π * (rout^3 - rin^3) / volume
        norm = npairs_ideal * shell * frames
        g[k] = norm > 0 ? counts[k] / norm : 0.0
    end
    return g
end

"Accumulate the O-O pair histogram for one frame (minimum image, O(N^2))."
function _accumulate_oo_rdf!(counts::Vector{Int}, q::AbstractMatrix, w::TIP3PWater,
                             rmax::Float64, dr::Float64)
    L = w.box.L
    n = length(w.molecules)
    inv = 1.0 / dr
    rc2 = rmax * rmax
    nb = length(counts)
    @inbounds for b in 2:n, a in 1:(b - 1)
        oa = 3a - 2
        ob = 3b - 2
        dx = q[1, oa] - q[1, ob]
        dy = q[2, oa] - q[2, ob]
        dz = q[3, oa] - q[3, ob]
        dx -= L[1] * round(dx / L[1])
        dy -= L[2] * round(dy / L[2])
        dz -= L[3] * round(dz / L[3])
        r2 = dx * dx + dy * dy + dz * dz
        r2 > rc2 && continue
        k = floor(Int, sqrt(r2) * inv) + 1
        k <= nb && (counts[k] += 1)
    end
    return counts
end

"""
O-O radial distribution function of one configuration.

    radial_distribution(q, w; rmax=1.2, nbins=300) -> (r, g)

Present for the tests and for a single-frame check; trajectories use the
streaming accumulation inside `production!` (same normalisation).
"""
function radial_distribution(q::AbstractMatrix, w::TIP3PWater; rmax::Real = 1.2,
                             nbins::Integer = 300)
    nb = Int(nbins)
    dr = Float64(rmax) / nb
    counts = zeros(Int, nb)
    _accumulate_oo_rdf!(counts, q, w, Float64(rmax), dr)
    r = _rdf_axis(Float64(rmax), nb)
    return r, _rdf_g(counts, r, length(w.molecules), _volume(w.box.L), 1)
end

"""
Integrated autocorrelation time and the corrected standard error of the mean.

`g = 1 + 2 sum_{k>=1} rho_k` is estimated with Geyer's initial positive
sequence (paired sums, truncated at the first non-positive pair), which is the
standard conservative estimator for a stationary series. Returns
`(mean, stderr, g)` with `stderr = sqrt(var(x)/n * g)`.
"""
function mean_with_error(x::AbstractVector{<:Real})
    n = length(x)
    n == 0 && return (NaN, NaN, NaN)
    n == 1 && return (Float64(x[1]), NaN, 1.0)
    μ = mean(x)
    v = var(x)
    g, _ = autocorrelation_time(x)
    return (μ, sqrt(v / n * g), g)
end

"Integrated autocorrelation factor `g = 1 + 2 sum rho_k` (Geyer paired sum)."
function autocorrelation_time(x::AbstractVector{<:Real})
    n = length(x)
    n < 8 && return (1.0, n)
    μ = mean(x)
    xc = x .- μ
    σ2 = sum(abs2, xc) / n
    σ2 <= 0 && return (1.0, n)
    maxlag = min(n - 2, max(1, ceil(Int, 10 * log10(n))))
    ρ = Vector{Float64}(undef, maxlag)
    for k in 1:maxlag
        s = 0.0
        @inbounds for i in 1:(n - k)
            s += xc[i] * xc[i + k]
        end
        ρ[k] = s / (n * σ2)
    end
    g = 1.0
    k = 1
    while k + 1 <= maxlag
        pair = ρ[k] + ρ[k + 1]
        pair <= 0 && break
        g += 2 * pair
        k += 2
    end
    return (max(1.0, g), n)
end

"""
Ordinary-least-squares slope with a Newey-West (Bartlett) HAC standard error.

A plain OLS standard error assumes independent residuals; an MD energy (or
temperature) series is strongly autocorrelated, so the naive error can be off by
an order of magnitude. The HAC lag is the usual `4 (n/100)^(2/9)` rule unless
`lag` is given. Returns `(slope, stderr, lag)`.
"""
function ols_slope_hac(t::AbstractVector{<:Real}, y::AbstractVector{<:Real};
                       lag::Union{Nothing,Integer} = nothing)
    n = length(y)
    n == length(t) || throw(ArgumentError("t and y must have equal length"))
    n >= 3 || return (NaN, NaN, 0)
    tb = mean(t)
    yb = mean(y)
    sxx = sum(abs2, ti - tb for ti in t)
    sxx <= 0 && return (NaN, NaN, 0)
    slope = sum((t[i] - tb) * (y[i] - yb) for i in 1:n) / sxx
    resid = [y[i] - (yb + slope * (t[i] - tb)) for i in 1:n]
    L = lag === nothing ? min(n - 1, max(1, ceil(Int, 4 * (n / 100)^(2 / 9)))) : Int(lag)
    v = sum(abs2, resid) / n
    for l in 1:L
        w = 1.0 - l / (L + 1)
        c = sum(resid[i] * resid[i + l] for i in 1:(n - l)) / n
        v += 2 * w * c
    end
    se = sqrt(max(v, 0.0) / sxx)
    return (slope, se, L)
end

"""
    diffusion_from_msd(t, msd) -> (D, se, (t_lo, t_hi))

Fit `MSD(t) = 6 D t + c` over the second half of the window (the part where the
motion is diffusive rather than ballistic) and report `D` in nm^2/ps with the
ordinary least-squares standard error of the slope. The fit window is returned
so the caller can report what was actually used.
"""
function diffusion_from_msd(t::Vector{Float64}, msd::Vector{Float64})
    n = length(t)
    n < 8 && return (NaN, NaN, (NaN, NaN))
    lo = max(1, n ÷ 2)
    tt = t[lo:end]
    yy = msd[lo:end]
    slope, se, _ = ols_slope_hac(tt, yy)
    return (slope / 6, se / 6, (tt[1], tt[end]))
end

"""
NVE velocity-Verlet step with the principal-axis rigid-body drift plus the final velocity
projection. Stages match `water_integrator_spec(:rigid_water_nve)` (all `:reversible`).
"""
function _nve_step!(q, p, F, Ftan, field::WaterField, m, cs, molecules, dt;
                    stage_hook = nothing, constraint_solver::Symbol = :principal_axis,
                    q_ref::Union{Nothing,Matrix{Float64}} = nothing)
    constraint_solver in (:principal_axis, :settle) ||
        throw(ArgumentError("unsupported constraint_solver: $constraint_solver (expected :principal_axis or :settle)"))
    n = size(q, 2)
    _emit_water_stage(stage_hook, :half_kick)
    @inbounds for j in 1:n, d in 1:3
        p[d, j] += 0.5 * dt * F[d, j]
    end
    _emit_water_stage(stage_hook, :rigid_body_drift)
    if constraint_solver === :settle
        qold = q_ref === nothing ? copy(q) : copyto!(q_ref, q)
        @inbounds for j in 1:n, d in 1:3
            q[d, j] += dt * (p[d, j] / m[j])
        end
        Settle.settle_step!(qold, q, p, m, cs, dt; box = field.system.ctx.box)
    else
        _drift!(q, p, m, molecules, dt)
    end
    _emit_water_stage(stage_hook, :force_evaluation)
    U = _tangent_forces!(F, Ftan, q, m, cs, field)
    _emit_water_stage(stage_hook, :half_kick)
    @inbounds for j in 1:n, d in 1:3
        p[d, j] += 0.5 * dt * F[d, j]
    end
    _emit_water_stage(stage_hook, :velocity_constraints)
    if constraint_solver === :settle
        Settle.project_momenta!(q, p, m, cs; box = field.system.ctx.box)
    else
        Constraints.project_velocities!(q, p, m, cs; tolerance = CONSTRAINT_TOL)
    end
    return U
end

function _total_molecular_angmom(q::AbstractMatrix, p::AbstractMatrix,
                                 m::AbstractVector, molecules)
    Lx = 0.0; Ly = 0.0; Lz = 0.0
    @inbounds for (o, h1, h2) in molecules
        M = m[o] + m[h1] + m[h2]
        x1 = (m[o] * q[1, o] + m[h1] * q[1, h1] + m[h2] * q[1, h2]) / M
        x2 = (m[o] * q[2, o] + m[h1] * q[2, h1] + m[h2] * q[2, h2]) / M
        x3 = (m[o] * q[3, o] + m[h1] * q[3, h1] + m[h2] * q[3, h2]) / M
        v1 = (p[1, o] + p[1, h1] + p[1, h2]) / M
        v2 = (p[2, o] + p[2, h1] + p[2, h2]) / M
        v3 = (p[3, o] + p[3, h1] + p[3, h2]) / M
        for i in (o, h1, h2)
            rx = q[1, i] - x1; ry = q[2, i] - x2; rz = q[3, i] - x3
            px = p[1, i] - m[i] * v1; py = p[2, i] - m[i] * v2; pz = p[3, i] - m[i] * v3
            Lx += ry * pz - rz * py
            Ly += rz * px - rx * pz
            Lz += rx * py - ry * px
        end
    end
    return (Lx, Ly, Lz)
end

"Comprehensive diagnostics from an unthermostatted rigid-water NVE run."
struct NVEDiagnostics
    nsteps::Int
    dt_ps::Float64
    energy_band::Float64
    energy_slope::Float64
    energy_slope_se::Float64
    potential_slope::Float64
    potential_slope_se::Float64
    kinetic_slope::Float64
    kinetic_slope_se::Float64
    max_pos_residual::Float64
    max_tan_residual::Float64
    max_bond_error::Float64
    max_momentum_drift::Float64
    max_angmom_drift::Float64
end

"Samples and diagnostics from `run_nve!`."
struct NVERunResult
    samples::Vector{Sample}
    diagnostics::NVEDiagnostics
end

"""
    run_nve!(tr, nsteps; dt=0.001, sample_every=1, mutate=false, stage_hook=nothing)
        -> NVERunResult

Run pure unthermostatted NVE velocity-Verlet with rigid-body drift on `tr` (or
a copy when `mutate = false`). Diagnostics use every integration step,
independently of `sample_every`. Tracks energy band, HAC slopes of `E(t)`,
`U(t)`, and `K(t)`, constraint residuals, bond errors, linear momentum drift,
and internal molecular angular momentum drift.
"""
function run_nve!(tr::WaterTrajectory, nsteps::Integer;
                  dt::Real = 0.001, sample_every::Integer = 1,
                  mutate::Bool = false, stage_hook = nothing,
                  constraint_solver::Symbol = :principal_axis)
    constraint_solver in (:principal_axis, :settle) ||
        throw(ArgumentError("unsupported constraint_solver: $constraint_solver (expected :principal_axis or :settle)"))
    nsteps >= 1 || throw(ArgumentError("nsteps must be >= 1"))
    sample_every >= 1 || throw(ArgumentError("sample_every must be >= 1"))
    dtf = Float64(dt)
    (isfinite(dtf) && dtf > 0) || throw(ArgumentError("dt must be positive and finite"))

    q = mutate ? tr.st.q : copy(tr.st.q)
    p = mutate ? tr.st.p : copy(tr.st.p)
    F = mutate ? tr.st.F : copy(tr.st.F)
    Ftan = similar(q)
    q_ref = constraint_solver === :settle ? similar(q) : nothing
    m = tr.water.masses
    cs = tr.water.constraints
    mols = tr.water.molecules

    U0 = _tangent_forces!(F, Ftan, q, m, cs, tr.field)
    if mutate
        tr.st.force_valid = true
    end
    K0 = Dynamics.kinetic_energy(tr.field.system, p)
    P0 = vec(sum(p, dims = 2))
    L0 = _total_molecular_angmom(q, p, m, mols)

    samples = Sample[]
    t_arr = Float64[0.0]
    E_arr = Float64[U0 + K0]
    U_arr = Float64[U0]
    K_arr = Float64[K0]

    max_pos = maximum(abs, Constraints.constraint_residuals(cs, q))
    max_tan = maximum(abs, Constraints.tangent_residuals(cs, q, p, m))
    max_bond = water_residuals(tr).max_bond_error
    max_dP = 0.0
    max_dL = 0.0
    r_oh = TIP3P_PARAMETERS.r_OH

    for s in 1:Int(nsteps)
        U = _nve_step!(q, p, F, Ftan, tr.field, m, cs, mols, dtf;
                       stage_hook = stage_hook,
                       constraint_solver = constraint_solver,
                       q_ref = q_ref)
        if mutate
            tr.step += 1
            tr.time_ps += dtf
        end
        t_cur = s * dtf
        K = Dynamics.kinetic_energy(tr.field.system, p)
        E = U + K
        push!(t_arr, t_cur)
        push!(E_arr, E)
        push!(U_arr, U)
        push!(K_arr, K)
        step_idx = mutate ? tr.step : s
        time_val = mutate ? tr.time_ps : t_cur
        if s % Int(sample_every) == 0 || s == Int(nsteps)
            push!(samples, Sample(step_idx, time_val, U, K, E, temperature_of(tr, p)))
        end

        max_pos = max(max_pos, maximum(abs, Constraints.constraint_residuals(cs, q)))
        max_tan = max(max_tan, maximum(abs, Constraints.tangent_residuals(cs, q, p, m)))
        @inbounds for (o, h1, h2) in mols
            d1 = sqrt((q[1, h1] - q[1, o])^2 + (q[2, h1] - q[2, o])^2 + (q[3, h1] - q[3, o])^2)
            d2 = sqrt((q[1, h2] - q[1, o])^2 + (q[2, h2] - q[2, o])^2 + (q[3, h2] - q[3, o])^2)
            d3 = sqrt((q[1, h2] - q[1, h1])^2 + (q[2, h2] - q[2, h1])^2 + (q[3, h2] - q[3, h1])^2)
            max_bond = max(max_bond, abs(d1 - r_oh), abs(d2 - r_oh), abs(d3 - R_HH))
        end
        Pt = vec(sum(p, dims = 2))
        max_dP = max(max_dP, maximum(abs, Pt .- P0))
        Lt = _total_molecular_angmom(q, p, m, mols)
        max_dL = max(max_dL, max(abs(Lt[1] - L0[1]), abs(Lt[2] - L0[2]), abs(Lt[3] - L0[3])))
    end

    band = maximum(abs, E_arr .- E_arr[1])
    sE, seE, _ = ols_slope_hac(t_arr, E_arr)
    sU, seU, _ = ols_slope_hac(t_arr, U_arr)
    sK, seK, _ = ols_slope_hac(t_arr, K_arr)
    diag = NVEDiagnostics(Int(nsteps), dtf, band, sE, seE, sU, seU, sK, seK,
                          max_pos, max_tan, max_bond, max_dP, max_dL)
    return NVERunResult(samples, diag)
end

"""
    nve_energy_band(tr; nsteps=1000, dt=0.001) -> (band, slope, se, nsteps)

Velocity-Verlet (NVE) run from a copy of the current state, with the same
composed field and the same rigid constraints. `band = max|E - E0|` in kJ/mol is
the shadow-energy excursion; the slope is a Newey-West HAC fit. This is the
energy-conservation check: the Langevin production run has no conserved energy,
so drift there measures thermostat exchange, not integrator quality.
"""
function nve_energy_band(tr::WaterTrajectory; nsteps::Integer = 1000,
                         dt::Real = 0.001)
    q = copy(tr.st.q)
    p = copy(tr.st.p)
    F = copy(tr.st.F)
    m = tr.water.masses
    cs = tr.water.constraints
    Ftan = similar(q)
    U = _tangent_forces!(F, Ftan, q, m, tr.water.constraints, tr.field)
    E = Vector{Float64}(undef, nsteps + 1)
    t = Vector{Float64}(undef, nsteps + 1)
    E[1] = U + Dynamics.kinetic_energy(tr.field.system, p)
    t[1] = 0.0
    for s in 1:Int(nsteps)
        U = _nve_step!(q, p, F, Ftan, tr.field, m, cs, tr.water.molecules,
                       Float64(dt))
        E[s + 1] = U + Dynamics.kinetic_energy(tr.field.system, p)
        t[s + 1] = s * Float64(dt)
    end
    band = maximum(abs, E .- E[1])
    slope, se, _ = ols_slope_hac(t, E)
    return (band, slope, se, Int(nsteps))
end

# ---------------------------------------------------------------------------
# run summary
# ---------------------------------------------------------------------------

"Everything a run is allowed to claim, with the measurement that supports it."
struct RunReport
    label::String
    molecules::Int
    atoms::Int
    box_nm::NTuple{3,Float64}
    volume_nm3::Float64
    density::Float64
    rcut::Float64
    pme_rcut::Float64
    alpha::Float64
    grid::NTuple{3,Int}
    order::Int
    relax_steps::Int
    relax_u_initial::Float64
    relax_u_final::Float64
    relax_fmax_final::Float64
    equil_ps::Float64
    equil_temp::Float64
    equil_temp_se::Float64
    prod_steps::Int
    prod_ns::Float64
    prod_wall_s::Float64
    ns_per_day::Float64
    peak_rss_mib::Float64
    drift_slope::Float64
    drift_se::Float64
    drift_window_kj::Float64
    temp_mean::Float64
    temp_se::Float64
    temp_iat::Float64
    n_samples::Int
    potential_mean::Float64
    potential_per_molecule::Float64
    nve_band_kj::Float64
    nve_slope::Float64
    nve_slope_se::Float64
    nve_steps::Int
    diffusion::Float64
    diffusion_se::Float64
    msd_window::NTuple{2,Float64}
    rdf_frames::Int
    first_peak_nm::Float64
    first_min_nm::Float64
    rss_source::Symbol
end

"Assemble a `RunReport` from the staged results (all numbers measured, not assumed)."
function summarize_run(; label::String, water::TIP3PWater, field::WaterField,
                       relax::RelaxReport, equil::EquilibrationResult,
                       prod::ProductionResult, nve = nothing,
                       peak_rss_mib::Float64 = Sys.maxrss() / 2.0^20,
                       rss_source::Symbol = :self)
    E = [s.E for s in prod.samples]
    t = [s.time_ps for s in prod.samples]
    n = length(E)
    slope = NaN
    se = NaN
    window = NaN
    if n >= 3
        slope, se, _ = ols_slope_hac(t, E)
        window = abs(slope) * (t[end] - t[1])
    end
    temps = [s.T_inst for s in prod.samples]
    tm, tse, tiat = mean_with_error(temps)
    U = [s.U for s in prod.samples]
    umean = isempty(U) ? NaN : mean(U)
    if nve === nothing
        nve_band, nve_slope, nve_se, nve_steps = NaN, NaN, NaN, 0
    else
        nve_band, nve_slope, nve_se, nve_steps = nve.band, nve.slope, nve.se, nve.steps
    end
    pk, pm = rdf_extrema(prod.rdf_r, prod.rdf_g)
    rss = peak_rss_mib > 0 ? peak_rss_mib : Sys.maxrss() / 2.0^20
    return RunReport(label, length(water.molecules), size(water.positions, 2),
                     water.box.L, _volume(water.box.L), water.density, field.rcut,
                     field.pme_rcut, field.alpha, field.grid, field.order,
                     relax.steps,
                     relax.u_initial, relax.u_final, relax.max_force_final,
                     equil.ps, equil.temp_mean, equil.temp_se, prod.steps,
                     prod.ps / 1000, prod.wall_s,
                     prod.ps / 1000 / (prod.wall_s / 86400), rss, slope, se, window,
                     tm, tse, tiat, n, umean, umean / length(water.molecules),
                     nve_band, nve_slope, nve_se,
                     nve_steps, prod.diffusion, prod.diffusion_se,
                     prod.msd_fit_window_ps, prod.rdf_frames, pk, pm, rss_source)
end

"Box volume (nm^3) of an orthorhombic cell."
_volume(L) = L[1] * L[2] * L[3]

"""
First RDF peak height/position and the following minimum position.

Returns `(peak_r, min_r)`; NaN when the curve has no interior peak.
"""
function rdf_extrema(r::Vector{Float64}, g::Vector{Float64})
    n = length(g)
    n < 5 && return (NaN, NaN)
    imax = 0
    gmax = -Inf
    for k in 2:(n - 1)
        # A real peak: strictly above both neighbours and not in the excluded
        # (g == 0) region below the closest possible contact.
        if g[k] > g[k - 1] && g[k] >= g[k + 1] && g[k] > gmax && g[k] > 1e-3
            gmax = g[k]
            imax = k
            break   # first peak only
        end
    end
    imax == 0 && return (NaN, NaN)
    kmin = imax
    while kmin < n && g[kmin + 1] <= g[kmin]
        kmin += 1
    end
    return (r[imax], kmin < n ? r[kmin] : NaN)
end

"Human-readable report table row block."
function print_report(io::IO, rep::RunReport)
    @printf(io, "run: %s\n", rep.label)
    @printf(io, "  system        %d molecules, %d atoms\n", rep.molecules, rep.atoms)
    @printf(io, "  box           %.4f x %.4f x %.4f nm  (V = %.4f nm^3)\n",
            rep.box_nm[1], rep.box_nm[2], rep.box_nm[3], rep.volume_nm3)
    @printf(io, "  density       %.4f g/cm^3\n", rep.density)
    @printf(io, "  force field   LJ :lfs rcut=%.3f nm + PME rcut=%.3f nm alpha=%.2f 1/nm,",
            rep.rcut, rep.pme_rcut, rep.alpha)
    @printf(io, " grid %dx%dx%d order %d\n", rep.grid[1], rep.grid[2], rep.grid[3],
            rep.order)
    @printf(io, "  relaxation    %d damped steps, U %.1f -> %.1f kJ/mol, max|F| %.1f kJ/mol/nm\n",
            rep.relax_steps, rep.relax_u_initial, rep.relax_u_final,
            rep.relax_fmax_final)
    @printf(io, "  equilibration %.3f ps, T = %.2f +/- %.2f K\n", rep.equil_ps,
            rep.equil_temp, rep.equil_temp_se)
    @printf(io, "  production    %d steps, %.4f ns, wall %.1f s, %.2f ns/day\n",
            rep.prod_steps, rep.prod_ns, rep.prod_wall_s, rep.ns_per_day)
    @printf(io, "  memory        peak RSS %.0f MiB (%s)\n", rep.peak_rss_mib,
            rep.rss_source === :self ? "Sys.maxrss, this process" : "reported by launcher")
    @printf(io, "  energy        drift slope %.4f +/- %.4f kJ/mol/ps (HAC), total %.1f kJ/mol over window\n",
            rep.drift_slope, rep.drift_se, rep.drift_window_kj)
    @printf(io, "  temperature   %.2f +/- %.2f K (IAT-corrected SE, g = %.2f, %d samples)\n",
            rep.temp_mean, rep.temp_se, rep.temp_iat, rep.n_samples)
    @printf(io, "  potential     %.1f kJ/mol (%.2f kJ/mol per molecule)\n",
            rep.potential_mean, rep.potential_per_molecule)
    if !isnan(rep.nve_band_kj)
        @printf(io, "  NVE check     %d steps: band %.3g kJ/mol, slope %.3g +/- %.3g kJ/mol/ps\n",
                rep.nve_steps, rep.nve_band_kj, rep.nve_slope, rep.nve_slope_se)
    end
    @printf(io, "  diffusion     %.4f +/- %.4f nm^2/ns (MSD fit %.2f-%.2f ps)\n",
            rep.diffusion * 1000, rep.diffusion_se * 1000, rep.msd_window[1],
            rep.msd_window[2])
    @printf(io, "  O-O RDF       first peak %.3f nm, first minimum %.3f nm (%d frames)\n",
            rep.first_peak_nm, rep.first_min_nm, rep.rdf_frames)
    return nothing
end

print_report(rep::RunReport) = print_report(stdout, rep)

end # module
