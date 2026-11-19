"""
Scalar-charge particle-mesh Ewald (PME) evaluator for orthorhombic cells.

This module is standalone (it does not import `Electrostatics`) but deliberately
uses the same energy convention:

    U = prefactor * ( U_real + U_recip + U_self + U_background + U_pair )

with

    U_real       = sum_{i<j} q_i q_j erfc(alpha r_ij) / r_ij
    U_recip      = (2pi/V) sum_{k != 0} exp(-k^2/(4 alpha^2))/k^2 |S(k)|^2
    U_self       = -alpha/sqrt(pi) sum_i q_i^2
    U_background = -pi Q^2/(2 alpha^2 V)                 (uniform background)
    U_pair       = sum_{canonical (i,j)} (s_ij - 1) q_i q_j / r_ij

`S(k)` is evaluated with cardinal B-spline charge spreading and an FFT, and
forces are the exact negative gradient of the *discrete* PME energy (analytic
derivative of the interpolation weights), not an `i k` spectral derivative.
The reciprocal potential is interpolated back with the same stencil, so
`energy`, `forces`, and `charge gradients` are mutually consistent.

## Conventions and limitations

* The uniform neutralizing background is always present. Charges do not need to
  sum to zero. A tin-foil (conducting) boundary is used: **no** surface/dipole
  term is added. This is *not* a slab or vacuum boundary correction.
* `PairCorrections` has exactly the `Electrostatics.PairCorrections` convention:
  for each canonical pair, `scaled=(i,j,s)` adds `(s-1)*prefactor*q_i*q_j/r` at
  the minimum-image distance; `excluded=(i,j)` is `s=0`. No bonded topology is
  inferred.
* Orthorhombic periodic cells only. The real-space cutoff must satisfy
  `0 < rcut < min(box)/2`; the cell-linked traversal assumes that strict bound.
  Linked cells are sized at least `rcut` wide (`floor(L/rcut)` cells per
  dimension), so exactly one ring of neighbouring cells is visited.
* Spline order is restricted to 4 or 6. The default grid targets roughly three
  grid points per reciprocal decay length `1/alpha` and rounds up to a power of
  two; pass `grid=(Nx,Ny,Nz)` to control accuracy/cost explicitly. Accuracy is
  controlled by `alpha`, `rcut`, `grid`, and `order`; this module makes no claim
  of broad validation.
* The PME grid, FFTW plans, spline stencils, linked-cell arrays, and input
  scratch buffers are mutable state inside `PMEPlan`. A plan is reusable for
  repeated evaluations with the same atom count and box; every warm evaluation
  reuses that state and does not allocate. A plan is **not** thread-safe and
  must not be used concurrently.
* Coordinates may lie outside the primary cell; minimum-image distances and
  periodic stencils are used throughout.
"""
module PME

using LinearAlgebra
using FFTW

const _PARENT = parentmodule(@__MODULE__)
const _HAS_FACTOR_PROTOCOL = isdefined(_PARENT, :Factors) && isdefined(_PARENT, :Types)
if _HAS_FACTOR_PROTOCOL
    const Factors = getfield(_PARENT, :Factors)
    const Types = getfield(_PARENT, :Types)
end

export PMEPlan, PairCorrections, evaluate_pme, pme_evaluate, pme_energy,
       pme_energy_forces, pme_energy_forces!, pme_charge_gradient!,
       pme_box_derivative, pme_box_derivative!,
       pme_energy_forces_box_derivative, pme_energy_forces_box_derivative!,
       pme_error_estimate, pme_recommend_parameters, pme_validate_budget,
       PMEFactor, pme_factor_energy, pme_factor_charge_gradient!,
       pme_factor_box_derivative, pme_factor_box_derivative!,
       pme_factor_energy_forces_box_derivative!

const _SQRTPI = sqrt(pi)
const _FFTW_PLAN_LOCK = ReentrantLock()

_erfc(x::Float64) = ccall(:erfc, Cdouble, (Cdouble,), x)

# ---------------------------------------------------------------------------
# Pair corrections (same convention as Electrostatics.PairCorrections)
# ---------------------------------------------------------------------------

"""
    PairCorrections(; excluded=[(i,j), ...], scaled=[(i,j,s), ...])

Point-Coulomb pair corrections applied after the Ewald sum. `scaled` multiplies
only the correction relative to the full Ewald result, i.e. it adds
`(s-1)*prefactor*q_i*q_j/r` at the minimum-image distance. `excluded` is `s=0`.
This matches `Electrostatics.PairCorrections` exactly; it is not an alternative
physical model and does not infer topology.
"""
struct PairCorrections
    weights::Dict{Tuple{Int,Int},Float64}
    function PairCorrections(; excluded=Tuple{Int,Int}[], scaled=Tuple{Int,Int,Float64}[])
        w = Dict{Tuple{Int,Int},Float64}()
        for p in excluded
            length(p) == 2 || throw(ArgumentError("excluded entries must be (i,j)"))
            i, j = _canonical(p[1], p[2])
            haskey(w, (i, j)) && throw(ArgumentError("duplicate pair correction"))
            w[(i, j)] = 0.0
        end
        for p in scaled
            length(p) == 3 || throw(ArgumentError("scaled entries must be (i,j,weight)"))
            i, j = _canonical(p[1], p[2])
            s = Float64(p[3])
            isfinite(s) && s >= 0 || throw(ArgumentError("scale must be finite and nonnegative"))
            haskey(w, (i, j)) && throw(ArgumentError("duplicate pair correction"))
            w[(i, j)] = s
        end
        return new(w)
    end
end

# Shared empty correction set. Using it as a keyword default keeps the hot
# evaluators allocation-free: `PairCorrections()` would build a fresh Dict on
# every call. It is never mutated, so sharing it is safe.
const _EMPTY_CORRECTIONS = PairCorrections()

function _canonical(i, j)
    i isa Integer && j isa Integer && i > 0 && j > 0 && i != j ||
        throw(ArgumentError("pair indices must be distinct positive integers"))
    return min(Int(i), Int(j)), max(Int(i), Int(j))
end

# ---------------------------------------------------------------------------
# Cardinal B-splines
# ---------------------------------------------------------------------------

"""Centered cardinal B-spline of even order `p`, support `[-p/2, p/2]`."""
function cardinal_bspline(p::Int, x::Float64)
    abs(x) >= p / 2 && return 0.0
    z = x + p / 2
    s = 0.0
    for k in 0:p
        t = z - k
        if t > 0
            s += (isodd(k) ? -1.0 : 1.0) * binomial(p, k) * t^(p - 1)
        end
    end
    return s / factorial(p - 1)
end

"""Derivative of the centered cardinal B-spline: `B_p'(x) = B_{p-1}(x+1/2) - B_{p-1}(x-1/2)`."""
cardinal_bspline_deriv(p::Int, x::Float64) =
    cardinal_bspline(p - 1, x + 0.5) - cardinal_bspline(p - 1, x - 0.5)

# ---------------------------------------------------------------------------
# Reusable plan
# ---------------------------------------------------------------------------

"""
    PMEPlan(q, box; alpha, rcut, grid=nothing, order=4, prefactor=1.0)

Precompute an FFTW plan, the B-spline influence function, the periodic
cell-pair topology, and every working array for a system with `size(q,2)`
atoms and orthorhombic `box`. The plan can be reused for repeated
`pme_energy`, `pme_energy_forces!`, and `pme_charge_gradient!` calls with the
same atom count and box; all per-call workspace (including copies of the
coordinates and charges) lives in the plan, so a warm call allocates nothing.
The plan is mutable and not thread-safe.
"""
function _parse_box(box)
    if hasproperty(box, :H) && hasproperty(box, :Hinv)
        H = Matrix{Float64}(box.H)
        Hinv = Matrix{Float64}(box.Hinv)
        widths = hasproperty(box, :widths) ? NTuple{3,Float64}(box.widths) :
                 ntuple(i -> 1.0 / norm(view(Hinv, i, :)), 3)
        V = det(H)
        isfinite(V) && V > 0 || throw(ArgumentError("cell matrix must have finite positive volume"))
        hinv_norms = ntuple(i -> norm(view(Hinv, i, :)), 3)
        return (box=box, H=H, Hinv=Hinv, widths=widths, hinv_norms=hinv_norms, is_triclinic=true, volume=V)
    elseif hasproperty(box, :L)
        L = (Float64(box.L[1]), Float64(box.L[2]), Float64(box.L[3]))
        all(isfinite, L) && all(>(0), L) ||
            throw(ArgumentError("box lengths must be finite and positive"))
        H = [L[1] 0.0 0.0; 0.0 L[2] 0.0; 0.0 0.0 L[3]]
        Hinv = [1.0 / L[1] 0.0 0.0; 0.0 1.0 / L[2] 0.0; 0.0 0.0 1.0 / L[3]]
        widths = L
        hinv_norms = (1.0 / L[1], 1.0 / L[2], 1.0 / L[3])
        return (box=L, H=H, Hinv=Hinv, widths=widths, hinv_norms=hinv_norms, is_triclinic=false, volume=L[1]*L[2]*L[3])
    elseif box isa AbstractMatrix
        size(box) == (3, 3) || throw(ArgumentError("cell matrix must be 3×3"))
        all(isfinite, box) || throw(ArgumentError("cell matrix must be finite"))
        H = Matrix{Float64}(box)
        V = det(H)
        isfinite(V) && V > 0 || throw(ArgumentError("cell matrix must have finite positive volume"))
        Hinv = inv(H)
        hinv_norms = ntuple(i -> norm(view(Hinv, i, :)), 3)
        all(w -> isfinite(w) && w > 0, hinv_norms) || throw(ArgumentError("invalid cell matrix"))
        widths = ntuple(i -> 1.0 / hinv_norms[i], 3)
        is_ortho = maximum(abs, H .- Diagonal(diag(H))) <= 1e-12 && all(>(0), diag(H))
        if is_ortho
            L = (H[1, 1], H[2, 2], H[3, 3])
            return (box=L, H=H, Hinv=Hinv, widths=widths, hinv_norms=hinv_norms, is_triclinic=false, volume=V)
        else
            return (box=H, H=H, Hinv=Hinv, widths=widths, hinv_norms=hinv_norms, is_triclinic=true, volume=V)
        end
    else
        Lv = Float64.(collect(box))
        length(Lv) == 3 || throw(ArgumentError("box must contain three lengths or be a 3×3 matrix"))
        all(isfinite, Lv) && all(>(0), Lv) ||
            throw(ArgumentError("box lengths must be finite and positive"))
        L = (Lv[1], Lv[2], Lv[3])
        H = [L[1] 0.0 0.0; 0.0 L[2] 0.0; 0.0 0.0 L[3]]
        Hinv = [1.0 / L[1] 0.0 0.0; 0.0 1.0 / L[2] 0.0; 0.0 0.0 1.0 / L[3]]
        widths = L
        hinv_norms = (1.0 / L[1], 1.0 / L[2], 1.0 / L[3])
        return (box=L, H=H, Hinv=Hinv, widths=widths, hinv_norms=hinv_norms, is_triclinic=false, volume=L[1]*L[2]*L[3])
    end
end

struct PMEPlan{P, B, FP, IFP}
    N::Int
    box::B
    H::Matrix{Float64}
    Hinv::Matrix{Float64}
    widths::NTuple{3,Float64}
    hinv_row_norms::NTuple{3,Float64}
    half_min_w2::Float64
    is_triclinic::Bool
    volume::Float64
    alpha::Float64
    rcut::Float64
    prefactor::Float64
    dims::NTuple{3,Int}
    order::Int
    influence::Array{Float64,3}
    work_real::Array{Float64,3}
    work_k::Array{ComplexF64,3}
    rfftplan::FP
    brfftplan::IFP
    cell_n::NTuple{3,Int}
    cell_pairs::Vector{Tuple{Int,Int}}
    cell_counts::Vector{Int}
    cell_offsets::Vector{Int}
    cell_id::Vector{Int}
    perm::Vector{Int}
    sxbuf::Matrix{Float64}
    sqbuf::Vector{Float64}
    sfbuf::Matrix{Float64}
    nodes::Array{Int,3}
    weights::Array{Float64,3}
    dweights::Array{Float64,3}
    xbuf::Matrix{Float64}
    qqbuf::Vector{Float64}
end

function PMEPlan(q::AbstractMatrix, box; alpha::Real, rcut::Real,
                 grid=nothing, order::Integer=4, prefactor::Real=1.0)
    order isa Integer && (order == 4 || order == 6) ||
        throw(ArgumentError("spline order must be 4 or 6"))
    P = Int(order)

    size(q, 1) == 3 || throw(ArgumentError("q must have shape 3×N"))
    N = size(q, 2)
    all(isfinite, q) || throw(ArgumentError("coordinates must be finite"))

    pbox = _parse_box(box)
    a = Float64(alpha)
    rc = Float64(rcut)
    p = Float64(prefactor)
    isfinite(a) && a > 0 || throw(ArgumentError("alpha must be finite and positive"))
    min_w = minimum(pbox.widths)
    isfinite(rc) && rc > 0 && rc < min_w / 2 ||
        throw(ArgumentError("require 0 < rcut < min(box)/2"))
    isfinite(p) || throw(ArgumentError("prefactor must be finite"))

    lengths = ntuple(i -> norm(view(pbox.H, :, i)), 3)
    dims = _grid_dims(grid, lengths, a, P)
    V = pbox.volume
    influence = _influence(dims, pbox.Hinv, a, P, V)

    work_real = zeros(Float64, dims...)
    work_k = Array{ComplexF64,3}(undef, dims[1] ÷ 2 + 1, dims[2], dims[3])
    rfftplan, brfftplan = lock(_FFTW_PLAN_LOCK) do
        (plan_rfft(work_real; flags=FFTW.ESTIMATE),
         plan_brfft(work_k, dims[1]; flags=FFTW.ESTIMATE))
    end

    if pbox.is_triclinic
        nc = ntuple(d -> max(1, floor(Int, pbox.widths[d] / rc)), 3)
        cell_pairs = _triclinic_cell_pairs(nc)
    else
        L = (pbox.H[1, 1], pbox.H[2, 2], pbox.H[3, 3])
        nc2 = ntuple(d -> max(1, floor(Int, 2.0 * L[d] / rc)), 3)
        nc = prod(nc2) <= max(32768, 4 * max(N, 1)) ? nc2 :
             ntuple(d -> max(1, floor(Int, L[d] / rc)), 3)
        cell_pairs = _cell_pairs(nc, L, rc)
    end
    ncell = prod(nc)
    cell_counts = zeros(Int, ncell)
    cell_offsets = zeros(Int, ncell + 1)
    cell_id = Vector{Int}(undef, N)
    perm = Vector{Int}(undef, N)
    sxbuf = Matrix{Float64}(undef, 3, N)
    sqbuf = Vector{Float64}(undef, N)
    sfbuf = Matrix{Float64}(undef, 3, N)

    nodes = Array{Int,3}(undef, P, 3, N)
    weights = Array{Float64,3}(undef, P, 3, N)
    dweights = Array{Float64,3}(undef, P, 3, N)
    xbuf = Matrix{Float64}(undef, 3, N)
    qqbuf = Vector{Float64}(undef, N)
    half_min_w2 = (0.5 * min_w)^2

    return PMEPlan{P, typeof(pbox.box), typeof(rfftplan), typeof(brfftplan)}(
        N, pbox.box, pbox.H, pbox.Hinv, pbox.widths, pbox.hinv_norms, half_min_w2,
        pbox.is_triclinic, V, a, rc, p, dims, P, influence, work_real, work_k,
        rfftplan, brfftplan, nc, cell_pairs, cell_counts, cell_offsets, cell_id,
        perm, sxbuf, sqbuf, sfbuf, nodes, weights, dweights, xbuf, qqbuf)
end

function _grid_dims(grid, lengths::NTuple{3,Float64}, alpha::Float64, order::Int)
    if grid === nothing
        return ntuple(d -> begin
            # About three grid points per reciprocal decay length 1/alpha.
            n = ceil(Int, 3 * alpha * lengths[d])
            n = max(n, 2 * order)
            nextpow(2, n)
        end, 3)
    end
    length(grid) == 3 || throw(ArgumentError("grid must contain three dimensions"))
    dims = ntuple(d -> begin
        g = grid[d]
        g isa Integer || throw(ArgumentError("grid dimensions must be integers"))
        Int(g)
    end, 3)
    all(>=(2 * order), dims) ||
        throw(ArgumentError("each grid dimension must be at least 2*order"))
    return dims
end

# FFTW mode index: 0, 1, ..., N/2, -N/2+1, ..., -1 (Nyquist may be +N/2).
function _mode(i::Int, n::Int)
    m = i - 1
    return m <= n ÷ 2 ? m : m - n
end

function _influence(dims::NTuple{3,Int}, Hinv::AbstractMatrix{Float64},
                    alpha::Float64, order::Int, volume::Float64)
    n1, n2, n3 = dims
    nk1 = n1 ÷ 2 + 1
    G = zeros(Float64, nk1, n2, n3)
    four_pi_over_v = 4pi / volume
    inv_4a2 = 1.0 / (4.0 * alpha * alpha)
    @inbounds for k in 1:n3, j in 1:n2, i in 1:nk1
        m1 = i - 1
        m2 = _mode(j, n2)
        m3 = _mode(k, n3)

        m1 == 0 && m2 == 0 && m3 == 0 && continue
        # Nyquist planes are omitted to avoid an ambiguous half-mode.
        if (iseven(n1) && m1 == n1 ÷ 2) ||
           (iseven(n2) && abs(m2) == n2 ÷ 2) ||
           (iseven(n3) && abs(m3) == n3 ÷ 2)
            continue
        end

        kx = 2pi * (Hinv[1, 1] * m1 + Hinv[2, 1] * m2 + Hinv[3, 1] * m3)
        ky = 2pi * (Hinv[1, 2] * m1 + Hinv[2, 2] * m2 + Hinv[3, 2] * m3)
        kz = 2pi * (Hinv[1, 3] * m1 + Hinv[2, 3] * m2 + Hinv[3, 3] * m3)
        k2 = kx * kx + ky * ky + kz * kz
        k2 == 0.0 && continue

        B = (sinc(m1 / n1) * sinc(m2 / n2) * sinc(m3 / n3))^order
        B == 0.0 && continue

        # Potential-grid coefficient for unnormalized backward rfft (`brfft`):
        # `brfft` carries no 1/N^3 factor, so no N^3 factor is needed here.
        G[i, j, k] = four_pi_over_v * exp(-k2 * inv_4a2) / k2 / (B * B)
    end
    return G
end

function _triclinic_cell_pairs(nc::NTuple{3,Int})
    ncell = prod(nc)
    pairs = Tuple{Int,Int}[]
    seen = zeros(Int, ncell)
    sizehint!(pairs, min(ncell * 14, ncell * (ncell + 1) ÷ 2))
    for c in 1:ncell
        c1 = (c - 1) % nc[1] + 1
        c2 = ((c - 1) ÷ nc[1]) % nc[2] + 1
        c3 = (c - 1) ÷ (nc[1] * nc[2]) + 1
        for dz in -1:1, dy in -1:1, dx in -1:1
            t1 = mod(c1 + dx - 1, nc[1]) + 1
            t2 = mod(c2 + dy - 1, nc[2]) + 1
            t3 = mod(c3 + dz - 1, nc[3]) + 1
            t = t1 + nc[1] * (t2 - 1) + nc[1] * nc[2] * (t3 - 1)
            t < c && continue
            seen[t] == c && continue
            seen[t] = c
            push!(pairs, (c, t))
        end
    end
    return pairs
end

function _cell_pairs(nc::NTuple{3,Int}, L::NTuple{3,Float64}, rcut::Float64)
    ncell = prod(nc)
    w = ntuple(d -> L[d] / nc[d], 3)
    # Any periodic cell difference can be represented by an offset in
    # [-floor(n/2), floor(n/2)]; a cutoff below min(L)/2 never needs more.
    radius = ntuple(d -> max(1, min(nc[d] ÷ 2, ceil(Int, rcut / w[d]))), 3)
    pairs = Tuple{Int,Int}[]
    seen = zeros(Int, ncell)
    rc2 = rcut * rcut

    for c in 1:ncell
        c1 = (c - 1) % nc[1] + 1
        c2 = ((c - 1) ÷ nc[1]) % nc[2] + 1
        c3 = (c - 1) ÷ (nc[1] * nc[2]) + 1
        for dz in -radius[3]:radius[3],
            dy in -radius[2]:radius[2],
            dx in -radius[1]:radius[1]
            t1 = mod(c1 + dx - 1, nc[1]) + 1
            t2 = mod(c2 + dy - 1, nc[2]) + 1
            t3 = mod(c3 + dz - 1, nc[3]) + 1
            t = t1 + nc[1] * (t2 - 1) + nc[1] * nc[2] * (t3 - 1)
            t < c && continue
            seen[t] == c && continue
            seen[t] = c

            d1 = t1 - c1
            d1 -= nc[1] * round(Int, d1 / nc[1])
            d2 = t2 - c2
            d2 -= nc[2] * round(Int, d2 / nc[2])
            d3 = t3 - c3
            d3 -= nc[3] * round(Int, d3 / nc[3])
            lx = max(0.0, (abs(d1) - 1) * w[1])
            ly = max(0.0, (abs(d2) - 1) * w[2])
            lz = max(0.0, (abs(d3) - 1) * w[3])
            if lx * lx + ly * ly + lz * lz <= rc2
                push!(pairs, (c, t))
            end
        end
    end
    return pairs
end

# ---------------------------------------------------------------------------
# Internal evaluation
# ---------------------------------------------------------------------------

@inline function _triclinic_min_image(plan::PMEPlan, dx::Float64, dy::Float64, dz::Float64)
    H = plan.H
    Hinv = plan.Hinv
    f1 = Hinv[1, 1] * dx + Hinv[1, 2] * dy + Hinv[1, 3] * dz
    f2 = Hinv[2, 1] * dx + Hinv[2, 2] * dy + Hinv[2, 3] * dz
    f3 = Hinv[3, 1] * dx + Hinv[3, 2] * dy + Hinv[3, 3] * dz

    n1 = round(Int, f1); n2 = round(Int, f2); n3 = round(Int, f3)
    rx0 = dx - (H[1, 1] * n1 + H[1, 2] * n2 + H[1, 3] * n3)
    ry0 = dy - (H[2, 1] * n1 + H[2, 2] * n2 + H[2, 3] * n3)
    rz0 = dz - (H[3, 1] * n1 + H[3, 2] * n2 + H[3, 3] * n3)
    best2 = rx0 * rx0 + ry0 * ry0 + rz0 * rz0
    if best2 < plan.half_min_w2
        return (rx0, ry0, rz0), (f1 - n1, f2 - n2, f3 - n3), best2
    end
    best_x = rx0; best_y = ry0; best_z = rz0
    bn1 = n1; bn2 = n2; bn3 = n3

    R = sqrt(best2)
    b1 = plan.hinv_row_norms[1] * R
    b2 = plan.hinv_row_norms[2] * R
    b3 = plan.hinv_row_norms[3] * R

    r1_min = ceil(Int, f1 - b1); r1_max = floor(Int, f1 + b1)
    r2_min = ceil(Int, f2 - b2); r2_max = floor(Int, f2 + b2)
    r3_min = ceil(Int, f3 - b3); r3_max = floor(Int, f3 + b3)

    for c in r3_min:r3_max, b in r2_min:r2_max, a in r1_min:r1_max
        (a == n1 && b == n2 && c == n3) && continue
        rx = dx - (H[1, 1] * a + H[1, 2] * b + H[1, 3] * c)
        ry = dy - (H[2, 1] * a + H[2, 2] * b + H[2, 3] * c)
        rz = dz - (H[3, 1] * a + H[3, 2] * b + H[3, 3] * c)
        r2_val = rx * rx + ry * ry + rz * rz
        if r2_val < best2
            best2 = r2_val
            best_x = rx; best_y = ry; best_z = rz
            bn1 = a; bn2 = b; bn3 = c
        end
    end
    w1 = f1 - bn1; w2 = f2 - bn2; w3 = f3 - bn3
    return (best_x, best_y, best_z), (w1, w2, w3), best2
end

function _validate_eval(plan::PMEPlan, q::AbstractMatrix, charges::AbstractVector,
                        F, grad, G, corrections::PairCorrections;
                        accumulate::Bool = false,
                        accumulate_G::Bool = false)
    size(q) == (3, plan.N) || throw(ArgumentError("q must be 3×$(plan.N)"))
    length(charges) == plan.N || throw(ArgumentError("charge count must match coordinates"))
    all(isfinite, q) && all(isfinite, charges) ||
        throw(ArgumentError("coordinates and charges must be finite"))
    all(i <= plan.N && j <= plan.N for ((i, j), _) in corrections.weights) ||
        throw(ArgumentError("correction index exceeds particle count"))

    if F !== nothing
        size(F) == (3, plan.N) || throw(ArgumentError("F must be 3×$(plan.N)"))
        Base.mightalias(F, q) && throw(ArgumentError("force accumulator must not alias q"))
        accumulate || fill!(F, 0.0)
    end
    if grad !== nothing
        length(grad) == plan.N || throw(ArgumentError("gradient length must match atom count"))
        Base.mightalias(grad, q) && throw(ArgumentError("gradient accumulator must not alias q"))
        Base.mightalias(grad, charges) &&
            throw(ArgumentError("gradient accumulator must not alias charges"))
        fill!(grad, 0.0)
    end
    if G !== nothing
        size(G) == (3, 3) || throw(ArgumentError("G must be 3×3"))
        Base.mightalias(G, q) && throw(ArgumentError("box derivative accumulator must not alias q"))
        accumulate_G || fill!(G, 0.0)
    end
    return nothing
end

@inline function _bspline_p4(t::Float64)
    om = 1.0 - t
    t2 = t * t
    om2 = om * om
    w1 = (1.0 / 6.0) * om2 * om
    w4 = (1.0 / 6.0) * t2 * t
    w2 = (1.0 / 6.0) * (4.0 - 6.0 * t2 + 3.0 * t2 * t)
    w3 = 1.0 - w1 - w2 - w4
    dw1 = -0.5 * om2
    dw4 = 0.5 * t2
    dw2 = -2.0 * t + 1.5 * t2
    dw3 = -(dw1 + dw2 + dw4)
    return (w1, w2, w3, w4), (dw1, dw2, dw3, dw4)
end

@inline function _bspline_p6(t::Float64)
    a1 = t; a2 = 1.0 - t
    b3 = 0.5 * (1.0 - t) * a2
    b2 = 0.5 * ((t + 1.0) * a2 + (2.0 - t) * a1)
    b1 = 0.5 * t * a1
    c4 = (1.0 / 3.0) * (1.0 - t) * b3
    c3 = (1.0 / 3.0) * ((t + 2.0) * b3 + (2.0 - t) * b2)
    c2 = (1.0 / 3.0) * ((t + 1.0) * b2 + (3.0 - t) * b1)
    c1 = (1.0 / 3.0) * t * b1
    d5 = 0.25 * (1.0 - t) * c4
    d4 = 0.25 * ((t + 3.0) * c4 + (2.0 - t) * c3)
    d3 = 0.25 * ((t + 2.0) * c3 + (3.0 - t) * c2)
    d2 = 0.25 * ((t + 1.0) * c2 + (4.0 - t) * c1)
    d1 = 0.25 * t * c1
    e6 = 0.2 * (1.0 - t) * d5
    e5 = 0.2 * ((t + 4.0) * d5 + (2.0 - t) * d4)
    e4 = 0.2 * ((t + 3.0) * d4 + (3.0 - t) * d3)
    e3 = 0.2 * ((t + 2.0) * d3 + (4.0 - t) * d2)
    e2 = 0.2 * ((t + 1.0) * d2 + (5.0 - t) * d1)
    e1 = 0.2 * t * d1
    return (e6, e5, e4, e3, e2, e1), (-d5, d5 - d4, d4 - d3, d3 - d2, d2 - d1, d1)
end

function _prepare_stencils!(plan::PMEPlan{P}, x::AbstractMatrix) where {P}
    Hinv = plan.Hinv
    dims = plan.dims
    n1, n2, n3 = dims
    nodes = plan.nodes
    weights = plan.weights
    dweights = plan.dweights
    @inbounds for i in 1:plan.N
        x1 = x[1, i]; x2 = x[2, i]; x3 = x[3, i]
        s1 = Hinv[1, 1] * x1 + Hinv[1, 2] * x2 + Hinv[1, 3] * x3
        s2 = Hinv[2, 1] * x1 + Hinv[2, 2] * x2 + Hinv[2, 3] * x3
        s3 = Hinv[3, 1] * x1 + Hinv[3, 2] * x2 + Hinv[3, 3] * x3
        for d in 1:3
            sd = d == 1 ? s1 : (d == 2 ? s2 : s3)
            nd = d == 1 ? n1 : (d == 2 ? n2 : n3)
            u = sd * nd
            uf = floor(u)
            f = unsafe_trunc(Int, uf)
            t = u - uf
            node_idx = mod(f - P ÷ 2 + 1, nd) + 1
            ws, dws = P == 4 ? _bspline_p4(t) : _bspline_p6(t)
            for k in 1:P
                nodes[k, d, i] = node_idx
                weights[k, d, i] = ws[k]
                dweights[k, d, i] = dws[k] * nd
                node_idx = node_idx == nd ? 1 : node_idx + 1
            end
        end
    end
    return nothing
end

function _rebuild_sorted_cells!(plan::PMEPlan, x::AbstractMatrix, qq::AbstractVector{Float64})
    counts = plan.cell_counts
    offsets = plan.cell_offsets
    cell_id = plan.cell_id
    perm = plan.perm
    sx = plan.sxbuf
    sq = plan.sqbuf
    fill!(counts, 0)

    nc = plan.cell_n
    nc1, nc2, nc3 = nc
    nc12 = nc1 * nc2
    N = plan.N
    if plan.is_triclinic
        Hinv = plan.Hinv
        @inbounds for i in 1:N
            x1 = x[1, i]; x2 = x[2, i]; x3 = x[3, i]
            s1 = Hinv[1, 1] * x1 + Hinv[1, 2] * x2 + Hinv[1, 3] * x3
            s2 = Hinv[2, 1] * x1 + Hinv[2, 2] * x2 + Hinv[2, 3] * x3
            s3 = Hinv[3, 1] * x1 + Hinv[3, 2] * x2 + Hinv[3, 3] * x3
            c1 = mod(floor(Int, s1 * nc1), nc1) + 1
            c2 = mod(floor(Int, s2 * nc2), nc2) + 1
            c3 = mod(floor(Int, s3 * nc3), nc3) + 1
            cid = c1 + nc1 * (c2 - 1) + nc12 * (c3 - 1)
            cell_id[i] = cid
            counts[cid] += 1
        end
    else
        iL1 = 1.0 / plan.H[1, 1]
        iL2 = 1.0 / plan.H[2, 2]
        iL3 = 1.0 / plan.H[3, 3]
        @inbounds for i in 1:N
            c1 = mod(floor(Int, x[1, i] * iL1 * nc1), nc1) + 1
            c2 = mod(floor(Int, x[2, i] * iL2 * nc2), nc2) + 1
            c3 = mod(floor(Int, x[3, i] * iL3 * nc3), nc3) + 1
            cid = c1 + nc1 * (c2 - 1) + nc12 * (c3 - 1)
            cell_id[i] = cid
            counts[cid] += 1
        end
    end

    ncell = length(counts)
    @inbounds begin
        acc = 0
        offsets[1] = 0
        for c in 1:ncell
            cnt = counts[c]
            counts[c] = acc
            acc += cnt
            offsets[c + 1] = acc
        end
        for i in 1:N
            cid = cell_id[i]
            pos = counts[cid] + 1
            counts[cid] = pos
            perm[pos] = i
            sx[1, pos] = x[1, i]
            sx[2, pos] = x[2, i]
            sx[3, pos] = x[3, i]
            sq[pos] = qq[i]
        end
    end
    return nothing
end

function _reciprocal_potential!(plan::PMEPlan{P}, qq::AbstractVector{Float64},
                                G::Union{Nothing,AbstractMatrix{Float64}}) where {P}
    work_real = plan.work_real
    work_k = plan.work_k
    fill!(work_real, 0.0)

    nodes = plan.nodes
    weights = plan.weights
    @inbounds for i in 1:plan.N
        qi = qq[i]
        qi == 0.0 && continue
        if P == 4
            n1_1 = nodes[1, 1, i]; n1_2 = nodes[2, 1, i]; n1_3 = nodes[3, 1, i]; n1_4 = nodes[4, 1, i]
            w1_1 = weights[1, 1, i]; w1_2 = weights[2, 1, i]; w1_3 = weights[3, 1, i]; w1_4 = weights[4, 1, i]
            for k3 in 1:4
                n3 = nodes[k3, 3, i]
                q3 = qi * weights[k3, 3, i]
                for k2 in 1:4
                    n2 = nodes[k2, 2, i]
                    q23 = q3 * weights[k2, 2, i]
                    work_real[n1_1, n2, n3] += q23 * w1_1
                    work_real[n1_2, n2, n3] += q23 * w1_2
                    work_real[n1_3, n2, n3] += q23 * w1_3
                    work_real[n1_4, n2, n3] += q23 * w1_4
                end
            end
        else
            for k3 in 1:P
                n3 = nodes[k3, 3, i]
                q3 = qi * weights[k3, 3, i]
                for k2 in 1:P
                    n2 = nodes[k2, 2, i]
                    q23 = q3 * weights[k2, 2, i]
                    for k1 in 1:P
                        work_real[nodes[k1, 1, i], n2, n3] += q23 * weights[k1, 1, i]
                    end
                end
            end
        end
    end

    FFTW.unsafe_execute!(plan.rfftplan, work_real, work_k)

    p = plan.prefactor
    a2_inv = 1.0 / (2.0 * plan.alpha * plan.alpha)
    Hinv = plan.Hinv
    influence = plan.influence

    m_tens11 = 0.0
    m_tens12 = 0.0
    m_tens13 = 0.0
    m_tens22 = 0.0
    m_tens23 = 0.0
    m_tens33 = 0.0

    n1, n2, n3 = plan.dims
    nk1 = n1 ÷ 2 + 1
    even_n1 = iseven(n1)
    @inbounds for k in 1:n3, j in 1:n2, i in 1:nk1
        inf_val = influence[i, j, k]
        Q_val = work_k[i, j, k]
        work_k[i, j, k] = Q_val * inf_val
        if G !== nothing && inf_val != 0.0
            m1 = i - 1
            m2 = _mode(j, n2)
            m3 = _mode(k, n3)
            kx = 2pi * (Hinv[1, 1] * m1 + Hinv[2, 1] * m2 + Hinv[3, 1] * m3)
            ky = 2pi * (Hinv[1, 2] * m1 + Hinv[2, 2] * m2 + Hinv[3, 2] * m3)
            kz = 2pi * (Hinv[1, 3] * m1 + Hinv[2, 3] * m2 + Hinv[3, 3] * m3)
            k2 = kx * kx + ky * ky + kz * kz
            Q2 = abs2(Q_val)
            mult = (i == 1 || (even_n1 && i == nk1)) ? 1.0 : 2.0
            wm = 0.5 * p * mult * Q2 * inf_val
            c_k = wm * (2.0 / k2 + a2_inv)
            m_tens11 += -wm + c_k * kx * kx
            m_tens12 += c_k * kx * ky
            m_tens13 += c_k * kx * kz
            m_tens22 += -wm + c_k * ky * ky
            m_tens23 += c_k * ky * kz
            m_tens33 += -wm + c_k * kz * kz
        end
    end

    FFTW.unsafe_execute!(plan.brfftplan, work_k, work_real)

    if G !== nothing
        # G_recip += M_tens * transpose(Hinv)
        # where M is symmetric: M[1,2] = M[2,1], etc.
        M11 = m_tens11; M12 = m_tens12; M13 = m_tens13
        M21 = m_tens12; M22 = m_tens22; M23 = m_tens23
        M31 = m_tens13; M32 = m_tens23; M33 = m_tens33

        G[1, 1] += M11 * Hinv[1, 1] + M12 * Hinv[1, 2] + M13 * Hinv[1, 3]
        G[1, 2] += M11 * Hinv[2, 1] + M12 * Hinv[2, 2] + M13 * Hinv[2, 3]
        G[1, 3] += M11 * Hinv[3, 1] + M12 * Hinv[3, 2] + M13 * Hinv[3, 3]

        G[2, 1] += M21 * Hinv[1, 1] + M22 * Hinv[1, 2] + M23 * Hinv[1, 3]
        G[2, 2] += M21 * Hinv[2, 1] + M22 * Hinv[2, 2] + M23 * Hinv[2, 3]
        G[2, 3] += M21 * Hinv[3, 1] + M22 * Hinv[3, 2] + M23 * Hinv[3, 3]

        G[3, 1] += M31 * Hinv[1, 1] + M32 * Hinv[1, 2] + M33 * Hinv[1, 3]
        G[3, 2] += M31 * Hinv[2, 1] + M32 * Hinv[2, 2] + M33 * Hinv[2, 3]
        G[3, 3] += M31 * Hinv[3, 1] + M32 * Hinv[3, 2] + M33 * Hinv[3, 3]
    end
    return nothing
end

function _real_space_ortho!(F, grad, G, plan::PMEPlan,
                            a::Float64, p::Float64, rc2::Float64)
    U = 0.0
    plan.N <= 1 && return U
    sx = plan.sxbuf
    sq = plan.sqbuf
    sf = plan.sfbuf
    perm = plan.perm
    offsets = plan.cell_offsets
    H = plan.H
    L1 = H[1, 1]; L2 = H[2, 2]; L3 = H[3, 3]
    iL1 = 1.0 / L1; iL2 = 1.0 / L2; iL3 = 1.0 / L3
    two_a_over_sqrtpi = 2.0 * a / _SQRTPI

    F !== nothing && fill!(sf, 0.0)

    @inbounds for (c, t) in plan.cell_pairs
        i0 = offsets[c] + 1
        i1 = offsets[c + 1]
        i0 > i1 && continue
        if c == t
            for si in i0:(i1 - 1)
                xi = sx[1, si]; yi = sx[2, si]; zi = sx[3, si]
                qi = p * sq[si]
                fxi = 0.0; fyi = 0.0; fzi = 0.0
                for sj in (si + 1):i1
                    dx = xi - sx[1, sj]
                    dy = yi - sx[2, sj]
                    dz = zi - sx[3, sj]
                    dx -= L1 * round(dx * iL1)
                    dy -= L2 * round(dy * iL2)
                    dz -= L3 * round(dz * iL3)
                    r2 = dx * dx + dy * dy + dz * dz
                    if r2 < rc2
                        r2 > 0.0 || throw(ArgumentError("coincident charges are singular"))
                        inv_r2 = 1.0 / r2
                        r = sqrt(r2)
                        inv_r = 1.0 / r
                        z = a * r
                        er = _erfc(z)
                        er_over_r = er * inv_r
                        qij = qi * sq[sj]
                        U += qij * er_over_r
                        cff = qij * (-two_a_over_sqrtpi * exp(-z * z) - er_over_r) * inv_r2
                        if F !== nothing
                            fx = cff * dx; fy = cff * dy; fz = cff * dz
                            fxi -= fx; fyi -= fy; fzi -= fz
                            sf[1, sj] += fx; sf[2, sj] += fy; sf[3, sj] += fz
                        end
                        if grad !== nothing
                            v = p * er_over_r
                            ii = perm[si]; jj = perm[sj]
                            grad[ii] += sq[sj] * v
                            grad[jj] += sq[si] * v
                        end
                        if G !== nothing
                            w1 = dx * iL1; w2 = dy * iL2; w3 = dz * iL3
                            G[1, 1] += cff * dx * w1; G[1, 2] += cff * dx * w2; G[1, 3] += cff * dx * w3
                            G[2, 1] += cff * dy * w1; G[2, 2] += cff * dy * w2; G[2, 3] += cff * dy * w3
                            G[3, 1] += cff * dz * w1; G[3, 2] += cff * dz * w2; G[3, 3] += cff * dz * w3
                        end
                    end
                end
                if F !== nothing
                    sf[1, si] += fxi; sf[2, si] += fyi; sf[3, si] += fzi
                end
            end
        else
            j0 = offsets[t] + 1
            j1 = offsets[t + 1]
            j0 > j1 && continue
            for si in i0:i1
                xi = sx[1, si]; yi = sx[2, si]; zi = sx[3, si]
                qi = p * sq[si]
                fxi = 0.0; fyi = 0.0; fzi = 0.0
                for sj in j0:j1
                    dx = xi - sx[1, sj]
                    dy = yi - sx[2, sj]
                    dz = zi - sx[3, sj]
                    dx -= L1 * round(dx * iL1)
                    dy -= L2 * round(dy * iL2)
                    dz -= L3 * round(dz * iL3)
                    r2 = dx * dx + dy * dy + dz * dz
                    if r2 < rc2
                        r2 > 0.0 || throw(ArgumentError("coincident charges are singular"))
                        inv_r2 = 1.0 / r2
                        r = sqrt(r2)
                        inv_r = 1.0 / r
                        z = a * r
                        er = _erfc(z)
                        er_over_r = er * inv_r
                        qij = qi * sq[sj]
                        U += qij * er_over_r
                        cff = qij * (-two_a_over_sqrtpi * exp(-z * z) - er_over_r) * inv_r2
                        if F !== nothing
                            fx = cff * dx; fy = cff * dy; fz = cff * dz
                            fxi -= fx; fyi -= fy; fzi -= fz
                            sf[1, sj] += fx; sf[2, sj] += fy; sf[3, sj] += fz
                        end
                        if grad !== nothing
                            v = p * er_over_r
                            ii = perm[si]; jj = perm[sj]
                            grad[ii] += sq[sj] * v
                            grad[jj] += sq[si] * v
                        end
                        if G !== nothing
                            w1 = dx * iL1; w2 = dy * iL2; w3 = dz * iL3
                            G[1, 1] += cff * dx * w1; G[1, 2] += cff * dx * w2; G[1, 3] += cff * dx * w3
                            G[2, 1] += cff * dy * w1; G[2, 2] += cff * dy * w2; G[2, 3] += cff * dy * w3
                            G[3, 1] += cff * dz * w1; G[3, 2] += cff * dz * w2; G[3, 3] += cff * dz * w3
                        end
                    end
                end
                if F !== nothing
                    sf[1, si] += fxi; sf[2, si] += fyi; sf[3, si] += fzi
                end
            end
        end
    end

    if F !== nothing
        @inbounds for s in 1:plan.N
            idx = perm[s]
            F[1, idx] += sf[1, s]
            F[2, idx] += sf[2, s]
            F[3, idx] += sf[3, s]
        end
    end
    return U
end

function _real_space_triclinic!(F, grad, G, plan::PMEPlan,
                                a::Float64, p::Float64, rc2::Float64)
    U = 0.0
    plan.N <= 1 && return U
    sx = plan.sxbuf
    sq = plan.sqbuf
    sf = plan.sfbuf
    perm = plan.perm
    offsets = plan.cell_offsets
    two_a_over_sqrtpi = 2.0 * a / _SQRTPI

    F !== nothing && fill!(sf, 0.0)

    @inbounds for (c, t) in plan.cell_pairs
        i0 = offsets[c] + 1
        i1 = offsets[c + 1]
        i0 > i1 && continue
        if c == t
            for si in i0:(i1 - 1)
                xi = sx[1, si]; yi = sx[2, si]; zi = sx[3, si]
                qi = p * sq[si]
                fxi = 0.0; fyi = 0.0; fzi = 0.0
                for sj in (si + 1):i1
                    (dx, dy, dz), (w1, w2, w3), r2 = _triclinic_min_image(
                        plan, xi - sx[1, sj], yi - sx[2, sj], zi - sx[3, sj])
                    if r2 < rc2
                        r2 > 0.0 || throw(ArgumentError("coincident charges are singular"))
                        inv_r2 = 1.0 / r2
                        r = sqrt(r2)
                        inv_r = 1.0 / r
                        z = a * r
                        er = _erfc(z)
                        er_over_r = er * inv_r
                        qij = qi * sq[sj]
                        U += qij * er_over_r
                        cff = qij * (-two_a_over_sqrtpi * exp(-z * z) - er_over_r) * inv_r2
                        if F !== nothing
                            fx = cff * dx; fy = cff * dy; fz = cff * dz
                            fxi -= fx; fyi -= fy; fzi -= fz
                            sf[1, sj] += fx; sf[2, sj] += fy; sf[3, sj] += fz
                        end
                        if grad !== nothing
                            v = p * er_over_r
                            ii = perm[si]; jj = perm[sj]
                            grad[ii] += sq[sj] * v
                            grad[jj] += sq[si] * v
                        end
                        if G !== nothing
                            G[1, 1] += cff * dx * w1; G[1, 2] += cff * dx * w2; G[1, 3] += cff * dx * w3
                            G[2, 1] += cff * dy * w1; G[2, 2] += cff * dy * w2; G[2, 3] += cff * dy * w3
                            G[3, 1] += cff * dz * w1; G[3, 2] += cff * dz * w2; G[3, 3] += cff * dz * w3
                        end
                    end
                end
                if F !== nothing
                    sf[1, si] += fxi; sf[2, si] += fyi; sf[3, si] += fzi
                end
            end
        else
            j0 = offsets[t] + 1
            j1 = offsets[t + 1]
            j0 > j1 && continue
            for si in i0:i1
                xi = sx[1, si]; yi = sx[2, si]; zi = sx[3, si]
                qi = p * sq[si]
                fxi = 0.0; fyi = 0.0; fzi = 0.0
                for sj in j0:j1
                    (dx, dy, dz), (w1, w2, w3), r2 = _triclinic_min_image(
                        plan, xi - sx[1, sj], yi - sx[2, sj], zi - sx[3, sj])
                    if r2 < rc2
                        r2 > 0.0 || throw(ArgumentError("coincident charges are singular"))
                        inv_r2 = 1.0 / r2
                        r = sqrt(r2)
                        inv_r = 1.0 / r
                        z = a * r
                        er = _erfc(z)
                        er_over_r = er * inv_r
                        qij = qi * sq[sj]
                        U += qij * er_over_r
                        cff = qij * (-two_a_over_sqrtpi * exp(-z * z) - er_over_r) * inv_r2
                        if F !== nothing
                            fx = cff * dx; fy = cff * dy; fz = cff * dz
                            fxi -= fx; fyi -= fy; fzi -= fz
                            sf[1, sj] += fx; sf[2, sj] += fy; sf[3, sj] += fz
                        end
                        if grad !== nothing
                            v = p * er_over_r
                            ii = perm[si]; jj = perm[sj]
                            grad[ii] += sq[sj] * v
                            grad[jj] += sq[si] * v
                        end
                        if G !== nothing
                            G[1, 1] += cff * dx * w1; G[1, 2] += cff * dx * w2; G[1, 3] += cff * dx * w3
                            G[2, 1] += cff * dy * w1; G[2, 2] += cff * dy * w2; G[2, 3] += cff * dy * w3
                            G[3, 1] += cff * dz * w1; G[3, 2] += cff * dz * w2; G[3, 3] += cff * dz * w3
                        end
                    end
                end
                if F !== nothing
                    sf[1, si] += fxi; sf[2, si] += fyi; sf[3, si] += fzi
                end
            end
        end
    end

    if F !== nothing
        @inbounds for s in 1:plan.N
            idx = perm[s]
            F[1, idx] += sf[1, s]
            F[2, idx] += sf[2, s]
            F[3, idx] += sf[3, s]
        end
    end
    return U
end

function _pme_core!(F, grad, G, plan::PMEPlan{P}, q::AbstractMatrix,
                    charges::AbstractVector;
                    corrections::PairCorrections=_EMPTY_CORRECTIONS,
                    accumulate::Bool = false,
                    accumulate_G::Bool = false) where {P}
    _validate_eval(plan, q, charges, F, grad, G, corrections;
                   accumulate=accumulate, accumulate_G=accumulate_G)

    # Reuse plan-owned scratch so a warm call does not allocate for the input
    # conversion. The plan is documented as not thread-safe.
    x = plan.xbuf
    qq = plan.qqbuf
    copyto!(x, q)
    copyto!(qq, charges)
    N = plan.N
    H = plan.H
    Hinv = plan.Hinv
    p = plan.prefactor
    a = plan.alpha
    rc = plan.rcut
    V = plan.volume
    is_triclinic = plan.is_triclinic

    _prepare_stencils!(plan, x)
    _reciprocal_potential!(plan, qq, G)
    work_real = plan.work_real
    nodes = plan.nodes
    weights = plan.weights
    dweights = plan.dweights

    # Reciprocal energy, forces, and charge gradient.
    U = 0.0
    @inbounds for i in 1:N
        phi = 0.0
        g1 = 0.0
        g2 = 0.0
        g3 = 0.0
        if P == 4
            n1_1 = nodes[1, 1, i]; n1_2 = nodes[2, 1, i]; n1_3 = nodes[3, 1, i]; n1_4 = nodes[4, 1, i]
            w1_1 = weights[1, 1, i]; w1_2 = weights[2, 1, i]; w1_3 = weights[3, 1, i]; w1_4 = weights[4, 1, i]
            dw1_1 = dweights[1, 1, i]; dw1_2 = dweights[2, 1, i]; dw1_3 = dweights[3, 1, i]; dw1_4 = dweights[4, 1, i]
            for k3 in 1:4
                n3 = nodes[k3, 3, i]; w3 = weights[k3, 3, i]; dw3 = dweights[k3, 3, i]
                p2 = 0.0; g1_2 = 0.0; g2_2 = 0.0
                for k2 in 1:4
                    n2 = nodes[k2, 2, i]; w2 = weights[k2, 2, i]; dw2 = dweights[k2, 2, i]
                    v1 = work_real[n1_1, n2, n3]
                    v2 = work_real[n1_2, n2, n3]
                    v3 = work_real[n1_3, n2, n3]
                    v4 = work_real[n1_4, n2, n3]
                    s0 = v1 * w1_1 + v2 * w1_2 + v3 * w1_3 + v4 * w1_4
                    p2 += s0 * w2
                    if F !== nothing
                        sd1 = v1 * dw1_1 + v2 * dw1_2 + v3 * dw1_3 + v4 * dw1_4
                        g1_2 += sd1 * w2
                        g2_2 += s0 * dw2
                    end
                end
                phi += p2 * w3
                if F !== nothing
                    g1 += g1_2 * w3
                    g2 += g2_2 * w3
                    g3 += p2 * dw3
                end
            end
        else
            for k3 in 1:P
                n3 = nodes[k3, 3, i]; w3 = weights[k3, 3, i]; dw3 = dweights[k3, 3, i]
                p2 = 0.0; g1_2 = 0.0; g2_2 = 0.0
                for k2 in 1:P
                    n2 = nodes[k2, 2, i]; w2 = weights[k2, 2, i]; dw2 = dweights[k2, 2, i]
                    s0 = 0.0; sd1 = 0.0
                    for k1 in 1:P
                        v = work_real[nodes[k1, 1, i], n2, n3]
                        s0 += v * weights[k1, 1, i]
                        if F !== nothing
                            sd1 += v * dweights[k1, 1, i]
                        end
                    end
                    p2 += s0 * w2
                    if F !== nothing
                        g1_2 += sd1 * w2
                        g2_2 += s0 * dw2
                    end
                end
                phi += p2 * w3
                if F !== nothing
                    g1 += g1_2 * w3
                    g2 += g2_2 * w3
                    g3 += p2 * dw3
                end
            end
        end
        U += 0.5 * p * qq[i] * phi
        grad !== nothing && (grad[i] += p * phi)
        if F !== nothing
            gr1 = Hinv[1, 1] * g1 + Hinv[2, 1] * g2 + Hinv[3, 1] * g3
            gr2 = Hinv[1, 2] * g1 + Hinv[2, 2] * g2 + Hinv[3, 2] * g3
            gr3 = Hinv[1, 3] * g1 + Hinv[2, 3] * g2 + Hinv[3, 3] * g3
            F[1, i] -= p * qq[i] * gr1
            F[2, i] -= p * qq[i] * gr2
            F[3, i] -= p * qq[i] * gr3
        end
    end

    # Real-space erfc sum
    rc2 = rc * rc
    _rebuild_sorted_cells!(plan, x, qq)
    if is_triclinic
        U += _real_space_triclinic!(F, grad, G, plan, a, p, rc2)
    else
        U += _real_space_ortho!(F, grad, G, plan, a, p, rc2)
    end

    Q = sum(qq)
    U -= p * a / _SQRTPI * sum(abs2, qq)
    U_bg = -p * pi * Q * Q / (2 * a * a * V)
    U += U_bg

    if grad !== nothing
        self_coeff = 2 * p * a / _SQRTPI
        bg_coeff = p * pi * Q / (a * a * V)
        @inbounds for i in 1:N
            grad[i] -= self_coeff * qq[i] + bg_coeff
        end
    end

    if G !== nothing
        for a_idx in 1:3, b_idx in 1:3
            G[a_idx, b_idx] -= U_bg * Hinv[b_idx, a_idx]
        end
    end

    # Point-Coulomb pair corrections (same convention as Electrostatics).
    if !isempty(corrections.weights)
        if is_triclinic
            for ((i, j), s) in corrections.weights
                (dx, dy, dz), (w1, w2, w3), r2 = _triclinic_min_image(plan, x[1, i] - x[1, j], x[2, i] - x[2, j], x[3, i] - x[3, j])
                r2 > 0.0 || throw(ArgumentError("coincident corrected charges are singular"))
                r = sqrt(r2)
                c = p * (s - 1) * qq[i] * qq[j]
                U += c / r
                cff = c / (r2 * r)
                if F !== nothing
                    F[1, i] += cff * dx
                    F[2, i] += cff * dy
                    F[3, i] += cff * dz
                    F[1, j] -= cff * dx
                    F[2, j] -= cff * dy
                    F[3, j] -= cff * dz
                end
                if grad !== nothing
                    v = p * (s - 1) / r
                    grad[i] += v * qq[j]
                    grad[j] += v * qq[i]
                end
                if G !== nothing
                    G[1, 1] -= cff * dx * w1; G[1, 2] -= cff * dx * w2; G[1, 3] -= cff * dx * w3
                    G[2, 1] -= cff * dy * w1; G[2, 2] -= cff * dy * w2; G[2, 3] -= cff * dy * w3
                    G[3, 1] -= cff * dz * w1; G[3, 2] -= cff * dz * w2; G[3, 3] -= cff * dz * w3
                end
            end
        else
            L = (H[1, 1], H[2, 2], H[3, 3])
            iL1 = 1.0 / L[1]; iL2 = 1.0 / L[2]; iL3 = 1.0 / L[3]
            for ((i, j), s) in corrections.weights
                dx = x[1, i] - x[1, j]
                dy = x[2, i] - x[2, j]
                dz = x[3, i] - x[3, j]
                dx -= L[1] * round(dx * iL1)
                dy -= L[2] * round(dy * iL2)
                dz -= L[3] * round(dz * iL3)
                r2 = dx * dx + dy * dy + dz * dz
                r2 > 0.0 || throw(ArgumentError("coincident corrected charges are singular"))
                r = sqrt(r2)
                c = p * (s - 1) * qq[i] * qq[j]
                U += c / r
                cff = c / (r2 * r)
                if F !== nothing
                    F[1, i] += cff * dx
                    F[2, i] += cff * dy
                    F[3, i] += cff * dz
                    F[1, j] -= cff * dx
                    F[2, j] -= cff * dy
                    F[3, j] -= cff * dz
                end
                if grad !== nothing
                    v = p * (s - 1) / r
                    grad[i] += v * qq[j]
                    grad[j] += v * qq[i]
                end
                if G !== nothing
                    w1 = dx * iL1; w2 = dy * iL2; w3 = dz * iL3
                    G[1, 1] -= cff * dx * w1; G[1, 2] -= cff * dx * w2; G[1, 3] -= cff * dx * w3
                    G[2, 1] -= cff * dy * w1; G[2, 2] -= cff * dy * w2; G[2, 3] -= cff * dy * w3
                    G[3, 1] -= cff * dz * w1; G[3, 2] -= cff * dz * w2; G[3, 3] -= cff * dz * w3
                end
            end
        end
    end

    return U
end

# Backwards-compatible dispatcher without G
_pme_core!(F, grad, plan::PMEPlan, q::AbstractMatrix, charges::AbstractVector; kwargs...) =
    _pme_core!(F, grad, nothing, plan, q, charges; kwargs...)

# ---------------------------------------------------------------------------
# Public evaluators
# ---------------------------------------------------------------------------

"""
    pme_energy(plan, q, charges; corrections=PairCorrections()) -> U

Evaluate the PME energy for coordinates `q` (3×N) and `charges` (length N) with
a prebuilt plan. The uniform neutralizing background and tin-foil boundary are
included.
"""
function pme_energy(plan::PMEPlan, q::AbstractMatrix, charges::AbstractVector;
                    corrections::PairCorrections=_EMPTY_CORRECTIONS)
    return _pme_core!(nothing, nothing, nothing, plan, q, charges; corrections=corrections)
end

"""
    pme_energy_forces!(F, plan, q, charges; corrections=PairCorrections()) -> U

Fill `F` (3×N) with `-∇U` and return the energy. The plane-wave forces are the
analytic derivative of the discrete PME energy; they are not `i k` forces.
"""
function pme_energy_forces!(F::AbstractMatrix{Float64}, plan::PMEPlan, q::AbstractMatrix,
                            charges::AbstractVector;
                            corrections::PairCorrections=_EMPTY_CORRECTIONS)
    return _pme_core!(F, nothing, nothing, plan, q, charges; corrections=corrections)
end

"""
    pme_energy_forces(plan, q, charges; corrections=PairCorrections()) -> (U, F)

Convenience wrapper around [`pme_energy_forces!`](@ref).
"""
function pme_energy_forces(plan::PMEPlan, q::AbstractMatrix, charges::AbstractVector;
                           corrections::PairCorrections=_EMPTY_CORRECTIONS)
    F = zeros(Float64, 3, plan.N)
    U = pme_energy_forces!(F, plan, q, charges; corrections=corrections)
    return U, F
end

"""
    pme_charge_gradient!(out, plan, q, charges; corrections=PairCorrections()) -> out

Fill `out` with the analytic derivative of the PME energy with respect to each
charge. `out` is returned for chaining. The convention matches
`Electrostatics.ewald_charge_gradient!`, including self, background, and pair
correction terms.
"""
function pme_charge_gradient!(out::AbstractVector{Float64}, plan::PMEPlan, q::AbstractMatrix,
                              charges::AbstractVector;
                              corrections::PairCorrections=_EMPTY_CORRECTIONS)
    _pme_core!(nothing, out, nothing, plan, q, charges; corrections=corrections)
    return out
end

"""
    pme_box_derivative!(G, plan, q, charges; corrections=PairCorrections(), accumulate=false) -> G

Fill `G` (3×3) with the analytic fixed-fractional box derivative `∂U/∂h` and return `G`.
"""
function pme_box_derivative!(G::AbstractMatrix{Float64}, plan::PMEPlan, q::AbstractMatrix,
                             charges::AbstractVector;
                             corrections::PairCorrections=_EMPTY_CORRECTIONS,
                             accumulate::Bool=false)
    _pme_core!(nothing, nothing, G, plan, q, charges;
               corrections=corrections, accumulate_G=accumulate)
    return G
end

"""
    pme_box_derivative(plan, q, charges; corrections=PairCorrections()) -> G

Return a 3×3 matrix with the analytic fixed-fractional box derivative `∂U/∂h`.
"""
function pme_box_derivative(plan::PMEPlan, q::AbstractMatrix, charges::AbstractVector;
                            corrections::PairCorrections=_EMPTY_CORRECTIONS)
    G = zeros(Float64, 3, 3)
    pme_box_derivative!(G, plan, q, charges; corrections=corrections)
    return G
end

"""
    pme_box_derivative(q, charges, box; alpha, rcut, grid=nothing, order=4, prefactor=1.0, corrections=PairCorrections()) -> G

One-shot calculation of the analytic fixed-fractional box derivative `∂U/∂h`.
"""
function pme_box_derivative(q::AbstractMatrix, charges::AbstractVector, box;
                            alpha::Real, rcut::Real, grid=nothing, order::Integer=4,
                            prefactor::Real=1.0, corrections::PairCorrections=_EMPTY_CORRECTIONS)
    plan = PMEPlan(q, box; alpha=alpha, rcut=rcut, grid=grid, order=order, prefactor=prefactor)
    return pme_box_derivative(plan, q, charges; corrections=corrections)
end

"""
    pme_energy_forces_box_derivative!(F, G, plan, q, charges; corrections=PairCorrections(), accumulate_F=false, accumulate_G=false) -> U

Fill `F` (3×N) with forces, `G` (3×3) with box derivative `∂U/∂h`, and return the energy `U`.
"""
function pme_energy_forces_box_derivative!(F::AbstractMatrix{Float64}, G::AbstractMatrix{Float64},
                                          plan::PMEPlan, q::AbstractMatrix, charges::AbstractVector;
                                          corrections::PairCorrections=_EMPTY_CORRECTIONS,
                                          accumulate_F::Bool=false, accumulate_G::Bool=false)
    return _pme_core!(F, nothing, G, plan, q, charges;
                      corrections=corrections, accumulate=accumulate_F, accumulate_G=accumulate_G)
end

"""
    pme_energy_forces_box_derivative(plan, q, charges; corrections=PairCorrections()) -> (U, F, G)

Evaluate energy `U`, forces `F` (3×N), and box derivative `G` (3×3) together.
"""
function pme_energy_forces_box_derivative(plan::PMEPlan, q::AbstractMatrix, charges::AbstractVector;
                                         corrections::PairCorrections=_EMPTY_CORRECTIONS)
    F = zeros(Float64, 3, plan.N)
    G = zeros(Float64, 3, 3)
    U = pme_energy_forces_box_derivative!(F, G, plan, q, charges; corrections=corrections)
    return U, F, G
end

"""
    pme_evaluate(q, charges, box; alpha, rcut, grid=nothing, order=4,
                 prefactor=1.0, corrections=PairCorrections()) -> (U, F)

Build a one-shot plan and return `(energy, forces)`. For repeated evaluations,
build a [`PMEPlan`](@ref) once and call [`pme_energy_forces`](@ref) instead.
"""
function pme_evaluate(q::AbstractMatrix, charges::AbstractVector, box;
                      alpha::Real, rcut::Real, grid=nothing, order::Integer=4,
                      prefactor::Real=1.0,
                      corrections::PairCorrections=_EMPTY_CORRECTIONS)
    plan = PMEPlan(q, box; alpha=alpha, rcut=rcut, grid=grid, order=order,
                   prefactor=prefactor)
    return pme_energy_forces(plan, q, charges; corrections=corrections)
end

# ---------------------------------------------------------------------------
# Error budget & parameter selection helpers
# ---------------------------------------------------------------------------

"""
    pme_error_estimate(box, alpha, rcut; grid=nothing, order=4, N=1, sum_q2=1.0) -> NamedTuple

Estimate the root-mean-square force error budget for PME partitioned into
real-space cutoff truncation error and reciprocal-space B-spline mesh interpolation error.

Based on the Kolafa-Perram real-space cutoff bound and the Petersen / Deserno-Holm
mesh interpolation error model:
- `real_error`: estimated RMS force error from real-space truncation at `rcut`
- `recip_error`: estimated RMS force error from reciprocal B-spline mesh discretization
- `total_error`: combined error estimate `sqrt(real_error^2 + recip_error^2)`
- `mesh_spacing`: maximum grid spacing
- `grid`: grid dimensions used
"""
function pme_error_estimate(box, alpha::Real, rcut::Real;
                            grid=nothing, order::Integer=4, N::Integer=1, sum_q2::Real=1.0)
    order isa Integer && (order == 4 || order == 6) ||
        throw(ArgumentError("spline order must be 4 or 6"))
    P = Int(order)
    pbox = _parse_box(box)
    a = Float64(alpha)
    rc = Float64(rcut)
    a > 0 || throw(ArgumentError("alpha must be positive"))
    rc > 0 || throw(ArgumentError("rcut must be positive"))
    V = pbox.volume
    Natoms = max(Int(N), 1)
    q_rms = sqrt(max(Float64(sum_q2), 1e-12) / Natoms)

    # Kolafa & Perram (1992) real-space RMS force error bound
    real_error = 2.0 * q_rms * q_rms * sqrt(rc / V) * exp(-a * a * rc * rc)

    # Reciprocal mesh error (Petersen 1995 / Deserno & Holm 1998)
    lengths = ntuple(i -> norm(view(pbox.H, :, i)), 3)
    dims = _grid_dims(grid, lengths, a, P)
    h_max = maximum(lengths[d] / dims[d] for d in 1:3)
    # Dimensionless factor (alpha * h_max / 2)^P / sqrt(P) scaled by field magnitude
    recip_error = (q_rms * q_rms / sqrt(rc * V)) * ((a * h_max / 2.0)^P / sqrt(P))
    total_error = sqrt(real_error^2 + recip_error^2)

    return (real_error=real_error, recip_error=recip_error, total_error=total_error,
            mesh_spacing=h_max, grid=dims, order=P)
end

"""
    pme_recommend_parameters(box; tol=1e-5, rcut=nothing, order=4) -> NamedTuple

Recommend PME parameters `(alpha, rcut, grid, order, estimated_error)` achieving
the target error tolerance `tol` for a given periodic `box`.
"""
function pme_recommend_parameters(box; tol::Real=1e-5, rcut=nothing, order::Integer=4)
    tol > 0 || throw(ArgumentError("tol must be positive"))
    (order == 4 || order == 6) || throw(ArgumentError("spline order must be 4 or 6"))
    P = Int(order)
    pbox = _parse_box(box)
    min_w = minimum(pbox.widths)

    rc = rcut === nothing ? 0.45 * min_w : Float64(rcut)
    (rc > 0 && rc < min_w / 2) || throw(ArgumentError("rcut must satisfy 0 < rcut < min_width/2"))

    # Choose alpha so real-space error factor exp(-alpha^2 * rc^2) <= tol
    alpha = sqrt(-log(min(Float64(tol), 0.5))) / rc

    # Choose mesh spacing h such that reciprocal error factor <= tol
    # (alpha * h / 2)^P / sqrt(P) <= tol  =>  h = (2 / alpha) * (tol * sqrt(P))^(1/P)
    h_target = (2.0 / alpha) * (Float64(tol) * sqrt(P))^(1.0 / P)
    lengths = ntuple(i -> norm(view(pbox.H, :, i)), 3)
    dims = ntuple(d -> begin
        n = ceil(Int, lengths[d] / h_target)
        n = max(n, 2 * P)
        nextpow(2, n)
    end, 3)

    est = pme_error_estimate(box, alpha, rc; grid=dims, order=P)
    return (alpha=alpha, rcut=rc, grid=dims, order=P, estimated_error=est.total_error)
end

"""
    pme_validate_budget(box; alpha::Real, rcut::Real, grid=nothing, order::Integer=4, tol::Real=1e-4) -> (Bool, NamedTuple)

Validate whether the given PME parameter combination satisfies the target error budget `tol`.
"""
function pme_validate_budget(box; alpha::Real, rcut::Real, grid=nothing, order::Integer=4, tol::Real=1e-4)
    est = pme_error_estimate(box, alpha, rcut; grid=grid, order=order)
    return (est.total_error <= tol, est)
end

"""
    evaluate_pme(q, charges, box; kwargs...)

Alias for [`pme_evaluate`](@ref), named by analogy with
`Electrostatics.evaluate_ewald`.
"""
evaluate_pme(args...; kwargs...) = pme_evaluate(args...; kwargs...)

if _HAS_FACTOR_PROTOCOL
    """Cached PME factor adapter for `Factors`/`Dynamics`, with a fixed periodic box and atom count."""
    mutable struct PMEFactor{B<:Union{Types.OrthorhombicBox, Types.TriclinicBox}, PL<:PMEPlan} <: Factors.AbstractFactor
        N::Int
        charges::Vector{Float64}
        box::B
        plan::PL
        corrections::PairCorrections
    end

    function PMEFactor(q::AbstractMatrix{<:Real}, charges::AbstractVector{<:Real},
                       box::Union{Types.OrthorhombicBox, Types.TriclinicBox};
                       alpha::Real, rcut::Real,
                       grid=nothing, order::Integer=4, prefactor::Real=1.0,
                       corrections::PairCorrections=_EMPTY_CORRECTIONS)
        size(q, 1) == 3 || throw(ArgumentError("q must have shape 3×N"))
        length(charges) == size(q, 2) || throw(ArgumentError("charge count must match coordinates"))
        all(isfinite, charges) || throw(ArgumentError("charges must be finite"))
        box_arg = box isa Types.OrthorhombicBox ? collect(box.L) : box
        plan = PMEPlan(q, box_arg; alpha=alpha, rcut=rcut, grid=grid,
                       order=order, prefactor=prefactor)
        all(i <= plan.N && j <= plan.N for ((i, j), _) in corrections.weights) ||
            throw(ArgumentError("correction index exceeds particle count"))
        PMEFactor(plan.N, Float64.(charges), box, plan, corrections)
    end

    function PMEFactor(q::AbstractMatrix{<:Real}, charges::AbstractVector{<:Real},
                       box::AbstractMatrix{<:Real}; kwargs...)
        return PMEFactor(q, charges, Types.TriclinicBox(box); kwargs...)
    end

    function PMEFactor(::AbstractMatrix{<:Real}, ::AbstractVector{<:Real}, box;
                       kwargs...)
        throw(ArgumentError("PMEFactor requires a periodic box (OrthorhombicBox, TriclinicBox, or 3×3 matrix); got $(typeof(box))"))
    end

    Factors.factor_support_size(f::PMEFactor) = f.N
    Factors.read_support(f::PMEFactor) = Tuple(1:f.N)
    Factors.is_local(::PMEFactor) = false
    Factors.force_consistent(::PMEFactor) = true

    function _validate_factor(f::PMEFactor, q::AbstractMatrix, ctx::Factors.Context)
        size(q) == (3, f.N) || throw(ArgumentError("q must be 3×$(f.N)"))
        ctx.N == f.N || throw(ArgumentError("context atom count does not match PME factor"))
        if f.box isa Types.OrthorhombicBox
            ctx.box isa Types.OrthorhombicBox || throw(Types.UnsupportedGeometry(
                "PMEFactor requires an orthorhombic periodic context box; got $(typeof(ctx.box))"))
            ctx.box.L == f.box.L || throw(ArgumentError("context box differs from PME factor box"))
        else
            ctx.box isa Types.TriclinicBox || throw(Types.UnsupportedGeometry(
                "PMEFactor requires a triclinic periodic context box; got $(typeof(ctx.box))"))
            ctx.box.H == f.box.H || throw(ArgumentError("context box differs from PME factor box"))
        end
        f.plan.N == f.N || throw(ArgumentError("PME plan capacity differs from factor atom count"))
        length(f.charges) == f.N || throw(ArgumentError("charge count no longer matches PME factor"))
        return nothing
    end

    function Factors.evaluate!(F::AbstractMatrix{Float64}, f::PMEFactor,
                               q::AbstractMatrix{Float64}, ctx::Factors.Context)
        size(F) == (3, f.N) || throw(ArgumentError("F must be 3×$(f.N)"))
        Base.mightalias(F, q) && throw(ArgumentError("force accumulator must not alias q"))
        _validate_factor(f, q, ctx)
        # Factor protocol: every factor ADDS to the shared accumulator. PME's
        # direct API zeroes by default; the factor must not erase bonded or
        # Lennard-Jones forces that were scattered before it.
        return _pme_core!(F, nothing, f.plan, q, f.charges;
                          corrections=f.corrections, accumulate=true)
    end

    "Evaluate PME factor energy with optional replacement charges."
    function pme_factor_energy(f::PMEFactor, q::AbstractMatrix{<:Real}, ctx::Factors.Context;
                               charges::AbstractVector{<:Real}=f.charges)
        _validate_factor(f, q, ctx)
        return pme_energy(f.plan, q, charges; corrections=f.corrections)
    end

    "Fill `out` with analytic PME factor derivatives by charge."
    function pme_factor_charge_gradient!(out::AbstractVector{Float64}, f::PMEFactor,
                                         q::AbstractMatrix{<:Real}, ctx::Factors.Context;
                                         charges::AbstractVector{<:Real}=f.charges)
        _validate_factor(f, q, ctx)
        return pme_charge_gradient!(out, f.plan, q, charges; corrections=f.corrections)
    end

    "Write or accumulate analytic PME factor box derivative `∂U/∂h` into `G` (3×3)."
    function pme_factor_box_derivative!(G::AbstractMatrix{Float64}, f::PMEFactor,
                                        q::AbstractMatrix{<:Real}, ctx::Factors.Context;
                                        charges::AbstractVector{<:Real}=f.charges,
                                        accumulate::Bool=false)
        _validate_factor(f, q, ctx)
        return pme_box_derivative!(G, f.plan, q, charges;
                                   corrections=f.corrections, accumulate=accumulate)
    end

    "Compute analytic PME factor box derivative `∂U/∂h` (3×3)."
    function pme_factor_box_derivative(f::PMEFactor, q::AbstractMatrix{<:Real},
                                       ctx::Factors.Context;
                                       charges::AbstractVector{<:Real}=f.charges)
        G = zeros(Float64, 3, 3)
        return pme_factor_box_derivative!(G, f, q, ctx; charges=charges, accumulate=false)
    end

    "Evaluate energy, forces, and box derivative together for a `PMEFactor`."
    function pme_factor_energy_forces_box_derivative!(F::AbstractMatrix{Float64},
                                                      G::AbstractMatrix{Float64},
                                                      f::PMEFactor,
                                                      q::AbstractMatrix{<:Real},
                                                      ctx::Factors.Context;
                                                      charges::AbstractVector{<:Real}=f.charges,
                                                      accumulate_F::Bool=false,
                                                      accumulate_G::Bool=false)
        _validate_factor(f, q, ctx)
        return pme_energy_forces_box_derivative!(F, G, f.plan, q, charges;
                                                 corrections=f.corrections,
                                                 accumulate_F=accumulate_F,
                                                 accumulate_G=accumulate_G)
    end
end

end # module
