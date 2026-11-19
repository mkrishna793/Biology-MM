# NIST Lennard-Jones reference validation.
#
# Reference: NIST Standard Reference Simulation Website,
#   "Lennard-Jones Fluid Reference Calculations: Cuboid Cell"
#   https://www.nist.gov/chemical-informatics-research-group/lennard-jones-fluid-reference-calculations
#   Sample configurations: https://www.nist.gov/document/ljsampleconfigurations-targz
#   (retrieved 2026-11-11; archive metadata.README dated 2013-01-08)
#
# What NIST provides, exactly:
#   V_LJ(r) = 4 eps [ (sigma/r)^12 - (sigma/r)^6 ],  reduced units eps = sigma = 1
#   U_pair  = sum_{i<j} V(r_ij)   ... but see the convention note below
#   W_pair  = -sum_{i<j} r_ij V'(r_ij)
#   U_LRC   = N * (1/2) * 4 pi rho * int_{rc}^inf r^2 V_LJ(r) dr
#   LFS:    V(r) -> V(r) - V(rc) - V'(rc)(r - rc) for r <= rc
#   PBC:    minimum image in all three Cartesian directions (cuboid cells only)
#
# Three truncation schemes per configuration: rc = 3 sigma (LRC), rc = 4 sigma
# (LRC), and rc = 3 sigma with the linear-force shift.
#
# SCOPE — read this before citing a pass:
#   * Matching NIST's tabulated 4-5 significant figures is evidence that this
#     code's minimum-image LJ energy, virial, and LRC agree with a third-party
#     reference implementation. It is NOT a certification, NOT a universal
#     tolerance, and NOT a statement about any other potential, cutoff, or
#     system.
#   * NIST's table gives 4-5 significant digits, so the achievable agreement is
#     bounded by that rounding (~5e-5 relative), not by this code.
#   * These are *static single-configuration* quantities. Nothing here validates
#     dynamics, integration error, or thermodynamics.

module NISTValidate

using FoundationCore
using FoundationCore.Types
using FoundationCore.Factors
using Printf, Statistics

export load_nist_config, nist_reference, evaluate_config, run_nist_validation,
       NIST_REFERENCE

"""
Reference table transcribed from the NIST page. Values are in reduced units and
reproduced to the precision NIST printed (4-5 significant figures). `l3`/`l4` are
*total* U_LRC for the configuration (NIST tabulates the per-particle form; the
total is what the energy comparison needs, and the transcription multiplies by N).
"""
const NIST_REFERENCE = Dict(
    1 => (N = 800, box = 10.0,
          u3 = -4.3515e3, w3 = -5.6867e2, l3 = -1.9849e2,
          u4 = -4.4675e3, w4 = -1.2639e3, l4 = -8.3769e1,
          uL = -3.8709e3, wL = 3.1754e2),
    2 => (N = 200, box = 8.0,
          u3 = -6.9000e2, w3 = -5.6846e2, l3 = -2.4230e1,
          u4 = -7.0460e2, w4 = -6.5599e2, l4 = -1.0226e1,
          uL = -6.2012e2, wL = -4.4533e2),
    3 => (N = 400, box = 10.0,
          u3 = -1.1467e3, w3 = -1.1649e3, l3 = -4.9622e1,
          u4 = -1.1754e3, w4 = -1.3371e3, l4 = -2.0942e1,
          uL = -1.0210e3, wL = -9.3578e2),
    4 => (N = 30, box = 8.0,
          u3 = -1.6790e1, w3 = -4.6249e1, l3 = -5.4517e-1,
          u4 = -1.7060e1, w4 = -4.7869e1, l4 = -2.3008e-1,
          uL = -1.5001e1, wL = -4.3096e1),
)

"Read a NIST sample configuration. Returns `(L, q::3xN)`."
function load_nist_config(path::AbstractString)
    lines = readlines(path)
    boxdims = parse.(Float64, split(strip(lines[1])))
    n = parse(Int, split(strip(lines[2]))[1])
    q = zeros(3, n)
    for i in 1:n
        parts = split(strip(lines[2 + i]))
        # column 1 is the atom id; columns 2-4 are x, y, z
        q[1, i] = parse(Float64, parts[2])
        q[2, i] = parse(Float64, parts[3])
        q[3, i] = parse(Float64, parts[4])
    end
    @assert length(boxdims) == 3
    @assert (boxdims[1] == boxdims[2] == boxdims[3]) "expected a cuboid cell"
    return (L = boxdims[1], q = q, N = n)
end

"Total U_LRC for an N-particle configuration at cutoff `rc`."
function total_lrc(N::Int, box::Float64, rc::Float64)
    rho = N / box^3
    per_particle = 0.5 * 4.0 * pi * rho * ((4.0 / 9.0) * rc^(-9) - (4.0 / 3.0) * rc^(-3))
    return per_particle * N
end

"""
Evaluate `U_pair`, `W_pair` for one configuration using the package's own factor
machinery (a `LennardJonesPair` per pair via minimum image), plus the analytic LRC.

`mode` is `:trunc` (with LRC) or `:lfs`.
"""
function evaluate_config(cfg; rc::Float64, mode::Symbol)
    q, L, N = cfg.q, cfg.L, cfg.N
    box = Types.OrthorhombicBox((L, L, L))
    ctx = Factors.Context(box, Types.REDUCED, N)
    # Enumerate the static reference directly. Some NIST fixtures use rc=L/2;
    # the production neighbor-list contract intentionally rejects that ambiguous
    # boundary, while the pairwise reference can evaluate it unambiguously.
    pairs = ((i, j) for i in 1:N-1 for j in i+1:N
             if begin
                 d = Types.unwrap_displacement(box, q, i, j)
                 d.x^2 + d.y^2 + d.z^2 <= rc^2
             end)
    Fv = zeros(3, N)
    U = 0.0
    W = 0.0
    for (i, j) in pairs
        U += Factors.evaluate!(Fv, Factors.LennardJonesPair(i, j, 1.0, 1.0; mode = mode, rc = rc), q, ctx)
        d = Types.unwrap_displacement(box, q, i, j)
        r = sqrt(d.x^2 + d.y^2 + d.z^2)
        # W_pair = -sum r V'(r). For :trunc, V' is the bare LJ derivative inside
        # the cutoff (NIST's definition), for :lfs it is shifted.
        dudr = Factors.lj_dudr(1.0, 1.0, r)
        if mode === :lfs
            dudr -= Factors.lj_dudr(1.0, 1.0, rc)
        end
        W += -r * dudr
    end
    return (U_pair = U, W_pair = W)
end

"""
Run the full comparison and print a table. Returns a results vector of named
tuples with the measured relative deviations.

Tolerance: `rel_tol = 5e-4`, chosen because NIST prints 4-5 significant figures,
so the *reference itself* is only good to ~5e-5 to 5e-4 relative. Anything looser
would hide a real bug; anything tighter would fail on NIST's rounding alone.
"""
function run_nist_validation(dir::AbstractString; rel_tol::Float64 = 5e-4, verbose::Bool = true)
    results = NamedTuple[]
    allpass = true
    for cfgid in sort(collect(keys(NIST_REFERENCE)))
        d = NIST_REFERENCE[cfgid]
        path = joinpath(dir, "lj_sample_config_periodic$(cfgid).txt")
        if !isfile(path)
            push!(results, (config = cfgid, quantity = "FILE", rel = NaN,
                            tol = rel_tol, pass = false, got = NaN, want = NaN))
            allpass = false
            continue
        end
        cfg = load_nist_config(path)
        @assert cfg.N == d.N "N mismatch for config $cfgid"
        @assert cfg.L == d.box "box mismatch for config $cfgid"

        r3 = evaluate_config(cfg; rc = 3.0, mode = :trunc)
        r4 = evaluate_config(cfg; rc = 4.0, mode = :trunc)
        rL = evaluate_config(cfg; rc = 3.0, mode = :lfs)
        l3 = total_lrc(cfg.N, cfg.L, 3.0)
        l4 = total_lrc(cfg.N, cfg.L, 4.0)

        checks = [
            ("U_pair(rc=3,LRC)", r3.U_pair, d.u3),
            ("W_pair(rc=3,LRC)", r3.W_pair, d.w3),
            ("U_LRC(rc=3)", l3, d.l3),
            ("U_pair(rc=4,LRC)", r4.U_pair, d.u4),
            ("W_pair(rc=4,LRC)", r4.W_pair, d.w4),
            ("U_LRC(rc=4)", l4, d.l4),
            ("U_pair(rc=3,LFS)", rL.U_pair, d.uL),
            ("W_pair(rc=3,LFS)", rL.W_pair, d.wL),
        ]
        if verbose
            @printf("config %d  N=%d  box=%.1f^3\n", cfgid, cfg.N, cfg.L)
        end
        for (qname, got, want) in checks
            rel = abs(got - want) / max(abs(want), 1e-30)
            pass = rel <= rel_tol
            allpass &= pass
            push!(results, (config = cfgid, quantity = qname, rel = rel, tol = rel_tol,
                            pass = pass, got = got, want = want))
            if verbose
                @printf("  %-20s got=%16.7G want=%16.7G rel=%.2e %s\n",
                        qname, got, want, rel, pass ? "ok" : "FAIL")
            end
        end
    end
    if verbose
        rels = [r.rel for r in results if isfinite(r.rel)]
        @printf("\nworst relative deviation across %d checks: %.3e (tolerance %.1e)\n",
                length(rels), maximum(rels), rel_tol)
        println(allpass ? "NIST VALIDATION: PASS" : "NIST VALIDATION: FAIL")
    end
    return (allpass = allpass, results = results)
end

end # module NISTValidate
