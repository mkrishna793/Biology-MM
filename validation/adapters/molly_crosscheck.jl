"""
Independent cross-check of this core's Lennard-Jones energy against Molly.jl.

Molly.jl is a separate pure-Julia MD package with its own LJ implementation, so
agreement is evidence from two independent codebases rather than a code checked
against itself.

Scope, stated before the result
-------------------------------
Establishes: the LJ pair energy with minimum-image periodic boundaries agrees with
Molly's on a real NIST configuration, *in a matched cutoff convention*.

Does not establish: agreement on dynamics, integrators, neighbour policy, or
thermodynamics. The comparison is only meaningful because both codes use the same
potential:

    this core `LennardJonesPair(mode = :shift, rc)`  ≡  Molly `ShiftedPotentialCutoff(rc)`

both of which compute `V_LJ(r) − V_LJ(rc)` inside the cutoff and 0 outside, with
no force shift.

Molly's default `cutoff = NoCutoff()` cannot be used with a periodic boundary
(it has no `pe_cutoff` method for `Nothing`), which is a Molly API detail recorded
here rather than worked around silently.
"""

using FoundationCore
using FoundationCore.Types
using FoundationCore.Factors
using FoundationCore.Neighbors
using Molly
using Unitful
using Printf

const T = FoundationCore.Types
const FA = FoundationCore.Factors
const NB = FoundationCore.Neighbors

# Molly works in real units with molar-consistent mass units. We use nm and
# kJ/mol, so a reduced LJ site (σ=1, ϵ=1, m=1) becomes σ=1 nm, ϵ=1 kJ/mol,
# m=1 g/mol. The *dimensionless ratio* E_core / E_molly is then the comparison,
# which is convention-independent.
const SIGMA = 1.0u"nm"
const EPS = 1.0u"kJ/mol"

function load_nist(path)
    lines = readlines(path)
    box = parse.(Float64, split(strip(lines[1])))
    n = parse(Int, split(strip(lines[2]))[1])
    q = zeros(3, n)
    for i in 1:n
        p = split(strip(lines[2 + i]))
        q[1, i] = parse(Float64, p[2])
        q[2, i] = parse(Float64, p[3])
        q[3, i] = parse(Float64, p[4])
    end
    return box[1], q
end

"LJ energy through this core: minimum image, :shift mode at rc."
function core_lj_energy(q, L, rc)
    N = size(q, 2)
    box = T.OrthorhombicBox((L, L, L))
    ctx = FA.Context(box, T.REDUCED, N)
    Fv = zeros(3, N)
    nl = NB.build_neighbors(q, box, rc, 0.0)
    U = 0.0
    for (i, j) in NB.neighbor_pairs(nl)
        U += FA.evaluate!(Fv,
                          FA.LennardJonesPair(i, j, 1.0, 1.0; mode = :shift, rc = rc),
                          q, ctx)
    end
    return U
end

"LJ energy through Molly on the same coordinates and the matched cutoff convention."
function molly_lj_energy(q, L, rc)
    N = size(q, 2)
    atoms = [Atom(index = i, atom_type = nothing, mass = 1.0u"g/mol",
                  charge = 0.0u"C", σ = SIGMA, ϵ = EPS) for i in 1:N]
    coords = [SVector{3}(q[1, i], q[2, i], q[3, i])u"nm" for i in 1:N]
    b = CubicBoundary(L * u"nm")
    lj = LennardJones(cutoff = ShiftedPotentialCutoff(rc * u"nm"), use_neighbors = false)
    sys = System(atoms = atoms, coords = coords, boundary = b,
                 pairwise_inters = (lj,))
    return ustrip(potential_energy(sys))
end

function main()
    dir = get(ENV, "NIST_DIR", "")
    if isempty(dir)
        println("Set NIST_DIR to the extracted NIST sample-configuration directory.")
        return 1
    end
    path = joinpath(dir, "lj_sample_config_periodic4.txt")
    isfile(path) || (println("missing $path"); return 1)
    L, q = load_nist(path)
    rc = 3.0
    Ucore = core_lj_energy(q, L, rc)
    Um = molly_lj_energy(q, L, rc)
    @printf("configuration: NIST periodic config 4 (N=%d, box=%.1f sigma)\n", size(q, 2), L)
    @printf("  this core   : %.12f  (reduced units, :shift at rc=%.1f)\n", Ucore, rc)
    @printf("  Molly.jl    : %.12f  (kJ/mol, ShiftedPotentialCutoff %.1f nm)\n", Um, rc)
    rel = abs(Ucore - Um) / max(abs(Ucore), abs(Um), 1e-30)
    @printf("  relative difference: %.3e\n", rel)
    # Two algorithms summing the same terms in different orders: agreement is
    # required only to accumulation roundoff, not bitwise.
    ok = rel < 1e-12
    println(ok ? "MOLLY CROSS-CHECK: PASS (independent implementations agree)" :
                 "MOLLY CROSS-CHECK: DISAGREE")
    return ok ? 0 : 2
end

exit(main())
