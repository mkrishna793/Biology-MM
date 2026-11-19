# Imported-system dynamics smoke test.
#
# This is the boundary between "parameter import + force evaluation" and
# "simulation": build the real DNA fixture through the normalized schema and
# push it through the full factor -> Dynamics -> velocity-Verlet path. It is a
# three-step smoke test, not a trajectory qualification. The force-consistent
# :lfs mode is required; :trunc must be refused by the integrator gate.

using FoundationCore
using Printf

const CH = FoundationCore.Chemistry
const D = FoundationCore.Dynamics
const F = FoundationCore.Factors

const REPO = normpath(joinpath(@__DIR__, "..", ".."))
const PYTHON = let p = get(ENV, "CHEM_PYTHON", "")
    !isempty(p) ? p : something(Sys.which("python3"), Sys.which("python"), "python3")
end

function read_crd(path)
    lines = readlines(path)
    head = split(lines[2])
    n = parse(Int, head[1])
    toks = split(join(lines[2:end]), r"\s+"; keepempty = false)
    offset = length(head) > 1 ? 3 : 2
    vals = parse.(Float64, toks[offset:offset+3n-1])
    q = Matrix{Float64}(undef, 3, n)
    k = 1
    for i in 1:n, d in 1:3
        q[d, i] = vals[k] / 10.0
        k += 1
    end
    return q
end

function main()
    parm = joinpath(REPO, "wrappers", "python", "tests", "systems",
                    "DNA_mbondi3.prmtop")
    crd = joinpath(REPO, "wrappers", "python", "tests", "systems",
                   "DNA_mbondi3.inpcrd")
    isfile(parm) && isfile(crd) || error("DNA fixture inputs missing")
    out = tempname() * ".json"
    cmd = `$PYTHON $(joinpath(REPO, "chemistry", "tools", "extract_amber.py"))
           --parm $parm --crd $crd --out $out --force-field "DNA dynamics smoke"
           --repo-root $REPO`
    run(pipeline(cmd; stdout = devnull, stderr = devnull))
    ns = CH.read_normalized(out)
    q = read_crd(crd)
    n = CH.n_atoms(ns)
    @assert size(q, 2) == n

    prot = CH.default_protocol(ns; nonbonded = :direct, lj_mode = :lfs)
    rs = CH.build_resolved(ns; protocol = prot)
    ctx = F.Context(rs.box, rs.units, n)
    hs = D.HarmonicSystem(rs.factors, rs.atoms.masses, ctx)
    @assert D.force_model_supported(hs) "the :lfs imported system is not force-consistent"
    st = D.State(copy(q), zeros(3, n))
    dt = 1.0e-4
    steps = 3
    @time records, series = D.run_trajectory(st, hs, dt, steps; record_every = 1)
    E = series.compensated
    @assert length(records) == steps
    @assert all(isfinite, E) "non-finite imported-system energy"
    drift = maximum(abs, E .- E[1])
    Fmax = maximum(abs, st.F)
    @printf("  imported dynamics: N=%d steps=%d dt=%.1e ps time=%.2e ps E0=%.6e dEmax=%.3e maxF=%.3e\n",
            n, steps, dt, dt * steps, E[1], drift, Fmax)
    @assert drift < 1.0e3 "smoke-step energy drifted beyond 1e3 kJ/mol"
    @assert Fmax > 1.0 "forces are not a real force field"

    # `:trunc` is only unsafe for the periodic StreamedLJ path; the direct
    # NoBox builder uses an explicit all-pair table with no cutoff, so the
    # periodic `:trunc` refusal is covered by test_pme_dynamics.jl /
    # test_streamed_lj_dynamics.jl rather than asserted here.
    println("CHEMISTRY IMPORTED DYNAMICS SMOKE PASSED")
    return true
end

main()
