# Reproduce the PME scaling evidence into `results/`.
#
#     export JULIA_DEPOT_PATH=~/.cache/biology-mm/julia-depot
#     julia --project=julia julia/experiments/pme_scaling/run.jl
#
# This is a thin driver: all measurements live in bench/pme_scaling.jl, and the
# output files under results/ are written by that benchmark, not edited here.

const HERE = @__DIR__
const JULIA_ROOT = normpath(joinpath(HERE, "..", ".."))   # julia
const BENCH = joinpath(JULIA_ROOT, "bench", "pme_scaling.jl")
const OUT = joinpath(HERE, "results")

isfile(BENCH) || error("benchmark not found at $BENCH")
mkpath(OUT)

cmd = `$(joinpath(Sys.BINDIR, "julia")) --project=$JULIA_ROOT $BENCH --all --big --repeat=3 --out=$OUT`
println("running: ", cmd)
run(cmd)
println("\nartifacts:")
for f in sort(readdir(OUT))
    path = joinpath(OUT, f)
    println("  ", f, "  (", filesize(path), " bytes)")
end
