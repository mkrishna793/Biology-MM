# Reproduce the reported NVE energy band at w216/w1728-like settings.
#
#   cd julia
#   JULIA_DEPOT_PATH=~/.cache/biology-mm/julia-depot \
#   ~/.cache/biology-mm/julia-1.13.3/bin/julia --project=. \
#   experiments/nve_diagnosis/probe_band.jl
#
# No file is modified; each configuration builds its own box.

using FoundationCore
using Printf

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
isdefined(FoundationCore, :WaterBox) ||
    Base.include(FoundationCore, joinpath(ROOT, "src", "WaterBox.jl"))
const WB = FoundationCore.WaterBox
const D = FoundationCore.Dynamics

function probe(; nmol, seed, dt, alpha, pme_rcut = nothing, nsteps = 1000)
    w = WB.tip3p_water(nmol; density = 1.0, seed = seed)
    field = WB.water_field(w; alpha = alpha, pme_rcut = pme_rcut)
    tr = WB.WaterTrajectory(w, field; temperature = 300.0, seed = seed)
    WB.relax!(tr; max_steps = 3000)
    WB.rethermalize!(tr; temperature = 300.0)
    t0 = time()
    band, slope, se, n = WB.nve_energy_band(tr; nsteps = nsteps, dt = dt)
    wall = time() - t0
    U0 = WB.potential(field, tr.st.q)
    @printf("%5d mol | alpha %.3f rcut_pme %.4f | dt %.3f fs | rcut_lj %.4f | band %10.4f | slope %10.2f +/- %7.2f | band/mol %9.3e | U_start %9.2f | %.1fs\n",
            nmol, field.alpha, field.pme_rcut, dt * 1000, field.rcut,
            band, slope, se, band / nmol, U0, wall)
    return band
end

println("== current-code NVE band vs the legacy reported settings ==")
# Trusted reference: 200 molecules, default split, 2 fs.
probe(nmol = 200, seed = 11, dt = 0.002, alpha = nothing)
# w216 settings from runs/w216/report.json: alpha = 3.5, pme_rcut = 0.9, 1 fs.
probe(nmol = 216, seed = 11, dt = 0.001, alpha = 3.5, pme_rcut = 0.9)
# Same box, default split: isolates alpha/rcut from size.
probe(nmol = 216, seed = 11, dt = 0.001, alpha = nothing)
# w1728 settings, small stand-in box (same alpha, same density).
probe(nmol = 216, seed = 5, dt = 0.001, alpha = 3.5)
