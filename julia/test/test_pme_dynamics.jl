using FoundationCore
using LinearAlgebra

const T = FoundationCore.Types
const F = FoundationCore.Factors
const D = FoundationCore.Dynamics
const EW = FoundationCore.Electrostatics
const PME = FoundationCore.PME

box = T.OrthorhombicBox([8.0, 8.0, 8.0])
ctx = F.Context(box, T.REDUCED, 4)
q = [1.0 3.1 5.2 6.0; 1.0 1.7 4.8 6.4; 2.0 2.2 5.1 6.0]
charges = [1.0, -0.7, 0.4, -0.7]
corrections = PME.PairCorrections(excluded=[(1, 2)], scaled=[(2, 3, 0.5)])
kw = (alpha=0.38, rcut=3.7, grid=(48, 48, 48), order=6,
      prefactor=0.5, corrections=corrections)
factor = PME.PMEFactor(q, charges, box; kw...)

@assert !F.is_local(factor)
@assert F.factor_support_size(factor) == size(q, 2)
@assert F.read_support(factor) == Tuple(1:size(q, 2))
@assert F.force_consistent(factor)

# Full factor path force check includes spreading/interpolation, real-space,
# self/background, and explicit pair corrections.
ctxenergy(x) = PME.pme_factor_energy(factor, x, ctx)
Fmesh = zeros(size(q))
Umesh = F.evaluate!(Fmesh, factor, q, ctx)
Ffd = zeros(size(q))
h = 1e-6
for idx in eachindex(q)
    qp = copy(q); qm = copy(q)
    qp[idx] += h; qm[idx] -= h
    Ffd[idx] = -(ctxenergy(qp) - ctxenergy(qm)) / (2h)
end
@assert maximum(abs, Fmesh - Ffd) < 2e-7

# The independent direct Ewald factor has the same physics/conventions. PME
# mesh error at this grid is small relative to the explicit tolerance.
ewfactor = EW.EwaldFactor(q, charges, box; alpha=kw.alpha, rcut=kw.rcut,
                          kmax=10, prefactor=kw.prefactor,
                          corrections=EW.PairCorrections(excluded=[(1, 2)],
                                                         scaled=[(2, 3, 0.5)]))
Uew, Few = EW.evaluate_ewald(q, charges, collect(box.L); alpha=kw.alpha,
                             rcut=kw.rcut, kmax=10, prefactor=kw.prefactor,
                             corrections=ewfactor.corrections)
@assert abs(Umesh - Uew) < 2e-5
@assert maximum(abs, Fmesh - Few) < 2e-5

function run_steps(factor, ctx, q, p, nsteps, dt)
    sys = D.HarmonicSystem([factor], ones(size(q, 2)), ctx)
    st = D.State(q, p)
    recs, energies = D.run_trajectory(st, sys, dt, nsteps; record_every=1)
    return D.energy_drift_report(recs, energies), energies
end

p0 = [0.0 0.01 -0.01 0.0; 0.01 -0.01 0.0 0.0; -0.01 0.0 0.01 0.0]
coarse, ecoarse = run_steps(factor, ctx, q, p0, 1000, 0.001)
fine, efine = run_steps(factor, ctx, q, p0, 1000, 0.0005)
@assert coarse.bounded && fine.bounded
@assert maximum(abs, ecoarse.compensated .- ecoarse.compensated[1]) < 2e-4
@assert maximum(abs, efine.compensated .- efine.compensated[1]) <
        maximum(abs, ecoarse.compensated .- ecoarse.compensated[1])

# EwaldFactor refuses triclinic geometry, while PMEFactor now supports TriclinicBox dynamics.
triclinic = T.TriclinicBox([8.0 0.2 0.0; 0.0 8.0 0.0; 0.0 0.0 8.0])
ewald_refused = try
    EW.EwaldFactor(q, charges, triclinic; alpha=kw.alpha, rcut=kw.rcut)
    false
catch e
    e isa T.UnsupportedGeometry
end
@assert ewald_refused
tri_factor = PME.PMEFactor(q, charges, triclinic; kw...)
tri_ctx = F.Context(triclinic, T.REDUCED, 4)
tri_coarse, _ = run_steps(tri_factor, tri_ctx, q, p0, 200, 0.001)
@assert tri_coarse.bounded

println("test_pme_dynamics PASS | PME/Ewald dU=$(abs(Umesh-Uew)) maxdF=$(maximum(abs, Fmesh-Few)) | coarse_band=$(coarse.max_abs_dev) fine_band=$(fine.max_abs_dev) coarse_slope=$(coarse.slope) +/- $(coarse.slope_stderr)")
