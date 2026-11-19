using FoundationCore
using LinearAlgebra
const T = FoundationCore.Types
const F = FoundationCore.Factors
const D = FoundationCore.Dynamics
const EW = FoundationCore.Electrostatics

box = T.OrthorhombicBox([12.0, 12.0, 12.0])
q = [1.0 4.0 7.1; 1.2 5.1 2.9; 2.0 4.4 8.0]
charges = [1.0, -0.7, -0.3]
corrections = EW.PairCorrections(excluded=[(1, 2)], scaled=[(2, 3, 0.5)])
kw = (alpha=0.34, rcut=5.5, kmax=5, prefactor=1.2, corrections=corrections)
factor = EW.EwaldFactor(q, charges, box; kw...)
ctx = F.Context(box, T.REDUCED, size(q, 2))

@assert !F.is_local(factor)
@assert F.factor_support_size(factor) == size(q, 2)
@assert F.force_consistent(factor)
@assert isapprox(EW.ewald_energy(factor, q, ctx), EW.evaluate_ewald(q, charges, collect(box.L); kw...)[1]; atol=1e-13)

forces = zeros(size(q))
energy = F.evaluate!(forces, factor, q, ctx)
fd_forces = zeros(size(q)); h = 1e-6
for idx in eachindex(q)
    qp = copy(q); qm = copy(q); qp[idx] += h; qm[idx] -= h
    fd_forces[idx] = -(EW.ewald_energy(factor, qp, ctx) - EW.ewald_energy(factor, qm, ctx))/(2h)
end
@assert isfinite(energy)
@assert maximum(abs, forces-fd_forces) < 3e-7

grad = zeros(length(charges))
EW.ewald_charge_gradient!(grad, factor, q, ctx)
for i in eachindex(charges)
    cp = copy(charges); cm = copy(charges); cp[i] += h; cm[i] -= h
    fd = (EW.ewald_energy(factor, q, ctx; charges=cp) - EW.ewald_energy(factor, q, ctx; charges=cm))/(2h)
    @assert isapprox(grad[i], fd; atol=2e-8, rtol=2e-7)
end

# The exact pair-correction convention is shared by energy and force, also
# when the corrected pair is outside the real-space cutoff.
qpair = [1.0 7.2; 1.0 1.0; 1.0 1.0]
chpair = [1.0, -1.0]
boxpair = T.OrthorhombicBox([12.0, 12.0, 12.0])
ctxpair = F.Context(boxpair, T.REDUCED, 2)
plain = EW.EwaldFactor(qpair, chpair, boxpair; alpha=0.35, rcut=4.0, kmax=4)
excluded = EW.EwaldFactor(qpair, chpair, boxpair; alpha=0.35, rcut=4.0, kmax=4,
                          corrections=EW.PairCorrections(excluded=[(1,2)]))
scaled = EW.EwaldFactor(qpair, chpair, boxpair; alpha=0.35, rcut=4.0, kmax=4,
                        corrections=EW.PairCorrections(scaled=[(1,2,0.5)]))
d = qpair[:,1]-qpair[:,2]; d .-= collect(boxpair.L) .* round.(d ./ collect(boxpair.L))
r = norm(d)
@assert isapprox(EW.ewald_energy(excluded, qpair, ctxpair)-EW.ewald_energy(plain, qpair, ctxpair), 1/r; atol=1e-12)
@assert isapprox(EW.ewald_energy(scaled, qpair, ctxpair)-EW.ewald_energy(plain, qpair, ctxpair), 0.5/r; atol=1e-12)

# Integrate a mild neutral three-charge fixture with the same factor and check
# that the symplectic trajectory's energy remains in a narrow bounded band.
qmd = [1.2 4.1 7.0; 2.0 6.0 3.3; 1.5 5.2 9.0]
chmd = [1.0, -0.7, -0.3]
boxmd = T.OrthorhombicBox([12.0, 13.0, 14.0])
fmd = EW.EwaldFactor(qmd, chmd, boxmd; alpha=0.34, rcut=5.5, kmax=4, prefactor=0.2)
sys = D.HarmonicSystem([fmd], ones(3), F.Context(boxmd, T.REDUCED, 3))
st = D.State(qmd, [0.0 0.01 -0.01; 0.01 -0.01 0.0; -0.01 0.0 0.01])
records, energies = D.run_trajectory(st, sys, 0.001, 1200; record_every=2)
report = D.energy_drift_report(records, energies)
@assert report.bounded
@assert maximum(abs, energies.compensated .- energies.compensated[1]) < 2e-4

for build in (() -> EW.EwaldFactor(q, charges, T.NoBox(); kw...),
              () -> EW.EwaldFactor(q, charges, box; alpha=0.34, rcut=5.5, kmax=17))
    failed = false
    try build() catch e; failed = e isa ArgumentError end
    @assert failed
end
println("test_electrostatics_dynamics PASS")
