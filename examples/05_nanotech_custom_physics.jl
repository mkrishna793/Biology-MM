# ============================================================================
#  examples/05_nanotech_custom_physics.jl
#
#  Flagship Nanotech, Custom Physics, Ternary Lattice GCMC & Enhanced Sampling.
#
#  Run from julia:
#      julia --project=. ../examples/05_nanotech_custom_physics.jl
# ============================================================================

using Printf
using LinearAlgebra
using Random

include(joinpath(@__DIR__, "..", "julia", "src", "FoundationCore.jl"))
using .FoundationCore

const T  = FoundationCore.Types
const F  = FoundationCore.Factors
const D  = FoundationCore.Dynamics
const SC = FoundationCore.SystemContract
const CP = FoundationCore.CustomPhysics
const RN = FoundationCore.ReactiveNano
const GC = FoundationCore.GrandCanonical
const ES = FoundationCore.EnhancedSampling
const EN = FoundationCore.Engine

const T0 = time_ns()
wall() = (time_ns() - T0) / 1e9

nfail = 0

function checkgate(name::AbstractString, ok::Bool, detail...)
    ok || (global nfail = nfail + 1)
    print(rpad("  [$name] $(ok ? "PASS" : "FAIL")", 38), " ")
    for d in detail
        print(d)
    end
    println()
    return ok
end

println("=" ^ 78)
println(" FoundationCore :: 05 - Nanotech, Custom Physics, Lattice GCMC & Sampling")
println("=" ^ 78)

# ---------------------------------------------------------------------------
# 1. (5,5) Armchair CNT with TersoffFactor: NVE Conservation & Stress-Strain
# ---------------------------------------------------------------------------
println("\n[1] (5,5) Armchair Carbon Nanotube with Reactive Tersoff Potential")
cnt = RN.build_cnt(5, 5, 2)
n_cnt = size(cnt.q, 2)
rng = MersenneTwister(2026)
q_cnt = cnt.q .+ 0.0015 .* randn(rng, 3, n_cnt)
p_cnt = 0.08 .* randn(rng, 3, n_cnt)
p_cnt .-= sum(p_cnt, dims = 2) ./ n_cnt

sys_cnt = T.System(T.AtomSet(q_cnt, p_cnt, cnt.masses, zeros(n_cnt), cnt.names),
                   cnt.box, T.ChemicalTopology(), CP.MD_UNITS, 300.0)
f_tersoff = RN.TersoffFactor(n_cnt, RN.tersoff_carbon(; units = :md))
eng_cnt = EN.create_engine(sys_cnt, [f_tersoff]; backend = :cpu, integrator = :verlet, dt_ps = 0.0002)
cons_cnt = EN.verify_engine_conservation(eng_cnt, D.State(copy(q_cnt), copy(p_cnt)); n_steps = 200)

@printf("   atoms = %d   radius = %.4f nm   Lz = %.4f nm\n",
        n_cnt, cnt.radius_nm, cnt.box.L[3])
@printf("   200-step NVE rel_energy_drift = %.3e   momentum_drift = %.3e\n",
        cons_cnt.rel_energy_drift, cons_cnt.momentum_drift_norm)
checkgate("CNT Tersoff NVE conservation", cons_cnt.rel_energy_drift < 1e-5,
          @sprintf("rel_drift = %.2e", cons_cnt.rel_energy_drift))

println("\n   Axial tensile stress-strain scan (epsilon_zz from -0.01 to +0.04):")
@printf("   %-10s %-16s %-16s\n", "strain", "U (kJ/mol)", "sigma_zz (kJ/mol/nm^3)")
F_tmp = zeros(3, n_cnt)
G_tmp = zeros(3, 3)
sigma_vals = Float64[]
for eps_z in (-0.01, 0.0, 0.01, 0.02, 0.04)
    L0 = cnt.box.L
    Lz_s = L0[3] * (1.0 + eps_z)
    box_s = T.OrthorhombicBox((L0[1], L0[2], Lz_s))
    qs = copy(cnt.q)
    qs[3, :] .*= (1.0 + eps_z)
    U_s = EN.evaluate_step!(eng_cnt, F_tmp, qs, box_s; box_deriv = G_tmp)
    sig_zz = (G_tmp[3, 3] * Lz_s) / T.box_volume(box_s)
    push!(sigma_vals, sig_zz)
    @printf("   %-+10.3f %-16.4f %-+16.4e\n", eps_z, U_s, sig_zz)
end
checkgate("CNT positive tensile stiffness", sigma_vals[end] > sigma_vals[1] && sigma_vals[end] > 0.0,
          @sprintf("sigma(+4%%) = %.2e", sigma_vals[end]))

# ---------------------------------------------------------------------------
# 2. ParameterEncoder + Yukawa & CubicSpline Custom Physics Laws
# ---------------------------------------------------------------------------
println("\n[2] ParameterEncoder + Yukawa & Tabulated Spline Verification")
enc = CP.ParameterEncoder([
    CP.ParameterSpec(:A, 45.0; unit = "kJ/mol*nm", lower = 0.0, encoding = :log),
    CP.ParameterSpec(:kappa, 3.2; unit = "nm^-1", lower = 0.0, encoding = :log),
    CP.ParameterSpec(:mix, 0.4; lower = 0.0, upper = 1.0, encoding = :logit),
])
z_enc = CP.encode_parameters(enc, Dict(:A => 45.0, :kappa => 3.2, :mix => 0.4))
p_dec = CP.decode_parameters(enc, z_enc)

q4 = [0.25  0.58  0.34  0.80;
      0.30  0.35  0.66  0.71;
      0.40  0.44  0.42  0.75]
box4 = T.OrthorhombicBox((1.8, 1.8, 1.8))
pairs4 = [(1, 2), (1, 3), (2, 3), (2, 4), (3, 4)]

f_yukawa = CP.CustomPairFactor(:yukawa, Dict(:A => p_dec[:A], :kappa => p_dec[:kappa]), pairs4;
                               rc = 0.85, mode = :lfs, n_atoms = 4)
res_yuk = CP.verify_custom_factor_gradients(f_yukawa, q4, box4, CP.MD_UNITS;
                                            h = 1e-5, force_tol = 1e-6, box_tol = 1e-6, order = 4)
checkgate("Yukawa 4th-order FD forces & stress", res_yuk.passed,
          @sprintf("max_f=%.2e, max_G=%.2e", res_yuk.max_force_err, res_yuk.max_box_err))

r_grid = collect(range(0.15, 1.0; length = 64))
u_grid = 200.0 .* (1.0 .- exp.(-8.0 .* (r_grid .- 0.30))) .^ 2
spline = CP.CubicSplineTable(r_grid, u_grid)
f_spline = CP.CustomPairFactor(:tabulated_spline, Dict{Symbol,Float64}(), pairs4;
                               rc = 0.90, mode = :lfs, n_atoms = 4, spline = spline)
res_sp = CP.verify_custom_factor_gradients(f_spline, q4, box4, CP.MD_UNITS;
                                           h = 1e-5, force_tol = 1e-6, box_tol = 1e-6, order = 4)
checkgate("CubicSpline 4th-order FD forces & stress", res_sp.passed,
          @sprintf("max_f=%.2e, max_G=%.2e", res_sp.max_force_err, res_sp.max_box_err))

# ---------------------------------------------------------------------------
# 3. TernaryLatticeGas (TritWord64) Competitive Adsorption Isotherm
# ---------------------------------------------------------------------------
println("\n[3] TernaryLatticeGas (TritWord64) Competitive Adsorption Isotherm")
@printf("   %-8s %-12s %-12s %-12s\n", "mu_A", "theta_A", "theta_B", "selectivity")
thetas_A = Float64[]
for mu_A in (-1.0, 0.0, 1.0)
    lg = GC.TernaryLatticeGas(8, 8, 2; J_AA = -0.15, J_BB = 0.05, J_AB = 0.25,
                              eps_A = -0.4, eps_B = 0.1, mu_A = mu_A, mu_B = 0.0,
                              temperature = 1.0)
    lres = GC.run_lattice_gcmc!(lg, 500; burnin = 120, rng = MersenneTwister(100 + round(Int, 10 * mu_A)))
    push!(thetas_A, lres.theta_A)
    @printf("   %-+8.2f %-12.4f %-12.4f %-12.4f\n", mu_A, lres.theta_A, lres.theta_B, lres.selectivity)
end
checkgate("Lattice isotherm monotonic loading", thetas_A[1] < thetas_A[2] < thetas_A[3],
          @sprintf("%.3f < %.3f < %.3f", thetas_A[1], thetas_A[2], thetas_A[3]))

# ---------------------------------------------------------------------------
# 4. Enhanced Sampling: UmbrellaFactor & Well-Tempered MetadynamicsBias
# ---------------------------------------------------------------------------
println("\n[4] Enhanced Sampling: UmbrellaFactor & MetadynamicsBias")
cv_d = ES.DistanceCV(1, 2)
s_cur = ES.cv_value_and_grad!(zeros(3, 4), cv_d, q4, box4)
f_umb = ES.UmbrellaFactor(cv_d, 30.0, s_cur + 0.05, 4)
meta = ES.MetadynamicsBias(cv_d, 4; height = 0.6, sigma = 0.2, bias_factor = 4.0)
ES.deposit_hill!(meta, s_cur - 0.1)
ES.deposit_hill!(meta, s_cur + 0.1)

res_umb = CP.verify_custom_factor_gradients(f_umb, q4, box4, CP.MD_UNITS;
                                            h = 1e-5, force_tol = 1e-6, box_tol = 1e-6, order = 4)
res_meta = CP.verify_custom_factor_gradients(meta, q4, box4, CP.MD_UNITS;
                                             h = 1e-5, force_tol = 1e-6, box_tol = 1e-6, order = 4)
checkgate("UmbrellaFactor exact forces & stress", res_umb.passed,
          @sprintf("max_f=%.2e, max_G=%.2e", res_umb.max_force_err, res_umb.max_box_err))
checkgate("MetadynamicsBias exact forces & stress", res_meta.passed,
          @sprintf("max_f=%.2e, max_G=%.2e", res_meta.max_force_err, res_meta.max_box_err))

println()
println("=" ^ 78)
@printf(" RESULT: %s   %d gate(s) failed   wall time = %.2f s\n",
        nfail == 0 ? "PASS" : "FAIL", nfail, wall())
println("=" ^ 78)
exit(nfail == 0 ? 0 : 1)
