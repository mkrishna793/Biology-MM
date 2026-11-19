# ============================================================================
#  examples/equilibrium_sensitivity.jl
#
#  Fixed-measure equilibrium parameter sensitivities from correlated samples.
#
#  Run from julia:
#      julia --project=. ../examples/equilibrium_sensitivity.jl
#  Or from the repository root:
#      julia --project=julia examples/equilibrium_sensitivity.jl
#
#  Deterministic: every RNG is an explicitly seeded MersenneTwister.
#
#  This example connects three pieces that already exist in the core:
#    - `Sampling.hmc_trajectory` samples q from exp(-beta U(q; theta));
#    - `ParameterGradients.evaluate_parameter_gradients!` gives dU/dtheta at
#      fixed q;
#    - `EquilibriumSensitivities` averages those gradients and attaches an
#      autocorrelation-aware standard error.
#
#  The exactly solvable target is a one-dimensional harmonic bond,
#
#      U(x; k) = (k/2) (x - r0)^2,
#
#  with beta = 1.  Its free energy is
#
#      A(k) = -(1/beta) log sqrt(2*pi/(beta*k)) + const,
#      dA/dk = 1/(2*beta*k).
#
#  The equilibrium identity used here is dA/dk = <dU/dk>_beta.  For this
#  harmonic target dU/dk = (x-r0)^2/2 and <(x-r0)^2> = 1/(beta*k), so the
#  gradient estimator has the same exact value, 1/(2*beta*k), as dA/dk.
#  The sample estimate is checked against both that closed form and a central
#  finite difference of A(k).
#
#  Scope, stated before any number:
#    - beta is fixed, and the measure and domain do not depend on the
#      parameter.  Parameter-dependent restraints/Jacobians are out of scope.
#    - A gradient is a derivative at a point, NOT a finite free-energy change.
#      Integrating over a finite parameter path is not done here.
#    - The two-leg section is a TOY arithmetic difference between two harmonic
#      ensembles.  It is not a binding affinity, has no receptor, no solvent, no
#      standard-state correction, and no physical claim.
# ============================================================================

using Printf
using Random
using Statistics

include(joinpath(@__DIR__, "..", "julia", "src", "FoundationCore.jl"))
using .FoundationCore

const T = FoundationCore.Types
const F = FoundationCore.Factors
const SP = FoundationCore.Sampling
const PG = FoundationCore.ParameterGradients
const ES = FoundationCore.EquilibriumSensitivities

nfail = 0

"""Record and print one gate. `detail` is pre-formatted text."""
function checkgate(name::AbstractString, ok::Bool, detail...)
    ok || (global nfail = nfail + 1)
    print(rpad("  [$name] $(ok ? "PASS" : "FAIL")", 43), " ")
    for d in detail
        print(d)
    end
    println()
    return ok
end

println("=" ^ 78)
println(" Foundation core :: equilibrium parameter sensitivities")
println("=" ^ 78)

# ---------------------------------------------------------------------------
# 1. The target's exact answer, before any sampling.
# ---------------------------------------------------------------------------
println("\n[1] Exact harmonic free energy and its derivative")
beta = 1.0
k0 = 4.0
r0 = 0.0
A(k) = -log(sqrt(2.0 * pi / (beta * k))) / beta

exact_dA = 1.0 / (2.0 * beta * k0)
h = 1e-6
fd_dA = (A(k0 + h) - A(k0 - h)) / (2h)
fd_rel = abs(fd_dA - exact_dA) / max(abs(exact_dA), 1.0)

@printf("   A(k) = -(1/beta) log sqrt(2*pi/(beta*k)) + const\n")
@printf("   exact dA/dk at k0 = %.1f: %.10f\n", k0, exact_dA)
@printf("   central difference h = %.0e: %.10f  (relative miss %.2e)\n",
        h, fd_dA, fd_rel)
println("   The additive constant is arbitrary and drops out of the derivative.")
checkgate("closed-form harmonic derivative", fd_rel < 1e-8,
          "relative miss ", @sprintf("%.2e", fd_rel))

# ---------------------------------------------------------------------------
# 2. One harmonic bond sampled by real HMC.  The sampled coordinate is mapped
#    to two atoms at the origin and at x; the factor evaluator supplies U and
#    the force, ParameterGradients supplies dU/dk and dU/dr0.
# ---------------------------------------------------------------------------
println("\n[2] HMC on U(x) = (k/2)x^2, gradient estimator <x^2/2>")
ctx2 = F.Context(T.REDUCED, 2)
ligand_factors = [F.HarmonicBond(1, 2, k0, r0)]

function q_of_ligand(x::Real)
    q = zeros(3, 2)
    q[1, 2] = x
    return q
end

function U_ligand(x::Vector{Float64})
    return F.total_energy_forces(ligand_factors, q_of_ligand(x[1]), ctx2)[1]
end

function gradU_ligand(x::Vector{Float64})
    Fv = F.total_energy_forces(ligand_factors, q_of_ligand(x[1]), ctx2)[2]
    return [-Fv[1, 2]]
end

m1 = [1.0]
dt = 0.2
nleap = 10
nsteps = 40_000
ligand_run = SP.hmc_trajectory([1.0], U_ligand, gradU_ligand, m1,
                               dt, nleap, nsteps, beta, MersenneTwister(101))

# Shared parameter ids: both k and r0 are driven by one dictionary, exactly as
# a force-field parameter would be.  The factor fields are the fallback values.
ligand_ids = Dict{Tuple{Int,Symbol},Symbol}((1, :k) => :bond_k,
                                             (1, :r0) => :bond_r0)
ligand_params = Dict{Symbol,Float64}(:bond_k => k0, :bond_r0 => r0)
ligand_grad = ES.factor_gradient_function(ligand_factors, ctx2;
                                          parameter_ids = ligand_ids,
                                          parameters = ligand_params)
grad_fn_ligand(x) = ligand_grad(q_of_ligand(x[1]))

ligand_sens = ES.estimate_sensitivity(ligand_run.samples, grad_fn_ligand;
                                      min_ess = 200)
k_mean = ligand_sens.means[:bond_k]
k_se = ligand_sens.se[:bond_k]
k_z = (k_mean - exact_dA) / k_se
probe = ligand_grad(q_of_ligand(1.3))[:bond_k]
probe_exact = 0.5 * (1.3 - r0)^2
probe_q = q_of_ligand(1.3)
direct = Dict{Symbol,Float64}()
PG.evaluate_parameter_gradients!(zeros(3, 2), direct, ligand_factors, probe_q,
                                 ctx2; parameters = ligand_params,
                                 parameter_ids = ligand_ids)
probe_direct = direct[:bond_k]

@printf("   HMC: dt=%.2f, L=%d, %d trajectories, acceptance %.4f\n",
        dt, nleap, nsteps, ligand_run.acceptance_rate)
@printf("   tau_int = %.2f, ESS = %.0f of %d, status = %s\n",
        ligand_sens.tau_int[:bond_k], ligand_sens.ess[:bond_k], nsteps,
        ligand_sens.status)
@printf("   <dU/dk> = %.6f +- %.6f   exact 1/(2*beta*k) = %.6f   z = %+.2f\n",
        k_mean, k_se, exact_dA, k_z)
@printf("   factor check: dU/dk at x = 1.3 is %.6f, (x-r0)^2/2 = %.6f\n",
        probe, probe_exact)
checkgate("factor dU/dk is (x-r0)^2/2", abs(probe - probe_exact) < 1e-14,
          "difference ", @sprintf("%.1e", abs(probe - probe_exact)))
checkgate("factor closure matches direct evaluator", probe == probe_direct,
          "closure ", @sprintf("%.6f", probe), ", direct ",
          @sprintf("%.6f", probe_direct))
checkgate("HMC sensitivity matches exact dA/dk", abs(k_z) < 5.0,
          "z = ", @sprintf("%+.2f", k_z), ", ESS = ",
          @sprintf("%.0f", ligand_sens.ess[:bond_k]))
checkgate("HMC result is not low-ESS", ligand_sens.status == :ok,
          "min_ess = 200, ESS = ",
          @sprintf("%.0f", ligand_sens.ess[:bond_k]))

# ---------------------------------------------------------------------------
# 3. Two-leg toy: a "complex" with two independent harmonic bonds sharing k and
#    r0, and a "ligand" with one.  The API computes only
#        mean complex - mean ligand + correction.
#    The correction is empty.  This is arithmetic, not a binding free energy.
# ---------------------------------------------------------------------------
println("\n[3] Toy two-leg difference: two harmonic bonds minus one")
ctx4 = F.Context(T.REDUCED, 4)
complex_factors = [F.HarmonicBond(1, 2, k0, r0),
                   F.HarmonicBond(3, 4, k0, r0)]

function q_of_complex(x)
    q = zeros(3, 4)
    q[1, 2] = x[1]
    q[1, 4] = x[2]
    return q
end

function U_complex(x::Vector{Float64})
    return F.total_energy_forces(complex_factors, q_of_complex(x), ctx4)[1]
end

function gradU_complex(x::Vector{Float64})
    Fv = F.total_energy_forces(complex_factors, q_of_complex(x), ctx4)[2]
    return [-Fv[1, 2], -Fv[1, 4]]
end

complex_run = SP.hmc_trajectory([0.5, -0.3], U_complex, gradU_complex,
                                [1.0, 1.0], dt, nleap, 30_000, beta,
                                MersenneTwister(202))

complex_ids = Dict{Tuple{Int,Symbol},Symbol}(
    (1, :k) => :bond_k, (1, :r0) => :bond_r0,
    (2, :k) => :bond_k, (2, :r0) => :bond_r0)
complex_params = Dict{Symbol,Float64}(:bond_k => k0, :bond_r0 => r0)
complex_grad = ES.factor_gradient_function(complex_factors, ctx4;
                                           parameter_ids = complex_ids,
                                           parameters = complex_params)
grad_fn_complex(x) = complex_grad(q_of_complex(x))

binding = ES.binding_sensitivity(complex_run.samples, ligand_run.samples,
                                 (grad_fn_complex, grad_fn_ligand);
                                 corrections = Dict{Symbol,Float64}(),
                                 min_ess = 200)

exact_delta = 2 * exact_dA - exact_dA       # two shared-k bonds minus one
delta_se = binding.se[:bond_k]
delta_z = (binding.means[:bond_k] - exact_delta) / delta_se

@printf("   complex: <dU/dk> = %.6f +- %.6f, ESS = %.0f\n",
        binding.complex.means[:bond_k], binding.complex.se[:bond_k],
        binding.complex.ess[:bond_k])
@printf("   ligand:  <dU/dk> = %.6f +- %.6f, ESS = %.0f\n",
        binding.ligand.means[:bond_k], binding.ligand.se[:bond_k],
        binding.ligand.ess[:bond_k])
@printf("   difference = %.6f +- %.6f   toy exact = %.6f   z = %+.2f\n",
        binding.means[:bond_k], delta_se, exact_delta, delta_z)
@printf("   corrections = %s (empty: no standard-state or restraint term)\n",
        binding.corrections)
println("   The 'complex' is two independent bonds, not a receptor-ligand system.")
println("   The API subtracts the two legs; it does not assert receptor identity,")
println("   does not add a standard-state term, and does not produce a binding")
println("   affinity.  The exact answer here is arithmetic on the harmonic model.")
checkgate("toy two-leg difference matches exact arithmetic", abs(delta_z) < 5.0,
          "z = ", @sprintf("%+.2f", delta_z))
checkgate("binding result is not low-ESS", binding.status == :ok,
          "status = ", binding.status)

# ---------------------------------------------------------------------------
# 4. Charge tangent projection.  This is a linear algebraic constraint on a
#    supplied group of charge-gradient ids, not a charge-transfer model.
# ---------------------------------------------------------------------------
println("\n[4] Charge tangent: subtract the group mean from each id")
charge_grad = Dict{Symbol,Float64}(:coulomb_qi_1 => 0.8,
                                   :coulomb_qj_1 => -0.3,
                                   :unrelated => 2.0)
charge_group = [:coulomb_qi_1, :coulomb_qj_1]
projected = ES.project_charge_tangent(charge_grad, [charge_group])
tangent_sum = projected[:coulomb_qi_1] + projected[:coulomb_qj_1]

@printf("   before: q_i=%.3f q_j=%.3f sum=%.3f\n",
        charge_grad[:coulomb_qi_1], charge_grad[:coulomb_qj_1],
        charge_grad[:coulomb_qi_1] + charge_grad[:coulomb_qj_1])
@printf("   after:  q_i=%.3f q_j=%.3f sum=%.2e (orthogonal to the sum-zero tangent)\n",
        projected[:coulomb_qi_1], projected[:coulomb_qj_1], tangent_sum)
println("   The non-charge id is untouched.  In binding_sensitivity the same")
println("   projection is applied per sample to both legs, and to an explicit")
println("   correction vector, so the reported difference is also tangent.")
checkgate("charge tangent has zero group sum", abs(tangent_sum) < 1e-14,
          "sum = ", @sprintf("%.1e", tangent_sum))
checkgate("charge tangent leaves unrelated ids untouched",
          projected[:unrelated] == charge_grad[:unrelated],
          "unrelated = ", projected[:unrelated])

# The same projection is a keyword of the two-leg API.  These gradients are
# fabricated (not sampled from an energy) solely to exercise that code path.
rng_toy = MersenneTwister(303)
n_toy = 500
c_charge = [Dict(:coulomb_qi_1 => 0.8 + 0.1randn(rng_toy),
                 :coulomb_qj_1 => -0.3 + 0.1randn(rng_toy),
                 :other => randn(rng_toy)) for _ in 1:n_toy]
l_charge = [Dict(:coulomb_qi_1 => 0.2 + 0.1randn(rng_toy),
                 :coulomb_qj_1 => -0.1 + 0.1randn(rng_toy),
                 :other => randn(rng_toy)) for _ in 1:n_toy]
binding_tangent = ES.binding_sensitivity(c_charge, l_charge;
                                         corrections = Dict(:coulomb_qi_1 => 0.2),
                                         charge_tangent_groups = [charge_group],
                                         min_ess = 0)
binding_tangent_sum = binding_tangent.means[:coulomb_qi_1] +
                      binding_tangent.means[:coulomb_qj_1]
@printf("   binding with charge_tangent_groups: difference group sum = %.2e\n",
        binding_tangent_sum)
checkgate("binding projects both legs and correction",
          abs(binding_tangent_sum) < 1e-12,
          "sum = ", @sprintf("%.1e", binding_tangent_sum))

# ---------------------------------------------------------------------------
# 5. Reading the result honestly.
# ---------------------------------------------------------------------------
println("\n[5] What is shown, and what is not")
println("   shown by measurement:")
println("    - the exact harmonic dA/dk agrees with a central difference of A(k);")
println("    - real HMC samples give <dU/dk> consistent with that exact value, with")
println("      an error bar built from the chain's own ESS rather than from N;")
println("    - the factor closure returns the same dU/dtheta as the direct")
println("      ParameterGradients evaluator, including shared parameter ids;")
println("    - the two-leg API computes exactly the requested arithmetic difference,")
println("      with independent-leg standard errors in quadrature;")
println("    - charge tangent projection removes the group mean and reports no")
println("      silent change to unrelated ids.")
println("   not shown, and not implied:")
println("    - finite free-energy changes for finite parameter moves.  A derivative")
println("      at k0 is not Delta A; integration over the path is not implemented.")
println("    - parameter-dependent measures, restraints, Jacobians, or domains.")
println("    - a covariance matrix across parameters: each component has its own tau")
println("      and ESS, and only marginal standard errors are reported.")
println("    - a physical binding affinity, receptor cancellation, solvent, or a")
println("      standard-state correction.  The two-leg section is a toy.")
println("    - convergence.  A large ESS can coexist with a biased, trapped chain;")
println("      ESS measures noise, not accuracy.")

println()
println("=" ^ 78)
@printf(" RESULT: %s   %d gate(s) failed\n", nfail == 0 ? "PASS" : "FAIL", nfail)
println("=" ^ 78)
exit(nfail == 0 ? 0 : 1)
