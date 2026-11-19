# Does the Lean shadow-invariant theorem describe the code's actual integrator?
#
# The Lean theorem `Foundation.shadow_invariant` proves, in exact rational
# arithmetic, that with D = w2*(1 - h^2*w2/4), the quadratic form
#     Htilde = p^2/2 + D*q^2/2
# is invariant under one velocity-Verlet step. This script checks the same
# statement against the *compiled Julia integrator*, so the proof and the
# implementation are tied together rather than living in separate worlds.
#
# Two independent checks:
#   1. arbitrary-precision (BigFloat) evaluation of the algebraic identity, so
#      roundoff cannot mask a discrepancy;
#   2. a 200k-step run of the real Julia integrator, where the invariant must hold
#      to float64 roundoff and the true energy must stay inside the predicted
#      O(h^2) band.

using FoundationCore
using FoundationCore.Types
using FoundationCore.Factors
using FoundationCore.Dynamics
using Printf, Random

const T = FoundationCore.Types
const FA = FoundationCore.Factors
const DY = FoundationCore.Dynamics

function one_step(q, p, h, w2)
    ph = p - (h / 2) * w2 * q
    qn = q + h * ph
    pn = ph - (h / 2) * w2 * qn
    return qn, pn
end

shadowH(D, q, p) = p^2 / 2 + D * q^2 / 2
harmH(w2, q, p) = p^2 / 2 + w2 * q^2 / 2

function test_algebraic_identity_bigfloat()
    Random.seed!(4)
    worst_rel = big(0.0)
    worst_defect = big(0.0)
    for trial in 1:200
        h  = big(rand()) * big(0.4)
        w2 = big(rand()) * big(3.0)
        q  = big(randn())
        p  = big(randn())
        D  = w2 * (1 - h^2 * w2 / 4)
        qn, pn = one_step(q, p, h, w2)
        lhs = shadowH(D, qn, pn)
        rhs = shadowH(D, q, p)
        worst_rel = max(worst_rel, abs(lhs - rhs) / max(abs(rhs), big(1e-30)))
        l2 = harmH(w2, qn, pn)
        r2 = shadowH(D, qn, pn) + h^2 * w2^2 / 8 * qn^2
        worst_defect = max(worst_defect, abs(l2 - r2) / max(abs(l2), big(1e-30)))
    end
    @printf("  shadow invariant, worst relative residual (200 random cases, BigFloat): %s\n",
            string(worst_rel))
    @printf("  energy defect identity, worst relative residual:                       %s\n",
            string(worst_defect))
    @assert worst_rel < big(1e-60) "shadow invariant fails algebraically"
    @assert worst_defect < big(1e-60) "energy-defect identity fails algebraically"
    return true
end

function test_compiled_integrator_obeys_shadow()
    w2 = 2.3
    k = w2 / 2
    h = 0.05
    q = zeros(3, 2); q[1, 1] = -0.5; q[1, 2] = 0.5
    p = zeros(3, 2)
    sys = DY.HarmonicSystem([FA.HarmonicBond(1, 2, k, 0.0)], [1.0, 1.0], FA.Context(T.REDUCED, 2))
    st = DY.State(q, p)
    st.p[1, 1] = 0.13; st.p[1, 2] = -0.13
    D = w2 * (1 - h^2 * w2 / 4)
    x0 = st.q[1, 2] - st.q[1, 1]
    p0 = st.p[1, 2] - st.p[1, 1]
    H0 = shadowH(D, x0, p0)
    n = 200_000
    worstH = 0.0
    Emin = Inf; Emax = -Inf
    maxq2 = 0.0
    for _ in 1:n
        DY.velocity_verlet_step!(st, sys, h)
        x = st.q[1, 2] - st.q[1, 1]
        pv = st.p[1, 2] - st.p[1, 1]
        worstH = max(worstH, abs(shadowH(D, x, pv) - H0) / abs(H0))
        E = harmH(w2, x, pv)
        Emin = min(Emin, E); Emax = max(Emax, E)
        maxq2 = max(maxq2, x^2)
    end
    predicted = h^2 * w2^2 / 8 * maxq2
    @printf("  compiled integrator, %d steps: shadow drift %.3e (relative)\n", n, worstH)
    @printf("  observed energy band width = %.3e\n", Emax - Emin)
    @printf("  theorem-predicted width h^2 w2^2/8 * max q^2 = %.3e (ratio %.3f)\n",
            predicted, (Emax - Emin) / predicted)
    @assert worstH < 1e-12 "compiled integrator does not conserve the shadow form"
    @assert (Emax - Emin) <= 2 * predicted + 1e-12 "energy escaped the predicted band"
    return true
end

for (name, fn) in [("algebraic_identity_bigfloat", test_algebraic_identity_bigfloat),
                   ("compiled_integrator_obeys_shadow", test_compiled_integrator_obeys_shadow)]
    println("TEST ", name)
    fn()
end
println("")
println("SHADOW THEOREM <-> IMPLEMENTATION: PASS")
