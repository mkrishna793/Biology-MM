"""Plain-assert tests for the group-harmonic experiment.

Run with:  python3 experiment/test_group_harmonic.py
No pytest dependency: every ``test_*`` function is executed by ``__main__``.
"""

from __future__ import annotations

import os
import sys
from fractions import Fraction

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from group_harmonic import (  # noqa: E402
    direct_energy_forces,
    exact_direct_energy_forces,
    exact_energy_forces,
    gradient_finite_difference,
    operation_counts,
    summarized_energy_forces,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _mat(rows):
    return [[Fraction(x) for x in row] for row in rows]


def _sum_frac(rows):
    return sum((sum(r) for r in rows), Fraction(0))


def _max_abs(arr):
    arr = np.asarray(arr, dtype=np.float64)
    return float(np.max(np.abs(arr))) if arr.size else 0.0


def _rel(a, b, tiny=1e-300):
    denom = max(abs(float(a)), abs(float(b)))
    return 0.0 if denom == 0.0 else abs(float(a) - float(b)) / max(denom, tiny)


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


def test_exact_rational_equality():
    """direct and summarized agree EXACTLY (Fraction equality, zero tolerance)."""
    cases = [
        (1, 1, Fraction(0)),
        (1, 1, Fraction(1)),
        (1, 1, Fraction(7, 3)),
        (1, 4, Fraction(3)),
        (2, 3, Fraction(0)),
        (3, 2, Fraction(7, 3)),
        (4, 4, Fraction(5, 2)),
        (6, 6, Fraction(1)),
        (5, 6, Fraction(4)),
    ]
    rng = np.random.default_rng(11)
    for (n, m, k) in cases:
        A = rng.integers(-3, 4, size=(n, 3)).tolist()
        B = rng.integers(-3, 4, size=(m, 3)).tolist()
        Ud, FAd, FBd = exact_direct_energy_forces(A, B, k)
        Us, FAs, FBs = exact_energy_forces(A, B, k)
        assert Ud == Us, "energy mismatch for n=%d m=%d k=%s: %s vs %s" % (n, m, k, Ud, Us)
        assert FAd == FAs, "group A force mismatch for n=%d m=%d k=%s" % (n, m, k)
        assert FBd == FBs, "group B force mismatch for n=%d m=%d k=%s" % (n, m, k)

    # coincident-centroid case constructed with integer arithmetic
    A = [[0, 0, 0], [2, 2, 2], [-1, 3, 0], [4, -2, 1]]
    # B has 8 points whose centroid equals A's centroid (1.25, 0.75, 0.75)
    B = [[1, 1, 1], [2, 0, 0], [0, 2, 1], [3, 3, 1],
         [-1, 0, 2], [2, -1, 0], [1, 1, 2], [2, 0, -1]]
    k = Fraction(7, 3)
    Ud, FAd, FBd = exact_direct_energy_forces(A, B, k)
    Us, FAs, FBs = exact_energy_forces(A, B, k)
    assert Ud == Us and FAd == FAs and FBd == FBs


def test_energy_nonnegative_and_zero_when_coincident():
    rng = np.random.default_rng(12)
    for (n, m, k) in [(1, 1, 1.0), (1, 7, 3.7), (8, 13, 0.0), (64, 97, 1.0)]:
        A = rng.standard_normal((n, 3))
        B = rng.standard_normal((m, 3))
        Ud, _, _ = direct_energy_forces(A, B, k)
        Us, _, _ = summarized_energy_forces(A, B, k)
        assert Ud >= 0.0 and Us >= 0.0, "negative energy for n=%d m=%d" % (n, m)

        c = np.array([1.5, -2.0, 0.25])
        Ac = np.tile(c, (n, 1))
        Bc = np.tile(c, (m, 1))
        Ud0, FAd0, FBd0 = direct_energy_forces(Ac, Bc, k)
        Us0, FAs0, FBs0 = summarized_energy_forces(Ac, Bc, k)
        assert Ud0 == 0.0 and Us0 == 0.0, "energy not exactly zero when coincident"
        assert _max_abs(FAd0) == 0.0 and _max_abs(FBd0) == 0.0
        assert _max_abs(FAs0) == 0.0 and _max_abs(FBs0) == 0.0

        Ue, FAe, FBe = exact_energy_forces(Ac.tolist(), Bc.tolist(), Fraction(k))
        assert Ue == Fraction(0)
        assert all(v == 0 for row in FAe for v in row)
        assert all(v == 0 for row in FBe for v in row)


def test_total_force_zero():
    rng = np.random.default_rng(13)
    for (n, m, k) in [(1, 1, 1.0), (2, 5, 3.7), (8, 13, 0.5), (40, 37, 1.0)]:
        A = rng.standard_normal((n, 3))
        B = rng.standard_normal((m, 3))
        for fn in (direct_energy_forces, summarized_energy_forces):
            _, FA, FB = fn(A, B, k)
            total = np.sum(FA, axis=0) + np.sum(FB, axis=0)
            scale = max(_max_abs(FA), _max_abs(FB), 1.0)
            assert float(np.linalg.norm(total)) <= 1e-12 * scale, \
                "total force residual too large in %s" % fn.__name__

    # exact rational: the total force is exactly zero
    A = [[1, 2, -3], [0, 0, 0], [4, -1, 2]]
    B = [[2, 1, 1], [-1, 3, 0], [0, -2, 4], [5, 1, -1]]
    for k in (Fraction(0), Fraction(1), Fraction(7, 3)):
        _, FAd, FBd = exact_direct_energy_forces(A, B, k)
        assert _sum_frac(FAd) + _sum_frac(FBd) == Fraction(0)
        assert _sum_frac(FAd) == -_sum_frac(FBd)
        _, FAs, FBs = exact_energy_forces(A, B, k)
        assert _sum_frac(FAs) + _sum_frac(FBs) == Fraction(0)
        assert _sum_frac(FAs) == -_sum_frac(FBs)


def test_finite_difference_matches_analytic():
    rng = np.random.default_rng(14)
    for (n, m, k) in [(1, 1, 2.5), (3, 4, 1.0), (5, 7, 3.7), (8, 13, 0.5)]:
        A = rng.standard_normal((n, 3))
        B = rng.standard_normal((m, 3))
        _, FAs, FBs = summarized_energy_forces(A, B, k)
        analytic = np.vstack([FAs, FBs])
        numeric = gradient_finite_difference(
            lambda X, Y: summarized_energy_forces(X, Y, k)[0], A, B, h=1e-6
        )
        # forces are F = -dU/dx, so the numerical gradient must equal -F
        diff = float(np.linalg.norm(numeric + analytic))
        scale = max(float(np.linalg.norm(analytic)), 1e-300)
        rel = diff / scale
        assert rel < 1e-5, "FD gradient relative error %.3e too large" % rel


def test_summarized_is_linear_in_k():
    rng = np.random.default_rng(15)
    for (n, m) in [(1, 1), (1, 7), (8, 13)]:
        A = rng.standard_normal((n, 3))
        B = rng.standard_normal((m, 3))
        k1, k2 = 1.3, 2.4
        U1, _, _ = summarized_energy_forces(A, B, k1)
        U2, _, _ = summarized_energy_forces(A, B, k2)
        U12, _, _ = summarized_energy_forces(A, B, k1 + k2)
        ref = max(abs(U1), abs(U2), abs(U12), 1e-300)
        assert abs(U12 - (U1 + U2)) <= 1e-12 * ref, "U not linear in k"

        # rigid translation of both groups leaves each decomposition term intact
        shift = np.array([1.0e3, -2.5e3, 0.75e3])
        A2, B2 = A + shift, B + shift
        U2s, _, _ = summarized_energy_forces(A2, B2, k1)
        assert _rel(U1, U2s) <= 1e-12, "energy not translation invariant"

        def terms(X, Y, k):
            a = X.mean(axis=0)
            b = Y.mean(axis=0)
            da = X - a
            db = Y - b
            ab = a - b
            nx, ny = X.shape[0], Y.shape[0]
            return (0.5 * k * nx * ny * float(np.dot(ab, ab)),
                    0.5 * k * ny * float(np.sum(da * da)),
                    0.5 * k * nx * float(np.sum(db * db)))

        t1 = terms(A, B, k1)
        t2 = terms(A2, B2, k1)
        for x, y in zip(t1, t2):
            assert _rel(x, y) <= 1e-12, "decomposition term not translation invariant"
        assert _rel(sum(t1), U1) <= 1e-12


def test_operation_counts():
    counts = operation_counts(64, 97)
    assert counts["pair_terms_direct"] == 64 * 97
    assert counts["energy_terms_direct"] == 64 * 97
    assert counts["force_components_direct"] == 2 * 64 * 97 * 3
    assert counts["summary_terms"] == 64 + 97
    assert counts["pair_terms_summarized"] == 0
    assert counts["memory_direct_pairs"] == 64 * 97
    assert counts["memory_summarized"] == 64 + 97
    assert 64 * 97 > 64 + 97

    large = operation_counts(1000, 1000)
    assert large["pair_terms_direct"] == 1000000
    assert large["summary_terms"] == 2000
    assert large["pair_terms_direct"] > large["summary_terms"]


def _lj_radial(r2, eps=1.0, sigma=1.0):
    r2 = np.maximum(np.asarray(r2, dtype=np.float64), 1e-12)
    inv6 = (sigma * sigma / r2) ** 3
    return 4.0 * eps * (inv6 * inv6 - inv6)


def _lj_pair_energy(A, B, eps=1.0, sigma=1.0):
    """Plain pairwise LJ 12-6 energy over the same bipartite pairs."""
    diff = A[:, None, :] - B[None, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    return float(np.sum(_lj_radial(r2, eps, sigma)))


def _lj_naive_closure(A, B, eps=1.0, sigma=1.0):
    """Apply the quadratic three-term bookkeeping to the LJ radial function.

    This is the natural (but wrong) guess that a potential depending only on
    ||q_i - q_j|| can be summarized from ||a - b||, ||da_i||, ||db_j|| the same
    way the squared distance can.  It is used only as a negative control.
    """
    n, m = A.shape[0], B.shape[0]
    a = A.mean(axis=0)
    b = B.mean(axis=0)
    da = A - a
    db = B - b
    ab2 = float(np.dot(a - b, a - b))
    return 0.5 * (
        n * m * float(_lj_radial(ab2, eps, sigma))
        + m * float(np.sum(_lj_radial(np.sum(da * da, axis=1), eps, sigma)))
        + n * float(np.sum(_lj_radial(np.sum(db * db, axis=1), eps, sigma)))
    )


def test_scope_limit_lennard_jones():
    """NEGATIVE CONTROL: the centroid/residual closed form is specific to the
    quadratic all-to-all model.  It does not reproduce Lennard-Jones, and any
    such pairwise/tabulated/cutoff potential breaks the O(n*m) -> O(n+m)
    reduction, because its value is not determined by the three quadratic
    invariants.
    """
    rng = np.random.default_rng(7)
    A = rng.standard_normal((6, 3))
    B = rng.standard_normal((8, 3)) + np.array([2.0, 0.5, -0.3])

    true_lj = _lj_pair_energy(A, B)
    closure = _lj_naive_closure(A, B)
    rel = abs(true_lj - closure) / max(abs(true_lj), abs(closure))
    assert rel > 0.1, "expected a large LJ closure error, got %.3e" % rel

    # Independent structural argument: rotating one group about its own centroid
    # preserves all quadratic invariants (so the quadratic U is unchanged) but
    # changes LJ pair distances, hence its energy.
    t = np.deg2rad(90.0)
    rot = np.array([[np.cos(t), -np.sin(t), 0.0],
                    [np.sin(t), np.cos(t), 0.0],
                    [0.0, 0.0, 1.0]])
    A_rot = (A - A.mean(axis=0)) @ rot.T + A.mean(axis=0)
    U_quad, FAs, FBs = summarized_energy_forces(A, B, 1.0)
    U_quad_rot = summarized_energy_forces(A_rot, B, 1.0)[0]
    assert _rel(U_quad, U_quad_rot) <= 1e-12, "quadratic U should be rotation invariant"
    lj_rot = _lj_pair_energy(A_rot, B)
    rot_rel = abs(true_lj - lj_rot) / max(abs(true_lj), abs(lj_rot))
    assert rot_rel > 0.1, "expected LJ energy to change under one-group rotation, got %.3e" % rot_rel
    assert _max_abs(FAs) > 0.0 and _max_abs(FBs) > 0.0


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------


def _run_all():
    tests = [
        ("exact_rational_equality", test_exact_rational_equality),
        ("energy_nonnegative_and_zero_when_coincident",
         test_energy_nonnegative_and_zero_when_coincident),
        ("total_force_zero", test_total_force_zero),
        ("finite_difference_matches_analytic", test_finite_difference_matches_analytic),
        ("summarized_is_linear_in_k", test_summarized_is_linear_in_k),
        ("operation_counts", test_operation_counts),
        ("scope_limit_lennard_jones", test_scope_limit_lennard_jones),
    ]
    for name, fn in tests:
        fn()
        print("PASS %s" % name)
    print("")
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    _run_all()
