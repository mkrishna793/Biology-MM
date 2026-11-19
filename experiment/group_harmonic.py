"""Group-harmonic interaction: direct O(n*m) pair enumeration versus the exact
O(n+m) centroid/residual closed form.

Model (finite, nonperiodic, two-group complete-bipartite quadratic):

    U = (k/2) * sum_{i in A} sum_{j in B} ||q_i - q_j||^2 ,   k >= 0

with groups A (n sites) and B (m sites), q in R^3.  Writing centroids
a = mean(A), b = mean(B) and centered residuals da_i = q_i - a,
db_j = q_j - b, the algebraic identity

    sum_{ij} ||q_i - q_j||^2 = n*m*||a - b||^2 + m*sum_i ||da_i||^2
                                                + n*sum_j ||db_j||^2

gives the "summarized" closed form.  Exact forces from the same potential are

    F_i = -k*m*(q_i - b)   (i in A) ,
    F_j = -k*n*(q_j - a)   (j in B) ,

and sum_i F_i + sum_j F_j = 0 exactly.

The two paths below are deliberately independent implementations:

* ``direct_*``   : enumerates all n*m pairs, knows nothing about centroids.
* ``summarized_*``: O(n+m) centroid/residual reductions only.

``exact_*`` repeats the algebra over ``fractions.Fraction`` so that the
direct-versus-summarized agreement can be checked with zero tolerance.  The
module has no import-time side effects; run it with ``python3 group_harmonic.py``
to print a comparison table.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np

_TINY = 1e-300


# ---------------------------------------------------------------------------
# shape helpers
# ---------------------------------------------------------------------------


def _as_float_matrix(X, name):
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("%s must have shape (N, 3), got %r" % (name, arr.shape))
    if arr.shape[0] < 1:
        raise ValueError("%s must contain at least one row" % name)
    return arr


def _frac(x):
    """Convert a scalar (int, numpy scalar, float, Fraction) to Fraction.

    ``Fraction(float)`` is exact for the binary float that is passed in, so
    integer inputs stay integers and float inputs are represented exactly.
    """
    if hasattr(x, "item"):        # numpy scalar -> python scalar
        x = x.item()
    return Fraction(x)


def _as_fraction_rows(X, name):
    rows = [list(r) for r in X]
    if not rows:
        raise ValueError("%s must contain at least one row" % name)
    out = []
    for r in rows:
        if len(r) != 3:
            raise ValueError("%s must have shape (N, 3)" % name)
        out.append([_frac(x) for x in r])
    return out


# ---------------------------------------------------------------------------
# direct pair enumeration
# ---------------------------------------------------------------------------


def direct_energy_forces(A, B, k):
    """O(n*m) enumeration over all pairs; returns (U, FA, FB) as float64.

    Convention: the (n, m, 3) array ``diff[i, j] = q_i - q_j`` is the only
    geometric input.  No centroid is ever formed.
    """
    A = _as_float_matrix(A, "A")
    B = _as_float_matrix(B, "B")
    k = float(k)

    diff = A[:, None, :] - B[None, :, :]          # (n, m, 3) pair array
    U = 0.5 * k * float(np.sum(diff * diff))
    FA = -k * np.sum(diff, axis=1)                # -k sum_j (q_i - q_j)
    FB = k * np.sum(diff, axis=0)                 # -k sum_i (q_j - q_i)
    return U, FA, FB


def exact_direct_energy_forces(A, B, k):
    """Exact-rational O(n*m) pair enumeration; returns (Fraction, FA, FB).

    ``FA``/``FB`` are lists of length n/m of 3-element lists of Fractions.
    """
    A_rows = _as_fraction_rows(A, "A")
    B_rows = _as_fraction_rows(B, "B")
    n = len(A_rows)
    m = len(B_rows)
    kf = _frac(k)

    total = Fraction(0)
    FA = [[Fraction(0), Fraction(0), Fraction(0)] for _ in range(n)]
    FB = [[Fraction(0), Fraction(0), Fraction(0)] for _ in range(m)]
    for i in range(n):
        qi = A_rows[i]
        for j in range(m):
            qj = B_rows[j]
            for d in range(3):
                delta = qi[d] - qj[d]
                total += delta * delta
                FA[i][d] -= kf * delta
                FB[j][d] += kf * delta
    U = kf * total / Fraction(2)
    return U, FA, FB


# ---------------------------------------------------------------------------
# summarized centroid/residual closed form
# ---------------------------------------------------------------------------


def summarized_energy_forces(A, B, k):
    """Exact closed form using only O(n+m) centroid/residual work."""
    A = _as_float_matrix(A, "A")
    B = _as_float_matrix(B, "B")
    k = float(k)
    n = A.shape[0]
    m = B.shape[0]

    a = np.sum(A, axis=0) / n
    b = np.sum(B, axis=0) / m
    da = A - a
    db = B - b
    ab = a - b

    U = 0.5 * k * (
        n * m * float(np.dot(ab, ab))
        + m * float(np.sum(da * da))
        + n * float(np.sum(db * db))
    )
    FA = -k * m * (A - b)
    FB = -k * n * (B - a)
    return U, FA, FB


def exact_energy_forces(A, B, k):
    """Exact-rational closed form; returns (Fraction, FA, FB)."""
    A_rows = _as_fraction_rows(A, "A")
    B_rows = _as_fraction_rows(B, "B")
    n = len(A_rows)
    m = len(B_rows)
    kf = _frac(k)

    a = [sum(A_rows[i][d] for i in range(n)) / n for d in range(3)]
    b = [sum(B_rows[j][d] for j in range(m)) / m for d in range(3)]

    ab2 = sum((a[d] - b[d]) * (a[d] - b[d]) for d in range(3))
    da2 = sum(
        (A_rows[i][d] - a[d]) * (A_rows[i][d] - a[d])
        for i in range(n)
        for d in range(3)
    )
    db2 = sum(
        (B_rows[j][d] - b[d]) * (B_rows[j][d] - b[d])
        for j in range(m)
        for d in range(3)
    )

    U = kf / Fraction(2) * (n * m * ab2 + m * da2 + n * db2)
    FA = [[-kf * m * (A_rows[i][d] - b[d]) for d in range(3)] for i in range(n)]
    FB = [[-kf * n * (B_rows[j][d] - a[d]) for d in range(3)] for j in range(m)]
    return U, FA, FB


# ---------------------------------------------------------------------------
# numeric gradient
# ---------------------------------------------------------------------------


def gradient_finite_difference(energy_fn, A, B, h=1e-6):
    """Central-difference gradient of ``energy_fn(A, B)`` w.r.t. all coords.

    Returns dU/dx of shape (n+m, 3), laid out as ``vstack([A, B])``.  Forces
    are the negative gradient, so compare this against ``-F``.
    """
    A0 = _as_float_matrix(A, "A")
    B0 = _as_float_matrix(B, "B")
    n = A0.shape[0]
    X = np.vstack([A0, B0])
    grad = np.empty_like(X)
    for i in range(X.shape[0]):
        for d in range(3):
            Xp = X.copy()
            Xm = X.copy()
            Xp[i, d] += h
            Xm[i, d] -= h
            ep = float(energy_fn(Xp[:n], Xp[n:]))
            em = float(energy_fn(Xm[:n], Xm[n:]))
            grad[i, d] = (ep - em) / (2.0 * h)
    return grad


# ---------------------------------------------------------------------------
# operation accounting
# ---------------------------------------------------------------------------


def operation_counts(n, m):
    """Counted algebraic work terms (NOT measured wall time).

    Conventions
    -----------
    * direct energy: one squared displacement per pair, n*m pairs.
    * direct forces: each pair contributes the 3-vector (q_i - q_j) once and
      it is accumulated into F_i and (negated) into F_j, i.e. 2*3 component
      updates per pair -> 2*n*m*3 component accumulations.
    * summarized: n residual rows + m residual rows (plus two centroids).
    * memory figures count the number of per-pair or per-atom vector entries
      that must be materialized, not bytes.
    """
    return {
        "pair_terms_direct": n * m,
        "energy_terms_direct": n * m,
        "force_components_direct": 2 * n * m * 3,
        "summary_terms": n + m,
        "pair_terms_summarized": 0,
        "memory_direct_pairs": n * m,
        "memory_summarized": n + m,
    }


# ---------------------------------------------------------------------------
# demo / comparison table
# ---------------------------------------------------------------------------

SIZES = [(1, 1), (1, 7), (7, 1), (8, 13), (64, 97), (1000, 1000)]
K_VALUES = [0.0, 1.0, 3.7]
GEOMETRIES = ["gaussian", "shifted_far", "coincident_centroids", "stretched"]

_STRETCH = np.array([1.0e3, 1.0, 1.0e-3])


def _make_geometries(n, m, rng):
    out = {}
    out["gaussian"] = (rng.standard_normal((n, 3)), rng.standard_normal((m, 3)))

    shift = np.array([3.0e3, -4.0e3, 1.2e4])
    out["shifted_far"] = (
        rng.standard_normal((n, 3)),
        rng.standard_normal((m, 3)) + shift,
    )

    A = rng.standard_normal((n, 3))
    B = rng.standard_normal((m, 3))
    B = B + (A.mean(axis=0) - B.mean(axis=0))       # centroids coincide exactly
    out["coincident_centroids"] = (A, B)

    out["stretched"] = (
        rng.standard_normal((n, 3)) * _STRETCH,
        rng.standard_normal((m, 3)) * _STRETCH,
    )
    return out


def _max_abs(arr):
    arr = np.asarray(arr, dtype=np.float64)
    return float(np.max(np.abs(arr))) if arr.size else 0.0


def _rel_err(x, y):
    denom = max(abs(float(x)), abs(float(y)))
    return 0.0 if denom == 0.0 else abs(float(x) - float(y)) / denom


def _force_rel_err(FA_a, FB_a, FA_b, FB_b):
    diff = max(_max_abs(np.asarray(FA_a) - np.asarray(FA_b)),
               _max_abs(np.asarray(FB_a) - np.asarray(FB_b)))
    scale = max(_max_abs(FA_a), _max_abs(FB_a),
                _max_abs(FA_b), _max_abs(FB_b), _TINY)
    return diff, diff / scale


def _fmt(x):
    if x == 0.0:
        return "0"
    return "%.2e" % x


def _sum_frac(rows):
    return sum((sum(r) for r in rows), Fraction(0))


def _print_float_table(rng):
    print("")
    print("=" * 118)
    print("FLOAT PATH: summarized O(n+m) closed form vs direct O(n*m) pair enumeration")
    print("Errors below are floating-point roundoff only; relative errors use max(|x|,|y|).")
    print("=" * 118)
    header = ("%5s %5s %6s %-21s %11s %10s %11s %10s %11s %11s"
              % ("n", "m", "k", "geometry", "max|dU|", "rel U",
                 "max|dF|", "rel F", "|sumF| dir", "|sumF| sum"))
    print(header)
    print("-" * len(header))
    for (n, m) in SIZES:
        geoms = _make_geometries(n, m, rng)
        for k in K_VALUES:
            for gname in GEOMETRIES:
                A, B = geoms[gname]
                Ud, FAd, FBd = direct_energy_forces(A, B, k)
                Us, FAs, FBs = summarized_energy_forces(A, B, k)
                dU = abs(Ud - Us)
                relU = _rel_err(Ud, Us)
                dF, relF = _force_rel_err(FAd, FBd, FAs, FBs)
                sf_d = float(np.linalg.norm(np.sum(FAd, axis=0) + np.sum(FBd, axis=0)))
                sf_s = float(np.linalg.norm(np.sum(FAs, axis=0) + np.sum(FBs, axis=0)))
                print("%5d %5d %6.1f %-21s %11s %10s %11s %10s %11s %11s"
                      % (n, m, k, gname, _fmt(dU), _fmt(relU), _fmt(dF),
                         _fmt(relF), _fmt(sf_d), _fmt(sf_s)))


def _print_operation_table():
    print("")
    print("=" * 118)
    print("COUNTED ALGEBRAIC WORK (not wall time; see operation_counts docstring)")
    print("=" * 118)
    header = ("%5s %5s %14s %16s %16s %14s %16s %13s"
              % ("n", "m", "pair_terms_dir", "energy_terms_dir",
                 "force_comp_dir", "summary_terms", "pair_terms_sum", "memory_ratio"))
    print(header)
    print("-" * len(header))
    for (n, m) in SIZES:
        c = operation_counts(n, m)
        ratio = float(c["pair_terms_direct"]) / float(c["summary_terms"])
        print("%5d %5d %14d %16d %16d %14d %16d %13.2f"
              % (n, m, c["pair_terms_direct"], c["energy_terms_direct"],
                 c["force_components_direct"], c["summary_terms"],
                 c["pair_terms_summarized"], ratio))


def _fd_energy(A, B, k):
    return summarized_energy_forces(A, B, k)[0]


def _print_fd_table(rng):
    print("")
    print("=" * 118)
    print("FINITE-DIFFERENCE CHECK: d(summarized U)/dx vs -F (forces are F = -dU/dx)")
    print("Central differences; h is absolute and is scaled with the geometry so that")
    print("h/|x| stays about 1e-6 on stretched configs. Relative error is L2(numeric - (-F))/L2(F).")
    print("=" * 118)
    header = ("%5s %5s %6s %-21s %9s %14s %14s"
              % ("n", "m", "k", "geometry", "h", "rel grad err", "max|F|"))
    print(header)
    print("-" * len(header))
    mild = np.array([1.0e2, 1.0, 1.0e-2])
    for (n, m) in SIZES:
        geoms = _make_geometries(n, m, rng)
        local = {
            "gaussian": geoms["gaussian"],
            "stretched_1e2": (geoms["gaussian"][0] * mild,
                              geoms["gaussian"][1] * mild),
            "coincident_centroids": geoms["coincident_centroids"],
        }
        cases = [(3.7, "gaussian", 1e-6), (1.0, "gaussian", 1e-6),
                 (3.7, "stretched_1e2", 1e-4)]
        if n * m > 1:
            cases.append((3.7, "coincident_centroids", 1e-6))
        for k, gname, h in cases:
            A, B = local[gname]
            _, FAs, FBs = summarized_energy_forces(A, B, k)
            neg_force = -np.vstack([FAs, FBs])
            numeric = gradient_finite_difference(
                lambda X, Y: _fd_energy(X, Y, k), A, B, h=h)
            diff = float(np.linalg.norm(numeric - neg_force))
            scale = max(float(np.linalg.norm(neg_force)), _TINY)
            print("%5d %5d %6.1f %-21s %9.0e %14s %14s"
                  % (n, m, k, gname, h, _fmt(diff / scale),
                     _fmt(float(np.max(np.abs(neg_force))))))


def _integer_configs(rng):
    """Small integer-coordinate configs for the exact-rational cross-check."""
    configs = []
    for (n, m) in [(1, 1), (2, 3), (3, 2), (4, 4), (5, 6), (6, 6)]:
        A = rng.integers(-3, 4, size=(n, 3))
        B = rng.integers(-3, 4, size=(m, 3))
        configs.append((n, m, A, B, "integer_random"))
    for (n, m) in [(1, 1), (2, 6), (3, 3), (6, 6)]:
        A = rng.integers(-3, 4, size=(n, 3))
        B = rng.integers(-3, 4, size=(m, 3))
        ratio = m // n                                  # m is a multiple of n here
        target = ratio * A.sum(axis=0)
        B = B.copy()
        B[0] = target - (B.sum(axis=0) - B[0])          # force sum(B) = target
        configs.append((n, m, A, B, "integer_coincident"))
    return configs


def _print_exact_table(rng):
    print("")
    print("=" * 118)
    print("EXACT-RATIONAL CROSS-CHECK: Fraction equality of direct vs summarized (zero tolerance)")
    print("=" * 118)
    header = ("%5s %5s %10s %-21s %8s %10s %10s %12s %12s"
              % ("n", "m", "k", "geometry", "U eq", "FA eq", "FB eq",
                 "sumF direct", "sumF sum"))
    print(header)
    print("-" * len(header))
    k_values = [Fraction(0), Fraction(1), Fraction(7, 3), Fraction(5, 2)]
    failures = 0
    for (n, m, A, B, label) in _integer_configs(rng):
        for k in k_values:
            Ud, FAd, FBd = exact_direct_energy_forces(A, B, k)
            Us, FAs, FBs = exact_energy_forces(A, B, k)
            u_eq = (Ud == Us)
            fa_eq = (FAd == FAs)
            fb_eq = (FBd == FBs)
            sf_d = _sum_frac(FAd) + _sum_frac(FBd)
            sf_s = _sum_frac(FAs) + _sum_frac(FBs)
            if not (u_eq and fa_eq and fb_eq):
                failures += 1
            print("%5d %5d %10s %-21s %8s %10s %10s %12s %12s"
                  % (n, m, k, label,
                     "EXACT" if u_eq else "MISMATCH",
                     "EXACT" if fa_eq else "MISMATCH",
                     "EXACT" if fb_eq else "MISMATCH",
                     str(sf_d), str(sf_s)))
    if failures:
        raise AssertionError("exact rational cross-check failed for %d cases" % failures)
    print("All exact-rational cases matched with Fraction equality (zero tolerance).")


def main():
    rng = np.random.default_rng(20261111)
    print("Group-harmonic model:", "U = (k/2) sum_{i in A} sum_{j in B} ||q_i - q_j||^2")
    print("Direct path enumerates all n*m pairs; summarized path uses centroids/residuals only.")
    _print_float_table(rng)
    _print_operation_table()
    _print_fd_table(rng)
    _print_exact_table(rng)
    print("")
    print("Notes: float agreement is roundoff-limited; the rational path is exact.")
    print("The O(n*m) -> O(n+m) reduction applies to pair evaluations only; atom-state")
    print("reads/writes and the output force array remain Omega(n+m).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
