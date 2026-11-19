#!/usr/bin/env python3
"""Cross-check the Julia MBAR lane against pymbar on the *same* dataset.

Reads the artifacts written by `lane_mbar.jl` (`u_kn.csv`, `n_k.csv`,
`f_julia.csv`, `meta.txt`), solves the identical MBAR equations with pymbar,
and asserts that

  1. pymbar's Delta f agrees with the analytic 0.5*log(k) within 4 combined
     standard errors, and
  2. pymbar and the Julia fixed point agree to 1e-3 kT.

Exit status is nonzero on any failure, so `run_all.sh` stops.
"""

import json
import os
import sys

import numpy as np

try:
    import pymbar
except Exception as exc:  # pragma: no cover - recorded by the runner
    print(f"SKIP pymbar unavailable: {type(exc).__name__}: {exc}")
    sys.exit(2)

ARTIFACTS = os.environ.get(
    "VERIFICATION_ARTIFACTS",
    os.path.expanduser("~/.cache/biology-mm/verification-artifacts"),
)

u_kn = np.loadtxt(os.path.join(ARTIFACTS, "u_kn.csv"), delimiter=",")
N_k = np.loadtxt(os.path.join(ARTIFACTS, "n_k.csv"), delimiter=",").astype(int)
f_julia = np.loadtxt(os.path.join(ARTIFACTS, "f_julia.csv"), delimiter=",")
k_values = np.loadtxt(os.path.join(ARTIFACTS, "k_values.csv"), delimiter=",")

analytic = 0.5 * np.log(k_values)
mbar = pymbar.MBAR(u_kn, N_k, verbose=False)
result = mbar.compute_free_energy_differences()
delta_f = np.asarray(result["Delta_f"][0], dtype=float)
d_delta_f = np.asarray(result["dDelta_f"][0], dtype=float)

err_analytic = np.abs(delta_f - analytic)
err_julia = np.abs(delta_f - (f_julia - f_julia[0]))
tolerance_analytic = 4.0 * np.maximum(d_delta_f, 1e-3)

print("pymbar", pymbar.__version__, "numpy", np.__version__)
print("Delta_f pymbar :", np.array2string(delta_f, precision=5))
print("Delta_f analytic:", np.array2string(analytic, precision=5))
print("Delta_f julia  :", np.array2string(f_julia - f_julia[0], precision=5))
print("max |pymbar-analytic| =", float(err_analytic.max()),
      " tolerance =", float(tolerance_analytic.max()))
print("max |pymbar-julia|    =", float(err_julia.max()))

payload = {
    "pymbar_version": pymbar.__version__,
    "numpy_version": np.__version__,
    "delta_f_pymbar": delta_f.tolist(),
    "delta_f_analytic": analytic.tolist(),
    "max_abs_err_analytic": float(err_analytic.max()),
    "max_abs_err_julia": float(err_julia.max()),
    "max_stat_tolerance": float(tolerance_analytic.max()),
}
with open(os.path.join(ARTIFACTS, "pymbar_check.json"), "w") as handle:
    json.dump(payload, handle, indent=2)

ok = True
if not np.all(err_analytic <= tolerance_analytic):
    print("FAIL pymbar vs analytic exceeds 4 combined standard errors")
    ok = False
else:
    print("PASS pymbar vs analytic within 4 combined standard errors")
if err_julia.max() > 1e-3:
    print("FAIL pymbar vs Julia estimate exceeds 1e-3 kT")
    ok = False
else:
    print("PASS pymbar vs Julia estimate within 1e-3 kT")

sys.exit(0 if ok else 1)
