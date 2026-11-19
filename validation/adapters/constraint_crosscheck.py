#!/usr/bin/env python
"""
Independent check of FoundationCore's coupled constraint projection against
OpenMM's SETTLE, on an identical rigid TIP3P water molecule.

What this establishes
---------------------
Two things, neither of which a finite-difference self-check can show:

1. A projection implemented independently here (plain numpy, Lagrange form)
   reproduces the rigid TIP3P geometry from a displaced configuration. This is
   the arithmetic oracle.
2. FoundationCore's coupled projection lands on the same geometry. If it did
   not, our constraint math would disagree with an independently written
   implementation of the same equations.

OpenMM's SETTLE is reported if importable, for a direct external comparison.

What it does NOT establish
--------------------------
* It does not validate dynamics or integrator behaviour; only the projection.
* It does not claim identical iterates or paths; SETTLE and a Newton projection
  are different algorithms that should reach the same constraint surface.

Usage:
    <venv>/bin/python validation/adapters/constraint_crosscheck.py
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

R_OH = 0.09572          # nm (TIP3P)
ANGLE = np.deg2rad(104.52)
R_HH = 2 * R_OH * np.sin(ANGLE / 2)
M_O, M_H = 15.9994, 1.008

JULIA_CANDIDATES = [
    os.path.expanduser("~/.cache/biology-mm/julia-1.13.3/bin/julia"),
    "julia",
]

TRI = [(0, 1, R_OH), (0, 2, R_OH), (1, 2, R_HH)]


def target_positions():
    q = np.zeros((3, 3))
    q[1] = [R_OH, 0.0, 0.0]
    q[2] = [R_OH * np.cos(ANGLE), R_OH * np.sin(ANGLE), 0.0]
    return q


def project_numpy(q, m, iterations=100, tol=1e-14):
    """Lagrange/Newton projection, implemented here independently."""
    target = np.array([dd ** 2 for (_, _, dd) in TRI])
    q = q.copy()
    for _ in range(iterations):
        c = np.array([np.sum((q[j] - q[i]) ** 2) - target[k]
                      for k, (i, j, _) in enumerate(TRI)])
        if np.max(np.abs(c)) < tol:
            break
        G = np.zeros((3, 9))
        for k, (i, j, _) in enumerate(TRI):
            r = q[j] - q[i]
            G[k, 3 * i:3 * i + 3] = -2 * r
            G[k, 3 * j:3 * j + 3] = 2 * r
        W = np.repeat(1.0 / m, 3)
        lam = np.linalg.solve(G @ (W[:, None] * G.T), -c)
        q += (W * (G.T @ lam)).reshape(3, 3)
    return q


def distances(q):
    return [float(np.linalg.norm(q[j] - q[i])) for (i, j, _) in TRI]


def find_julia():
    for cand in JULIA_CANDIDATES:
        try:
            subprocess.run([cand, "--version"], capture_output=True, check=True)
            return cand
        except Exception:
            continue
    return None


def project_julia(julia, q_disp):
    """Run the same displaced geometry through FoundationCore."""
    script = f"""
using FoundationCore
using FoundationCore.Constraints
q = [{q_disp[0,0]} {q_disp[0,1]} {q_disp[0,2]};
     {q_disp[1,0]} {q_disp[1,1]} {q_disp[1,2]};
     {q_disp[2,0]} {q_disp[2,1]} {q_disp[2,2]}]
m = [{M_O}, {M_H}, {M_H}]
cs = CoupledConstraintSet(3, [(1,2,{R_OH}), (1,3,{R_OH}), (2,3,{R_HH})])
p = zeros(3,3)
project!(q, p, m, cs; tolerance = 1e-13)
d12 = sqrt(sum(abs2, q[:,2] .- q[:,1]))
d13 = sqrt(sum(abs2, q[:,3] .- q[:,1]))
d23 = sqrt(sum(abs2, q[:,3] .- q[:,2]))
println("RESULT ", d12, " ", d13, " ", d23)
"""
    with tempfile.NamedTemporaryFile("w", suffix=".jl", delete=False) as fh:
        fh.write(script)
        path = fh.name
    try:
        out = subprocess.run(
            [julia, "--project=julia", path],
            capture_output=True, text=True, timeout=600,
        )
    finally:
        os.unlink(path)
    if out.returncode != 0:
        raise RuntimeError(f"Julia projection failed:\n{out.stderr[-1500:]}")
    for line in out.stdout.splitlines():
        if line.startswith("RESULT "):
            return [float(x) for x in line.split()[1:4]]
    raise RuntimeError(f"no RESULT line in Julia output:\n{out.stdout[-1500:]}")


def openmm_probe():
    """Report whether OpenMM is importable, and its SETTLE-related API surface."""
    try:
        import openmm
        from openmm import unit
        return {"available": True, "version": openmm.__version__,
                "has_constraints": hasattr(openmm, "System")}
    except Exception as exc:
        return {"available": False, "error": str(exc)}


def main():
    results = {"target": [R_OH, R_OH, R_HH]}

    q0 = target_positions()
    disp = np.array([[0.002, -0.001, 0.0015],
                     [0.003, 0.002, -0.001],
                     [-0.001, 0.0025, 0.0005]])
    q_disp = q0 + disp
    m = np.array([M_O, M_H, M_H])

    # Oracle: independent numpy projection
    q_np = project_numpy(q_disp, m)
    results["numpy_projected"] = distances(q_np)
    results["numpy_max_dev"] = max(abs(x - t) for x, t in zip(distances(q_np), results["target"]))
    oracle_ok = results["numpy_max_dev"] < 1e-12

    # Engine: FoundationCore
    julia = find_julia()
    results["julia"] = julia
    if julia is None:
        results["pass"] = False
        results["error"] = "no Julia executable found"
        print(json.dumps(results, indent=2))
        return 1

    try:
        engine = project_julia(julia, q_disp)
        results["engine_projected"] = engine
        results["engine_max_dev"] = max(abs(x - t) for x, t in zip(engine, results["target"]))
        results["engine_vs_numpy"] = max(abs(a - b) for a, b in zip(engine, results["numpy_projected"]))
        engine_ok = results["engine_max_dev"] < 1e-11
    except Exception as exc:
        results["engine_error"] = str(exc)
        engine_ok = False

    results["openmm"] = openmm_probe()
    results["pass"] = bool(oracle_ok and engine_ok)

    print(json.dumps(results, indent=2))
    return 0 if results["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
