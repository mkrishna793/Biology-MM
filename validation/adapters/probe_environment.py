"""Probe which optional scientific dependencies are actually importable.

This module is deliberately the *only* place that decides what is available. The
rest of the validation code asks it and reports a status, so there is one honest
answer to "is X installed" rather than scattered try/except blocks with different
behaviour.

Status vocabulary, used consistently across the validation matrix:

  AVAILABLE      importable now, version reported
  BLOCKED        installable in principle but not installed here; the exact
                 blocker is recorded
  UNSUPPORTED    cannot work on this platform/version combination

Run:  python3 validation/adapters/probe_environment.py
"""

from __future__ import annotations

import importlib
import json
import platform
import sys
from typing import Any, Dict, List

# (import name, human name, why it matters, install hint)
_PROBES = [
    ("numpy", "numpy", "arrays, reference numerics", "pip install numpy"),
    ("scipy", "scipy", "quadrature, optimisation, sparse solves", "pip install scipy"),
    ("MDAnalysis", "MDAnalysis", "PDB/mmCIF parsing, trajectory I/O, order parameters",
     "pip install MDAnalysis"),
    ("mdtraj", "MDTraj", "trajectory I/O and RMSD/order-parameter pipeline",
     "pip install mdtraj"),
    ("physical_validation", "physical_validation",
     "energy/kinetic/ensemble consistency diagnostics", "pip install physical_validation"),
    ("openmm", "OpenMM", "mature MD engine; AMOEBA and Drude polarizable force fields",
     "conda install -c conda-forge openmm"),
    ("openff.toolkit", "OpenFF Toolkit", "molecular parameterisation from SMILES",
     "pip install openff-toolkit"),
    ("openff.evaluator", "OpenFF Evaluator",
     "property calculation against ThermoML experimental data",
     "pip install openff-evaluator  (heavy: conda recommended)"),
    ("pandas", "pandas", "tabular results", "pip install pandas"),
    ("pytest", "pytest", "test runner", "pip install pytest"),
]


def probe_one(import_name: str) -> Dict[str, Any]:
    try:
        mod = importlib.import_module(import_name)
    except Exception as exc:  # noqa: BLE001 - we deliberately record any failure
        return {"status": "BLOCKED", "version": None,
                "error": f"{type(exc).__name__}: {exc}"}
    ver = getattr(mod, "__version__", None)
    if ver is None:
        # some packages hide the version; report what we can rather than guessing
        ver = getattr(getattr(mod, "version", None), "version", "unknown")
    return {"status": "AVAILABLE", "version": str(ver), "error": None}


def probe_all() -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for import_name, human, role, hint in _PROBES:
        r = probe_one(import_name)
        r.update({"name": human, "role": role, "install": hint})
        out[import_name] = r
    return out


def summary_lines() -> List[str]:
    lines = []
    res = probe_all()
    for import_name, r in res.items():
        ver = r["version"] or "-"
        lines.append(f"{r['status']:<12} {r['name']:<20} {ver:<14} {r['role']}")
        if r["status"] != "AVAILABLE":
            lines.append(f"{'':<12} -> blocker: {r['error']}")
            lines.append(f"{'':<12} -> install: {r['install']}")
    return lines


def environment() -> Dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


def main() -> int:
    env = environment()
    print("== environment ==")
    for k, v in env.items():
        print(f"  {k:<12} {v}")
    print()
    print("== optional dependencies ==")
    for line in summary_lines():
        print("  " + line)
    print()
    res = probe_all()
    n_avail = sum(1 for r in res.values() if r["status"] == "AVAILABLE")
    print(f"== {n_avail}/{len(res)} optional dependencies available ==")
    out = {"environment": env, "dependencies": res}
    print()
    print("JSON:")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
