"""Matched, bounded OpenMM checks for FoundationCore LJ and direct Ewald.

Run with the repository's explicit Python environment. FoundationCore is called
through Julia; both engines receive the constants declared in this file.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
JULIA_DEFAULT = Path.home() / ".cache/biology-mm/julia-1.13.3/bin/julia"
PYTHON_DEFAULT = Path.home() / ".cache/biology-mm/.venv/bin/python"
LJ_TOL = (2e-10, 2e-9)  # absolute kJ/mol and kJ/mol/nm
PME_TOL = (2e-5, 2e-4)  # absolute kJ/mol and kJ/mol/nm

BOX = (3.0, 3.0, 3.0)  # nm
POSITIONS = ((0.35, 0.45, 0.55), (1.05, 0.60, 0.50), (0.55, 1.25, 0.75))
EPSILON = (0.70, 0.90, 0.60)  # kJ/mol
SIGMA = (0.32, 0.29, 0.34)  # nm
CHARGES = (1.0, -0.4, -0.6)  # e; neutral
ALPHA = 3.0  # nm^-1
RCUT = 1.4  # nm
KMAX = 16
PME_GRID = (128, 128, 128)
COULOMB = 138.935456  # kJ nm mol^-1 e^-2


def julia_payload() -> str:
    q = "[" + "; ".join(
        " ".join(f"{POSITIONS[i][axis]:.17g}" for i in range(3))
        for axis in range(3)
    ) + "]"
    return f'''include(joinpath(raw"{ROOT / 'julia/src/FoundationCore.jl'}"))
using .FoundationCore
q = {q}
box = FoundationCore.Types.OrthorhombicBox(collect({BOX}))
lj = FoundationCore.StreamedLJ
ws = lj.LJWorkspace(q, box, {RCUT})
le, lf, _ = lj.evaluate!(ws, q, collect({EPSILON}), collect({SIGMA}), box; rcut={RCUT}, mode=:lfs)
println("LJ|", repr(le), "|", join(vec(lf), ","))
ee, ef = FoundationCore.Electrostatics.evaluate_ewald(q, collect({CHARGES}), collect({BOX}); alpha={ALPHA}, rcut={RCUT}, kmax={KMAX}, prefactor={COULOMB})
println("EWALD|", repr(ee), "|", join(vec(ef), ","))
'''


def parse_julia(stdout: str) -> dict[str, tuple[float, list[float]]]:
    results = {}
    for line in stdout.splitlines():
        fields = line.split("|")
        if len(fields) == 3 and fields[0] in {"LJ", "EWALD"}:
            if fields[0] in results:
                raise RuntimeError(f"duplicate Julia result record: {fields[0]}")
            results[fields[0]] = (float(fields[1]), [float(x) for x in fields[2].split(",")])
    if set(results) != {"LJ", "EWALD"}:
        raise RuntimeError(f"Julia output did not contain both result records: {stdout}")
    return results


def openmm_lj(openmm, unit):
    system = openmm.System()
    for _ in POSITIONS:
        system.addParticle(1.0)
    force = openmm.CustomNonbondedForce(
        "4*sqrt(e1*e2)*((s/r)^12-(s/r)^6) "
        "- 4*sqrt(e1*e2)*((s/rc)^12-(s/rc)^6) "
        "- 24*sqrt(e1*e2)*((s/rc)^6-2*(s/rc)^12)/rc*(r-rc); "
        "s=(s1+s2)/2"
    )
    force.addPerParticleParameter("e")
    force.addPerParticleParameter("s")
    force.addGlobalParameter("rc", RCUT)
    force.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
    force.setCutoffDistance(RCUT)
    force.setUseSwitchingFunction(False)
    force.setUseLongRangeCorrection(False)
    for e, s in zip(EPSILON, SIGMA):
        force.addParticle([e, s])
    system.addForce(force)
    return evaluate_openmm(openmm, unit, system)


def openmm_pme(openmm, unit):
    system = openmm.System()
    force = openmm.NonbondedForce()
    for charge in CHARGES:
        system.addParticle(1.0)
        force.addParticle(charge, 1.0, 0.0)
    force.setNonbondedMethod(openmm.NonbondedForce.PME)
    force.setCutoffDistance(RCUT)
    force.setEwaldErrorTolerance(1e-7)
    force.setPMEParameters(ALPHA, *PME_GRID)
    system.addForce(force)
    return evaluate_openmm(openmm, unit, system)


def evaluate_openmm(openmm, unit, system):
    vectors = [openmm.Vec3(BOX[0], 0, 0), openmm.Vec3(0, BOX[1], 0), openmm.Vec3(0, 0, BOX[2])]
    system.setDefaultPeriodicBoxVectors(*(v * unit.nanometer for v in vectors))
    integrator = openmm.VerletIntegrator(0.001)
    context = openmm.Context(system, integrator, openmm.Platform.getPlatformByName("Reference"))
    context.setPositions([openmm.Vec3(*p) * unit.nanometer for p in POSITIONS])
    state = context.getState(getEnergy=True, getForces=True)
    energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    forces = state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
    flat = [float(forces[i, axis]) for i in range(3) for axis in range(3)]
    del context, integrator
    return float(energy), flat


def compare(name, reference, actual, tolerances):
    if len(reference[1]) != len(actual[1]) or not reference[1]:
        raise ValueError(f"{name}: force-array length mismatch: {len(reference[1])} vs {len(actual[1])}")
    de = abs(reference[0] - actual[0])
    df = max(abs(a - b) for a, b in zip(reference[1], actual[1]))
    ok = de <= tolerances[0] and df <= tolerances[1]
    print(f"{name}: {'PASS' if ok else 'FAIL'} | dE={de:.9g} (tol {tolerances[0]:g}) kJ/mol | max_dF={df:.9g} (tol {tolerances[1]:g}) kJ/mol/nm")
    return ok


def main() -> int:
    julia = Path(os.environ.get("JULIA_BIN", JULIA_DEFAULT))
    configured_python = Path(os.environ.get("PYTHON_BIN", PYTHON_DEFAULT))
    print(f"Julia interpreter: {julia}")
    print(f"Python interpreter: {sys.executable} (configured: {configured_python})")
    if Path(sys.executable).resolve() != configured_python.resolve():
        print("OPENMM MATCHED MODELS: BLOCKED (must run with the configured absolute Python interpreter)")
        return 2
    if not julia.is_file() or not os.access(julia, os.X_OK):
        print(f"OPENMM MATCHED MODELS: BLOCKED (Julia executable unavailable: {julia})")
        return 2
    try:
        import openmm
        from openmm import unit
    except Exception as exc:  # noqa: BLE001
        print(f"OPENMM MATCHED MODELS: BLOCKED (OpenMM import: {type(exc).__name__}: {exc})")
        return 2
    try:
        run = subprocess.run([str(julia), "--startup-file=no", "-e", julia_payload()],
                             cwd=ROOT / "julia", text=True, capture_output=True, timeout=45)
    except subprocess.TimeoutExpired:
        print("OPENMM MATCHED MODELS: BLOCKED (Julia fixture exceeded 45-second bound)")
        return 2
    if run.returncode:
        print(f"OPENMM MATCHED MODELS: BLOCKED (Julia evaluator failed, exit {run.returncode})")
        print(run.stderr.strip())
        return 2
    try:
        reference = parse_julia(run.stdout)
        print(f"OpenMM version: {openmm.__version__}; platform: Reference")
        print(f"LJ: LFS; box={BOX} nm; cutoff={RCUT} nm; explicit per-particle epsilon/sigma")
        print(f"PME: neutral charges={CHARGES}; alpha={ALPHA} nm^-1; cutoff={RCUT} nm; grid={PME_GRID}; direct Ewald kmax={KMAX}")
        results = [compare("Matched LJ", reference["LJ"], openmm_lj(openmm, unit), LJ_TOL),
                   compare("Direct Ewald vs PME", reference["EWALD"], openmm_pme(openmm, unit), PME_TOL)]
    except Exception as exc:  # noqa: BLE001
        print(f"OPENMM MATCHED MODELS: FAIL ({type(exc).__name__}: {exc})")
        return 1
    print("OPENMM MATCHED MODELS:", "PASS" if all(results) else "FAIL")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
