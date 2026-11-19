"""Bounded explicit-parameter TIP3P site comparison with OpenMM.

This is a static two-water point-parameter fixture, not automatic chemical
parameterization or a general water/force-field validation. Core contributes
direct orthorhombic Ewald electrostatics and periodic force-shifted LJ; OpenMM
uses PME and an algebraically matched custom LJ force. Constraints/trajectory
are unavailable because FoundationCore's current RATTLE API refuses projection.
"""
from __future__ import annotations

import os
import math
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
JULIA_DEFAULT = Path.home() / ".cache/biology-mm/julia-1.13.3/bin/julia"
PYTHON_DEFAULT = Path.home() / ".cache/biology-mm/.venv/bin/python"

# Explicit rigid TIP3P geometry and parameters; coordinates are nm, charges e,
# epsilon kJ/mol, and sigma nm. Hydrogen LJ parameters are zero.
OH = 0.09572
HOH_DEG = 104.52
HALF_ANGLE = HOH_DEG * 3.141592653589793 / 360.0
COS_HALF = math.cos(HALF_ANGLE)
SIN_HALF = math.sin(HALF_ANGLE)
BOX = (2.5, 2.5, 2.5)
POSITIONS = (
    (0.45, 0.50, 0.55),
    (0.45 + OH * COS_HALF, 0.50 + OH * SIN_HALF, 0.55),
    (0.45 + OH * COS_HALF, 0.50 - OH * SIN_HALF, 0.55),
    (1.12, 0.95, 0.80),
    (1.12 + OH * COS_HALF, 0.95, 0.80 + OH * SIN_HALF),
    (1.12 + OH * COS_HALF, 0.95, 0.80 - OH * SIN_HALF),
)
CHARGES = (-0.834, 0.417, 0.417, -0.834, 0.417, 0.417)
INTRAMOLECULAR_EXCLUSIONS = ((1, 2), (1, 3), (2, 3), (4, 5), (4, 6), (5, 6))
EPSILON = (0.636386, 0.0, 0.0, 0.636386, 0.0, 0.0)
SIGMA = (0.315075, 1.0, 1.0, 0.315075, 1.0, 1.0)
COULOMB = 138.935456
ALPHA = 3.0
RCUT = 1.0
KMAX = 16
PME_GRID = (64, 64, 64)
ENERGY_TOL = 2e-4
FORCE_TOL = 2e-3


def julia_payload() -> str:
    q = "[" + "; ".join(" ".join(f"{p[a]:.17g}" for p in POSITIONS) for a in range(3)) + "]"
    return f'''include(joinpath(raw"{ROOT / 'julia/src/FoundationCore.jl'}"))
using .FoundationCore
q = {q}
box = FoundationCore.Types.OrthorhombicBox(collect({BOX}))
ws = FoundationCore.StreamedLJ.LJWorkspace(q, box, {RCUT})
ljtop = FoundationCore.StreamedLJ.PairTopology(excluded={list(INTRAMOLECULAR_EXCLUSIONS)!r})
electro_corrections = FoundationCore.Electrostatics.PairCorrections(excluded={list(INTRAMOLECULAR_EXCLUSIONS)!r})
le, lf, _ = FoundationCore.StreamedLJ.evaluate!(ws, q, collect({EPSILON}), collect({SIGMA}), box; rcut={RCUT}, mode=:lfs, topology=ljtop)
ee, ef = FoundationCore.Electrostatics.evaluate_ewald(q, collect({CHARGES}), collect({BOX}); alpha={ALPHA}, rcut={RCUT}, kmax={KMAX}, prefactor={COULOMB}, corrections=electro_corrections)
println("LJ|", repr(le), "|", join(vec(lf), ","))
println("EWALD|", repr(ee), "|", join(vec(ef), ","))
'''


def parse_julia(stdout: str) -> dict[str, tuple[float, list[float]]]:
    results = {}
    for line in stdout.splitlines():
        fields = line.split("|")
        if len(fields) == 3 and fields[0] in {"LJ", "EWALD"}:
            results[fields[0]] = (float(fields[1]), [float(x) for x in fields[2].split(",")])
    if set(results) != {"LJ", "EWALD"} or any(len(v[1]) != 18 for v in results.values()):
        raise RuntimeError(f"Julia output missing valid results: {stdout}")
    return results


def evaluate_openmm(openmm, unit, system):
    vectors = [openmm.Vec3(BOX[0], 0, 0), openmm.Vec3(0, BOX[1], 0), openmm.Vec3(0, 0, BOX[2])]
    system.setDefaultPeriodicBoxVectors(*(v * unit.nanometer for v in vectors))
    integrator = openmm.VerletIntegrator(0.001)
    context = openmm.Context(system, integrator, openmm.Platform.getPlatformByName("Reference"))
    context.setPositions([openmm.Vec3(*p) * unit.nanometer for p in POSITIONS])
    state = context.getState(getEnergy=True, getForces=True)
    energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    forces = state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
    flat = [float(forces[i, axis]) for i in range(len(POSITIONS)) for axis in range(3)]
    del context, integrator
    return float(energy), flat


def openmm_forces(openmm, unit):
    system = openmm.System()
    electrostatics = openmm.NonbondedForce()
    for charge in CHARGES:
        system.addParticle(15.999 if charge < 0 else 1.008)
        electrostatics.addParticle(charge, 1.0, 0.0)
    for i, j in INTRAMOLECULAR_EXCLUSIONS:
        electrostatics.addException(i - 1, j - 1, 0.0, 1.0, 0.0)
    electrostatics.setNonbondedMethod(openmm.NonbondedForce.PME)
    electrostatics.setCutoffDistance(RCUT)
    electrostatics.setEwaldErrorTolerance(1e-8)
    electrostatics.setPMEParameters(ALPHA, *PME_GRID)
    system.addForce(electrostatics)

    return evaluate_openmm(openmm, unit, system)


def compare(name, reference, actual):
    de = abs(reference[0] - actual[0])
    df = max(abs(a - b) for a, b in zip(reference[1], actual[1]))
    ok = de <= ENERGY_TOL and df <= FORCE_TOL
    print(f"{name}: {'PASS' if ok else 'FAIL'} | dE={de:.9g} (tol {ENERGY_TOL:g}) kJ/mol | max_dF={df:.9g} (tol {FORCE_TOL:g}) kJ/mol/nm")
    return ok


def main() -> int:
    julia = Path(os.environ.get("JULIA_BIN", JULIA_DEFAULT))
    configured_python = Path(os.environ.get("PYTHON_BIN", PYTHON_DEFAULT))
    print(f"Julia interpreter: {julia}")
    print(f"Python interpreter: {sys.executable} (configured: {configured_python})")
    if Path(sys.executable).resolve() != configured_python.resolve():
        print("TIP3P MATCHED: BLOCKED (run with the configured absolute Python interpreter)")
        return 2
    if not julia.is_file() or not os.access(julia, os.X_OK):
        print(f"TIP3P MATCHED: BLOCKED (Julia executable unavailable: {julia})")
        return 2
    try:
        import openmm
        from openmm import unit
    except Exception as exc:  # noqa: BLE001
        print(f"TIP3P MATCHED: BLOCKED (OpenMM import: {type(exc).__name__}: {exc})")
        return 2
    run = subprocess.run([str(julia), "--startup-file=no", "-e", julia_payload()],
                         cwd=ROOT / "julia", text=True, capture_output=True, timeout=60)
    if run.returncode:
        print(f"TIP3P MATCHED: BLOCKED (Core Julia evaluator failed, exit {run.returncode})")
        print(run.stderr.strip())
        return 2
    try:
        reference = parse_julia(run.stdout)
        actual = openmm_forces(openmm, unit)
        print(f"OpenMM version: {openmm.__version__}; platform: Reference")
        print(f"TIP3P sites: 2 rigid-geometry waters; charges={CHARGES}; O epsilon/sigma={(EPSILON[0], SIGMA[0])}; H epsilon/sigma=(0, 1.0) (zero LJ epsilon)")
        print(f"Box={BOX} nm; LJ force-shift cutoff={RCUT} nm; Ewald alpha={ALPHA} nm^-1; Core direct Ewald kmax={KMAX}; OpenMM PME grid={PME_GRID}")
        print(f"Scope: static explicit TIP3P site parameters with intramolecular LJ/Coulomb exclusions {INTRAMOLECULAR_EXCLUSIONS}; direct Ewald (Core) vs PME (OpenMM), same real cutoff/split and tin-foil convention.")
        print("Trajectory: unavailable; constrained periodic electrostatics/dynamics are not composed end-to-end.")
        ewald_result = compare("Direct Ewald vs PME", reference["EWALD"], actual)
        # Evaluate the independently matched LJ term in its own context.
        lj_system = openmm.System()
        for _ in POSITIONS:
            lj_system.addParticle(1.0)
        lj_force = openmm.CustomNonbondedForce(
            "4*sqrt(e1*e2)*((s/r)^12-(s/r)^6)"
            "-4*sqrt(e1*e2)*((s/rc)^12-(s/rc)^6)"
            "-24*sqrt(e1*e2)*((s/rc)^6-2*(s/rc)^12)/rc*(r-rc); s=(s1+s2)/2"
        )
        lj_force.addPerParticleParameter("e")
        lj_force.addPerParticleParameter("s")
        lj_force.addGlobalParameter("rc", RCUT)
        lj_force.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
        lj_force.setCutoffDistance(RCUT)
        for e, s in zip(EPSILON, SIGMA):
            lj_force.addParticle([e, s])
        for i, j in INTRAMOLECULAR_EXCLUSIONS:
            lj_force.addExclusion(i - 1, j - 1)
        lj_system.addForce(lj_force)
        lj_result = compare("Matched force-shifted LJ", reference["LJ"], evaluate_openmm(openmm, unit, lj_system))
    except Exception as exc:  # noqa: BLE001
        print(f"TIP3P MATCHED: FAIL ({type(exc).__name__}: {exc})")
        return 1
    print("TIP3P MATCHED:", "PASS" if ewald_result and lj_result else "FAIL")
    return 0 if ewald_result and lj_result else 1


if __name__ == "__main__":
    raise SystemExit(main())
