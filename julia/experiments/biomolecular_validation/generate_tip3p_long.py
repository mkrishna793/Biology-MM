#!/usr/bin/env python3
"""Generate a longer OpenMM TIP3P PME reference (100 ps equil + 200 ps production).

This is a new fixture for the biomolecular-validation MSD work. It does not
overwrite the original 0.2 ps `reference/` directory. The goal is a trajectory
long enough to show whether the oxygen MSD reaches a diffusive slope; it is not
a converged bulk-water thermodynamic average.

Defaults: 2 nm cubic box (258 TIP3P waters, 774 atoms), PME, 0.9 nm cutoff,
rigid water/H-bond constraints, 2 fs LangevinMiddle at 300 K, 100 ps
equilibration, 200 ps production, frames and energies every 1 ps.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default="reference_long")
    parser.add_argument("--platform", default="OpenCL",
                        help="OpenMM platform name; falls back to CPU then Reference")
    parser.add_argument("--equil-ps", type=float, default=100.0)
    parser.add_argument("--prod-ps", type=float, default=200.0)
    parser.add_argument("--dt-fs", type=float, default=2.0)
    parser.add_argument("--frame-ps", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--friction", type=float, default=1.0, help="1/ps")
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--minimize-iterations", type=int, default=200)
    parser.add_argument("--box-nm", type=float, default=2.0)
    parser.add_argument("--cutoff-nm", type=float, default=0.9)
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        import openmm
        from openmm import app, unit
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(f"OpenMM is required: {exc}")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name in ("trajectory.dcd", "energy.csv", "equil_energy.csv",
                 "topology.pdb", "metadata.json"):
        if (output / name).exists():
            raise SystemExit(f"refusing to overwrite {output / name}")

    forcefield = app.ForceField("tip3p.xml")
    modeller = app.Modeller(app.Topology(), [])
    modeller.addSolvent(forcefield, model="tip3p",
                        boxSize=openmm.Vec3(args.box_nm, args.box_nm, args.box_nm)
                        * unit.nanometer, neutralize=True)
    system = forcefield.createSystem(
        modeller.topology, nonbondedMethod=app.PME,
        nonbondedCutoff=args.cutoff_nm * unit.nanometer,
        constraints=app.HBonds, rigidWater=True, ewaldErrorTolerance=5e-4)
    dt = args.dt_fs * unit.femtosecond
    integrator = openmm.LangevinMiddleIntegrator(
        args.temperature * unit.kelvin, args.friction / unit.picosecond, dt)
    integrator.setRandomNumberSeed(args.seed)

    platform = None
    platform_properties = {}
    errors = []
    for name in (args.platform, "CPU", "Reference"):
        try:
            candidate = openmm.Platform.getPlatformByName(name)
            if name == "CPU":
                platform_properties = {"Threads": str(max(1, (__import__("os").cpu_count() or 1)))}
            else:
                platform_properties = {}
            # Building a Context is the real availability test.
            test = openmm.Context(system, openmm.VerletIntegrator(0.001), candidate,
                                  platform_properties)
            del test
            platform = candidate
            break
        except Exception as exc:  # platform may be absent or fail to load
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    if platform is None:
        raise SystemExit("no usable OpenMM platform: " + "; ".join(errors))

    simulation = app.Simulation(modeller.topology, system, integrator,
                               platform, platform_properties)
    simulation.context.setPositions(modeller.positions)
    initial = simulation.context.getState(getEnergy=True).getPotentialEnergy()
    t0 = time.perf_counter()
    simulation.minimizeEnergy(maxIterations=args.minimize_iterations)
    minimized = simulation.context.getState(getEnergy=True).getPotentialEnergy()
    minimize_s = time.perf_counter() - t0
    simulation.context.setVelocitiesToTemperature(args.temperature * unit.kelvin,
                                                 args.seed)

    equil_steps = max(1, round(args.equil_ps / (args.dt_fs / 1000.0)))
    prod_steps = max(1, round(args.prod_ps / (args.dt_fs / 1000.0)))
    sample_steps = max(1, round(args.frame_ps / (args.dt_fs / 1000.0)))
    equil_reporter = app.StateDataReporter(
        str(output / "equil_energy.csv"), sample_steps, step=True, time=True,
        potentialEnergy=True, temperature=True, separator=",")
    simulation.reporters.append(equil_reporter)
    t0 = time.perf_counter()
    simulation.step(equil_steps)
    equil_s = time.perf_counter() - t0
    simulation.reporters.clear()

    simulation.reporters.append(app.DCDReporter(
        str(output / "trajectory.dcd"), sample_steps))
    simulation.reporters.append(app.StateDataReporter(
        str(output / "energy.csv"), sample_steps, step=True, time=True,
        potentialEnergy=True, temperature=True, separator=","))
    t0 = time.perf_counter()
    simulation.step(prod_steps)
    prod_s = time.perf_counter() - t0
    simulation.reporters.clear()

    final_state = simulation.context.getState(getPositions=True, getEnergy=True)
    final_positions = final_state.getPositions()
    with (output / "topology.pdb").open("w") as stream:
        app.PDBFile.writeFile(modeller.topology, final_positions, stream, keepIds=True)

    temperatures = []
    energies = []
    with (output / "energy.csv").open() as stream:
        next(stream)
        for line in stream:
            fields = line.strip().split(",")
            if len(fields) >= 4:
                energies.append(float(fields[2]))
                temperatures.append(float(fields[3]))

    info = {
        "openmm_version": openmm.__version__,
        "platform": platform.getName(),
        "platform_properties": platform_properties,
        "platform_errors": errors,
        "force_field": "OpenMM tip3p.xml",
        "electrostatics": "PME",
        "box_nm": [args.box_nm, args.box_nm, args.box_nm],
        "atoms": modeller.topology.getNumAtoms(),
        "waters": modeller.topology.getNumResidues(),
        "equil_ps": args.equil_ps,
        "prod_ps": args.prod_ps,
        "dt_fs": args.dt_fs,
        "frame_ps": args.frame_ps,
        "integrator": "LangevinMiddle",
        "temperature_K": args.temperature,
        "friction_per_ps": args.friction,
        "nonbonded_cutoff_nm": args.cutoff_nm,
        "ewald_error_tolerance": 5e-4,
        "constraints": "HBonds",
        "rigid_water": True,
        "seed": args.seed,
        "minimize_iterations": args.minimize_iterations,
        "minimize_wall_s": minimize_s,
        "equil_wall_s": equil_s,
        "prod_wall_s": prod_s,
        "total_wall_s": minimize_s + equil_s + prod_s,
        "equil_steps": equil_steps,
        "prod_steps": prod_steps,
        "sample_steps": sample_steps,
        "initial_potential_kj_mol": initial.value_in_unit(unit.kilojoule_per_mole),
        "minimized_potential_kj_mol": minimized.value_in_unit(unit.kilojoule_per_mole),
        "final_potential_kj_mol": final_state.getPotentialEnergy().value_in_unit(
            unit.kilojoule_per_mole),
        "prod_temperature_mean_K": float(np.mean(temperatures)) if temperatures else None,
        "prod_temperature_std_K": float(np.std(temperatures)) if temperatures else None,
        "prod_potential_mean_kj_mol": float(np.mean(energies)) if energies else None,
        "warning": ("Longer diagnostic fixture than reference/ but still one 2 nm "
                    "box, one trajectory, and a model-dependent D; not an "
                    "equilibrated bulk-water reference or an engine result."),
    }
    (output / "metadata.json").write_text(json.dumps(info, indent=2) + "\n",
                                          encoding="utf8")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
