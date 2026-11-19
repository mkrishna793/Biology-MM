#!/usr/bin/env python3
"""Generate a small OpenMM TIP3P PME energy/short-trajectory fixture."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="tip3p_pme_reference")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    try:
        import openmm
        from openmm import app, unit
    except ImportError as exc:
        parser.error("OpenMM is required: {}".format(exc))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    forcefield = app.ForceField("tip3p.xml")
    modeller = app.Modeller(app.Topology(), [])
    modeller.addSolvent(forcefield, model="tip3p", boxSize=openmm.Vec3(2, 2, 2) * unit.nanometer,
                        neutralize=True)
    system = forcefield.createSystem(modeller.topology, nonbondedMethod=app.PME,
                                     nonbondedCutoff=0.9 * unit.nanometer,
                                     constraints=app.HBonds, rigidWater=True,
                                     ewaldErrorTolerance=5e-4)
    integrator = openmm.LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picoseconds,
                                                 0.002 * unit.picoseconds)
    integrator.setRandomNumberSeed(args.seed)
    platform = openmm.Platform.getPlatformByName("Reference")
    simulation = app.Simulation(modeller.topology, system, integrator, platform)
    simulation.context.setPositions(modeller.positions)
    initial = simulation.context.getState(getEnergy=True).getPotentialEnergy()
    simulation.minimizeEnergy(maxIterations=100)
    minimized = simulation.context.getState(getEnergy=True).getPotentialEnergy()
    simulation.context.setVelocitiesToTemperature(300 * unit.kelvin, args.seed)
    simulation.reporters.append(app.DCDReporter(str(output / "trajectory.dcd"), 10))
    simulation.reporters.append(app.StateDataReporter(str(output / "energy.csv"), 10,
        step=True, potentialEnergy=True, temperature=True, separator=","))
    simulation.step(args.steps)
    simulation.reporters.clear()
    with (output / "topology.pdb").open("w") as stream:
        app.PDBFile.writeFile(modeller.topology, modeller.positions, stream, keepIds=True)
    info = {"openmm_version": openmm.__version__, "platform": platform.getName(),
            "force_field": "OpenMM tip3p.xml", "electrostatics": "PME",
            "box_nm": [2, 2, 2], "atoms": modeller.topology.getNumAtoms(), "steps": args.steps,
            "seed": args.seed, "initial_potential_kj_mol": initial.value_in_unit(unit.kilojoule_per_mole),
            "minimized_potential_kj_mol": minimized.value_in_unit(unit.kilojoule_per_mole),
            "warning": "Small generated fixture only; not an equilibrated reference or accepted scientific range."}
    (output / "metadata.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf8")
    print(json.dumps(info, indent=2))

if __name__ == "__main__":
    main()
