"""Example 01: Carbon nanotube geometry construction and nanomechanics workflow."""

from __future__ import annotations

import numpy as np

from biology_mm import (
    CustomPotential,
    NanostructureBuilder,
    Simulation,
    SimulationConfig,
)


def main() -> None:
    cnt = NanostructureBuilder.carbon_nanotube(n=5, m=5, n_cells=2, bond_length_nm=0.142)
    print(f"Built (5,5) armchair CNT: {cnt.n_atoms} atoms, {len(cnt.bonds)} C-C bonds")
    print(f"Box (nm): {tuple(round(v, 4) for v in cnt.box_lengths)}")
    print(f"Minimum periodic C-C distance: {cnt.minimum_distance(periodic=True):.5f} nm")

    # Attach harmonic C-C bonds (0-based indices) for Wave-1 md.jl execution,
    # and inspect a reactive Tersoff C potential spec for Phase 9 integration.
    potentials = [
        CustomPotential.harmonic_bond(i, j, k=250000.0, r0=0.142)
        for i, j in cnt.bonds
    ]
    # Add a weak nonbonded LJ pair across non-bonded neighbors as an example
    potentials.append(CustomPotential.lj_pair(0, 10, epsilon=0.28, sigma=0.34, rc=1.0, mode="lfs"))

    tersoff_c = CustomPotential.tersoff(element="C", units="md")
    print("Tersoff spec:", tersoff_c.to_spec_dict())

    cfg = SimulationConfig(
        ensemble="nve",
        integrator="velocity_verlet",
        dt=0.0005,
        temperature=0.0,
        minimize_max_steps=5,
        equilibrate_steps=5,
        produce_steps=25,
        frame_every=1,
        checkpoint_every=0,
        contact_tol=0.05,
        run_id="example-cnt-55",
    )
    sim = Simulation(cnt, potentials=potentials, config=cfg)
    result = sim.run()
    print(f"Workflow stages: {result.stages}, frames: {result.n_frames}")
    disp = np.linalg.norm(result.frames[-1]["positions"] - result.frames[0]["positions"], axis=1)
    print(f"Max atomic displacement over trajectory: {disp.max():.6f} nm")


if __name__ == "__main__":
    main()
