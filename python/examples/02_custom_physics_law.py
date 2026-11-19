"""Example 02: Custom potential tabulation and unconstrained parameter encoding."""

from __future__ import annotations

import math

import numpy as np

from biology_mm import (
    CustomPotential,
    ParameterEncoder,
    ParameterSpec,
    Simulation,
    SimulationConfig,
    SystemBuilder,
)


def main() -> None:
    encoder = ParameterEncoder(
        [
            ParameterSpec("k_bond", default=500.0, unit="kJ/(mol*nm^2)", lower=1.0, encoding="log"),
            ParameterSpec("r0", default=0.38, unit="nm", lower=0.1, upper=1.0, encoding="logit"),
            ParameterSpec("phi0", default=0.5, unit="rad", encoding="angle"),
            ParameterSpec("sign_trit", default=1.0, trit_scale=1.0, encoding="ternary_quantized"),
        ]
    )
    theta0 = encoder.defaults
    z0 = encoder.encode(theta0)
    theta_rt = encoder.decode(z0)
    jac = encoder.jacobian_diag(z0)
    print("Encoded z0     :", np.round(z0, 5))
    print("Round-trip     :", np.round(theta_rt, 5))
    print("Jacobian diag  :", np.round(jac, 5))

    # Sample a custom Buckingham-like potential U(r) onto a 256-point spline grid
    def buckingham_u(r: float) -> float:
        return 1.2e4 * math.exp(-3.4 * r) - 0.15 / (r ** 6)

    tab_pot = CustomPotential.from_callable(
        buckingham_u,
        r_min=0.25,
        r_max=1.10,
        pairs=[(0, 1)],
        n_grid=256,
        mode="lfs",
    )
    spec = tab_pot.to_spec_dict(atom_index_base=1)
    print(
        f"Tabulated custom potential '{spec['type']}' with {len(spec['r_grid'])} grid points "
        f"for 1-based pairs={spec['pairs']}"
    )

    # Run a 2-atom oscillator + LJ dimer through Julia CLI
    sb = SystemBuilder(box_lengths=(10.0, 10.0, 10.0))
    a0 = sb.add_atom((4.0, 5.0, 5.0), mass=12.0, name="C")
    a1 = sb.add_atom((4.42, 5.0, 5.0), mass=12.0, name="C")
    sys_dimer = sb.build()

    pots = [
        CustomPotential.harmonic_bond(a0, a1, k=500.0, r0=0.38),
        CustomPotential.lj_pair(a0, a1, epsilon=0.25, sigma=0.34, rc=1.5, mode="lfs"),
    ]
    cfg = SimulationConfig(
        ensemble="nve",
        integrator="velocity_verlet",
        dt=0.001,
        temperature=0.0,
        minimize_max_steps=0,
        equilibrate_steps=0,
        produce_steps=50,
        frame_every=2,
        checkpoint_every=0,
        contact_tol=0.05,
        run_id="example-custom-law",
    )
    res = Simulation(sys_dimer, potentials=pots, config=cfg).run()
    print(f"Dimer simulation completed: {res.n_frames} frames, D={res.diffusion_coefficient:.6e}")


if __name__ == "__main__":
    main()
