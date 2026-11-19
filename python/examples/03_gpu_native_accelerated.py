"""Example 03: Native Apple Metal GPU / ARM64 NEON nonbonded, MSM stencil, and ternary algebra."""

from __future__ import annotations

import numpy as np

from biology_mm import (
    GPUVerletPlan,
    SystemBuilder,
    csa_accumulate,
    gpu_msm_stencil27,
    gpu_ternary_matvec,
    probe_hardware,
    run_batched_gpu_steps,
)


def main() -> None:
    prof = probe_hardware()
    print(
        f"Hardware: backend={prof.backend}, device='{prof.device_name}', "
        f"has_gpu={prof.has_gpu}, unified={prof.unified_memory}, simd={prof.simd_width}"
    )

    # 1. Exact Carry-Save Balanced Ternary Accumulation
    csa = csa_accumulate([1000, -450, 225, -775, 42])
    print(f"CSA exact sum: resolved={csa.resolved} (expected 42), overflow={csa.overflow}")

    # 2. FCC Argon crystal LJ + DSF energy, forces, and 3x3 virial on Metal GPU
    sys_fcc = SystemBuilder.fcc_lattice(n_cells=3, rho=0.8, mass=39.948, name="Ar")
    n = sys_fcc.n_atoms
    rng = np.random.default_rng(7)
    positions = sys_fcc.positions + 0.02 * rng.standard_normal((n, 3))
    sigma = np.full(n, 0.34, dtype=np.float64)
    epsilon = np.full(n, 0.996, dtype=np.float64)
    charges = np.zeros(n, dtype=np.float64)
    charges[::2] = 0.25
    charges[1::2] = -0.25

    with GPUVerletPlan(
        positions=positions,
        sigma=sigma,
        epsilon=epsilon,
        charges=charges,
        box=sys_fcc.box_lengths,
        rcut=1.5,
        skin=0.25,
        alpha_dsf=0.22,
        backend="auto",
    ) as plan:
        energy, forces, virial = plan.eval(positions=positions, compute_virial=True)
        print(
            f"GPUVerletPlan (N={n}, pairs={plan.num_pairs}, using_gpu={plan.using_gpu}): "
            f"U={energy:.6f}, |sum(F)|={np.linalg.norm(forces.sum(axis=0)):.3e}"
        )
        print("Virial dU/dH diag:", np.round(np.diag(virial), 5))

    # 3. GPU-resident batched Velocity Verlet
    momenta = np.zeros((n, 3), dtype=np.float64)
    batch_res = run_batched_gpu_steps(
        positions=positions,
        momenta=momenta,
        masses=sys_fcc.masses,
        sigma=sigma,
        epsilon=epsilon,
        charges=charges,
        box=sys_fcc.box_lengths,
        dt=0.001,
        n_steps=20,
        rcut=1.5,
        skin=0.25,
        alpha_dsf=0.22,
    )
    print(
        f"Batched GPU MD (20 steps): PE={batch_res['potential_energy']:.4f}, "
        f"KE={batch_res['kinetic_energy']:.4f}, E_tot={batch_res['total_energy']:.4f}"
    )

    # 4. Dual-rail ternary MatVec on Metal GPU
    M, K = 8, 2
    a = rng.integers(0, 2**63, size=(M, K), dtype=np.uint64)
    b = rng.integers(0, 2**63, size=(M, K), dtype=np.uint64)
    W_P = a & ~b
    W_N = b & ~a
    xa = rng.integers(0, 2**63, size=(K,), dtype=np.uint64)
    xb = rng.integers(0, 2**63, size=(K,), dtype=np.uint64)
    x_P = xa & ~xb
    x_N = xb & ~xa
    y = gpu_ternary_matvec(W_P, W_N, x_P, x_N, x_scale=0.5)
    print("GPU ternary matvec sample:", np.round(y[:4], 3))

    # 5. 27-point periodic 3D MSM stencil convolution on Metal GPU
    Q = rng.standard_normal((8, 8, 8))
    k27 = np.zeros(27, dtype=np.float64)
    k27[13] = 2.0  # center (0,0,0)
    k27[14] = -0.25  # (+1,0,0)
    u_msm, phi = gpu_msm_stencil27(Q, k27, prefactor=1.0)
    print(f"GPU MSM stencil27 energy: {u_msm:.6f}, Phi norm: {np.linalg.norm(phi):.6f}")


if __name__ == "__main__":
    main()
