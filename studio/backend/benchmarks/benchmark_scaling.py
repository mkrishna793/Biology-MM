import sys, os, time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / 'python'))
sys.path.insert(0, str(REPO_ROOT / 'studio' / 'backend'))

import numpy as np
from trium_studio.engine_core import _build_spatial_gpu_verlet_plan, get_bridge

bridge = get_bridge()
print("GPU Hardware:", bridge.probe_hardware(), flush=True)

for N in [100_000, 500_000, 1_000_000, 5_000_000]:
    print(f"Starting N = {N:,}...", flush=True)
    np.random.seed(42)
    L = float(pow(N / 30.0, 1.0/3.0))
    box = np.array([L, L, L])
    pos = np.random.uniform(0.1, L - 0.1, size=(N, 3)).astype(np.float64)
    sigma = np.full(N, 0.3, dtype=np.float64)
    epsilon = np.full(N, 0.5, dtype=np.float64)
    charge = np.random.uniform(-0.5, 0.5, size=N).astype(np.float64)

    t0 = time.perf_counter()
    plan = _build_spatial_gpu_verlet_plan(
        positions=pos,
        sigma=sigma,
        epsilon=epsilon,
        charges=charge,
        box=box,
        exclusions=None,
        rcut=1.0,
        skin=0.2,
        alpha_dsf=0.2,
        coulomb_prefactor=138.935456,
        backend="gpu",
    )
    t_init = time.perf_counter() - t0
    print(f"Plan initialized in {t_init*1000:.2f} ms, using_gpu={plan.using_gpu}, pairs={plan.num_pairs:,}", flush=True)

    t1 = time.perf_counter()
    energy, forces, virial = plan.eval(pos, box, compute_virial=False)
    t_eval = time.perf_counter() - t1
    print(f"Force Eval in {t_eval*1000:.2f} ms, energy={energy:.2f}, max_f={np.max(np.abs(forces)):.2f}", flush=True)

    # Run batched MD step
    masses = np.full(N, 12.011, dtype=np.float64)
    momenta = np.zeros_like(pos)
    t2 = time.perf_counter()
    res = plan.run_batched_steps(
        positions=pos,
        momenta=momenta,
        masses=masses,
        dt=0.001,
        n_steps=5,
        box=box,
    )
    t_md = (time.perf_counter() - t2) / 5.0
    print(f"Batched GPU MD: {t_md*1000:.2f} ms/step ({1.0/t_md:.1f} steps/s)\n", flush=True)

    plan.close()
