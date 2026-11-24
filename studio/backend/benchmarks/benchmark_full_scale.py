
import sys, os, time
from pathlib import Path

import psutil

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / 'python'))
sys.path.insert(0, str(REPO_ROOT / 'studio' / 'backend'))

import numpy as np
from trium_studio.engine_core import _build_spatial_gpu_verlet_plan, get_bridge

proc = psutil.Process(os.getpid())

def get_ram_mb():
    return proc.memory_info().rss / (1024 * 1024)

bridge = get_bridge()
hw = bridge.probe_hardware()
print(f"=== Metal 4 GPU Benchmark on {hw.device_name} (VRAM: {hw.vram_bytes/(1024**3):.2f} GB) ===", flush=True)

# 1. Scaling benchmarks: 1M, 5M, 10M
for N in [1_000_000, 5_000_000, 10_000_000]:
    ram_before = get_ram_mb()
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
    ram_init = get_ram_mb()

    # Force Eval
    t1 = time.perf_counter()
    energy, forces, virial = plan.eval(pos, box, compute_virial=False)
    t_eval = time.perf_counter() - t1

    # Batched GPU MD
    masses = np.full(N, 12.011, dtype=np.float64)
    momenta = np.zeros_like(pos)
    t2 = time.perf_counter()
    n_steps = 5 if N <= 5_000_000 else 2
    res = plan.run_batched_steps(
        positions=pos,
        momenta=momenta,
        masses=masses,
        dt=0.001,
        n_steps=n_steps,
        box=box,
    )
    t_md = (time.perf_counter() - t2) / float(n_steps)

    print(f"\n[N = {N:,} atoms (Dense Fluid Box: {L:.1f} nm)]", flush=True)
    print(f"  Plan Init:    {t_init*1000:7.2f} ms", flush=True)
    print(f"  RAM Usage:    {ram_init:7.1f} MB (Delta: +{ram_init - ram_before:.1f} MB)", flush=True)
    print(f"  Force Eval:   {t_eval*1000:7.2f} ms ({1.0/t_eval:5.1f} evals/s)", flush=True)
    print(f"  MD Step:      {t_md*1000:7.2f} ms/step ({1.0/t_md:5.1f} steps/s)", flush=True)
    print(f"  Tile Pairs:   {plan.num_pairs // 1024:,} tiles", flush=True)
    print(f"  Energy:       {energy:11.4e}, Max Force: {np.max(np.abs(forces)):11.4e}", flush=True)
    plan.close()
    del pos, sigma, epsilon, charge, plan, forces, masses, momenta

# 2. Faustovirus 5J7V 40M Capsid Benchmark
coords_path = (
    sys.argv[1]
    if len(sys.argv) > 1
    else os.environ.get("BIOLOGY_MM_5J7V_COORDS", str(Path.home() / "Desktop" / "5j7v_40m_coords.bin"))
)
if os.path.exists(coords_path):
    print("\n=== Loading Faustovirus 5J7V 40-Million Atom Capsid ===", flush=True)
    ram_before = get_ram_mb()
    t0 = time.perf_counter()
    coords_flat = np.fromfile(coords_path, dtype=np.float32)
    N_40m = coords_flat.size // 3
    pos_40m = (coords_flat.reshape(N_40m, 3).astype(np.float64) * 0.1) # Angstroms -> nm
    t_load = time.perf_counter() - t0
    print(f"Loaded {N_40m:,} atoms from flat binary in {t_load*1000:.1f} ms! RAM: {get_ram_mb():.1f} MB", flush=True)

    span = np.ptp(pos_40m, axis=0)
    print(f"Capsid Span: {span[0]:.1f} x {span[1]:.1f} x {span[2]:.1f} nm (Diameter: {np.max(span)*10:.1f} Angstroms)", flush=True)
    box_40m = span + 4.0 # 4 nm margin

    sigma_40m = np.full(N_40m, 0.3, dtype=np.float64)
    epsilon_40m = np.full(N_40m, 0.5, dtype=np.float64)
    charges_40m = np.zeros(N_40m, dtype=np.float64)

    t0 = time.perf_counter()
    plan_40m = _build_spatial_gpu_verlet_plan(
        positions=pos_40m,
        sigma=sigma_40m,
        epsilon=epsilon_40m,
        charges=charges_40m,
        box=box_40m,
        exclusions=None,
        rcut=1.0,
        skin=0.2,
        backend="gpu",
    )
    t_init_40m = time.perf_counter() - t0
    ram_40m = get_ram_mb()

    t1 = time.perf_counter()
    energy_40m, forces_40m, _ = plan_40m.eval(pos_40m, box_40m, compute_virial=False)
    t_eval_40m = time.perf_counter() - t1

    masses_40m = np.full(N_40m, 12.011, dtype=np.float64)
    momenta_40m = np.zeros_like(pos_40m)
    t2 = time.perf_counter()
    res_40m = plan_40m.run_batched_steps(
        positions=pos_40m,
        momenta=momenta_40m,
        masses=masses_40m,
        dt=0.001,
        n_steps=2,
        box=box_40m,
    )
    t_md_40m = (time.perf_counter() - t2) / 2.0

    print(f"\n[Faustovirus 5J7V: {N_40m:,} ATOMS SIMULATED ON APPLE M5 GPU]")
    print(f"  Tile Plan Init:  {t_init_40m:7.2f} seconds")
    print(f"  Total Host RAM:  {ram_40m/1024:7.2f} GB (Delta: +{(ram_40m - ram_before)/1024:.2f} GB)")
    print(f"  Force Eval Time: {t_eval_40m*1000:7.2f} ms")
    print(f"  Batched GPU MD:  {t_md_40m*1000:7.2f} ms/step ({1.0/t_md_40m:5.2f} steps/s)")
    print(f"  Total Energy:    {energy_40m:11.4e}")
    print(f"  Max Force:       {np.max(np.abs(forces_40m)):11.4e} kJ/(mol*nm)")
    plan_40m.close()
