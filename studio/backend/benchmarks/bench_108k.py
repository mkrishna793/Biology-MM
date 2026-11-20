"""Benchmark: 108,000-atom FCC Argon, N MD steps. Reports wall time, ms/step, ns/day, CPU & RSS."""
import resource
import sys
import time

from biology_mm import SystemBuilder
from trium_studio.engine_core import SimulationSession

N_STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
CHUNK = 100

system = SystemBuilder.fcc_lattice(30, rho=0.84, mass=1.0, name="Ar", units="reduced")
sess = SimulationSession(system, mode="metal_verlet", dt=0.005, temperature=1.2, sigma=1.0, epsilon=1.0, rcut=2.5)
print(f"atoms={system.n_atoms} backend={sess.hardware.backend} gpu={sess.hardware.has_gpu}")

import os
t_cpu0 = sum(os.times()[:2])
t0 = time.perf_counter()
done = 0
while done < N_STEPS:
    m = sess.step(CHUNK)
    done += CHUNK
wall = time.perf_counter() - t0
cpu = sum(os.times()[:2]) - t_cpu0
rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)  # macOS: bytes
ms_step = wall / done * 1e3
sim_ps = done * 0.005
print(f"steps={done} wall={wall:.2f}s ms/step={ms_step:.3f} steps/s={done / wall:.1f}")
print(f"process_cpu_time={cpu:.2f}s cpu_utilisation={cpu / wall * 100:.0f}% peak_rss={rss_mb:.0f}MB")
print(f"atom-steps/s={sess.n_real * done / wall:.3e}  final T*={m['temperature_K']:.3f}  Etot={m['total_energy']:.2f}")
print(f"extrapolated 23,021 steps = {23021 * wall / done:.1f}s ({23021 * wall / done / 60:.1f} min)")
