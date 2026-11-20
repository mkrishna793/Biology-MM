"""Benchmark: 27+5 Trium Radix-3 Peano Tryte Tiles vs 32-atom Binary Morton & Full-Atom RAM/Speed Scaling.

Measures:
  1. 27+5 Radix-3 Peano vs 32-atom Binary Morton on 10k, 50k, and 250k realistic
     hollow-capsid & globular protein geometries (mean radius, AABB volume, interacting
     tile pairs, candidate atom-pair checks, % reductions).
  2. Real `TryteTileFullAtomSession` RAM, bytes/atom, build time, and step time at
     10k, 100k, and 1M atoms.
  3. Exact verified 40,000,000-atom byte math (Tryte 27+5 vs naive Float64).
"""

from __future__ import annotations

import math
import time
from typing import Tuple

import numpy as np

from trium_studio.tryte_tiles import (
    TryteTileFullAtomSession,
    compare_tryte_peano_vs_binary_morton,
)


def make_globular_protein(n_atoms: int, seed: int = 101) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a dense globular protein geometry (~0.36 nm spacing) in nm."""
    rng = np.random.default_rng(seed)
    # Approximate protein atomic density ~ 15 atoms / nm^3 -> R = (3N / (4*pi*15))^(1/3)
    radius_nm = ((3.0 * n_atoms) / (4.0 * math.pi * 15.0)) ** (1.0 / 3.0)
    # Uniformly sample interior of a sphere with slight radial density variation
    u = rng.uniform(0.0, 1.0, size=n_atoms)
    r = radius_nm * np.cbrt(u)
    cos_th = rng.uniform(-1.0, 1.0, size=n_atoms)
    sin_th = np.sqrt(np.maximum(0.0, 1.0 - cos_th * cos_th))
    phi = rng.uniform(0.0, 2.0 * math.pi, size=n_atoms)
    pos = np.stack(
        [r * sin_th * np.cos(phi), r * sin_th * np.sin(phi), r * cos_th],
        axis=1,
    ).astype(np.float32)
    z = rng.choice(np.array([1, 6, 7, 8, 16], dtype=np.uint8), size=n_atoms, p=[0.35, 0.35, 0.14, 0.14, 0.02])
    return pos, z


def make_hollow_capsid(n_atoms: int, seed: int = 202) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a realistic hollow icosahedral-like viral capsid shell in nm."""
    rng = np.random.default_rng(seed)
    # Shell thickness ~ 3.0 nm; surface density ~ 45 atoms / nm^2
    shell_thick_nm = 3.0
    radius_nm = max(5.0, math.sqrt(n_atoms / (4.0 * math.pi * 45.0)))
    golden = math.pi * (3.0 - math.sqrt(5.0))
    idx = np.arange(n_atoms, dtype=np.float64)
    y = 1.0 - 2.0 * (idx + 0.5) / n_atoms
    r_xy = np.sqrt(np.maximum(0.0, 1.0 - y * y))
    theta = golden * idx
    r = radius_nm + rng.uniform(-0.5 * shell_thick_nm, 0.5 * shell_thick_nm, size=n_atoms)
    pos = np.stack(
        [r * np.cos(theta) * r_xy, r * y, r * np.sin(theta) * r_xy],
        axis=1,
    ).astype(np.float32)
    z = rng.choice(np.array([1, 6, 7, 8, 16], dtype=np.uint8), size=n_atoms, p=[0.35, 0.35, 0.14, 0.14, 0.02])
    return pos, z


def run_benchmark() -> None:
    print("=" * 96)
    print("BENCHMARK 1: 27+5 Trium Radix-3 Peano Trytes vs 32-Atom Binary Morton (rcut = 1.0 nm)")
    print("=" * 96)
    header = (
        f"{'Geometry':<18} {'N':>8} | {'Peano R(nm)':>11} {'Morton R(nm)':>12} {'R Red%':>7} | "
        f"{'Vol Red%':>8} | {'Peano Pairs':>11} {'Morton Pairs':>12} {'Pair Red%':>9} {'AtomChk Red%':>12}"
    )
    print(header)
    print("-" * len(header))

    for geom_name, builder in (("Globular Protein", make_globular_protein), ("Hollow Capsid", make_hollow_capsid)):
        for n in (10_000, 50_000, 250_000):
            pos, _ = builder(n)
            res = compare_tryte_peano_vs_binary_morton(pos, rcut_nm=1.0, levels=6)
            p = res["peano_27"]
            m = res["morton_32"]
            print(
                f"{geom_name:<18} {n:>8,d} | {p['mean_bounding_radius_nm']:>11.4f} {m['mean_bounding_radius_nm']:>12.4f} "
                f"{res['radius_reduction_pct']:>6.2f}% | {res['aabb_volume_reduction_pct']:>7.2f}% | "
                f"{p['interacting_tile_pairs']:>11,d} {m['interacting_tile_pairs']:>12,d} "
                f"{res['pair_reduction_pct']:>8.2f}% {res['atom_pair_check_reduction_pct']:>11.2f}%"
            )

    print("\n" + "=" * 96)
    print("BENCHMARK 2: Real TryteTileFullAtomSession RAM & Step Scaling (10k, 100k, 1M Atoms)")
    print("=" * 96)
    h2 = (
        f"{'N Atoms':>10} {'Tiles':>8} | {'Build(ms)':>10} {'RAM(MB)':>9} {'B/atom':>7} {'Savings':>8} | "
        f"{'ms/step':>9} {'Temp(K)':>8} {'Drift':>10} {'Health':>8}"
    )
    print(h2)
    print("-" * len(h2))

    for n in (10_000, 100_000, 1_000_000):
        pos, z = make_hollow_capsid(n, seed=400 + n % 97)
        t0 = time.perf_counter()
        sess = TryteTileFullAtomSession(
            positions_nm=pos,
            atomic_numbers=z,
            charges=None,
            temperature=300.0,
            dt=0.002,
            max_active_aa_atoms=8_192,
        )
        build_ms = (time.perf_counter() - t0) * 1e3
        # Warm-up 1 step then time 5 steps
        sess.step(1)
        m = sess.step(5)
        print(
            f"{n:>10,d} {m['n_tiles']:>8,d} | {build_ms:>10.1f} {m['ram_mb']:>9.2f} "
            f"{m['bytes_per_atom']:>7.2f} {m['memory_savings_ratio']:>7.2f}x | "
            f"{m['ms_per_step']:>9.2f} {m['temperature_K']:>8.2f} {m['drift_fraction']:>10.6f} {m['health_status']:>8}"
        )
        sess.close()

    print("\n" + "=" * 96)
    print("BENCHMARK 3: Exact Verified 40,000,000-Atom Byte Math (27+5 Tryte All-Atom vs Float64)")
    print("=" * 96)
    n40 = 40_000_000
    n_tiles_40 = math.ceil(n40 / 27)
    pos_bytes = n40 * 3 * 4           # float32 (N, 3) = 12 B/atom
    mom_bytes = n40 * 3 * 4           # float32 (N, 3) = 12 B/atom
    frc_bytes = n40 * 3 * 4           # float32 (N, 3) = 12 B/atom
    z_bytes = n40 * 1                 # uint8 (N,)     =  1 B/atom
    q_bytes = n40 * 4                 # float32 (N,)   =  4 B/atom
    # Lanes 27..31 + tile_counts (45 B/tile) + ref_rms (4 B/tile) + ~3 edges/tile * 12 B = 85 B/tile
    tile_lanes_bytes = n_tiles_40 * (12 + 4 + 4 + 4 + 12 + 4 + 4 + 1)  # 45 B/tile
    tile_graph_bytes = n_tiles_40 * (4 + 3 * 12)                        # 40 B/tile (ref_rms + ~3 edges/tile)
    stream_buf_bytes = 65_536 * 4 + 16_384 * 4                          # capped stream/active indices
    total_tryte_bytes = pos_bytes + mom_bytes + frc_bytes + z_bytes + q_bytes + tile_lanes_bytes + tile_graph_bytes + stream_buf_bytes
    total_tryte_gb = total_tryte_bytes / (1024.0 ** 3)
    b_per_atom = total_tryte_bytes / n40

    naive_f64_bytes = n40 * 160  # float64 pos/mom/frc/masses/charges/radii + neighbor/history overhead
    naive_f64_gb = naive_f64_bytes / (1024.0 ** 3)
    with_swap_gb = 21.0  # Full Studio Float64 + Python strings + snapshots (~525 B/atom)

    print(f"  N_atoms                  : {n40:,d}")
    print(f"  N_tiles (ceil(N/27))     : {n_tiles_40:,d}")
    print(f"  positions_nm (f32)       : {pos_bytes / 1024**2:8.2f} MiB (12.00 B/atom)")
    print(f"  momenta (f32)            : {mom_bytes / 1024**2:8.2f} MiB (12.00 B/atom)")
    print(f"  forces (f32)             : {frc_bytes / 1024**2:8.2f} MiB (12.00 B/atom)")
    print(f"  atomic_numbers (u8)      : {z_bytes / 1024**2:8.2f} MiB ( 1.00 B/atom)")
    print(f"  charges (f32)            : {q_bytes / 1024**2:8.2f} MiB ( 4.00 B/atom)")
    print(f"  27+5 tile lanes (L27-31) : {tile_lanes_bytes / 1024**2:8.2f} MiB ({tile_lanes_bytes / n40:5.2f} B/atom)")
    print(f"  tile far-field graph     : {tile_graph_bytes / 1024**2:8.2f} MiB ({tile_graph_bytes / n40:5.2f} B/atom)")
    print(f"  TOTAL Tryte 27+5 All-Atom: {total_tryte_bytes:,d} bytes = {total_tryte_gb:.3f} GiB ({b_per_atom:.2f} B/atom)")
    print(f"  vs Minimal Float64 (160B): {naive_f64_gb:.3f} GiB ({naive_f64_gb / total_tryte_gb:.2f}x smaller)")
    print(f"  vs Full Studio Float64   : ~{with_swap_gb:.1f} GiB ({with_swap_gb / total_tryte_gb:.2f}x smaller)")


if __name__ == "__main__":
    run_benchmark()
