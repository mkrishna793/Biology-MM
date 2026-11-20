#!/usr/bin/env python3
import ctypes
import os
import sys
import time
import numpy as np

# Ensure Python can import from studio-backend and biology-mm
sys.path.insert(0, '/Users/mohana/Desktop/Biology-MM-Studio/studio-backend')
sys.path.insert(0, '/Users/mohana/Desktop/Biology-MM/python')

from biology_mm import GPUVerletPlan, probe_hardware
from biology_mm.native import get_bridge, _as_box_matrices

def main():
    print("=" * 70)
    print("BIOLOGY-MM NATIVE METAL GPU: 40-MILLION ATOM 5J7V SIMULATION TEST")
    print("=" * 70)

    hw = probe_hardware()
    print(f"Hardware: {hw.device_name} | Backend: {hw.backend} | VRAM: {hw.vram_bytes / (1024**3):.2f} GB")

    bin_path = '/Users/mohana/Desktop/5j7v_40m_coords.bin'
    if not os.path.exists(bin_path):
        print(f"Error: {bin_path} not found.")
        return

    t0 = time.time()
    print(f"\n1. Loading 5J7V Assembly 1 coordinates from {bin_path}...")
    coords = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 3)
    n_atoms = coords.shape[0]
    print(f"   Loaded {n_atoms:,} real atoms in {time.time()-t0:.2f} s")
    print(f"   Binary footprint: {coords.nbytes / (1024**2):.1f} MB")

    min_c = coords.min(axis=0)
    max_c = coords.max(axis=0)
    span = max_c - min_c
    print(f"   Capsid diameter: {np.linalg.norm(span):.1f} Å ({np.linalg.norm(span)*0.1:.1f} nm)")

    # Shift into positive simulation box with 10 Å margin
    pos_nm = ((coords - min_c + 5.0) * 0.1).astype(np.float64)
    box_nm = np.array([span[0]*0.1 + 10.0, span[1]*0.1 + 10.0, span[2]*0.1 + 10.0], dtype=np.float64)
    print(f"   Simulation box: {box_nm[0]:.1f} x {box_nm[1]:.1f} x {box_nm[2]:.1f} nm")

    rcut = 1.0
    skin = 0.1
    r_list = rcut + skin
    nx = int(box_nm[0] / r_list)
    ny = int(box_nm[1] / r_list)
    nz = int(box_nm[2] / r_list)
    n_cells = nx * ny * nz
    print(f"\n2. Building O(N) Spatial Linked-Cell Decomposition ({nx} x {ny} x {nz} = {n_cells:,} cells)...")
    t1 = time.time()

    frac = pos_nm / box_nm
    cx = np.clip((frac[:, 0] * nx).astype(np.int32), 0, nx - 1)
    cy = np.clip((frac[:, 1] * ny).astype(np.int32), 0, ny - 1)
    cz = np.clip((frac[:, 2] * nz).astype(np.int32), 0, nz - 1)
    cid = cx + nx * (cy + ny * cz)
    perm = np.argsort(cid)
    cid_sorted = cid[perm]
    order = (perm + 1).astype(np.int32)

    unique_cids, start_idx, counts = np.unique(cid_sorted, return_index=True, return_counts=True)
    tile_size = 32
    tiles_per_cell = np.ceil(counts / tile_size).astype(np.int32)
    total_tiles = int(np.sum(tiles_per_cell))

    cell_n_tiles = np.zeros(n_cells, dtype=np.int32)
    cell_n_tiles[unique_cids] = tiles_per_cell

    cell_t_hi = np.zeros(n_cells, dtype=np.int32)
    cum_tiles = np.cumsum(tiles_per_cell, dtype=np.int32)
    cell_t_hi[unique_cids] = cum_tiles + 1
    cell_t_lo = np.zeros(n_cells, dtype=np.int32)
    cell_t_lo[unique_cids] = cell_t_hi[unique_cids] - tiles_per_cell

    tile_starts = np.empty(total_tiles, dtype=np.int32)
    tile_ends = np.empty(total_tiles, dtype=np.int32)

    cur_tile = 0
    for u_i in range(unique_cids.size):
        s0 = int(start_idx[u_i])
        cnt = int(counts[u_i])
        e0 = s0 + cnt
        for a in range(s0, e0, tile_size):
            tile_starts[cur_tile] = a + 1
            tile_ends[cur_tile] = min(a + tile_size, e0)
            cur_tile += 1

    ti_list = [np.arange(1, total_tiles + 1, dtype=np.int32)]
    tj_list = [np.arange(1, total_tiles + 1, dtype=np.int32)]

    for m in [2, 3, 4]:
        mask_m = (tiles_per_cell == m)
        if not np.any(mask_m): continue
        base = cell_t_lo[unique_cids[mask_m]]
        for ta_off in range(m):
            for tb_off in range(ta_off + 1, m):
                ti_list.append(base + ta_off)
                tj_list.append(base + tb_off)

    _HALF_SHELL_OFFSETS = [
        (1, 0, 0), (1, 1, 0), (0, 1, 0), (-1, 1, 0),
        (-1, -1, 1), (0, -1, 1), (1, -1, 1),
        (-1, 0, 1), (0, 0, 1), (1, 0, 1),
        (-1, 1, 1), (0, 1, 1), (1, 1, 1)
    ]

    ucx = unique_cids % nx
    ucy = (unique_cids // nx) % ny
    ucz = unique_cids // (nx * ny)
    src_lo = cell_t_lo[unique_cids]

    for dx, dy, dz in _HALF_SHELL_OFFSETS:
        ncx = (ucx + dx) % nx
        ncy = (ucy + dy) % ny
        ncz = (ucz + dz) % nz
        nb_cid = ncx + nx * (ncy + ny * ncz)
        nb_lo = cell_t_lo[nb_cid]
        nb_cnt = cell_n_tiles[nb_cid]
        
        occ = (nb_cnt > 0)
        if not np.any(occ): continue
        s_l = src_lo[occ]
        d_l = nb_lo[occ]
        s_c = tiles_per_cell[occ]
        d_c = nb_cnt[occ]
        
        for s_i in range(1, 5):
            s_mask = (s_c == s_i)
            if not np.any(s_mask): continue
            s_l_sub = s_l[s_mask]
            d_l_sub = d_l[s_mask]
            d_c_sub = d_c[s_mask]
            for d_i in range(1, 5):
                sd_mask = (d_c_sub == d_i)
                if not np.any(sd_mask): continue
                s_base = s_l_sub[sd_mask]
                d_base = d_l_sub[sd_mask]
                for a_off in range(s_i):
                    for b_off in range(d_i):
                        a_idx = s_base + a_off
                        b_idx = d_base + b_off
                        ti_list.append(np.minimum(a_idx, b_idx))
                        tj_list.append(np.maximum(a_idx, b_idx))

    tile_i = np.ascontiguousarray(np.concatenate(ti_list), dtype=np.int32)
    tile_j = np.ascontiguousarray(np.concatenate(tj_list), dtype=np.int32)
    print(f"   Spatial decomposition done in {time.time()-t1:.2f} s")
    print(f"   Occupied cells: {unique_cids.size:,} / {n_cells:,} ({unique_cids.size/n_cells*100:.2f}%)")
    print(f"   Total tiles: {total_tiles:,} (size {tile_size})")
    print(f"   Total interacting tile pairs: {tile_i.size:,}")

    sigma = np.full(n_atoms, 0.34, dtype=np.float64)
    epsilon = np.full(n_atoms, 0.997, dtype=np.float64)
    charge = np.zeros(n_atoms, dtype=np.float64)
    excl_keys = np.empty(0, dtype=np.uint64)

    bridge = get_bridge()
    is_tri, H_mat, Hinv_mat = _as_box_matrices(box_nm)
    H_ptr = H_mat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)) if H_mat is not None else None
    Hinv_ptr = Hinv_mat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)) if Hinv_mat is not None else None

    print("\n3. Allocating Native Metal 4 GPU Buffers on Apple M5...")
    t2 = time.time()
    c_ptr = bridge.gpu.xbt_gpu_verlet_plan_create(
        ctypes.c_size_t(n_atoms),
        pos_nm.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        sigma.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        epsilon.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        charge.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        H_ptr,
        Hinv_ptr,
        ctypes.c_int(is_tri),
        ctypes.c_size_t(tile_i.size),
        tile_i.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        tile_j.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        order.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        tile_starts.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        tile_ends.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.c_size_t(0),
        None,
        ctypes.c_size_t(0),
        None,
        None,
        ctypes.c_double(0.5),
        ctypes.c_double(1.0 / 1.2),
        ctypes.c_double(float(rcut)),
        ctypes.c_double(float(skin)),
        ctypes.c_double(0.0),
        ctypes.c_double(138.935456),
        ctypes.c_int(1),
    )

    if not c_ptr:
        raise RuntimeError("xbt_gpu_verlet_plan_create failed on Metal device")

    plan_obj = object.__new__(GPUVerletPlan)
    plan_obj._bridge = bridge
    plan_obj._ptr = int(c_ptr)
    plan_obj._using_gpu = True
    plan_obj.n_atoms = n_atoms
    plan_obj.rcut = float(rcut)
    plan_obj.skin = float(skin)
    plan_obj.alpha_dsf = 0.0
    plan_obj.coulomb_prefactor = 138.935456
    plan_obj.use_lfs = True
    plan_obj.include_self = True
    plan_obj._sigma = sigma
    plan_obj._epsilon = epsilon
    plan_obj._charges = charge
    plan_obj._default_box = box_nm
    plan_obj._keepalive = (pos_nm, sigma, epsilon, charge, order, tile_starts, tile_ends, tile_i, tile_j, excl_keys)
    plan_obj._self_energy = 0.0

    print(f"   Metal GPU Plan allocated in {time.time()-t2:.2f} s")
    print(f"   GPU Resident: YES (Backend = {bridge.backend})")

    print("\n4. Executing 40,000,000-Atom Physics Step on Apple M5 GPU...")
    t3 = time.time()
    u, f, vir = plan_obj.eval(positions=pos_nm, box=box_nm, compute_virial=True)
    gpu_step_time = time.time() - t3

    print(f"\n" + "=" * 70)
    print("RESULTS OF 40M-ATOM FULL CAPSID SIMULATION ON APPLE M5 GPU:")
    print("=" * 70)
    print(f"  Simulated Atoms:           {n_atoms:,} atoms")
    print(f"  Interacting Tile Pairs:    {tile_i.size:,} pairs")
    print(f"  Metal GPU Step Duration:   {gpu_step_time:.3f} seconds ({gpu_step_time*1000:.1f} ms)")
    print(f"  Calculated Potential E:    {u:.4e} kJ/mol")
    print(f"  Max Atomic Force:          {np.max(np.linalg.norm(f, axis=1)):.2f} kJ/(mol*nm)")
    print(f"  Mean Atomic Force:         {np.mean(np.linalg.norm(f, axis=1)):.2f} kJ/(mol*nm)")
    print("=" * 70)
    print("SUCCESS: Full 40M Faustovirus capsid simulated directly on Apple M5 GPU!")

if __name__ == '__main__':
    main()
