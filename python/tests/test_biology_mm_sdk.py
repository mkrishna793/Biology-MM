"""Comprehensive pytest suite for the Biology MM Python SDK."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pytest

import biology_mm
from biology_mm import (
    AtomicSystem,
    CustomPotential,
    GPUVerletPlan,
    NanostructureBuilder,
    ParameterEncoder,
    ParameterSpec,
    Simulation,
    SimulationConfig,
    SystemBuilder,
    csa_accumulate,
    erfc_fast,
    exp_radix3,
    get_bridge,
    gpu_msm_stencil27,
    gpu_ternary_matvec,
    mm_abi_version,
    mm_version,
    probe_hardware,
    rsqrt_fast,
    run_batched_gpu_steps,
    tanh_radix3,
)

PYTHON_ROOT = Path(__file__).resolve().parents[1]


def _mic_disp_np(
    dx0: float,
    dy0: float,
    dz0: float,
    box: Optional[Sequence[float] | np.ndarray],
) -> np.ndarray:
    d0 = np.array([dx0, dy0, dz0], dtype=np.float64)
    if box is None:
        return d0
    arr = np.asarray(box, dtype=np.float64)
    if arr.shape == (3,):
        return d0 - arr * np.rint(d0 / arr)
    H = arr
    Hinv = np.linalg.inv(H)
    f0 = Hinv @ d0
    n0 = np.rint(f0)
    best = d0 - H @ n0
    best2 = float(np.dot(best, best))
    for da in (-1, 0, 1):
        for db in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if da == 0 and db == 0 and dc == 0:
                    continue
                cand_n = n0 + np.array([da, db, dc], dtype=np.float64)
                cand = d0 - H @ cand_n
                c2 = float(np.dot(cand, cand))
                if c2 < best2:
                    best2 = c2
                    best = cand
    return best


def _reference_lj_dsf_np(
    positions: np.ndarray,
    sigma: np.ndarray,
    epsilon: np.ndarray,
    charges: np.ndarray,
    box: Optional[Sequence[float] | np.ndarray],
    rcut: float,
    alpha_dsf: float = 0.0,
    coulomb_prefactor: float = 138.935456,
    use_lfs: bool = True,
    include_self: bool = True,
    exclusions: Sequence[Tuple[int, int]] = (),
    scaled_14: Sequence[Tuple[int, int, float]] = (),
    lj14_scale: float = 0.5,
    coul14_scale: float = 1.0 / 1.2,
) -> Tuple[float, np.ndarray, np.ndarray]:
    n = positions.shape[0]
    forces = np.zeros((n, 3), dtype=np.float64)
    virial = np.zeros((3, 3), dtype=np.float64)
    energy = 0.0

    excl_set = {(min(i, j), max(i, j)) for i, j in exclusions}
    scaled_map = {(min(i, j), max(i, j)): w for i, j, w in scaled_14}

    do_coul = alpha_dsf > 0.0 and coulomb_prefactor != 0.0
    if do_coul:
        two_a_sqrtpi = 2.0 * alpha_dsf / math.sqrt(math.pi)
        erfc_rc = math.erfc(alpha_dsf * rcut)
        exp_rc = math.exp(-alpha_dsf * alpha_dsf * rcut * rcut)
        arc = erfc_rc / (rcut * rcut) + two_a_sqrtpi * exp_rc / rcut
        brc = erfc_rc / rcut + arc * rcut
        if include_self:
            energy -= coulomb_prefactor * (erfc_rc / (2.0 * rcut) + alpha_dsf / math.sqrt(math.pi)) * float(
                np.sum(charges * charges)
            )
    else:
        two_a_sqrtpi = arc = brc = 0.0

    Hinv = None
    if box is not None:
        arr = np.asarray(box, dtype=np.float64)
        Hinv = np.diag(1.0 / arr) if arr.shape == (3,) else np.linalg.inv(arr)

    rc2 = rcut * rcut
    for i in range(n - 1):
        for j in range(i + 1, n):
            key = (i, j)
            if key in excl_set:
                continue
            if key in scaled_map:
                w_raw = scaled_map[key]
                w_lj = w_raw if w_raw != 1.0 else lj14_scale
                w_coul = coul14_scale
            else:
                w_lj = 1.0
                w_coul = 1.0

            d = _mic_disp_np(
                positions[i, 0] - positions[j, 0],
                positions[i, 1] - positions[j, 1],
                positions[i, 2] - positions[j, 2],
                box,
            )
            r2 = float(np.dot(d, d))
            if r2 >= rc2:
                continue
            r = math.sqrt(r2)
            inv_r = 1.0 / r
            inv_r2 = 1.0 / r2

            u_pair = 0.0
            c_total = 0.0

            eij = math.sqrt(epsilon[i] * epsilon[j])
            sij = 0.5 * (sigma[i] + sigma[j])
            if eij > 0.0 and sij > 0.0 and w_lj != 0.0:
                s2 = (sij * sij) * inv_r2
                s6 = s2 * s2 * s2
                s12 = s6 * s6
                s2c = (sij * sij) / rc2
                s6c = s2c * s2c * s2c
                s12c = s6c * s6c
                u_lj = 4.0 * eij * ((s12 - s6) - (s12c - s6c))
                c_lj = 24.0 * eij * (2.0 * s12 - s6) * inv_r2
                if use_lfs:
                    duc = 24.0 * eij * (s6c - 2.0 * s12c) / rcut
                    u_lj -= duc * (r - rcut)
                    c_lj += duc * inv_r
                u_pair += w_lj * u_lj
                c_total += w_lj * c_lj

            if do_coul and w_coul != 0.0 and charges[i] != 0.0 and charges[j] != 0.0:
                kq = w_coul * coulomb_prefactor * charges[i] * charges[j]
                z = alpha_dsf * r
                er = math.erfc(z)
                ex = math.exp(-z * z)
                u_pair += kq * (er * inv_r - brc + arc * r)
                c_total += kq * (er * inv_r2 * inv_r + two_a_sqrtpi * ex * inv_r2 - arc * inv_r)

            energy += u_pair
            f_ij = c_total * d
            forces[i] += f_ij
            forces[j] -= f_ij

            if Hinv is not None:
                w = Hinv @ d
                virial += np.outer(-f_ij, w)

    return energy, forces, virial


def test_native_cpu_bridge_math_and_csa():
    bridge = get_bridge()
    assert bridge.has_cpu_bridge
    assert mm_version() == 202611
    assert mm_abi_version() == 2

    for x in (-3.25, -0.4, 0.0, 0.75, 4.1):
        assert math.isclose(exp_radix3(x), math.exp(x), rel_tol=1e-12, abs_tol=1e-14)
        assert math.isclose(tanh_radix3(x), math.tanh(x), rel_tol=1e-10, abs_tol=1e-12)
    for x in (0.05, 0.5, 1.0, 3.5, 25.0):
        assert math.isclose(rsqrt_fast(x), 1.0 / math.sqrt(x), rel_tol=1e-12)
    for x in (-2.0, 0.0, 0.35, 1.5, 4.0):
        assert abs(erfc_fast(x) - math.erfc(x)) < 2e-7

    rng = np.random.default_rng(12345)
    terms = rng.integers(-50_000, 50_000, size=200, dtype=np.int64)
    res1 = csa_accumulate(terms)
    res2 = csa_accumulate(rng.permutation(terms))
    assert not res1.overflow
    assert res1.resolved == int(np.sum(terms))
    assert res2.resolved == res1.resolved
    assert (res1.s_p & res1.s_n) == 0
    assert (res1.c_p & res1.c_n) == 0


def test_apple_metal_probe_and_verlet_lj_dsf_ortho_and_triclinic():
    prof = probe_hardware()
    has_gpu = get_bridge().has_gpu_bridge and prof.has_gpu
    if has_gpu:
        assert prof.backend == "metal"
        assert prof.unified_memory is True
        assert prof.simd_width >= 32
        assert prof.vram_bytes > 0
    else:
        assert prof.backend == "cpu"

    rng = np.random.default_rng(202611)
    n = 48
    positions = np.empty((n, 3), dtype=np.float64)
    idx = 0
    for ix in range(4):
        for iy in range(4):
            for iz in range(3):
                positions[idx] = [
                    (ix + 0.3 + 0.25 * rng.random()) * (4.6 / 4.0),
                    (iy + 0.3 + 0.25 * rng.random()) * (4.6 / 4.0),
                    (iz + 0.3 + 0.25 * rng.random()) * (4.6 / 3.0),
                ]
                idx += 1

    sigma = 0.28 + 0.05 * rng.random(n)
    epsilon = 0.35 + 0.25 * rng.random(n)
    charges = 0.45 * (rng.random(n) - 0.5)
    charges -= np.mean(charges)

    exclusions = [(i, i + 1) for i in range(0, n - 1, 2)]
    scaled_14 = [(i, i + 2, 0.5) for i in range(0, n - 2, 4)]

    box_ortho = (4.8, 5.0, 5.2)
    box_tri = np.array([[5.1, 0.35, 0.15], [0.0, 5.0, 0.25], [0.0, 0.0, 4.9]], dtype=np.float64)
    rcut = 1.9
    alpha = 0.24

    for box in (box_ortho, box_tri):
        u_ref, f_ref, v_ref = _reference_lj_dsf_np(
            positions,
            sigma,
            epsilon,
            charges,
            box=box,
            rcut=rcut,
            alpha_dsf=alpha,
            use_lfs=True,
            include_self=True,
            exclusions=exclusions,
            scaled_14=scaled_14,
        )

        with GPUVerletPlan(
            positions=positions,
            sigma=sigma,
            epsilon=epsilon,
            charges=charges,
            box=box,
            exclusions=exclusions,
            scaled_14=scaled_14,
            rcut=rcut,
            skin=0.25,
            alpha_dsf=alpha,
            use_lfs=True,
            include_self=True,
            backend="gpu" if has_gpu else "cpu",
        ) as plan:
            assert plan.using_gpu is has_gpu
            assert plan.num_pairs > 0
            u_eval, f_eval, v_eval = plan.eval(positions=positions, compute_virial=True)

        rel_u = abs(u_eval - u_ref) / max(1.0, abs(u_ref))
        rel_f = np.max(np.abs(f_eval - f_ref)) / max(1.0, float(np.max(np.abs(f_ref))))
        rel_v = np.max(np.abs(v_eval - v_ref)) / max(1.0, float(np.max(np.abs(v_ref))))

        assert rel_u < 1e-5, f"Energy rel error {rel_u}"
        assert rel_f < 1e-5, f"Force rel error {rel_f}"
        assert rel_v < 1e-5, f"Virial rel error {rel_v}"

    # Central finite-difference check of forces and virial on triclinic cell
    with GPUVerletPlan(
        positions=positions,
        sigma=sigma,
        epsilon=epsilon,
        charges=charges,
        box=box_tri,
        exclusions=exclusions,
        scaled_14=scaled_14,
        rcut=rcut,
        skin=0.25,
        alpha_dsf=alpha,
        backend="cpu",
    ) as cpu_plan:
        _, f_cpu, v_cpu = cpu_plan.eval(positions=positions, compute_virial=True)
        h = 1e-6
        for atom in (0, 17, n - 1):
            for dim in range(3):
                qp = positions.copy()
                qm = positions.copy()
                qp[atom, dim] += h
                qm[atom, dim] -= h
                up, _, _ = cpu_plan.eval(positions=qp, compute_virial=False)
                um, _, _ = cpu_plan.eval(positions=qm, compute_virial=False)
                f_fd = -(up - um) / (2.0 * h)
                assert abs(f_cpu[atom, dim] - f_fd) < 1e-6

        # Box derivative FD check: q_scaled = Hinv @ q
        Hinv_tri = np.linalg.inv(box_tri)
        s_coords = positions @ Hinv_tri.T
        for a in range(3):
            for b in range(3):
                Hp = box_tri.copy()
                Hm = box_tri.copy()
                Hp[a, b] += h
                Hm[a, b] -= h
                up, _, _ = cpu_plan.eval(positions=s_coords @ Hp.T, box=Hp, compute_virial=False)
                um, _, _ = cpu_plan.eval(positions=s_coords @ Hm.T, box=Hm, compute_virial=False)
                v_fd = (up - um) / (2.0 * h)
                assert abs(v_cpu[a, b] - v_fd) < 1e-5


def test_gpu_ternary_matvec_msm_stencil27_and_batched_md():
    rng = np.random.default_rng(7711)

    # Dual-rail ternary matvec exact integer parity
    M, K = 24, 5
    a = rng.integers(0, np.iinfo(np.uint64).max, size=(M, K), dtype=np.uint64)
    b = rng.integers(0, np.iinfo(np.uint64).max, size=(M, K), dtype=np.uint64)
    W_P = a & ~b
    W_N = b & ~a
    xa = rng.integers(0, np.iinfo(np.uint64).max, size=(K,), dtype=np.uint64)
    xb = rng.integers(0, np.iinfo(np.uint64).max, size=(K,), dtype=np.uint64)
    x_P = xa & ~xb
    x_N = xb & ~xa
    row_scales = rng.uniform(0.5, 2.0, size=M)
    x_scale = 0.25

    y_gpu = gpu_ternary_matvec(W_P, W_N, x_P, x_N, row_scales=row_scales, x_scale=x_scale, backend="auto")
    y_cpu = gpu_ternary_matvec(W_P, W_N, x_P, x_N, row_scales=row_scales, x_scale=x_scale, backend="cpu")

    # Exact bit-count reference
    y_ref = np.zeros(M, dtype=np.float64)
    for m in range(M):
        acc = 0
        for k in range(K):
            pos = int((int(W_P[m, k]) & int(x_P[k])) | (int(W_N[m, k]) & int(x_N[k]))).bit_count()
            neg = int((int(W_P[m, k]) & int(x_N[k])) | (int(W_N[m, k]) & int(x_P[k]))).bit_count()
            acc += pos - neg
        y_ref[m] = row_scales[m] * x_scale * acc

    np.testing.assert_allclose(y_gpu, y_ref, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(y_cpu, y_ref, rtol=1e-14, atol=1e-14)

    # 27-point periodic stencil convolution parity
    ml = 8
    Q = rng.standard_normal((ml, ml, ml))
    k27 = rng.standard_normal(27)
    u_gpu, phi_gpu = gpu_msm_stencil27(Q, k27, prefactor=1.7, backend="auto")
    u_ref, phi_ref = gpu_msm_stencil27(Q, k27, prefactor=1.7, backend="cpu")
    assert abs(u_gpu - u_ref) / max(1.0, abs(u_ref)) < 1e-5
    assert np.max(np.abs(phi_gpu - phi_ref)) < 1e-4

    # Batched GPU Velocity Verlet (when native GPU bridge is active)
    if get_bridge().has_gpu_bridge:
        fcc = SystemBuilder.fcc_lattice(n_cells=2, rho=0.7, mass=39.948)
        n = fcc.n_atoms
        momenta = np.zeros((n, 3), dtype=np.float64)
        sig = np.full(n, 0.34)
        eps = np.full(n, 0.996)
        res0 = run_batched_gpu_steps(fcc.positions, momenta, fcc.masses, sig, eps, dt=0.001, n_steps=0, box=fcc.box_lengths, rcut=1.4)
        res1 = run_batched_gpu_steps(fcc.positions, momenta, fcc.masses, sig, eps, dt=0.001, n_steps=20, box=fcc.box_lengths, rcut=1.4)
        assert abs(res1["total_energy"] - res0["total_energy"]) / max(1.0, abs(res0["total_energy"])) < 1e-4


def test_parameter_encoder_all_encodings_and_jacobian_fd():
    specs = [
        ParameterSpec("k", default=250.0, unit="kJ/mol/nm^2", lower=-1000.0, upper=1000.0, encoding="identity"),
        ParameterSpec("sigma", default=0.34, unit="nm", lower=0.05, encoding="log"),
        ParameterSpec("scale", default=0.5, lower=0.0, upper=1.0, encoding="logit"),
        ParameterSpec("theta", default=0.85, unit="rad", encoding="angle"),
        ParameterSpec("w_trit", default=-0.25, trit_scale=0.25, encoding="ternary_quantized"),
    ]
    enc = ParameterEncoder(specs)
    theta0 = enc.defaults
    z0 = enc.encode(theta0)
    theta_rt = enc.decode(z0)
    np.testing.assert_allclose(theta_rt, theta0, rtol=1e-12, atol=1e-12)

    # Central FD check on continuous encodings (indices 0..3) and STE check on index 4
    jac = enc.jacobian_diag(z0)
    h = 1e-6
    for idx in range(4):
        zp = z0.copy()
        zm = z0.copy()
        zp[idx] += h
        zm[idx] -= h
        fd = (enc.decode(zp)[idx] - enc.decode(zm)[idx]) / (2.0 * h)
        assert abs(jac[idx] - fd) < 1e-7
    assert jac[4] == 0.25

    # Bounds validation & serialization round-trip
    with pytest.raises(ValueError):
        enc.validate([250.0, 0.01, 0.5, 0.85, -0.25])  # sigma < lower=0.05
    with pytest.raises(ValueError):
        enc.validate([250.0, 0.34, 1.5, 0.85, -0.25])  # scale > upper=1.0

    rt_enc = ParameterEncoder.from_dict(enc.to_dict())
    np.testing.assert_allclose(rt_enc.defaults, enc.defaults)
    assert [s.encoding for s in rt_enc.specs] == [s.encoding for s in enc.specs]


def _pbc_degrees_and_lengths(sys: AtomicSystem) -> Tuple[np.ndarray, np.ndarray]:
    n = sys.n_atoms
    deg = np.zeros(n, dtype=int)
    lengths = []
    lx, ly, lz = sys.box_lengths
    for i, j in sys.bonds:
        deg[i] += 1
        deg[j] += 1
        d = sys.positions[i] - sys.positions[j]
        d[0] -= lx * round(d[0] / lx)
        d[1] -= ly * round(d[1] / ly)
        d[2] -= lz * round(d[2] / lz)
        lengths.append(float(np.linalg.norm(d)))
    return deg, np.array(lengths, dtype=np.float64)


def test_nanostructure_builders_exact_geometry():
    # (5,5) armchair CNT: 20 atoms per cell -> 40 atoms for n_cells=2
    cnt55 = NanostructureBuilder.carbon_nanotube(5, 5, n_cells=2, bond_length_nm=0.142)
    assert cnt55.n_atoms == 40
    deg55, len55 = _pbc_degrees_and_lengths(cnt55)
    assert np.all(deg55 == 3)
    assert np.all((len55 > 0.138) & (len55 < 0.145))

    # (6,0) zigzag CNT: 24 atoms per cell -> 48 atoms for n_cells=2
    cnt60 = NanostructureBuilder.carbon_nanotube(6, 0, n_cells=2, bond_length_nm=0.142)
    assert cnt60.n_atoms == 48
    deg60, len60 = _pbc_degrees_and_lengths(cnt60)
    assert np.all(deg60 == 3)
    assert np.all((len60 > 0.138) & (len60 < 0.145))

    # Periodic graphene sheet
    graph = NanostructureBuilder.graphene_sheet(3, 2, bond_length_nm=0.142)
    assert graph.n_atoms == 24
    deg_g, len_g = _pbc_degrees_and_lengths(graph)
    assert np.all(deg_g == 3)
    np.testing.assert_allclose(len_g, 0.142, atol=1e-12)

    # Nanopore membrane: no atom closer than pore_radius_nm to XY center
    pore = NanostructureBuilder.nanopore_membrane(5, 4, pore_radius_nm=0.45, bond_length_nm=0.142)
    assert pore.n_atoms < 80
    cx, cy = 0.5 * pore.box_lengths[0], 0.5 * pore.box_lengths[1]
    r_xy = np.hypot(pore.positions[:, 0] - cx, pore.positions[:, 1] - cy)
    assert np.all(r_xy >= 0.45 - 1e-12)

    # Silicon diamond: 8 * 2^3 = 64 atoms, 4-coordinated tetrahedral
    si = NanostructureBuilder.silicon_diamond(2, 2, 2, a_nm=0.5431)
    assert si.n_atoms == 64
    deg_si, len_si = _pbc_degrees_and_lengths(si)
    assert np.all(deg_si == 4)
    np.testing.assert_allclose(len_si, math.sqrt(3.0) / 4.0 * 0.5431, atol=1e-12)


def test_custom_potential_from_callable_and_indexing():
    pot = CustomPotential.from_callable(
        lambda r: 2.5 * (r - 0.5) ** 2,
        r_min=0.2,
        r_max=1.2,
        pairs=[(0, 1), (2, 3)],
        n_grid=128,
        mode="lfs",
    )
    spec = pot.to_spec_dict(atom_index_base=1)
    assert spec["type"] == "tabulated_spline"
    assert spec["pairs"] == [[1, 2], [3, 4]]
    assert len(spec["r_grid"]) == 128
    assert len(spec["u_grid"]) == 128
    assert abs(spec["u_grid"][0] - 2.5 * (0.2 - 0.5) ** 2) < 1e-12

    hb = CustomPotential.harmonic_bond(0, 1, k=10.0, r0=1.3).to_spec_dict(atom_index_base=1)
    assert hb == {"type": "harmonic_bond", "i": 1, "j": 2, "k": 10.0, "r0": 1.3}

    wall = CustomPotential.wall_restraint("harmonic", axis=2, lo=0.2, hi=3.8, k=50.0).to_spec_dict(atom_index_base=1)
    assert wall["axis"] == 3


def test_simulation_end_to_end_julia_md(tmp_path):
    sb = SystemBuilder(box_lengths=(20.0, 20.0, 20.0))
    i0 = sb.add_atom((1.0, 1.0, 1.0), mass=1.0, name="A")
    i1 = sb.add_atom((2.35, 1.0, 1.0), mass=1.0, name="A")
    sys = sb.build()

    pots = [
        CustomPotential.harmonic_bond(i0, i1, k=2.0, r0=1.3),
        CustomPotential.lj_pair(i0, i1, epsilon=0.05, sigma=1.1, rc=4.0, mode="lfs"),
    ]
    cfg = SimulationConfig(
        ensemble="nve",
        integrator="velocity_verlet",
        dt=0.002,
        temperature=0.0,
        minimize_max_steps=0,
        equilibrate_steps=0,
        produce_steps=60,
        frame_every=2,
        checkpoint_every=0,
        contact_tol=0.05,
        run_id="pytest-e2e",
    )
    sim = Simulation(sys, potentials=pots, config=cfg)
    result = sim.run(workdir=tmp_path / "md_work")

    assert result.stages == ["validate", "prepare", "minimize", "equilibrate", "produce", "analyze"]
    assert result.n_frames == 30
    assert len(result.frames) == 30
    assert result.frames[0]["positions"].shape == (2, 3)
    assert len(result.log_records) == 30

    # Also test native evaluator directly on Simulation
    u_nat, f_nat, v_nat = sim.evaluate_lj_dsf_native(sigma=1.1, epsilon=0.05, rcut=4.0, backend="gpu")
    assert math.isfinite(u_nat)
    assert f_nat.shape == (2, 3)
    assert v_nat.shape == (3, 3)


def test_python_cnt_tersoff_e2e_julia_md(tmp_path):
    cnt = NanostructureBuilder.carbon_nanotube(4, 4, n_cells=1, bond_length_nm=0.142)
    assert cnt.units == "md"
    pot = CustomPotential.tersoff(element="C", units="md")
    cfg = SimulationConfig(
        ensemble="nve",
        integrator="velocity_verlet",
        dt=0.0005,
        temperature=0.0,
        minimize_max_steps=0,
        equilibrate_steps=0,
        produce_steps=20,
        frame_every=1,
        checkpoint_every=0,
        contact_tol=0.05,
        run_id="pytest-cnt-tersoff",
    )
    sim = Simulation(cnt, potentials=[pot], config=cfg)
    spec = sim.to_spec_dict(tmp_path / "cnt_work")
    assert spec["units"] == "md"
    result = sim.run(workdir=tmp_path / "cnt_work")
    assert result.stages == ["validate", "prepare", "minimize", "equilibrate", "produce", "analyze"]
    assert result.n_frames == 20
    assert len(result.frames) == 20
    assert result.frames[0]["positions"].shape == (cnt.n_atoms, 3)
    assert len(result.log_records) == 20


def test_python_from_callable_tabulated_spline_e2e_julia_md(tmp_path):
    sb = SystemBuilder(box_lengths=(12.0, 12.0, 12.0))
    i0 = sb.add_atom((2.0, 2.0, 2.0), mass=1.0, name="A")
    i1 = sb.add_atom((3.25, 2.0, 2.0), mass=1.0, name="A")
    sys = sb.build()

    pot = CustomPotential.from_callable(
        lambda r: 2.0 * (r - 1.1) ** 2,
        r_min=0.5,
        r_max=3.0,
        pairs=[(i0, i1)],
        n_grid=128,
        rc=2.8,
        mode="lfs",
    )
    cfg = SimulationConfig(
        ensemble="nve",
        integrator="velocity_verlet",
        dt=0.002,
        temperature=0.0,
        minimize_max_steps=0,
        equilibrate_steps=0,
        produce_steps=20,
        frame_every=1,
        checkpoint_every=0,
        contact_tol=0.05,
        run_id="pytest-spline-e2e",
    )
    sim = Simulation(sys, potentials=[pot], config=cfg)
    spec = sim.to_spec_dict(tmp_path / "spline_work")
    assert "units" not in spec
    result = sim.run(workdir=tmp_path / "spline_work")
    assert result.stages == ["validate", "prepare", "minimize", "equilibrate", "produce", "analyze"]
    assert result.n_frames == 20
    assert len(result.frames) == 20
    assert result.frames[0]["positions"].shape == (2, 3)


def test_cli_info():
    import os

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PYTHON_ROOT)
    proc = subprocess.run(
        [sys.executable, "-m", "biology_mm", "info", "--json"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["sdk_version"] == "3.0.0"
    assert payload["cpu_bridge_loaded"] is True
    if get_bridge().has_gpu_bridge:
        assert payload["gpu_bridge_loaded"] is True
        assert payload["hardware"]["backend"] == "metal"
        assert payload["hardware"]["has_gpu"] is True
    else:
        assert payload["gpu_bridge_loaded"] is False
        assert payload["hardware"]["backend"] == "cpu"
        assert payload["hardware"]["has_gpu"] is False
    assert payload["julia_toolchain"]["available"] is True
    assert biology_mm.__version__ == "3.0.0"
