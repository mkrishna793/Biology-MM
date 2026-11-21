#!/usr/bin/env python3
"""Biology-MM Open Physics Validation Benchmark Suite (Reproducible Open-Science Proof).

Runs 5 self-contained, reproducible physical & interoperability benchmarks and writes:
  - validation/OPEN_VALIDATION_REPORT.md
  - validation/open_validation_results.json

Benchmarks:
  1. NVE Symplectic Velocity-Verlet Energy Conservation (LJ Liquid Argon, 94.4 K)
  2. Bulk TIP3P Water Thermodynamic Density & O-O Radial Distribution Function g(r) (300 K, 1 bar NPT)
  3. Alanine Dipeptide Enhanced Sampling (Well-Tempered Metadynamics + Analytical CV VJP Check)
  4. Multi-Domain Chemistry & Physics Verification (Morse Bond Oscillator + Yukawa Screened Plasma)
  5. Standard Binary .dcd / .xtc / .pdb Trajectory Interoperability Round-Trip Audit

Usage:
  PYTHONPATH="python:studio/backend" python3 validation/run_open_validation.py
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "python") not in sys.path:
    sys.path.insert(0, str(ROOT / "python"))

from biology_mm import (
    DihedralCV,
    DistanceCV,
    EnhancedSamplingSuite,
    OpenScienceEngine,
    PhysicsParameters,
    WellTemperedMetadynamics,
    build_alanine_dipeptide,
    build_lj_argon_fluid,
    build_morse_diatomic,
    build_tip3p_water_box,
    build_yukawa_plasma,
    radial_distribution_function,
    read_dcd,
    read_xtc,
)


def _finite_diff_grad(cv, pos: np.ndarray, box=None, eps: float = 1e-6) -> np.ndarray:
    num = np.zeros_like(pos)
    for i in range(pos.shape[0]):
        for d in range(3):
            p_plus = pos.copy()
            p_minus = pos.copy()
            p_plus[i, d] += eps
            p_minus[i, d] -= eps
            vp = cv.evaluate(p_plus, box)
            vm = cv.evaluate(p_minus, box)
            diff = vp - vm
            if cv.periodic is not None:
                diff = (diff + 0.5 * cv.periodic) % cv.periodic - 0.5 * cv.periodic
            num[i, d] = diff / (2.0 * eps)
    return num


def bench_1_nve_argon() -> Dict[str, Any]:
    """Benchmark 1: NVE symplectic energy conservation on periodic LJ liquid Argon."""
    t0 = time.perf_counter()
    top = build_lj_argon_fluid(n_side=4, density_g_cm3=1.374)  # 64 Ar atoms
    # Brief NVT equilibration, then switch to pure NVE Velocity-Verlet
    params_eq = PhysicsParameters(
        integrator="baoab_nvt",
        pair_style="lj_lfs",
        dt_ps=0.002,
        temperature_K=94.4,
        rcut_nm=0.80,
        seed=101,
    )
    eng = OpenScienceEngine(top, params_eq)
    eng.minimize(max_steps=40)
    eng.run(n_steps=80, report_interval=20)

    # Pure NVE production run
    eng.set_parameters(integrator="nve_verlet", dt_ps=0.002)
    rep = eng.run(n_steps=250, report_interval=5)
    elapsed = time.perf_counter() - t0

    e_std = float(np.std(rep.total_energy_kj_mol))
    e_mean = float(np.mean(rep.total_energy_kj_mol))
    rel_fluct = e_std / max(1e-9, abs(e_mean))
    passed = bool(rel_fluct < 5e-3 and np.all(np.isfinite(rep.total_energy_kj_mol)))
    return {
        "name": "NVE Symplectic Energy Conservation (LJ Argon, C1 LFS)",
        "domain": "Physics",
        "n_atoms": top.n_atoms,
        "steps": 250,
        "dt_ps": 0.002,
        "mean_total_energy_kj_mol": round(e_mean, 4),
        "std_total_energy_kj_mol": round(e_std, 6),
        "relative_energy_fluctuation": round(rel_fluct, 7),
        "mean_temperature_K": round(rep.mean_temperature_K, 2),
        "elapsed_s": round(elapsed, 3),
        "passed": passed,
    }


def bench_2_tip3p_water_npt() -> Dict[str, Any]:
    """Benchmark 2: Bulk TIP3P water NPT density & O-O radial distribution function g(r)."""
    t0 = time.perf_counter()
    top = build_tip3p_water_box(n_side=4)  # 64 H2O = 192 atoms
    params = PhysicsParameters(
        integrator="baoab_npt",
        pair_style="lj_lfs",
        dt_ps=0.0005,  # 0.5 fs for flexible O-H bonds
        temperature_K=300.0,
        pressure_bar=1.0,
        gamma_ps_inv=15.0,
        rcut_nm=0.58,
        coulomb_kappa_nm_inv=2.5,
        barostat_interval=15,
        barostat_max_dV_frac=0.015,
        seed=202,
    )
    eng = OpenScienceEngine(top, params)
    eng.minimize(max_steps=80, step_size_nm=0.001)
    eng._init_velocities(300.0)
    # Equilibrate in NPT at 300 K
    eng.run(n_steps=160, report_interval=80)

    # Sample production frames for density and g_OO(r)
    frames = []
    boxes = []
    densities = []
    temps = []
    for s in range(1, 161):
        eng.step()
        if s % 8 == 0:
            frames.append(eng.positions.copy())
            boxes.append(eng.box.copy())
            densities.append(eng.density_g_cm3())
            temps.append(eng.temperature())

    o_indices = [i for i, el in enumerate(top.elements) if el == "O"]
    r_nm, g_oo = radial_distribution_function(
        np.stack(frames, axis=0),
        np.stack(boxes, axis=0),
        indices=o_indices,
        r_max_nm=0.60,
        n_bins=60,
    )
    peak_idx = int(np.argmax(g_oo))
    peak_r_ang = float(r_nm[peak_idx] * 10.0)
    peak_val = float(g_oo[peak_idx])
    mean_rho = float(np.mean(densities))
    mean_T = float(np.mean(temps))
    elapsed = time.perf_counter() - t0

    passed = bool(0.75 <= mean_rho <= 1.25 and 2.4 <= peak_r_ang <= 3.6 and peak_val > 1.2)
    return {
        "name": "Bulk TIP3P Water NPT Density & g_OO(r) (300 K, 1 bar)",
        "domain": "Biology / Chemistry",
        "n_molecules": 64,
        "n_atoms": top.n_atoms,
        "mean_density_g_cm3": round(mean_rho, 4),
        "reference_water_density_g_cm3": 0.997,
        "g_oo_first_peak_ang": round(peak_r_ang, 3),
        "g_oo_first_peak_height": round(peak_val, 3),
        "mean_temperature_K": round(mean_T, 2),
        "elapsed_s": round(elapsed, 3),
        "passed": passed,
    }


def bench_3_alanine_dipeptide_metad() -> Dict[str, Any]:
    """Benchmark 3: Alanine dipeptide backbone phi/psi Well-Tempered Metadynamics + VJP audit."""
    t0 = time.perf_counter()
    top = build_alanine_dipeptide()
    phi_cv = DihedralCV(1, 3, 4, 6, name="phi")
    psi_cv = DihedralCV(3, 4, 6, 8, name="psi")

    # 1. Exact analytical vs finite-difference VJP gradient verification
    _, g_ana = phi_cv.evaluate_with_grad(top.positions_nm)
    g_num = _finite_diff_grad(phi_cv, top.positions_nm)
    max_grad_err = float(np.max(np.abs(g_ana - g_num)))

    # 2. Well-Tempered Metadynamics on phi
    metad = WellTemperedMetadynamics(
        phi_cv, height_kj_mol=1.5, sigma=0.35, bias_factor=8.0, temperature_K=300.0, stride=15
    )
    suite = EnhancedSamplingSuite()
    suite.add_cv(phi_cv)
    suite.add_cv(psi_cv)
    suite.add_bias(metad)

    params = PhysicsParameters(
        integrator="baoab_nvt",
        pair_style="lj_lfs",
        dt_ps=0.001,
        temperature_K=300.0,
        gamma_ps_inv=5.0,
        seed=303,
    )
    eng = OpenScienceEngine(top, params, enhanced_sampling=suite)
    rep = eng.run(n_steps=300, report_interval=10)
    grid, fes = metad.free_energy_surface(np.linspace(-math.pi, math.pi, 36))
    elapsed = time.perf_counter() - t0

    phi_vals = [h["phi"] for h in rep.cv_history if "phi" in h]
    phi_span_deg = float(math.degrees(max(phi_vals) - min(phi_vals))) if phi_vals else 0.0
    fes_barrier = float(np.max(fes) - np.min(fes))
    passed = bool(max_grad_err < 1e-6 and metad.n_hills == 20 and fes_barrier > 0.5)
    return {
        "name": "Alanine Dipeptide Well-Tempered Metadynamics & CV VJP Verification",
        "domain": "Biology (Enhanced Sampling)",
        "n_atoms": top.n_atoms,
        "vjp_max_grad_error": float(f"{max_grad_err:.3e}"),
        "metad_hills_deposited": metad.n_hills,
        "phi_sampled_span_deg": round(phi_span_deg, 2),
        "reconstructed_fes_range_kj_mol": round(fes_barrier, 3),
        "elapsed_s": round(elapsed, 3),
        "passed": passed,
    }


def bench_4_chemistry_and_physics_styles() -> Dict[str, Any]:
    """Benchmark 4: Morse bond oscillator (Chemistry) + Yukawa screened plasma (Physics)."""
    t0 = time.perf_counter()
    # 4a. Morse diatomic NVE energy conservation
    morse_top = build_morse_diatomic(r_init_nm=0.165)
    morse_params = PhysicsParameters(
        integrator="nve_verlet",
        pair_style="morse",
        dt_ps=0.0005,
        temperature_K=0.0,
        rcut_nm=1.0,
        morse_De_kj_mol=420.0,
        morse_a_nm_inv=19.0,
        morse_r0_nm=0.140,
        seed=404,
    )
    morse_eng = OpenScienceEngine(morse_top, morse_params)
    morse_rep = morse_eng.run(n_steps=200, report_interval=5)
    morse_drift = float(np.std(morse_rep.total_energy_kj_mol) / max(1e-9, abs(morse_rep.total_energy_kj_mol[0])))

    # 4b. Yukawa screened Coulomb plasma NVE
    plasma_top = build_yukawa_plasma(n_side=3, box_nm=2.0)
    plasma_params = PhysicsParameters(
        integrator="nve_verlet",
        pair_style="yukawa",
        dt_ps=0.001,
        temperature_K=300.0,
        rcut_nm=0.95,
        coulomb_kappa_nm_inv=2.5,
        seed=405,
    )
    plasma_eng = OpenScienceEngine(plasma_top, plasma_params)
    plasma_rep = plasma_eng.run(n_steps=150, report_interval=5)
    plasma_drift = float(np.std(plasma_rep.total_energy_kj_mol) / max(1e-9, abs(plasma_rep.total_energy_kj_mol[0])))
    elapsed = time.perf_counter() - t0

    passed = bool(morse_drift < 1e-3 and plasma_drift < 5e-3)
    return {
        "name": "Multi-Domain Chemistry (Morse) & Physics (Yukawa Plasma) NVE Parity",
        "domain": "Chemistry & Physics",
        "morse_relative_energy_drift": float(f"{morse_drift:.3e}"),
        "yukawa_relative_energy_drift": float(f"{plasma_drift:.3e}"),
        "elapsed_s": round(elapsed, 3),
        "passed": passed,
    }


def bench_5_dcd_xtc_interoperability() -> Dict[str, Any]:
    """Benchmark 5: Standard .dcd, .xtc (3DF compressed), and .pdb binary round-trip audit."""
    t0 = time.perf_counter()
    top = build_lj_argon_fluid(n_side=3, density_g_cm3=1.374)  # 27 atoms (> 9 -> compressed XTC)
    params = PhysicsParameters(integrator="baoab_nvt", pair_style="lj_lfs", dt_ps=0.002, temperature_K=94.4, seed=505)
    eng = OpenScienceEngine(top, params)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        dcd_file = tmp / "argon.dcd"
        xtc_file = tmp / "argon.xtc"
        pdb_file = tmp / "argon.pdb"
        eng.run(n_steps=50, report_interval=5, dcd_path=dcd_file, xtc_path=xtc_file, pdb_path=pdb_file)

        dcd_traj = read_dcd(dcd_file)
        xtc_traj = read_xtc(xtc_file)
        pdb_text = pdb_file.read_text(encoding="utf-8")

        dcd_bytes = dcd_file.stat().st_size
        xtc_bytes = xtc_file.stat().st_size
        max_diff_nm = float(np.max(np.abs(dcd_traj.positions_nm - xtc_traj.positions_nm)))
        box_diff_nm = float(np.max(np.abs(dcd_traj.box_nm - xtc_traj.box_nm)))

    elapsed = time.perf_counter() - t0
    passed = bool(
        dcd_traj.n_frames == 10
        and xtc_traj.n_frames == 10
        and dcd_traj.n_atoms == 27
        and xtc_traj.n_atoms == 27
        and max_diff_nm < 0.001  # XTC precision=1000 -> max quantization error 0.0005 nm
        and box_diff_nm < 1e-5
        and "CRYST1" in pdb_text
    )
    return {
        "name": "Standard .dcd / .xtc / .pdb Binary Trajectory Interoperability",
        "domain": "Open-Science Interoperability",
        "n_frames": dcd_traj.n_frames,
        "n_atoms": dcd_traj.n_atoms,
        "dcd_size_bytes": dcd_bytes,
        "xtc_size_bytes": xtc_bytes,
        "compression_ratio_dcd_vs_xtc": round(dcd_bytes / max(1, xtc_bytes), 2),
        "max_dcd_vs_xtc_coord_diff_nm": float(f"{max_diff_nm:.6f}"),
        "max_box_diff_nm": float(f"{box_diff_nm:.3e}"),
        "elapsed_s": round(elapsed, 3),
        "passed": passed,
    }


def write_markdown_report(results: List[Dict[str, Any]], out_path: Path) -> None:
    all_passed = all(r["passed"] for r in results)
    lines = [
        "# Biology-MM Open Physics Validation Benchmark Report",
        "",
        f"**Overall Status:** {'PASS (5/5 Benchmarks Verified)' if all_passed else 'FAIL'}",
        "**Reproduce:** `PYTHONPATH=\"python:studio/backend\" python3 validation/run_open_validation.py`",
        "",
        "## Summary Table",
        "",
        "| # | Benchmark | Domain | Key Metric | Status |",
        "| :- | :--- | :--- | :--- | :--- |",
    ]
    for idx, r in enumerate(results, 1):
        if idx == 1:
            metric = f"Rel. NVE fluctuation = `{r['relative_energy_fluctuation']}` (`T={r['mean_temperature_K']} K`)"
        elif idx == 2:
            metric = f"`rho={r['mean_density_g_cm3']} g/cm³`, `g_OO` peak at `{r['g_oo_first_peak_ang']} Å` (`h={r['g_oo_first_peak_height']}`)"
        elif idx == 3:
            metric = f"VJP grad err = `{r['vjp_max_grad_error']}`, `{r['metad_hills_deposited']}` hills, `ΔF={r['reconstructed_fes_range_kj_mol']} kJ/mol`"
        elif idx == 4:
            metric = f"Morse drift = `{r['morse_relative_energy_drift']}`, Yukawa drift = `{r['yukawa_relative_energy_drift']}`"
        else:
            metric = f"`{r['n_frames']}` frames, DCD vs XTC max diff = `{r['max_dcd_vs_xtc_coord_diff_nm']} nm` ({r['compression_ratio_dcd_vs_xtc']}x XTC compression)"
        status = "PASS" if r["passed"] else "FAIL"
        lines.append(f"| {idx} | {r['name']} | {r['domain']} | {metric} | **{status}** |")

    lines.extend([
        "",
        "## Standard Workflow Compatibility (No Closed Tools Required)",
        "",
        "- **MDTraj:** `traj = mdtraj.load('argon.xtc', top='argon.pdb')` or `mdtraj.load('argon.dcd', top='argon.pdb')`",
        "- **MDAnalysis:** `u = MDAnalysis.Universe('argon.pdb', 'argon.dcd')`",
        "- **VMD / GROMACS:** Standard CHARMM `CORD` DCD (Å, AKMA) and GROMACS `1995` XDR/3DF XTC (nm, ps).",
        "- **PLUMED-style Enhanced Sampling:** Declarative `DISTANCE`, `ANGLE`, `TORSION`, `GYRATION`, `COORDINATION`, `RESTRAINT`, `MOVINGRESTRAINT`, and `METAD` directives.",
        "",
        "## Raw Benchmark Telemetry",
        "",
        "```json",
        json.dumps(results, indent=2),
        "```",
        "",
    ])
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    results = [
        bench_1_nve_argon(),
        bench_2_tip3p_water_npt(),
        bench_3_alanine_dipeptide_metad(),
        bench_4_chemistry_and_physics_styles(),
        bench_5_dcd_xtc_interoperability(),
    ]
    out_dir = ROOT / "validation"
    (out_dir / "open_validation_results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    write_markdown_report(results, out_dir / "OPEN_VALIDATION_REPORT.md")

    for r in results:
        tag = "PASS" if r["passed"] else "FAIL"
        print(f"[{tag}] {r['name']} ({r['elapsed_s']}s)")
    return 0 if all(r["passed"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
