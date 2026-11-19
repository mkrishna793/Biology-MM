"""Simulation configuration, Julia CLI orchestration, and native GPU evaluator."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .builders import AtomicSystem
from .native import GPUVerletPlan, get_bridge
from .parameters import CustomPotential

__all__ = [
    "Simulation",
    "SimulationConfig",
    "SimulationResult",
    "discover_julia_toolchain",
    "parse_extxyz",
]


def discover_julia_toolchain() -> Dict[str, Any]:
    """Locate the Julia binary, depot path, `julia` project, and `bin/md.jl`."""
    here = Path(__file__).resolve()
    env_root = os.environ.get("BIOLOGY_MM_ROOT")
    if env_root:
        julia_project = Path(env_root).expanduser() / "julia"
    else:
        julia_project = here.parents[2] / "julia"

    md_cli = julia_project / "bin" / "md.jl"

    julia_bin: Optional[str] = os.environ.get("BIOLOGY_MM_JULIA") or os.environ.get("JULIA")
    if not julia_bin:
        julia_bin = shutil.which("julia")
    if not julia_bin:
        cached = Path.home() / ".cache" / "biology-mm" / "julia-1.13.3" / "bin" / "julia"
        if cached.is_file():
            julia_bin = str(cached)

    depot_path: Optional[str] = os.environ.get("JULIA_DEPOT_PATH") or os.environ.get("BIOLOGY_MM_JULIA_DEPOT")
    if not depot_path:
        cached_depot = Path.home() / ".cache" / "biology-mm" / "julia-depot"
        if cached_depot.is_dir():
            depot_path = str(cached_depot)

    return {
        "julia_bin": julia_bin,
        "julia_depot_path": depot_path,
        "julia_project": str(julia_project),
        "md_cli": str(md_cli),
        "available": bool(julia_bin and md_cli.is_file()),
    }


def parse_extxyz(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Parse an extended XYZ trajectory file written by `TrajectoryFormat.write_extxyz_frame`."""
    p = Path(path)
    frames: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as fh:
        while True:
            header_n = fh.readline()
            if not header_n:
                break
            stripped = header_n.strip()
            if not stripped:
                continue
            n_atoms = int(stripped)
            comment = fh.readline().strip()

            time_m = re.search(r"(?:^|\s)Time=([^\s]+)", comment)
            time_val = float(time_m.group(1)) if time_m else 0.0

            lat_m = re.search(r'Lattice="([^"]+)"', comment)
            box_mat = None
            if lat_m:
                vals = [float(x) for x in lat_m.group(1).split()]
                if len(vals) == 9:
                    # Julia writes vec(box_matrix) in column-major order
                    box_mat = np.array(vals, dtype=np.float64).reshape((3, 3)).T.copy()

            species: List[str] = []
            positions = np.empty((n_atoms, 3), dtype=np.float64)
            for idx in range(n_atoms):
                parts = fh.readline().split()
                if len(parts) < 4:
                    raise ValueError(f"Malformed XYZ atom line in {p}: {parts}")
                species.append(parts[0])
                positions[idx, 0] = float(parts[1])
                positions[idx, 1] = float(parts[2])
                positions[idx, 2] = float(parts[3])

            frames.append(
                {
                    "species": species,
                    "positions": positions,
                    "time": time_val,
                    "box_matrix": box_mat,
                }
            )
    return frames


@dataclass
class SimulationConfig:
    """Configuration for a Biology MM workflow run (`bin/md.jl`)."""

    ensemble: str = "nve"
    integrator: str = "velocity_verlet"
    dt: float = 0.002
    temperature: float = 300.0
    gamma: float = 1.0
    equilibrate_steps: int = 10
    produce_steps: int = 100
    frame_every: int = 1
    checkpoint_every: int = 50
    minimize_max_steps: int = 50
    backend: str = "auto"
    run_id: str = "py-run"
    seed: int = 1
    contact_tol: float = 0.05
    analysis: Optional[Dict[str, Any]] = None


@dataclass
class SimulationResult:
    """Outputs and parsed trajectory frames from a completed `Simulation.run()`."""

    run_id: str
    workdir: Path
    trajectory_path: Path
    checkpoint_path: Optional[Path]
    n_frames: int
    diffusion_coefficient: float
    stages: List[str]
    frames: List[Dict[str, Any]] = field(default_factory=list)
    log_records: List[Dict[str, Any]] = field(default_factory=list)
    raw_output: Dict[str, Any] = field(default_factory=dict)


class Simulation:
    """High-level simulation driver bridging Python `AtomicSystem` + `CustomPotential` to Julia & Metal GPU."""

    def __init__(
        self,
        system: AtomicSystem,
        potentials: Optional[Sequence[CustomPotential]] = None,
        config: Optional[SimulationConfig] = None,
    ) -> None:
        self.system = system
        self.potentials: List[CustomPotential] = (
            list(potentials) if potentials is not None else list(system.potentials)
        )
        self.config = config or SimulationConfig()

    def to_spec_dict(self, workdir: Union[str, Path]) -> Dict[str, Any]:
        """Build a `schema_version=1` JSON specification dictionary for `bin/md.jl`."""
        cfg = self.config
        if not self.potentials:
            raise ValueError("Simulation requires at least one CustomPotential factor")

        n_frames = max(1, cfg.produce_steps // max(1, cfg.frame_every))
        analysis = dict(cfg.analysis) if cfg.analysis is not None else {}
        if "min_frames" not in analysis:
            analysis["min_frames"] = min(20, max(4, n_frames))
        if "min_fit_points" not in analysis:
            analysis["min_fit_points"] = min(5, max(2, n_frames // 5))
        if "fit_window" not in analysis and "lag_window" not in analysis:
            analysis["fit_window"] = [0.10, 0.30] if n_frames >= 25 else [0.10, 0.80]
        if "min_ess" not in analysis:
            analysis["min_ess"] = 1.0

        workflow: Dict[str, Any] = {
            "run_id": cfg.run_id,
            "dt": float(cfg.dt),
            "seed": int(cfg.seed),
            "temperature": float(cfg.temperature),
            "minimize_max_steps": int(cfg.minimize_max_steps),
            "equilibrate_steps": int(cfg.equilibrate_steps),
            "produce_steps": int(cfg.produce_steps),
            "frame_every": int(cfg.frame_every),
            "checkpoint_every": int(cfg.checkpoint_every),
            "contact_tol": float(cfg.contact_tol),
        }
        if cfg.integrator != "respa_nve":
            workflow["gamma"] = float(cfg.gamma)

        spec: Dict[str, Any] = {
            "schema_version": 1,
            "coordinates": self.system.positions.tolist(),
            "masses": self.system.masses.tolist(),
            "box": list(self.system.box_lengths),
            "factors": [p.to_spec_dict(atom_index_base=1) for p in self.potentials],
            "ensemble": cfg.ensemble,
            "integrator": cfg.integrator,
            "workdir": str(Path(workdir)),
            "workflow": workflow,
            "analysis": analysis,
        }
        if self.system.units != "reduced":
            spec["units"] = self.system.units
        return spec

    def run(self, workdir: Optional[Union[str, Path]] = None) -> SimulationResult:
        """Execute the full simulation workflow via `julia/bin/md.jl`."""
        tc = discover_julia_toolchain()
        if not tc["available"]:
            raise RuntimeError(
                f"Julia toolchain not found (julia_bin={tc['julia_bin']}, md_cli={tc['md_cli']})"
            )

        own_tmp = workdir is None
        run_root = Path(tempfile.mkdtemp(prefix="biology_mm_run_")) if own_tmp else Path(workdir).resolve()
        run_root.mkdir(parents=True, exist_ok=True)
        out_dir = run_root / "run" if own_tmp else run_root
        spec_path = run_root / "spec.json"

        spec = self.to_spec_dict(out_dir)
        spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")

        env = os.environ.copy()
        if tc["julia_depot_path"]:
            env["JULIA_DEPOT_PATH"] = tc["julia_depot_path"]

        cmd = [
            tc["julia_bin"],
            f"--project={tc['julia_project']}",
            tc["md_cli"],
            "--backend",
            self.config.backend,
            str(spec_path),
        ]
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(
                f"md.jl exited with code {proc.returncode}:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
            )

        payload = json.loads(proc.stdout.strip().splitlines()[-1])
        traj_path = Path(payload["trajectory"])
        ckpt_raw = payload.get("checkpoint")
        ckpt_path = Path(ckpt_raw) if ckpt_raw else None

        frames = parse_extxyz(traj_path) if traj_path.is_file() else []
        jsonl_path = traj_path.with_suffix(".jsonl")
        log_records: List[Dict[str, Any]] = []
        if jsonl_path.is_file():
            for line in jsonl_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    log_records.append(json.loads(line))

        return SimulationResult(
            run_id=str(payload["run_id"]),
            workdir=out_dir,
            trajectory_path=traj_path,
            checkpoint_path=ckpt_path,
            n_frames=int(payload["frames"]),
            diffusion_coefficient=float(payload["diffusion_coefficient"]),
            stages=[str(s) for s in payload["stages"]],
            frames=frames,
            log_records=log_records,
            raw_output=payload,
        )

    def evaluate_lj_dsf_native(
        self,
        sigma: Union[float, Sequence[float], np.ndarray],
        epsilon: Union[float, Sequence[float], np.ndarray],
        rcut: float = 1.2,
        skin: float = 0.2,
        alpha_dsf: float = 0.0,
        coulomb_prefactor: float = 138.935456,
        use_lfs: bool = True,
        include_self: bool = True,
        exclusions: Optional[Sequence[Sequence[int]]] = None,
        scaled_14: Optional[Sequence[Sequence[float]]] = None,
        lj14_scale: float = 0.5,
        coul14_scale: float = 1.0 / 1.2,
        backend: Optional[str] = None,
        compute_virial: bool = True,
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """Evaluate LJ + DSF energy, forces `(N, 3)`, and virial `(3, 3)` directly on native Metal/CPU."""
        n = self.system.n_atoms
        sig_arr = np.full(n, float(sigma), dtype=np.float64) if np.ndim(sigma) == 0 else np.asarray(sigma, dtype=np.float64)
        eps_arr = np.full(n, float(epsilon), dtype=np.float64) if np.ndim(epsilon) == 0 else np.asarray(epsilon, dtype=np.float64)
        excl = self.system.bonds if exclusions is None and self.system.bonds else exclusions

        with GPUVerletPlan(
            positions=self.system.positions,
            sigma=sig_arr,
            epsilon=eps_arr,
            charges=self.system.charges,
            box=self.system.box_lengths,
            exclusions=excl,
            scaled_14=scaled_14,
            lj14_scale=lj14_scale,
            coul14_scale=coul14_scale,
            rcut=rcut,
            skin=skin,
            alpha_dsf=alpha_dsf,
            coulomb_prefactor=coulomb_prefactor,
            use_lfs=use_lfs,
            include_self=include_self,
            backend=backend or self.config.backend,
            bridge=get_bridge(),
        ) as plan:
            return plan.eval(compute_virial=compute_virial)
