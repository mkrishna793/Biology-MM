"""Parameter encoding specs and custom potential factories."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = [
    "CustomPotential",
    "ParameterEncoder",
    "ParameterSpec",
    "VALID_ENCODINGS",
]

VALID_ENCODINGS = ("identity", "log", "logit", "angle", "ternary_quantized")


def _sigmoid(z: float) -> float:
    if z >= 0.0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


@dataclass(frozen=True)
class ParameterSpec:
    """Specification for a physical force-field parameter and its unconstrained encoding."""

    name: str
    default: float
    unit: str = ""
    lower: float = -math.inf
    upper: float = math.inf
    encoding: str = "identity"
    trit_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.encoding not in VALID_ENCODINGS:
            raise ValueError(f"Unsupported encoding '{self.encoding}'; expected one of {VALID_ENCODINGS}")
        if self.lower > self.upper:
            raise ValueError(f"Parameter '{self.name}' lower bound {self.lower} > upper bound {self.upper}")
        if not math.isfinite(self.trit_scale) or self.trit_scale <= 0.0:
            raise ValueError(f"Parameter '{self.name}' trit_scale must be positive and finite")
        if self.encoding == "logit" and not (math.isfinite(self.lower) and math.isfinite(self.upper) and self.upper > self.lower):
            raise ValueError(f"Parameter '{self.name}' with 'logit' encoding requires finite lower < upper")
        if not math.isfinite(self.default):
            raise ValueError(f"Parameter '{self.name}' default must be finite")
        if self.default < self.lower or self.default > self.upper:
            raise ValueError(f"Parameter '{self.name}' default {self.default} outside [{self.lower}, {self.upper}]")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "default": float(self.default),
            "unit": self.unit,
            "lower": float(self.lower) if math.isfinite(self.lower) else None,
            "upper": float(self.upper) if math.isfinite(self.upper) else None,
            "encoding": self.encoding,
            "trit_scale": float(self.trit_scale),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ParameterSpec":
        lo = data.get("lower")
        hi = data.get("upper")
        return cls(
            name=str(data["name"]),
            default=float(data["default"]),
            unit=str(data.get("unit", "")),
            lower=-math.inf if lo is None else float(lo),
            upper=math.inf if hi is None else float(hi),
            encoding=str(data.get("encoding", "identity")),
            trit_scale=float(data.get("trit_scale", 1.0)),
        )


class ParameterEncoder:
    """
    Bidirectional parameter transform between constrained physical space `theta`
    and unconstrained optimization space `z`, with exact diagonal Jacobian `d(theta)/d(z)`.
    """

    def __init__(self, specs: Sequence[ParameterSpec]) -> None:
        self.specs: Tuple[ParameterSpec, ...] = tuple(specs)
        names = [s.name for s in self.specs]
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate parameter names in ParameterEncoder: {names}")

    def __len__(self) -> int:
        return len(self.specs)

    @property
    def defaults(self) -> np.ndarray:
        return np.array([s.default for s in self.specs], dtype=np.float64)

    def validate(self, theta: Sequence[float]) -> np.ndarray:
        arr = np.asarray(theta, dtype=np.float64)
        if arr.shape != (len(self.specs),):
            raise ValueError(f"Expected parameter vector of shape ({len(self.specs)},), got {arr.shape}")
        if not np.all(np.isfinite(arr)):
            raise ValueError("Parameter vector contains non-finite values")
        for spec, val in zip(self.specs, arr):
            if val < spec.lower or val > spec.upper:
                raise ValueError(
                    f"Parameter '{spec.name}' value {val} violates bounds [{spec.lower}, {spec.upper}]"
                )
        return arr

    def encode(self, theta: Sequence[float]) -> np.ndarray:
        arr = self.validate(theta)
        out = np.empty(len(self.specs), dtype=np.float64)
        for idx, (spec, val) in enumerate(zip(self.specs, arr)):
            x = float(val)
            if spec.encoding == "identity":
                out[idx] = x
            elif spec.encoding == "log":
                shift = spec.lower if math.isfinite(spec.lower) else 0.0
                diff = x - shift
                if diff <= 0.0:
                    raise ValueError(f"Parameter '{spec.name}' value {x} must exceed {shift} for log encoding")
                out[idx] = math.log(diff)
            elif spec.encoding == "logit":
                span = spec.upper - spec.lower
                u = (x - spec.lower) / span
                if u <= 0.0 or u >= 1.0:
                    raise ValueError(f"Parameter '{spec.name}' value {x} must lie strictly inside ({spec.lower}, {spec.upper}) for logit encoding")
                out[idx] = math.log(u / (1.0 - u))
            elif spec.encoding == "angle":
                out[idx] = math.atan2(math.sin(x), math.cos(x))
            elif spec.encoding == "ternary_quantized":
                out[idx] = float(np.clip(np.rint(x / spec.trit_scale), -1.0, 1.0))
        return out

    def decode(self, z: Sequence[float]) -> np.ndarray:
        arr = np.asarray(z, dtype=np.float64)
        if arr.shape != (len(self.specs),):
            raise ValueError(f"Expected encoded vector of shape ({len(self.specs)},), got {arr.shape}")
        if not np.all(np.isfinite(arr)):
            raise ValueError("Encoded vector contains non-finite values")

        out = np.empty(len(self.specs), dtype=np.float64)
        for idx, (spec, val) in enumerate(zip(self.specs, arr)):
            zv = float(val)
            if spec.encoding == "identity":
                out[idx] = zv
            elif spec.encoding == "log":
                shift = spec.lower if math.isfinite(spec.lower) else 0.0
                out[idx] = shift + math.exp(zv)
            elif spec.encoding == "logit":
                s = _sigmoid(zv)
                out[idx] = spec.lower + (spec.upper - spec.lower) * s
            elif spec.encoding == "angle":
                out[idx] = math.atan2(math.sin(zv), math.cos(zv))
            elif spec.encoding == "ternary_quantized":
                trit = float(np.clip(np.rint(zv), -1.0, 1.0))
                out[idx] = trit * spec.trit_scale
        return out

    def jacobian_diag(self, z: Sequence[float]) -> np.ndarray:
        """
        Return diagonal `d(decode(z)) / d(z)` (using Straight-Through Estimator `trit_scale`
        for `ternary_quantized`).
        """
        arr = np.asarray(z, dtype=np.float64)
        if arr.shape != (len(self.specs),):
            raise ValueError(f"Expected encoded vector of shape ({len(self.specs)},), got {arr.shape}")

        jac = np.empty(len(self.specs), dtype=np.float64)
        for idx, (spec, val) in enumerate(zip(self.specs, arr)):
            zv = float(val)
            if spec.encoding == "identity":
                jac[idx] = 1.0
            elif spec.encoding == "log":
                jac[idx] = math.exp(zv)
            elif spec.encoding == "logit":
                s = _sigmoid(zv)
                jac[idx] = (spec.upper - spec.lower) * s * (1.0 - s)
            elif spec.encoding == "angle":
                jac[idx] = 1.0
            elif spec.encoding == "ternary_quantized":
                jac[idx] = spec.trit_scale
        return jac

    def to_dict(self) -> Dict[str, Any]:
        return {"specs": [s.to_dict() for s in self.specs]}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ParameterEncoder":
        return cls([ParameterSpec.from_dict(item) for item in data["specs"]])


@dataclass
class CustomPotential:
    """
    Declarative potential factor specification.

    All atom indices stored in Python (`i`, `j`, `pairs`, `atoms`, `atom_indices`) use
    **0-based** indexing and are converted to **1-based** indices when serialized via
    `to_spec_dict(atom_index_base=1)` for the Julia CLI (`bin/md.jl`).
    """

    kind: str
    params: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def harmonic_bond(cls, i: int, j: int, k: float, r0: float) -> "CustomPotential":
        return cls("harmonic_bond", {"i": int(i), "j": int(j), "k": float(k), "r0": float(r0)})

    @classmethod
    def lj_pair(
        cls,
        i: int,
        j: int,
        epsilon: float,
        sigma: float,
        rc: float,
        mode: str = "lfs",
    ) -> "CustomPotential":
        if mode not in ("none", "trunc", "shift", "lfs"):
            raise ValueError(f"Unsupported LJ mode '{mode}'")
        return cls(
            "lj_pair",
            {
                "i": int(i),
                "j": int(j),
                "epsilon": float(epsilon),
                "sigma": float(sigma),
                "rc": float(rc),
                "mode": str(mode),
            },
        )

    @classmethod
    def morse_bond(cls, i: int, j: int, De: float, a: float, r0: float) -> "CustomPotential":
        return cls("morse_bond", {"i": int(i), "j": int(j), "De": float(De), "a": float(a), "r0": float(r0)})

    @classmethod
    def tersoff(
        cls,
        element: str = "C",
        units: str = "md",
        atoms: Optional[Sequence[int]] = None,
    ) -> "CustomPotential":
        params: Dict[str, Any] = {"element": str(element), "units": str(units)}
        if atoms is not None:
            params["atoms"] = [int(a) for a in atoms]
        return cls("tersoff", params)

    @classmethod
    def stillinger_weber(
        cls,
        element: str = "Si",
        units: str = "md",
        atoms: Optional[Sequence[int]] = None,
    ) -> "CustomPotential":
        params: Dict[str, Any] = {"element": str(element), "units": str(units)}
        if atoms is not None:
            params["atoms"] = [int(a) for a in atoms]
        return cls("stillinger_weber", params)

    @classmethod
    def custom_pair(
        cls,
        kind: str,
        pairs: Sequence[Tuple[int, int]],
        params: Dict[str, Any],
        rc: float,
        mode: str = "lfs",
    ) -> "CustomPotential":
        return cls(
            "custom_pair",
            {
                "kernel": str(kind),
                "pairs": [(int(i), int(j)) for i, j in pairs],
                "params": dict(params),
                "rc": float(rc),
                "mode": str(mode),
            },
        )

    @classmethod
    def from_callable(
        cls,
        fn: Callable[[float], float],
        r_min: float,
        r_max: float,
        pairs: Sequence[Tuple[int, int]],
        n_grid: int = 256,
        rc: Optional[float] = None,
        mode: str = "lfs",
    ) -> "CustomPotential":
        """Sample any scalar Python potential `U(r)` onto a uniform grid for cubic spline evaluation."""
        r0 = float(r_min)
        r1 = float(r_max)
        ng = int(n_grid)
        if not (0.0 < r0 < r1):
            raise ValueError(f"Expected 0 < r_min < r_max, got ({r0}, {r1})")
        if ng < 4:
            raise ValueError(f"n_grid must be >= 4, got {ng}")
        r_grid = np.linspace(r0, r1, ng, dtype=np.float64)
        u_vals: List[float] = []
        for rv in r_grid:
            uv = float(fn(float(rv)))
            if not math.isfinite(uv):
                raise ValueError(f"Potential callable returned non-finite value {uv} at r={rv}")
            u_vals.append(uv)
        cutoff = r1 if rc is None else float(rc)
        return cls(
            "tabulated_spline",
            {
                "pairs": [(int(i), int(j)) for i, j in pairs],
                "r_grid": [float(v) for v in r_grid],
                "u_grid": u_vals,
                "rc": cutoff,
                "mode": str(mode),
            },
        )

    @classmethod
    def external_efield(
        cls,
        E: Sequence[float],
        charges: Sequence[float],
        atom_indices: Optional[Sequence[int]] = None,
    ) -> "CustomPotential":
        e_vec = [float(v) for v in E]
        if len(e_vec) != 3:
            raise ValueError("E field must be a 3-component vector")
        params: Dict[str, Any] = {
            "E": e_vec,
            "charges": [float(c) for c in charges],
        }
        if atom_indices is not None:
            params["atom_indices"] = [int(a) for a in atom_indices]
        return cls("external_efield", params)

    @classmethod
    def wall_restraint(
        cls,
        kind: str,
        axis: int,
        lo: float,
        hi: float,
        k: Optional[float] = None,
        epsilon: Optional[float] = None,
        sigma: Optional[float] = None,
        rc: Optional[float] = None,
    ) -> "CustomPotential":
        if axis not in (0, 1, 2):
            raise ValueError(f"axis must be 0, 1, or 2; got {axis}")
        params: Dict[str, Any] = {
            "kind": str(kind),
            "axis": int(axis),
            "lo": float(lo),
            "hi": float(hi),
        }
        if k is not None:
            params["k"] = float(k)
        if epsilon is not None:
            params["epsilon"] = float(epsilon)
        if sigma is not None:
            params["sigma"] = float(sigma)
        if rc is not None:
            params["rc"] = float(rc)
        return cls("wall_restraint", params)

    @classmethod
    def umbrella(cls, i: int, j: int, k: float, r0: float) -> "CustomPotential":
        return cls("umbrella", {"i": int(i), "j": int(j), "k": float(k), "r0": float(r0)})

    def to_spec_dict(self, atom_index_base: int = 1) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dictionary with optional 1-based Julia atom indexing."""
        shift = int(atom_index_base)
        d: Dict[str, Any] = {"type": self.kind}
        for key, val in self.params.items():
            if key in ("i", "j"):
                d[key] = int(val) + shift
            elif key in ("atoms", "atom_indices"):
                d[key] = [int(a) + shift for a in val]
            elif key == "pairs":
                d[key] = [[int(p[0]) + shift, int(p[1]) + shift] for p in val]
            elif key == "axis" and self.kind == "wall_restraint":
                d[key] = int(val) + shift
            else:
                d[key] = val
        return d
