"""Native ctypes bridge to Trium-XBT (CPU/NEON) and TriumGPU (Apple Metal) libraries."""

from __future__ import annotations

import ctypes
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = [
    "CSAResult",
    "GPUProfile",
    "GPUVerletPlan",
    "NativeBridge",
    "csa_accumulate",
    "erfc_fast",
    "exp_radix3",
    "get_bridge",
    "gpu_msm_stencil27",
    "gpu_ternary_matvec",
    "mm_abi_version",
    "mm_version",
    "probe_hardware",
    "rsqrt_fast",
    "run_batched_gpu_steps",
    "tanh_radix3",
]


def _dlext() -> str:
    if sys.platform == "darwin":
        return "dylib"
    if sys.platform.startswith("win"):
        return "dll"
    return "so"


def _candidate_lib_dirs() -> list[Path]:
    dirs: list[Path] = []
    env_lib = os.environ.get("BIOLOGY_MM_LIB_DIR")
    if env_lib:
        dirs.append(Path(env_lib).expanduser())
    env_root = os.environ.get("BIOLOGY_MM_ROOT")
    if env_root:
        root_p = Path(env_root).expanduser()
        dirs.append(root_p / "julia" / "lib")
        dirs.append(root_p / "native" / "build")

    # .../python/biology_mm/native.py -> .../julia/lib and .../native/build
    here = Path(__file__).resolve()
    repo_root = here.parents[2]
    dirs.append(repo_root / "julia" / "lib")
    dirs.append(repo_root / "native" / "build")
    return dirs


def _discover_library(basename: str) -> Optional[Path]:
    ext = _dlext()
    stem = basename[3:] if basename.startswith("lib") else basename
    filenames = (f"lib{stem}.{ext}", f"{stem}.{ext}")
    for d in _candidate_lib_dirs():
        for filename in filenames:
            cand = d / filename
            if cand.is_file():
                return cand
    return None


@dataclass(frozen=True)
class GPUProfile:
    """Hardware capability telemetry from the active GPU or CPU backend."""

    has_gpu: bool
    backend: str
    device_name: str
    unified_memory: bool
    max_threads_per_tg: int
    simd_width: int
    vram_bytes: int


@dataclass(frozen=True)
class CSAResult:
    """Result of balanced ternary carry-save accumulation over int64 terms."""

    resolved: int
    s_p: int
    s_n: int
    c_p: int
    c_n: int
    overflow: bool


def _as_box_matrices(
    box: Optional[Union[Sequence[float], np.ndarray]]
) -> Tuple[int, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Convert a box specification to column-major 3x3 double arrays matching Julia/C++/Metal ABI.

    Note on ABI:
    Both `trium_mm_bridge.cpp` and `trium_gpu_metal.mm` index `H` and `Hinv` as column-major
    3x3 matrices where columns are lattice vectors (`H[0..2]` = a1, `H[3..5]` = a2, `H[6..8]` = a3).
    """
    if box is None:
        return 0, None, None

    arr = np.asarray(box, dtype=np.float64)
    if arr.shape == (3,):
        lx, ly, lz = float(arr[0]), float(arr[1]), float(arr[2])
        if not (math.isfinite(lx) and math.isfinite(ly) and math.isfinite(lz)) or lx <= 0 or ly <= 0 or lz <= 0:
            raise ValueError(f"Orthorhombic box lengths must be positive and finite, got {arr}")
        H = np.diag([lx, ly, lz]).astype(np.float64)
        H_col = np.ascontiguousarray(H.T, dtype=np.float64)
        return 0, H_col, None

    if arr.shape == (3, 3):
        if not np.all(np.isfinite(arr)):
            raise ValueError("Triclinic box matrix H contains non-finite values")
        det = float(np.linalg.det(arr))
        if not math.isfinite(det) or abs(det) < 1e-15:
            raise ValueError("Triclinic box matrix H is singular")
        # Check if diagonal orthorhombic was passed as 3x3
        Hinv = np.linalg.inv(arr)
        H_col = np.ascontiguousarray(arr.T, dtype=np.float64)
        Hinv_col = np.ascontiguousarray(Hinv.T, dtype=np.float64)
        return 1, H_col, Hinv_col

    raise ValueError(f"box must be None, (3,) lengths, or (3, 3) matrix; got shape {arr.shape}")


def _encode_pair_keys(pairs: Optional[Sequence[Sequence[int]]], n_atoms: int, with_weights: bool = False):
    """Encode 0-based atom index pairs into sorted 1-based uint64 keys `(lo1 << 32) | hi1`."""
    if not pairs:
        if with_weights:
            return np.empty(0, dtype=np.uint64), np.empty(0, dtype=np.float64)
        return np.empty(0, dtype=np.uint64)

    entries = []
    for item in pairs:
        if with_weights:
            if len(item) == 2:
                i0, j0 = int(item[0]), int(item[1])
                w = 1.0
            elif len(item) == 3:
                i0, j0, w = int(item[0]), int(item[1]), float(item[2])
            else:
                raise ValueError(f"scaled_14 entry must have length 2 or 3, got {item}")
        else:
            if len(item) != 2:
                raise ValueError(f"exclusion entry must have length 2, got {item}")
            i0, j0 = int(item[0]), int(item[1])
            w = 0.0

        if not (0 <= i0 < n_atoms and 0 <= j0 < n_atoms) or i0 == j0:
            raise ValueError(f"Invalid pair ({i0}, {j0}) for n_atoms={n_atoms}")
        lo1 = min(i0, j0) + 1
        hi1 = max(i0, j0) + 1
        key = (lo1 << 32) | hi1
        entries.append((key, w))

    entries.sort(key=lambda t: t[0])
    keys = np.array([e[0] for e in entries], dtype=np.uint64)
    if with_weights:
        weights = np.array([e[1] for e in entries], dtype=np.float64)
        return keys, weights
    return keys


class NativeBridge:
    """Wrapper holding ctypes bindings to `libtrium_mm_bridge` and `libtrium_gpu_bridge`."""

    def __init__(
        self,
        mm_lib_path: Optional[Union[str, Path]] = None,
        gpu_lib_path: Optional[Union[str, Path]] = None,
    ) -> None:
        disable_gpu = os.environ.get("BIOLOGY_MM_DISABLE_GPU", "").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        self.mm_lib_path = Path(mm_lib_path) if mm_lib_path else _discover_library("libtrium_mm_bridge")
        if disable_gpu and gpu_lib_path is None:
            self.gpu_lib_path = None
        else:
            self.gpu_lib_path = Path(gpu_lib_path) if gpu_lib_path else _discover_library("libtrium_gpu_bridge")

        self.mm: Optional[ctypes.CDLL] = None
        self.gpu: Optional[ctypes.CDLL] = None

        if self.mm_lib_path and self.mm_lib_path.is_file():
            self.mm = ctypes.CDLL(str(self.mm_lib_path))
            self._bind_mm_symbols()

        if self.gpu_lib_path and self.gpu_lib_path.is_file():
            self.gpu = ctypes.CDLL(str(self.gpu_lib_path))
            self._bind_gpu_symbols()

    @property
    def has_cpu_bridge(self) -> bool:
        return self.mm is not None

    @property
    def has_gpu_bridge(self) -> bool:
        return self.gpu is not None

    def _bind_mm_symbols(self) -> None:
        lib = self.mm
        assert lib is not None

        lib.xbt_mm_version.argtypes = []
        lib.xbt_mm_version.restype = ctypes.c_int64

        lib.xbt_mm_abi_version.argtypes = []
        lib.xbt_mm_abi_version.restype = ctypes.c_int64

        for fn_name in ("xbt_exp_radix3", "xbt_tanh_radix3", "xbt_rsqrt_fast", "xbt_erfc_fast"):
            fn = getattr(lib, fn_name)
            fn.argtypes = [ctypes.c_double]
            fn.restype = ctypes.c_double

        lib.xbt_csa_accumulate_array.argtypes = [
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_int64),
        ]
        lib.xbt_csa_accumulate_array.restype = ctypes.c_int

        lib.xbt_symplectic_kdk_step_int64.argtypes = [
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        lib.xbt_symplectic_kdk_step_int64.restype = ctypes.c_int

        lib.xbt_ternary_matvec_dualrail.argtypes = [
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_double,
            ctypes.POINTER(ctypes.c_double),
        ]
        lib.xbt_ternary_matvec_dualrail.restype = None

        lib.xbt_verlet_plan_create.argtypes = [
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_int,
        ]
        lib.xbt_verlet_plan_create.restype = ctypes.c_void_p

        lib.xbt_verlet_plan_eval_mt.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_double),
        ]
        lib.xbt_verlet_plan_eval_mt.restype = ctypes.c_double

        lib.xbt_verlet_plan_destroy.argtypes = [ctypes.c_void_p]
        lib.xbt_verlet_plan_destroy.restype = None

        lib.xbt_verlet_plan_num_pairs.argtypes = [ctypes.c_void_p]
        lib.xbt_verlet_plan_num_pairs.restype = ctypes.c_size_t

    def _bind_gpu_symbols(self) -> None:
        lib = self.gpu
        assert lib is not None

        lib.xbt_gpu_probe_device.argtypes = [
            ctypes.c_char_p,
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_uint64),
        ]
        lib.xbt_gpu_probe_device.restype = ctypes.c_int32

        lib.xbt_gpu_verlet_plan_create.argtypes = [
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_int,
        ]
        lib.xbt_gpu_verlet_plan_create.restype = ctypes.c_void_p

        lib.xbt_gpu_verlet_plan_eval.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
        ]
        lib.xbt_gpu_verlet_plan_eval.restype = ctypes.c_double

        lib.xbt_gpu_verlet_plan_destroy.argtypes = [ctypes.c_void_p]
        lib.xbt_gpu_verlet_plan_destroy.restype = None

        lib.xbt_gpu_verlet_plan_num_pairs.argtypes = [ctypes.c_void_p]
        lib.xbt_gpu_verlet_plan_num_pairs.restype = ctypes.c_size_t

        lib.xbt_gpu_ternary_matvec_dualrail.argtypes = [
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_double,
            ctypes.POINTER(ctypes.c_double),
        ]
        lib.xbt_gpu_ternary_matvec_dualrail.restype = None

        lib.xbt_gpu_msm_stencil27.argtypes = [
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_double,
            ctypes.POINTER(ctypes.c_double),
        ]
        lib.xbt_gpu_msm_stencil27.restype = ctypes.c_double

        lib.xbt_gpu_md_batch_run.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_double,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
        ]
        lib.xbt_gpu_md_batch_run.restype = ctypes.c_double

    def probe_hardware(self) -> GPUProfile:
        """Probe active Apple Metal / GPU hardware or report CPU fallback."""
        if self.gpu is not None:
            name_buf = ctypes.create_string_buffer(256)
            backend_id = ctypes.c_int32(0)
            unified = ctypes.c_int32(0)
            max_tg = ctypes.c_int32(0)
            simd_w = ctypes.c_int32(0)
            vram = ctypes.c_uint64(0)
            ok = self.gpu.xbt_gpu_probe_device(
                name_buf,
                ctypes.c_int32(256),
                ctypes.byref(backend_id),
                ctypes.byref(unified),
                ctypes.byref(max_tg),
                ctypes.byref(simd_w),
                ctypes.byref(vram),
            )
            if ok != 0:
                backend_map = {1: "metal", 2: "vulkan", 3: "sycl"}
                return GPUProfile(
                    has_gpu=True,
                    backend=backend_map.get(int(backend_id.value), "metal"),
                    device_name=name_buf.value.decode("utf-8", errors="replace") or "Apple Metal GPU",
                    unified_memory=bool(unified.value != 0),
                    max_threads_per_tg=max(1, int(max_tg.value)),
                    simd_width=max(1, int(simd_w.value)),
                    vram_bytes=int(vram.value),
                )

        cpus = os.cpu_count() or 1
        return GPUProfile(
            has_gpu=False,
            backend="cpu",
            device_name=f"CPU ({cpus} cores)",
            unified_memory=True,
            max_threads_per_tg=cpus,
            simd_width=1,
            vram_bytes=0,
        )

    def mm_version(self) -> int:
        if self.mm is None:
            raise RuntimeError("libtrium_mm_bridge not loaded")
        return int(self.mm.xbt_mm_version())

    def mm_abi_version(self) -> int:
        if self.mm is None:
            raise RuntimeError("libtrium_mm_bridge not loaded")
        return int(self.mm.xbt_mm_abi_version())

    def exp_radix3(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        if self.mm is None:
            return np.exp(x) if isinstance(x, np.ndarray) else math.exp(float(x))
        if np.ndim(x) == 0:
            return float(self.mm.xbt_exp_radix3(ctypes.c_double(float(x))))
        arr = np.asarray(x, dtype=np.float64)
        out = np.empty_like(arr)
        fn = self.mm.xbt_exp_radix3
        for idx, val in np.ndenumerate(arr):
            out[idx] = fn(ctypes.c_double(float(val)))
        return out

    def tanh_radix3(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        if self.mm is None:
            return np.tanh(x) if isinstance(x, np.ndarray) else math.tanh(float(x))
        if np.ndim(x) == 0:
            return float(self.mm.xbt_tanh_radix3(ctypes.c_double(float(x))))
        arr = np.asarray(x, dtype=np.float64)
        out = np.empty_like(arr)
        fn = self.mm.xbt_tanh_radix3
        for idx, val in np.ndenumerate(arr):
            out[idx] = fn(ctypes.c_double(float(val)))
        return out

    def rsqrt_fast(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        if self.mm is None:
            return 1.0 / np.sqrt(x) if isinstance(x, np.ndarray) else 1.0 / math.sqrt(float(x))
        if np.ndim(x) == 0:
            return float(self.mm.xbt_rsqrt_fast(ctypes.c_double(float(x))))
        arr = np.asarray(x, dtype=np.float64)
        out = np.empty_like(arr)
        fn = self.mm.xbt_rsqrt_fast
        for idx, val in np.ndenumerate(arr):
            out[idx] = fn(ctypes.c_double(float(val)))
        return out

    def erfc_fast(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        if self.mm is None:
            if isinstance(x, np.ndarray):
                return np.vectorize(math.erfc)(x)
            return math.erfc(float(x))
        if np.ndim(x) == 0:
            return float(self.mm.xbt_erfc_fast(ctypes.c_double(float(x))))
        arr = np.asarray(x, dtype=np.float64)
        out = np.empty_like(arr)
        fn = self.mm.xbt_erfc_fast
        for idx, val in np.ndenumerate(arr):
            out[idx] = fn(ctypes.c_double(float(val)))
        return out

    def csa_accumulate(self, terms: Iterable[int]) -> CSAResult:
        if self.mm is None:
            raise RuntimeError("libtrium_mm_bridge not loaded")
        arr = np.ascontiguousarray(list(terms), dtype=np.int64)
        s_p = ctypes.c_uint64(0)
        s_n = ctypes.c_uint64(0)
        c_p = ctypes.c_uint64(0)
        c_n = ctypes.c_uint64(0)
        resolved = ctypes.c_int64(0)
        ptr = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)) if arr.size > 0 else ctypes.POINTER(ctypes.c_int64)()
        ovf = self.mm.xbt_csa_accumulate_array(
            ptr,
            ctypes.c_size_t(arr.size),
            ctypes.byref(s_p),
            ctypes.byref(s_n),
            ctypes.byref(c_p),
            ctypes.byref(c_n),
            ctypes.byref(resolved),
        )
        return CSAResult(
            resolved=int(resolved.value),
            s_p=int(s_p.value),
            s_n=int(s_n.value),
            c_p=int(c_p.value),
            c_n=int(c_n.value),
            overflow=bool(ovf != 0),
        )

    def gpu_ternary_matvec(
        self,
        W_P: np.ndarray,
        W_N: np.ndarray,
        x_P: np.ndarray,
        x_N: np.ndarray,
        row_scales: Optional[np.ndarray] = None,
        x_scale: float = 1.0,
        backend: str = "auto",
    ) -> np.ndarray:
        """
        Bit-sliced dual-rail balanced ternary matrix-vector product:
        `y[m] = row_scales[m] * x_scale * <W[m, :], x>`.
        `W_P, W_N` have Python shape `(M, K_words)` (uint64, disjoint `(P & N) == 0`).
        """
        wp = np.asarray(W_P, dtype=np.uint64)
        wn = np.asarray(W_N, dtype=np.uint64)
        xp = np.ascontiguousarray(x_P, dtype=np.uint64)
        xn = np.ascontiguousarray(x_N, dtype=np.uint64)

        if wp.ndim != 2 or wn.shape != wp.shape:
            raise ValueError(f"W_P and W_N must be 2D with matching shapes, got {wp.shape} and {wn.shape}")
        M, K_words = wp.shape
        if xp.shape != (K_words,) or xn.shape != (K_words,):
            raise ValueError(f"x_P and x_N must have shape ({K_words},)")
        if np.any((wp & wn) != 0) or np.any((xp & xn) != 0):
            raise ValueError("Dual-rail invariant violated: P and N bitplanes must be disjoint")

        # C/Metal kernels index W[m, k] column-major: W_flat[m + M * k].
        wp_col = np.ascontiguousarray(wp.T, dtype=np.uint64)
        wn_col = np.ascontiguousarray(wn.T, dtype=np.uint64)

        rs_arr = None
        rs_ptr = ctypes.POINTER(ctypes.c_double)()
        if row_scales is not None:
            rs_arr = np.ascontiguousarray(row_scales, dtype=np.float64)
            if rs_arr.shape != (M,):
                raise ValueError(f"row_scales must have shape ({M},)")
            rs_ptr = rs_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        y_out = np.zeros(M, dtype=np.float64)
        if M == 0 or K_words == 0:
            return y_out

        use_gpu = (backend != "cpu") and (self.gpu is not None)
        if use_gpu:
            assert self.gpu is not None
            self.gpu.xbt_gpu_ternary_matvec_dualrail(
                ctypes.c_size_t(M),
                ctypes.c_size_t(K_words),
                wp_col.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
                wn_col.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
                rs_ptr,
                xp.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
                xn.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
                ctypes.c_double(float(x_scale)),
                y_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            )
            return y_out

        if self.mm is not None:
            self.mm.xbt_ternary_matvec_dualrail(
                ctypes.c_size_t(M),
                ctypes.c_size_t(K_words),
                wp_col.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
                wn_col.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
                rs_ptr,
                xp.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
                xn.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
                ctypes.c_double(float(x_scale)),
                y_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            )
            return y_out

        raise RuntimeError("Neither libtrium_gpu_bridge nor libtrium_mm_bridge is loaded")

    def gpu_msm_stencil27(
        self,
        Q_grid: np.ndarray,
        stencil_k27: np.ndarray,
        prefactor: float = 1.0,
        backend: str = "auto",
    ) -> Tuple[float, np.ndarray]:
        """
        27-point periodic 3D stencil convolution:
        `Phi[x, y, z] = sum_{dx,dy,dz in {-1,0,1}} k27[(dx+1) + 3*(dy+1) + 9*(dz+1)] * Q[x+dx, y+dy, z+dz]`
        Returns `(energy, Phi_grid)` where `energy = 0.5 * prefactor * sum(Q * Phi)`.
        """
        q_arr = np.asarray(Q_grid, dtype=np.float64)
        k_arr = np.ascontiguousarray(stencil_k27, dtype=np.float64)
        if q_arr.ndim != 3 or q_arr.shape[0] != q_arr.shape[1] or q_arr.shape[1] != q_arr.shape[2]:
            raise ValueError(f"Q_grid must be cubic 3D (Ml, Ml, Ml), got {q_arr.shape}")
        if k_arr.shape != (27,):
            raise ValueError(f"stencil_k27 must have shape (27,), got {k_arr.shape}")

        ml = q_arr.shape[0]
        if ml == 0:
            return 0.0, np.zeros((0, 0, 0), dtype=np.float64)

        # Kernel indexes grid[x + Ml*y + Ml^2*z], i.e. Fortran-contiguous (x, y, z).
        q_fort = np.asfortranarray(q_arr, dtype=np.float64)
        phi_fort = np.zeros((ml, ml, ml), dtype=np.float64, order="F")

        use_gpu = (backend != "cpu") and (self.gpu is not None)
        if use_gpu:
            assert self.gpu is not None
            energy = float(
                self.gpu.xbt_gpu_msm_stencil27(
                    ctypes.c_size_t(ml),
                    q_fort.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                    k_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                    ctypes.c_double(float(prefactor)),
                    phi_fort.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                )
            )
            return energy, np.ascontiguousarray(phi_fort, dtype=np.float64)

        # Pure-NumPy periodic 3D convolution fallback
        phi = np.zeros((ml, ml, ml), dtype=np.float64)
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    idx = (dx + 1) + 3 * (dy + 1) + 9 * (dz + 1)
                    kval = k_arr[idx]
                    if kval != 0.0:
                        phi += kval * np.roll(q_arr, shift=(-dx, -dy, -dz), axis=(0, 1, 2))
        energy = 0.5 * float(prefactor) * float(np.sum(q_arr * phi))
        return energy, phi


_DEFAULT_BRIDGE: Optional[NativeBridge] = None


def get_bridge() -> NativeBridge:
    global _DEFAULT_BRIDGE
    if _DEFAULT_BRIDGE is None:
        _DEFAULT_BRIDGE = NativeBridge()
    return _DEFAULT_BRIDGE


def probe_hardware() -> GPUProfile:
    return get_bridge().probe_hardware()


def mm_version() -> int:
    return get_bridge().mm_version()


def mm_abi_version() -> int:
    return get_bridge().mm_abi_version()


def exp_radix3(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return get_bridge().exp_radix3(x)


def tanh_radix3(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return get_bridge().tanh_radix3(x)


def rsqrt_fast(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return get_bridge().rsqrt_fast(x)


def erfc_fast(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return get_bridge().erfc_fast(x)


def csa_accumulate(terms: Iterable[int]) -> CSAResult:
    return get_bridge().csa_accumulate(terms)


def gpu_ternary_matvec(
    W_P: np.ndarray,
    W_N: np.ndarray,
    x_P: np.ndarray,
    x_N: np.ndarray,
    row_scales: Optional[np.ndarray] = None,
    x_scale: float = 1.0,
    backend: str = "auto",
) -> np.ndarray:
    return get_bridge().gpu_ternary_matvec(W_P, W_N, x_P, x_N, row_scales=row_scales, x_scale=x_scale, backend=backend)


def gpu_msm_stencil27(
    Q_grid: np.ndarray,
    stencil_k27: np.ndarray,
    prefactor: float = 1.0,
    backend: str = "auto",
) -> Tuple[float, np.ndarray]:
    return get_bridge().gpu_msm_stencil27(Q_grid, stencil_k27, prefactor=prefactor, backend=backend)


class GPUVerletPlan:
    """
    Context-managed compiled CSR Verlet neighbor plan for unified LJ + DSF Coulomb
    evaluation on Apple Metal GPU (with automatic CPU/NEON fallback).
    """

    TILE_SIZE = 32

    def __init__(
        self,
        positions: np.ndarray,
        sigma: np.ndarray,
        epsilon: np.ndarray,
        charges: Optional[np.ndarray] = None,
        box: Optional[Union[Sequence[float], np.ndarray]] = None,
        exclusions: Optional[Sequence[Sequence[int]]] = None,
        scaled_14: Optional[Sequence[Sequence[float]]] = None,
        lj14_scale: float = 0.5,
        coul14_scale: float = 1.0 / 1.2,
        rcut: float = 1.2,
        skin: float = 0.2,
        alpha_dsf: float = 0.0,
        coulomb_prefactor: float = 138.935456,
        use_lfs: bool = True,
        include_self: bool = True,
        backend: str = "auto",
        bridge: Optional[NativeBridge] = None,
    ) -> None:
        self._bridge = bridge or get_bridge()
        self._ptr: Optional[int] = None
        self._using_gpu: bool = False

        pos = np.ascontiguousarray(positions, dtype=np.float64)
        if pos.ndim != 2 or pos.shape[1] != 3 or pos.shape[0] == 0:
            raise ValueError(f"positions must have shape (N, 3) with N >= 1, got {pos.shape}")
        n = pos.shape[0]
        self.n_atoms = n
        self.rcut = float(rcut)
        self.skin = float(skin)
        self.alpha_dsf = float(alpha_dsf)
        self.coulomb_prefactor = float(coulomb_prefactor)
        self.use_lfs = bool(use_lfs)
        self.include_self = bool(include_self)

        sig = np.ascontiguousarray(sigma, dtype=np.float64)
        eps = np.ascontiguousarray(epsilon, dtype=np.float64)
        if sig.shape != (n,) or eps.shape != (n,):
            raise ValueError(f"sigma and epsilon must have shape ({n},)")
        chg = (
            np.zeros(n, dtype=np.float64)
            if charges is None
            else np.ascontiguousarray(charges, dtype=np.float64)
        )
        if chg.shape != (n,):
            raise ValueError(f"charges must have shape ({n},)")

        self._sigma = sig
        self._epsilon = eps
        self._charges = chg
        self._default_box = box

        is_tri, H_col, Hinv_col = _as_box_matrices(box)
        H_ptr = H_col.ctypes.data_as(ctypes.POINTER(ctypes.c_double)) if H_col is not None else ctypes.POINTER(ctypes.c_double)()
        Hinv_ptr = Hinv_col.ctypes.data_as(ctypes.POINTER(ctypes.c_double)) if Hinv_col is not None else ctypes.POINTER(ctypes.c_double)()

        # Partition 1..N into tiles of size 32 (1-based indices matching Julia/C++ ABI)
        n_tiles = (n + self.TILE_SIZE - 1) // self.TILE_SIZE
        order = np.arange(1, n + 1, dtype=np.int32)
        tile_starts = np.array([t * self.TILE_SIZE + 1 for t in range(n_tiles)], dtype=np.int32)
        tile_ends = np.array([min((t + 1) * self.TILE_SIZE, n) for t in range(n_tiles)], dtype=np.int32)

        ti_list = []
        tj_list = []
        for ti in range(1, n_tiles + 1):
            for tj in range(ti, n_tiles + 1):
                ti_list.append(ti)
                tj_list.append(tj)
        tile_i = np.array(ti_list, dtype=np.int32)
        tile_j = np.array(tj_list, dtype=np.int32)

        excl_keys = _encode_pair_keys(exclusions, n, with_weights=False)
        scaled_keys, scaled_weights = _encode_pair_keys(scaled_14, n, with_weights=True)

        excl_ptr = excl_keys.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)) if excl_keys.size else ctypes.POINTER(ctypes.c_uint64)()
        sk_ptr = scaled_keys.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)) if scaled_keys.size else ctypes.POINTER(ctypes.c_uint64)()
        sw_ptr = scaled_weights.ctypes.data_as(ctypes.POINTER(ctypes.c_double)) if scaled_weights.size else ctypes.POINTER(ctypes.c_double)()

        # Keep arrays alive for plan construction
        self._keepalive = (pos, sig, eps, chg, order, tile_starts, tile_ends, tile_i, tile_j, excl_keys, scaled_keys, scaled_weights)

        if backend not in ("auto", "cpu", "gpu", "metal"):
            raise ValueError(f"Unsupported backend '{backend}'")

        want_gpu = (backend != "cpu") and self._bridge.has_gpu_bridge
        if want_gpu:
            assert self._bridge.gpu is not None
            ptr = self._bridge.gpu.xbt_gpu_verlet_plan_create(
                ctypes.c_size_t(n),
                pos.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                sig.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                eps.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                chg.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                H_ptr,
                Hinv_ptr,
                ctypes.c_int(is_tri),
                ctypes.c_size_t(tile_i.size),
                tile_i.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                tile_j.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                order.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                tile_starts.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                tile_ends.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                ctypes.c_size_t(excl_keys.size),
                excl_ptr,
                ctypes.c_size_t(scaled_keys.size),
                sk_ptr,
                sw_ptr,
                ctypes.c_double(float(lj14_scale)),
                ctypes.c_double(float(coul14_scale)),
                ctypes.c_double(self.rcut),
                ctypes.c_double(self.skin),
                ctypes.c_double(self.alpha_dsf),
                ctypes.c_double(self.coulomb_prefactor),
                ctypes.c_int(1 if self.use_lfs else 0),
            )
            if ptr:
                self._ptr = int(ptr)
                self._using_gpu = True

        if self._ptr is None:
            if not self._bridge.has_cpu_bridge:
                raise RuntimeError("No native Trium-XBT CPU or GPU bridge library available")
            assert self._bridge.mm is not None
            ptr = self._bridge.mm.xbt_verlet_plan_create(
                ctypes.c_size_t(n),
                pos.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                sig.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                eps.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                chg.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                H_ptr,
                Hinv_ptr,
                ctypes.c_int(is_tri),
                ctypes.c_size_t(tile_i.size),
                tile_i.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                tile_j.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                order.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                tile_starts.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                tile_ends.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                ctypes.c_size_t(excl_keys.size),
                excl_ptr,
                ctypes.c_size_t(scaled_keys.size),
                sk_ptr,
                sw_ptr,
                ctypes.c_double(float(lj14_scale)),
                ctypes.c_double(float(coul14_scale)),
                ctypes.c_double(self.rcut),
                ctypes.c_double(self.skin),
                ctypes.c_double(self.alpha_dsf),
                ctypes.c_double(self.coulomb_prefactor),
                ctypes.c_int(1 if self.use_lfs else 0),
            )
            if not ptr:
                raise RuntimeError("xbt_verlet_plan_create failed")
            self._ptr = int(ptr)
            self._using_gpu = False

        # Compute DSF self-energy term matching TriumGPU.jl / TriumXBT.jl
        self._self_energy = 0.0
        if self.include_self and self.alpha_dsf > 0.0 and self.coulomb_prefactor != 0.0:
            erfc_rc = math.erfc(self.alpha_dsf * self.rcut)
            self._self_energy = -self.coulomb_prefactor * (
                erfc_rc / (2.0 * self.rcut) + self.alpha_dsf / math.sqrt(math.pi)
            ) * float(np.sum(chg * chg))

    @property
    def using_gpu(self) -> bool:
        return self._using_gpu

    @property
    def num_pairs(self) -> int:
        if self._ptr is None:
            return 0
        if self._using_gpu:
            assert self._bridge.gpu is not None
            return int(self._bridge.gpu.xbt_gpu_verlet_plan_num_pairs(ctypes.c_void_p(self._ptr)))
        assert self._bridge.mm is not None
        return int(self._bridge.mm.xbt_verlet_plan_num_pairs(ctypes.c_void_p(self._ptr)))

    def eval(
        self,
        positions: Optional[np.ndarray] = None,
        box: Optional[Union[Sequence[float], np.ndarray]] = None,
        compute_virial: bool = True,
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Evaluate nonbonded LJ + DSF energy, Cartesian forces `(N, 3)`, and virial `dU/dH` `(3, 3)`.
        """
        if self._ptr is None:
            raise RuntimeError("GPUVerletPlan has been closed")

        pos = self._keepalive[0] if positions is None else np.ascontiguousarray(positions, dtype=np.float64)
        if pos.shape != (self.n_atoms, 3):
            raise ValueError(f"positions must have shape ({self.n_atoms}, 3), got {pos.shape}")

        use_box = self._default_box if box is None else box
        is_tri, H_col, Hinv_col = _as_box_matrices(use_box)
        H_ptr = H_col.ctypes.data_as(ctypes.POINTER(ctypes.c_double)) if H_col is not None else ctypes.POINTER(ctypes.c_double)()
        Hinv_ptr = Hinv_col.ctypes.data_as(ctypes.POINTER(ctypes.c_double)) if Hinv_col is not None else ctypes.POINTER(ctypes.c_double)()

        need_vir = bool(compute_virial and use_box is not None)
        forces = np.zeros((self.n_atoms, 3), dtype=np.float64)
        virial9 = np.zeros(9, dtype=np.float64)
        vir_ptr = virial9.ctypes.data_as(ctypes.POINTER(ctypes.c_double)) if need_vir else ctypes.POINTER(ctypes.c_double)()

        if self._using_gpu:
            assert self._bridge.gpu is not None
            u_pair = float(
                self._bridge.gpu.xbt_gpu_verlet_plan_eval(
                    ctypes.c_void_p(self._ptr),
                    pos.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                    H_ptr,
                    Hinv_ptr,
                    ctypes.c_int(is_tri),
                    ctypes.c_int(1 if need_vir else 0),
                    forces.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                    vir_ptr,
                )
            )
        else:
            assert self._bridge.mm is not None
            u_pair = float(
                self._bridge.mm.xbt_verlet_plan_eval_mt(
                    ctypes.c_void_p(self._ptr),
                    pos.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                    H_ptr,
                    Hinv_ptr,
                    ctypes.c_int(is_tri),
                    ctypes.c_int(0),
                    ctypes.c_int(0),
                    forces.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                    ctypes.POINTER(ctypes.c_int64)(),
                    vir_ptr,
                )
            )

        # Native virial9 is column-major dU/dH[a, b] = [v00, v10, v20, v01, ...]
        virial_3x3 = virial9.reshape((3, 3)).T.copy() if need_vir else np.zeros((3, 3), dtype=np.float64)
        return self._self_energy + u_pair, forces, virial_3x3

    def run_batched_steps(
        self,
        positions: np.ndarray,
        momenta: np.ndarray,
        masses: np.ndarray,
        dt: float,
        n_steps: int,
        bonds: Optional[Sequence[Tuple[int, int, float, float]]] = None,
        angles: Optional[Sequence[Tuple[int, int, int, float, float]]] = None,
        box: Optional[Union[Sequence[float], np.ndarray]] = None,
    ) -> dict[str, Union[np.ndarray, float]]:
        """
        Execute `n_steps` of GPU-resident Velocity Verlet integration via `xbt_gpu_md_batch_run`.
        `bonds` entries are `(i, j, k, r0)` (0-based); `angles` entries are `(i, j, k, ktheta, theta0)` (0-based).
        """
        if self._ptr is None:
            raise RuntimeError("GPUVerletPlan has been closed")
        if not self._using_gpu or self._bridge.gpu is None:
            raise RuntimeError("run_batched_steps requires an active GPU plan")

        q = np.ascontiguousarray(positions, dtype=np.float64).copy()
        p = np.ascontiguousarray(momenta, dtype=np.float64).copy()
        m = np.ascontiguousarray(masses, dtype=np.float64)
        if q.shape != (self.n_atoms, 3) or p.shape != (self.n_atoms, 3) or m.shape != (self.n_atoms,):
            raise ValueError("positions, momenta, and masses shapes do not match n_atoms")
        if np.any(m <= 0) or not np.all(np.isfinite(m)):
            raise ValueError("masses must be positive and finite")
        inv_mass = np.ascontiguousarray(1.0 / m, dtype=np.float64)

        use_box = self._default_box if box is None else box
        is_tri, H_col, Hinv_col = _as_box_matrices(use_box)
        H_ptr = H_col.ctypes.data_as(ctypes.POINTER(ctypes.c_double)) if H_col is not None else ctypes.POINTER(ctypes.c_double)()
        Hinv_ptr = Hinv_col.ctypes.data_as(ctypes.POINTER(ctypes.c_double)) if Hinv_col is not None else ctypes.POINTER(ctypes.c_double)()

        nb = len(bonds) if bonds else 0
        if nb > 0:
            assert bonds is not None
            bi = np.array([b[0] for b in bonds], dtype=np.int32)
            bj = np.array([b[1] for b in bonds], dtype=np.int32)
            bk = np.array([b[2] for b in bonds], dtype=np.float64)
            br0 = np.array([b[3] for b in bonds], dtype=np.float64)
            bi_p = bi.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
            bj_p = bj.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
            bk_p = bk.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            br0_p = br0.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        else:
            bi_p = bj_p = ctypes.POINTER(ctypes.c_int32)()
            bk_p = br0_p = ctypes.POINTER(ctypes.c_double)()

        na = len(angles) if angles else 0
        if na > 0:
            assert angles is not None
            ai = np.array([a[0] for a in angles], dtype=np.int32)
            aj = np.array([a[1] for a in angles], dtype=np.int32)
            ak = np.array([a[2] for a in angles], dtype=np.int32)
            akth = np.array([a[3] for a in angles], dtype=np.float64)
            ath0 = np.array([a[4] for a in angles], dtype=np.float64)
            ai_p = ai.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
            aj_p = aj.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
            ak_p = ak.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
            akth_p = akth.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            ath0_p = ath0.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        else:
            ai_p = aj_p = ak_p = ctypes.POINTER(ctypes.c_int32)()
            akth_p = ath0_p = ctypes.POINTER(ctypes.c_double)()

        forces = np.zeros((self.n_atoms, 3), dtype=np.float64)
        pe_ref = ctypes.c_double(0.0)
        ke_ref = ctypes.c_double(0.0)

        e_raw = float(
            self._bridge.gpu.xbt_gpu_md_batch_run(
                ctypes.c_void_p(self._ptr),
                q.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                p.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                inv_mass.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                H_ptr,
                Hinv_ptr,
                ctypes.c_int(is_tri),
                ctypes.c_size_t(nb),
                bi_p,
                bj_p,
                bk_p,
                br0_p,
                ctypes.c_size_t(na),
                ai_p,
                aj_p,
                ak_p,
                akth_p,
                ath0_p,
                ctypes.c_double(float(dt)),
                ctypes.c_int(int(n_steps)),
                forces.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                ctypes.byref(pe_ref),
                ctypes.byref(ke_ref),
            )
        )
        pe = float(pe_ref.value) + self._self_energy
        ke = float(ke_ref.value)
        return {
            "positions": q,
            "momenta": p,
            "forces": forces,
            "potential_energy": pe,
            "kinetic_energy": ke,
            "total_energy": e_raw + self._self_energy,
        }

    def close(self) -> None:
        if self._ptr is not None:
            ptr = self._ptr
            self._ptr = None
            if self._using_gpu and self._bridge.gpu is not None:
                self._bridge.gpu.xbt_gpu_verlet_plan_destroy(ctypes.c_void_p(ptr))
            elif self._bridge.mm is not None:
                self._bridge.mm.xbt_verlet_plan_destroy(ctypes.c_void_p(ptr))

    def __enter__(self) -> "GPUVerletPlan":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def run_batched_gpu_steps(
    positions: np.ndarray,
    momenta: np.ndarray,
    masses: np.ndarray,
    sigma: np.ndarray,
    epsilon: np.ndarray,
    dt: float,
    n_steps: int,
    charges: Optional[np.ndarray] = None,
    box: Optional[Union[Sequence[float], np.ndarray]] = None,
    bonds: Optional[Sequence[Tuple[int, int, float, float]]] = None,
    angles: Optional[Sequence[Tuple[int, int, int, float, float]]] = None,
    exclusions: Optional[Sequence[Sequence[int]]] = None,
    scaled_14: Optional[Sequence[Sequence[float]]] = None,
    rcut: float = 1.2,
    skin: float = 0.2,
    alpha_dsf: float = 0.0,
    coulomb_prefactor: float = 138.935456,
    use_lfs: bool = True,
) -> dict[str, Union[np.ndarray, float]]:
    """Convenience wrapper building a `GPUVerletPlan` and running `n_steps` on Metal GPU."""
    with GPUVerletPlan(
        positions=positions,
        sigma=sigma,
        epsilon=epsilon,
        charges=charges,
        box=box,
        exclusions=exclusions,
        scaled_14=scaled_14,
        rcut=rcut,
        skin=skin,
        alpha_dsf=alpha_dsf,
        coulomb_prefactor=coulomb_prefactor,
        use_lfs=use_lfs,
        backend="gpu",
    ) as plan:
        return plan.run_batched_steps(
            positions=positions,
            momenta=momenta,
            masses=masses,
            dt=dt,
            n_steps=n_steps,
            bonds=bonds,
            angles=angles,
            box=box,
        )
