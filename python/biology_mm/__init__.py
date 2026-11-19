"""Biology MM Python SDK: Native Trium-XBT / Apple Metal MM acceleration and simulation orchestration."""

from .builders import AtomicSystem, NanostructureBuilder, SystemBuilder
from .native import (
    CSAResult,
    GPUProfile,
    GPUVerletPlan,
    NativeBridge,
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
from .parameters import CustomPotential, ParameterEncoder, ParameterSpec
from .simulation import Simulation, SimulationConfig, SimulationResult

__version__ = "1.0.0"

__all__ = [
    "AtomicSystem",
    "CSAResult",
    "CustomPotential",
    "GPUProfile",
    "GPUVerletPlan",
    "NanostructureBuilder",
    "NativeBridge",
    "ParameterEncoder",
    "ParameterSpec",
    "Simulation",
    "SimulationConfig",
    "SimulationResult",
    "SystemBuilder",
    "__version__",
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
