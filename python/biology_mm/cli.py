"""Command-line interface for Biology MM Python SDK."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional, Sequence

from . import __version__
from .native import get_bridge, mm_abi_version, mm_version, probe_hardware
from .simulation import discover_julia_toolchain


def build_info_payload() -> dict:
    bridge = get_bridge()
    prof = probe_hardware()
    tc = discover_julia_toolchain()
    return {
        "sdk_version": __version__,
        "cpu_bridge_loaded": bridge.has_cpu_bridge,
        "cpu_bridge_path": str(bridge.mm_lib_path) if bridge.mm_lib_path else None,
        "mm_version": mm_version() if bridge.has_cpu_bridge else None,
        "mm_abi_version": mm_abi_version() if bridge.has_cpu_bridge else None,
        "gpu_bridge_loaded": bridge.has_gpu_bridge,
        "gpu_bridge_path": str(bridge.gpu_lib_path) if bridge.gpu_lib_path else None,
        "hardware": {
            "has_gpu": prof.has_gpu,
            "backend": prof.backend,
            "device_name": prof.device_name,
            "unified_memory": prof.unified_memory,
            "max_threads_per_tg": prof.max_threads_per_tg,
            "simd_width": prof.simd_width,
            "vram_bytes": prof.vram_bytes,
        },
        "julia_toolchain": tc,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="biology_mm", description="Biology MM Python SDK CLI")
    sub = parser.add_subparsers(dest="command")
    info_p = sub.add_parser("info", help="Print native hardware and Julia toolchain diagnostics")
    info_p.add_argument("--json", action="store_true", help="Emit machine-readable JSON")

    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command == "info":
        payload = build_info_payload()
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            hw = payload["hardware"]
            tc = payload["julia_toolchain"]
            print("Biology MM Python SDK v1.0.0")
            print(f"  CPU Bridge : {payload['cpu_bridge_loaded']} ({payload['cpu_bridge_path']})")
            print(f"  MM Version : {payload['mm_version']} (ABI v{payload['mm_abi_version']})")
            print(f"  GPU Bridge : {payload['gpu_bridge_loaded']} ({payload['gpu_bridge_path']})")
            print(f"  Device     : {hw['device_name']} [{hw['backend']}]")
            print(f"  UnifiedMem : {hw['unified_memory']}, SIMD={hw['simd_width']}, MaxTG={hw['max_threads_per_tg']}")
            print(f"  VRAM Bytes : {hw['vram_bytes']}")
            print(f"  Julia      : {tc['julia_bin']} (available={tc['available']})")
            print(f"  Depot      : {tc['julia_depot_path']}")
        return 0

    parser.print_help(sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
