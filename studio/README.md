# Biology-MM Studio

Biology-MM Studio is an interactive, GPU-accelerated biomolecular modeling and nanotech simulation workbench built on top of **Mol\* v6** and the **Biology-MM Metal 4 / C++ simulation engine**.

It enables real-time interactive visualization, simulation steering, live telemetry analysis, and custom physics scripting for biomacromolecules and multi-million-atom assemblies (including full viral capsids like 40M-atom Faustovirus 5J7V).

---

## Architecture Overview

```
studio/
├── backend/                        # FastAPI simulation server & engine bridge
│   ├── trium_studio/               # Core backend package
│   │   ├── app.py                  # FastAPI REST + WebSocket endpoints
│   │   ├── engine_core.py          # Native Metal/C++ simulation session bridge
│   │   ├── protocol.py             # Binary protocol packet definitions (48-byte header)
│   │   ├── cif_builder.py          # mmCIF parser & binary coordinate stream generator
│   │   ├── mvs_exporter.py         # MolViewSpec declarative state exporter
│   │   ├── structure_loader.py     # RCSB PDB / PDBe structure fetcher
│   │   ├── sandbox.py              # In-process Python simulation code execution
│   │   └── ws_manager.py           # WebSocket connection pool & telemetry broadcast
│   ├── tests/                      # Pytest unit & integration test suite
│   ├── benchmarks/                 # Benchmarks (e.g. 108k Argon, multi-million scaling)
│   └── requirements.txt            # Backend dependencies
├── frontend/
│   ├── biology-mm-studio/          # React + Mol* v6 Studio application
│   │   ├── src/
│   │   │   ├── components/         # UI Panes (Toolbar, Viewport, Telemetry, CodeLab)
│   │   │   ├── molstar-ui/         # Native Mol* canvas overlay controls
│   │   │   ├── hooks/              # Real-time WebSocket streaming hooks
│   │   │   └── studio-store.ts     # Global reactive state management
│   │   ├── vite.config.ts
│   │   └── package.json
│   ├── biology-mm-extension/       # Mol* v6 GPU-accelerated extension
│   │   └── src/
│   │       ├── structure-bridge.ts # 48-byte aligned binary coordinate loader
│   │       ├── gpu-buffer-updater.ts# WebGL2 direct GPU attribute buffer updates
│   │       ├── live-stream-behavior.ts# Real-time state synchronizer
│   │       ├── physics-color-themes.ts# Kinetic energy, velocity, force colormaps
│   │       └── physics-overlays.ts # Force vector & velocity vector overlays
│   └── molstar-patches/            # Upstream Mol* core patches for large assemblies
└── start_studio.sh                 # Launcher script
```

---

## Key Features

1. **High-Performance Binary Streaming**:
   - Uses an aligned 48-byte binary header (`<4sIIffIIfffII`) followed by contiguous Float32 coordinate arrays.
   - Eliminates JSON serialization overhead, streaming coordinates directly into Mol\* WebGL2 buffer arrays at 60 FPS.

2. **Metal 4 Streaming Tile Kernel**:
   - Integrates with `trium_gpu_metal.mm` $32 \times 32$ SIMD threadgroup streaming tile kernel with hardware atomic float accumulation.
   - Scales past 40,000,000 atoms without neighbor list RAM exhaustion.

3. **Multi-Scale Visualization**:
   - Seamless assembly symmetry support for giant viral capsids.
   - Physics-informed color themes: kinetic energy, temperature, force gradient, and root-mean-square displacement (RMSD).

4. **Live Code Lab & Simulation Steering**:
   - Interactive in-browser Python Code Lab for applying real-time force fields, optical tweezers, or pulling forces during active simulation runs.

---

## Quickstart

### Backend Setup
```bash
# Navigate to studio directory
cd studio

# Launch studio in production mode (starts API on http://127.0.0.1:8765)
./start_studio.sh
```

### Development Mode
```bash
./start_studio.sh --dev
```
