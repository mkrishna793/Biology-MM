"""FastAPI application and REST/WebSocket routes for Biology-MM-Studio."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from biology_mm import AtomicSystem, probe_hardware

from .cif_builder import build_mmcif_from_session
from .engine_core import (
    PHYSICS_CHANNEL_NAMES,
    CubicSplineRadialTable,
    SimulationSession,
    compile_safe_radial_expr,
)
try:
    from .mvs_exporter import export_mvsj, export_mvsx_bytes
    _HAS_MVS_EXPORTER = True
except ImportError:
    export_mvsj = None  # type: ignore[assignment]
    export_mvsx_bytes = None  # type: ignore[assignment]
    _HAS_MVS_EXPORTER = False

from .multiscale import _decode_element_byte, expand_symmetry_assembly, unpack_tsym_binary
from .ternary_compact import CompactSymmetryMultiscaleSession, encode_elements_uint8
from .tryte_tiles import TryteTileFullAtomSession
from .presets import PRESET_DEFINITIONS, create_preset_session, get_preset, list_presets
from .sandbox import execute_user_code
from .structure_loader import load_structure_from_bytes, load_structure_pair_from_bytes
from .ws_manager import PALETTE_NAMES, ManagedSession, SessionManager

STUDIO_VERSION = "1.0.0"
_DIST_CANDIDATES = [
    Path(__file__).resolve().parents[2] / "molstar" / "apps" / "biology-mm-studio" / "dist",
    Path(__file__).resolve().parents[1] / "frontend" / "biology-mm-studio" / "dist",
    Path("/Users/mohana/Desktop/Biology-MM-Studio/molstar/apps/biology-mm-studio/dist"),
]
DEFAULT_DIST_DIR = next((p for p in _DIST_CANDIDATES if (p / "index.html").is_file()), _DIST_CANDIDATES[0])


class CreateSessionRequest(BaseModel):
    preset_id: str = "cnt-tersoff-tensile"


class SteerPayload(BaseModel):
    temperature: Optional[float] = None
    efield_z: Optional[float] = None
    gcmc_mu: Optional[float] = None
    strain_rate: Optional[float] = None
    active_channel: Optional[str] = None
    palette: Optional[str] = None


class ScrubPayload(BaseModel):
    target_step: int = Field(..., ge=0)


class StepPayload(BaseModel):
    n_steps: int = Field(1, ge=1, le=5000)


class CustomPhysicsPayload(BaseModel):
    potential_expr: str
    r_min: float = Field(0.08, gt=0.0)
    r_max: float = Field(0.50, gt=0.0)
    n_points: int = Field(256, ge=16, le=2048)
    temperature: float = Field(300.0, ge=0.1)
    efield_z: float = 0.0
    gcmc_mu: float = -2.0
    strain_rate: float = 0.0
    dt: Optional[float] = Field(None, gt=0.0, le=0.05)
    seed_preset_id: Optional[str] = None
    params: Optional[Dict[str, float]] = None


class ActiveStructurePayload(BaseModel):
    title: str = "Imported Mol* Structure"
    coords_ang: List[List[float]]
    elements: List[str]
    charges: Optional[List[float]] = None
    bonds: Optional[List[List[int]]] = None
    box_ang: Optional[List[float]] = None
    structure_metadata: Optional[Dict[str, Any]] = None


class CreateFromStructurePayload(BaseModel):
    structure: ActiveStructurePayload
    mode: str = "metal_verlet"
    temperature: float = Field(300.0, ge=0.1)
    dt: float = Field(0.001, gt=0.0, le=0.05)


class CodeRunPayload(BaseModel):
    code: str
    timeout_s: float = Field(10.0, gt=0.1, le=30.0)
    active_structure: Optional[ActiveStructurePayload] = None


class RelaxPayload(BaseModel):
    max_steps: int = Field(40, ge=1, le=500)
    max_force_cap: float = Field(5000.0, gt=0.0)
    alpha_sc: float = Field(0.5, gt=0.0)
    lam: float = Field(0.5, gt=0.0, le=1.0)


class MultiscalePayload(BaseModel):
    center_ang: List[float]
    aa_radius_ang: float = Field(15.0, ge=0.0)
    cg_radius_ang: float = Field(35.0, ge=0.0)


class SteerAtomPayload(BaseModel):
    atom_index: Optional[int] = None
    target_ang: Optional[List[float]] = None
    k_spring: float = Field(500.0, ge=0.0)


_ELEMENT_MASSES: Dict[str, float] = {
    "H": 1.008,
    "HE": 4.0026,
    "LI": 6.94,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "F": 18.998,
    "NE": 20.180,
    "NA": 22.990,
    "MG": 24.305,
    "AL": 26.982,
    "SI": 28.085,
    "P": 30.974,
    "S": 32.06,
    "CL": 35.45,
    "K": 39.098,
    "AR": 39.948,
    "CA": 40.078,
    "FE": 55.845,
    "ZN": 65.38,
    "SE": 78.971,
    "BR": 79.904,
    "I": 126.90,
}


def _build_system_from_structure_payload(
    payload: ActiveStructurePayload,
) -> Tuple[AtomicSystem, Optional[Dict[str, Any]]]:
    n = len(payload.coords_ang)
    if n == 0 or len(payload.elements) != n:
        raise HTTPException(
            status_code=400,
            detail="coords_ang and elements must be non-empty and match in length",
        )
    coords_ang = np.asarray(payload.coords_ang, dtype=np.float64)
    if coords_ang.shape != (n, 3) or not np.all(np.isfinite(coords_ang)):
        raise HTTPException(status_code=400, detail="coords_ang must have finite shape (N, 3)")
    pos_nm = coords_ang * 0.1

    names: List[str] = []
    masses = np.empty(n, dtype=np.float64)
    for i, raw_el in enumerate(payload.elements):
        cleaned = "".join(ch for ch in str(raw_el).strip() if ch.isalpha())
        sym = (cleaned[0].upper() + cleaned[1:].lower()) if cleaned else "C"
        names.append(sym)
        masses[i] = _ELEMENT_MASSES.get(sym.upper(), 12.011)

    if payload.charges is not None and len(payload.charges) == n:
        charges = np.asarray(payload.charges, dtype=np.float64)
    else:
        charges = np.zeros(n, dtype=np.float64)

    margin = 0.9  # 9 Angstroms uniform margin inside simulation box
    min_xyz = np.min(pos_nm, axis=0) if n > 0 else np.zeros(3, dtype=np.float64)
    pos_shifted = pos_nm - min_xyz
    span = np.ptp(pos_shifted, axis=0) if n > 1 else np.zeros(3, dtype=np.float64)
    bx, by, bz = 0.0, 0.0, 0.0
    if payload.box_ang and len(payload.box_ang) == 3:
        bx, by, bz = (max(0.0, float(v) * 0.1) for v in payload.box_ang)
    box_nm = (
        max(bx, float(span[0]) + 2 * margin, 2.8),
        max(by, float(span[1]) + 2 * margin, 2.8),
        max(bz, float(span[2]) + 2 * margin, 2.8),
    )
    offset = 0.5 * (np.array(box_nm) - span)
    pos_nm = pos_shifted + offset

    bonds: List[Tuple[int, int]] = []
    seen_bonds = set()
    if payload.bonds:
        for pair in payload.bonds:
            if len(pair) != 2:
                continue
            i0, j0 = int(pair[0]), int(pair[1])
            if 0 <= i0 < n and 0 <= j0 < n and i0 != j0:
                key = (min(i0, j0), max(i0, j0))
                if key not in seen_bonds:
                    seen_bonds.add(key)
                    bonds.append(key)

    sys = AtomicSystem(
        positions=pos_nm,
        masses=masses,
        charges=charges,
        box_lengths=box_nm,
        names=names,
        bonds=bonds,
        units="md",
    )
    return sys, payload.structure_metadata


def _require_session(manager: SessionManager, sid: str) -> ManagedSession:
    record = manager.get_session(sid)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Session '{sid}' not found")
    return record


def create_app(dist_dir: Optional[Path] = None) -> FastAPI:
    manager = SessionManager()
    resolved_dist = Path(dist_dir) if dist_dir is not None else DEFAULT_DIST_DIR

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        try:
            yield
        finally:
            manager.close_all()

    application = FastAPI(
        title="Biology-MM-Studio API",
        version=STUDIO_VERSION,
        lifespan=lifespan,
    )
    application.state.session_manager = manager

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @application.middleware("http")
    async def add_no_cache_headers(request: Request, call_next):
        response = await call_next(request)
        if request.url.path == "/" or request.url.path.endswith(".html") or request.url.path.endswith(".js"):
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response

    @application.get("/api/health")
    async def health() -> Dict[str, str]:
        return {"status": "ok", "version": STUDIO_VERSION}

    @application.get("/api/capabilities")
    async def capabilities() -> Dict[str, Any]:
        hw = probe_hardware()
        total_ram_mb = 0.0
        with manager._lock:
            for rec in manager._sessions.values():
                try:
                    total_ram_mb += float(rec.session.get_metrics().get("ram_mb", 0.0))
                except Exception:
                    pass
        return {
            "studio_version": STUDIO_VERSION,
            "hardware": asdict(hw),
            "channels": list(PHYSICS_CHANNEL_NAMES),
            "palettes": list(PALETTE_NAMES),
            "presets_count": len(PRESET_DEFINITIONS),
            "max_interactive_atoms": 262_144,
            "supports_mvsx": _HAS_MVS_EXPORTER,
            "supports_symmetry_compact": True,
            "supports_base243_ternary": True,
            "supports_tryte_27_plus_5": True,
            "storage_mode": "symmetry-compact",
            "ram_mb": round(total_ram_mb, 4),
        }

    @application.get("/api/presets")
    async def presets() -> list[Dict[str, Any]]:
        return list_presets()

    @application.post("/api/sessions")
    async def create_session(payload: Optional[CreateSessionRequest] = None) -> Dict[str, Any]:
        preset_id = payload.preset_id if payload is not None else "cnt-tersoff-tensile"
        try:
            preset_info = get_preset(preset_id)
            sim = create_preset_session(preset_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        record = manager.register_session(
            sim,
            default_channel=preset_info["default_channel"],
            default_palette=preset_info["default_palette"],
        )
        metrics = sim.get_metrics()
        metrics["session_id"] = record.session_id
        return {
            "session_id": record.session_id,
            "preset": preset_info,
            "metrics": metrics,
        }


    @application.post("/api/sessions/create-from-binary")
    async def create_session_from_binary(
        request: Request,
        storage_mode: str = Query("auto"),
    ) -> Dict[str, Any]:
        """
        Stream multi-million atom structures directly from Mol* as flat binary Float32Array,
        completely bypassing V8 JSON string length limits and RAM heap explosions.
        """
        body = await request.body()
        if len(body) < 48:
            raise HTTPException(status_code=400, detail="Binary payload too short (< 48 bytes)")

        # Header: magic (4s), n_atoms (I), title_len (I), temperature (f), dt (f), mode_len (I), flags (I), box (3f), reserved (2I) = 48 bytes
        import struct
        magic, n_atoms, title_len, temperature, dt, mode_len, flags, bx, by, bz, _r1, _r2 = struct.unpack(
            "<4sIIffIIfffII", body[:48]
        )
        if magic != b"TBIN":
            raise HTTPException(status_code=400, detail="Invalid binary format magic (expected TBIN)")

        offset = 48
        coords_bytes = n_atoms * 3 * 4
        if len(body) < offset + coords_bytes:
            raise HTTPException(status_code=400, detail=f"Incomplete coordinates: expected {coords_bytes} bytes, got {len(body) - offset}")

        mode_sel = str(storage_mode or "auto").strip().lower()
        if mode_sel not in ("auto", "tryte-all-atom", "full", "float64-full"):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid storage_mode '{storage_mode}'; expected 'auto', 'tryte-all-atom', or 'float64-full'",
            )

        coords_ang = np.frombuffer(body[offset : offset + coords_bytes], dtype=np.float32).reshape(n_atoms, 3)
        offset += coords_bytes

        has_elements = bool(flags & 0x01)
        if has_elements and len(body) >= offset + n_atoms:
            elem_bytes = body[offset : offset + n_atoms]
            names = [_decode_element_byte(b) for b in elem_bytes]
            offset += n_atoms
        else:
            names = ["C"] * n_atoms

        title = body[offset : offset + title_len].decode("utf-8", errors="ignore") if title_len > 0 else "Imported Mol* Structure"
        offset += title_len
        mode = body[offset : offset + mode_len].decode("utf-8", errors="ignore") if mode_len > 0 else "metal_verlet"
        offset += mode_len

        if mode_sel in ("tryte-all-atom", "full") or (mode_sel == "auto" and n_atoms > 50_000):
            pos_nm_f32 = (coords_ang * np.float32(0.1)).astype(np.float32, copy=True)
            z_u8 = encode_elements_uint8(names)
            box_nm_opt = (float(bx) * 0.1, float(by) * 0.1, float(bz) * 0.1) if (bx > 0 and by > 0 and bz > 0) else None
            sim_tryte = TryteTileFullAtomSession(
                positions_nm=pos_nm_f32,
                atomic_numbers=z_u8,
                charges=None,
                box_nm=box_nm_opt,
                mode=mode,
                dt=dt if dt > 0 else 0.002,
                temperature=temperature if temperature > 0 else 300.0,
                structure_metadata={"source": "molstar-binary-tryte", "title": title, "atom_count": n_atoms},
            )
            minimization = sim_tryte.minimize()
            record = manager.register_session(
                sim_tryte,
                default_channel="force-magnitude",
                default_palette="plasma",
            )
            metrics = sim_tryte.get_metrics()
            metrics["session_id"] = record.session_id
            return {
                "session_id": record.session_id,
                "title": title,
                "storage_mode": metrics["storage_mode"],
                "minimization": minimization,
                "metrics": metrics,
            }

        pos_nm = coords_ang.astype(np.float64) * 0.1
        masses = np.array([_ELEMENT_MASSES.get(s.upper(), 12.011) for s in names], dtype=np.float64)
        charges = np.zeros(n_atoms, dtype=np.float64)

        margin = 0.9  # 9 Angstroms margin
        min_xyz = np.min(pos_nm, axis=0) if n_atoms > 0 else np.zeros(3, dtype=np.float64)
        pos_shifted = pos_nm - min_xyz
        span = np.ptp(pos_shifted, axis=0) if n_atoms > 1 else np.zeros(3, dtype=np.float64)
        if bx > 0 and by > 0 and bz > 0:
            box_nm = (float(bx) * 0.1, float(by) * 0.1, float(bz) * 0.1)
        else:
            box_nm = (
                max(float(span[0]) + 2 * margin, 2.8),
                max(float(span[1]) + 2 * margin, 2.8),
                max(float(span[2]) + 2 * margin, 2.8),
            )
        offset_pos = 0.5 * (np.array(box_nm) - span)
        pos_nm = pos_shifted + offset_pos

        sys = AtomicSystem(
            positions=pos_nm,
            masses=masses,
            charges=charges,
            box_lengths=box_nm,
            names=names,
            bonds=[],
            units="md",
        )

        sim = SimulationSession(
            sys,
            mode=mode,
            dt=dt if dt > 0 else 0.001,
            temperature=temperature if temperature > 0 else 300.0,
            rcut=1.0,
            alpha_dsf=0.0,
            structure_metadata={"source": "molstar-binary", "atom_count": n_atoms},
        )
        minimization = sim.minimize()
        record = manager.register_session(
            sim,
            default_channel="force-magnitude",
            default_palette="plasma",
        )
        metrics = sim.get_metrics()
        metrics["session_id"] = record.session_id
        return {
            "session_id": record.session_id,
            "title": title,
            "minimization": minimization,
            "metrics": metrics,
        }

    @application.post("/api/sessions/create-from-symmetry")
    async def create_session_from_symmetry(
        request: Request,
        storage_mode: str = Query("auto"),
    ) -> Dict[str, Any]:
        """
        Ingest a compact 4-byte-aligned `TSYM` biological symmetry assembly packet.
        For large assemblies (> 50,000 atoms in `auto` mode, or explicit `symmetry-compact`),
        instantiates `CompactSymmetryMultiscaleSession` directly WITHOUT expanding 40M+
        float64 arrays in RAM.
        """
        mode_sel = str(storage_mode or "auto").strip().lower()
        if mode_sel not in ("auto", "symmetry-compact", "tryte-all-atom", "full"):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid storage_mode '{storage_mode}'; expected 'auto', 'symmetry-compact', 'tryte-all-atom', or 'full'",
            )

        body = await request.body()
        try:
            sym = unpack_tsym_binary(body)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        n_total = int(sym.n_asu_atoms) * int(sym.n_transforms)
        if n_total == 0:
            raise HTTPException(status_code=400, detail="Symmetry assembly contains 0 atoms")

        if mode_sel == "symmetry-compact" or (mode_sel == "auto" and n_total > 50_000):
            sim_compact = CompactSymmetryMultiscaleSession.from_tsym_payload(sym)
            relaxation = sim_compact.relax_clashes()
            record = manager.register_session(
                sim_compact,
                default_channel="force-magnitude",
                default_palette="plasma",
            )
            metrics = sim_compact.get_metrics()
            metrics["session_id"] = record.session_id
            return {
                "session_id": record.session_id,
                "title": sym.title,
                "n_asu_atoms": sym.n_asu_atoms,
                "n_transforms": sym.n_transforms,
                "n_atoms": n_total,
                "storage_mode": metrics["storage_mode"],
                "relaxation": relaxation,
                "metrics": metrics,
            }

        if mode_sel in ("tryte-all-atom", "full"):
            sim_tryte = TryteTileFullAtomSession.from_tsym_payload(sym)
            relaxation = sim_tryte.relax_clashes()
            record = manager.register_session(
                sim_tryte,
                default_channel="force-magnitude",
                default_palette="plasma",
            )
            metrics = sim_tryte.get_metrics()
            metrics["session_id"] = record.session_id
            return {
                "session_id": record.session_id,
                "title": sym.title,
                "n_asu_atoms": sym.n_asu_atoms,
                "n_transforms": sym.n_transforms,
                "n_atoms": n_total,
                "storage_mode": metrics["storage_mode"],
                "relaxation": relaxation,
                "metrics": metrics,
            }

        expanded_coords_ang, expanded_elements = expand_symmetry_assembly(
            sym.asu_coords_ang,
            sym.asu_elements,
            sym.transforms_4x4,
        )
        n_atoms = int(expanded_coords_ang.shape[0])
        if n_atoms == 0:
            raise HTTPException(status_code=400, detail="Symmetry expansion produced 0 atoms")

        pos_nm = expanded_coords_ang.astype(np.float64) * 0.1
        masses = np.array([_ELEMENT_MASSES.get(s.upper(), 12.011) for s in expanded_elements], dtype=np.float64)
        charges = np.zeros(n_atoms, dtype=np.float64)

        margin = 0.9
        min_xyz = np.min(pos_nm, axis=0)
        pos_shifted = pos_nm - min_xyz
        span = np.ptp(pos_shifted, axis=0) if n_atoms > 1 else np.zeros(3, dtype=np.float64)
        bx, by, bz = sym.box_ang
        if bx > 0 and by > 0 and bz > 0:
            box_nm = (float(bx) * 0.1, float(by) * 0.1, float(bz) * 0.1)
        else:
            box_nm = (
                max(float(span[0]) + 2 * margin, 2.8),
                max(float(span[1]) + 2 * margin, 2.8),
                max(float(span[2]) + 2 * margin, 2.8),
            )
        offset_pos = 0.5 * (np.array(box_nm) - span)
        pos_nm = pos_shifted + offset_pos

        sys = AtomicSystem(
            positions=pos_nm,
            masses=masses,
            charges=charges,
            box_lengths=box_nm,
            names=expanded_elements,
            bonds=[],
            units="md",
        )
        sim = SimulationSession(
            sys,
            mode=sym.mode or "metal_verlet",
            dt=sym.dt if sym.dt > 0 else 0.002,
            temperature=sym.temperature if sym.temperature > 0 else 300.0,
            rcut=1.0,
            alpha_dsf=0.0,
            structure_metadata={
                "source": "molstar-symmetry",
                "n_asu_atoms": sym.n_asu_atoms,
                "n_transforms": sym.n_transforms,
                "atom_count": n_atoms,
            },
        )
        relaxation = sim.relax_clashes()
        record = manager.register_session(
            sim,
            default_channel="force-magnitude",
            default_palette="plasma",
        )
        metrics = sim.get_metrics()
        metrics["session_id"] = record.session_id
        return {
            "session_id": record.session_id,
            "title": sym.title,
            "n_asu_atoms": sym.n_asu_atoms,
            "n_transforms": sym.n_transforms,
            "n_atoms": n_atoms,
            "relaxation": relaxation,
            "metrics": metrics,
        }

    @application.post("/api/sessions/from-structure")
    async def create_session_from_structure(payload: CreateFromStructurePayload) -> Dict[str, Any]:
        sys, meta = _build_system_from_structure_payload(payload.structure)
        sim = SimulationSession(
            sys,
            mode=payload.mode,
            dt=payload.dt,
            temperature=payload.temperature,
            rcut=1.0,
            alpha_dsf=0.25 if np.any(sys.charges != 0.0) else 0.0,
            structure_metadata=meta,
        )
        minimization = sim.minimize()
        record = manager.register_session(
            sim,
            default_channel="force-magnitude",
            default_palette="plasma",
        )
        metrics = sim.get_metrics()
        metrics["session_id"] = record.session_id
        return {
            "session_id": record.session_id,
            "title": payload.structure.title,
            "minimization": minimization,
            "metrics": metrics,
        }

    @application.post("/api/sessions/upload")
    async def upload_structure_file(
        request: Request,
        filename: str = Query("uploaded_structure.pdb"),
        mode: str = Query("metal_verlet"),
        dt: float = Query(0.002),
        temperature: float = Query(300.0),
        storage_mode: str = Query("auto"),
    ) -> Dict[str, Any]:
        """Ingest raw structure bytes (.pdb, .cif/.mmcif, .gro, .xyz, .data) without JSON size bottlenecks."""
        mode_sel = str(storage_mode or "auto").strip().lower()
        if mode_sel not in ("auto", "tryte-all-atom", "full", "float64-full"):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid storage_mode '{storage_mode}'; expected 'auto', 'tryte-all-atom', or 'float64-full'",
            )
        content_type = (request.headers.get("content-type") or "").lower()
        if "multipart/form-data" in content_type:
            form = await request.form()
            uploaded_files = []
            for _key, val in form.multi_items():
                if hasattr(val, "read"):
                    try:
                        data_bytes = await val.read()
                        if data_bytes:
                            uploaded_files.append((data_bytes, getattr(val, "filename", None) or filename))
                    finally:
                        if hasattr(val, "close"):
                            await val.close()
            if not uploaded_files:
                raise HTTPException(status_code=400, detail="Empty multipart upload. Attach at least one structure file.")
            try:
                if len(uploaded_files) == 1:
                    sys, meta = load_structure_from_bytes(uploaded_files[0][0], uploaded_files[0][1])
                else:
                    sys, meta = load_structure_pair_from_bytes(
                        uploaded_files[0][0],
                        uploaded_files[0][1],
                        uploaded_files[1][0],
                        uploaded_files[1][1],
                    )
            except Exception as e:
                names_str = ", ".join(fn for _, fn in uploaded_files[:2])
                raise HTTPException(status_code=400, detail=f"Failed to parse uploaded files ({names_str}): {e}")
        else:
            body = await request.body()
            if not body:
                raise HTTPException(status_code=400, detail="Empty request body. Upload raw file bytes.")
            try:
                sys, meta = load_structure_from_bytes(body, filename)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to parse structure file '{filename}': {e}")

        if mode_sel in ("tryte-all-atom", "full") or (mode_sel == "auto" and sys.n_atoms > 50_000):
            charges_f32 = sys.charges.astype(np.float32) if np.any(sys.charges != 0.0) else None
            sim: Any = TryteTileFullAtomSession(
                positions_nm=sys.positions.astype(np.float32),
                atomic_numbers=sys.names,
                charges=charges_f32,
                bonds=sys.bonds,
                box_nm=sys.box_lengths,
                mode=mode,
                dt=dt,
                temperature=temperature,
                structure_metadata=meta,
            )
        else:
            sim = SimulationSession(
                sys,
                mode=mode,
                dt=dt,
                temperature=temperature,
                rcut=1.0,
                alpha_dsf=0.25 if np.any(sys.charges != 0.0) else 0.0,
                structure_metadata=meta,
            )
        minimization = sim.minimize()
        record = manager.register_session(
            sim,
            default_channel="force-magnitude",
            default_palette="plasma",
        )
        metrics = sim.get_metrics()
        metrics["session_id"] = record.session_id
        return {
            "session_id": record.session_id,
            "title": meta.get("title", filename),
            "format": meta.get("format", "unknown"),
            "coords_synthesized": bool(meta.get("coords_synthesized", False)),
            "num_atoms": sys.n_atoms,
            "num_bonds": len(sys.bonds),
            "storage_mode": metrics["storage_mode"],
            "minimization": minimization,
            "metrics": metrics,
        }

    @application.get("/api/sessions/{sid}/metrics")
    async def session_metrics(sid: str) -> Dict[str, Any]:
        record = _require_session(manager, sid)
        metrics = record.session.get_metrics()
        metrics["session_id"] = sid
        return metrics

    @application.post("/api/sessions/{sid}/steer")
    async def steer_session(sid: str, payload: SteerPayload) -> Dict[str, Any]:
        record = _require_session(manager, sid)
        if payload.active_channel is not None:
            if payload.active_channel not in PHYSICS_CHANNEL_NAMES:
                raise HTTPException(status_code=400, detail=f"Unknown channel '{payload.active_channel}'")
            record.active_channel = payload.active_channel
        if payload.palette is not None:
            if payload.palette not in PALETTE_NAMES:
                raise HTTPException(status_code=400, detail=f"Unknown palette '{payload.palette}'")
            record.palette = payload.palette

        metrics = record.session.steer(
            temperature=payload.temperature,
            efield_z=payload.efield_z,
            gcmc_mu=payload.gcmc_mu,
            strain_rate=payload.strain_rate,
            active_channel=payload.active_channel,
        )
        metrics["session_id"] = sid
        manager.push_current_frame(record, metrics=metrics, reversed_step=False)
        return metrics

    @application.post("/api/sessions/{sid}/scrub")
    async def scrub_session(sid: str, payload: ScrubPayload) -> Dict[str, Any]:
        record = _require_session(manager, sid)
        metrics, was_reverse = record.session.scrub_to_step(payload.target_step)
        metrics["session_id"] = sid
        manager.push_current_frame(record, metrics=metrics, reversed_step=was_reverse)
        return {"metrics": metrics, "reversed": was_reverse}

    @application.post("/api/sessions/{sid}/step")
    async def step_session(sid: str, payload: Optional[StepPayload] = None) -> Dict[str, Any]:
        record = _require_session(manager, sid)
        n_steps = payload.n_steps if payload is not None else 1
        metrics = record.session.step(n_steps=n_steps)
        metrics["session_id"] = sid
        manager.push_current_frame(record, metrics=metrics, reversed_step=False)
        return metrics

    @application.post("/api/sessions/{sid}/relax")
    async def relax_session(sid: str, payload: Optional[RelaxPayload] = None) -> Dict[str, Any]:
        record = _require_session(manager, sid)
        p = payload or RelaxPayload()
        report = record.session.relax_clashes(
            max_steps=p.max_steps,
            max_force_cap=p.max_force_cap,
            alpha_sc=p.alpha_sc,
            lam=p.lam,
        )
        metrics = record.session.get_metrics()
        metrics["session_id"] = sid
        manager.push_current_frame(record, metrics=metrics, reversed_step=False)
        return {"session_id": sid, "relaxation": report, "metrics": metrics}

    @application.post("/api/sessions/{sid}/multiscale")
    async def set_session_multiscale(sid: str, payload: MultiscalePayload) -> Dict[str, Any]:
        record = _require_session(manager, sid)
        if len(payload.center_ang) != 3:
            raise HTTPException(status_code=400, detail="center_ang must be a 3-element list [x, y, z]")
        summary = record.session.set_focus_zone(
            center_ang=payload.center_ang,
            aa_radius_ang=payload.aa_radius_ang,
            cg_radius_ang=payload.cg_radius_ang,
        )
        metrics = record.session.get_metrics()
        metrics["session_id"] = sid
        manager.push_current_frame(record, metrics=metrics, reversed_step=False)
        return {"session_id": sid, "multiscale": summary, "metrics": metrics}

    @application.post("/api/sessions/{sid}/steer-atom")
    async def steer_session_atom(sid: str, payload: SteerAtomPayload) -> Dict[str, Any]:
        record = _require_session(manager, sid)
        if payload.target_ang is not None and len(payload.target_ang) != 3:
            raise HTTPException(status_code=400, detail="target_ang must be a 3-element list [x, y, z]")
        try:
            smd = record.session.steer_atom(
                atom_index=payload.atom_index,
                target_ang=payload.target_ang,
                k_spring=payload.k_spring,
            )
        except IndexError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        metrics = record.session.get_metrics()
        metrics["session_id"] = sid
        manager.push_current_frame(record, metrics=metrics, reversed_step=False)
        return {"session_id": sid, "smd": smd, "metrics": metrics}

    @application.post("/api/sessions/{sid}/custom-physics")
    async def apply_custom_physics(sid: str, payload: CustomPhysicsPayload) -> Dict[str, Any]:
        record = _require_session(manager, sid)
        if payload.r_max <= payload.r_min:
            raise HTTPException(status_code=400, detail="r_max must be greater than r_min")
        try:
            evaluator = compile_safe_radial_expr(payload.potential_expr)
            r_grid = np.linspace(payload.r_min, payload.r_max, payload.n_points, dtype=np.float64)
            default_params = {
                "De": 480.0, "a": 21.0, "r0": 0.142, "eps": 0.997, "sig": 0.34,
                "A": 100.0, "kappa": 2.0, "q1": 1.0, "q2": -1.0, "lambda_": 1.0,
                "k": 1000.0, "C6": 0.002, "C12": 1e-6, "alpha": 12.0, "beta": 6.0,
            }
            params = {**default_params, **(payload.params or {})}
            u_grid = evaluator(r_grid, params)
            if not np.all(np.isfinite(u_grid)):
                raise ValueError("Potential expression produced non-finite values over [r_min, r_max]")
            spline = CubicSplineRadialTable(r_grid, u_grid, mode="lfs")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid custom potential: {exc}") from exc

        sim = record.session
        with sim._lock:
            sim.custom_expr = payload.potential_expr
            sim.custom_params = params
            sim._custom_spline = spline
            sim.mode = "custom_spline"
            sim.rcut = float(payload.r_max)
            sim.temperature = float(payload.temperature)
            sim.efield_z = float(payload.efield_z)
            sim.gcmc_mu = float(payload.gcmc_mu)
            sim.strain_rate = float(payload.strain_rate)
            if payload.dt is not None:
                sim.dt = float(payload.dt)
            sim._evaluate_state()

        metrics = sim.get_metrics()
        metrics["session_id"] = sid
        manager.push_current_frame(record, metrics=metrics, reversed_step=False)
        _, du_vals = spline.eval_array(r_grid)
        return {
            "ok": True,
            "metrics": metrics,
            "r_grid": r_grid.tolist(),
            "u_grid": u_grid.tolist(),
            "force_grid": (-du_vals).tolist(),
        }

    @application.get("/api/sessions/{sid}/atom/{idx}")
    async def inspect_atom(sid: str, idx: int) -> Dict[str, Any]:
        record = _require_session(manager, sid)
        try:
            return record.session.inspect_atom(idx)
        except IndexError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @application.get("/api/sessions/{sid}/structure.cif")
    async def get_structure_cif(
        sid: str,
        channel: Optional[str] = Query(None),
    ) -> Response:
        record = _require_session(manager, sid)
        cif_text = build_mmcif_from_session(record.session, channel=channel or record.active_channel)
        return Response(
            content=cif_text,
            media_type="chemical/x-cif",
            headers={"Content-Disposition": f'attachment; filename="{sid}.cif"'},
        )

    @application.get("/api/sessions/{sid}/export/mvsj")
    async def get_export_mvsj(
        sid: str,
        channel: Optional[str] = Query(None),
        palette: Optional[str] = Query(None),
    ) -> JSONResponse:
        if not _HAS_MVS_EXPORTER or export_mvsj is None:
            raise HTTPException(status_code=501, detail="MolViewSpec export unavailable: 'molviewspec' is not installed")
        record = _require_session(manager, sid)
        ch = channel or record.active_channel
        pal = palette or record.palette
        try:
            mvsj_dict = export_mvsj(
                record.session,
                channel=ch,
                palette=pal,
                title=f"Biology-MM-Studio ({sid})",
                cif_url=f"/api/sessions/{sid}/structure.cif",
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return JSONResponse(content=mvsj_dict)

    @application.get("/api/sessions/{sid}/export/mvsx")
    async def get_export_mvsx(
        sid: str,
        channel: Optional[str] = Query(None),
        palette: Optional[str] = Query(None),
    ) -> Response:
        if not _HAS_MVS_EXPORTER or export_mvsx_bytes is None:
            raise HTTPException(status_code=501, detail="MolViewSpec export unavailable: 'molviewspec' is not installed")
        record = _require_session(manager, sid)
        ch = channel or record.active_channel
        pal = palette or record.palette
        try:
            data = export_mvsx_bytes(
                record.session,
                channel=ch,
                palette=pal,
                title=f"Biology-MM-Studio ({sid})",
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return Response(
            content=data,
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="biology-mm-{sid}.mvsx"'},
        )

    @application.post("/api/code/run")
    async def run_code(payload: CodeRunPayload) -> Dict[str, Any]:
        active_sys = None
        active_meta = None
        if payload.active_structure is not None:
            active_sys, active_meta = _build_system_from_structure_payload(payload.active_structure)

        result, sim = execute_user_code(
            payload.code,
            timeout_s=payload.timeout_s,
            active_structure=active_sys,
            active_structure_metadata=active_meta,
        )
        if sim is not None:
            if sim.structure_metadata is None and active_meta is not None and active_sys is not None and sim.n_real == active_sys.n_atoms:
                sim.structure_metadata = dict(active_meta)
            record = manager.register_session(sim)
            result["session_id"] = record.session_id
            if result.get("metrics") is not None:
                result["metrics"]["session_id"] = record.session_id
        return result

    @application.websocket("/ws/stream/{sid}")
    async def ws_stream(websocket: WebSocket, sid: str) -> None:
        await manager.handle_websocket(websocket, sid)

    index_html = resolved_dist / "index.html"
    if index_html.is_file():
        application.mount("/", StaticFiles(directory=str(resolved_dist), html=True), name="studio-spa")

    return application


app = create_app()
