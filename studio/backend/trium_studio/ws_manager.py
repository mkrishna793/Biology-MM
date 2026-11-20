"""Background simulation runner and TriumStream v1 WebSocket manager."""

from __future__ import annotations

import asyncio
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from fastapi import WebSocket, WebSocketDisconnect

from .engine_core import (
    PHYSICS_CHANNEL_NAMES,
    LatestFrameBuffer,
    SimulationSession,
)

PALETTE_NAMES: Tuple[str, ...] = (
    "plasma",
    "thermal-div",
    "electrostatic-bwr",
    "bond-viridis",
    "stress-inferno",
    "ternary-triad",
)


@dataclass
class ManagedSession:
    """State wrapper around a `SimulationSession` for REST + WebSocket streaming."""

    session_id: str
    session: SimulationSession
    active_channel: str = "force-magnitude"
    palette: str = "plasma"
    playing: bool = False
    show_overlays: bool = False
    show_volume: bool = False
    camera_roi: Optional[Tuple[List[float], float, int]] = None
    frame_buffer: LatestFrameBuffer = field(default_factory=LatestFrameBuffer)
    lock: threading.RLock = field(default_factory=threading.RLock)
    stop_event: threading.Event = field(default_factory=threading.Event)
    worker_thread: Optional[threading.Thread] = None
    last_error: Optional[str] = None
    created_at: float = field(default_factory=time.time)


class SessionManager:
    """Thread-safe registry and background stream coordinator for Studio sessions."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._sessions: Dict[str, ManagedSession] = {}

    def register_session(
        self,
        session: SimulationSession,
        session_id: Optional[str] = None,
        default_channel: str = "force-magnitude",
        default_palette: str = "plasma",
    ) -> ManagedSession:
        sid = session_id or uuid.uuid4().hex[:12]
        ch = default_channel if default_channel in PHYSICS_CHANNEL_NAMES else "force-magnitude"
        pal = default_palette if default_palette in PALETTE_NAMES else "plasma"
        session.active_channel = PHYSICS_CHANNEL_NAMES.index(ch)
        record = ManagedSession(
            session_id=sid,
            session=session,
            active_channel=ch,
            palette=pal,
        )
        with self._lock:
            self._sessions[sid] = record
        return record

    def get_session(self, session_id: str) -> Optional[ManagedSession]:
        with self._lock:
            return self._sessions.get(session_id)

    def remove_session(self, session_id: str) -> bool:
        with self._lock:
            record = self._sessions.pop(session_id, None)
        if record is None:
            return False
        record.playing = False
        record.stop_event.set()
        if record.worker_thread is not None and record.worker_thread.is_alive():
            record.worker_thread.join(timeout=0.5)
        record.session.close()
        return True

    def close_all(self) -> None:
        with self._lock:
            sids = list(self._sessions.keys())
        for sid in sids:
            self.remove_session(sid)

    def _ensure_worker(self, record: ManagedSession) -> None:
        with record.lock:
            if record.worker_thread is not None and record.worker_thread.is_alive():
                return
            record.stop_event.clear()
            t = threading.Thread(
                target=self._simulation_loop,
                args=(record,),
                name=f"sim-runner-{record.session_id}",
                daemon=True,
            )
            record.worker_thread = t
            t.start()

    @staticmethod
    def _steps_per_tick(session: Any) -> int:
        n_atoms = int(getattr(session, "n_atoms", getattr(session, "n_real", 0)))
        if n_atoms > 500_000:
            return 1
        if n_atoms > 50_000:
            return 2
        return 10 if getattr(session, "mode", "") == "metal_verlet" else 2

    def _simulation_loop(self, record: ManagedSession) -> None:
        while not record.stop_event.is_set():
            if not record.playing:
                time.sleep(0.02)
                continue
            t0 = time.perf_counter()
            try:
                steps = self._steps_per_tick(record.session)
                metrics = record.session.step(n_steps=steps)
                frame_pkt = record.session.encode_frame_packet(
                    channel_name=record.active_channel,
                    camera_roi=record.camera_roi,
                )
                vol_pkt = record.session.encode_volume_packet() if record.show_volume else None
                record.frame_buffer.push(frame_pkt, metrics, volume_packet=vol_pkt)
            except Exception as exc:
                with record.lock:
                    record.last_error = str(exc) or repr(exc)
                    record.playing = False
            elapsed = time.perf_counter() - t0
            sleep_s = max(0.005, 0.033 - elapsed)
            time.sleep(sleep_s)

    def push_current_frame(
        self,
        record: ManagedSession,
        metrics: Optional[Dict[str, Any]] = None,
        reversed_step: bool = False,
        include_volume: bool = False,
    ) -> None:
        m = metrics if metrics is not None else record.session.get_metrics()
        frame_pkt = record.session.encode_frame_packet(
            channel_name=record.active_channel,
            reversed_step=reversed_step,
            camera_roi=record.camera_roi,
        )
        vol_pkt = record.session.encode_volume_packet() if (include_volume or record.show_volume) else None
        record.frame_buffer.push(frame_pkt, m, volume_packet=vol_pkt)

    async def handle_websocket(self, websocket: WebSocket, session_id: str) -> None:
        await websocket.accept()
        record = self.get_session(session_id)
        if record is None:
            await websocket.send_json({"type": "error", "error": f"Unknown session '{session_id}'"})
            await websocket.close(code=4404)
            return

        # Send initial topology, frame, overlays, and metrics snapshot
        try:
            topo_pkt = await asyncio.to_thread(record.session.encode_topology_packet)
            await websocket.send_bytes(topo_pkt)

            frame_pkt = await asyncio.to_thread(
                record.session.encode_frame_packet,
                record.active_channel,
                False,
                record.camera_roi,
            )
            await websocket.send_bytes(frame_pkt)

            overlay_pkt = await asyncio.to_thread(record.session.encode_overlay_packet)
            await websocket.send_bytes(overlay_pkt)

            init_metrics = await asyncio.to_thread(record.session.get_metrics)
            init_metrics["session_id"] = session_id
            init_metrics["dropped_frames"] = 0
            await websocket.send_json({"type": "metrics", "metrics": init_metrics})
            await websocket.send_json(
                {
                    "type": "state",
                    "playing": record.playing,
                    "active_channel": record.active_channel,
                    "palette": record.palette,
                }
            )
        except Exception as exc:
            await websocket.send_json({"type": "error", "error": str(exc)})
            await websocket.close(code=1011)
            return

        send_task = asyncio.create_task(self._sender_loop(websocket, record))
        recv_task = asyncio.create_task(self._receiver_loop(websocket, record))
        done, pending = await asyncio.wait(
            {send_task, recv_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
        for task in done:
            # Consume exceptions cleanly on disconnect
            try:
                task.result()
            except (WebSocketDisconnect, asyncio.CancelledError):
                pass
            except Exception:
                pass

    async def _sender_loop(self, websocket: WebSocket, record: ManagedSession) -> None:
        while True:
            err: Optional[str] = None
            with record.lock:
                if record.last_error is not None:
                    err = record.last_error
                    record.last_error = None
            if err is not None:
                await websocket.send_json({"type": "error", "error": err})
                await websocket.send_json(
                    {
                        "type": "state",
                        "playing": record.playing,
                        "active_channel": record.active_channel,
                        "palette": record.palette,
                    }
                )
            frame_pkt, metrics, vol_pkt, dropped = record.frame_buffer.pop_latest()
            if frame_pkt is not None:
                await websocket.send_bytes(frame_pkt)
                if vol_pkt is not None:
                    await websocket.send_bytes(vol_pkt)
                if record.show_overlays:
                    ov_pkt = await asyncio.to_thread(record.session.encode_overlay_packet)
                    await websocket.send_bytes(ov_pkt)
                if metrics is not None:
                    payload = dict(metrics)
                    payload["session_id"] = record.session_id
                    payload["dropped_frames"] = dropped
                    await websocket.send_json({"type": "metrics", "metrics": payload})
            await asyncio.sleep(0.016)

    async def _receiver_loop(self, websocket: WebSocket, record: ManagedSession) -> None:
        while True:
            msg = await websocket.receive_json()
            if not isinstance(msg, dict):
                await websocket.send_json({"type": "error", "error": "Expected JSON object message"})
                continue

            kind = str(msg.get("type") or msg.get("action") or "").strip()
            try:
                if kind == "play":
                    with record.lock:
                        record.last_error = None
                        record.playing = True
                    self._ensure_worker(record)
                    await websocket.send_json(
                        {
                            "type": "state",
                            "playing": True,
                            "active_channel": record.active_channel,
                            "palette": record.palette,
                        }
                    )
                elif kind == "pause":
                    record.playing = False
                    await websocket.send_json(
                        {
                            "type": "state",
                            "playing": False,
                            "active_channel": record.active_channel,
                            "palette": record.palette,
                        }
                    )
                elif kind == "step":
                    n_steps = max(1, int(msg.get("n_steps", 1)))
                    metrics = await asyncio.to_thread(record.session.step, n_steps)
                    await asyncio.to_thread(self.push_current_frame, record, metrics, False)
                elif kind == "scrub":
                    target_step = int(msg.get("target_step", msg.get("step", 0)))
                    metrics, was_rev = await asyncio.to_thread(record.session.scrub_to_step, target_step)
                    await asyncio.to_thread(self.push_current_frame, record, metrics, was_rev)
                elif kind == "steer":
                    if "palette" in msg and msg["palette"] in PALETTE_NAMES:
                        record.palette = str(msg["palette"])
                    ch = msg.get("active_channel")
                    if ch in PHYSICS_CHANNEL_NAMES:
                        record.active_channel = str(ch)
                    metrics = await asyncio.to_thread(
                        record.session.steer,
                        temperature=msg.get("temperature"),
                        efield_z=msg.get("efield_z"),
                        gcmc_mu=msg.get("gcmc_mu"),
                        strain_rate=msg.get("strain_rate"),
                        active_channel=ch,
                    )
                    await asyncio.to_thread(self.push_current_frame, record, metrics, False)
                elif kind == "set_channel":
                    ch = str(msg.get("channel", ""))
                    if ch not in PHYSICS_CHANNEL_NAMES:
                        await websocket.send_json({"type": "error", "error": f"Unknown channel '{ch}'"})
                        continue
                    record.active_channel = ch
                    record.session.steer(active_channel=ch)
                    await asyncio.to_thread(self.push_current_frame, record, None, False)
                    await websocket.send_json(
                        {
                            "type": "state",
                            "playing": record.playing,
                            "active_channel": record.active_channel,
                            "palette": record.palette,
                        }
                    )
                elif kind == "set_palette":
                    pal = str(msg.get("palette", ""))
                    if pal not in PALETTE_NAMES:
                        await websocket.send_json({"type": "error", "error": f"Unknown palette '{pal}'"})
                        continue
                    record.palette = pal
                    await websocket.send_json(
                        {
                            "type": "state",
                            "playing": record.playing,
                            "active_channel": record.active_channel,
                            "palette": record.palette,
                        }
                    )
                elif kind == "request_volume":
                    ml = int(msg.get("ml", 27))
                    record.show_volume = bool(msg.get("enabled", True))
                    vol_pkt = await asyncio.to_thread(record.session.encode_volume_packet, ml)
                    await websocket.send_bytes(vol_pkt)
                elif kind == "request_overlays":
                    max_v = int(msg.get("max_vectors", 2000))
                    if "enabled" in msg:
                        record.show_overlays = bool(msg["enabled"])
                    ov_pkt = await asyncio.to_thread(record.session.encode_overlay_packet, max_v)
                    await websocket.send_bytes(ov_pkt)
                elif kind == "camera_roi":
                    center = msg.get("center_ang")
                    if center is None:
                        record.camera_roi = None
                    else:
                        radius = float(msg.get("radius_ang", 30.0))
                        max_atoms = int(msg.get("max_atoms", 262_144))
                        record.camera_roi = ([float(c) for c in center], radius, max_atoms)
                    await asyncio.to_thread(self.push_current_frame, record, None, False)
                elif kind == "inspect_atom":
                    idx = int(msg["index"])
                    atom = await asyncio.to_thread(record.session.inspect_atom, idx)
                    await websocket.send_json({"type": "atom_inspection", "atom": atom})
                elif kind == "set_focus_zone":
                    center = msg.get("center_ang")
                    if not isinstance(center, (list, tuple)) or len(center) != 3:
                        await websocket.send_json({"type": "error", "error": "set_focus_zone requires center_ang=[x, y, z]"})
                        continue
                    aa_r = float(msg.get("aa_radius_ang", 15.0))
                    cg_r = float(msg.get("cg_radius_ang", 35.0))
                    summary = await asyncio.to_thread(record.session.set_focus_zone, center, aa_r, cg_r)
                    await asyncio.to_thread(self.push_current_frame, record, None, False)
                    await websocket.send_json({"type": "multiscale", "multiscale": summary})
                elif kind == "relax_clashes":
                    max_steps = int(msg.get("max_steps", 40))
                    max_force_cap = float(msg.get("max_force_cap", 5000.0))
                    alpha_sc = float(msg.get("alpha_sc", 0.5))
                    lam = float(msg.get("lam", 0.5))
                    report = await asyncio.to_thread(
                        record.session.relax_clashes, max_steps, max_force_cap, alpha_sc, lam
                    )
                    await asyncio.to_thread(self.push_current_frame, record, None, False)
                    await websocket.send_json({"type": "relax_result", "relaxation": report})
                elif kind == "steer_atom":
                    atom_idx = msg.get("atom_index")
                    target_ang = msg.get("target_ang")
                    k_spring = float(msg.get("k_spring", 500.0))
                    smd = await asyncio.to_thread(record.session.steer_atom, atom_idx, target_ang, k_spring)
                    await asyncio.to_thread(self.push_current_frame, record, None, False)
                    await websocket.send_json({"type": "smd_state", "smd": smd})
                elif kind == "set_delta_threshold":
                    thr = float(msg.get("threshold_ang", 0.05))
                    enabled = msg.get("enabled")
                    res = await asyncio.to_thread(record.session.set_delta_threshold, thr, enabled)
                    await websocket.send_json({"type": "delta_threshold", **res})
                else:
                    await websocket.send_json({"type": "error", "error": f"Unsupported WS message type '{kind}'"})
            except Exception as exc:
                await websocket.send_json({"type": "error", "error": str(exc)})
