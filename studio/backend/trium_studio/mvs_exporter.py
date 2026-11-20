"""MolViewSpec (MVSJ / MVSX) exporter for Biology-MM-Studio sessions."""

from __future__ import annotations

import io
import json
import zipfile
from contextlib import nullcontext
from typing import Any, Dict, List, Sequence, Tuple

import molviewspec as mvs
import numpy as np

from .cif_builder import _safe_element_symbol, build_mmcif_from_session

# RGB color stops (0..255) for all 6 Biology-MM-Studio palettes
_PALETTE_STOPS: Dict[str, List[Tuple[int, int, int]]] = {
    "plasma": [
        (13, 8, 135),
        (126, 3, 168),
        (204, 71, 120),
        (248, 149, 64),
        (240, 249, 33),
    ],
    "thermal-div": [
        (33, 102, 172),
        (146, 197, 222),
        (247, 247, 247),
        (244, 165, 130),
        (178, 24, 43),
    ],
    "electrostatic-bwr": [
        (43, 92, 224),
        (158, 188, 245),
        (245, 245, 245),
        (244, 152, 138),
        (214, 39, 40),
    ],
    "bond-viridis": [
        (68, 1, 84),
        (59, 82, 139),
        (33, 145, 140),
        (94, 201, 98),
        (253, 231, 37),
    ],
    "stress-inferno": [
        (0, 0, 4),
        (87, 16, 110),
        (188, 55, 84),
        (249, 142, 9),
        (252, 255, 164),
    ],
    "ternary-triad": [
        (59, 130, 246),  # -1: blocked / water
        (148, 163, 184),  # 0: framework / empty
        (245, 158, 11),  # +1: adsorbed / active guest
    ],
}


def _interpolate_palette_hex(t_values: np.ndarray, palette: str) -> List[str]:
    """Map normalized scalar values in ``[0, 1]`` to ``#rrggbb`` hex colors."""
    stops = _PALETTE_STOPS.get(palette)
    if stops is None:
        raise ValueError(f"Unknown palette '{palette}'; expected one of {tuple(_PALETTE_STOPS.keys())}")

    arr = np.asarray(stops, dtype=np.float64)
    n_stops = arr.shape[0]
    t = np.clip(np.asarray(t_values, dtype=np.float64), 0.0, 1.0)
    scaled = t * float(n_stops - 1)
    idx0 = np.floor(scaled).astype(np.int64)
    idx0 = np.clip(idx0, 0, n_stops - 2)
    idx1 = idx0 + 1
    frac = (scaled - idx0)[:, None]
    rgb = np.round(arr[idx0] * (1.0 - frac) + arr[idx1] * frac).astype(np.int64)
    rgb = np.clip(rgb, 0, 255)
    return [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in rgb]


def build_annotations_from_session(
    session,
    channel: str = "force-magnitude",
    palette: str = "plasma",
) -> List[Dict[str, Any]]:
    """Build per-atom MVS JSON annotation rows for active atoms in ``session``.

    Each row carries ``atom_id`` (1-based ``_atom_site.id``), ``atom_index`` (0-based),
    ``slot_index``, ``element``, ``channel``, ``scalar``, and ``color`` (``#rrggbb``).
    """
    if palette not in _PALETTE_STOPS:
        raise ValueError(f"Unknown palette '{palette}'; expected one of {tuple(_PALETTE_STOPS.keys())}")

    lock = getattr(session, "_lock", None)
    ctx = lock if lock is not None else nullcontext()

    with ctx:
        scalars_all = np.asarray(session.compute_scalar_channel(channel), dtype=np.float64)
        active_mask = np.asarray(session.active_mask, dtype=bool)
        active_slots = np.flatnonzero(active_mask)
        atomic_numbers = np.asarray(session.atomic_numbers, dtype=np.int64)
        names = list(session.names)

    if active_slots.size == 0:
        return []

    active_scalars = scalars_all[active_slots]
    if palette == "ternary-triad" or channel == "ternary-state":
        # Map {-1, 0, +1} -> {0.0, 0.5, 1.0}
        t_vals = np.clip((active_scalars + 1.0) * 0.5, 0.0, 1.0)
    else:
        s_min = float(np.min(active_scalars))
        s_max = float(np.max(active_scalars))
        span = s_max - s_min
        if span <= 1e-12:
            t_vals = np.full(active_scalars.shape, 0.5, dtype=np.float64)
        else:
            t_vals = (active_scalars - s_min) / span

    hex_colors = _interpolate_palette_hex(t_vals, palette)

    rows: List[Dict[str, Any]] = []
    for idx, slot in enumerate(active_slots):
        slot_int = int(slot)
        rows.append(
            {
                "atom_id": idx + 1,
                "atom_index": idx,
                "slot_index": slot_int,
                "element": _safe_element_symbol(names[slot_int], int(atomic_numbers[slot_int])),
                "channel": channel,
                "scalar": float(active_scalars[idx]),
                "color": hex_colors[idx],
            }
        )
    return rows


def export_mvsj(
    session,
    channel: str = "force-magnitude",
    palette: str = "plasma",
    title: str = "Biology-MM Scene",
    cif_url: str = "structure.cif",
) -> dict:
    """Build a Mol* v6-compatible MVSJ state dictionary using ``molviewspec``."""
    # Validate channel and palette early so invalid inputs fail fast
    if palette not in _PALETTE_STOPS:
        raise ValueError(f"Unknown palette '{palette}'; expected one of {tuple(_PALETTE_STOPS.keys())}")
    lock = getattr(session, "_lock", None)
    ctx = lock if lock is not None else nullcontext()
    with ctx:
        _ = session.compute_scalar_channel(channel)
        step_count = int(getattr(session, "step_count", 0))
        sim_time_ps = float(getattr(session, "sim_time_ps", 0.0))

    builder = mvs.create_builder()
    builder.canvas(background_color="#0b101b")
    structure = (
        builder.download(url=cif_url)
        .parse(format="mmcif")
        .model_structure()
    )
    component = structure.component(selector="all")
    representation = component.representation(type="ball_and_stick")
    representation.color_from_uri(
        uri="annotations.json",
        format="json",
        schema="atom",
        field_name="color",
    )

    description = (
        f"Biology-MM-Studio | channel={channel} | palette={palette} "
        f"| step={step_count} | time_ps={sim_time_ps:.4f}"
    )
    state = builder.get_state(title=title, description=description)
    return state.model_dump(mode="json", exclude_none=True)


def export_mvsx_bytes(
    session,
    channel: str = "force-magnitude",
    palette: str = "plasma",
    title: str = "Biology-MM Scene",
) -> bytes:
    """Export a complete ``.mvsx`` ZIP archive containing ``index.mvsj``, ``structure.cif``, and ``annotations.json``."""
    mvsj_dict = export_mvsj(
        session,
        channel=channel,
        palette=palette,
        title=title,
        cif_url="structure.cif",
    )
    cif_text = build_mmcif_from_session(session)
    annotations = build_annotations_from_session(session, channel=channel, palette=palette)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("index.mvsj", json.dumps(mvsj_dict, indent=2))
        zf.writestr("structure.cif", cif_text)
        zf.writestr("annotations.json", json.dumps(annotations, indent=2))
    return buf.getvalue()
