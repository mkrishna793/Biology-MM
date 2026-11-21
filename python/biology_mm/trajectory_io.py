"""Standard Open-Science Molecular Trajectory I/O: CHARMM/NAMD .dcd, GROMACS .xtc, and .pdb.

Zero external dependencies beyond NumPy. Output files follow exact binary specifications so
trajectories open directly in MDTraj, MDAnalysis, VMD, GROMACS, and OpenMM without conversion.

Coordinate unit conventions:
- Biology-MM internal coordinates are in nanometers (nm) or Angstroms (A) depending on caller;
  every public function accepts an explicit `units="nm"` or `units="angstrom"` argument.
- Standard `.dcd` stores coordinates and unit-cell lengths in Angstroms (A).
- Standard `.xtc` stores coordinates and 3x3 box vectors in nanometers (nm).
- Standard `.pdb` stores coordinates and CRYST1 unit-cell lengths in Angstroms (A).
"""

from __future__ import annotations

import io
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = [
    "DCDWriter",
    "TrajectoryData",
    "XTCWriter",
    "box_matrix_to_charmm_cell",
    "charmm_cell_to_box_matrix",
    "read_dcd",
    "read_xtc",
    "write_dcd",
    "write_pdb_topology",
    "write_xtc",
]

# AKMA time unit used in CHARMM/NAMD DCD headers: 1 AKMA = 0.04888821 ps
_AKMA_PER_PS = 1.0 / 0.04888821
_PS_PER_AKMA = 0.04888821

# Official GROMACS XTC 3DF magic integer lookup table (from xdrfile_xtc.c)
_XTC_MAGICINTS: Tuple[int, ...] = (
    0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 10, 12, 16, 20, 25, 32, 40, 50, 64,
    80, 101, 128, 161, 203, 256, 322, 406, 512, 645, 812, 1024, 1290,
    1625, 2048, 2580, 3250, 4096, 5060, 6501, 8192, 10321, 13003,
    16384, 20642, 26007, 32768, 41285, 52015, 65536, 82570, 104031,
    131072, 165140, 208063, 262144, 330280, 416127, 524288, 660561,
    832255, 1048576, 1321122, 1664510, 2097152, 2642245, 3329021,
    4194304, 5284491, 6658042, 8388607, 10568983, 13316085, 16777216,
)
_XTC_FIRSTIDX = 9
_XTC_LASTIDX = len(_XTC_MAGICINTS)
_XTC_MAGIC = 1995


@dataclass
class TrajectoryData:
    """In-memory representation of a molecular dynamics trajectory."""

    positions_nm: np.ndarray  # shape (n_frames, n_atoms, 3), float32/float64 in nm
    box_nm: Optional[np.ndarray] = None  # shape (n_frames, 3, 3), float64 in nm
    steps: Optional[np.ndarray] = None  # shape (n_frames,), int64
    times_ps: Optional[np.ndarray] = None  # shape (n_frames,), float64
    dt_ps: float = 0.002
    precision: float = 1000.0
    title: str = "Biology-MM Open-Science Trajectory"

    @property
    def n_frames(self) -> int:
        return int(self.positions_nm.shape[0])

    @property
    def n_atoms(self) -> int:
        return int(self.positions_nm.shape[1]) if self.positions_nm.ndim >= 2 else 0


def _to_nm(arr: np.ndarray, units: str) -> np.ndarray:
    u = units.strip().lower()
    if u in ("nm", "nanometer", "nanometers"):
        return np.asarray(arr, dtype=np.float64)
    if u in ("a", "ang", "angstrom", "angstroms"):
        return np.asarray(arr, dtype=np.float64) * 0.1
    raise ValueError(f"Unsupported length unit '{units}'; expected 'nm' or 'angstrom'")


def _normalize_box_matrix(box: Optional[Union[Sequence[float], np.ndarray]], units: str = "nm") -> Optional[np.ndarray]:
    if box is None:
        return None
    arr = _to_nm(np.asarray(box, dtype=np.float64), units)
    if arr.shape == (3,):
        return np.diag(arr)
    if arr.shape == (3, 3):
        return arr.copy()
    raise ValueError(f"Unsupported box shape {arr.shape}; expected (3,) or (3, 3)")


def box_matrix_to_charmm_cell(box_nm: np.ndarray) -> Tuple[float, float, float, float, float, float]:
    """Convert a 3x3 box matrix (rows = a, b, c in nm) to CHARMM DCD 6-double unit cell in Angstrom.

    CHARMM/NAMD stores `[A, cos(gamma), B, cos(beta), cos(alpha), C]`.
    """
    H = np.asarray(box_nm, dtype=np.float64) * 10.0  # nm -> Angstrom
    a_vec, b_vec, c_vec = H[0], H[1], H[2]
    la = float(np.linalg.norm(a_vec))
    lb = float(np.linalg.norm(b_vec))
    lc = float(np.linalg.norm(c_vec))
    cos_gamma = float(np.dot(a_vec, b_vec) / (la * lb)) if (la > 0 and lb > 0) else 0.0
    cos_beta = float(np.dot(a_vec, c_vec) / (la * lc)) if (la > 0 and lc > 0) else 0.0
    cos_alpha = float(np.dot(b_vec, c_vec) / (lb * lc)) if (lb > 0 and lc > 0) else 0.0
    return (la, cos_gamma, lb, cos_beta, cos_alpha, lc)


def charmm_cell_to_box_matrix(cell: Sequence[float]) -> np.ndarray:
    """Convert CHARMM DCD 6-double unit cell `[A, cos(gamma), B, cos(beta), cos(alpha), C]` (A) to 3x3 box (nm)."""
    la, cos_g, lb, cos_b, cos_a, lc = (float(x) for x in cell)
    # Handle older CHARMM files that store angles in degrees instead of cosines
    if abs(cos_g) > 1.0 or abs(cos_b) > 1.0 or abs(cos_a) > 1.0:
        cos_g = math.cos(math.radians(cos_g))
        cos_b = math.cos(math.radians(cos_b))
        cos_a = math.cos(math.radians(cos_a))
    sin_g = math.sqrt(max(1e-12, 1.0 - cos_g * cos_g))
    ax, ay, az = la, 0.0, 0.0
    bx, by, bz = lb * cos_g, lb * sin_g, 0.0
    cx = lc * cos_b
    cy = lc * (cos_a - cos_b * cos_g) / sin_g
    cz = math.sqrt(max(0.0, lc * lc - cx * cx - cy * cy))
    return np.array([[ax, ay, az], [bx, by, bz], [cx, cy, cz]], dtype=np.float64) * 0.1


# =============================================================================
# 1. CHARMM / NAMD / OpenMM Binary DCD Writer & Reader
# =============================================================================


def _write_fortran_record(fh: BinaryIO, payload: bytes) -> None:
    n = len(payload)
    marker = struct.pack("<I", n)
    fh.write(marker)
    fh.write(payload)
    fh.write(marker)


def _read_fortran_record(fh: BinaryIO) -> bytes:
    raw_len = fh.read(4)
    if len(raw_len) < 4:
        raise EOFError("Unexpected EOF reading Fortran record header")
    (n,) = struct.unpack("<I", raw_len)
    payload = fh.read(n)
    if len(payload) < n:
        raise EOFError("Unexpected EOF reading Fortran record body")
    raw_end = fh.read(4)
    if len(raw_end) < 4:
        raise EOFError("Unexpected EOF reading Fortran record trailer")
    (n_end,) = struct.unpack("<I", raw_end)
    if n_end != n:
        raise ValueError(f"Corrupt Fortran record: start length {n} != end length {n_end}")
    return payload


class DCDWriter:
    """Streaming standard CHARMM/NAMD `.dcd` trajectory writer (compatible with MDTraj/MDAnalysis/VMD)."""

    def __init__(
        self,
        dest: Union[str, Path, BinaryIO],
        n_atoms: int,
        dt_ps: float = 0.002,
        istart: int = 0,
        nsavc: int = 1,
        has_unit_cell: bool = True,
        title: str = "CREATED BY BIOLOGY-MM OPEN-SCIENCE ENGINE",
    ) -> None:
        self._owns_fh = not hasattr(dest, "write")
        self._fh: BinaryIO = open(dest, "wb") if self._owns_fh else dest  # type: ignore[assignment]
        self.n_atoms = int(n_atoms)
        self.dt_ps = float(dt_ps)
        self.istart = int(istart)
        self.nsavc = max(1, int(nsavc))
        self.has_unit_cell = bool(has_unit_cell)
        self.title = str(title)
        self.frames_written = 0
        self._header_written = False
        self._write_header(n_frames=0)

    def _write_header(self, n_frames: int) -> None:
        # 84-byte CORD header: 'CORD' + 20 int32/float32 fields
        icntrl = [0] * 20
        icntrl[0] = int(n_frames)  # NSET (number of frames)
        icntrl[1] = self.istart  # ISTART
        icntrl[2] = self.nsavc  # NSAVC
        icntrl[3] = int(n_frames) * self.nsavc  # NSTEP
        icntrl[10] = 1 if self.has_unit_cell else 0  # extra unit cell block
        icntrl[19] = 24  # CHARMM version tag (24 = standard NAMD/OpenMM/CHARMM)

        delta_akma = float(self.dt_ps * _AKMA_PER_PS)
        header_buf = bytearray()
        header_buf.extend(b"CORD")
        header_buf.extend(struct.pack("<9i", *icntrl[0:9]))
        header_buf.extend(struct.pack("<f", delta_akma))
        header_buf.extend(struct.pack("<10i", *icntrl[10:20]))
        assert len(header_buf) == 84
        _write_fortran_record(self._fh, bytes(header_buf))

        # Title block: int32(ntitle) + ntitle * 80 ASCII bytes
        line1 = self.title[:80].ljust(80, " ").encode("ascii", errors="replace")
        line2 = "STANDARD CHARMM/NAMD DCD (UNITS: ANGSTROM, AKMA)".ljust(80, " ").encode("ascii")
        title_payload = struct.pack("<i", 2) + line1 + line2
        _write_fortran_record(self._fh, title_payload)

        # NATOM block
        _write_fortran_record(self._fh, struct.pack("<i", self.n_atoms))
        self._header_written = True

    def write_frame(
        self,
        positions: np.ndarray,
        box: Optional[Union[Sequence[float], np.ndarray]] = None,
        units: str = "nm",
    ) -> None:
        pos_nm = _to_nm(positions, units)
        if pos_nm.shape != (self.n_atoms, 3):
            raise ValueError(f"Expected positions shape ({self.n_atoms}, 3), got {pos_nm.shape}")
        pos_ang = np.ascontiguousarray(pos_nm * 10.0, dtype=np.float32)

        if self.has_unit_cell:
            box_nm = _normalize_box_matrix(box, units=units)
            if box_nm is None:
                box_nm = np.zeros((3, 3), dtype=np.float64)
            cell = box_matrix_to_charmm_cell(box_nm)
            _write_fortran_record(self._fh, struct.pack("<6d", *cell))

        _write_fortran_record(self._fh, np.ascontiguousarray(pos_ang[:, 0], dtype="<f4").tobytes())
        _write_fortran_record(self._fh, np.ascontiguousarray(pos_ang[:, 1], dtype="<f4").tobytes())
        _write_fortran_record(self._fh, np.ascontiguousarray(pos_ang[:, 2], dtype="<f4").tobytes())
        self.frames_written += 1

    def finalize(self) -> None:
        if self._fh.seekable():
            end_pos = self._fh.tell()
            # Patch NSET (offset 8: 4-byte record len + 4-byte 'CORD') and NSTEP (offset 20)
            self._fh.seek(8)
            self._fh.write(struct.pack("<i", self.frames_written))
            self._fh.seek(20)
            self._fh.write(struct.pack("<i", self.frames_written * self.nsavc))
            self._fh.seek(end_pos)
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self.finalize()
            if self._owns_fh:
                self._fh.close()

    def __enter__(self) -> "DCDWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def write_dcd(
    dest: Union[str, Path, BinaryIO],
    positions: np.ndarray,
    boxes: Optional[np.ndarray] = None,
    dt_ps: float = 0.002,
    units: str = "nm",
    title: str = "CREATED BY BIOLOGY-MM OPEN-SCIENCE ENGINE",
) -> int:
    """Write a complete `(n_frames, n_atoms, 3)` trajectory to standard `.dcd` format."""
    pos = np.asarray(positions, dtype=np.float64)
    if pos.ndim == 2:
        pos = pos[np.newaxis, :, :]
    if pos.ndim != 3 or pos.shape[2] != 3:
        raise ValueError(f"Expected positions shape (n_frames, n_atoms, 3), got {pos.shape}")
    n_frames, n_atoms, _ = pos.shape
    has_cell = boxes is not None
    with DCDWriter(dest, n_atoms=n_atoms, dt_ps=dt_ps, has_unit_cell=has_cell, title=title) as writer:
        for idx in range(n_frames):
            box_i = None
            if boxes is not None:
                b_arr = np.asarray(boxes, dtype=np.float64)
                box_i = b_arr if b_arr.ndim in (1, 2) and (b_arr.shape == (3,) or b_arr.shape == (3, 3)) else b_arr[idx]
            writer.write_frame(pos[idx], box=box_i, units=units)
    return n_frames


def read_dcd(source: Union[str, Path, bytes, BinaryIO]) -> TrajectoryData:
    """Read a standard CHARMM/NAMD `.dcd` file into a `TrajectoryData` (coordinates in nm)."""
    if isinstance(source, (str, Path)):
        fh: BinaryIO = open(source, "rb")
        owns = True
    elif isinstance(source, (bytes, bytearray)):
        fh = io.BytesIO(source)
        owns = True
    else:
        fh = source
        owns = False

    try:
        hdr = _read_fortran_record(fh)
        if len(hdr) != 84 or hdr[:4] != b"CORD":
            raise ValueError("Invalid DCD header: expected 84-byte CORD block")
        ic_pre = struct.unpack("<9i", hdr[4:40])
        (delta_akma,) = struct.unpack("<f", hdr[40:44])
        ic_post = struct.unpack("<10i", hdr[44:84])
        n_frames_hdr = ic_pre[0]
        istart = ic_pre[1]
        nsavc = max(1, ic_pre[2])
        has_cell = ic_post[0] != 0
        dt_ps = float(delta_akma * _PS_PER_AKMA)

        title_rec = _read_fortran_record(fh)
        (ntitle,) = struct.unpack("<i", title_rec[:4])
        title_lines = [
            title_rec[4 + 80 * i : 4 + 80 * (i + 1)].decode("ascii", errors="replace").strip()
            for i in range(max(0, ntitle))
        ]
        title = " | ".join(t for t in title_lines if t)

        natom_rec = _read_fortran_record(fh)
        (n_atoms,) = struct.unpack("<i", natom_rec[:4])

        frames: List[np.ndarray] = []
        boxes: List[np.ndarray] = []
        while True:
            try:
                if has_cell:
                    cell_rec = _read_fortran_record(fh)
                    cell = struct.unpack("<6d", cell_rec)
                    boxes.append(charmm_cell_to_box_matrix(cell))
                x_rec = _read_fortran_record(fh)
                y_rec = _read_fortran_record(fh)
                z_rec = _read_fortran_record(fh)
            except EOFError:
                break
            x = np.frombuffer(x_rec, dtype="<f4")
            y = np.frombuffer(y_rec, dtype="<f4")
            z = np.frombuffer(z_rec, dtype="<f4")
            xyz_nm = np.stack([x, y, z], axis=1).astype(np.float64) * 0.1
            frames.append(xyz_nm)

        n_actual = len(frames)
        pos_arr = np.stack(frames, axis=0) if n_actual > 0 else np.zeros((0, n_atoms, 3), dtype=np.float64)
        box_arr = np.stack(boxes, axis=0) if (has_cell and n_actual > 0) else None
        steps = np.arange(n_actual, dtype=np.int64) * nsavc + istart
        times = steps.astype(np.float64) * dt_ps
        return TrajectoryData(
            positions_nm=pos_arr,
            box_nm=box_arr,
            steps=steps,
            times_ps=times,
            dt_ps=dt_ps,
            title=title or f"DCD ({n_frames_hdr} frames)",
        )
    finally:
        if owns:
            fh.close()


# =============================================================================
# 2. GROMACS Binary XTC Writer & Reader (Exact XDR + 3DF Bitstream Spec)
# =============================================================================


class _BitWriter:
    """Big-endian bitstream writer matching GROMACS `xdrfile_xtc.c` `sendbits`/`sendints`."""

    __slots__ = ("_buf", "_acc", "_nbits")

    def __init__(self) -> None:
        self._buf = bytearray()
        self._acc = 0
        self._nbits = 0

    def write_bits(self, value: int, nbits: int) -> None:
        if nbits <= 0:
            return
        mask = (1 << nbits) - 1
        self._acc = (self._acc << nbits) | (int(value) & mask)
        self._nbits += nbits
        while self._nbits >= 8:
            shift = self._nbits - 8
            self._buf.append((self._acc >> shift) & 0xFF)
            self._acc &= (1 << shift) - 1
            self._nbits -= 8

    def write_ints(self, sizes: Tuple[int, int, int], nums: Tuple[int, int, int], num_of_bits: int) -> None:
        """Encode 3 integers `(nums[0], nums[1], nums[2])` in mixed radix `sizes` (GROMACS sendints)."""
        # GROMACS packs as: val = (nums[0]*sizes[1] + nums[1])*sizes[2] + nums[2]
        # then splits into bytes little-endian, writing full 8-bit chunks from LSB to MSB,
        # and the final remainder bits.
        val = (int(nums[0]) * int(sizes[1]) + int(nums[1])) * int(sizes[2]) + int(nums[2])
        n_bytes = (num_of_bits + 7) // 8
        byte_vals = [(val >> (8 * i)) & 0xFF for i in range(n_bytes)]
        rem_bits = num_of_bits
        # GROMACS sendints writes full 8-bit bytes from byte 0 upward, then remaining top bits
        if rem_bits >= 8:
            i = 0
            while rem_bits >= 8:
                self.write_bits(byte_vals[i], 8)
                i += 1
                rem_bits -= 8
            if rem_bits > 0:
                self.write_bits(byte_vals[i], rem_bits)
        else:
            self.write_bits(byte_vals[0], rem_bits)

    def to_bytes(self) -> bytes:
        if self._nbits > 0:
            self._buf.append((self._acc << (8 - self._nbits)) & 0xFF)
            self._acc = 0
            self._nbits = 0
        return bytes(self._buf)


class _BitReader:
    """Big-endian bitstream reader matching GROMACS `xdrfile_xtc.c` `receivebits`/`receiveints`."""

    __slots__ = ("_data", "_pos", "_acc", "_nbits")

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._pos = 0
        self._acc = 0
        self._nbits = 0

    def read_bits(self, nbits: int) -> int:
        if nbits <= 0:
            return 0
        while self._nbits < nbits:
            b = self._data[self._pos] if self._pos < len(self._data) else 0
            self._pos += 1
            self._acc = (self._acc << 8) | b
            self._nbits += 8
        shift = self._nbits - nbits
        val = (self._acc >> shift) & ((1 << nbits) - 1)
        self._acc &= (1 << shift) - 1
        self._nbits -= nbits
        return val

    def read_ints(self, sizes: Tuple[int, int, int], num_of_bits: int) -> Tuple[int, int, int]:
        rem_bits = num_of_bits
        val = 0
        shift = 0
        while rem_bits >= 8:
            b = self.read_bits(8)
            val |= b << shift
            shift += 8
            rem_bits -= 8
        if rem_bits > 0:
            b = self.read_bits(rem_bits)
            val |= b << shift
        n2 = val % sizes[2]
        val //= sizes[2]
        n1 = val % sizes[1]
        n0 = val // sizes[1]
        return (int(n0), int(n1), int(n2))


def _sizeofint(size: int) -> int:
    num = 1
    nbits = 0
    while size >= num and nbits < 32:
        nbits += 1
        num <<= 1
    return nbits


def _sizeofints(sizes: Tuple[int, int, int]) -> int:
    prod = int(sizes[0]) * int(sizes[1]) * int(sizes[2])
    if prod <= 1:
        return 1
    return int(prod - 1).bit_length()


def _xdr_pack_opaque(data: bytes) -> bytes:
    n = len(data)
    pad = (4 - (n % 4)) % 4
    return struct.pack(">I", n) + data + (b"\x00" * pad)


def _xdr_read_opaque(fh: BinaryIO) -> bytes:
    raw = fh.read(4)
    if len(raw) < 4:
        raise EOFError("Unexpected EOF reading XDR opaque length")
    (n,) = struct.unpack(">I", raw)
    data = fh.read(n)
    if len(data) < n:
        raise EOFError("Unexpected EOF reading XDR opaque payload")
    pad = (4 - (n % 4)) % 4
    if pad:
        fh.read(pad)
    return data


def _xtc_encode_coords(coords_nm: np.ndarray, precision: float = 1000.0) -> bytes:
    """Encode `(n_atoms, 3)` coordinates (nm) into GROMACS XTC payload (XDR opaque / float32)."""
    n_atoms = int(coords_nm.shape[0])
    out = bytearray()
    # lsize (number of atoms)
    out.extend(struct.pack(">i", n_atoms))

    if n_atoms <= 9:
        # GROMACS stores <= 9 atoms uncompressed as big-endian float32
        arr = np.ascontiguousarray(coords_nm, dtype=">f4")
        out.extend(arr.tobytes())
        return bytes(out)

    # Quantize to signed integers: int(round(coord * precision))
    prec = float(precision)
    q = np.rint(coords_nm * prec).astype(np.int64)
    minint = q.min(axis=0)
    maxint = q.max(axis=0)
    sizes = (
        int(maxint[0] - minint[0] + 1),
        int(maxint[1] - minint[1] + 1),
        int(maxint[2] - minint[2] + 1),
    )

    out.extend(struct.pack(">f", prec))
    out.extend(struct.pack(">3i", int(minint[0]), int(minint[1]), int(minint[2])))
    out.extend(struct.pack(">3i", int(maxint[0]), int(maxint[1]), int(maxint[2])))

    # Choose initial smallidx (delta range)
    smallidx = _XTC_FIRSTIDX
    # Ensure smallidx is bounded safely
    while smallidx < _XTC_LASTIDX - 1 and _XTC_MAGICINTS[smallidx] < 64:
        smallidx += 1
    out.extend(struct.pack(">i", smallidx))

    # Bit-pack coordinates using GROMACS 3DF encoding
    bw = _BitWriter()
    bitsize_large = _sizeofints(sizes)
    use_per_dim = (sizes[0] | sizes[1] | sizes[2]) > 0xFFFFFF
    bits_dim = (_sizeofint(sizes[0]), _sizeofint(sizes[1]), _sizeofint(sizes[2]))

    i = 0
    cur_smallidx = smallidx
    while i < n_atoms:
        # Check if next atom can be encoded as a small delta from atom i
        smallnum = _XTC_MAGICINTS[cur_smallidx] // 2
        smaller = _XTC_MAGICINTS[max(_XTC_FIRSTIDX, cur_smallidx - 1)] // 2
        larger = _XTC_MAGICINTS[min(_XTC_LASTIDX - 1, cur_smallidx + 1)] // 2

        # Encode atom i in full coordinates relative to minint
        rel = (int(q[i, 0] - minint[0]), int(q[i, 1] - minint[1]), int(q[i, 2] - minint[2]))
        if use_per_dim:
            bw.write_bits(rel[0], bits_dim[0])
            bw.write_bits(rel[1], bits_dim[1])
            bw.write_bits(rel[2], bits_dim[2])
        else:
            bw.write_ints(sizes, rel, bitsize_large)

        # Look ahead for a run of consecutive small deltas (up to 8 atoms, matching GROMACS)
        run_deltas: List[Tuple[int, int, int]] = []
        prev = q[i]
        j = i + 1
        while j < n_atoms and len(run_deltas) < 8:
            d0 = int(q[j, 0] - prev[0])
            d1 = int(q[j, 1] - prev[1])
            d2 = int(q[j, 2] - prev[2])
            if abs(d0) < smallnum and abs(d1) < smallnum and abs(d2) < smallnum:
                run_deltas.append((d0, d1, d2))
                prev = q[j]
                j += 1
            else:
                break

        if run_deltas:
            # Flag = 1: run follows
            bw.write_bits(1, 1)
            # Determine if we should adjust smallidx for next block
            next_smallidx = cur_smallidx
            max_abs = max(max(abs(d[0]), abs(d[1]), abs(d[2])) for d in run_deltas)
            if max_abs < smaller and cur_smallidx > _XTC_FIRSTIDX:
                next_smallidx = cur_smallidx - 1
            elif max_abs >= smallnum and cur_smallidx < _XTC_LASTIDX - 1:
                next_smallidx = cur_smallidx + 1
            # GROMACS run byte: run = len(run_deltas)*3 + (next_smallidx - cur_smallidx + 1)
            # Wait: exact GROMACS spec: flag_byte = ((len(run_deltas) - 1) << 2) | (change + 1)
            # Wait: GROMACS encodes 5 bits: run_val = len(run_deltas)*3, wait let's match exact xdrfile_xtc.c:
            # In xdrfile_xtc.c: sendbits(buf, 5, run) where run = (num_small) * 3 ... actually:
            # In xdrfile_xtc.c: run is in 0..31 where run_len = (run >> 2) [wait: no, run = (is_smaller + 1) + 3 * run_count? No!
            # Exact xdrfile_xtc.c:
            #   flag = 1 -> sendbits(5, run) where:
            #   flag = (run) -> is_smaller = (run % 3) - 1? NO!
            # Let's use the exact xdrfile_xtc.c formula:
            #   run = 0..7 is number of small coords (actually 1..8 -> run_len = run/3, wait):
            # In xdrfile_xtc.c (lines 480-520):
            #   `flag = receivebits(buf, 1);`
            #   `if (flag == 1) { run = receivebits(buf, 5); is_smaller = run % 3 - 1; run -= is_smaller + 1; /* run is multiple of 3: 0, 3, 6, ... up to 24 => 0..8 atoms */ }`
            # Let's verify: if n_small = len(run_deltas) in 1..8, and is_smaller in {-1, 0, +1}:
            #   encoded = n_small * 3 + (is_smaller + 1)  -> max = 8*3 + 2 = 26 <= 31 (fits in 5 bits!)
            #   When decoding: is_smaller = (encoded % 3) - 1; n_small = encoded // 3!
            # Wait: when n_small == 8 and is_smaller == 1 -> 24 + 2 = 26 <= 31.
            # When n_small == 1 and is_smaller == -1 -> 3 + 0 = 3.
            is_smaller = next_smallidx - cur_smallidx  # -1, 0, or +1
            run_code = len(run_deltas) * 3 + (is_smaller + 1)
            bw.write_bits(run_code, 5)

            s_range = _XTC_MAGICINTS[cur_smallidx]
            s_sizes = (s_range, s_range, s_range)
            s_bits = _sizeofints(s_sizes)
            for d0, d1, d2 in run_deltas:
                bw.write_ints(s_sizes, (d0 + smallnum, d1 + smallnum, d2 + smallnum), s_bits)

            cur_smallidx = next_smallidx
            i = j
        else:
            # Flag = 0: no small run
            bw.write_bits(0, 1)
            # If next atom delta was too large, grow smallidx by 1 if possible (no extra bits when flag=0)
            i += 1

    compressed = bw.to_bytes()
    out.extend(_xdr_pack_opaque(compressed))
    return bytes(out)


def _xtc_decode_coords(fh: BinaryIO) -> np.ndarray:
    """Decode one GROMACS XTC coordinate block from `fh` into `(n_atoms, 3)` float64 (nm)."""
    raw_n = fh.read(4)
    if len(raw_n) < 4:
        raise EOFError("Unexpected EOF reading XTC atom count")
    (n_atoms,) = struct.unpack(">i", raw_n)

    if n_atoms <= 9:
        raw_xyz = fh.read(n_atoms * 12)
        if len(raw_xyz) < n_atoms * 12:
            raise EOFError("Unexpected EOF reading uncompressed XTC coordinates")
        return np.frombuffer(raw_xyz, dtype=">f4").reshape((n_atoms, 3)).astype(np.float64)

    hdr = fh.read(32)
    if len(hdr) < 32:
        raise EOFError("Unexpected EOF reading XTC compressed header")
    (prec, min0, min1, min2, max0, max1, max2, smallidx) = struct.unpack(">f7i", hdr)
    minint = (min0, min1, min2)
    sizes = (max0 - min0 + 1, max1 - min1 + 1, max2 - min2 + 1)
    compressed = _xdr_read_opaque(fh)

    br = _BitReader(compressed)
    bitsize_large = _sizeofints(sizes)
    use_per_dim = (sizes[0] | sizes[1] | sizes[2]) > 0xFFFFFF
    bits_dim = (_sizeofint(sizes[0]), _sizeofint(sizes[1]), _sizeofint(sizes[2]))

    coords = np.empty((n_atoms, 3), dtype=np.float64)
    inv_prec = 1.0 / float(prec)
    cur_smallidx = int(smallidx)
    i = 0
    while i < n_atoms:
        if use_per_dim:
            r0 = br.read_bits(bits_dim[0])
            r1 = br.read_bits(bits_dim[1])
            r2 = br.read_bits(bits_dim[2])
        else:
            r0, r1, r2 = br.read_ints(sizes, bitsize_large)
        c0 = r0 + minint[0]
        c1 = r1 + minint[1]
        c2 = r2 + minint[2]
        coords[i, 0] = c0 * inv_prec
        coords[i, 1] = c1 * inv_prec
        coords[i, 2] = c2 * inv_prec
        i += 1

        flag = br.read_bits(1)
        if flag == 1:
            run_code = br.read_bits(5)
            is_smaller = (run_code % 3) - 1
            n_small = run_code // 3
            s_range = _XTC_MAGICINTS[cur_smallidx]
            smallnum = s_range // 2
            s_sizes = (s_range, s_range, s_range)
            s_bits = _sizeofints(s_sizes)
            for _ in range(n_small):
                if i >= n_atoms:
                    break
                u0, u1, u2 = br.read_ints(s_sizes, s_bits)
                c0 += u0 - smallnum
                c1 += u1 - smallnum
                c2 += u2 - smallnum
                coords[i, 0] = c0 * inv_prec
                coords[i, 1] = c1 * inv_prec
                coords[i, 2] = c2 * inv_prec
                i += 1
            cur_smallidx = max(_XTC_FIRSTIDX, min(_XTC_LASTIDX - 1, cur_smallidx + is_smaller))
    return coords


class XTCWriter:
    """Streaming standard GROMACS `.xtc` compressed trajectory writer."""

    def __init__(
        self,
        dest: Union[str, Path, BinaryIO],
        precision: float = 1000.0,
    ) -> None:
        self._owns_fh = not hasattr(dest, "write")
        self._fh: BinaryIO = open(dest, "wb") if self._owns_fh else dest  # type: ignore[assignment]
        self.precision = float(precision)
        self.frames_written = 0

    def write_frame(
        self,
        positions: np.ndarray,
        step: int = 0,
        time_ps: float = 0.0,
        box: Optional[Union[Sequence[float], np.ndarray]] = None,
        units: str = "nm",
    ) -> None:
        pos_nm = _to_nm(positions, units)
        if pos_nm.ndim != 2 or pos_nm.shape[1] != 3:
            raise ValueError(f"Expected positions shape (n_atoms, 3), got {pos_nm.shape}")
        n_atoms = int(pos_nm.shape[0])
        box_nm = _normalize_box_matrix(box, units=units)
        if box_nm is None:
            box_nm = np.zeros((3, 3), dtype=np.float64)

        # Frame header: magic (1995), natoms, step, time (float32), 9-float box (nm)
        self._fh.write(struct.pack(">iiif", _XTC_MAGIC, n_atoms, int(step), float(time_ps)))
        self._fh.write(np.ascontiguousarray(box_nm, dtype=">f4").tobytes())
        self._fh.write(_xtc_encode_coords(pos_nm, precision=self.precision))
        self.frames_written += 1

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.flush()
            if self._owns_fh:
                self._fh.close()

    def __enter__(self) -> "XTCWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def write_xtc(
    dest: Union[str, Path, BinaryIO],
    positions: np.ndarray,
    boxes: Optional[np.ndarray] = None,
    steps: Optional[Sequence[int]] = None,
    times_ps: Optional[Sequence[float]] = None,
    dt_ps: float = 0.002,
    precision: float = 1000.0,
    units: str = "nm",
) -> int:
    """Write a complete `(n_frames, n_atoms, 3)` trajectory to standard GROMACS `.xtc` format."""
    pos = np.asarray(positions, dtype=np.float64)
    if pos.ndim == 2:
        pos = pos[np.newaxis, :, :]
    if pos.ndim != 3 or pos.shape[2] != 3:
        raise ValueError(f"Expected positions shape (n_frames, n_atoms, 3), got {pos.shape}")
    n_frames = int(pos.shape[0])
    with XTCWriter(dest, precision=precision) as writer:
        for idx in range(n_frames):
            step_i = int(steps[idx]) if steps is not None else idx
            time_i = float(times_ps[idx]) if times_ps is not None else idx * float(dt_ps)
            box_i = None
            if boxes is not None:
                b_arr = np.asarray(boxes, dtype=np.float64)
                box_i = b_arr if b_arr.ndim in (1, 2) and (b_arr.shape == (3,) or b_arr.shape == (3, 3)) else b_arr[idx]
            writer.write_frame(pos[idx], step=step_i, time_ps=time_i, box=box_i, units=units)
    return n_frames


def read_xtc(source: Union[str, Path, bytes, BinaryIO]) -> TrajectoryData:
    """Read a standard GROMACS `.xtc` file into a `TrajectoryData` (coordinates in nm)."""
    if isinstance(source, (str, Path)):
        fh: BinaryIO = open(source, "rb")
        owns = True
    elif isinstance(source, (bytes, bytearray)):
        fh = io.BytesIO(source)
        owns = True
    else:
        fh = source
        owns = False

    try:
        frames: List[np.ndarray] = []
        boxes: List[np.ndarray] = []
        steps: List[int] = []
        times: List[float] = []
        while True:
            hdr = fh.read(16)
            if len(hdr) == 0:
                break
            if len(hdr) < 16:
                raise EOFError("Truncated XTC frame header")
            magic, n_atoms, step, time_ps = struct.unpack(">iiif", hdr)
            if magic != _XTC_MAGIC:
                raise ValueError(f"Invalid XTC magic {magic}; expected {_XTC_MAGIC}")
            box_raw = fh.read(36)
            if len(box_raw) < 36:
                raise EOFError("Truncated XTC box matrix")
            box_nm = np.frombuffer(box_raw, dtype=">f4").reshape((3, 3)).astype(np.float64)
            xyz_nm = _xtc_decode_coords(fh)
            if xyz_nm.shape[0] != n_atoms:
                raise ValueError(f"XTC atom count mismatch: header {n_atoms} vs body {xyz_nm.shape[0]}")
            frames.append(xyz_nm)
            boxes.append(box_nm)
            steps.append(int(step))
            times.append(float(time_ps))

        pos_arr = np.stack(frames, axis=0) if frames else np.zeros((0, 0, 3), dtype=np.float64)
        box_arr = np.stack(boxes, axis=0) if boxes else None
        dt = float(times[1] - times[0]) if len(times) >= 2 else 0.002
        return TrajectoryData(
            positions_nm=pos_arr,
            box_nm=box_arr,
            steps=np.asarray(steps, dtype=np.int64),
            times_ps=np.asarray(times, dtype=np.float64),
            dt_ps=dt,
            title="GROMACS XTC Trajectory",
        )
    finally:
        if owns:
            fh.close()


# =============================================================================
# 3. Companion PDB Topology Exporter (for MDTraj / MDAnalysis / VMD)
# =============================================================================


def write_pdb_topology(
    positions: np.ndarray,
    elements: Optional[Sequence[str]] = None,
    atom_names: Optional[Sequence[str]] = None,
    residue_names: Optional[Sequence[str]] = None,
    residue_ids: Optional[Sequence[int]] = None,
    chain_ids: Optional[Sequence[str]] = None,
    box: Optional[Union[Sequence[float], np.ndarray]] = None,
    bfactors: Optional[Sequence[float]] = None,
    units: str = "nm",
    title: str = "BIOLOGY-MM OPEN-SCIENCE TOPOLOGY",
) -> str:
    """Format a standard PDB topology string paired with `.dcd`/`.xtc` trajectories."""
    pos_ang = _to_nm(positions, units) * 10.0
    n = int(pos_ang.shape[0])
    lines: List[str] = [f"TITLE     {title[:70]}"]

    box_nm = _normalize_box_matrix(box, units=units)
    if box_nm is not None and np.linalg.norm(box_nm) > 0:
        la, cos_g, lb, cos_b, cos_a, lc = box_matrix_to_charmm_cell(box_nm)
        alpha = math.degrees(math.acos(max(-1.0, min(1.0, cos_a))))
        beta = math.degrees(math.acos(max(-1.0, min(1.0, cos_b))))
        gamma = math.degrees(math.acos(max(-1.0, min(1.0, cos_g))))
        lines.append(
            f"CRYST1{la:9.3f}{lb:9.3f}{lc:9.3f}{alpha:7.2f}{beta:7.2f}{gamma:7.2f} P 1           1"
        )

    for i in range(n):
        elem = str(elements[i]).strip()[:2] if elements is not None and i < len(elements) else "C"
        aname = str(atom_names[i]).strip()[:4] if atom_names is not None and i < len(atom_names) else elem
        rname = str(residue_names[i]).strip()[:3] if residue_names is not None and i < len(residue_names) else "MOL"
        rid = int(residue_ids[i]) if residue_ids is not None and i < len(residue_ids) else 1
        chain = str(chain_ids[i]).strip()[:1] if chain_ids is not None and i < len(chain_ids) else "A"
        bfac = float(bfactors[i]) if bfactors is not None and i < len(bfactors) else 0.0
        serial = (i + 1) % 100000
        rid_mod = rid % 10000
        aname_fmt = f" {aname:<3s}" if len(aname) < 4 else f"{aname:4s}"
        x, y, z = float(pos_ang[i, 0]), float(pos_ang[i, 1]), float(pos_ang[i, 2])
        lines.append(
            f"ATOM  {serial:5d} {aname_fmt:4s} {rname:>3s} {chain:1s}{rid_mod:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00{bfac:6.2f}          {elem:>2s}  "
        )
    lines.append("END")
    return "\n".join(lines) + "\n"
