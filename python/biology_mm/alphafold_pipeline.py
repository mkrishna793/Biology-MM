"""AlphaFold DB + PAE Domain Clustering + pLDDT-Guided Multiscale & ATLAS Flexibility Pipeline.

Provides end-to-end post-prediction physics preparation and validation for AlphaFold2 /
AlphaFold DB / ESMFold / ATLAS protein structures on a single laptop:
  1. `fetch_alphafold_entry(uniprot_id)`: Fetches predicted mmCIF/PDB + PAE JSON from the
     official EMBL-EBI AlphaFold Protein Structure Database REST API.
  2. `parse_alphafold_plddt_any(text, fmt)`: Extracts per-atom and per-residue pLDDT scores
     from both `.pdb` (B-factor column) and `.cif`/`.mmcif` (`_atom_site.B_iso_or_equiv` /
     `_ma_qa_metric_local.metric_value`).
  3. `cluster_pae_domains(pae_matrix)`: Graph-based rigid domain & inter-domain hinge
     detection from the Predicted Aligned Error (PAE) matrix.
  4. `compute_confidence_guided_setup(...)`: Automatically centers Biology-MM's All-Atom
     Focus Zone on flexible loops (pLDDT < 70) or uncertain inter-domain hinges and computes
     smooth per-atom pLDDT-weighted harmonic scaffold restraints.
  5. `compute_per_residue_rmsf(...)` & `evaluate_atlas_flexibility(...)`: Kabsch-aligned
     per-residue RMSF and Pearson/Spearman correlation against ATLAS reference MD dynamics.
"""

from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import pearsonr, spearmanr

ALPHAFOLD_API_BASE = "https://alphafold.ebi.ac.uk/api/prediction"
_USER_AGENT = "Biology-MM/2.6.0 (Open-Science Molecular Dynamics; https://github.com/mkrishna793/Biology-MM)"


@dataclass
class AlphaFoldStructureConfidence:
    """Per-atom and per-residue pLDDT confidence parsed from PDB or mmCIF."""

    plddt_per_atom: np.ndarray
    plddt_per_residue: np.ndarray
    residue_ids: np.ndarray
    mean_plddt: float
    low_confidence_atoms: np.ndarray      # pLDDT < 70
    medium_confidence_atoms: np.ndarray   # 70 <= pLDDT < 90
    high_confidence_atoms: np.ndarray     # pLDDT >= 90
    is_likely_alphafold: bool = True

    def to_dict(self) -> Dict[str, Any]:
        n = max(1, int(self.plddt_per_atom.size))
        return {
            "n_atoms": int(self.plddt_per_atom.size),
            "n_residues": int(self.plddt_per_residue.size),
            "mean_plddt": round(float(self.mean_plddt), 2),
            "frac_low_confidence": round(float(self.low_confidence_atoms.size) / n, 4),
            "frac_medium_confidence": round(float(self.medium_confidence_atoms.size) / n, 4),
            "frac_high_confidence": round(float(self.high_confidence_atoms.size) / n, 4),
            "is_likely_alphafold": bool(self.is_likely_alphafold),
        }


@dataclass
class PAEDomainPartition:
    """Rigid domains and flexible linker/hinge residues extracted from a PAE matrix."""

    n_residues: int
    domain_labels: np.ndarray            # (L,) int32: 0..D-1 for rigid domains, -1 for linkers/hinges
    domains: List[List[int]]             # 0-based residue indices per rigid domain
    hinge_residues: List[int]            # 0-based flexible linker / inter-domain hinge residues
    inter_domain_pae_mean: float
    pae_cutoff_ang: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_residues": int(self.n_residues),
            "n_domains": len(self.domains),
            "domain_sizes": [len(d) for d in self.domains],
            "n_hinge_residues": len(self.hinge_residues),
            "inter_domain_pae_mean": round(float(self.inter_domain_pae_mean), 2),
            "pae_cutoff_ang": float(self.pae_cutoff_ang),
        }


@dataclass
class ConfidenceGuidedSetup:
    """Automatic multiscale focus zone and per-atom pLDDT restraint schedule."""

    focus_center_nm: List[float]
    focus_center_ang: List[float]
    aa_radius_nm: float
    cg_radius_nm: float
    aa_radius_ang: float
    cg_radius_ang: float
    restraint_k_per_atom: np.ndarray     # (N,) float32 kJ/(mol*nm^2)
    focus_reason: str
    confidence_summary: Dict[str, Any] = field(default_factory=dict)
    pae_summary: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "focus_center_nm": [round(float(v), 4) for v in self.focus_center_nm],
            "focus_center_ang": [round(float(v), 2) for v in self.focus_center_ang],
            "aa_radius_nm": round(float(self.aa_radius_nm), 3),
            "cg_radius_nm": round(float(self.cg_radius_nm), 3),
            "aa_radius_ang": round(float(self.aa_radius_ang), 2),
            "cg_radius_ang": round(float(self.cg_radius_ang), 2),
            "focus_reason": self.focus_reason,
            "mean_restraint_k": round(float(np.mean(self.restraint_k_per_atom)), 2) if self.restraint_k_per_atom.size else 0.0,
            "unrestrained_atoms": int(np.sum(self.restraint_k_per_atom <= 1e-6)),
            "confidence": self.confidence_summary,
            "pae": self.pae_summary,
        }


@dataclass
class AlphaFoldEntry:
    """Downloaded AlphaFold DB record with structure text and optional PAE matrix."""

    uniprot_id: str
    entry_id: str
    gene: str
    organism: str
    structure_text: str
    structure_format: str                # "cif" or "pdb"
    filename: str
    pae_matrix: Optional[np.ndarray] = None
    source_url: str = ""
    pae_url: str = ""


@dataclass
class ATLASFlexibilityReport:
    """Quantitative comparison between predicted MD RMSF and ATLAS reference RMSF."""

    target_id: str
    n_residues: int
    pearson_r: float
    spearman_rho: float
    rmse_ang: float
    mean_pred_rmsf_ang: float
    mean_ref_rmsf_ang: float


def _http_get_text(url: str, timeout: float = 20.0) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def parse_pae_json(pae_json_text: str) -> Optional[np.ndarray]:
    """Parse AlphaFold DB v2/v3/v4 PAE JSON into a dense `(L, L)` float32 matrix in Å."""
    data = json.loads(pae_json_text)
    if isinstance(data, list) and len(data) > 0:
        data = data[0]
    if not isinstance(data, dict):
        return None
    if "predicted_aligned_error" in data:
        arr = np.asarray(data["predicted_aligned_error"], dtype=np.float32)
        if arr.ndim == 2 and arr.shape[0] == arr.shape[1] and arr.shape[0] > 0:
            return arr
    if "distance" in data and "residue1" in data and "residue2" in data:
        r1 = np.asarray(data["residue1"], dtype=np.int32) - 1
        r2 = np.asarray(data["residue2"], dtype=np.int32) - 1
        dist = np.asarray(data["distance"], dtype=np.float32)
        L = int(max(r1.max(initial=0), r2.max(initial=0))) + 1
        mat = np.zeros((L, L), dtype=np.float32)
        mat[r1, r2] = dist
        return mat
    return None


def fetch_alphafold_entry(
    uniprot_id: str,
    timeout: float = 20.0,
    prefer_format: str = "cif",
) -> AlphaFoldEntry:
    """Fetch an AlphaFold DB structure (mmCIF or PDB) and PAE matrix by UniProt accession."""
    uid = str(uniprot_id).strip().upper()
    if not uid or any(c.isspace() or c in "/\\?#" for c in uid):
        raise ValueError(f"Invalid UniProt accession '{uniprot_id}'")

    api_url = f"{ALPHAFOLD_API_BASE}/{uid}"
    try:
        meta_raw = _http_get_text(api_url, timeout=timeout)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise ValueError(f"UniProt accession '{uid}' was not found in AlphaFold DB") from exc
        raise RuntimeError(f"AlphaFold DB API returned HTTP {exc.code} for '{uid}'") from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to contact AlphaFold DB API for '{uid}': {exc}") from exc

    records = json.loads(meta_raw)
    if isinstance(records, dict):
        records = [records]
    if not records:
        raise ValueError(f"No AlphaFold DB prediction records found for '{uid}'")

    rec = records[0]
    entry_id = str(rec.get("entryId") or f"AF-{uid}-F1")
    gene = str(rec.get("gene") or uid)
    organism = str(rec.get("organismScientificName") or "")

    cif_url = str(rec.get("cifUrl") or "")
    pdb_url = str(rec.get("pdbUrl") or "")
    pae_url = str(rec.get("paeDocUrl") or "")

    if prefer_format == "pdb" and pdb_url:
        struct_url, fmt = pdb_url, "pdb"
    elif cif_url:
        struct_url, fmt = cif_url, "cif"
    elif pdb_url:
        struct_url, fmt = pdb_url, "pdb"
    else:
        raise RuntimeError(f"AlphaFold DB record for '{uid}' contained no cifUrl or pdbUrl")

    structure_text = _http_get_text(struct_url, timeout=timeout)
    pae_matrix: Optional[np.ndarray] = None
    if pae_url:
        try:
            pae_matrix = parse_pae_json(_http_get_text(pae_url, timeout=timeout))
        except Exception:
            pae_matrix = None

    return AlphaFoldEntry(
        uniprot_id=uid,
        entry_id=entry_id,
        gene=gene,
        organism=organism,
        structure_text=structure_text,
        structure_format=fmt,
        filename=f"{entry_id}.{fmt}",
        pae_matrix=pae_matrix,
        source_url=struct_url,
        pae_url=pae_url,
    )


def _tokenize_cif_line(line: str) -> List[str]:
    tokens: List[str] = []
    i, n = 0, len(line)
    while i < n:
        while i < n and line[i].isspace():
            i += 1
        if i >= n or line[i] == "#":
            break
        if line[i] in ("'", '"'):
            quote = line[i]
            i += 1
            start = i
            while i < n and not (line[i] == quote and (i + 1 == n or line[i + 1].isspace())):
                i += 1
            tokens.append(line[start:i])
            if i < n:
                i += 1
        else:
            start = i
            while i < n and not line[i].isspace():
                i += 1
            tokens.append(line[start:i])
    return tokens


def parse_alphafold_plddt_any(
    structure_text: str,
    fmt: str = "auto",
) -> AlphaFoldStructureConfidence:
    """Parse per-atom & per-residue pLDDT scores from AlphaFold PDB or mmCIF text."""
    kind = fmt.lower().strip()
    if kind == "auto":
        kind = "cif" if ("_atom_site." in structure_text or structure_text.lstrip().startswith("data_")) else "pdb"

    atom_plddt: List[float] = []
    atom_res: List[int] = []
    header_hint = any(
        tag in structure_text[:8000].upper()
        for tag in ("ALPHAFOLD", "PLDDT", "_MA_QA_METRIC_LOCAL", "COLABFOLD", "ESMFOLD")
    )

    if kind == "pdb":
        for line in structure_text.splitlines():
            if line.startswith(("ATOM  ", "HETATM")) and len(line) >= 66:
                try:
                    b_val = float(line[60:66].strip())
                except ValueError:
                    b_val = 70.0
                try:
                    res_id = int(line[22:26].strip())
                except ValueError:
                    res_id = len(atom_res) + 1
                atom_plddt.append(b_val)
                atom_res.append(res_id)
    else:
        lines = structure_text.splitlines()
        i, n_lines = 0, len(lines)
        while i < n_lines:
            if lines[i].strip() == "loop_":
                i += 1
                cols: List[str] = []
                while i < n_lines and lines[i].strip().startswith("_"):
                    cols.append(lines[i].strip())
                    i += 1
                if any(c.startswith("_atom_site.") for c in cols):
                    col_map = {c: idx for idx, c in enumerate(cols)}
                    b_idx = col_map.get("_atom_site.B_iso_or_equiv")
                    seq_idx = col_map.get("_atom_site.label_seq_id", col_map.get("_atom_site.auth_seq_id"))
                    grp_idx = col_map.get("_atom_site.group_PDB")
                    while i < n_lines:
                        raw = lines[i].strip()
                        if not raw or raw.startswith(("#", "loop_", "data_", "_")):
                            break
                        toks = _tokenize_cif_line(raw)
                        if len(toks) >= len(cols):
                            if grp_idx is not None and toks[grp_idx] not in ("ATOM", "HETATM"):
                                i += 1
                                continue
                            b_val = 70.0
                            if b_idx is not None:
                                try:
                                    b_val = float(toks[b_idx])
                                except ValueError:
                                    b_val = 70.0
                            res_id = len(atom_res) + 1
                            if seq_idx is not None:
                                try:
                                    res_id = int(toks[seq_idx])
                                except ValueError:
                                    res_id = len(atom_res) + 1
                            atom_plddt.append(b_val)
                            atom_res.append(res_id)
                        i += 1
                    continue
            i += 1

    arr = np.asarray(atom_plddt, dtype=np.float32)
    res_arr = np.asarray(atom_res, dtype=np.int32)
    if arr.size == 0:
        empty = np.zeros(0, dtype=np.int32)
        return AlphaFoldStructureConfidence(
            plddt_per_atom=np.zeros(0, dtype=np.float32),
            plddt_per_residue=np.zeros(0, dtype=np.float32),
            residue_ids=empty,
            mean_plddt=0.0,
            low_confidence_atoms=empty,
            medium_confidence_atoms=empty,
            high_confidence_atoms=empty,
            is_likely_alphafold=False,
        )

    # Normalize 0..1 fractional pLDDT (e.g. ESMFold) to 0..100 scale
    if float(np.max(arr)) <= 1.05 and float(np.min(arr)) >= 0.0 and header_hint:
        arr = arr * np.float32(100.0)

    # Compute per-residue mean pLDDT in residue order
    unique_res, inv = np.unique(res_arr, return_inverse=True)
    res_sums = np.bincount(inv, weights=arr.astype(np.float64))
    res_cnts = np.maximum(np.bincount(inv), 1)
    plddt_res = (res_sums / res_cnts).astype(np.float32)

    in_plddt_range = bool(np.min(arr) >= 0.0 and np.max(arr) <= 100.5)
    is_af = bool(header_hint or (in_plddt_range and float(np.mean(arr)) >= 35.0 and float(np.std(arr)) > 0.5))

    return AlphaFoldStructureConfidence(
        plddt_per_atom=arr,
        plddt_per_residue=plddt_res,
        residue_ids=res_arr,
        mean_plddt=float(np.mean(arr)),
        low_confidence_atoms=np.flatnonzero(arr < 70.0).astype(np.int32),
        medium_confidence_atoms=np.flatnonzero((arr >= 70.0) & (arr < 90.0)).astype(np.int32),
        high_confidence_atoms=np.flatnonzero(arr >= 90.0).astype(np.int32),
        is_likely_alphafold=is_af,
    )


def cluster_pae_domains(
    pae_matrix: np.ndarray,
    pae_cutoff_ang: float = 5.0,
    min_domain_size: int = 15,
) -> PAEDomainPartition:
    """Partition residues into rigid domains and flexible linkers/hinges from a PAE matrix.

    Inspired by Tristan Croll's `pae_to_domains`: builds a symmetrized low-PAE contact graph
    (edges where `0.5 * (PAE_ij + PAE_ji) <= pae_cutoff_ang` and sequence separation >= 2),
    identifies rigid domain components (`size >= min_domain_size`), and marks smaller
    segments or high-PAE boundary residues as flexible hinges/linkers.
    """
    pae = np.asarray(pae_matrix, dtype=np.float64)
    if pae.ndim != 2 or pae.shape[0] != pae.shape[1] or pae.shape[0] == 0:
        raise ValueError(f"Expected square non-empty PAE matrix, got shape {pae.shape}")

    L = int(pae.shape[0])
    sym = 0.5 * (pae + pae.T)

    # Require non-local low-PAE contacts (|i - j| >= 2) plus local backbone continuity
    idx = np.arange(L)
    sep = np.abs(idx[:, None] - idx[None, :])
    adj = (sym <= float(pae_cutoff_ang)) & (sep >= 2)
    # Also connect immediate neighbors only if both participate in a low-PAE contact
    has_nonlocal = np.any(adj, axis=1)
    for k in range(L - 1):
        if has_nonlocal[k] and has_nonlocal[k + 1] and sym[k, k + 1] <= float(pae_cutoff_ang):
            adj[k, k + 1] = True
            adj[k + 1, k] = True

    n_comp, raw_labels = connected_components(csr_matrix(adj.astype(np.int8)), directed=False)
    domains: List[List[int]] = []
    domain_labels = np.full(L, -1, dtype=np.int32)
    hinges: List[int] = []

    comp_lists: List[List[int]] = [[] for _ in range(n_comp)]
    for res_idx, cid in enumerate(raw_labels):
        comp_lists[int(cid)].append(res_idx)

    # Sort rigid domains largest-first
    rigid_comps = [c for c in comp_lists if len(c) >= int(min_domain_size)]
    rigid_comps.sort(key=len, reverse=True)
    for dom_id, members in enumerate(rigid_comps):
        domains.append(sorted(members))
        for r in members:
            domain_labels[r] = dom_id

    for r in range(L):
        if domain_labels[r] < 0:
            hinges.append(r)

    # Compute mean inter-domain PAE (measures orientational uncertainty between rigid domains)
    if len(domains) >= 2:
        inter_vals: List[float] = []
        for a in range(len(domains)):
            da = np.asarray(domains[a], dtype=np.int32)
            for b in range(a + 1, len(domains)):
                db = np.asarray(domains[b], dtype=np.int32)
                inter_vals.append(float(np.mean(sym[np.ix_(da, db)])))
        inter_mean = float(np.mean(inter_vals))
    else:
        inter_mean = float(np.mean(sym))

    return PAEDomainPartition(
        n_residues=L,
        domain_labels=domain_labels,
        domains=domains,
        hinge_residues=hinges,
        inter_domain_pae_mean=inter_mean,
        pae_cutoff_ang=float(pae_cutoff_ang),
    )


def compute_plddt_restraint_weights(
    plddt_per_atom: np.ndarray,
    k_high: float = 250.0,
    low_cutoff: float = 70.0,
    high_cutoff: float = 90.0,
) -> np.ndarray:
    """Map per-atom pLDDT scores to smooth harmonic restraint force constants (kJ/(mol*nm^2)).

    - `pLDDT <= low_cutoff (70)`: `k = 0.0` (flexible loops / disordered tails move freely).
    - `low_cutoff < pLDDT < high_cutoff`: smooth cubic Hermite ramp `3t^2 - 2t^3`.
    - `pLDDT >= high_cutoff (90)`: `k = k_high` (rigid folded core gently anchored).
    """
    p = np.asarray(plddt_per_atom, dtype=np.float64).reshape(-1)
    if p.size == 0:
        return np.zeros(0, dtype=np.float32)
    span = max(1e-3, float(high_cutoff) - float(low_cutoff))
    t = np.clip((p - float(low_cutoff)) / span, 0.0, 1.0)
    smooth = t * t * (3.0 - 2.0 * t)
    return (float(max(0.0, k_high)) * smooth).astype(np.float32)


def compute_confidence_guided_setup(
    positions_nm: np.ndarray,
    confidence: AlphaFoldStructureConfidence,
    pae_matrix: Optional[np.ndarray] = None,
    k_high: float = 250.0,
    low_cutoff: float = 70.0,
    high_cutoff: float = 90.0,
) -> ConfidenceGuidedSetup:
    """Compute optimal All-Atom Focus Zone + pLDDT restraints for an AlphaFold structure."""
    pos = np.asarray(positions_nm, dtype=np.float64)
    n = int(pos.shape[0])
    if n == 0:
        return ConfidenceGuidedSetup(
            focus_center_nm=[0.0, 0.0, 0.0],
            focus_center_ang=[0.0, 0.0, 0.0],
            aa_radius_nm=2.0,
            cg_radius_nm=4.0,
            aa_radius_ang=20.0,
            cg_radius_ang=40.0,
            restraint_k_per_atom=np.zeros(0, dtype=np.float32),
            focus_reason="empty-structure",
        )

    plddt = confidence.plddt_per_atom
    if plddt.size != n:
        plddt = np.resize(plddt, n) if plddt.size > 0 else np.full(n, 70.0, dtype=np.float32)

    restraint_k = compute_plddt_restraint_weights(
        plddt, k_high=k_high, low_cutoff=low_cutoff, high_cutoff=high_cutoff
    )

    pae_part: Optional[PAEDomainPartition] = None
    if pae_matrix is not None and np.asarray(pae_matrix).ndim == 2:
        pae_part = cluster_pae_domains(pae_matrix)

    # Priority 1: PAE inter-domain hinge residues (if multiple rigid domains exist)
    focus_atoms: Optional[np.ndarray] = None
    reason = "global-centroid"
    if pae_part is not None and len(pae_part.hinge_residues) > 0 and confidence.residue_ids.size == n:
        unique_res = np.unique(confidence.residue_ids)
        hinge_res_ids = {int(unique_res[h]) for h in pae_part.hinge_residues if h < unique_res.size}
        mask = np.isin(confidence.residue_ids, list(hinge_res_ids))
        if np.any(mask):
            focus_atoms = np.flatnonzero(mask)
            # Zero restraints on PAE hinge atoms so inter-domain motion is unimpeded
            restraint_k[focus_atoms] = 0.0
            reason = f"pae-interdomain-hinge ({len(pae_part.domains)} domains, {len(pae_part.hinge_residues)} hinge residues)"

    # Priority 2: Low-pLDDT (< 70) flexible loops
    if focus_atoms is None and confidence.low_confidence_atoms.size > 0:
        valid_low = confidence.low_confidence_atoms[confidence.low_confidence_atoms < n]
        if valid_low.size > 0:
            focus_atoms = valid_low
            reason = f"low-plddt-loops ({valid_low.size} atoms with pLDDT < {low_cutoff:.0f})"

    if focus_atoms is not None and focus_atoms.size > 0:
        sub = pos[focus_atoms]
        center = np.mean(sub, axis=0)
        spread = float(np.percentile(np.linalg.norm(sub - center[None, :], axis=1), 90.0))
        aa_r = max(1.5, min(6.0, spread + 0.8))
    else:
        center = np.mean(pos, axis=0)
        spread = float(np.percentile(np.linalg.norm(pos - center[None, :], axis=1), 85.0))
        aa_r = max(2.0, min(6.0, spread + 0.5))

    cg_r = max(aa_r * 1.8, aa_r + 1.5)
    return ConfidenceGuidedSetup(
        focus_center_nm=[float(c) for c in center],
        focus_center_ang=[float(c * 10.0) for c in center],
        aa_radius_nm=float(aa_r),
        cg_radius_nm=float(cg_r),
        aa_radius_ang=float(aa_r * 10.0),
        cg_radius_ang=float(cg_r * 10.0),
        restraint_k_per_atom=restraint_k,
        focus_reason=reason,
        confidence_summary=confidence.to_dict(),
        pae_summary=pae_part.to_dict() if pae_part is not None else None,
    )


def _kabsch_align(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Superpose coordinate set `P (N,3)` onto reference `Q (N,3)` via SVD Kabsch alignment."""
    pc = P - np.mean(P, axis=0, keepdims=True)
    qc = Q - np.mean(Q, axis=0, keepdims=True)
    H = pc.T @ qc
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    S = np.diag([1.0, 1.0, -1.0 if d < 0.0 else 1.0])
    R = Vt.T @ S @ U.T
    return (pc @ R.T) + np.mean(Q, axis=0, keepdims=True)


def compute_per_residue_rmsf(
    trajectory_nm: np.ndarray,
    residue_ids: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """Compute Kabsch-aligned per-residue RMSF in Ångströms from `(T, N, 3)` trajectory (nm)."""
    traj = np.asarray(trajectory_nm, dtype=np.float64)
    if traj.ndim != 3 or traj.shape[2] != 3 or traj.shape[0] < 2:
        raise ValueError(f"Expected trajectory shape (T>=2, N, 3), got {traj.shape}")

    T, N, _ = traj.shape
    ref = traj[0]
    aligned = np.empty_like(traj)
    aligned[0] = ref
    for t in range(1, T):
        aligned[t] = _kabsch_align(traj[t], ref)

    mean_pos = np.mean(aligned, axis=0)
    disp_sq = np.sum((aligned - mean_pos[None, :, :]) ** 2, axis=2)  # (T, N) nm^2
    atom_rmsf_ang = np.sqrt(np.mean(disp_sq, axis=0)) * 10.0

    if residue_ids is None:
        return atom_rmsf_ang.astype(np.float64)

    res_arr = np.asarray(residue_ids, dtype=np.int32).reshape(-1)
    if res_arr.size != N:
        raise ValueError(f"residue_ids length ({res_arr.size}) must match N ({N})")
    _, inv = np.unique(res_arr, return_inverse=True)
    sums = np.bincount(inv, weights=atom_rmsf_ang)
    cnts = np.maximum(np.bincount(inv), 1)
    return (sums / cnts).astype(np.float64)


def evaluate_atlas_flexibility(
    target_id: str,
    pred_rmsf_ang: np.ndarray,
    ref_rmsf_ang: np.ndarray,
) -> ATLASFlexibilityReport:
    """Evaluate Pearson r, Spearman rho, and RMSE between predicted and ATLAS reference RMSF."""
    p = np.asarray(pred_rmsf_ang, dtype=np.float64).reshape(-1)
    r = np.asarray(ref_rmsf_ang, dtype=np.float64).reshape(-1)
    m = min(p.size, r.size)
    p, r = p[:m], r[:m]
    if m < 3 or np.std(p) < 1e-12 or np.std(r) < 1e-12:
        pr, sr = 0.0, 0.0
    else:
        pr = float(pearsonr(p, r).statistic)
        sr = float(spearmanr(p, r).statistic)
    rmse = float(np.sqrt(np.mean((p - r) ** 2))) if m > 0 else 0.0
    return ATLASFlexibilityReport(
        target_id=str(target_id),
        n_residues=int(m),
        pearson_r=pr,
        spearman_rho=sr,
        rmse_ang=rmse,
        mean_pred_rmsf_ang=float(np.mean(p)) if m > 0 else 0.0,
        mean_ref_rmsf_ang=float(np.mean(r)) if m > 0 else 0.0,
    )
