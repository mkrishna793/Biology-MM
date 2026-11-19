"""Independent, offline verification of the ``protein_ligand`` chemistry fixture.

The fixture is a solute-only T4 lysozyme (PDB 1XEP, L99A/M102Q) + catechol
complex: AMBER ``prmtop``/``inpcrd`` (AMBER ff14SB protein, OpenFF-parameterized
``UNL`` catechol) plus a PDB written from the same coordinates.  This module
loads the parameter file with ParmEd and OpenMM, checks the structural and
parameter invariants, and recomputes the OpenMM ``NoCutoff`` config-0 energy and
forces against the committed reference (``reference/protein_ligand.json`` and
``reference/protein_ligand.f32``).

It performs no network access and never writes to the fixture.  Run it directly
for a JSON report, or import ``run_verification`` from the pytest module.

    python chemistry/tools/verify_protein_ligand.py
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = REPO_ROOT / "chemistry/fixtures/protein_ligand"
REFERENCE_JSON = REPO_ROOT / "chemistry/reference/protein_ligand.json"
REFERENCE_F32 = REPO_ROOT / "chemistry/reference/protein_ligand.f32"

PARMTOP = FIXTURE_DIR / "protein_ligand.prmtop"
INPCRD = FIXTURE_DIR / "protein_ligand.inpcrd"
PDB = FIXTURE_DIR / "protein_ligand.pdb"

EXPECTED_ATOMS = 2615
EXPECTED_LIGAND = "UNL"
EXPECTED_LIGAND_ATOMS = 14
EXPECTED_LIGAND_ELEMENTS = {"C": 6, "H": 6, "O": 2}
EXPECTED_CHARGE = 8.0
CHARGE_TOL = 0.01

BOND_MEDIAN_NM = (0.09, 0.15)
BOND_MAX_NM = 0.20
LIGAND_GAP_NM = (0.14, 0.35)

ENERGY_TOL_KJMOL = 0.05
FORCE_TOL_KJMOL_PER_NM = 0.05


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Report:
    """Accumulates named pass/fail checks plus measured values."""

    def __init__(self) -> None:
        self.checks: list[dict] = []
        self.values: dict = {}

    def check(self, name: str, ok: bool, detail: str) -> bool:
        self.checks.append({"check": name, "ok": bool(ok), "detail": detail})
        return bool(ok)

    def close(self, name: str, ok: bool, measured, low, high, unit: str = "") -> bool:
        suffix = f" {unit}".rstrip()
        return self.check(
            name,
            ok,
            f"{measured:.6g}{suffix} in [{low:.6g}, {high:.6g}]{suffix}",
        )

    def as_dict(self) -> dict:
        return {
            "fixture": "protein_ligand",
            "ok": all(item["ok"] for item in self.checks),
            "checks": self.checks,
            "values": self.values,
        }


def _mdin(nm: np.ndarray, i: int, j: int) -> float:
    return float(np.linalg.norm(nm[i] - nm[j]))


def run_verification() -> dict:
    """Run every check and return the JSON-serializable report."""
    import openmm
    import parmed as pmd
    from openmm import Context, Platform, VerletIntegrator, unit
    from openmm import app

    report = Report()

    # --- files and recorded hashes -------------------------------------
    reference = json.loads(REFERENCE_JSON.read_text())
    for path in (PARMTOP, INPCRD, PDB):
        report.check(f"file:{path.name}", path.is_file() and path.stat().st_size > 0,
                     f"{path.stat().st_size if path.is_file() else 0} bytes")

    hashes = {path.name: sha256(path) for path in (PARMTOP, INPCRD, PDB)}
    report.values["sha256"] = hashes
    recorded = reference["source"]
    report.check(
        "sha256:prmtop-matches-reference",
        hashes["protein_ligand.prmtop"] == recorded["parameter_sha256"],
        hashes["protein_ligand.prmtop"],
    )
    report.check(
        "sha256:inpcrd-matches-reference",
        hashes["protein_ligand.inpcrd"] == recorded["coordinate_sha256"],
        hashes["protein_ligand.inpcrd"],
    )

    # --- ParmEd topology ------------------------------------------------
    parm = pmd.load_file(str(PARMTOP))
    inpcrd = pmd.load_file(str(INPCRD))
    n_atoms = len(parm.atoms)
    report.values["n_atoms"] = n_atoms
    report.check("parmed:atom-count", n_atoms == EXPECTED_ATOMS,
                 f"{n_atoms} atoms (expected {EXPECTED_ATOMS})")

    charge = math.fsum(atom.charge for atom in parm.atoms)
    report.values["net_charge_e"] = charge
    report.check("charge:net", abs(charge - EXPECTED_CHARGE) <= CHARGE_TOL,
                 f"{charge:.6f} e (expected {EXPECTED_CHARGE:+.2f} +/- {CHARGE_TOL})")

    report.check("topology:no-box", parm.box is None, "prmtop has no periodic box")
    report.check("topology:coordinate-count", inpcrd.natom == n_atoms,
                 f"inpcrd natom={inpcrd.natom}, prmtop atoms={n_atoms}")

    # --- ligand identity ------------------------------------------------
    ligand_residues = [res for res in parm.residues if res.name == EXPECTED_LIGAND]
    report.check("ligand:single-UNL", len(ligand_residues) == 1,
                 f"{len(ligand_residues)} UNL residues")
    if len(ligand_residues) != 1:
        return report.as_dict()

    ligand = ligand_residues[0]
    ligand_atoms = list(ligand.atoms)
    ligand_indices = [atom.idx for atom in ligand_atoms]
    counts: dict[str, int] = {}
    for atom in ligand_atoms:
        counts[atom.element_name] = counts.get(atom.element_name, 0) + 1
    report.values["ligand"] = {
        "residue": ligand.name,
        "n_atoms": len(ligand_atoms),
        "elements": counts,
        "first_atom_index": ligand_indices[0],
        "last_atom_index": ligand_indices[-1],
    }
    report.check("ligand:atom-count", len(ligand_atoms) == EXPECTED_LIGAND_ATOMS,
                 f"{len(ligand_atoms)} atoms (expected {EXPECTED_LIGAND_ATOMS})")
    report.check("ligand:elements", counts == EXPECTED_LIGAND_ELEMENTS,
                 f"{counts} (expected {EXPECTED_LIGAND_ELEMENTS})")

    # --- coordinates ----------------------------------------------------
    # ParmEd supplies topology/charges/bonds; OpenMM reads the coordinate file
    # for the positions used by the reference protocol.
    openmm_inpcrd = app.AmberInpcrdFile(str(INPCRD))
    report.check("openmm:coordinate-count",
                 len(openmm_inpcrd.positions) == n_atoms,
                 f"inpcrd positions={len(openmm_inpcrd.positions)}")
    report.check("inpcrd:no-box", openmm_inpcrd.boxVectors is None,
                 f"boxVectors={openmm_inpcrd.boxVectors}")
    positions_nm = np.array(
        openmm_inpcrd.positions.value_in_unit(unit.nanometer), dtype=np.float64
    )
    report.check("coordinates:finite", bool(np.isfinite(positions_nm).all()),
                 f"shape {positions_nm.shape}")
    report.check("coordinates:nm-scale", float(np.nanmax(np.abs(positions_nm))) < 100.0,
                 "all |x|,|y|,|z| < 100 nm")

    # The inpcrd has no atom names; the PDB is the human-readable sidecar and
    # must list the same atoms in the same prmtop order.
    pdb = pmd.load_file(str(PDB))
    report.check("pdb:atom-count", len(pdb.atoms) == n_atoms,
                 f"{len(pdb.atoms)} atoms (expected {n_atoms})")
    same_names = [a.name for a in pdb.atoms] == [a.name for a in parm.atoms]
    same_residues = [a.residue.name for a in pdb.atoms] == [a.residue.name for a in parm.atoms]
    report.check("pdb:atom-names-order", same_names, "PDB names/order equal prmtop")
    report.check("pdb:residue-names-order", same_residues, "PDB residue names equal prmtop")

    pdb_nm = np.array([[a.xx, a.xy, a.xz] for a in pdb.atoms], dtype=np.float64) * 0.1
    pdb_delta = float(np.abs(pdb_nm - positions_nm).max())
    report.values["pdb_vs_inpcrd_max_delta_nm"] = pdb_delta
    report.check("pdb:coordinates-match-inpcrd", pdb_delta <= 1e-3,
                 f"max delta {pdb_delta:.2e} nm (PDB precision is 1e-3 nm)")

    # --- geometry -------------------------------------------------------
    bond_lengths = np.array(
        [_mdin(positions_nm, bond.atom1.idx, bond.atom2.idx) for bond in parm.bonds],
        dtype=np.float64,
    )
    report.values["bonds"] = {
        "count": int(bond_lengths.size),
        "median_nm": float(np.median(bond_lengths)),
        "min_nm": float(bond_lengths.min()),
        "max_nm": float(bond_lengths.max()),
    }
    median = float(np.median(bond_lengths))
    report.close("geometry:bond-length-median", BOND_MEDIAN_NM[0] <= median <= BOND_MEDIAN_NM[1],
                 median, *BOND_MEDIAN_NM, unit="nm")
    report.check("geometry:bond-length-max", float(bond_lengths.max()) < BOND_MAX_NM,
                 f"{bond_lengths.max():.6g} nm < {BOND_MAX_NM} nm")

    protein_heavy = [a.idx for a in parm.atoms
                     if a.residue.name != EXPECTED_LIGAND and a.element_name != "H"]
    ligand_heavy = [a.idx for a in ligand_atoms if a.element_name != "H"]
    protein_xyz = positions_nm[protein_heavy]
    best = min(
        (float(np.min(np.linalg.norm(protein_xyz - positions_nm[idx], axis=1))), idx)
        for idx in ligand_heavy
    )
    report.values["ligand_protein_min_heavy_distance_nm"] = best[0]
    report.values["ligand_protein_closest_ligand_atom"] = ligand_atoms[
        ligand_indices.index(best[1])
    ].name
    report.close("geometry:ligand-protein-min-heavy-distance",
                 LIGAND_GAP_NM[0] <= best[0] <= LIGAND_GAP_NM[1],
                 best[0], *LIGAND_GAP_NM, unit="nm")

    # --- ligand bond graph (catechol: one aromatic ring, two hydroxyls) --
    local = {atom.idx: i for i, atom in enumerate(ligand_atoms)}
    adjacency = {i: set() for i in range(len(ligand_atoms))}
    for bond in parm.bonds:
        if bond.atom1.idx in local and bond.atom2.idx in local:
            i, j = local[bond.atom1.idx], local[bond.atom2.idx]
            adjacency[i].add(j)
            adjacency[j].add(i)
    n_bonds = sum(len(neigh) for neigh in adjacency.values()) // 2
    degrees = {atom.name: len(adjacency[i]) for i, atom in enumerate(ligand_atoms)}
    carbons = [i for i, atom in enumerate(ligand_atoms) if atom.element_name == "C"]
    oxygens = [i for i, atom in enumerate(ligand_atoms) if atom.element_name == "O"]
    hydrogens = [i for i, atom in enumerate(ligand_atoms) if atom.element_name == "H"]
    ring_ok = len(carbons) == 6 and all(len(adjacency[i] & set(carbons)) == 2 for i in carbons)
    cycle_rank = n_bonds - len(ligand_atoms) + 1
    report.values["ligand_graph"] = {
        "bonds": n_bonds,
        "degrees": degrees,
        "cycle_rank": cycle_rank,
        "carbons_in_ring": len(carbons),
    }
    report.check("ligand:bond-count", n_bonds == 14,
                 f"{n_bonds} bonds (6 ring + 4 C-H + 2 C-O + 2 O-H)")
    report.check("ligand:ring", ring_ok, "six carbons each bonded to two ring carbons")
    report.check("ligand:one-cycle", cycle_rank == 1, f"cycle rank {cycle_rank} (one ring)")
    report.check("ligand:hydroxyls",
                 len(oxygens) == 2 and all(len(adjacency[i]) == 2 for i in oxygens),
                 f"{len(oxygens)} divalent oxygens")
    report.check("ligand:hydrogen-termini",
                 len(hydrogens) == 6 and all(len(adjacency[i]) == 1 for i in hydrogens),
                 f"{len(hydrogens)} monovalent hydrogens")
    c_h = sum(1 for i in carbons for j in adjacency[i] if ligand_atoms[j].element_name == "H")
    c_o = sum(1 for i in carbons for j in adjacency[i] if ligand_atoms[j].element_name == "O")
    report.check("ligand:substitution", c_h == 4 and c_o == 2,
                 f"{c_h} C-H and {c_o} C-O bonds (4 aromatic H, 2 hydroxyls)")

    # --- protein parameter provenance (ff14SB partial charges) -----------
    ff14sb = app.__file__.replace("app/__init__.py", "app/data/amber14/protein.ff14SB.xml")
    templates: dict[str, dict[str, float]] = {}
    for residue in ET.parse(ff14sb).getroot().iter("Residue"):
        templates[residue.get("name")] = {
            atom.get("name"): float(atom.get("charge"))
            for atom in residue.findall("Atom")
        }
    compared = matched_charges = 0
    max_charge_delta = 0.0
    for residue in parm.residues:
        template = templates.get(residue.name)
        if template is None:
            continue
        names = {atom.name for atom in residue.atoms}
        if names != set(template):
            continue
        compared += 1
        delta = max(abs(atom.charge - template[atom.name]) for atom in residue.atoms)
        max_charge_delta = max(max_charge_delta, delta)
        matched_charges += int(delta < 1e-4)
    report.values["ff14sb_residues"] = {
        "compared": compared,
        "matched": matched_charges,
        "max_charge_delta_e": max_charge_delta,
        "templates": str(Path(ff14sb).relative_to(Path(app.__file__).parents[1])),
    }
    report.check(
        "parameters:ff14sb-charges",
        compared >= 150 and matched_charges == compared,
        f"{matched_charges}/{compared} templateable residues equal ff14SB charges "
        f"(max delta {max_charge_delta:.1e} e; N/C termini and the HIS name differ by construction)",
    )

    # --- reference metadata and protocol --------------------------------
    protocol = reference["protocol"]
    report.check("reference:n-atoms", reference["n_atoms"] == n_atoms,
                 f"reference n_atoms={reference['n_atoms']}")
    report.check("reference:configs", reference["n_configs"] == 3,
                 f"n_configs={reference['n_configs']}")
    report.check("reference:no-cutoff",
                 protocol["nonbonded"] == "NoCutoff" and reference["periodic"] is False,
                 f"nonbonded={protocol['nonbonded']}, periodic={reference['periodic']}")
    report.check("reference:no-constraints",
                 protocol["constraints"] == "none" and protocol["rigid_water"] is False,
                 "constraints=None, rigidWater=False")
    report.check("reference:no-box", reference["box_nm"] is None,
                 f"box_nm={reference['box_nm']}")

    # --- OpenMM recomputation -------------------------------------------
    prmtop = app.AmberPrmtopFile(str(PARMTOP))
    system = prmtop.createSystem(
        nonbondedMethod=app.NoCutoff,
        constraints=None,
        rigidWater=False,
        removeCMMotion=False,
        hydrogenMass=None,
    )
    for force in system.getForces():
        kind = type(force).__name__
        force.setForceGroup(1 if kind in ("NonbondedForce", "CustomNonbondedForce") else 0)
        if kind == "NonbondedForce":
            force.setUseDispersionCorrection(False)

    context = Context(
        system,
        VerletIntegrator(0.001),
        Platform.getPlatformByName("CPU"),
        {"Threads": "1", "DeterministicForces": "true"},
    )
    context.setPositions(positions_nm * unit.nanometer)
    state = context.getState(energy=True, forces=True)
    energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    forces = np.asarray(
        state.getForces(asNumpy=True).value_in_unit(
            unit.kilojoule_per_mole / unit.nanometer
        ),
        dtype=np.float64,
    )
    report.check("openmm:energy-finite", math.isfinite(energy), f"{energy:.6f} kJ/mol")
    report.check("openmm:forces-finite", bool(np.isfinite(forces).all()),
                 f"shape {forces.shape}")

    config0 = reference["configs"][0]
    d_energy = abs(energy - config0["energy_kj_mol"])
    d_force = abs(float(np.abs(forces).max()) - config0["force_absmax"])
    report.values["config0"] = {
        "energy_kj_mol": energy,
        "reference_energy_kj_mol": config0["energy_kj_mol"],
        "energy_delta_kj_mol": d_energy,
        "max_force_kj_mol_per_nm": float(np.abs(forces).max()),
        "reference_max_force_kj_mol_per_nm": config0["force_absmax"],
        "max_force_delta_kj_mol_per_nm": d_force,
    }
    report.check("config0:energy-vs-reference", d_energy <= ENERGY_TOL_KJMOL,
                 f"|delta|={d_energy:.3e} <= {ENERGY_TOL_KJMOL} kJ/mol")
    report.check("config0:max-force-vs-reference", d_force <= FORCE_TOL_KJMOL_PER_NM,
                 f"|delta|={d_force:.3e} <= {FORCE_TOL_KJMOL_PER_NM} kJ/mol/nm")

    # --- stored Float32 force file, config 0 ----------------------------
    raw = np.fromfile(REFERENCE_F32, dtype="<f4")
    expected = reference["n_configs"] * 3 * n_atoms
    report.check("f32:length", raw.size == expected,
                 f"{raw.size} floats (expected {expected})")
    if raw.size == expected:
        config0_f32 = raw[: 3 * n_atoms].reshape(n_atoms, 3).astype(np.float64)
        f32_delta = float(np.abs(forces - config0_f32).max())
        report.values["config0_f32"] = {
            "max_abs_delta_kj_mol_per_nm": f32_delta,
            "norm_double": float(np.linalg.norm(forces)),
            "norm_f32": float(np.linalg.norm(config0_f32)),
        }
        report.check("config0:f32-vs-double", f32_delta <= FORCE_TOL_KJMOL_PER_NM,
                     f"max |delta|={f32_delta:.3e} <= {FORCE_TOL_KJMOL_PER_NM}")

    report.values["openmm"] = {
        "installed": openmm.version.version,
        "reference_written_with": reference["openmm_version"],
    }
    return report.as_dict()


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv:
        print(f"usage: {Path(__file__).name}", file=sys.stderr)
        return 2
    report = run_verification()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
