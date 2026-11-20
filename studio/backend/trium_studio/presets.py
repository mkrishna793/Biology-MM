"""Scientific simulation presets for Biology-MM-Studio."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from biology_mm import AtomicSystem, NanostructureBuilder, SystemBuilder

from .engine_core import SimulationSession

PRESET_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "id": "biomolecular-complex",
        "title": "Solvated Biomolecular Complex (Helix + Ligand + Ions + Water)",
        "category": "Biomolecular",
        "description": (
            "16-residue alpha-helical peptide bound to a small-molecule ligand in explicit "
            "NaCl water solvent — enables native Mol* Polymer Cartoon, Ligand, Ion, Water & Sequence layers."
        ),
        "n_atoms": 169,
        "capacity": 169,
        "mode": "metal_verlet",
        "units": "md",
        "default_channel": "force-magnitude",
        "default_palette": "plasma",
        "temperature": 300.0,
        "efield_z": 0.0,
        "gcmc_mu": 0.0,
        "tags": ["Protein Helix", "Ligand", "Ions", "Water", "Cartoon"],
        "code_template": (
            "from trium_studio.engine_core import SimulationSession\n"
            "from trium_studio.presets import build_biomolecular_complex_system\n\n"
            "system, structure_metadata = build_biomolecular_complex_system()\n"
            "session = SimulationSession(\n"
            "    system,\n"
            "    mode='metal_verlet',\n"
            "    dt=0.001,\n"
            "    temperature=300.0,\n"
            "    rcut=1.0,\n"
            "    alpha_dsf=0.25,\n"
            "    structure_metadata=structure_metadata,\n"
            ")\n"
            "print(f'Complex atoms={system.n_atoms}, bonds={len(system.bonds)}, entities={len(structure_metadata[\"entities\"])}')\n"
        ),
    },
    {
        "id": "cnt-tersoff-tensile",
        "title": "CNT (10,10) Tersoff Tensile Strain",
        "category": "Nanomechanics",
        "description": (
            "Armchair (10,10) single-walled carbon nanotube (320 C atoms) under axial "
            "tensile strain with reactive Morse-Tersoff bond-order & stress fields."
        ),
        "n_atoms": 320,
        "capacity": 320,
        "mode": "cnt_tersoff",
        "units": "md",
        "default_channel": "bond-order",
        "default_palette": "bond-viridis",
        "temperature": 300.0,
        "efield_z": 0.0,
        "gcmc_mu": 0.0,
        "tags": ["CNT", "Tersoff", "Tensile", "Bond Order"],
        "code_template": (
            "from biology_mm import NanostructureBuilder\n"
            "from trium_studio.engine_core import SimulationSession\n\n"
            "system = NanostructureBuilder.carbon_nanotube(10, 10, n_cells=8)\n"
            "session = SimulationSession(\n"
            "    system,\n"
            "    mode='cnt_tersoff',\n"
            "    dt=0.001,\n"
            "    temperature=300.0,\n"
            "    strain_rate=0.02,\n"
            ")\n"
            "print(f'CNT atoms={system.n_atoms}, bonds={len(system.bonds)}')\n"
        ),
    },
    {
        "id": "nanopore-gcmc-ion",
        "title": "Graphene Nanopore GCMC + NEMD Ion Flux",
        "category": "Nanofluidics",
        "description": (
            "Circular graphene nanopore membrane with live Grand Canonical Monte Carlo "
            "ion/gas insertion-deletion and Z-axis electric-field NEMD transport."
        ),
        "n_atoms": 227,
        "capacity": 355,
        "mode": "nanopore_gcmc_nemd",
        "units": "md",
        "default_channel": "ternary-state",
        "default_palette": "ternary-triad",
        "temperature": 300.0,
        "efield_z": 0.5,
        "gcmc_mu": -2.0,
        "tags": ["GCMC", "NEMD", "Nanopore", "Ternary"],
        "code_template": (
            "import numpy as np\n"
            "from biology_mm import AtomicSystem, NanostructureBuilder\n"
            "from trium_studio.engine_core import SimulationSession\n\n"
            "membrane = NanostructureBuilder.nanopore_membrane(10, 6, pore_radius_nm=0.45, box_z_nm=4.0)\n"
            "lx, ly, lz = membrane.box_lengths\n"
            "guest_pos = np.array([\n"
            "    [0.25 * lx, 0.25 * ly, 0.8], [0.75 * lx, 0.25 * ly, 0.9],\n"
            "    [0.25 * lx, 0.75 * ly, 3.1], [0.75 * lx, 0.75 * ly, 3.2],\n"
            "    [0.50 * lx, 0.50 * ly, 1.0], [0.50 * lx, 0.50 * ly, 3.0],\n"
            "], dtype=np.float64)\n"
            "pos = np.vstack([membrane.positions, guest_pos])\n"
            "masses = np.concatenate([membrane.masses, np.array([22.99, 35.45, 22.99, 35.45, 22.99, 35.45])])\n"
            "charges = np.concatenate([membrane.charges, np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])])\n"
            "names = list(membrane.names) + ['Na', 'Cl', 'Na', 'Cl', 'Na', 'Cl']\n"
            "system = AtomicSystem(pos, masses, charges, membrane.box_lengths, names=names, bonds=membrane.bonds, units='md')\n"
            "session = SimulationSession(\n"
            "    system,\n"
            "    mode='nanopore_gcmc_nemd',\n"
            "    dt=0.002,\n"
            "    temperature=300.0,\n"
            "    efield_z=0.5,\n"
            "    gcmc_mu=-2.0,\n"
            "    max_atoms_cap=system.n_atoms + 128,\n"
            ")\n"
            "print(f'Nanopore active={session.n_real}, capacity={session.capacity}')\n"
        ),
    },
    {
        "id": "argon-metal-verlet",
        "title": "Argon FCC Fluid (Metal 4 Verlet)",
        "category": "GPU Liquids",
        "description": (
            "10,976-atom Lennard-Jones FCC argon system running on Apple Metal 4 "
            "batched Velocity-Verlet kernels with real-time force-magnitude coloring."
        ),
        "n_atoms": 10976,
        "capacity": 10976,
        "mode": "metal_verlet",
        "units": "reduced",
        "default_channel": "force-magnitude",
        "default_palette": "plasma",
        "temperature": 1.2,
        "efield_z": 0.0,
        "gcmc_mu": 0.0,
        "tags": ["Metal 4", "Verlet", "LJ", "11k Atoms"],
        "code_template": (
            "from biology_mm import SystemBuilder\n"
            "from trium_studio.engine_core import SimulationSession\n\n"
            "system = SystemBuilder.fcc_lattice(14, rho=0.84, mass=1.0, name='Ar', units='reduced')\n"
            "session = SimulationSession(\n"
            "    system,\n"
            "    mode='metal_verlet',\n"
            "    dt=0.005,\n"
            "    temperature=1.2,\n"
            "    sigma=1.0,\n"
            "    epsilon=1.0,\n"
            "    rcut=2.5,\n"
            ")\n"
            "print(f'Argon FCC atoms={system.n_atoms}, backend={session.hardware.backend}')\n"
        ),
    },
    {
        "id": "graphene-custom-spline",
        "title": "Graphene Sheet (Custom Spline U(r))",
        "category": "Custom Potentials",
        "description": (
            "Periodic 384-atom graphene sheet driven by a live C2 cubic-spline "
            "tabulated Morse potential U(r) that can be hot-swapped from Physics Builder."
        ),
        "n_atoms": 384,
        "capacity": 384,
        "mode": "custom_spline",
        "units": "md",
        "default_channel": "atomic-stress",
        "default_palette": "stress-inferno",
        "temperature": 300.0,
        "efield_z": 0.0,
        "gcmc_mu": 0.0,
        "tags": ["Graphene", "Cubic Spline", "Custom U(r)"],
        "code_template": (
            "from biology_mm import NanostructureBuilder\n"
            "from trium_studio.engine_core import SimulationSession\n\n"
            "system = NanostructureBuilder.graphene_sheet(12, 8)\n"
            "session = SimulationSession(\n"
            "    system,\n"
            "    mode='custom_spline',\n"
            "    dt=0.001,\n"
            "    temperature=300.0,\n"
            "    rcut=0.25,\n"
            "    custom_expr='De * (1.0 - exp(-a * (r - r0)))**2',\n"
            "    custom_params={'De': 480.0, 'a': 21.0, 'r0': 0.142},\n"
            ")\n"
            "print(f'Graphene atoms={system.n_atoms}, bonds={len(system.bonds)}')\n"
        ),
    },
    {
        "id": "silicon-crystal-phonons",
        "title": "Silicon Diamond Crystal Phonons",
        "category": "Solid State",
        "description": (
            "4x4x4 diamond-cubic Silicon crystal (512 atoms, 1,024 tetrahedral bonds) "
            "visualizing per-atom kinetic temperature and lattice phonon modes."
        ),
        "n_atoms": 512,
        "capacity": 512,
        "mode": "custom_spline",
        "units": "md",
        "default_channel": "kinetic-temp",
        "default_palette": "thermal-div",
        "temperature": 300.0,
        "efield_z": 0.0,
        "gcmc_mu": 0.0,
        "tags": ["Silicon", "Diamond", "Phonons", "Thermal"],
        "code_template": (
            "from biology_mm import NanostructureBuilder\n"
            "from trium_studio.engine_core import SimulationSession\n\n"
            "system = NanostructureBuilder.silicon_diamond(4, 4, 4)\n"
            "session = SimulationSession(\n"
            "    system,\n"
            "    mode='custom_spline',\n"
            "    dt=0.001,\n"
            "    temperature=300.0,\n"
            "    rcut=0.32,\n"
            "    custom_expr='De * (1.0 - exp(-a * (r - r0)))**2',\n"
            "    custom_params={'De': 220.0, 'a': 15.0, 'r0': 0.235},\n"
            ")\n"
            "print(f'Silicon atoms={system.n_atoms}, bonds={len(system.bonds)}')\n"
        ),
    },
    {
        "id": "argon-100k-gpu-scale",
        "title": "Argon 108k GPU Scale Benchmark",
        "category": "GPU Scale",
        "description": (
            "108,000-atom FCC Argon benchmark (30x30x30 unit cells) with int16 "
            "fractional TriumStream v1 compression and Morton-tiled Metal 4 execution."
        ),
        "n_atoms": 108000,
        "capacity": 108000,
        "mode": "metal_verlet",
        "units": "reduced",
        "default_channel": "force-magnitude",
        "default_palette": "plasma",
        "temperature": 1.2,
        "efield_z": 0.0,
        "gcmc_mu": 0.0,
        "tags": ["108k Atoms", "Metal 4", "int16", "Benchmark"],
        "code_template": (
            "from biology_mm import SystemBuilder\n"
            "from trium_studio.engine_core import SimulationSession\n\n"
            "system = SystemBuilder.fcc_lattice(30, rho=0.84, mass=1.0, name='Ar', units='reduced')\n"
            "session = SimulationSession(\n"
            "    system,\n"
            "    mode='metal_verlet',\n"
            "    dt=0.005,\n"
            "    temperature=1.2,\n"
            "    sigma=1.0,\n"
            "    epsilon=1.0,\n"
            "    rcut=2.5,\n"
            "    use_int16=True,\n"
            ")\n"
            "print(f'100k GPU Scale atoms={system.n_atoms}, backend={session.hardware.backend}')\n"
        ),
    },
]

_PRESET_MAP: Dict[str, Dict[str, Any]] = {p["id"]: p for p in PRESET_DEFINITIONS}


def list_presets() -> List[Dict[str, Any]]:
    """Return copies of all preset metadata dictionaries matching `PresetInfo`."""
    return [dict(p) for p in PRESET_DEFINITIONS]


def get_preset(preset_id: str) -> Dict[str, Any]:
    """Return metadata for `preset_id` or raise `KeyError`."""
    if preset_id not in _PRESET_MAP:
        raise KeyError(f"Unknown preset '{preset_id}'")
    return dict(_PRESET_MAP[preset_id])


def _build_nanopore_gcmc_system() -> tuple[AtomicSystem, np.ndarray]:
    membrane = NanostructureBuilder.nanopore_membrane(10, 6, pore_radius_nm=0.45, box_z_nm=4.0)
    lx, ly, lz = membrane.box_lengths
    rng = np.random.default_rng(42)

    n_guests = 12
    guest_pos = np.empty((n_guests, 3), dtype=np.float64)
    guest_masses = np.empty(n_guests, dtype=np.float64)
    guest_charges = np.empty(n_guests, dtype=np.float64)
    guest_names: List[str] = []
    guest_ternary = np.empty(n_guests, dtype=np.float32)

    for i in range(n_guests):
        is_cation = (i % 2) == 0
        z_val = rng.uniform(0.3, 1.3) if (i < n_guests // 2) else rng.uniform(2.7, lz - 0.3)
        guest_pos[i] = [rng.uniform(0.15 * lx, 0.85 * lx), rng.uniform(0.15 * ly, 0.85 * ly), z_val]
        if is_cation:
            guest_masses[i] = 22.99
            guest_charges[i] = 1.0
            guest_names.append("Na")
            guest_ternary[i] = 1.0
        else:
            guest_masses[i] = 35.45
            guest_charges[i] = -1.0
            guest_names.append("Cl")
            guest_ternary[i] = -1.0

    pos = np.vstack([membrane.positions, guest_pos])
    masses = np.concatenate([membrane.masses, guest_masses])
    charges = np.concatenate([membrane.charges, guest_charges])
    names = list(membrane.names) + guest_names
    sys = AtomicSystem(
        positions=pos,
        masses=masses,
        charges=charges,
        box_lengths=membrane.box_lengths,
        names=names,
        bonds=membrane.bonds,
        units="md",
    )
    return sys, guest_ternary


def build_biomolecular_complex_system() -> tuple[AtomicSystem, Dict[str, Any]]:
    """Build a solvated 16-residue alpha-helix + aromatic ligand + Na+/Cl- ions + explicit HOH waters."""
    seq = [
        "ALA", "GLU", "LEU", "LYS", "VAL", "THR", "ALA", "ILE",
        "GLY", "SER", "LEU", "ARG", "ALA", "VAL", "ASP", "ALA",
    ]
    pos_list: List[List[float]] = []
    masses: List[float] = []
    charges: List[float] = []
    names: List[str] = []
    bonds: List[tuple[int, int]] = []

    group_pdb: List[str] = []
    atom_id: List[str] = []
    comp_id: List[str] = []
    asym_id: List[str] = []
    entity_id: List[str] = []
    seq_id: List[int] = []

    elem_mass = {"N": 14.007, "C": 12.011, "O": 15.999, "H": 1.008, "Na": 22.99, "Cl": 35.45}

    def add_atom(
        xyz: List[float],
        sym: str,
        chg: float,
        g_pdb: str,
        a_lbl: str,
        c_lbl: str,
        ch_lbl: str,
        e_id: str,
        s_num: int,
    ) -> int:
        idx = len(pos_list)
        pos_list.append(xyz)
        masses.append(elem_mass.get(sym, 12.011))
        charges.append(chg)
        names.append(sym)
        group_pdb.append(g_pdb)
        atom_id.append(a_lbl)
        comp_id.append(c_lbl)
        asym_id.append(ch_lbl)
        entity_id.append(e_id)
        seq_id.append(s_num)
        return idx

    # Entity 1: Chain A 16-residue right-handed alpha helix along Z
    lx, ly, lz = 3.8, 3.8, 4.2
    cx, cy, z0 = 1.55, 1.85, 0.85
    dtheta = 1.745329  # 100 degrees per residue
    dz = 0.15  # 1.5 A rise per residue
    prev_c: Optional[int] = None

    for r_i, resname in enumerate(seq):
        theta = r_i * dtheta
        z_ca = z0 + r_i * dz
        r_ca = 0.230
        n_xyz = [cx + 0.210 * np.cos(theta - 0.36), cy + 0.210 * np.sin(theta - 0.36), z_ca - 0.075]
        ca_xyz = [cx + r_ca * np.cos(theta), cy + r_ca * np.sin(theta), z_ca]
        c_xyz = [cx + 0.205 * np.cos(theta + 0.36), cy + 0.205 * np.sin(theta + 0.36), z_ca + 0.065]
        o_xyz = [cx + 0.220 * np.cos(theta + 0.42), cy + 0.220 * np.sin(theta + 0.42), z_ca + 0.185]

        res_num = r_i + 1
        i_n = add_atom(n_xyz, "N", -0.35, "ATOM", "N", resname, "A", "1", res_num)
        i_ca = add_atom(ca_xyz, "C", 0.12, "ATOM", "CA", resname, "A", "1", res_num)
        i_c = add_atom(c_xyz, "C", 0.45, "ATOM", "C", resname, "A", "1", res_num)
        i_o = add_atom(o_xyz, "O", -0.45, "ATOM", "O", resname, "A", "1", res_num)

        bonds.extend([(i_n, i_ca), (i_ca, i_c), (i_c, i_o)])
        if prev_c is not None:
            bonds.append((prev_c, i_n))
        prev_c = i_c

        if resname != "GLY":
            cb_xyz = [cx + 0.375 * np.cos(theta - 0.05), cy + 0.375 * np.sin(theta - 0.05), z_ca - 0.025]
            i_cb = add_atom(cb_xyz, "C", 0.05, "ATOM", "CB", resname, "A", "1", res_num)
            bonds.append((i_ca, i_cb))

    # Entity 2: Chain B bound fused-ring ligand (12 heavy atoms) docked next to helix pocket
    lig_cx, lig_cy, lig_cz = 2.48, 2.02, 2.05
    lig_coords = [
        (0.000, 0.000, 0.000, "C", 0.05),
        (0.140, 0.000, 0.000, "C", 0.05),
        (0.210, 0.121, 0.000, "C", -0.05),
        (0.140, 0.242, 0.000, "C", 0.10),
        (0.000, 0.242, 0.000, "C", -0.05),
        (-0.070, 0.121, 0.000, "C", 0.05),
        (0.210, -0.121, 0.020, "C", 0.12),
        (0.345, -0.095, 0.025, "N", -0.40),
        (0.350, 0.048, 0.015, "C", 0.15),
        (0.220, 0.365, -0.020, "O", -0.45),
        (-0.205, 0.121, 0.020, "N", -0.35),
        (-0.285, 0.240, 0.030, "C", 0.25),
    ]
    lig_base = len(pos_list)
    for k, (dx, dy, dz_l, sym, chg) in enumerate(lig_coords, start=1):
        add_atom(
            [lig_cx + dx, lig_cy + dy, lig_cz + dz_l],
            sym,
            chg,
            "HETATM",
            f"{sym}{k}",
            "LIG",
            "B",
            "2",
            1,
        )
    lig_edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (1, 6), (6, 7), (7, 8), (8, 2), (3, 9), (5, 10), (10, 11)]
    for u, v in lig_edges:
        bonds.append((lig_base + u, lig_base + v))

    # Entity 3: Chain C mobile Na+ and Cl- ions in solvent
    ion_sites = [
        ("Na", 1.0, [0.80, 0.90, 1.35]),
        ("Cl", -1.0, [2.95, 1.05, 1.45]),
        ("Na", 1.0, [0.75, 2.95, 2.55]),
        ("Cl", -1.0, [2.90, 2.95, 2.75]),
        ("Na", 1.0, [2.85, 1.35, 3.30]),
        ("Cl", -1.0, [0.95, 1.40, 3.15]),
    ]
    for s_i, (sym, chg, xyz) in enumerate(ion_sites, start=1):
        add_atom(xyz, sym, chg, "HETATM", sym.upper(), sym.upper(), "C", "3", s_i)

    # Entity 4: Chain D explicit 3-site HOH solvent molecules (24 waters)
    existing = np.asarray(pos_list, dtype=np.float64)
    rng = np.random.default_rng(2026)
    waters_added = 0
    w_seq = 0
    attempts = 0
    while waters_added < 24 and attempts < 2000:
        attempts += 1
        cand = np.array([
            rng.uniform(0.35, lx - 0.35),
            rng.uniform(0.35, ly - 0.35),
            rng.uniform(0.35, lz - 0.35),
        ])
        d_min = float(np.min(np.linalg.norm(existing - cand[None, :], axis=1)))
        if d_min < 0.34:
            continue
        w_seq += 1
        theta_w = rng.uniform(0, 2.0 * np.pi)
        h1 = cand + 0.096 * np.array([np.cos(theta_w), np.sin(theta_w), 0.0])
        h2 = cand + 0.096 * np.array([np.cos(theta_w + 1.824), np.sin(theta_w + 1.824), 0.0])
        io = add_atom(cand.tolist(), "O", -0.82, "HETATM", "O", "HOH", "D", "4", w_seq)
        ih1 = add_atom(h1.tolist(), "H", 0.41, "HETATM", "H1", "HOH", "D", "4", w_seq)
        ih2 = add_atom(h2.tolist(), "H", 0.41, "HETATM", "H2", "HOH", "D", "4", w_seq)
        bonds.extend([(io, ih1), (io, ih2)])
        existing = np.vstack([existing, cand[None, :], h1[None, :], h2[None, :]])
        waters_added += 1

    system = AtomicSystem(
        positions=np.asarray(pos_list, dtype=np.float64),
        masses=np.asarray(masses, dtype=np.float64),
        charges=np.asarray(charges, dtype=np.float64),
        box_lengths=(lx, ly, lz),
        names=names,
        bonds=bonds,
        units="md",
    )
    meta: Dict[str, Any] = {
        "group_pdb": group_pdb,
        "atom_id": atom_id,
        "comp_id": comp_id,
        "asym_id": asym_id,
        "entity_id": entity_id,
        "seq_id": seq_id,
        "entities": [
            {"id": "1", "type": "polymer", "description": "Alpha-Helical Peptide (Chain A)"},
            {"id": "2", "type": "non-polymer", "description": "Bound Aromatic Ligand (LIG)"},
            {"id": "3", "type": "non-polymer", "description": "Mobile Na+/Cl- Electrolyte"},
            {"id": "4", "type": "water", "description": "Explicit HOH Solvent"},
        ],
    }
    return system, meta


def create_preset_session(preset_id: str) -> SimulationSession:
    """Instantiate a ready-to-stream `SimulationSession` for the given `preset_id`."""
    if preset_id not in _PRESET_MAP:
        raise KeyError(f"Unknown preset '{preset_id}'")

    if preset_id == "biomolecular-complex":
        system, meta = build_biomolecular_complex_system()
        sess = SimulationSession(
            system,
            mode="metal_verlet",
            dt=0.001,
            temperature=300.0,
            rcut=1.0,
            alpha_dsf=0.25,
            structure_metadata=meta,
        )
        sess.minimize()
        if sess.get_metrics().get("max_force", 0.0) > 5000.0:
            sess.relax_clashes(max_steps=30)
        return sess

    if preset_id == "cnt-tersoff-tensile":
        system = NanostructureBuilder.carbon_nanotube(10, 10, n_cells=8)
        return SimulationSession(
            system,
            mode="cnt_tersoff",
            dt=0.001,
            temperature=300.0,
            strain_rate=0.02,
        )

    if preset_id == "nanopore-gcmc-ion":
        system, guest_ternary = _build_nanopore_gcmc_system()
        n_mem = system.n_atoms - guest_ternary.size
        cap = system.n_atoms + 128
        session = SimulationSession(
            system,
            mode="nanopore_gcmc_nemd",
            dt=0.002,
            temperature=300.0,
            efield_z=0.5,
            gcmc_mu=-2.0,
            max_atoms_cap=cap,
        )
        session.ternary_states[n_mem : system.n_atoms] = guest_ternary
        return session

    if preset_id == "argon-metal-verlet":
        system = SystemBuilder.fcc_lattice(14, rho=0.84, mass=1.0, name="Ar", units="reduced")
        return SimulationSession(
            system,
            mode="metal_verlet",
            dt=0.005,
            temperature=1.2,
            sigma=1.0,
            epsilon=1.0,
            rcut=2.5,
        )

    if preset_id == "graphene-custom-spline":
        system = NanostructureBuilder.graphene_sheet(12, 8)
        return SimulationSession(
            system,
            mode="custom_spline",
            dt=0.001,
            temperature=300.0,
            rcut=0.25,
            custom_expr="De * (1.0 - exp(-a * (r - r0)))**2",
            custom_params={"De": 480.0, "a": 21.0, "r0": 0.142},
        )

    if preset_id == "silicon-crystal-phonons":
        system = NanostructureBuilder.silicon_diamond(4, 4, 4)
        return SimulationSession(
            system,
            mode="custom_spline",
            dt=0.001,
            temperature=300.0,
            rcut=0.32,
            custom_expr="De * (1.0 - exp(-a * (r - r0)))**2",
            custom_params={"De": 220.0, "a": 15.0, "r0": 0.235},
        )

    if preset_id == "argon-100k-gpu-scale":
        system = SystemBuilder.fcc_lattice(30, rho=0.84, mass=1.0, name="Ar", units="reduced")
        return SimulationSession(
            system,
            mode="metal_verlet",
            dt=0.005,
            temperature=1.2,
            sigma=1.0,
            epsilon=1.0,
            rcut=2.5,
            use_int16=True,
        )

    raise KeyError(f"Unhandled preset '{preset_id}'")
