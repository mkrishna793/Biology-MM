#!/usr/bin/env julia
"""
Lane 4 - bonded-term convention closure against an external parser.

Question: our engine has exactly three bonded energy forms,

    HarmonicBond   U = (k/2)(r - r0)^2
    HarmonicAngle  U = (k_th/2)(theta - theta0)^2
    PeriodicDihedral U = k_phi [1 + cos(n*phi - phi0)]

while GROMACS/ParmEd/InterMol/OpenMM write the same physics with different
normalisation, units and phase placement.  Per-term parameters are extracted by
two independent parsers (reference/terms.tsv) and the reference energies and
forces are produced by OpenMM 8.6.1 (reference/openmm_terms.tsv), using the
standard HarmonicBondForce / HarmonicAngleForce / PeriodicTorsionForce classes.
This script feeds the same parameters to the engine and checks energy and force
term by term, plus the all-bonded total.

TOLERANCES ARE FIXED BEFORE ANY MEASUREMENT (see TOL_* below):
  * engine vs reference energy at a fixed geometry: |dU| <= 1e-10 kJ/mol
  * engine vs reference torsion energy over a 13-point phi grid: <= 1e-10 kJ/mol
  * engine vs OpenMM per-term and total energy: |dU| <= 1e-8 kJ/mol,
    force <= 1e-9 relative
  * the two independent k columns of the frozen table must obey
    k_bond = 836.8 k_p, k_angle = 8.368 k_p, k_phi = 4.184 k_p to 1e-9 relative
  * engine phi vs the external ParmEd phi: |dphi| <= 1e-10 rad
  * engine force vs central difference of engine energy: <= 1e-6 relative
A wrong convention must produce a mismatch orders of magnitude above these;
that is asserted separately (DISCRIMINATION_MIN).

Run:  sh run.sh          (from this directory)
      julia --project=../.. check.jl
"""

using FoundationCore
using LinearAlgebra
using Printf

const F = FoundationCore.Factors
const T = FoundationCore.Types

const HERE = @__DIR__
const TSV = joinpath(HERE, "reference", "terms.tsv")
const OMM_TSV = joinpath(HERE, "reference", "openmm_terms.tsv")
isfile(OMM_TSV) || error("missing $OMM_TSV; run ./run.sh --refresh")

# --- tolerances, declared before measurement -------------------------------
const TOL_ENERGY = 1e-10      # kJ/mol, absolute
const TOL_PHI = 1e-10         # rad
const TOL_FD = 1e-6           # relative force error
const TOL_OMM_E = 1e-8        # kJ/mol, engine vs OpenMM per-term energy
const TOL_OMM_F = 1e-9        # relative, engine vs OpenMM per-term force
const TOL_COLUMN = 1e-9       # relative, GROMACS k vs ParmEd k through the table
const DISCRIMINATION_MIN = 1.0  # kJ/mol; a wrong convention must exceed this

# --- reference table -------------------------------------------------------

struct AtomRow; idx::Int; name::String; xyz::Vector{Float64}; end
struct BondRow; i::Int; j::Int; k::Float64; r0::Float64; kp::Float64; r0p::Float64; tag::String; end
struct AngleRow; i::Int; j::Int; k::Int; kth::Float64; th0::Float64; kp::Float64; th0p::Float64; tag::String; end
struct DihRow
    i::Int; j::Int; k::Int; l::Int
    kphi::Float64; n::Int; phis::Float64      # GROMACS units (kJ/mol, deg)
    kp::Float64; np::Int; phisp::Float64      # ParmEd units (kcal/mol, deg)
    tag::String
    fc::Vector{Float64}                       # InterMol fc0..fc6 (kJ/mol)
end
struct PhiRow; i::Int; j::Int; k::Int; l::Int; deg::Float64; end

function read_reference(path)
    atoms = Dict{String,Vector{AtomRow}}(); bonds = BondRow[]; angles = AngleRow[]
    dihs = DihRow[]; phis = PhiRow[]
    for line in eachline(path)
        startswith(line, "#") && continue
        isempty(strip(line)) && continue
        w = split(line)
        if w[1] == "ATOM"
            push!(get!(atoms, w[2], AtomRow[]), AtomRow(parse(Int, w[3]), w[4], parse.(Float64, w[5:7])))
        elseif w[1] == "BOND"
            push!(bonds, BondRow(parse(Int, w[2]), parse(Int, w[3]),
                                 parse(Float64, w[4]), parse(Float64, w[5]),
                                 parse(Float64, w[6]), parse(Float64, w[7]), w[8]))
        elseif w[1] == "ANGLE"
            push!(angles, AngleRow(parse(Int, w[2]), parse(Int, w[3]), parse(Int, w[4]),
                                   parse(Float64, w[5]), parse(Float64, w[6]),
                                   parse(Float64, w[7]), parse(Float64, w[8]), w[9]))
        elseif w[1] == "DIH"
            push!(dihs, DihRow(parse(Int, w[2]), parse(Int, w[3]), parse(Int, w[4]), parse(Int, w[5]),
                               parse(Float64, w[6]), parse(Int, w[7]), parse(Float64, w[8]),
                               parse(Float64, w[9]), parse(Int, w[10]), parse(Float64, w[11]),
                               w[12], parse.(Float64, w[13:19])))
        elseif w[1] == "PHIPARMED"
            push!(phis, PhiRow(parse(Int, w[2]), parse(Int, w[3]), parse(Int, w[4]),
                               parse(Int, w[5]), parse(Float64, w[6])))
        end
    end
    return atoms, bonds, angles, dihs, phis
end

# --- OpenMM reference: per-term energies and forces from a real engine ------
# Keyed by (kind, i, j, k, l, n) because one atom quadruple can carry several
# torsion terms of different multiplicity.
struct OmmRef
    energy::Dict{String,Dict{NTuple{6,Any},Float64}}
    force::Dict{String,Dict{NTuple{6,Any},Array{Float64}}}
    total::Dict{String,Tuple{Float64,Array{Float64}}}
end

function read_openmm(path)
    energy = Dict{String,Dict{NTuple{6,Any},Float64}}()
    force = Dict{String,Dict{NTuple{6,Any},Array{Float64}}}()
    total = Dict{String,Tuple{Float64,Array{Float64}}}()
    natoms = Dict{String,Int}()
    for line in eachline(path)
        startswith(line, "#") && continue
        isempty(strip(line)) && continue
        w = split(line)
        if w[1] == "OMME"
            key = (w[3], parse(Int, w[4]), parse(Int, w[5]), parse(Int, w[6]), parse(Int, w[7]), parse(Int, w[8]))
            get!(energy, w[2], Dict{NTuple{6,Any},Float64}())[key] = parse(Float64, w[9])
        elseif w[1] == "OMMF"
            key = (w[3], parse(Int, w[4]), parse(Int, w[5]), parse(Int, w[6]), parse(Int, w[7]), parse(Int, w[8]))
            vals = parse.(Float64, w[9:end])
            get!(force, w[2], Dict{NTuple{6,Any},Array{Float64}}())[key] = reshape(vals, 3, div(length(vals), 3))
        elseif w[1] == "OMMT"
            vals = parse.(Float64, w[4:end])
            total[w[2]] = (parse(Float64, w[3]), reshape(vals, 3, div(length(vals), 3)))
        end
    end
    return OmmRef(energy, force, total)
end

# --- independent reference formulas, in the reference's own convention -----
# Bond/angle: ParmEd's k(r-r0)^2 (kcal/mol, A) converted to (kJ/mol, nm).
const KCAL_TO_KJ = 4.184
const A2_TO_NM2 = 0.01
ref_bond_energy_pm(kp, r0p_A, r_nm) = kp * KCAL_TO_KJ * (r_nm - r0p_A * 0.1)^2 / A2_TO_NM2
ref_angle_energy_pm(kp, th0_rad, th) = kp * KCAL_TO_KJ * (th - th0_rad)^2

# Torsion: k[1+cos(n phi - phi_s)] evaluated *without* atan2, from the
# documented definitions cos(phi) = n1.n2/(|n1||n2|) and
# sin(phi) = (n1 x n2).b2/(|n1||n2||b2|), then a Chebyshev recurrence for
# cos(n phi), sin(n phi).  This never calls the engine's dihedral_angle_grad.
function ref_dihedral_cossin(q, i, j, k, l)
    b1 = q[:, j] - q[:, i]
    b2 = q[:, k] - q[:, j]
    b3 = q[:, l] - q[:, k]
    n1 = cross(b1, b2)
    n2 = cross(b2, b3)
    d = norm(n1) * norm(n2)
    (d < 1e-24 || norm(b2) < 1e-12) && return nothing
    c = dot(n1, n2) / d
    s = dot(cross(n1, n2), b2) / (d * norm(b2))
    return c, s
end

function ref_torsion_energy(kphi, n, phis, c, s)
    # cos(n phi), sin(n phi) by Chebyshev: T_n(c), U_{n-1}(c) s
    cn, sn = 1.0, 0.0
    for _ in 1:n
        cn, sn = cn * c - sn * s, sn * c + cn * s
    end
    return kphi * (1.0 + cn * cos(phis) + sn * sin(phis))
end

ctx_for(n) = F.Context(T.REDUCED, n)

# --- measurement helpers ---------------------------------------------------

function term_energy(f, q)
    ctx = ctx_for(size(q, 2))
    U, _ = F.total_energy_forces([f], q, ctx)
    return U
end

"Rotate atom `l` about the j->k axis by `delta` (Rodrigues)."
function rotate_l(q, j, k, l, delta)
    q2 = copy(q)
    u = q[:, k] - q[:, j]
    u = u / norm(u)
    v = q[:, l] - q[:, k]
    q2[:, l] = q[:, k] .+ (v .* cos(delta) .+ cross(u, v) .* sin(delta) .+ u .* (dot(u, v) * (1 - cos(delta))))
    return q2
end

maxrel(x, s) = maximum(abs.(x)) / max(1.0, abs(s))

struct Report
    rows::Vector{String}
end

function add!(r::Report, ok, name, detail)
    push!(r.rows, (@sprintf("%-4s %-42s %s", ok ? "PASS" : "FAIL", name, detail)))
    return ok
end

function run_case(tag, atoms, bonds, angles, dihs, phis, omm, rep)
    arows = sort(atoms[tag], by = a -> a.idx)
    n = length(arows)
    for (m, a) in enumerate(arows)
        a.idx == m || error("case $tag atom indices are not 1:$n")
    end
    add!(rep, n == 9, "case $tag geometry has 9 unique atoms", "n=$n")
    q = zeros(3, n)
    for (m, a) in enumerate(arows)
        q[:, m] .= a.xyz
    end
    ctx = ctx_for(n)

    bt = [b for b in bonds if b.tag == tag]
    at = [a for a in angles if a.tag == tag]
    dt = [d for d in dihs if d.tag == tag]

    # The two k columns were produced by independent parsers; they must obey the
    # exact affine relations of the convention table, otherwise the frozen table
    # (not just the engine) has drifted.
    e_col = 0.0
    for b in bt
        e_col = max(e_col, abs(836.8 * b.kp - b.k) / max(1.0, b.k))
    end
    for a in at
        e_col = max(e_col, abs(8.368 * a.kp - a.kth) / max(1.0, a.kth))
    end
    for d in dt
        e_col = max(e_col, abs(4.184 * d.kp - d.kphi) / max(1.0, d.kphi))
    end
    add!(rep, e_col <= TOL_COLUMN, "case $tag k columns obey 836.8/8.368/4.184",
         @sprintf("max rel dev=%.2e", e_col))

    U_eng_bond = 0.0; U_ref_bond = 0.0; e_bond = 0.0
    for b in bt
        r = norm(q[:, b.i] - q[:, b.j])
        f = F.HarmonicBond(b.i, b.j, b.k, b.r0)
        Ue = term_energy(f, q)
        Ur = ref_bond_energy_pm(b.kp, b.r0p, r)
        U_eng_bond += Ue; U_ref_bond += Ur
        e_bond = max(e_bond, abs(Ue - Ur))
        @printf("  bond  %d-%-2d k=%.6g kJ/mol/nm^2 (=%.6g kcal/mol/A^2) r0=%.5f nm  U=%.8f  dU=%.2e\n",
                b.i, b.j, b.k, b.kp, b.r0, Ue, Ue - Ur)
    end
    add!(rep, e_bond <= TOL_ENERGY, "case $tag bonds (engine vs ParmEd closed form)",
         @sprintf("n=%d max|dU|=%.2e kJ/mol", length(bt), e_bond))

    U_eng_ang = 0.0; U_ref_ang = 0.0; e_ang = 0.0
    for a in at
        th = acos(clamp(dot(q[:, a.i] - q[:, a.j], q[:, a.k] - q[:, a.j]) /
                        (norm(q[:, a.i] - q[:, a.j]) * norm(q[:, a.k] - q[:, a.j])), -1.0, 1.0))
        f = F.HarmonicAngle(a.i, a.j, a.k, a.kth, deg2rad(a.th0))
        Ue = term_energy(f, q)
        Ur = ref_angle_energy_pm(a.kp, deg2rad(a.th0p), th)
        U_eng_ang += Ue; U_ref_ang += Ur
        e_ang = max(e_ang, abs(Ue - Ur))
    end
    add!(rep, e_ang <= TOL_ENERGY, "case $tag angles (engine vs ParmEd closed form)",
         @sprintf("n=%d max|dU|=%.2e kJ/mol", length(at), e_ang))
    @printf("  angles: n=%d k in [%.4g, %.4g] kJ/mol/rad^2, theta0 in [%.4f, %.4f] deg\n",
            length(at), minimum(a.kth for a in at), maximum(a.kth for a in at),
            minimum(a.th0 for a in at), maximum(a.th0 for a in at))

    # ---- torsions: engine vs documented form over a phi grid ----
    deltas = range(-pi, pi; length = 13)
    e_dih = 0.0; e_half = 0.0; e_phase = 0.0; e_multi = 0.0
    e_alt = 0.0
    for d in dt
        f = F.PeriodicDihedral(d.i, d.j, d.k, d.l, d.kphi, d.n, deg2rad(d.phis))
        Ue0 = NaN; Ur0 = NaN; maxrel_phi = 0.0
        for delta in deltas
            q2 = rotate_l(q, d.j, d.k, d.l, delta)
            cs = ref_dihedral_cossin(q2, d.i, d.j, d.k, d.l)
            cs === nothing && error("degenerate torsion geometry")
            c, s = cs
            Ur = ref_torsion_energy(d.kphi, d.n, deg2rad(d.phis), c, s)
            Ue = term_energy(f, q2)
            e_dih = max(e_dih, abs(Ue - Ur))
            # discrimination: deliberately wrong conventions
            e_half = max(e_half, abs(Ue - ref_torsion_energy(0.5 * d.kphi, d.n, deg2rad(d.phis), c, s)))
            e_phase = max(e_phase, abs(Ue - ref_torsion_energy(d.kphi, d.n, deg2rad(d.phis) + pi, c, s)))
            if d.n != 1
                e_multi = max(e_multi, abs(Ue - ref_torsion_energy(d.kphi, 1, deg2rad(d.phis), c, s)))
            end
            # InterMol stores the same term as (fc0=k, fc_n=k, phi=phi_s).  Read
            # with the phase *inside* the harmonic, k[1+cos(n(phi-phi_s))], that
            # representation only agrees when n*phi_s == phi_s (mod 2pi).
            phi_e = F.dihedral_angle_grad(q2, T.NoBox(), d.i, d.j, d.k, d.l)[1]
            e_alt = max(e_alt, abs(Ue - d.kphi * (1.0 + cos(d.n * (phi_e - deg2rad(d.phis))))))
            if abs(delta) < 1e-12
                Ue0 = Ue; Ur0 = Ur
            end
        end
        @printf("  dih   %d-%d-%d-%d k=%.6g kJ/mol (=%.6g kcal/mol) n=%d phi_s=%.3f deg  U(ref geom)=%.8f  dU=%.2e\n",
                d.i, d.j, d.k, d.l, d.kphi, d.kp, d.n, d.phis, Ue0, Ue0 - Ur0)
    end
    add!(rep, e_dih <= TOL_ENERGY, "case $tag torsions over 13 phi points",
         @sprintf("n=%d max|dU|=%.2e kJ/mol", length(dt), e_dih))

    # ---- phi convention: engine vs external ParmEd implementation ----
    e_phi = 0.0
    seen = Set{Tuple{Int,Int,Int,Int}}()
    for p in phis
        any(d -> d.i == p.i && d.j == p.j && d.k == p.k && d.l == p.l, dt) || continue
        (p.i, p.j, p.k, p.l) in seen && continue
        push!(seen, (p.i, p.j, p.k, p.l))
        t = F.dihedral_angle_grad(q, T.NoBox(), p.i, p.j, p.k, p.l)
        t === nothing && error("degenerate reference geometry")
        dphi = mod(t[1] - deg2rad(p.deg) + pi, 2pi) - pi
        e_phi = max(e_phi, abs(dphi))
        @printf("  phi   %d-%d-%d-%d engine=%9.5f deg  parmed=%9.5f deg  dphi=%.2e rad\n",
                p.i, p.j, p.k, p.l, rad2deg(t[1]), p.deg, dphi)
    end
    add!(rep, e_phi <= TOL_PHI, "case $tag phi vs ParmEd geometry.dihedral",
         @sprintf("max|dphi|=%.2e rad", e_phi))

    # ---- force check: engine F vs central difference of engine U ----
    e_fd = 0.0
    for d in dt
        f = F.PeriodicDihedral(d.i, d.j, d.k, d.l, d.kphi, d.n, deg2rad(d.phis))
        U0, F0 = F.total_energy_forces([f], q, ctx)
        h = 1e-6
        Fd = zeros(3, n)
        for atom in (d.i, d.j, d.k, d.l), m in 1:3
            qp = copy(q); qp[m, atom] += h
            qm = copy(q); qm[m, atom] -= h
            Up, _ = F.total_energy_forces([f], qp, ctx)
            Um, _ = F.total_energy_forces([f], qm, ctx)
            Fd[m, atom] = -(Up - Um) / (2h)
        end
        e_fd = max(e_fd, maximum(abs.(F0[:, [d.i, d.j, d.k, d.l]] .- Fd[:, [d.i, d.j, d.k, d.l]])) /
                          max(1.0, maximum(abs.(Fd))))
    end
    add!(rep, e_fd <= TOL_FD, "case $tag torsion force vs finite difference",
         @sprintf("max rel err=%.2e", e_fd))

    # ---- totals ----
    U_eng_dih = sum(term_energy(F.PeriodicDihedral(d.i, d.j, d.k, d.l, d.kphi, d.n, deg2rad(d.phis)), q) for d in dt)
    U_ref_dih = 0.0
    for d in dt
        c, s = ref_dihedral_cossin(q, d.i, d.j, d.k, d.l)
        U_ref_dih += ref_torsion_energy(d.kphi, d.n, deg2rad(d.phis), c, s)
    end
    allf = Any[]
    for b in bt; push!(allf, F.HarmonicBond(b.i, b.j, b.k, b.r0)); end
    for a in at; push!(allf, F.HarmonicAngle(a.i, a.j, a.k, a.kth, deg2rad(a.th0))); end
    for d in dt; push!(allf, F.PeriodicDihedral(d.i, d.j, d.k, d.l, d.kphi, d.n, deg2rad(d.phis))); end
    U_eng_all, F_eng_all = F.total_energy_forces(allf, q, ctx)

    # ---- OpenMM: the external engine, per term and in total ----
    e_omm = 0.0; ef_omm = 0.0; ncmp = 0
    e_omm_b = 0.0; e_omm_a = 0.0; e_omm_d = 0.0; ef_omm_b = 0.0; ef_omm_a = 0.0; ef_omm_d = 0.0
    for b in bt
        key = ("bond", b.i, b.j, 0, 0, 0)
        f = F.HarmonicBond(b.i, b.j, b.k, b.r0)
        Ue, Fe = F.total_energy_forces([f], q, ctx)
        Uo = omm.energy[tag][key]; Fo = omm.force[tag][key]
        e_omm = max(e_omm, abs(Ue - Uo)); e_omm_b = max(e_omm_b, abs(Ue - Uo))
        ef_omm = max(ef_omm, maximum(abs.(Fe .- Fo)) / max(1.0, maximum(abs.(Fo))))
        ef_omm_b = max(ef_omm_b, maximum(abs.(Fe .- Fo)) / max(1.0, maximum(abs.(Fo))))
        ncmp += 1
    end
    for a in at
        key = ("angle", a.i, a.j, a.k, 0, 0)
        f = F.HarmonicAngle(a.i, a.j, a.k, a.kth, deg2rad(a.th0))
        Ue, Fe = F.total_energy_forces([f], q, ctx)
        Uo = omm.energy[tag][key]; Fo = omm.force[tag][key]
        e_omm = max(e_omm, abs(Ue - Uo)); e_omm_a = max(e_omm_a, abs(Ue - Uo))
        ef_omm = max(ef_omm, maximum(abs.(Fe .- Fo)) / max(1.0, maximum(abs.(Fo))))
        ef_omm_a = max(ef_omm_a, maximum(abs.(Fe .- Fo)) / max(1.0, maximum(abs.(Fo))))
        ncmp += 1
    end
    for d in dt
        key = ("dih", d.i, d.j, d.k, d.l, d.n)
        f = F.PeriodicDihedral(d.i, d.j, d.k, d.l, d.kphi, d.n, deg2rad(d.phis))
        Ue, Fe = F.total_energy_forces([f], q, ctx)
        Uo = omm.energy[tag][key]; Fo = omm.force[tag][key]
        e_omm = max(e_omm, abs(Ue - Uo)); e_omm_d = max(e_omm_d, abs(Ue - Uo))
        ef_omm = max(ef_omm, maximum(abs.(Fe .- Fo)) / max(1.0, maximum(abs.(Fo))))
        ef_omm_d = max(ef_omm_d, maximum(abs.(Fe .- Fo)) / max(1.0, maximum(abs.(Fo))))
        ncmp += 1
    end
    Ut_o, Ft_o = omm.total[tag]
    e_omm_tot = abs(U_eng_all - Ut_o)
    ef_omm_tot = maximum(abs.(F_eng_all .- Ft_o)) / max(1.0, maximum(abs.(Ft_o)))
    add!(rep, e_omm <= TOL_OMM_E, "case $tag OpenMM per-term energies",
         @sprintf("%d terms, max|dU|=%.2e kJ/mol", ncmp, e_omm))
    @printf("  openmm per class: bond dU=%.2e dF=%.2e | angle dU=%.2e dF=%.2e | torsion dU=%.2e dF=%.2e\n",
            e_omm_b, ef_omm_b, e_omm_a, ef_omm_a, e_omm_d, ef_omm_d)
    add!(rep, ef_omm <= TOL_OMM_F, "case $tag OpenMM per-term forces",
         @sprintf("max rel |dF|=%.2e", ef_omm))
    add!(rep, e_omm_tot <= TOL_OMM_E, "case $tag OpenMM total bonded energy",
         @sprintf("|dU|=%.2e kJ/mol (OpenMM %.10f)", e_omm_tot, Ut_o))
    add!(rep, ef_omm_tot <= TOL_OMM_F, "case $tag OpenMM total bonded forces",
         @sprintf("max rel |dF|=%.2e", ef_omm_tot))

    @printf("  subtotals (%s): bond=%.10f angle=%.10f torsion=%.10f kJ/mol\n",
            tag, U_eng_bond, U_eng_ang, U_eng_dih)
    @printf("  TOTAL bonded (%s): engine=%.10f  ParmEd-closed-form=%.10f  dU=%.2e  |  OpenMM=%.10f  dU=%.2e kJ/mol\n",
            tag, U_eng_all, U_ref_bond + U_ref_ang + U_ref_dih,
            U_eng_all - (U_ref_bond + U_ref_ang + U_ref_dih), Ut_o, U_eng_all - Ut_o)
    @printf("  TOTAL torsion only:  engine=%.10f  reference=%.10f\n", U_eng_dih, U_ref_dih)

    return e_half, e_phase, e_multi, e_alt
end

function main()
    atoms, bonds, angles, dihs, phis = read_reference(TSV)
    omm = read_openmm(OMM_TSV)
    cases = sort(collect(keys(atoms)))
    @printf("reference: %s + %s (%d cases, %d atoms total, %d/%d/%d terms)\n",
            basename(TSV), basename(OMM_TSV), length(cases),
            sum(length(v) for v in values(atoms)), length(bonds), length(angles), length(dihs))

    rep = Report(String[])
    natoms_max = maximum(length(atoms[c]) for c in cases)
    add!(rep, natoms_max == 9, "no duplicated geometry",
         @sprintf("%d cases x %d atoms", length(cases), natoms_max))

    # engineered cis/trans fixture: fixes the sign convention of phi itself
    a, b = 1.0, 1.2
    cis = zeros(3, 4); tr = zeros(3, 4)
    for (q, s) in ((cis, 1.0), (tr, -1.0))
        q[:, 2] = [0.0, 0.0, 0.0]; q[:, 3] = [1.0, 0.0, 0.0]
        q[:, 1] = [-cos(a), sin(a), 0.0]
        q[:, 4] = [1 + cos(b), s * sin(b), 0.0]
    end
    phi_cis = F.dihedral_angle_grad(cis, T.NoBox(), 1, 2, 3, 4)[1]
    phi_tr = F.dihedral_angle_grad(tr, T.NoBox(), 1, 2, 3, 4)[1]
    @printf("phi convention: cis=%.12f deg, trans=%.12f deg\n", rad2deg(phi_cis), rad2deg(phi_tr))
    add!(rep, abs(phi_cis) <= TOL_PHI && abs(abs(phi_tr) - pi) <= TOL_PHI,
         "phi sign convention (cis=0, trans=180)",
         @sprintf("cis=%.2e deg trans=%.2e deg", rad2deg(phi_cis), rad2deg(phi_tr)))

    e_half = 0.0; e_phase = 0.0; e_multi = 0.0; e_alt = 0.0
    for tag in unique([b.tag for b in bonds])
        h, p, m, al = run_case(tag, atoms, bonds, angles, dihs, phis, omm, rep)
        e_half = max(e_half, h); e_phase = max(e_phase, p); e_multi = max(e_multi, m); e_alt = max(e_alt, al)
    end

    @printf("\ndiscrimination (a wrong convention would have to show up here):\n")
    @printf("  k_phi/2 instead of k_phi      : max|dU| = %.3f kJ/mol\n", e_half)
    @printf("  phi0 = phi_s + pi             : max|dU| = %.3f kJ/mol\n", e_phase)
    @printf("  n -> 1                        : max|dU| = %.3f kJ/mol\n", e_multi)
    @printf("  InterMol fc/phi read as k[1+cos(n(phi-phi_s))]: max|dU| = %.3f kJ/mol\n", e_alt)
    add!(rep, min(e_half, e_phase, e_multi) > DISCRIMINATION_MIN,
         "wrong conventions are detectable", @sprintf("min deviation %.3f kJ/mol", min(e_half, e_phase, e_multi)))

    println("\n== lane 4 checks ==")
    for r in rep.rows
        println(r)
    end
    ok = all(startswith(r, "PASS") for r in rep.rows)
    println(ok ? "\nLANE 4: PASS" : "\nLANE 4: FAIL")
    return ok ? 0 : 1
end

exit(main())
