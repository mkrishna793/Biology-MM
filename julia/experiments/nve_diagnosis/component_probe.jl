# Reproducible component-level constrained NVE diagnostic for TIP3P WaterBox.
# Run from julia with the repository's configured Julia environment.
# Output is streamed to stdout; redirect it externally to preserve the full log.

using FoundationCore
using Printf
using Random

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
isdefined(FoundationCore, :WaterBox) ||
    Base.include(FoundationCore, joinpath(ROOT, "src", "WaterBox.jl"))
const WB = FoundationCore.WaterBox
const F = FoundationCore.Factors
const D = FoundationCore.Dynamics
const C = FoundationCore.Constraints

function zero_force_water(w; lj=false, pme=false)
    WB.TIP3PWater(w.positions, w.masses,
        pme ? w.charges : zero(w.charges), lj ? w.epsilon : zero(w.epsilon),
        w.sigma, w.molecules, w.constraints, w.lj_topology,
        w.pme_corrections, w.box, w.density, w.seed)
end

function components(field, q)
    flj = zeros(size(q)); fpme = similar(flj)
    ulj = F.evaluate!(flj, field.lj, q, field.ctx)
    upme = F.evaluate!(fpme, field.pme, q, field.ctx)
    return ulj, upme, flj .+ fpme
end

com_momentum_max(p) = maximum(abs(sum(@view p[d, :])) for d in 1:3)

function initialize(nmol, seed; relax_steps=100)
    w = WB.tip3p_water(nmol; density=1.0, seed=seed)
    field = WB.water_field(w)
    tr = WB.WaterTrajectory(w, field; temperature=300.0, seed=seed)
    rel = WB.relax!(tr; max_steps=relax_steps)
    # Relaxation ends at rest; initialize a common tangent Maxwell state.
    rng = MersenneTwister(seed + 9000)
    p = tr.st.p
    for j in axes(p, 2), d in 1:3
        p[d,j] = sqrt(w.masses[j] * WB.KB * 300.0) * randn(rng)
    end
    C.project_velocities!(tr.st.q, p, w.masses, w.constraints;
                          tolerance=WB.CONSTRAINT_TOL)
    for d in 1:3
        vcm = sum(p[d,j] for j in axes(p,2)) / sum(w.masses)
        for j in axes(p,2); p[d,j] -= w.masses[j] * vcm; end
    end
    C.project_velocities!(tr.st.q, p, w.masses, w.constraints;
                          tolerance=WB.CONSTRAINT_TOL)
    return tr, rel
end

function run_nve(tr, dt, nsteps, label)
    w, field = tr.water, tr.field
    q, p = copy(tr.st.q), copy(tr.st.p)
    f = zeros(size(q)); ftan = similar(f)
    u = WB._tangent_forces!(f, ftan, q, w.masses, w.constraints, field)
    ulj, upme, _ = components(field, q)
    k = D.kinetic_energy(field.system, p)
    e0 = ulj + upme + k
    band = 0.0
    maxforce = maximum(abs, f)
    maxres = maximum(abs, C.constraint_residuals(w.constraints, q))
    projwork = 0.0
    maxcom = com_momentum_max(p)
    for step in 1:nsteps
        p .+= 0.5dt .* f
        WB._drift!(q, p, w.masses, w.molecules, dt)
        u = WB._tangent_forces!(f, ftan, q, w.masses, w.constraints, field)
        p .+= 0.5dt .* f
        kpre = D.kinetic_energy(field.system, p)
        C.project_velocities!(q, p, w.masses, w.constraints;
                              tolerance=WB.CONSTRAINT_TOL)
        k = D.kinetic_energy(field.system, p)
        projwork += k - kpre
        maxcom = max(maxcom, com_momentum_max(p))
        ulj, upme, _ = components(field, q)
        band = max(band, abs(k + ulj + upme - e0))
        maxforce = max(maxforce, maximum(abs, f))
        maxres = max(maxres, maximum(abs, C.constraint_residuals(w.constraints, q)))
        @printf("RECORD %s step=%d t_ps=%.6g K=%.9g U_LJ=%.9g U_PME=%.9g E=%.9g dE=%.9g maxF=%.9g residual=%.3g com_momentum_max=%.9g proj_work=%.9g\n",
            label, step, step*dt, k, ulj, upme, k+ulj+upme,
            k+ulj+upme-e0, maximum(abs,f),
            maximum(abs,C.constraint_residuals(w.constraints,q)),
            com_momentum_max(p), projwork)
    end
    @printf("SUMMARY %s band=%.9g final_dE=%.9g projection_work=%.9g max_force=%.9g max_residual=%.3g max_com_momentum=%.9g\n",
        label, band, k+ulj+upme-e0, projwork,
        maxforce, maxres, maxcom)
end

function main()
    for (nmol, steps, seed) in ((27, 100, 27), (64, 100, 64),
                                (216, 100, 216), (1728, 100, 1728))
        tr, rel = initialize(nmol, seed)
        df = tr.field
        @printf("SETTINGS n=%d density=1 alpha=%.6g pme_rcut=%.6g lj_rcut=%.6g grid=%s; current defaults\n",
            nmol, df.alpha, df.pme_rcut, df.rcut, string(df.grid))
        @printf("START n=%d relax_steps=%d relax_converged=%s U0=%.9g U1=%.9g fmax=%.9g\n",
                nmol, rel.steps, rel.converged, rel.u_initial, rel.u_final,
                rel.max_force_final)
        # All comparisons start from exactly this relaxed and thermalized state.
        for (dt, n) in ((0.001, steps), (0.0005, 2steps))
            run_nve(tr, dt, n, "full_n$(nmol)_dt$(dt)")
        end
        if nmol in (27, 64)
            for (name, lj, pme) in (("constraint",false,false),
                                     ("lj",true,false), ("pme",false,true))
                wc = zero_force_water(tr.water; lj=lj, pme=pme)
                fc = WB.water_field(wc)
                tc = WB.WaterTrajectory(wc, fc; temperature=0.0, seed=seed)
                tc.st.q .= tr.st.q; tc.st.p .= tr.st.p
                tc.st.force_valid = false
                run_nve(tc, 0.001, name == "pme" ? 20 : steps,
                        "$(name)_n$(nmol)")
            end
        end
    end
end

main()
