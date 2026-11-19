using FoundationCore
using JSON

const FC = FoundationCore
const WF = FC.Workflow

struct CLIInputError <: Exception
    code::String
    message::String
end
Base.showerror(io::IO, e::CLIInputError) = print(io, e.code, ": ", e.message)

const _VALID_CLI_BACKENDS = ("auto", "cpu", "gpu", "metal", "vulkan", "sycl")

function usage(io=stdout)
    println(io, "Usage: julia --project=. bin/md.jl [--backend <auto|cpu|gpu|metal|vulkan|sycl>] [--info] SPEC.json")
    println(io, "       julia --project=. bin/md.jl --info [--backend <auto|cpu|gpu|metal|vulkan|sycl>]")
    println(io, "Runs validate -> prepare -> minimize -> equilibrate -> produce -> analyze.")
    println(io, "Options:")
    println(io, "  --backend <mode>  Hardware backend: auto (default), cpu, gpu, metal, vulkan, sycl")
    println(io, "  --info, --status  Print TriumGPU hardware telemetry")
    println(io, "Spec: schema_version=1, coordinates (N×3), masses, orthorhombic box, supported factors.")
    println(io, "Supported ensembles: nve, nvt; opt-in respa_nve. Imported normalized_amber requires explicit nm,kJ/mol,amu,ps units and protocol in (direct-nocutoff-flexible, pme-lfs-flexible, pme-lfs-rigid).")
end

function _print_gpu_telemetry(io::IO, requested_backend::Symbol)
    prof = FC.TriumGPU.probe_gpu_device(; backend=requested_backend)
    avail = join(string.(FC.TriumGPU.available_gpu_backends()), ", ")
    println(io, "TriumGPU Telemetry:")
    println(io, "  requested_backend        : ", requested_backend)
    println(io, "  active_backend           : ", prof.backend)
    println(io, "  device_name              : ", prof.device_name)
    println(io, "  has_gpu                  : ", prof.has_gpu)
    println(io, "  unified_memory           : ", prof.unified_memory)
    println(io, "  simd_width               : ", prof.simd_width)
    println(io, "  max_threads_per_tg       : ", prof.max_threads_per_tg)
    println(io, "  auto_workgroup_size      : ", prof.auto_workgroup_size)
    println(io, "  vram_bytes               : ", prof.vram_bytes)
    println(io, "  auto_gpu_crossover_pairs : ", prof.auto_gpu_crossover_pairs)
    println(io, "  available_backends       : ", avail)
end

function _required(obj, key, context)
    haskey(obj, key) || throw(CLIInputError("MissingParameter", "$context.$key is required"))
    obj[key]
end

function _number(obj, key, context)
    value = _required(obj, key, context)
    value isa Real && isfinite(value) || throw(CLIInputError("InvalidParameter", "$context.$key must be a finite number"))
    Float64(value)
end

function _analysis_config(doc)
    raw = get(doc, "analysis", Dict{String,Any}())
    allowed = Set(("min_frames", "fit_window", "lag_window", "min_fit_points", "min_time_span", "unwrap", "min_ess"))
    unknown = setdiff(Set(String.(keys(raw))), allowed)
    isempty(unknown) || throw(CLIInputError("InvalidParameter", "unsupported analysis fields: $(join(sort!(collect(unknown)), ", "))"))
    fit = get(raw, "fit_window", nothing)
    lag = get(raw, "lag_window", nothing)
    (fit === nothing || (fit isa AbstractVector && length(fit) == 2)) || throw(CLIInputError("InvalidParameter", "analysis.fit_window must contain two fractions"))
    (lag === nothing || (lag isa AbstractVector && length(lag) == 2)) || throw(CLIInputError("InvalidParameter", "analysis.lag_window must contain two times"))
    opts = Dict{Symbol,Any}(:min_frames=>Int(get(raw,"min_frames",20)),
        :min_fit_points=>Int(get(raw,"min_fit_points",5)), :min_time_span=>Float64(get(raw,"min_time_span",0.0)),
        :unwrap=>Bool(get(raw,"unwrap",true)), :min_ess=>Float64(get(raw,"min_ess",5.0)))
    fit === nothing || (opts[:fit_window] = (Float64(fit[1]), Float64(fit[2])))
    lag === nothing || (opts[:lag_window] = (Float64(lag[1]), Float64(lag[2])))
    WF.AnalysisConfig(; min_frames=Int(get(raw,"min_frames",20)), opts...)
end

function _parse_cli_units(doc)
    raw = get(doc, "units", "reduced")
    raw isa AbstractString || throw(CLIInputError("InvalidUnits", "units must be a string"))
    s = String(raw)
    if s == "reduced"
        return (FC.Types.REDUCED, :reduced)
    elseif s == "nm,kJ/mol,amu,ps" || s == "md"
        return (FC.CustomPhysics.MD_UNITS, :md)
    elseif s == "A,eV,amu,fs" || s == "ev_angstrom"
        return (FC.CustomPhysics.EV_ANGSTROM_UNITS, :ev_angstrom)
    else
        throw(CLIInputError("InvalidUnits", "unsupported units '$s'"))
    end
end

function _resolve_reactive_units(f, sys_units_sym::Symbol)::Symbol
    if !haskey(f, "units") || f["units"] === nothing
        return sys_units_sym === :ev_angstrom ? :ev : :md
    end
    raw = f["units"]
    raw isa AbstractString || throw(CLIInputError("InvalidUnits", "factor units must be a string"))
    s = String(raw)
    if s == "md" || s == "nm,kJ/mol,amu,ps"
        return :md
    elseif s == "ev" || s == "ev_angstrom" || s == "A,eV,amu,fs"
        return :ev
    else
        throw(CLIInputError("InvalidUnits", "unsupported reactive factor units '$s'"))
    end
end

function run_spec(path; backend::Symbol=:auto)
    Symbol(backend) in Symbol.(_VALID_CLI_BACKENDS) ||
        throw(CLIInputError("InvalidBackend", "unsupported backend '$backend'; expected one of $(join(_VALID_CLI_BACKENDS, ", "))"))
    spec_path = abspath(path)
    doc = JSON.parsefile(spec_path)
    doc isa AbstractDict || throw(CLIInputError("InvalidSpec", "root must be a JSON object"))
    get(doc, "schema_version", nothing) == 1 || throw(CLIInputError("InvalidSpec", "schema_version must be 1"))
    haskey(doc, "pdb") && throw(CLIInputError("UnsupportedChemistry", "parameter-file chemistry is not accepted; supply explicit supported factors"))
    haskey(doc, "constraints") && throw(CLIInputError("UnsupportedConstraint", "CLI constraint specifications are not supported"))
    imported = haskey(doc, "normalized_amber")
    rs = nothing
    constraints = nothing
    if imported
        get(doc, "units", "") == "nm,kJ/mol,amu,ps" || throw(CLIInputError("InvalidUnits", "normalized AMBER execution requires units=nm,kJ/mol,amu,ps"))
        protocol = _required(doc, "protocol", "spec")
        protocol in ("direct-nocutoff-flexible", "pme-lfs-flexible", "pme-lfs-rigid") ||
            throw(CLIInputError("UnsupportedProtocol", "supported protocols: direct-nocutoff-flexible, pme-lfs-flexible, pme-lfs-rigid"))
        source = String(doc["normalized_amber"])
        isabspath(source) || (source = joinpath(dirname(spec_path), source))
        ns = FC.Chemistry.read_normalized(source)
        if protocol == "direct-nocutoff-flexible"
            rs = FC.Chemistry.build_resolved(ns; protocol=FC.Chemistry.default_protocol(ns; nonbonded=:direct, lj_mode=:lfs, water_model=:flexible))
            rs.box isa FC.Types.NoBox || throw(CLIInputError("UnsupportedProtocol", "direct imported execution requires NoBox"))
            rs.constraints === nothing || throw(CLIInputError("UnsupportedConstraint", "direct-nocutoff-flexible refuses imported constraints"))
        elseif protocol == "pme-lfs-flexible"
            rs = FC.Chemistry.build_resolved(ns; protocol=FC.Chemistry.default_protocol(ns; nonbonded=:pme, lj_mode=:lfs, water_model=:flexible))
            rs.box isa FC.Types.NoBox && throw(CLIInputError("UnsupportedProtocol", "pme-lfs-flexible requires periodic box"))
            rs.constraints === nothing || throw(CLIInputError("UnsupportedConstraint", "pme-lfs-flexible refuses imported constraints"))
        elseif protocol == "pme-lfs-rigid"
            rs = FC.Chemistry.build_resolved(ns; protocol=FC.Chemistry.default_protocol(ns; nonbonded=:pme, lj_mode=:lfs, water_model=:rigid))
            rs.box isa FC.Types.NoBox && throw(CLIInputError("UnsupportedProtocol", "pme-lfs-rigid requires periodic box"))
            constraints = rs.constraints
        end
        isempty(rs.atoms.virtual_sites) || throw(CLIInputError("UnsupportedVirtualSite", "virtual sites are not executable by this CLI"))
    end
    rawcoords = _required(doc, "coordinates", "spec")
    rawcoords isa AbstractVector && !isempty(rawcoords) || throw(CLIInputError("InvalidCoordinates", "coordinates must be a non-empty N×3 array"))
    all(row -> row isa AbstractVector && length(row) == 3 && all(x -> x isa Real && isfinite(x), row), rawcoords) ||
        throw(CLIInputError("InvalidCoordinates", "each coordinate row must contain three finite numbers"))
    q = hcat((Float64.(row) for row in rawcoords)...)
    if imported
        size(q,2) == length(rs.atoms.masses) || throw(CLIInputError("InvalidCoordinates", "coordinates must match imported atom count"))
        masses_raw = rs.atoms.masses
        box = rs.box
        factors = FC.ChemistryFactors.with_lfs_c64(rs.factors)
        sys_units = rs.units
    else
        sys_units, sys_units_sym = _parse_cli_units(doc)
        masses_raw = _required(doc, "masses", "spec")
        masses_raw isa AbstractVector && length(masses_raw) == size(q,2) || throw(CLIInputError("InvalidParameter", "masses count must match coordinates"))
        all(x -> x isa Real && isfinite(x) && x > 0, masses_raw) || throw(CLIInputError("InvalidParameter", "masses must be positive finite numbers"))
        boxdata = _required(doc, "box", "spec")
        boxdata isa AbstractVector && length(boxdata) == 3 && all(x -> x isa Real && isfinite(x) && x > 0, boxdata) ||
            throw(CLIInputError("InvalidBox", "box must contain three positive finite orthorhombic lengths"))
        box = try FC.Types.OrthorhombicBox(Float64.(boxdata)) catch e; throw(CLIInputError("InvalidBox", sprint(showerror,e))) end
        rawfactors = _required(doc, "factors", "spec")
        rawfactors isa AbstractVector || throw(CLIInputError("InvalidParameter", "factors must be an array"))
        n_atoms = size(q, 2)
        factors = FC.Factors.AbstractFactor[]
        for (idx, f) in enumerate(rawfactors)
            f isa AbstractDict || throw(CLIInputError("InvalidParameter", "factors[$idx] must be an object"))
            kind = get(f, "type", "")
            ctx = "factors[$idx]"
            try
                if kind == "harmonic_bond"
                    push!(factors, FC.Factors.HarmonicBond(Int(_required(f,"i",ctx)), Int(_required(f,"j",ctx)), _number(f,"k",ctx), _number(f,"r0",ctx)))
                elseif kind == "lj_pair"
                    mode = Symbol(get(f, "mode", "lfs"))
                    mode in (:none, :trunc, :shift, :lfs) || throw(CLIInputError("InvalidParameter", "unsupported LJ mode '$mode'"))
                    push!(factors, FC.Factors.LennardJonesPair(Int(_required(f,"i",ctx)), Int(_required(f,"j",ctx)), _number(f,"epsilon",ctx), _number(f,"sigma",ctx); mode=mode, rc=_number(f,"rc",ctx)))
                elseif kind == "morse_bond"
                    push!(factors, FC.ReactiveNano.MorseBondFactor(Int(_required(f,"i",ctx)), Int(_required(f,"j",ctx)), _number(f,"De",ctx), _number(f,"a",ctx), _number(f,"r0",ctx)))
                elseif kind == "tersoff"
                    el = String(get(f, "element", "C"))
                    u_sym = _resolve_reactive_units(f, sys_units_sym)
                    tparams = if el == "C"
                        FC.ReactiveNano.tersoff_carbon(; units=u_sym)
                    elseif el == "Si"
                        FC.ReactiveNano.tersoff_silicon(; units=u_sym)
                    else
                        throw(CLIInputError("UnsupportedChemistry", "unsupported Tersoff element '$el'"))
                    end
                    push!(factors, FC.ReactiveNano.TersoffFactor(n_atoms, tparams))
                elseif kind == "stillinger_weber"
                    el = String(get(f, "element", "Si"))
                    el == "Si" || throw(CLIInputError("UnsupportedChemistry", "unsupported Stillinger-Weber element '$el'"))
                    u_sym = _resolve_reactive_units(f, sys_units_sym)
                    swparams = FC.ReactiveNano.stillinger_weber_silicon(; units=u_sym)
                    push!(factors, FC.ReactiveNano.StillingerWeberFactor(n_atoms, swparams))
                elseif kind == "custom_pair"
                    kernel = Symbol(_required(f, "kernel", ctx))
                    raw_pairs = _required(f, "pairs", ctx)
                    raw_pairs isa AbstractVector || throw(CLIInputError("InvalidParameter", "$ctx.pairs must be an array"))
                    pairs_vec = Tuple{Int,Int}[(Int(p[1]), Int(p[2])) for p in raw_pairs]
                    raw_params = get(f, "params", Dict{String,Any}())
                    raw_params isa AbstractDict || throw(CLIInputError("InvalidParameter", "$ctx.params must be an object"))
                    pdict = Dict{Symbol,Float64}(Symbol(k) => Float64(v) for (k, v) in pairs(raw_params))
                    rc = (haskey(f, "rc") && f["rc"] !== nothing) ? _number(f, "rc", ctx) : Inf
                    mode = Symbol(get(f, "mode", isfinite(rc) ? "lfs" : "none"))
                    push!(factors, FC.CustomPhysics.CustomPairFactor(kernel, pdict, pairs_vec; rc=rc, mode=mode, n_atoms=n_atoms))
                elseif kind == "tabulated_spline"
                    raw_pairs = _required(f, "pairs", ctx)
                    raw_pairs isa AbstractVector || throw(CLIInputError("InvalidParameter", "$ctx.pairs must be an array"))
                    pairs_vec = Tuple{Int,Int}[(Int(p[1]), Int(p[2])) for p in raw_pairs]
                    r_grid = Float64.(_required(f, "r_grid", ctx))
                    u_grid = Float64.(_required(f, "u_grid", ctx))
                    rc = (haskey(f, "rc") && f["rc"] !== nothing) ? _number(f, "rc", ctx) : r_grid[end]
                    mode = Symbol(get(f, "mode", "lfs"))
                    tab = FC.CustomPhysics.CubicSplineTable(r_grid, u_grid)
                    push!(factors, FC.CustomPhysics.CustomPairFactor(:tabulated_spline, Dict{Symbol,Float64}(), pairs_vec; rc=rc, mode=mode, n_atoms=n_atoms, spline=tab))
                elseif kind == "external_efield"
                    E_raw = Float64.(_required(f, "E", ctx))
                    ch_raw = Float64.(_required(f, "charges", ctx))
                    atom_idxs = (haskey(f, "atom_indices") && f["atom_indices"] !== nothing) ? Int.(f["atom_indices"]) : collect(1:n_atoms)
                    push!(factors, FC.NEMD.ExternalElectricFieldFactor(E_raw, ch_raw, atom_idxs, n_atoms))
                elseif kind == "wall_restraint"
                    wkind_raw = Symbol(_required(f, "kind", ctx))
                    wkind = wkind_raw === :harmonic ? :harmonic_flat_bottom : wkind_raw
                    axis = Int(_required(f, "axis", ctx))
                    lo = (haskey(f, "lo") && f["lo"] !== nothing) ? _number(f, "lo", ctx) : -Inf
                    hi = (haskey(f, "hi") && f["hi"] !== nothing) ? _number(f, "hi", ctx) : Inf
                    atom_idxs = (haskey(f, "atom_indices") && f["atom_indices"] !== nothing) ? Int.(f["atom_indices"]) : collect(1:n_atoms)
                    wk_kwargs = Dict{Symbol,Any}(:axis => axis, :lo => lo, :hi => hi)
                    haskey(f, "k") && f["k"] !== nothing && (wk_kwargs[:k] = _number(f, "k", ctx))
                    haskey(f, "epsilon") && f["epsilon"] !== nothing && (wk_kwargs[:epsilon] = _number(f, "epsilon", ctx))
                    haskey(f, "sigma") && f["sigma"] !== nothing && (wk_kwargs[:sigma] = _number(f, "sigma", ctx))
                    haskey(f, "rc") && f["rc"] !== nothing && (wk_kwargs[:rc] = _number(f, "rc", ctx))
                    haskey(f, "r_max") && f["r_max"] !== nothing && (wk_kwargs[:r_max] = _number(f, "r_max", ctx))
                    if haskey(f, "center") && f["center"] !== nothing
                        c = f["center"]
                        wk_kwargs[:center] = (Float64(c[1]), Float64(c[2]))
                    end
                    push!(factors, FC.NEMD.WallRestraintFactor(wkind, atom_idxs, n_atoms; wk_kwargs...))
                elseif kind == "umbrella"
                    cv = FC.EnhancedSampling.DistanceCV(Int(_required(f, "i", ctx)), Int(_required(f, "j", ctx)))
                    push!(factors, FC.EnhancedSampling.UmbrellaFactor(cv, _number(f, "k", ctx), _number(f, "r0", ctx), n_atoms))
                else
                    throw(CLIInputError("UnsupportedChemistry", "unsupported factor type '$kind'"))
                end
            catch e
                e isa KeyError && throw(CLIInputError("MissingParameter", "$ctx missing field $(e.key)"))
                (e isa ArgumentError || e isa DomainError || e isa BoundsError || e isa DimensionMismatch) &&
                    throw(CLIInputError("InvalidParameter", sprint(showerror, e)))
                rethrow()
            end
        end
    end
    isempty(factors) && throw(CLIInputError("MissingParameter", "at least one factor is required"))
    ensemble = get(doc, "ensemble", "nvt")
    ensemble in ("nve", "nvt") || throw(CLIInputError("UnsupportedEnsemble", "'$ensemble'; supported ensembles: nve, nvt"))
    cfgraw = get(doc, "workflow", Dict{String,Any}())
    cfgraw isa AbstractDict || throw(CLIInputError("InvalidParameter", "workflow must be an object"))
    allowed_cfg = Set(("run_id", "dt", "seed", "temperature", "gamma", "minimize_max_steps", "equilibrate_steps", "produce_steps", "frame_every", "checkpoint_every", "contact_tol", "fast_factors", "respa_m"))
    unknown_cfg = setdiff(Set(String.(keys(cfgraw))), allowed_cfg)
    isempty(unknown_cfg) || throw(CLIInputError("InvalidParameter", "unsupported workflow fields: $(join(sort!(collect(unknown_cfg)), ", "))"))
    cfgkwargs = Dict{Symbol,Any}(Symbol(k)=>v for (k,v) in cfgraw)
    integrator = get(doc,"integrator", ensemble == "nve" ? "velocity_verlet" : "langevin_baoab")
    integrator in ("velocity_verlet", "langevin_baoab", "respa_nve") || throw(CLIInputError("UnsupportedIntegrator", integrator))
    (ensemble == "nve" ? integrator in ("velocity_verlet", "respa_nve") : integrator == "langevin_baoab") || throw(CLIInputError("UnsupportedIntegrator", "integrator must match ensemble"))
    cfgkwargs[:algorithm] = Symbol(integrator)
    if integrator == "respa_nve"
        cfgkwargs[:temperature] = 0.0
        cfgkwargs[:gamma] = nothing
    end
    if imported
        cfgkwargs[:resolved_parameters] = Dict("normalized_amber"=>JSON.parsefile(source), "protocol"=>protocol, "units"=>doc["units"])
    end
    cfgkwargs[:analysis] = _analysis_config(doc)
    cfg = try WF.WorkflowConfig(; cfgkwargs...) catch e; throw(CLIInputError("InvalidParameter", sprint(showerror,e))) end
    at = FC.Types.AtomSet(q, zeros(size(q)), Float64.(masses_raw), zeros(length(masses_raw)))
    sys = FC.Types.System(at, box; units=sys_units, temperature=cfg.temperature)
    workdir = String(get(doc, "workdir", "md-run"))
    isabspath(workdir) || (workdir = joinpath(dirname(spec_path), workdir))
    if haskey(doc, "resume_from")
        rspec = doc["resume_from"]
        rspec isa AbstractDict || throw(CLIInputError("InvalidParameter", "resume_from must be an object"))
        ckpt_path = String(_required(rspec, "checkpoint", "resume_from"))
        manifest_path = String(_required(rspec, "manifest", "resume_from"))
        isabspath(ckpt_path) || (ckpt_path = joinpath(dirname(spec_path), ckpt_path))
        isabspath(manifest_path) || (manifest_path = joinpath(dirname(spec_path), manifest_path))
        mkpath(workdir)
        ckpt_dir = joinpath(workdir, "checkpoints")
        mkpath(ckpt_dir)
        traj_path = joinpath(workdir, "trajectory.xyz")
        prod_res = WF.resume_workflow(ckpt_path, manifest_path, factors, Float64.(masses_raw),
                                      sys_units, cfg;
                                      trajectory_path=traj_path, constraints=constraints,
                                      checkpoint_dir=ckpt_dir)
        ana_res = WF.analyze_stage(prod_res.trajectory_path, prod_res.energies, cfg)
        println(JSON.json(Dict("run_id"=>cfg.run_id, "trajectory"=>prod_res.trajectory_path,
            "checkpoint"=>prod_res.checkpoint_path, "frames"=>ana_res.n_frames,
            "diffusion_coefficient"=>ana_res.diffusion_coefficient,
            "stages"=>["resume", "produce", "analyze"])))
        return
    end
    result = WF.run_workflow(sys, factors, cfg; workdir=workdir, constraints=constraints)
    println(JSON.json(Dict("run_id"=>result.manifest.run_id, "trajectory"=>result.trajectory_path,
        "checkpoint"=>result.final_checkpoint_path, "frames"=>result.analysis.n_frames,
        "diffusion_coefficient"=>result.analysis.diffusion_coefficient,
        "stages"=>String.(getfield.(result.stages, :stage)))))
end

function _parse_cli_args(args::Vector{String})
    backend = :auto
    backend_requested = false
    show_info = false
    specs = String[]
    i = 1
    while i <= length(args)
        a = args[i]
        if a in ("-h", "--help")
            return (:help, backend, backend_requested, show_info, specs)
        elseif a in ("--info", "--status")
            show_info = true
        elseif a == "--backend"
            i < length(args) || throw(CLIInputError("InvalidBackend", "--backend requires a value in $(join(_VALID_CLI_BACKENDS, ", "))"))
            i += 1
            val = args[i]
            val in _VALID_CLI_BACKENDS || throw(CLIInputError("InvalidBackend", "unsupported backend '$val'; expected one of $(join(_VALID_CLI_BACKENDS, ", "))"))
            backend = Symbol(val)
            backend_requested = true
        elseif startswith(a, "--backend=")
            val = split(a, "="; limit=2)[2]
            val in _VALID_CLI_BACKENDS || throw(CLIInputError("InvalidBackend", "unsupported backend '$val'; expected one of $(join(_VALID_CLI_BACKENDS, ", "))"))
            backend = Symbol(val)
            backend_requested = true
        elseif startswith(a, "-")
            return (:badflag, backend, backend_requested, show_info, specs)
        else
            push!(specs, a)
        end
        i += 1
    end
    return (:ok, backend, backend_requested, show_info, specs)
end

if abspath(PROGRAM_FILE) == @__FILE__
    mode, backend, backend_requested, show_info, specs = try
        _parse_cli_args(String.(ARGS))
    catch err
        showerror(stderr, err)
        println(stderr)
        exit(2)
    end
    if mode === :help
        usage(stdout)
        exit(0)
    elseif mode === :badflag
        usage(stderr)
        exit(2)
    elseif show_info && isempty(specs)
        _print_gpu_telemetry(stdout, backend)
        exit(0)
    elseif length(specs) != 1
        usage(stderr)
        exit(2)
    end
    if backend_requested || show_info
        _print_gpu_telemetry(stderr, backend)
    end
    try
        run_spec(specs[1]; backend=backend)
    catch err
        showerror(stderr, err)
        println(stderr)
        exit(1)
    end
end
