module TrajectoryFormat

export write_extxyz_frame, read_extxyz_frames

_finite(x) = isfinite(x)
_unit_token(x) = occursin(r"^[A-Za-z][A-Za-z0-9_.+-]*$", String(x))

function write_extxyz_frame(io::IO; positions, species, time, box_matrix=nothing,
                            length_unit, time_unit)
    ndims(positions) == 2 && size(positions, 1) == 3 || throw(ArgumentError("positions must have shape 3×N"))
    size(positions, 2) > 0 || throw(ArgumentError("positions must contain at least one atom"))
    length(species) == size(positions, 2) || throw(ArgumentError("species count must match positions"))
    all(_finite, positions) || throw(ArgumentError("positions must be finite"))
    _finite(time) || throw(ArgumentError("time must be finite"))
    _unit_token(length_unit) || throw(ArgumentError("length_unit must be a single safe unit token"))
    _unit_token(time_unit) || throw(ArgumentError("time_unit must be a single safe unit token"))
    all(s -> occursin(r"^[A-Za-z][A-Za-z0-9_+-]*$", String(s)), species) ||
        throw(ArgumentError("species must be element-like labels"))
    if box_matrix !== nothing
        size(box_matrix) == (3, 3) || throw(ArgumentError("box_matrix must be 3×3"))
        all(_finite, box_matrix) || throw(ArgumentError("box_matrix must be finite"))
        d = box_matrix[1, 1] * (box_matrix[2, 2] * box_matrix[3, 3] - box_matrix[2, 3] * box_matrix[3, 2]) -
            box_matrix[1, 2] * (box_matrix[2, 1] * box_matrix[3, 3] - box_matrix[2, 3] * box_matrix[3, 1]) +
            box_matrix[1, 3] * (box_matrix[2, 1] * box_matrix[3, 2] - box_matrix[2, 2] * box_matrix[3, 1])
        isfinite(d) && d != 0 || throw(ArgumentError("box_matrix must be nonsingular"))
    end
    println(io, size(positions, 2))
    if box_matrix === nothing
        println(io, "Time=$(time) time_unit=\"$(time_unit)\" length_unit=\"$(length_unit)\" pbc=\"F F F\" Properties=species:S:1:pos:R:3")
    else
        lattice = join(string.(vec(box_matrix)), " ")
        println(io, "Lattice=\"$(lattice)\" Properties=species:S:1:pos:R:3 Time=$(time) time_unit=\"$(time_unit)\" length_unit=\"$(length_unit)\" pbc=\"T T T\"")
    end
    for i in axes(positions, 2)
        println(io, String(species[i]), " ", positions[1, i], " ", positions[2, i], " ", positions[3, i])
    end
    return nothing
end

function read_extxyz_frames(io::IO)
    frames = NamedTuple[]
    while !eof(io)
        nline = readline(io)
        isempty(strip(nline)) && continue
        n = tryparse(Int, strip(nline))
        (n === nothing || n <= 0) && throw(ArgumentError("atom count must be a positive integer"))
        comment = readline(io)
        props = match(r"Properties=species:S:1:pos:R:3(?:\s|$)", comment)
        props === nothing && throw(ArgumentError("unsupported or missing Properties metadata"))
        time_m = match(r"(?:^|\s)Time=([^\s]+)", comment)
        lu_m = match(r"(?:^|\s)length_unit=\"([A-Za-z][A-Za-z0-9_.+-]*)\"", comment)
        tu_m = match(r"(?:^|\s)time_unit=\"([A-Za-z][A-Za-z0-9_.+-]*)\"", comment)
        (time_m === nothing || lu_m === nothing || tu_m === nothing) && throw(ArgumentError("missing or invalid time/unit metadata"))
        time = tryparse(Float64, time_m.captures[1])
        (time !== nothing && isfinite(time)) || throw(ArgumentError("time must be finite"))
        pbc = match(r"(?:^|\s)pbc=\"([^\"]+)\"", comment)
        lat = match(r"(?:^|\s)Lattice=\"([^\"]+)\"", comment)
        box = nothing
        if lat !== nothing
            vals = parse.(Float64, split(lat.captures[1]))
            length(vals) == 9 || throw(ArgumentError("Lattice must contain 9 values"))
            all(isfinite, vals) || throw(ArgumentError("Lattice must be finite"))
            box = reshape(vals, 3, 3)
            d = box[1, 1] * (box[2, 2] * box[3, 3] - box[2, 3] * box[3, 2]) -
                box[1, 2] * (box[2, 1] * box[3, 3] - box[2, 3] * box[3, 1]) +
                box[1, 3] * (box[2, 1] * box[3, 2] - box[2, 2] * box[3, 1])
            isfinite(d) && d != 0 || throw(ArgumentError("Lattice must be nonsingular"))
        end
        (pbc !== nothing && pbc.captures[1] in ("T T T", "F F F")) || throw(ArgumentError("pbc must be \"T T T\" or \"F F F\""))
        ((box !== nothing) == (pbc.captures[1] == "T T T")) || throw(ArgumentError("Lattice and pbc metadata disagree"))
        positions = Matrix{Float64}(undef, 3, n)
        species = String[]
        for i in 1:n
            fields = split(strip(readline(io)))
            length(fields) == 4 || throw(ArgumentError("invalid atom row"))
            occursin(r"^[A-Za-z][A-Za-z0-9_+-]*$", fields[1]) || throw(ArgumentError("invalid species label"))
            push!(species, fields[1])
            coordinates = parse.(Float64, fields[2:4])
            all(isfinite, coordinates) || throw(ArgumentError("coordinates must be finite"))
            positions[:, i] .= coordinates
        end
        push!(frames, (positions=positions, species=species, time=time,
                       box_matrix=box, pbc=pbc === nothing ? nothing : split(pbc.captures[1]),
                       length_unit=lu_m.captures[1], time_unit=tu_m.captures[1]))
    end
    return frames
end

end
