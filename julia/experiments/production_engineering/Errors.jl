module ProductionErrors

export ProductionError, CheckpointCorruption, CheckpointIncompatibility,
       UnsupportedCheckpointVersion, InvalidCheckpoint, DeterminismError,
       InvalidInput, SingularGeometry, UnstableOperator, NonConvergence,
       UnsupportedFeature, ResourceRefusal

abstract type ProductionError <: Exception end

struct CheckpointCorruption <: ProductionError
    message::String
end
struct CheckpointIncompatibility <: ProductionError
    message::String
end
struct UnsupportedCheckpointVersion <: ProductionError
    found::Int
    supported::Int
end
struct InvalidCheckpoint <: ProductionError
    message::String
end
struct DeterminismError <: ProductionError
    message::String
end
struct InvalidInput <: ProductionError
    message::String
end
struct SingularGeometry <: ProductionError
    message::String
end
struct UnstableOperator <: ProductionError
    message::String
end
struct NonConvergence <: ProductionError
    message::String
end
struct UnsupportedFeature <: ProductionError
    message::String
end
struct ResourceRefusal <: ProductionError
    message::String
end

Base.showerror(io::IO, e::CheckpointCorruption) = print(io, "checkpoint corruption: ", e.message)
Base.showerror(io::IO, e::CheckpointIncompatibility) = print(io, "checkpoint incompatibility: ", e.message)
Base.showerror(io::IO, e::UnsupportedCheckpointVersion) = print(io, "unsupported checkpoint version ", e.found, " (supported ", e.supported, ")")
Base.showerror(io::IO, e::InvalidCheckpoint) = print(io, "invalid checkpoint: ", e.message)
Base.showerror(io::IO, e::DeterminismError) = print(io, "determinism error: ", e.message)
Base.showerror(io::IO, e::InvalidInput) = print(io, "invalid input: ", e.message)
Base.showerror(io::IO, e::SingularGeometry) = print(io, "singular geometry: ", e.message)
Base.showerror(io::IO, e::UnstableOperator) = print(io, "unstable operator: ", e.message)
Base.showerror(io::IO, e::NonConvergence) = print(io, "non-convergence: ", e.message)
Base.showerror(io::IO, e::UnsupportedFeature) = print(io, "unsupported feature: ", e.message)
Base.showerror(io::IO, e::ResourceRefusal) = print(io, "resource refusal: ", e.message)

end
