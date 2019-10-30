module TextSearch
    import Base: broadcastable
    import StatsBase: fit, predict
    using SparseArrays
    using LinearAlgebra
    include("textconfig.jl")
    include("dvec.jl")
    include("io.jl")
    include("basicmodels.jl")
    include("distmodel.jl")
    include("entmodel.jl")
    include("svecutils.jl")
    include("invindex.jl")
    include("rocchio.jl")
    include("neardup.jl")
end
