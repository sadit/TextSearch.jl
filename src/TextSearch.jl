# This file is a part of TextSearch.jl

module TextSearch
    import Base: broadcastable
    import StatsBase: fit, predict
    using SimilaritySearch, InvertedFiles, LinearAlgebra, SparseArrays

    include("textconfig.jl")
    include("normalize.jl")
    include("tokenize.jl")
    include("bow.jl")
    include("vmodel.jl")
    include("emodel.jl")
    include("neardup.jl")
#    include("multi.jl")
end
