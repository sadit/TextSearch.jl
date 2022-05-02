# This file is a part of TextSearch.jl

module TextSearch
    import Base: broadcastable
    import StatsBase: fit, predict
    using SimilaritySearch, InvertedFiles, LinearAlgebra, SparseArrays

    include("textconfig.jl")
    include("normalize.jl")
    include("tokenize.jl")
    include("dvec.jl")
    include("voc.jl")
    include("bow.jl")
    include("sparseconversions.jl")
    include("vmodel.jl")
    include("emodel.jl")
end
