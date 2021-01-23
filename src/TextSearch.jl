# This file is a part of TextSearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

module TextSearch
    import Base: broadcastable
    import StatsBase: fit, predict
    using SparseArrays
    using LinearAlgebra

    struct IdWeight
        id::Int32
        weight::Float32
    end

    include("textconfig.jl")
    include("normalize.jl")
    include("tokenize.jl")
    include("dvec.jl")
    include("vmodel.jl")
    include("distmodel.jl")
    include("entmodel.jl")
    include("multi.jl")
    include("svecutils.jl")
    include("invindex.jl")
    include("neardup.jl")
end
