# This file is a part of TextSearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt
module TextSearch
    import Base: broadcastable
    import StatsBase: fit, predict
    using LinearAlgebra
    
    struct IdWeight
        id::UInt64
        weight::Float64
    end

    include("textconfig.jl")
    include("normalize.jl")
#    include("tokenmap.jl")
    include("tokenize.jl")
    include("dvec.jl")
    include("bow.jl")
    include("vmodel.jl")
    include("emodel.jl")
#    include("distmodel.jl")
#    include("multi.jl")
    include("svecutils.jl")
    include("invindex.jl")
    include("neardup.jl")
end
