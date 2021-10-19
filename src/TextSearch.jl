# This file is a part of TextSearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt
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
