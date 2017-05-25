module TextModel
using SimilaritySearch
# TODO: move all dependent code of SimilaritySearch to TextModel
include("io.jl")
include("textconfig.jl")
include("basicmodels.jl")
include("distmodel.jl")
include("entmodel.jl")
end
