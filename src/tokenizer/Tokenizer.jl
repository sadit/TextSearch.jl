# This file is a part of TextSearch.jl

module Tokenizer

using SimilaritySearch: getminbatch, @BATCHES
using ProgressMeter

include("buffer.jl")
include("generators.jl")
include("defaults.jl")
include("tokentrans.jl")
include("normalization.jl")
include("tokenization.jl")
include("textconfig.jl")
include("normalize.jl")
include("tokenize.jl")

const TOKENIZER_CACHES = Channel{TokenizerBuffer}(Inf)

function __init__()
    for _ in 1:2*Threads.nthreads()+4
        put!(TOKENIZER_CACHES, TokenizerBuffer())
    end
end

end
