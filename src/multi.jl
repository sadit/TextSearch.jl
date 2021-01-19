# This file is a part of TextSearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

import KCenters
export update!, joinmodel
"""
    update!(a::VectorModel, b::VectorModel)

Updates `a` with `b` inplace; returns `a`.
"""
function update!(a::VectorModel, b::VectorModel)
m = a.m
for (token, idfreq) in b.tokens
    x = get(a.tokens, token, nothing)
    if x === nothing
        m += 1
        a.tokens[token] = IdFreq(m, x.freq)
    else
        a.tokens[token] = IdFreq(idfreq.id, idfreq.freq + x.freq)
    end
end

a.maxfreq = max(a.maxfreq, b.maxfreq)
a.n += b.n
a.m = m
a.id2token = Dict(idfreq.id => t for (t, idfreq) in a.tokens)
a
end

"""
    joinmodel(arr::AbstractVector{VectorModel})
    joinmodel(arr::AbstractVector{EntModel})

Joins a list of models into a single one; frequencies and weights are computed as the mean of its non-missing occurrences
"""
function joinmodel(arr::AbstractVector{VectorModel})
    item = first(arr)
    # Vocabulary = Dict{Symbol, IdFreq}
    tokens_ = Dict{Symbol,Int}()
    counts_ = Dict{Symbol,Int32}()

    maxfreq = 0
    for model in arr
        maxfreq = max(maxfreq, model.maxfreq)
        for (k, v) in model.tokens
            tokens_[k] = get(tokens_, k, 0.0) + v.freq # ceil(Int, 0.5 * (
            counts_[k] = get(counts_, k, 0) + 1
        end
    end

    tokens = TextSearch.Vocabulary()
    for (k, v) in tokens_
        tokens[k] = TextSearch.IdFreq(length(tokens) + 1, ceil(Int, v / counts_[k]))
    end

    id2token = Dict(idfreq.id => token for (token, idfreq) in tokens)
    VectorModel(item.config, tokens, id2token, maxfreq, length(tokens_), item.n)
end


function glue(arr::AbstractVector{EntModel})
    item = first(arr)
    tokens_ = Dict{Symbol,Float32}()
    counts_ = Dict{Symbol,Int32}()

    for model in arr
        for (k, v) in model.tokens
            tokens_[k] = get(tokens_, k, 0.0) + v.weight
            counts_[k] = get(counts_, k, 0) + 1
        end
    end

    tokens = TextSearch.WeightedVocabulary()
    for (k, v) in tokens_
        tokens[k] = TextSearch.IdWeight(length(tokens) + 1, v / counts_[k])
    end

    id2token = Dict(w.id => token for (token, w) in tokens)

    EntModel(item.config, tokens, id2token, length(tokens), item.n)
end
