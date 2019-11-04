export DistModel, feed!, fix!

const DistVocabulary = Dict{Symbol, Vector{Float64}}

mutable struct DistModel <: Model
    config::TextConfig
    tokens::DistVocabulary
    sizes::Vector{Int}
    initial_dist::Vector{Float64}
    m::Int
    n::Int
end

const EMPTY_TOKEN_DIST = Int[]

"""
    fit(::Type{DistModel}, config::TextConfig, corpus, y; nclasses=0, weights=nothing, fix=true)

Creates a DistModel object using the specified `corpus` (an array of strings or an array of arrays of strings);
and its associated labels `y`. Optional parameters:
- `nclasses`: the number of classes
- `weights`: It has three different kind of values
   - an array of `nclasses` floating point to scale the value of each bin in the computed histogram.
   - the keyword :balance` that indicates that `weights` must try to compensate the unbalance among classes
   - nothing: let the computed histogram untouched
- `fix`: if true, it stores the empirical probabilities instead of frequencies
"""
function fit(::Type{DistModel}, config::TextConfig, corpus, y; nclasses=0, weights=:balance, fix=false, smooth::Real=0)
    if nclasses == 0
        nclasses = unique(y) |> length
    end
    smooth = fill(convert(Float64, smooth), nclasses)
    counters = zeros(Int, nclasses)
    model = DistModel(config, DistVocabulary(), counters, smooth, 0, 0)
    feed!(model, corpus, y)
    if weights == :balance
        s = sum(model.sizes)
        weights = [s / x  for x in model.sizes]
    end

    if weights !== nothing
        normalize!(model, weights)
    end

    if fix
        fix!(model)
    end

    model
end

"""
    feed!(model::DistModel, corpus, y)

DistModel objects support for incremental feed if `fix!` method is not called on `fit`
"""
function feed!(model::DistModel, corpus, y)
    config = model.config
    nclasses = length(model.sizes)

    for (klass, text) in zip(y, corpus)
        for token in tokenize(config, text)
            token_dist = get(model.tokens, token, EMPTY_TOKEN_DIST)
            if length(token_dist) == 0
                token_dist = copy(model.initial_dist)
                model.tokens[token] = token_dist
            end
            token_dist[klass] += 1
        end

        model.sizes[klass] += 1
    end
    
    model.n += length(corpus)
    model.m = length(model.tokens)
    model
end

"""
    normalize!(model::DistModel, weights)

Multiply weights in each histogram, e.g., looking for compensating unbalance
"""
function normalize!(model::DistModel, weights)
    nclasses = length(model.sizes)

    for (token, hist) in model.tokens
        for i in 1:nclasses
            hist[i] *= weights[i]
        end
    end
end

"""
    fix!(model::DistModel)

Replaces frequencies by empirical probabilities in the model
"""
function fix!(model::DistModel)
    nclasses = length(model.sizes)

    for (token, dist) in model.tokens
        s = sum(dist)
        for i in 1:nclasses
            dist[i] /= s
        end

    end

    model
end
