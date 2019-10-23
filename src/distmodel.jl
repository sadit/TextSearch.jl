export DistModel, feed!, fix!

mutable struct DistModel <: Model
    tokens::Dict{Symbol, Vector{Float64}}
    config::TextConfig
    sizes::Vector{Int}
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
function fit(::Type{DistModel}, config::TextConfig, corpus, y; nclasses=0, weights=nothing, fix=true)
    if nclasses == 0
        nclasses = unique(y) |> length
    end
    
    model = DistModel(Dict{Symbol, Vector{Float64}}(), config, zeros(Int, nclasses))
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
end

"""
    feed!(model::DistModel, corpus, y)

DistModel objects support for incremental feed if `fix!` method is not called on `fit`
"""
function feed!(model::DistModel, corpus, y)
    config = model.config
    nclasses = length(model.sizes)
    n = 0
    println(stderr, "feeding DistModel with $(length(corpus)) items, classes: $(nclasses)")
    for (klass, text) in zip(y, corpus)
        for token in tokenize(config, text)
            token_dist = get(model.tokens, token, EMPTY_TOKEN_DIST)
            if length(token_dist) == 0
                token_dist = zeros(Float64, nclasses)
                model.tokens[token] = token_dist
            end
            token_dist[klass] += 1
        end

        model.sizes[klass] += 1
        n += 1
        n % 1000 == 0 && print(stderr, "*")
        n % 100000 == 0 && println(stderr, " dist: $(model.sizes), adv: $n")
    end
    println(stderr, "finished DistModel: $n processed items")

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
