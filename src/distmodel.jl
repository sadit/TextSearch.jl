# This file is a part of TextSearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt
using CategoricalArrays
export DistModel, push!, fix!, prune

const DistVocabulary = Dict{Symbol, Vector{Float64}}

mutable struct DistModel <: TextModel
    tokens::DistVocabulary
    sizes::Vector{Int}
    initial_dist::Vector{Float64}
    m::Int  # vocabulary size (preserved after vocabulary pruning)   
    n::Int  # collection size
end

const EMPTY_TOKEN_DIST = Int[]

"""
    DistModel(tokenized_corpus, y::CategoricalArray; nclasses=0, weights=nothing, minocc=1, fix=true)

Creates a DistModel object using the specified `tokenized_corpus`
and its associated labels `y`. Optional parameters:
- `nclasses`: the number of classes
- `weights`: It has three different kind of values
   - an array of `nclasses` floating point to scale the value of each bin in the computed histogram.
   - `:balance` that indicates that `weights` must try to compensate the unbalance among classes
   - `:rand` set random weights for each class
   - `:none` let the computed histogram untouched
- `minocc`: minimum population to consider a token (without considering the smoothing factor).
- `fix`: if true, it stores the empirical probabilities instead of frequencies
"""
function DistModel(corpus::AbstractVector{BOW}, y::CategoricalArray; nclasses=0, weights=:balance, smooth::Real=0, minocc::Integer=1, fix=false)
    if nclasses == 0
        nclasses = levels(y) |> length
    end
	
    smooth = fill(convert(Float64, smooth), nclasses)
    model = DistModel(DistVocabulary(), zeros(Int, nclasses), smooth, 0, 0)
    for (bow, label) in zip(corpus, y.refs)
        push!(model, bow, label)
    end

	prune(model, first(smooth) * nclasses + minocc)

	if weights == :none
		# do nothing
	else
		if weights == :balance
			s = sum(model.sizes)
			weights = [s / x for x in model.sizes]  ## this produces nicer numbers, but it is the same than weights = [1 / x for x in model.sizes]
		elseif weights == :rand
			weights = [rand() for x in model.sizes]
		else
			error("Unknown '$weights' for weights parameter")
		end
		
        normalize!(model, weights)
    end

    fix && fix!(model)

    model
end

"""
	prune(model::DistModel, minocc)

Deletes tokens having a sum less of `minocc` as entries of its distribution.
Note that the `smooth` factor, all balancing methods after calling the `fix! function the values will sum to 1.0.
"""
function prune(model::DistModel, minocc)
	for (k, v) in model.tokens
		if sum(v) < minocc
			delete!(model.tokens, k)
		end
	end
end


"""
    push!(model::DistModel, bow::BOW, y)

Adds an example to the model
"""
function Base.push!(model::DistModel, bow::BOW, label::Integer)
    nclasses = length(model.sizes)

    for (sym, freq) in bow
        token_dist = get(model.tokens, sym, EMPTY_TOKEN_DIST)
        if length(token_dist) == 0
            token_dist = copy(model.initial_dist)
            model.tokens[sym] = token_dist
        end

        token_dist[label] += freq
    end

    model.sizes[label] += 1
    model.n += 1
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
    fix!(model::DistModel, use_soft_max=false)

Replaces frequencies by empirical probabilities in the model
"""
function fix!(model::DistModel, use_soft_max=false)
    nclasses = length(model.sizes)

	if use_soft_max
		for (token, dist) in model.tokens
			s = sum(exp(d) for d in dist)
			for i in 1:nclasses
				dist[i] = exp(dist[i]) / s
			end

		end
	else
		for (token, dist) in model.tokens
			s = sum(dist)
			for i in 1:nclasses
				dist[i] /= s
			end
		end
	end
    model
end
