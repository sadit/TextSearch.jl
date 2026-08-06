# This file is a part of TextSearch.jl

export TokenizationConfig, alltokengenerators

"""
    TokenizationConfig(;
        nlist::Vector=Int8[],
        mark_token_type::Bool=true,
        generators::Vector{<:AbstractTokenGenerator}=AbstractTokenGenerator[]
    )

Defines the tokenization stage of a [`TextConfig`](@ref) (see its `tokenization` field):
unigrams and word n-grams, computed from the output of the normalization stage.

- `nlist`: a list of words n-grams to use (`1` emits plain unigrams via
  [`UnigramGenerator`](@ref), any other value emits word n-grams via
  [`NWordGenerator`](@ref)).
- `mark_token_type`: each token is `marked` with its type (nword) when is true.
- `generators`: extra [`AbstractTokenGenerator`](@ref)s to run in addition to the ones
  `nlist` builds; this is the extension point for adding new kinds of tokens (e.g.
  character q-grams, skip-grams, or collocations, none of which are built-in anymore)
  without needing a new `TokenizationConfig` keyword argument (see
  [`alltokengenerators`](@ref)).

Note: If nlist and generators are both empty, then it defaults to nlist=[1]

# Example

```julia
julia> cfg = TokenizationConfig(nlist=[1, 2]);

julia> collect(tokenize(TextConfig(tokenization=cfg), "cats sat"))
["cats", "sat", "cats sat\tn"]
```
"""
Base.@kwdef struct TokenizationConfig
    nlist::Vector{Int8} = Int8[]
    mark_token_type::Bool = true
    generators::Vector{AbstractTokenGenerator} = AbstractTokenGenerator[]

    function TokenizationConfig(nlist::AbstractVector, mark_token_type::Bool, generators::AbstractVector)
        if length(nlist) == length(generators) == 0
            nlist = [1]
        end
        nlist = sort!(Vector{Int8}(nlist))
        generators = Vector{AbstractTokenGenerator}(generators)

        new(nlist, mark_token_type, generators)
    end
end

function TokenizationConfig(c::TokenizationConfig;
        nlist=c.nlist,
        mark_token_type=c.mark_token_type,
        generators::AbstractVector=c.generators
    )
    TokenizationConfig(nlist, mark_token_type, generators)
end

function Base.show(io::IO, c::TokenizationConfig; prefix="", indent="  ")
    println(io, prefix, "TokenizationConfig:")
    prefix = indent * prefix
    for f in fieldnames(TokenizationConfig)
        print(io, prefix, indent)
        _show_field(io, f, getfield(c, f))
    end
end

"""
    alltokengenerators(cfg::TokenizationConfig)::Vector{AbstractTokenGenerator}

Builds the full, ordered list of [`AbstractTokenGenerator`](@ref)s `cfg` runs: the
built-in ones implied by `cfg.nlist`, followed by `cfg.generators` (any extra/custom
generators). Called once per [`tokenize`](@ref) invocation.

# Example

```julia
julia> alltokengenerators(TokenizationConfig(nlist=[1, 2]))
2-element Vector{AbstractTokenGenerator}:
 UnigramGenerator()
 NWordGenerator(2)
```
"""
function alltokengenerators(cfg::TokenizationConfig)
    gens = AbstractTokenGenerator[]

    for q in cfg.nlist
        push!(gens, q == 1 ? UnigramGenerator() : NWordGenerator(q))
    end

    append!(gens, cfg.generators)
    gens
end
