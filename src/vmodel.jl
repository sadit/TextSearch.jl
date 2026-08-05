# This file is part of TextSearch.jl

export TextModel, VectorModel, trainsize, vocsize,
    TfWeighting, IdfWeighting, TpWeighting,
    FreqWeighting, BinaryLocalWeighting, BinaryGlobalWeighting,
    LocalWeighting, GlobalWeighting, weight, fit, vectorize, vectorize_corpus

#####
##
## LocalWeighting
##
#####
"""
    LocalWeighting

Abstract type for local weighting
"""
abstract type LocalWeighting end

"""
    TfWeighting()

Term frequency weighting
"""
struct TfWeighting <: LocalWeighting end

"""
    TpWeighting()

Term probability weighting
"""
struct TpWeighting <: LocalWeighting end

"""
    FreqWeighting()

Frequency weighting
"""
struct FreqWeighting <: LocalWeighting end

"""
    BinaryLocalWeighting()

The weight is 1 for known tokens, 0 for out of vocabulary tokens
"""
struct BinaryLocalWeighting <: LocalWeighting end

#####
##
## GlobalWeighting
##
#####
"""
    GlobalWeighting

Abstract type for global weighting
"""
abstract type GlobalWeighting end


"""
    IdfWeighting()

Inverse document frequency weighting
"""
struct IdfWeighting <: GlobalWeighting end


"""
    BinaryGlobalWeighting()

The weight is 1 for known tokens, 0 for out of vocabulary tokens
"""
struct BinaryGlobalWeighting <: GlobalWeighting end

#####
##
## TextModels
##
#####
"""
    TextModel

Abstract type for text-to-vector weighting models (see [`VectorModel`](@ref)).
"""
abstract type TextModel end

"""
    VectorModel{_G<:GlobalWeighting, _L<:LocalWeighting}

Combines a [`Vocabulary`](@ref) with a local/global term-weighting scheme
(e.g. [`TfWeighting`](@ref)+[`IdfWeighting`](@ref) for classical TF-IDF) to turn
bags of words into weighted sparse vectors (`SparseVector{Float32,Int32}`) via
[`vectorize`](@ref)/[`vectorize!`](@ref).
Build one with [`VectorModel(gw, lw, voc)`](@ref VectorModel).

# Fields
- `global_weighting`: the [`GlobalWeighting`](@ref) scheme (e.g. IDF), applied per-token, corpus-wide.
- `local_weighting`: the [`LocalWeighting`](@ref) scheme (e.g. TF), applied per-token, per-document.
- `voc`: the underlying [`Vocabulary`](@ref).
- `maxoccs`: maximum per-token occurrence count in `voc`, used by some local weightings.
- `weight`: precomputed per-token global weight (`weight[tokenID]`).

# Example

```julia
julia> corpus = ["hello world", "hello there", "the cat sat"];

julia> voc = Vocabulary(TextConfig(), corpus; verbose=false);

julia> model = VectorModel(IdfWeighting(), TfWeighting(), voc);

julia> vectorize(model, "hello world")
6-element SparseArrays.SparseVector{Float32, Int32} with 2 stored entries:
  [1]  =  0.369076
  [2]  =  0.929399
```
"""
mutable struct VectorModel{_G<:GlobalWeighting, _L<:LocalWeighting} <: TextModel
    global_weighting::_G
    local_weighting::_L
    voc::Vocabulary
    maxoccs::Int32
    weight::Vector{Float32}
end

function Base.show(io::IO, model::VectorModel; prefix="", indent="  ")
    println(io, prefix, "VectorModel:")
    prefix = indent * prefix
    println(io, prefix, "global_weighting: ", model.global_weighting)
    println(io, prefix, "local_weighting: ", model.local_weighting)
    println(io, prefix, "maxoccs: ", model.maxoccs)
    show(io, model.voc; prefix, indent)
end

"""
    VectorModel(gw::GlobalWeighting, lw::LocalWeighting, voc::Vocabulary; weight=nothing)

Creates a [`VectorModel`](@ref) for the given vocabulary `voc` using the local weighting
`lw` (e.g. [`TfWeighting`](@ref)) and global weighting `gw` (e.g. [`IdfWeighting`](@ref)).
The per-token global weight vector is computed from `voc` unless `weight` is given
explicitly (e.g. when reusing weights computed elsewhere, such as [`EntropyWeighting`](@ref)).

# Example

```julia
julia> corpus = ["hello world", "hello there", "the cat sat"];

julia> voc = Vocabulary(TextConfig(), corpus; verbose=false);

julia> model = VectorModel(IdfWeighting(), TfWeighting(), voc);

julia> length(model.weight)
6
```
"""
function VectorModel(gw::GlobalWeighting, lw::LocalWeighting, voc::Vocabulary; weight=nothing)
    vocsize(voc) > 0 || error("empty vocabulary")
    maxoccs = convert(Int32, maximum(voc.occs))
    W = weight === nothing ? Vector{Float32}(undef, vocsize(voc)) : weight
    model = VectorModel(gw, lw, voc, maxoccs, W)

    if weight === nothing
        for tokenID in eachindex(voc)
            model.weight[tokenID] = w_ = global_weighting(model, tokenID)
            # @assert w_ >= 0 "NEGATIVE WEIGHT $tokenID -- $w_"
        end
    end

    model
end

@inline trainsize(model::VectorModel) = trainsize(model.voc)
@inline vocsize(model::VectorModel) = vocsize(model.voc)

@inline Base.length(model::VectorModel) = length(model.voc)
@inline occs(model::VectorModel, tokenID::Integer) = occs(model.voc, tokenID)
@inline ndocs(model::VectorModel, tokenID::Integer) = ndocs(model.voc, tokenID)
@inline token(model::VectorModel, tokenID::Integer) = token(model.voc, tokenID)
@inline Base.eachindex(model::VectorModel) = eachindex(model.voc)
@inline weight(model::VectorModel, tokenID::Integer) = tokenID == 0 ? zero(eltype(model.weight)) : model.weight[tokenID]
@inline weight(model::VectorModel) = model.weight
@inline occs(model::VectorModel) = occs(model.voc)
@inline ndocs(model::VectorModel) = ndocs(model.voc)
@inline token(model::VectorModel) = token(model.voc)

"""
    table(model::VectorModel, TableConstructor)

Builds a Tables.jl-compatible table (e.g., a `DataFrame`) with one row per token,
using `TableConstructor` (e.g. `DataFrame`) as the row-table constructor. Columns are
`token`, `ndocs`, `occs`, and `weight`.

# Example

```julia
julia> using DataFrames

julia> corpus = ["hello world", "hello there", "the cat sat"];

julia> voc = Vocabulary(TextConfig(), corpus; verbose=false);

julia> model = VectorModel(IdfWeighting(), TfWeighting(), voc);

julia> table(model, DataFrame)
6×4 DataFrame
 Row │ token   ndocs  occs   weight
     │ String  Int32  Int32  Float32
─────┼────────────────────────────────
   1 │ hello       2      2  0.485427
   2 │ world       1      1  1.22239
   3 │ there       1      1  1.22239
   4 │ the         1      1  1.22239
   5 │ cat         1      1  1.22239
   6 │ sat         1      1  1.22239
```
"""
function table(model::VectorModel, TableConstructor)
    TableConstructor(; token=token(model), ndocs=ndocs(model), occs=occs(model), weight=weight(model))
end

Base.getindex(model::VectorModel, token::AbstractString) = model[token2id(model.voc, token)]

function Base.getindex(model::VectorModel, tokenID::Integer)
    id = convert(UInt32, tokenID)
    voc = model.voc
    if id == 0
        (; id=id, occs=zero(eltype(voc.occs)), ndocs=zero(eltype(voc.ndocs)), weight=zero(eltype(model.weight)), token="")
    else
        (; id=id, occs=voc.occs[id], ndocs=voc.ndocs[id], weight=model.weight[id], token=voc.token[id])
    end
end

"""
    filter_tokens(pred::Function, model::VectorModel)

Returns a copy of `model` reduced to the tokens for which `pred(t)` is `true`, where
`t` is a `(; id, occs, ndocs, weight, token)` named tuple (see also
[`filter_tokens(pred, voc::Vocabulary)`](@ref filter_tokens)).

# Example

```julia
julia> corpus = ["hello world", "hello there", "the cat sat"];

julia> voc = Vocabulary(TextConfig(), corpus; verbose=false);

julia> model = VectorModel(IdfWeighting(), TfWeighting(), voc);

julia> vocsize(filter_tokens(t -> t.ndocs >= 2, model))
1
```
"""
function filter_tokens(pred::Function, model::VectorModel)
    voc = model.voc
    V = Vocabulary(voc.textconfig, trainsize(voc), numtokens(voc))
    W = Vector{Float32}(undef, 0)
    
    for i in eachindex(voc)
        t = model[i]
        if pred(t)
            push_token!(V, t.token, t.occs, t.ndocs)
            push!(W, t.weight)
        end
    end

    VectorModel(model.global_weighting, model.local_weighting, V, model.maxoccs, W)
end


"""
    VectorizeBuffer(n=128)

Pooled per-thread scratch space for [`vectorize!`](@ref): `ids` accumulates every
in-vocabulary token id seen in a document (with repeats), which is then sorted and
run-length-encoded to recover per-token occurrence counts — the same merge-based
strategy [`sum(::AbstractVector{<:SparseVector})`](@ref sum) uses, avoiding the `Dict`
allocation/hashing a [`BOW`](@ref) would need for this per-call, performance-sensitive
path.
"""
struct VectorizeBuffer
    ids::Vector{Int32}

    function VectorizeBuffer(n=128)
        ids = Vector{Int32}()
        sizehint!(ids, n)
        new(ids)
    end
end

const VECTORIZE_CACHES = Channel{VectorizeBuffer}(Inf)

Base.empty!(buff::VectorizeBuffer) = (empty!(buff.ids); buff)

function _collect_token_ids!(ids::Vector{Int32}, voc::Vocabulary, text::AbstractString)
    tokenizerbuffer() do tok
        tokenlist = tokenize(borrowtokenizedtext, voc.textconfig, text, tok)
        for token in tokenlist
            tokenID = token2id(voc, token)
            zero(UInt32) != tokenID && push!(ids, tokenID)
        end
    end
    ids
end

function _collect_token_ids!(ids::Vector{Int32}, voc::Vocabulary, tokens::TokenizedText)
    for token in tokens
        tokenID = token2id(voc, token)
        zero(UInt32) != tokenID && push!(ids, tokenID)
    end
    ids
end

function _collect_token_ids!(ids::Vector{Int32}, voc::Vocabulary, messages)
    for text in messages
        _collect_token_ids!(ids, voc, text)
    end
    ids
end

"""
    vectorize!(buff::VectorizeBuffer, model::VectorModel, text; normalize=true, minweight=1e-6)

Tokenizes `text` and weights it using `model`'s local/global weighting scheme, returning
the result as a `SparseVector{Float32,Int32}`; entries with a weight below `minweight`
are dropped, and the result is L2-normalized unless `normalize=false`. `buff` is used as
scratch space (see [`VectorizeBuffer`](@ref); tokenization scratch space is borrowed
separately via [`tokenizerbuffer`](@ref)). See [`vectorize`](@ref) for a version that
manages the scratch buffer for you.

# Example

```julia
julia> corpus = ["hello world", "hello there", "the cat sat"];

julia> voc = Vocabulary(TextConfig(), corpus; verbose=false);

julia> model = VectorModel(IdfWeighting(), TfWeighting(), voc);

julia> buff = TextSearch.VectorizeBuffer();

julia> TextSearch.vectorize!(buff, model, "hello world")
6-element SparseArrays.SparseVector{Float32, Int32} with 2 stored entries:
  [1]  =  0.369076
  [2]  =  0.929399
```
"""
function vectorize!(buff::VectorizeBuffer, model::VectorModel, text; normalize=true, minweight=1e-6)
    ids = buff.ids
    empty!(ids)
    _collect_token_ids!(ids, model.voc, text)
    sort!(ids)
    n = length(ids)
    numtokens::Int = n  # total in-vocabulary token occurrences (TpWeighting)

    maxoccs::Int = 0
    if model.local_weighting isa TfWeighting
        i = 1
        while i <= n
            j = i
            while j < n && ids[j+1] == ids[i]
                j += 1
            end
            occs = j - i + 1
            occs > maxoccs && (maxoccs = occs)
            i = j + 1
        end
    end

    nnzidx = Vector{Int32}(undef, n)
    nnzval = Vector{Float32}(undef, n)

    i = 1
    k = 0
    while i <= n
        j = i
        while j < n && ids[j+1] == ids[i]
            j += 1
        end
        tokenID = ids[i]
        occs = j - i + 1
        w = local_weighting(model.local_weighting, occs, maxoccs, numtokens) * weight(model, tokenID)
        if w >= minweight
            k += 1
            nnzidx[k] = tokenID
            nnzval[k] = w
        end
        i = j + 1
    end

    resize!(nnzidx, k)
    resize!(nnzval, k)

    vec = SparseVector(vocsize(model), nnzidx, nnzval)
    normalize && normalize!(vec)
    vec
end

"""
    vectorize(model::VectorModel, text; normalize=true, minweight=1e-6)

Computes the weighted sparse vector (a `SparseVector{Float32,Int32}`) representation of
`text` under `model`. `text` can be a string or a list of strings (a multi-field document).

# Example

```julia
julia> corpus = ["hello world", "hello there", "the cat sat"];

julia> voc = Vocabulary(TextConfig(), corpus; verbose=false);

julia> model = VectorModel(IdfWeighting(), TfWeighting(), voc);

julia> vectorize(model, "hello world")
6-element SparseArrays.SparseVector{Float32, Int32} with 2 stored entries:
  [1]  =  0.369076
  [2]  =  0.929399
```
"""
function vectorize(model::VectorModel, text; normalize=true, minweight=1e-6)
    buff = take!(VECTORIZE_CACHES)
    try
        vectorize!(buff, model, text; normalize, minweight)
    finally
        put!(VECTORIZE_CACHES, buff)
    end
end

"""
    vectorize_corpus(model::VectorModel, corpus; normalize=true, minweight=1e-6, verbose=true)

Computes the [`vectorize`](@ref) representation of every document in `corpus`,
processed in parallel across threads (the batch size is picked automatically from the
size of `corpus`, as in `SimilaritySearch.getminbatch`).

# Example

```julia
julia> corpus = ["hello world", "hello there", "the cat sat"];

julia> voc = Vocabulary(TextConfig(), corpus; verbose=false);

julia> model = VectorModel(IdfWeighting(), TfWeighting(), voc);

julia> vectorize_corpus(model, corpus; verbose=false)[1]
6-element SparseArrays.SparseVector{Float32, Int32} with 2 stored entries:
  [1]  =  0.369076
  [2]  =  0.929399
```
"""
function vectorize_corpus(model::VectorModel, corpus; normalize=true, minweight=1e-6, verbose=true)
    corpus = collect(corpus)
    n = length(corpus)
    V = Vector{SparseVector{Float32,Int32}}(undef, n)
    minbatch = getminbatch(n)
    prog = Progress(n; dt=1, enabled=verbose, desc="vectorizing corpus")

    @BATCHES minbatch for i in 1:n
        V[i] = vectorize(model, corpus[i]; normalize, minweight)
        next!(prog)
    end

    V
end

# local weightings: TfWeighting, TpWeighting, FreqWeighting, BinaryLocalWeighting
# global weightings: IdfWeighting, BinaryGlobalWeighting

@inline local_weighting(::TfWeighting, occs, maxoccs, numtokens) = occs / maxoccs
@inline local_weighting(::FreqWeighting, occs, maxoccs, numtokens) = occs
@inline local_weighting(::TpWeighting, occs, maxoccs, numtokens) = occs / numtokens
@inline local_weighting(::BinaryLocalWeighting, occs, maxoccs, numtokens) = 1.0
@inline global_weighting(model::VectorModel{IdfWeighting}, tokenID) = @inbounds log2((0.5 + trainsize(model)) / (0.5 + ndocs(model, tokenID)))
@inline global_weighting(model::VectorModel{BinaryGlobalWeighting}, tokenID) = 1.0
