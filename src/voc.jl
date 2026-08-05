# This file is a part of TextSearch.jl

export Vocabulary, occs, ndocs, token, vocsize, trainsize, numtokens, filter_tokens, tokenize_and_append!, merge_voc, update_voc!, vocabulary_from_thesaurus, token2id, encode, decode, table

"""
    Vocabulary

Holds the token ⇄ id mapping produced while parsing a corpus, along with per-token
occurrence and document-frequency counters. A `Vocabulary` is the entry point of the
processing pipeline: it is built from a [`TextConfig`](@ref) and a corpus, and is
later consumed by [`VectorModel`](@ref), [`BM25Scorer`](@ref), and [`bagofwords`](@ref).

# Fields
- `textconfig`: the [`TextConfig`](@ref) used to tokenize the corpus that produced this vocabulary.
- `token`: `id -> token` string table.
- `occs`: `id -> total number of occurrences` of the token across the corpus.
- `ndocs`: `id -> number of documents` containing the token.
- `token2id`: `token -> id` reverse mapping (`0` means "unknown token").
- `trainsize`: number of documents used to build the vocabulary.
- `numtokens`: total number of (non-unique) tokens seen while building the vocabulary.

# Example

```julia
julia> voc = Vocabulary(TextConfig(), ["hello world", "hello there"]; verbose=false);

julia> vocsize(voc)
3

julia> token2id(voc, "hello")
0x00000001
```
"""
struct Vocabulary
    textconfig::TextConfig
    token::Vector{String}
    occs::Vector{Int32}
    ndocs::Vector{Int32}
    token2id::Dict{String,UInt32}
    trainsize::Ref{Int64}
    numtokens::Ref{Int64}
end

function Base.show(io::IO, voc::Vocabulary; prefix="", indent="  ")
    println(io, prefix, "Vocabulary:")
    prefix = indent * prefix
    println(io, prefix, "vocsize: ", vocsize(voc))
    println(io, prefix, "trainsize: ", trainsize(voc))
    println(io, prefix, "numtokens: ", numtokens(voc))
    println(io, prefix, "avgdoclen: ", avgdoclen(voc))
    show(io, voc.textconfig; prefix, indent)
end

"""
    token2id(voc::Vocabulary, tok::AbstractString)::UInt32

Looks up the id of `tok` in `voc`; returns `0` when `tok` is out of vocabulary.

# Example

```julia
julia> voc = Vocabulary(TextConfig(), ["hello world"]; verbose=false);

julia> token2id(voc, "hello")
0x00000001

julia> token2id(voc, "unknown")
0x00000000
```
"""
token2id(voc::Vocabulary, tok::AbstractString) = get(voc.token2id, tok, zero(UInt32))

"""
    decode(voc::Vocabulary, bow::Dict)

Converts a `Dict` sparse vector indexed by token id (e.g., a [`BOW`](@ref)) into a
`Dict` indexed by the corresponding token string.

# Example

```julia
julia> voc = Vocabulary(TextConfig(), ["hello world"]; verbose=false);

julia> decode(voc, bagofwords(voc, "hello hello"))
Dict{String, Int32}("hello" => 2)
```
"""
function decode(voc::Vocabulary, bow::Dict)
    Dict(voc.token[k] => v for (k, v) in bow)
end

"""
    encode(voc::Vocabulary, bow::Dict)

Converts a `Dict` sparse vector indexed by token string into a `Dict` indexed by
token id, the inverse of [`decode`](@ref).

# Example

```julia
julia> voc = Vocabulary(TextConfig(), ["hello world"]; verbose=false);

julia> encode(voc, Dict("hello" => 2))
Dict{UInt32, Int64}(0x00000001 => 2)
```
"""
function encode(voc::Vocabulary, bow::Dict)
    Dict(token2id(voc, k) => v for (k, v) in bow)
end

"""
    table(voc::Vocabulary, TableConstructor)

Builds a Tables.jl-compatible table (e.g., a `DataFrame`) with one row per token,
using `TableConstructor` (e.g. `DataFrame`) as the row-table constructor. Columns are
`token`, `ndocs`, and `occs`.

# Example

```julia
julia> using DataFrames

julia> corpus = ["hello world", "hello there", "the cat sat"];

julia> voc = Vocabulary(TextConfig(), corpus; verbose=false);

julia> table(voc, DataFrame)
6×3 DataFrame
 Row │ token   ndocs  occs
     │ String  Int32  Int32
─────┼──────────────────────
   1 │ hello       2      2
   2 │ world       1      1
   3 │ there       1      1
   4 │ the         1      1
   5 │ cat         1      1
   6 │ sat         1      1
```
"""
function table(voc::Vocabulary, TableConstructor)
    TableConstructor(; voc.token, voc.ndocs, voc.occs)
end

"""
    vocabulary_from_thesaurus(textconfig::TextConfig, tokens::AbstractVector)

Creates a [`Vocabulary`](@ref) directly from a list of tokens (a thesaurus), instead
of tokenizing a corpus; every token is registered with `occs=1` and `ndocs=1`.

# Example

```julia
julia> voc = vocabulary_from_thesaurus(TextConfig(), ["cat", "dog", "bird"]);

julia> vocsize(voc)
3

julia> token2id(voc, "cat")
0x00000001
```
"""
function vocabulary_from_thesaurus(textconfig::TextConfig, tokens::AbstractVector)
    n = length(tokens)
    voc = Vocabulary(textconfig, n, n)
    for t in tokens
        push_token!(voc, t, 1, 1)
    end

    voc
end

"""
    Vocabulary(textconfig::TextConfig, trainsize::Int, numtokens::Int)

Creates an empty `Vocabulary` (no tokens registered yet) preallocated with capacity
hints based on `trainsize` (following Heaps' law). `trainsize` and `numtokens` may be
`0` when unknown ahead of time; use [`push_token!`](@ref) or [`tokenize_and_append!`](@ref)
to fill it, or use [`Vocabulary(textconfig, corpus)`](@ref Vocabulary) to build it directly
from a corpus.

# Example

```julia
julia> voc = Vocabulary(TextConfig(), 0, 0);

julia> TextSearch.push_token!(voc, "cat"; occs=1, ndocs=1)
0x00000001

julia> vocsize(voc)
1
```
"""
function Vocabulary(textconfig::TextConfig, trainsize::Int64, numtokens::Int64)
    # n == 0 means unknown
    voc = Vocabulary(textconfig, String[], Int32[], Int32[], Dict{String,UInt32}(), Ref(trainsize), Ref(numtokens))
    vocsize = ceil(Int, trainsize^0.6)  # approx based on Heaps law
    sizehint!(voc.token, vocsize)
    sizehint!(voc.occs, vocsize)
    sizehint!(voc.ndocs, vocsize)
    sizehint!(voc.token2id, vocsize)
    voc
end

function vocab_from_small_collection(textconfig::TextConfig, corpus::AbstractVector; minbatch::Int=0)
    voc = Vocabulary(textconfig, length(corpus), 0)
    tokenize_and_append!(voc, corpus; minbatch)
    voc
end

"""
    Vocabulary(textconfig::TextConfig, corpus; minbatch=0, buffsize=2^16, verbose=true)

Tokenizes `corpus` under `textconfig` and builds the resulting [`Vocabulary`](@ref).
`corpus` can be any vector of documents (each document a string or a list of strings)
or an iterable/generator of documents (useful for corpora too large to fit in memory);
in the generator case, documents are consumed and tokenized in batches of `buffsize`.
`minbatch` controls the batch size used for multithreading (`0` picks an automatic value).

# Example

```julia
julia> voc = Vocabulary(TextConfig(), ["hello world", "hello there"]; verbose=false);

julia> vocsize(voc), trainsize(voc)
(3, 2)
```
"""
function Vocabulary(textconfig::TextConfig, corpusgenerator; minbatch::Int=0, buffsize::Int=2^16, verbose::Bool=true)
    if corpusgenerator isa AbstractVector && length(corpusgenerator) <= buffsize
        return vocab_from_small_collection(textconfig, corpusgenerator; minbatch)
    end

    voc = Vocabulary(textconfig, 0, 0)
    len = 0
    corpus = []
    sizehint!(corpus, buffsize)
    @showprogress dt=1 enabled=verbose desc="vocabulary:" for doc in corpusgenerator
        push!(corpus, doc)

        if length(corpus) == buffsize
            # verbose && (@info "computing vocabulary -- advance: $len - buffsize: $buffsize")
            len += buffsize
            tokenize_and_append!(voc, corpus; minbatch)
            empty!(corpus)
        end 
    end

    if length(corpus) > 0
        len += length(corpus)
        tokenize_and_append!(voc, corpus; minbatch)
    end

    voc.trainsize[] = len
    voc
end

const BOW_CACHES = Channel{BOW}(Inf)

function _locked_tokenize_and_push(voc, doc, bow::BOW, l)
    tokenizerbuffer() do tok
        tokenlist = tokenize(borrowtokenizedtext, voc.textconfig, doc, tok)
        for token in tokenlist
            id = 0
            lock(l)
            try
                id = push_token!(voc, token, 1, 0)
            finally
                unlock(l)
                bow[id] = 1
            end
        end
    end
end

"""
    tokenize_and_append!(voc::Vocabulary, corpus; minbatch=0)

Parse each document in the given corpus and appends each token to the vocabulary.

# Example

```julia
julia> voc = Vocabulary(TextConfig(), 0, 0);

julia> tokenize_and_append!(voc, ["hello world", "hello there"]);

julia> vocsize(voc)
3
```
"""
function tokenize_and_append!(voc::Vocabulary, corpus; minbatch=0)
    l = Threads.SpinLock()
    n = length(corpus)
    minbatch = minbatch > 0 ? minbatch : getminbatch(n)

    @BATCHES minbatch begin
    @BEGINBATCH
        batch_numtokens = 0
        batch_ndocs = Dict{UInt32,Int}()
    @LOOP for i in 1:n
        doc = corpus[i]
        bow = take!(BOW_CACHES)

        try
            empty!(bow)
            if doc isa AbstractVector
                for text in doc
                    _locked_tokenize_and_push(voc, text, bow, l)
                end
            else # if doc isa AbstractString
                _locked_tokenize_and_push(voc, doc, bow, l)
            end

            batch_numtokens += length(bow)
            for id in keys(bow)
                batch_ndocs[id] = get(batch_ndocs, id, 0) + 1
            end
        finally
            put!(BOW_CACHES, bow)
        end
    end
    @ENDBATCH
        lock(l)
        try
            voc.numtokens[] += batch_numtokens
            for (id, c) in batch_ndocs
                voc.ndocs[id] += c
            end
        finally
            unlock(l)
        end
    end

    voc
end

Base.length(voc::Vocabulary) = length(voc.occs)
Base.eachindex(voc::Vocabulary) = eachindex(voc.occs)

"""
    vocsize(voc::Vocabulary)

Number of unique tokens in `voc`.
"""
vocsize(voc::Vocabulary) = length(voc)

"""
    trainsize(voc::Vocabulary)

Number of documents used to build `voc`.
"""
trainsize(voc::Vocabulary) = voc.trainsize[]

"""
    numtokens(voc::Vocabulary)

Total number of (non-unique) tokens seen while building `voc`.
"""
numtokens(voc::Vocabulary) = voc.numtokens[]

"""
    avgdoclen(voc::Vocabulary)

Average document length in tokens (`numtokens(voc) / trainsize(voc)`), used by [`BM25Scorer`](@ref).
"""
avgdoclen(voc::Vocabulary) = numtokens(voc) / trainsize(voc)

"""
    ndocs(voc::Vocabulary, tokenID::Integer)
    ndocs(voc::Vocabulary)

Number of documents containing the token `tokenID` (`0` is out-of-vocabulary and yields
`0` instead of erroring), or the whole per-token vector when called without a `tokenID`.

# Example

```julia
julia> voc = Vocabulary(TextConfig(), ["hello world", "hello there"]; verbose=false);

julia> ndocs(voc, token2id(voc, "hello"))
2
```
"""
ndocs(voc::Vocabulary, tokenID::Integer) = tokenID == 0 ? zero(eltype(voc.ndocs)) : voc.ndocs[tokenID]

"""
    occs(voc::Vocabulary, tokenID::Integer)
    occs(voc::Vocabulary)

Total occurrences of the token `tokenID` across the corpus (`0` is out-of-vocabulary and
yields `0` instead of erroring), or the whole per-token vector when called without a `tokenID`.

# Example

```julia
julia> voc = Vocabulary(TextConfig(), ["hello world", "hello there"]; verbose=false);

julia> occs(voc, token2id(voc, "hello"))
2
```
"""
occs(voc::Vocabulary, tokenID::Integer) = tokenID == 0 ? zero(eltype(voc.occs)) : voc.occs[tokenID]

"""
    token(voc::Vocabulary, tokenID::Integer)
    token(voc::Vocabulary)

The token string for `tokenID` (`0` is out-of-vocabulary and yields `""` instead of
erroring), or the whole token vector when called without a `tokenID`.

# Example

```julia
julia> voc = Vocabulary(TextConfig(), ["hello world", "hello there"]; verbose=false);

julia> token(voc, token2id(voc, "hello"))
"hello"
```
"""
token(voc::Vocabulary, tokenID::Integer) = tokenID == 0 ? "" : voc.token[tokenID]

@inline occs(voc::Vocabulary) = voc.occs
@inline ndocs(voc::Vocabulary) = voc.ndocs
@inline token(voc::Vocabulary) = voc.token

"""
    push_token!(voc::Vocabulary, token, occs::Integer, ndocs::Integer)
    push_token!(voc::Vocabulary, token; occs::Integer=0, ndocs::Integer=0)

Registers `token` in `voc` if not already present (assigning it a new id), or accumulates
`occs`/`ndocs` into its existing entry otherwise. Returns the token's id.

# Example

```julia
julia> voc = Vocabulary(TextConfig(), 0, 0);

julia> TextSearch.push_token!(voc, "cat"; occs=1, ndocs=1)
0x00000001
```
"""
function push_token!(voc::Vocabulary, token, occs::Integer, ndocs::Integer)
    id = token2id(voc, token)

    if id == 0
        id = length(voc) + 1
        push!(voc.token, token)
        push!(voc.occs, occs)
        push!(voc.ndocs, ndocs)
        voc.token2id[token] = id
    else
        voc.occs[id] += occs
        voc.ndocs[id] += ndocs
    end

    id
end

function push_token!(voc::Vocabulary, token; occs::Integer=0, ndocs::Integer=0)
    push_token!(voc, token, occs, ndocs)
end

function append_tokens!(voc::Vocabulary, tokens; occs::Integer=0, ndocs::Integer=0)
    for token in tokens
        push_token!(voc, token, occs, ndocs)
    end
end

itertokenid(idlist::AbstractVector) = idlist 
itertokenid(idlist::AbstractVector{IdWeight}) = (p.id for p in idlist) 
itertokenid(idlist::AbstractVector{IdIntWeight}) = (p.id for p in idlist) 
itertokenid(idlist::AbstractVector{<:NamedTuple}) = (p.id for p in idlist) 
itertokenid(idlist::Dict) = keys(idlist) 
itertokenid(idlist::AbstractKnn) = IdView(idlist)

Base.getindex(voc::Vocabulary, idlist) = [voc[i] for i in itertokenid(idlist)]
Base.getindex(voc::Vocabulary, token::AbstractString) = voc[get(voc.token2id, token, 0)]

function Base.getindex(voc::Vocabulary, tokenID::Integer)
    id = convert(UInt32, tokenID)

    if id == 0
        (; id=id, occs=zero(eltype(voc.occs)), ndocs=zero(eltype(voc.ndocs)), token="")
    else
        (; id=id, occs=voc.occs[id], ndocs=voc.ndocs[id], token=voc.token[id])
    end
end
