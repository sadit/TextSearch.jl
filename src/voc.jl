# This file is a part of TextSearch.jl

export Vocabulary, getoccs, getndocs, gettoken, vocsize, gettrainsize, getnumtokens, avgdoclen, filter_tokens, tokenize_and_append!, merge_voc, update_voc!, vocabulary_from_thesaurus, token2id, encode, decode, table

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
    println(io, prefix, "trainsize: ", gettrainsize(voc))
    println(io, prefix, "numtokens: ", getnumtokens(voc))
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

function vocab_from_small_collection(textconfig::TextConfig, corpus::AbstractVector; isnormalized::Bool=false)
    voc = Vocabulary(textconfig, length(corpus), 0)
    tokenize_and_append!(voc, corpus; isnormalized)
    voc
end

"""
    Vocabulary(textconfig::TextConfig, corpus; buffsize=2^16, isnormalized::Bool=false, verbose=true)

Tokenizes `corpus` under `textconfig` and builds the resulting [`Vocabulary`](@ref).
`corpus` can be any vector of documents (each document a string or a list of strings)
or an iterable/generator of documents (useful for corpora too large to fit in memory);
in the generator case, documents are consumed and tokenized in batches of `buffsize`.

# Example

```julia
julia> voc = Vocabulary(TextConfig(), ["hello world", "hello there"]; verbose=false);

julia> vocsize(voc), gettrainsize(voc)
(3, 2)
```
"""
function Vocabulary(textconfig::TextConfig, corpusgenerator; buffsize::Int=2^16, isnormalized::Bool=false, verbose::Bool=true)
    if corpusgenerator isa AbstractVector && length(corpusgenerator) <= buffsize
        return vocab_from_small_collection(textconfig, corpusgenerator; isnormalized)
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
            tokenize_and_append!(voc, corpus; isnormalized)
            empty!(corpus)
        end
    end

    if length(corpus) > 0
        len += length(corpus)
        tokenize_and_append!(voc, corpus; isnormalized)
    end

    voc.trainsize[] = len
    voc
end

const BOW_CACHES = Channel{BOW}(Inf)

"""
    _VocabularyBatch

The counts one batch of documents produces, before they reach the shared [`Vocabulary`](@ref):
`tokens` in order of first appearance within the batch, with `occs`/`ndocs` parallel to it and
`index` mapping a token to its position.
"""
struct _VocabularyBatch
    tokens::Vector{String}
    index::Dict{String,Int32}
    occs::Vector{Int64}
    ndocs::Vector{Int64}
    numtokens::Int
end

function _tokenize_and_count!(tokens, index, occs, ndocs, bow::BOW, textconfig::TextConfig, doc; isnormalized::Bool=false)
    # Returns how many tokens `doc` has. `bow` cannot answer that: it is filled as a SET
    # (`bow[i] = 1`) because its only job is to say which tokens the document contained, for
    # `ndocs`. Reading a token count off it -- `length(bow)`, or equivalently summing its
    # values -- yields the number of DISTINCT tokens instead, which is what made `numtokens`,
    # and therefore `avgdoclen`, mean something other than its documented "average document
    # length in tokens".
    ntokens = Ref(0)
    tokenizerbuffer() do tok
        tokenlist = tokenize(borrowtokenizedtext, textconfig, doc, tok; isnormalized)
        ntokens[] = length(tokenlist)
        for token in tokenlist
            i = get(index, token, zero(Int32))
            if i == 0
                push!(tokens, token)
                push!(occs, 0)
                push!(ndocs, 0)
                i = index[token] = length(tokens)
            end
            occs[i] += 1
            bow[i] = 1
        end
    end
    ntokens[]
end

function _count_batch(textconfig::TextConfig, corpus, range; isnormalized::Bool=false)
    tokens = String[]
    index = Dict{String,Int32}()
    occs = Int64[]
    ndocs = Int64[]
    numtokens = 0
    bow = take!(BOW_CACHES)

    try
        for i in range
            doc = corpus[i]
            empty!(bow)
            if doc isa AbstractVector
                for text in doc
                    numtokens += _tokenize_and_count!(tokens, index, occs, ndocs, bow, textconfig, text; isnormalized)
                end
            else # if doc isa AbstractString
                numtokens += _tokenize_and_count!(tokens, index, occs, ndocs, bow, textconfig, doc; isnormalized)
            end

            for j in keys(bow)
                ndocs[j] += 1
            end
        end
    finally
        put!(BOW_CACHES, bow)
    end

    _VocabularyBatch(tokens, index, occs, ndocs, numtokens)
end

"""
    tokenize_and_append!(voc::Vocabulary, corpus; isnormalized::Bool=false)

Parse each document in the given corpus and appends each token to the vocabulary.

# Example

```julia
julia> voc = Vocabulary(TextConfig(), 0, 0);

julia> tokenize_and_append!(voc, ["hello world", "hello there"]);

julia> vocsize(voc)
3
```
"""
function tokenize_and_append!(voc::Vocabulary, corpus; isnormalized::Bool=false)
    # Each batch counts into its own table, with no lock, and the tables are then folded
    # together in corpus order. What this replaces is a lock taken per token around
    # `push_token!`, which cost even without contention: measured on 23k Markdown paragraphs,
    # one thread built the vocabulary 4.8x faster without it, and sixteen threads only ran 1.7x
    # faster than one with it.
    #
    # Folding in corpus order also fixes the ids: every token gets the id a sequential scan
    # would give it (order of first appearance), whatever the thread count, where before it got
    # whichever id its thread won the lock with.
    n = length(corpus)
    n == 0 && return voc
    # About two batches per thread, rather than `getminbatch`'s many small ones: every batch
    # repeats the common tokens in its own table, so the fold grows with the number of batches.
    # With word bigrams over 185k documents, 64 batches doubled the build time of 16 or 32.
    nbatches = min(n, 2 * Threads.nthreads())
    batchsize = cld(n, nbatches)
    nbatches = cld(n, batchsize)
    batches = Vector{_VocabularyBatch}(undef, nbatches)
    textconfig = voc.textconfig

    @BATCHES 1 for b in 1:nbatches
        batches[b] = _count_batch(textconfig, corpus, (b-1)*batchsize+1:min(b*batchsize, n); isnormalized)
    end

    batch = _fold_batches!(batches)
    for (i, token) in enumerate(batch.tokens)
        push_token!(voc, token, batch.occs[i], batch.ndocs[i])
    end
    voc.numtokens[] += batch.numtokens

    voc
end

"""
    _absorb!(a::_VocabularyBatch, b::_VocabularyBatch) -> a

Adds `b`'s counts into `a`. Tokens new to `a` are appended in `b`'s order, so when `b` covers
the documents right after `a`'s, the result lists tokens in order of first appearance over both.
"""
function _absorb!(a::_VocabularyBatch, b::_VocabularyBatch)
    for (j, token) in enumerate(b.tokens)
        i = get(a.index, token, zero(Int32))
        if i == 0
            push!(a.tokens, token)
            push!(a.occs, b.occs[j])
            push!(a.ndocs, b.ndocs[j])
            a.index[token] = length(a.tokens)
        else
            a.occs[i] += b.occs[j]
            a.ndocs[i] += b.ndocs[j]
        end
    end
    _VocabularyBatch(a.tokens, a.index, a.occs, a.ndocs, a.numtokens + b.numtokens)
end

"""
    _fold_batches!(batches) -> _VocabularyBatch

Folds consecutive batches into one, pairwise and in parallel: each round absorbs batch `2k`
into batch `2k-1`, halving the list while keeping it in corpus order. Folding them one by one
into the vocabulary instead was the slow half of the build (2.4s of 5.3s with word bigrams over
185k documents and 16 batches), because it is sequential.
"""
function _fold_batches!(batches::Vector{_VocabularyBatch})
    while length(batches) > 1
        m = length(batches)
        folded = Vector{_VocabularyBatch}(undef, cld(m, 2))
        @BATCHES 1 for k in 1:cld(m, 2)
            folded[k] = 2k <= m ? _absorb!(batches[2k-1], batches[2k]) : batches[2k-1]
        end
        batches = folded
    end
    batches[1]
end

Base.length(voc::Vocabulary) = length(voc.occs)
Base.eachindex(voc::Vocabulary) = eachindex(voc.occs)

"""
    vocsize(voc::Vocabulary)

Number of unique tokens in `voc`.
"""
vocsize(voc::Vocabulary) = length(voc)

"""
    gettrainsize(voc::Vocabulary)

Number of documents used to build `voc`.
"""
gettrainsize(voc::Vocabulary) = voc.trainsize[]

"""
    getnumtokens(voc::Vocabulary)

Total number of (non-unique) tokens seen while building `voc`.
"""
getnumtokens(voc::Vocabulary) = voc.numtokens[]

"""
    avgdoclen(voc::Vocabulary)

Average document length in tokens (`getnumtokens(voc) / gettrainsize(voc)`), used by [`BM25Scorer`](@ref).
"""
avgdoclen(voc::Vocabulary) = getnumtokens(voc) / gettrainsize(voc)

"""
    getndocs(voc::Vocabulary, tokenID::Integer)
    getndocs(voc::Vocabulary)

Number of documents containing the token `tokenID` (`0` is out-of-vocabulary and yields
`0` instead of erroring), or the whole per-token vector when called without a `tokenID`.

# Example

```julia
julia> voc = Vocabulary(TextConfig(), ["hello world", "hello there"]; verbose=false);

julia> getndocs(voc, token2id(voc, "hello"))
2
```
"""
getndocs(voc::Vocabulary, tokenID::Integer) = tokenID == 0 ? zero(eltype(voc.ndocs)) : voc.ndocs[tokenID]

"""
    getoccs(voc::Vocabulary, tokenID::Integer)
    getoccs(voc::Vocabulary)

Total occurrences of the token `tokenID` across the corpus (`0` is out-of-vocabulary and
yields `0` instead of erroring), or the whole per-token vector when called without a `tokenID`.

# Example

```julia
julia> voc = Vocabulary(TextConfig(), ["hello world", "hello there"]; verbose=false);

julia> getoccs(voc, token2id(voc, "hello"))
2
```
"""
getoccs(voc::Vocabulary, tokenID::Integer) = tokenID == 0 ? zero(eltype(voc.occs)) : voc.occs[tokenID]

"""
    gettoken(voc::Vocabulary, tokenID::Integer)
    gettoken(voc::Vocabulary)

The token string for `tokenID` (`0` is out-of-vocabulary and yields `""` instead of
erroring), or the whole token vector when called without a `tokenID`.

# Example

```julia
julia> voc = Vocabulary(TextConfig(), ["hello world", "hello there"]; verbose=false);

julia> gettoken(voc, token2id(voc, "hello"))
"hello"
```
"""
gettoken(voc::Vocabulary, tokenID::Integer) = tokenID == 0 ? "" : voc.token[tokenID]

@inline getoccs(voc::Vocabulary) = voc.occs
@inline getndocs(voc::Vocabulary) = voc.ndocs
@inline gettoken(voc::Vocabulary) = voc.token

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
itertokenid(idlist::AbstractMetricQueue) = IdView(idlist)

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
