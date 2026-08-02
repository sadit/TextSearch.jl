# This file is a part of TextSearch.jl
export BM25Scorer, bm25score, tokenscore

"""
    BM25Scorer

Precomputed coefficients for the Okapi BM25 (BM25+) scoring function, used to rank
documents against a query given per-token document frequencies and document lengths.
Build one with [`BM25Scorer(voc)`](@ref BM25Scorer) or
[`BM25Scorer(trainsize, avgdoclen)`](@ref BM25Scorer); score individual
(token-frequency, document-length) pairs with [`tokenscore`](@ref), or whole
query/document bags of words with [`bm25score`](@ref). [`BM25InvertedFile`](@ref)
uses a `BM25Scorer` internally to answer top-k queries efficiently.

# Fields
`k1_plus_1`, `k1_mult_1_min_b`, and `k1_mult_b_div_avg_doc_len` are combinations of the
BM25 `k1`/`b` hyperparameters and the corpus' average document length, precomputed for
faster scoring; `δ` is the BM25+ lower-bound correction term; `trainsize` is the number
of documents the corpus statistics were computed from.

# Example

```julia
julia> corpus = ["hello world", "hello there", "the cat sat"];

julia> voc = Vocabulary(TextConfig(), corpus; verbose=false);

julia> bm25 = BM25Scorer(voc);

julia> bm25.trainsize
3
```
"""
struct BM25Scorer
    #k1::Float32
    #b::Float32
    k1_plus_1::Float32
    k1_mult_1_min_b::Float32
    k1_mult_b_div_avg_doc_len::Float32
    δ::Float32
    trainsize::Int32
end


function Base.show(io::IO, bm::BM25Scorer; prefix="", indent="  ")
    println(io, prefix, "BM25Scorer:")
    prefix = indent * prefix
    k1 = bm.k1_plus_1 - 1
    b = - (bm.k1_mult_1_min_b / k1 - 1)
    avgdoclen = 1 / (bm.k1_mult_b_div_avg_doc_len / b / k1)
    println(io, prefix, "k1: ", k1)
    println(io, prefix, "b: ", b)
    println(io, prefix, "avgdoclen: ", avgdoclen)
    println(io, prefix, "δ: ", bm.δ)
    println(io, prefix, "trainsize: ", bm.trainsize)
end


"""
    BM25Scorer(trainsize::Integer, avgdoclen::AbstractFloat; k1=1.2f0, b=0.75f0, δ=1f0)

Creates a [`BM25Scorer`](@ref) for a corpus of `trainsize` documents with average
document length `avgdoclen`. `k1` controls term-frequency saturation, `b` controls
document-length normalization, and `δ` is the BM25+ lower-bound correction.

# Example

```julia
julia> bm25 = BM25Scorer(3, 3.5);

julia> bm25.trainsize
3
```
"""
function BM25Scorer(trainsize::Integer, avgdoclen::AbstractFloat; k1=1.2f0, b=0.75f0, δ=1f0)
    BM25Scorer( #k1, b,
        convert(Float32, k1 + 1f0),
        convert(Float32, k1 * (1f0 - b)),
        convert(Float32, k1 * b / avgdoclen),
        δ,
        convert(Int32, trainsize)
    )
end

"""
    BM25Scorer(voc::Vocabulary; k1=1.2f0, b=0.75f0, δ=1f0)

Creates a [`BM25Scorer`](@ref) using the training size and average document length
already computed in `voc` (see [`trainsize`](@ref) and [`avgdoclen`](@ref)).

# Example

```julia
julia> corpus = ["hello world", "hello there", "the cat sat"];

julia> voc = Vocabulary(TextConfig(), corpus; verbose=false);

julia> BM25Scorer(voc).trainsize
3
```
"""
BM25Scorer(voc::Vocabulary; k1=1.2f0, b=0.75f0, δ=1f0) = BM25Scorer(trainsize(voc), avgdoclen(voc); k1, b, δ)

"""
    bm25score(bm25::BM25Scorer, voc::Vocabulary, query::Dict, doc::Dict)::Float32

Computes the BM25 relevance score of `doc` (a bag of words) for `query` (a bag of words),
summing [`tokenscore`](@ref) over every query token present in `doc`. Higher is more relevant.

# Example

```julia
julia> corpus = ["hello world", "hello there", "the cat sat"];

julia> voc = Vocabulary(TextConfig(), corpus; verbose=false);

julia> bm25 = BM25Scorer(voc);

julia> bm25score(bm25, voc, bagofwords(voc, "hello"), bagofwords(voc, "hello world"))
0.96917987f0
```
"""
function bm25score(bm25::BM25Scorer, voc::Vocabulary, query::Dict, doc::Dict)::Float32
    s = 0f0

    doclen = sum(f for f in values(doc))
    for tokenID in keys(query)
        w = get(doc, tokenID, 0f0)
        if w > 0f0
            s += tokenscore(bm25, ndocs(voc, tokenID), doclen, w)
        end
    end

    s
end

"""
    tokenscore(bm25::BM25Scorer, toknumdocs, doclen, tokfreqindoc)

Computes the BM25 contribution of a single token to a document's score, given the
number of documents containing the token (`toknumdocs`, i.e. its document frequency),
the document's length in tokens (`doclen`), and the token's frequency in the document
(`tokfreqindoc`). Used internally by [`bm25score`](@ref) and [`BM25InvertedFile`](@ref) search.

# Example

```julia
julia> corpus = ["hello world", "hello there", "the cat sat"];

julia> voc = Vocabulary(TextConfig(), corpus; verbose=false);

julia> tokenscore(BM25Scorer(voc), 2, 3, 1)
0.8908208f0
```
"""
function tokenscore(bm25::BM25Scorer, toknumdocs, doclen, tokfreqindoc)
    idf = log(1f0 + (bm25.trainsize - toknumdocs + 0.5f0) / (toknumdocs + 0.5f0))
    num = tokfreqindoc * bm25.k1_plus_1
    den = tokfreqindoc + bm25.k1_mult_1_min_b + doclen * bm25.k1_mult_b_div_avg_doc_len
    convert(Float32, idf * (num / den + bm25.δ))
end
