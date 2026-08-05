# This file is a part of TextSearch.jl

export bagofwords_corpus, bagofwords

"""
    bagofwords!(bow::BOW, voc::Vocabulary, tokenlist::TokenizedText)

Accumulates the tokens in `tokenlist` into `bow` (a [`BOW`](@ref)), looking up each
token's id in `voc`; out-of-vocabulary tokens are skipped. Returns `bow`.

# Example

```julia
julia> voc = Vocabulary(TextConfig(), ["hello world"]; verbose=false);

julia> TextSearch.bagofwords!(BOW(), voc, tokenize(TextConfig(), "hello hello"))
Dict{UInt32, Int32}(0x00000001 => 2)
```
"""
function bagofwords!(bow::BOW, voc::Vocabulary, tokenlist::TokenizedText)
    for token in tokenlist
        tokenID = token2id(voc, token)
        if zero(UInt32) != tokenID
            bow[tokenID] = get(bow, tokenID, zero(Int32)) + one(Int32)
        end
    end

    bow
end

"""
    bagofwords!(bow::BOW, voc::Vocabulary, messages)

Computes a bag of words from a multi-field document (a list of texts), accumulating the
result into `bow`. See [`bagofwords`](@ref) for the non-mutating version.

# Example

```julia
julia> voc = Vocabulary(TextConfig(), ["hello world"]; verbose=false);

julia> bow = BOW();

julia> TextSearch.bagofwords!(bow, voc, "hello hello world");

julia> bow
Dict{UInt32, Int32}(0x00000002 => 1, 0x00000001 => 2)
```
"""
function bagofwords!(bow::BOW, voc::Vocabulary, messages)
    for text in messages
        tokenizerbuffer() do tok
            tokens = tokenize(borrowtokenizedtext, voc.textconfig, text, tok)
            bagofwords!(bow, voc, tokens)
        end
    end

    bow
end

function bagofwords!(bow::BOW, voc::Vocabulary, text::AbstractString)
    tokenizerbuffer() do tok
        tokens = tokenize(borrowtokenizedtext, voc.textconfig, text, tok)
        bagofwords!(bow, voc, tokens)
    end
    bow
end

"""
    _bow_sizehint(voc::Vocabulary)

Estimated final size (number of unique tokens) of a [`BOW`](@ref) computed under `voc`,
used to `sizehint!` it up front and avoid rehashing while it's filled. Uses `voc`'s own
[`avgdoclen`](@ref) (average tokens per document across its training corpus) as the
estimate, falling back to a small default before `voc` has seen any training documents.
"""
_bow_sizehint(voc::Vocabulary) = trainsize(voc) > 0 ? ceil(Int, avgdoclen(voc)) : 16

"""
    bagofwords(voc::Vocabulary, messages)

Tokenizes `messages` (a string or a list of strings) under `voc`'s [`TextConfig`](@ref)
and returns its bag of words ([`BOW`](@ref)): a `token id => occurrence count` mapping.
An already-computed [`BOW`](@ref) is returned unchanged.

# Example

```julia
julia> voc = Vocabulary(TextConfig(), ["hello world"]; verbose=false);

julia> bagofwords(voc, "hello hello world")
Dict{UInt32, Int32}(0x00000002 => 1, 0x00000001 => 2)
```
"""
function bagofwords(voc::Vocabulary, messages)
    bow = BOW()
    sizehint!(bow, _bow_sizehint(voc))
    bagofwords!(bow, voc, messages)
end
bagofwords(voc::Vocabulary, messages::BOW) = messages

"""
    bagofwords_corpus(voc::Vocabulary, corpus::AbstractVector; minbatch=0, verbose=true)

Computes a list of bag of words ([`BOW`](@ref)s) from a corpus, one per document,
in parallel across threads.

# Example

```julia
julia> voc = Vocabulary(TextConfig(), ["hello world", "hello there"]; verbose=false);

julia> bagofwords_corpus(voc, ["hello world", "hello there"]; verbose=false)[1]
Dict{UInt32, Int32}(0x00000002 => 1, 0x00000001 => 1)
```
"""
bagofwords_corpus(voc::Vocabulary, corpus::AbstractVector{BOW}; minbatch=0, verbose=true) = corpus
function bagofwords_corpus(voc::Vocabulary, corpus::AbstractVector; minbatch=0, verbose=true)
    n = length(corpus)
    bowsize = _bow_sizehint(voc)
    X = Vector{BOW}(undef, n)
    minbatch = minbatch > 0 ? minbatch : getminbatch(n)
    prog = Progress(n; dt=1, enabled=verbose, desc="Bag of words")

    @BATCHES minbatch for i in 1:n
        doc = corpus[i]
        if doc isa BOW
            X[i] = doc
        else
            bow = BOW()
            sizehint!(bow, bowsize)
            X[i] = bagofwords!(bow, voc, doc)
        end
        next!(prog)
    end

    X
end

