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

function bagofwords_(copy_::Function, voc::Vocabulary, text)
    buff = take!(TEXT_SEARCH_CACHES)
    empty!(buff)
    try
        copy_(bagofwords!(buff, voc, text).bow)
    finally
        put!(TEXT_SEARCH_CACHES, buff)
    end
end

"""
    bagofwords!(buff::TextSearchBuffer, voc::Vocabulary, messages)

Computes a bag of words from a multi-field document (a list of texts), storing the
result in `buff.bow`. See [`bagofwords`](@ref) for the non-mutating version.

# Example

```julia
julia> voc = Vocabulary(TextConfig(), ["hello world"]; verbose=false);

julia> buff = TextSearch.TextSearchBuffer();

julia> TextSearch.bagofwords!(buff, voc, "hello hello world");

julia> buff.bow
Dict{UInt32, Int32}(0x00000002 => 1, 0x00000001 => 2)
```
"""
function bagofwords!(buff::TextSearchBuffer, voc::Vocabulary, messages)
    empty!(buff.bow)
    for text in messages
        empty!(buff.tok)
        tokens = tokenize(borrowtokenizedtext, voc.textconfig, text, buff.tok)
        bagofwords!(buff.bow, voc, tokens)
    end

    buff
end

function bagofwords!(buff::TextSearchBuffer, voc::Vocabulary, text::AbstractString)
    tokens = tokenize(borrowtokenizedtext, voc.textconfig, text, buff.tok)
    bagofwords!(buff.bow, voc, tokens)
    buff
end

function bagofwords!(buff::TextSearchBuffer, voc::Vocabulary, tokens::TokenizedText)
    bagofwords!(buff.bow, voc, tokens)
    buff
end

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
bagofwords(voc::Vocabulary, messages) = bagofwords_(copy, voc, messages)
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
    X = [bagofwords(voc, corpus[1])]
    resize!(X, n)
    minbatch = getminbatch(minbatch, n)

    #@batch minbatch=minbatch per=thread 
    @showprogress dt=1 enabled=verbose desc="Bag of words" Threads.@threads for i in 2:n
        X[i] = bagofwords(voc, corpus[i])
    end

    X
end

