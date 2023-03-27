# This file is a part of TextSearch.jl

export Vocabulary, occs, ndocs, token, vocsize, trainsize, filter_tokens, tokenize_and_append!, merge_voc, update_voc!, vocabulary_from_thesaurus, token2id, 
       encode, decode


struct Vocabulary
    textconfig::TextConfig
    token::Vector{String}
    occs::Vector{Int32}
    ndocs::Vector{Int32}
    token2id::Dict{String,UInt32}
    corpuslen::Int
end

token2id(voc::Vocabulary, tok::AbstractString) = get(voc.token2id, tok, zero(UInt32))

function decode(voc::Vocabulary, bow::Dict)
    Dict(voc.token[k] => v for (k, v) in bow)
end

function encode(voc::Vocabulary, bow::Dict)
    Dict(token2id(voc, k) => v for (k, v) in bow)
end

function vocabulary_from_thesaurus(textconfig::TextConfig, tokens::AbstractVector)
    n = length(tokens)
    token2id = Dict{String,UInt32}
    voc = Vocabulary(textconfig, n)
    for t in tokens
        push_token!(voc, t, occs(t), ndocs(t))
    end

    voc
end

"""
    Vocabulary(textconfig::TextConfig, n::Integer)

Creates a `Vocabulary` struct
"""
function Vocabulary(textconfig::TextConfig, n::Integer)
    # n == 0 means unknown
    voc = Vocabulary(textconfig, String[], Int32[], Int32[], Dict{String,UInt32}(), n)
    vocsize = ceil(Int, n^0.6)  # approx based on Heaps law
    sizehint!(voc.token, vocsize)
    sizehint!(voc.occs, vocsize)
    sizehint!(voc.ndocs, vocsize)
    sizehint!(voc.token2id, vocsize)
    voc 
end

"""
    Vocabulary(textconfig, corpus; minbatch=0)

Computes a vocabulary from a corpus using the TextConfig `textconfig`.
"""
function Vocabulary(textconfig::TextConfig, corpus::AbstractVector; minbatch=0)
    voc = Vocabulary(textconfig, length(corpus))
    tokenize_and_append!(voc, textconfig, corpus; minbatch)
    voc
end

function locked_tokenize_and_push(voc, textconfig, doc, buff, l)
    empty!(buff)

    for token in tokenize(borrowtokenizedtext, textconfig, doc, buff)
        id = 0
        lock(l)
        try
            id = push_token!(voc, token, 1, 0)
        finally
            unlock(l)
            buff.bow[id] = 1
        end
    end

    lock(l)
    try
        for id in keys(buff.bow)
            voc.ndocs[id] += 1
        end
    finally
        unlock(l)
    end
end

"""
    tokenize_and_append!(voc::Vocabulary, textconfig::TextConfig, corpus; minbatch=0)

Parse each document in the given corpus and appends each token to the vocabulary.
"""
function tokenize_and_append!(voc::Vocabulary, textconfig::TextConfig, corpus; minbatch=0)
    l = Threads.SpinLock()
    n = length(corpus)
    minbatch = getminbatch(minbatch, n)

    @batch per=thread minbatch=minbatch for i in 1:n
        doc = corpus[i]

        buff = take!(TEXT_SEARCH_CACHES)

        try
            if doc isa AbstractVector
                for text in doc
                    locked_tokenize_and_push(voc, textconfig, text, buff, l)
                end
            else # if doc isa AbstractString
                locked_tokenize_and_push(voc, textconfig, doc, buff, l)
            end
        finally
            put!(TEXT_SEARCH_CACHES, buff)
        end
    end

    voc
end

function filter_tokens!(voc::Vocabulary, text::TokenizedText)
    j = 0
    for i in eachindex(text.tokens)
        t = text.tokens[i]
        if haskey(voc.token2id, t)
            j += 1
            text.tokens[j] = t
        end
    end

    resize!(text.tokens, j)
    text
end

function filter_tokens!(voc::Vocabulary, arr::AbstractVector{TokenizedText})
    for t in arr
        filter_tokens!(voc, t)
    end

    arr
end

"""
    update_voc!(voc::Vocabulary, another::Vocabulary)
    update_voc!(pred::Function, voc::Vocabulary, another::Vocabulary)

Update `voc` vocabulary using another one.
Optionally a predicate can be given to filter vocabularies.

Note 1: `corpuslen` remains unchanged (the structure is immutable and a new `Vocabulary` should be created to update this field).
Note 2: Both `voc` and `another` vocabularies should had been created with a _compatible_ [`Textconfig`](@ref) to be able to work on them.
"""
update_voc!(voc::Vocabulary, another::Vocabulary) = update_voc!(t->true, voc, another)

function update_voc!(pred::Function, voc::Vocabulary, another::Vocabulary)
    for i in eachindex(another)
        v = another[i]
        if pred(v)
            push_token!(voc, v.token, v.occs, v.ndocs)
        end
    end

    voc
end

"""
    merge_voc(voc1::Vocabulary, voc2::Vocabulary[, ...])
    merge_voc(pred::Function, voc1::Vocabulary, voc2::Vocabulary[, ...])

Merges two or more vocabularies into a new one. A predicate function can be used to filter token entries.

Note: All vocabularies should had been created with a _compatible_ [`Textconfig`](@ref) to be able to work on them.
"""
merge_voc(voc1::Vocabulary, voc2::Vocabulary, voclist...) = merge_voc(x->true, voc1, voc2, voclist...)

function merge_voc(pred::Function, voc1::Vocabulary, voc2::Vocabulary, voclist...)
    #all(v -> v isa Vocabulary, voclist) || throw(ArgumentError("arguments should be of type `Vocabulary`"))
    
    L = [voc1, voc2]
    for v in voclist
        push!(L, v)
    end

    sort!(L, by=vocsize, rev=true)
    voc = Vocabulary(voc1.textconfig, sum(v.corpuslen for v in L))

    for v in L
        update_voc!(pred, voc, v)
    end

    voc
end

Base.length(voc::Vocabulary) = length(voc.occs)
Base.eachindex(voc::Vocabulary) = eachindex(voc.occs)
vocsize(voc::Vocabulary) = length(voc)
trainsize(voc::Vocabulary) = voc.corpuslen
ndocs(voc::Vocabulary, tokenID::Integer) = tokenID == 0 ? zero(eltype(voc.ndocs)) : voc.ndocs[tokenID]
occs(voc::Vocabulary, tokenID::Integer) = tokenID == 0 ? zero(eltype(voc.occs)) : voc.occs[tokenID]
token(voc::Vocabulary, tokenID::Integer) = tokenID == 0 ? "" : voc.token[tokenID]

@inline occs(voc::Vocabulary) = voc.occs
@inline ndocs(voc::Vocabulary) = voc.ndocs
@inline token(voc::Vocabulary) = voc.token

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
itertokenid(idlist::KnnResult) = IdView(idlist)

function Base.getindex(voc::Vocabulary, idlist)
    [voc[i] for i in itertokenid(idlist)]
end

function Base.getindex(voc::Vocabulary, tokenID::Integer)
    id = convert(UInt32, tokenID)

    if id == 0
        (; id=id, occs=zero(eltype(voc.occs)), ndocs=zero(eltype(voc.ndocs)), token="")
    else
        (; id=id, occs=voc.occs[id], ndocs=voc.ndocs[id], token=voc.token[id])
    end
end


"""
    filter_tokens(pred::Function, voc::Vocabulary)

Returns a copy of reduced vocabulary based on evaluating `pred` function for each entry in `voc`
"""
function filter_tokens(pred::Function, voc::Vocabulary)
    V = Vocabulary(voc.textconfig, voc.corpuslen)

    for i in eachindex(voc)
        v = voc[i]
        if pred(v)
            push_token!(V, v.token, v.occs, v.ndocs)
        end
    end

    V
end
