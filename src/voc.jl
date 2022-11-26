# This file is a part of TextSearch.jl

export Vocabulary, occs, ndocs, token, vocsize, trainsize, filter_tokens, tokenize_and_append!, merge_voc, update_voc!

struct Vocabulary
    token::Vector{String}
    occs::Vector{Int32}
    ndocs::Vector{Int32}
    token2id::Dict{String,UInt32}
    corpuslen::Int
end

"""
    Vocabulary(n::Integer)

Creates a `Vocabulary` struct
"""
function Vocabulary(n::Integer) 
    # n == 0 means unknown
    voc = Vocabulary(String[], Int32[], Int32[], Dict{String,UInt32}(), n)
    vocsize = ceil(Int, n^0.6)  # approx based on Heaps law
    sizehint!(voc.token, vocsize)
    sizehint!(voc.occs, vocsize)
    sizehint!(voc.ndocs, vocsize)
    sizehint!(voc.token2id, vocsize)
    voc
end

function locked_tokenize_and_push(voc, textconfig, doc, buff, l; ignore_new_tokens=false)
    empty!(buff)
    id = 0

    for token in tokenize(identity, textconfig, doc, buff)
        lock(l)
        try
            id = push_token!(voc, token, 1, 0; ignore_new_tokens)
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
    voc = Vocabulary(sum(v.corpuslen for v in L))

    for v in L
        update_voc!(pred, voc, v)
    end

    voc
end

"""
    Vocabulary(textconfig, corpus; minbatch=0, thesaurus=nothing)

Computes a vocabulary from a corpus using the TextConfig `textconfig`.
If thesaurus is a collection of strings, then it is used as a fixed list of tokens.
"""
function Vocabulary(textconfig::TextConfig, corpus::AbstractVector; minbatch=0, thesaurus=nothing)
    voc = Vocabulary(length(corpus))
    if thesaurus === nothing
        tokenize_and_append!(voc, textconfig, corpus; minbatch)
    else
        for v in thesaurus
            push_token!(voc, v)
        end

        tokenize_and_append!(voc, textconfig, corpus; minbatch, ignore_new_tokens=true)
    end
end

"""
    tokenize_and_append!(voc::Vocabulary, textconfig::TextConfig, corpus; minbatch=0, ignore_new_tokens=false)

Parse each document in the given corpus and appends each token to the vocabulary.
"""
function tokenize_and_append!(voc::Vocabulary, textconfig::TextConfig, corpus; minbatch=0, ignore_new_tokens=false)
    l = Threads.SpinLock()
    n = length(corpus)
    minbatch = getminbatch(minbatch, n)

    Threads.@threads for i in 1:n
        doc = corpus[i]
        
        buff = take!(TEXT_SEARCH_CACHES)
        try
            if doc isa AbstractString
                locked_tokenize_and_push(voc, textconfig, doc, buff, l; ignore_new_tokens)
            else
                for text in doc
                    locked_tokenize_and_push(voc, textconfig, text, buff, l; ignore_new_tokens)
                end
            end
        finally
            put!(TEXT_SEARCH_CACHES, buff)
        end
    end

    voc
end

"""
    Vocabulary(corpus)

Computes a vocabulary from an already tokenized corpus
"""
function Vocabulary(corpus)
    voc = Vocabulary(length(corpus))
    bow = Set{Int32}()
    for tokens in corpus
        empty!(bow)
        for token in tokens
            id = push_token!(voc, token, 1, 0)
            push!(bow, id)
        end

        for id in bow
            voc.ndocs[id] += 1
        end
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

function push_token!(voc::Vocabulary, token::String, occs::Integer, ndocs::Integer; ignore_new_tokens::Bool=false)
    id = get(voc.token2id, token, zero(UInt32))

    if id == 0
        if !ignore_new_tokens
            id = length(voc) + 1
            push!(voc.token, token)
            push!(voc.occs, occs)
            push!(voc.ndocs, ndocs)
            voc.token2id[token] = id
        end
    else
        voc.occs[id] += occs
        voc.ndocs[id] += ndocs
    end

    id
end

function push_token!(voc::Vocabulary, token::String; occs::Integer=0, ndocs::Integer=0, ignore_new_tokens=false)
    push_token!(voc, token, occs, ndocs; ignore_new_tokens)
end

Base.get(voc::Vocabulary, token::String, default::Integer)::UInt32 = get(voc.token2id, token, default)

function Base.getindex(voc::Vocabulary, token::String)
    getindex(voc, get(voc, token, 0))
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
    V = Vocabulary(voc.corpuslen)

    for i in eachindex(voc)
        v = voc[i]
        if pred(v)
            push_token!(V, v.token, v.occs, v.ndocs)
        end
    end

    V
end