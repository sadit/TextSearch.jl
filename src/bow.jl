# This file is a part of TextSearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

export compute_bow_multimessage, compute_bow_list, compute_bow

"""
    compute_bow(tokenlist::AbstractVector{S}, bow::BOW=BOW()) where {S<:AbstractString}
    compute_bow(config::TextConfig, text::AbstractString, bow::BOW=BOW())
    compute_bow(config::TextConfig, text::AbstractVector, bow::BOW=BOW())
    

Creates a bag of words from the given text (a string or a list of strings).
If bow is given then updates the bag with the text.
When `config` is given, the text is parsed according to it.
"""
function compute_bow(tokenlist::AbstractVector{S}, bow::BOW=BOW()) where {S<:AbstractString}
    for token in tokenlist
        sym = Symbol(token)
        m = get(bow, sym, 0) + 1
        bow[sym] = m
    end

    bow
end

compute_bow(config::TextConfig, text::AbstractString, bow::BOW=BOW()) = 
    compute_bow(tokenize(config, normalize_text(config, text)), bow)

compute_bow(config::TextConfig, text::AbstractVector, bow::BOW=BOW()) = 
    compute_bow_multimessage(config, text, bow)

"""
    compute_bow_multimessage(config::TextConfig, corpus::AbstractVector, bow=BOW())

Computes a bag of words from a multimessage corpus
"""
function compute_bow_multimessage(config::TextConfig, corpus::AbstractVector, bow=BOW())
    normbuffer = Vector{Char}()
    tokenlist = Vector{String}()
    buff = IOBuffer(Vector{UInt8}(undef, 16), write=true)

    for text in corpus
        empty!(tokenlist)
        if text isa AbstractString
            empty!(normbuffer)
            normalize_text(config, text, normbuffer)
            tokenize(config, normbuffer; output=tokenlist, buff=buff)
        else
            tokenize(config, text; output=tokenlist, buff=buff)
        end
        
        compute_bow(tokenlist, bow)
    end
    
    bow
end

"""
    compute_bow_list(config::TextConfig, corpus::AbstractVector)

Computes a list of bag of words from a corpus
"""
function compute_bow_list(config::TextConfig, corpus::AbstractVector)
    bow_vector = Vector{BOW}(undef, length(corpus))
    t = Char[]
    L = String[]
    buff = IOBuffer(Vector{UInt8}(undef, 16), write=true)

    for (i, text) in enumerate(corpus)
        empty!(t)
        empty!(L)
        normalize_text(config, text, t)
        tokenize(config, t; output=L, buff=buff)
        bow_vector[i] = compute_bow(tokenize(config, t))
    end

    bow_vector
end
