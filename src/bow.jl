# This file is a part of TextSearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

export compute_bow_multimessage, compute_bow_list, compute_bow

"""
    compute_bow(tokenlist::AbstractVector{S}, bow::BOW=BOW()) where {S<:AbstractString}
    compute_bow(config::TextConfig, text::AbstractString, bow::BOW=BOW())
    
Updates a DVEC using the given list of tokens; returns `bow`
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
