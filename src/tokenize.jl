# This file is a part of TextSearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

export Tokenizer, tokenize, qgrams, unigrams

struct Tokenizer{TMap<:TokenMap}
    config::TextConfig
    vocmap::TMap
    normtext::Vector{Char}
    tokens::Vector{UInt64}
    unigrams::Vector{String}
    io::IOBuffer
end

StructTypes.StructType(::Type{<:Tokenizer}) = StructTypes.Struct()

function Tokenizer(config::TextConfig, vocmap=TokenHash(true); n=128) 
    normtext = Vector{Char}(undef, n)
    tokens = Vector{UInt64}(undef, n)
    unigrams = Vector{String}(undef, n)
    resize!(normtext, 0)
    resize!(tokens, 0)
    resize!(unigrams, 0)

    Tokenizer(config, vocmap, normtext, tokens, unigrams, IOBuffer())
end

Base.broadcastable(m::Tokenizer) = (m,)

decode(tok::Tokenizer, id::UInt64) = decode(tok.vocmap, id)

function Base.empty!(tok::Tokenizer)
    empty!(tok.normtext)
    empty!(tok.tokens)
    empty!(tok.unigrams)
end

function tokenize__(config::TextConfig, textlist::AbstractVector, tok::Tokenizer=Tokenizer())
    n = length(textlist) * length(first(textlist))

    for text in textlist
        empty!(tok)
        if text isa AbstractString
            normalize_text(config, text, tok.normtext)
            tokenize_(tok)
        else
            tokenize(tok, text)
        end
    end

    tok.tokens
end

"""
    tokenize(tok::Tokenizer, text::AbstractString)

Tokenizes `text` using the given configuration
"""
function tokenize(tok::Tokenizer, text::AbstractString)
    empty!(tok)
    normalize_text(tok.config, text, tok.normtext)
    tokenize_(tok)
end

function tokenize_(tok::Tokenizer)
    config = tok.config
    for q in config.qlist
        qgrams(tok, q)
    end
    
    if length(config.nlist) > 0 || length(config.slist) > 0
        n1 = length(tok.tokens)
        unigrams(tok)  # unigrams are always activated if any |nlist| > 0 or |slist| > 0

        if length(config.nlist) == 0 || config.nlist[1] != 1 # always sorted
            resize!(tok.tokens, n1)
        end

        for q in config.nlist 
            q != 1 && nwords(tok, q)
        end

        for q in config.slist
            skipgrams(tok, q)
        end
    end

    tok.tokens
end

"""
    flush_token!(tok::Tokenizer)

Pushes the word inside buffer into token list; it discards empty strings.
"""
function flush_token!(tok::Tokenizer)
    io = tok.io
    if io.size > 0
        s = String(take!(io))
        push!(tok.tokens, encode(tok.vocmap, s))
        s
    else
        nothing
    end
end

"""
    qgrams(tok::Tokenizer, q::Integer)

Computes character q-grams for the given input
"""
function qgrams(tok::Tokenizer, q::Integer)
    n = length(tok.normtext)

    for i in 1:(n - q + 1)
        for j in i:i+q-1
            @inbounds write(tok.io, tok.normtext[j])
        end

        flush_token!(tok)
    end

    tok.tokens
end


"""
    unigrams(tok::Tokenizer, dropunigrams::Bool)

Performs the word tokenization
"""
function unigrams(tok::Tokenizer)
    n = length(tok.normtext)
    @inbounds for i in 1:n
        c = tok.normtext[i]

        if c == BLANK
            s = flush_token!(tok)
            s !== nothing && push!(tok.unigrams, s)
        else
            write(tok.io, c)
        end
    end
    s = flush_token!(tok)
    s !== nothing && push!(tok.unigrams, s)
    tok.tokens
end

"""
    nwords(tok::Tokenizer, q::Integer)
"""
function nwords(tok::Tokenizer, q::Integer)
    n = length(tok.unigrams)

    @inbounds for i in 1:(n - q + 1)
        _last = i + q - 1
        for j in i:_last-1
            write(tok.io, tok.unigrams[j])
            write(tok.io, BLANK)
        end

        write(tok.io, tok.unigrams[_last])
        flush_token!(tok)
    end

    tok.tokens
end

"""
    skipgrams(tok::Tokenizer, q::Skipgram)

Tokenizes using skipgrams
"""
function skipgrams(tok::Tokenizer, q::Skipgram)
    n = length(tok.unigrams)

    for start in 1:(n - (q.qsize + (q.qsize - 1) * q.skip) + 1)
        if q.qsize == 2
            write(tok.io, tok.unigrams[start])
            write(tok.io, BLANK)
            write(tok.io, tok.unigrams[start + 1 + q.skip])
        else
            ep = q.qsize-2
            for i in 0:ep
                write(tok.io, tok.unigrams[start + i * (1+q.skip)])
                write(tok.io, BLANK)
            end
            ep += 1
            write(tok.io, tok.unigrams[start + ep * (1+q.skip)])
        end

        flush_token!(tok)
    end

    tok.tokens
end
