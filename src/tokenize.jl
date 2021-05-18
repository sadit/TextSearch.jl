# This file is a part of TextSearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

export Tokenizer, tokenize, qgrams, unigrams, encode, decode

"""
    struct Tokenizer

A tokenizer converts a text into a set of tokens, and in particular, each token
in this implementation is represented as the hash code of the corresponding string token.
This methods also normalize and preprocess the text following instructions in the given `TextConfig` object.
The structure has several fields:
- the text config object
- an inverse map `invmap` that need to be captured at construction time (only needed if the tokens need to be inspected).
- `isconstruction` indicator (must be true only when the tokens are being parsed for building a model)
- the rest of the fields are used as buffers (multithreaded applications need independent copies of tokenizers)

"""
struct Tokenizer
    config::TextConfig
    invmap::Union{Dict{UInt64,String},Nothing}
    isconstruction::Bool
    normtext::Vector{Char}
    tokens::Vector{UInt64}
    unigrams::Vector{String}
    io::IOBuffer
end

StructTypes.StructType(::Type{<:Tokenizer}) = StructTypes.DictType()

function StructTypes.construct(::Type{<:Tokenizer}, d::Dict)
    c = d[:config]
    invmap = d[:invmap]
    c["slist"] = Skipgram[Skipgram(s["qsize"], s["skip"]) for s in c["slist"]]
    config = TextConfig(; (Symbol(k) => v for (k,v) in c)...)
    if invmap !== nothing
        invmap = Dict(parse(UInt64, k) => v for (k, v) in invmap)
    end
    Tokenizer(config, isconstruction=false, invmap=invmap)
end

StructTypes.keyvaluepairs(tok::Tokenizer) = [:config => tok.config, :invmap => tok.invmap]

function Tokenizer(
        config::TextConfig;
        isconstruction=true,
        invmap=Dict{UInt64,String}(),
        n=128
    )
    normtext = Vector{Char}(undef, n)
    tokens = Vector{UInt64}(undef, n)
    unigrams = Vector{String}(undef, n)
    resize!(normtext, 0)
    resize!(tokens, 0)
    resize!(unigrams, 0)

    Tokenizer(config, invmap, isconstruction, normtext, tokens, unigrams, IOBuffer())
end

function Tokenizer(tok::Tokenizer; isconstruction=false, n=128)
    Tokenizer(tok.config, invmap=tok.invmap, isconstruction=isconstruction, n=n)
end

function Base.show(io::IO, tok::Tokenizer) 
    print(io, "{Tokenizer isconstruction=$(tok.isconstruction) ")
    if tok.invmap === nothing
        print(io, "invmap=nothing")
    else
        print(io, "invmap=", length(tok.invmap))
    end
    print(io, " config=$(tok.config)\n}")
end

Base.broadcastable(m::Tokenizer) = (m,)

function encode(tok::Tokenizer, token::AbstractString)
    h = hash(token)
    tok.isconstruction && tok.invmap !== nothing && (tok.invmap[h] = token)
    h
end

function decode(tok::Tokenizer, id::UInt64)
    tok.invmap === nothing ? nothing : tok.invmap[id]
end

function decode(tok::Tokenizer, vec::Dict{UInt64,S}) where S
    tok.invmap === nothing ? nothing : Dict(tok.invmap[k] => v for (k,v) in vec)
end

function Base.empty!(tok::Tokenizer)
    empty!(tok.normtext)
    empty!(tok.tokens)
    empty!(tok.unigrams)
end

#=function tokenize__(config::TextConfig, textlist::AbstractVector, tok::Tokenizer=Tokenizer())
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
end=#

"""
    tokenize(tok::Tokenizer, text::AbstractString)

Tokenizes `text` using the given configuration
"""
function tokenize(tok::Tokenizer, text::AbstractString)
    #@info text
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
        push!(tok.tokens, encode(tok, s))
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
        write(tok.io, '\t', 'q')
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
    # @info tok.normtext
    @inbounds for i in 2:n  # normtext[1] is BLANK
        c = tok.normtext[i]
        p = tok.normtext[i-1]

        ## @show i, p, c
        if ispunct(c) && !ispunct(p) && p !== BLANK
            ## @show :a
            s = flush_token!(tok)
            s !== nothing && push!(tok.unigrams, s)
            write(tok.io, c)
        elseif ispunct(p) && !ispunct(c) && !(p in ('#', '@', '_'))
            ## @show :b
            s = flush_token!(tok)
            s !== nothing && push!(tok.unigrams, s)
            if c !== BLANK
                write(tok.io, c)
            end
        elseif c == BLANK
            ## @show :c
            if p !== BLANK
                s = flush_token!(tok)
                s !== nothing && push!(tok.unigrams, s)
            end
        else
            ## @show :d
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
        write(tok.io, '\t', 'n')
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
        write(tok.io, '\t', 's')
        if q.qsize == 2
            write(tok.io, tok.unigrams[start])
            write(tok.io, BLANK)
            write(tok.io, tok.unigrams[start + 1 + q.skip])
        else
            ep = q.qsize - 2
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
