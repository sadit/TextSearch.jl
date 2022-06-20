# This file is a part of TextSearch.jl

export tokenize, tokenize_corpus, qgrams, unigrams, encode, decode

const EXTRA_PUNCT = Set(['~', '+', '^', '$', '|', '<', '>'])

Base.broadcastable(m::TextConfig) = (m,)

"""
    tokenize_corpus(textconfig::TextConfig, arr; minbatch=0)

Tokenize a list of texts.
"""
function tokenize_corpus(textconfig::TextConfig, arr; minbatch=0)
    n = length(arr)
    L = Vector{Vector{String}}(undef, n)
    minbatch = getminbatch(minbatch, n)
    
    @batch minbatch=minbatch per=thread for i in 1:n
        buff = take!(CACHES)
        empty!(buff)
        try
            L[i] = copy(tokenize(textconfig, arr[i]))
        finally
            put!(CACHES, buff)
        end
    end

    L
end

"""
    tokenize(textconfig::TextConfig, text::AbstractString, buff=TextSearchBuffer())

Tokenizes `text` using the given configuration
"""
function tokenize(textconfig::TextConfig, text::AbstractString, buff=TextSearchBuffer())
    normalize_text(textconfig, text, buff.normtext)
    tokenize_(textconfig, buff)
end

function tokenize_(config::TextConfig, buff::TextSearchBuffer)
    for q in config.qlist
        qgrams(q, buff)
    end
    
    if length(config.nlist) > 0 || length(config.slist) > 0
        n1 = length(buff.tokens)
        unigrams(buff)  # unigrams are always activated if any |nlist| > 0 or |slist| > 0

        if length(config.nlist) == 0 || config.nlist[1] != 1 # always sorted
            resize!(buff.tokens, n1)
        end

        for q in config.nlist 
            q != 1 && nwords(q, buff)
        end

        for q in config.slist
            skipgrams(q, buff)
        end
    end

    buff.tokens
end

"""
    flush_token!(buff::TextSearchBuffer)

Pushes the word inside the buffer to the token list; it discards empty strings.
"""
function flush_token!(buff::TextSearchBuffer)
    io = buff.io
    if io.size > 0
        s = String(take!(io))
        push!(buff.tokens, s)
        s
    else
        nothing
    end
end

"""
    qgrams(q::Integer, buff::TextSearchBuffer)

Computes character q-grams for the given input
"""
function qgrams(q::Integer, buff::TextSearchBuffer)
    n = length(buff.normtext)

    for i in 1:(n - q + 1)
        write(buff.io, '\t', 'q')
        for j in i:i+q-1
            @inbounds write(buff.io, buff.normtext[j])
        end
        flush_token!(buff)
    end

    buff.tokens
end

ispunct2(c) = ispunct(c) || c in EXTRA_PUNCT

"""
    unigrams(buff::TextSearchBuffer)

Performs the word tokenization
"""
function unigrams(buff::TextSearchBuffer)
    n = length(buff.normtext)
    # @info buff.normtext
    @inbounds for i in 2:n  # normtext[1] is BLANK
        c = buff.normtext[i]
        p = buff.normtext[i-1]

        ## @show i, p, c
        if ispunct2(c) && !ispunct2(p) && p !== BLANK
            ## @show :a
            s = flush_token!(buff)
            s !== nothing && push!(buff.unigrams, s)
            write(buff.io, c)
        elseif ispunct2(p)
            if ispunct2(c) && buff.io.size > 2
                s = flush_token!(buff)
                s !== nothing && push!(buff.unigrams, s)
                write(buff.io, c)
            elseif !ispunct2(c) && !(p in ('#', '@', '_'))
                ## @show :b
                s = flush_token!(buff)
                s !== nothing && push!(buff.unigrams, s)
                c !== BLANK && write(buff.io, c)
            else
                write(buff.io, c)
            end
        elseif isemoji(c)
            s = flush_token!(buff)
            s !== nothing && push!(buff.unigrams, s)
            write(buff.io, c)
            s = flush_token!(buff)
            s !== nothing && push!(buff.unigrams, s)
        elseif c == BLANK
            if p !== BLANK
                s = flush_token!(buff)
                #write(buff.io, c)
                s !== nothing && push!(buff.unigrams, s)
            end
        else
            ## @show :d
            write(buff.io, c)
        end
    end

    s = flush_token!(buff)
    s !== nothing && push!(buff.unigrams, s)
    buff.tokens
end

"""
    nwords(q::Integer, buff::TextSearchBuffer)
"""
function nwords(q::Integer, buff::TextSearchBuffer)
    n = length(buff.unigrams)

    @inbounds for i in 1:(n - q + 1)
        _last = i + q - 1
        write(buff.io, '\t', 'n')
        for j in i:_last-1
            write(buff.io, buff.unigrams[j])
            write(buff.io, BLANK)
        end

        write(buff.io, buff.unigrams[_last])
        flush_token!(buff)
    end

    buff.tokens
end

"""
    skipgrams(q::Skipgram, buff::TextSearchBuffer)

Tokenizes using skipgrams
"""
function skipgrams(q::Skipgram, buff::TextSearchBuffer)
    n = length(buff.unigrams)

    for start in 1:(n - (q.qsize + (q.qsize - 1) * q.skip) + 1)
        write(buff.io, '\t', 's')
        if q.qsize == 2
            write(buff.io, buff.unigrams[start])
            write(buff.io, BLANK)
            write(buff.io, buff.unigrams[start + 1 + q.skip])
        else
            ep = q.qsize - 2
            for i in 0:ep
                write(buff.io, buff.unigrams[start + i * (1+q.skip)])
                write(buff.io, BLANK)
            end
            ep += 1
            write(buff.io, buff.unigrams[start + ep * (1+q.skip)])
        end

        flush_token!(buff)
    end

    buff.tokens
end
