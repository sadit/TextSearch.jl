# This file is a part of TextSearch.jl

export TokenizedText, tokenize, tokenize_corpus, qgrams, unigrams, filter_tokens!

struct TokenizedText{StringVector<:AbstractVector{String}}
    tokens::StringVector 
end

@inline Base.getindex(T::TokenizedText, i::Integer) = T.tokens[i]
@inline Base.firstindex(T::TokenizedText) = 1
@inline Base.lastindex(T::TokenizedText) = length(T)
@inline Base.length(T::TokenizedText) = length(T.tokens)
@inline Base.iterate(T::TokenizedText, s::Int=1) = iterate(T.tokens, s)
@inline Base.eltype(T::TokenizedText) = eltype(T.tokens)
@inline Base.push!(T::TokenizedText, a) = push!(T.tokens, a)
@inline Base.append!(T::TokenizedText, a) = append!(T.tokens, a)

tokenizedtext(s) = TokenizedText(Vector(s))
borrowtokenizedtext(s) = TokenizedText(s)

tokenize(copy_::Function, textconfig::TextConfig, text::TokenizedText, buff::TextSearchBuffer) = text 
tokenize(copy_::Function, textconfig::TextConfig, text::TokenizedText) = text 
tokenize(copy_::Function, textconfig::TextConfig, arr::AbstractVector{T}, buff::TextSearchBuffer) where {T<:TokenizedText} = arr
tokenize_corpus(copy_::Function, textconfig::TextConfig, arr::AbstractVector{T}; minbatch=0) where {T<:TokenizedText} = arr

const EXTRA_PUNCT = Set(['~', '+', '^', '$', '|', '<', '>'])

"""
    tokenize(textconfig::TextConfig, text)
    tokenize(copy_::Function, textconfig::TextConfig, text)

    tokenize(textconfig::TextConfig, text, buff)
    tokenize(copy_::Function, textconfig::TextConfig, text, buff)

Tokenizes `text` using the given configuration. The `tokenize` makes heavy usage of buffers,
and when these buffers are shared it is mandatory to create a copy of the result (`buff.tokens`).

Change the default `copy` function to make an additional filtering of the tokens.
You can also pass the `identity` function to avoid copying.

"""
function tokenize(copy_::Function, textconfig::TextConfig, text::AbstractString, buff::TextSearchBuffer)
    normalize_text(textconfig, text, buff.normtext)
    t = tokenize_(textconfig, buff)
    copy_(t)
end

function tokenize(copy_::Function, textconfig::TextConfig, arr::AbstractVector{T}, buff::TextSearchBuffer) where {T<:AbstractString}
    normalize_text(textconfig, arr[1], buff.normtext)
    tokenize_(textconfig, buff)

    for i in 2:length(arr)
        empty!(buff.normtext); empty!(buff.unigrams)
        normalize_text(textconfig, arr[i], buff.normtext)
        tokenize_(textconfig, buff)
    end

    copy_(buff.tokens)
end

tokenize(textconfig::TextConfig, text) = tokenize(tokenizedtext, textconfig, text)

function tokenize(copy_::Function, textconfig::TextConfig, text)
    buff = take!(TEXT_SEARCH_CACHES)
    empty!(buff)
    try
        tokenize(copy_, textconfig, text, buff)
    finally
        put!(TEXT_SEARCH_CACHES, buff)
    end
end

"""
    tokenize_corpus(textconfig::TextConfig, arr; minbatch=0)
    tokenize_corpus(copy_::Function, textconfig::TextConfig, arr; minbatch=0)

Tokenize a list of texts. The `copy_` function is passed to [`tokenize`](@ref) as first argument.
"""
function tokenize_corpus(copy_::Function, textconfig::TextConfig, arr; minbatch=0)
    n = length(arr)
    L = Vector{TokenizedText}(undef, n)
    minbatch = getminbatch(minbatch, n)
    
    # @batch minbatch=minbatch per=thread
    Threads.@threads for i in 1:n
        L[i] = tokenize(copy_, textconfig, arr[i])
    end

    L
end

tokenize_corpus(textconfig::TextConfig, arr; minbatch=0) = tokenize_corpus(tokenizedtext, textconfig, arr; minbatch)

function tokenize_(config::TextConfig, buff::TextSearchBuffer)
    for q in config.qlist
        qgrams(q, buff, config.tt, config.mark_token_type)
    end
    
    if length(config.nlist) > 0 || length(config.slist) > 0
        n1 = length(buff.tokens)
        unigrams(buff, config.tt)  # unigrams are always activated if any |nlist| > 0 or |slist| > 0

        if length(config.nlist) == 0 || config.nlist[1] != 1 # always sorted
            resize!(buff.tokens, n1)
        end

        for q in config.nlist 
            q != 1 && nwords(q, buff, config.tt, config.mark_token_type)
        end

        for q in config.slist
            skipgrams(q, buff, config.tt, config.mark_token_type)
        end
    end

    buff.tokens
end

"""
    flush_unigram!(buff::TextSearchBuffer, tt::AbstractTokenTransformation)

Pushes the word inside the buffer to the token list; it discards empty strings.
"""
function flush_unigram!(buff::TextSearchBuffer, tt::AbstractTokenTransformation)
    buff.io.size == 0 && return nothing
    s = transform_unigram(tt, String(take!(buff.io)))
    s === nothing && return nothing
    push!(buff.tokens, s)
    s
end

"""
    flush_nword!(buff::TextSearchBuffer, tt::AbstractTokenTransformation, mark_token_type)

Pushes the nword inside the buffer to the token list; it discards empty strings.
"""
function flush_nword!(buff::TextSearchBuffer, tt::AbstractTokenTransformation, mark_token_type)
    buff.io.size == 0 && return nothing
    mark_token_type && write(buff.io, '\t', 'n')
    s = transform_nword(tt, String(take!(buff.io)))
    s === nothing && return nothing
    push!(buff.tokens, s)
    s
end

"""
    flush_qgram!(buff::TextSearchBuffer, tt::AbstractTokenTransformation, mark_token_type)

Pushes the qgram inside the buffer to the token list; it discards empty strings.
"""
function flush_qgram!(buff::TextSearchBuffer, tt::AbstractTokenTransformation, mark_token_type)
    buff.io.size == 0 && return nothing
    mark_token_type && write(buff.io, '\t', 'q')
    s = transform_qgram(tt, String(take!(buff.io)))
    s === nothing && return nothing
    push!(buff.tokens, s)
    s
end

"""
    flush_skipgram!(buff::TextSearchBuffer, tt::AbstractTokenTransformation, mark_token_type)

Pushes the skipgram inside the buffer to the token list; it discards empty strings.
"""
function flush_skipgram!(buff::TextSearchBuffer, tt::AbstractTokenTransformation, mark_token_type)
    buff.io.size == 0 && return nothing
    mark_token_type && write(buff.io, '\t', 's')
    s = transform_skipgram(tt, String(take!(buff.io)))
    s === nothing && return nothing
    push!(buff.tokens, s)
    s
end

"""
    qgrams(q::Integer, buff::TextSearchBuffer, tt::AbstractTokenTransformation, mark_token_type)

Computes character q-grams for the given input
"""
function qgrams(q::Integer, buff::TextSearchBuffer, tt::AbstractTokenTransformation, mark_token_type)
    n = length(buff.normtext)

    for i in 1:(n - q + 1)
        for j in i:i+q-1
            @inbounds write(buff.io, buff.normtext[j])
        end
        flush_qgram!(buff, tt, mark_token_type)
    end

    buff.tokens
end

ispunct2(c) = ispunct(c) || c in EXTRA_PUNCT

"""
    unigrams(buff::TextSearchBuffer, tt::AbstractTokenTransformation)

Performs the word tokenization
"""
function unigrams(buff::TextSearchBuffer, tt::AbstractTokenTransformation)
    n = length(buff.normtext)
    # @info buff.normtext
    @inbounds for i in 2:n  # normtext[1] is BLANK
        c = buff.normtext[i]
        p = buff.normtext[i-1]

        ## @show i, p, c
        if ispunct2(c) && !ispunct2(p) && p !== BLANK
            ## @show :a
            s = flush_unigram!(buff, tt)
            s !== nothing && push!(buff.unigrams, s)
            write(buff.io, c)
        elseif ispunct2(p)
            if ispunct2(c) && buff.io.size > 2
                s = flush_unigram!(buff, tt)
                s !== nothing && push!(buff.unigrams, s)
                write(buff.io, c)
            elseif !ispunct2(c) && !(p in ('#', '@', '_'))
                ## @show :b
                s = flush_unigram!(buff, tt)
                s !== nothing && push!(buff.unigrams, s)
                c !== BLANK && write(buff.io, c)
            else
                write(buff.io, c)
            end
        elseif isemoji(c)
            s = flush_unigram!(buff, tt)
            s !== nothing && push!(buff.unigrams, s)
            write(buff.io, c)
            s = flush_unigram!(buff, tt)
            s !== nothing && push!(buff.unigrams, s)
        elseif c == BLANK
            if p !== BLANK
                s = flush_unigram!(buff, tt)
                #write(buff.io, c)
                s !== nothing && push!(buff.unigrams, s)
            end
        else
            ## @show :d
            write(buff.io, c)
        end
    end

    s = flush_unigram!(buff, tt)
    s !== nothing && push!(buff.unigrams, s)
    buff.tokens
end

"""
    nwords(q::Integer, buff::TextSearchBuffer, tt::AbstractTokenTransformation, mark_token_type)
"""
function nwords(q::Integer, buff::TextSearchBuffer, tt::AbstractTokenTransformation, mark_token_type)
    n = length(buff.unigrams)

    @inbounds for i in 1:(n - q + 1)
        _last = i + q - 1
        for j in i:_last-1
            write(buff.io, buff.unigrams[j])
            write(buff.io, BLANK)
        end

        write(buff.io, buff.unigrams[_last])
        flush_nword!(buff, tt, mark_token_type)
    end

    buff.tokens
end

"""
    skipgrams(q::Skipgram, buff::TextSearchBuffer, tt::AbstractTokenTransformation, mark_token_type)

Tokenizes using skipgrams
"""
function skipgrams(q::Skipgram, buff::TextSearchBuffer, tt::AbstractTokenTransformation, mark_token_type)
    n = length(buff.unigrams)

    for start in 1:(n - (q.qsize + (q.qsize - 1) * q.skip) + 1)
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

        flush_skipgram!(buff, tt, mark_token_type)
    end

    buff.tokens
end

