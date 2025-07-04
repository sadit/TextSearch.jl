# This file is a part of TextSearch.jl

export TokenizedText, tokenize, tokenize_corpus, qgrams, unigrams, filter_tokens!

struct TokenizedText{StringVector<:AbstractVector{String}}
    tokens::StringVector 
end

@inline Base.getindex(T::TokenizedText, i::Integer) = T.tokens[i]
@inline Base.setindex!(T::TokenizedText, v, i::Integer) = (T.tokens[i] = v)
@inline Base.firstindex(T::TokenizedText) = 1
@inline Base.lastindex(T::TokenizedText) = length(T)
@inline Base.eachindex(T::TokenizedText) = firstindex(T):lastindex(T)
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

function tokenize(copy_::Function, textconfig::TextConfig, arr::AbstractVector, buff::TextSearchBuffer)
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

function normalize_text(textconfig::TextConfig, text; limits::Bool=false)
    buff = take!(TEXT_SEARCH_CACHES)
    empty!(buff)
    try
        String(normalize_text(textconfig, text, buff.normtext; limits))
    finally
        put!(TEXT_SEARCH_CACHES, buff)
    end
end


"""
    tokenize_corpus(textconfig::TextConfig, arr; minbatch=0, verbose=true)
    tokenize_corpus(copy_::Function, textconfig::TextConfig, arr; minbatch=0, verbose=true)

Tokenize a list of texts. The `copy_` function is passed to [`tokenize`](@ref) as first argument.
"""
function tokenize_corpus(copy_::Function, textconfig::TextConfig, arr; minbatch::Int=0, verbose::Bool=true)
    n = length(arr)
    L = Vector{TokenizedText}(undef, n)
    minbatch = getminbatch(minbatch, n)
    
    # @batch minbatch=minbatch per=thread
    @showprogress dt=1 enabled=verbose desc="tokenizing" Threads.@threads for i in 1:n
        L[i] = tokenize(copy_, textconfig, arr[i])
    end

    L
end

tokenize_corpus(textconfig::TextConfig, arr; minbatch::Int=0, verbose::Bool=true) = tokenize_corpus(tokenizedtext, textconfig, arr; minbatch, verbose)

function tokenize_(config::TextConfig, buff::TextSearchBuffer)
    for q in config.qlist
        qgrams(q, buff, config.tt, config.mark_token_type)
    end
    
    if length(config.nlist) > 0 || length(config.slist) > 0 || config.collocations > 1
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

        if config.collocations > 1
            collocations(config.collocations, buff, config.tt, config.mark_token_type)
        end
    end

    buff.tokens
end

function push_token_from_transform!(tokens, s::Nothing)
end

function push_token_from_transform!(tokens, s::AbstractString)
    push!(tokens, s)
end

function push_token_from_transform!(tokens, slist::AbstractVector)
    for s in slist
        push!(tokens, s)
    end
end

"""
    flush_unigram!(buff::TextSearchBuffer, tt::AbstractTokenTransformation)

Pushes the word inside the buffer to the token list; it discards empty strings.
"""
function flush_unigram!(buff::TextSearchBuffer, tt::AbstractTokenTransformation)
    buff.io.size == 0 && return nothing
    s = transform_unigram(tt, String(take!(buff.io)))
    push_token_from_transform!(buff.tokens, s)
end

"""
    flush_nword!(buff::TextSearchBuffer, tt::AbstractTokenTransformation, mark_token_type)

Pushes the nword inside the buffer to the token list; it discards empty strings.
"""
function flush_nword!(buff::TextSearchBuffer, tt::AbstractTokenTransformation, mark_token_type)
    buff.io.size == 0 && return nothing
    mark_token_type && write(buff.io, '\t', 'n')
    s = transform_nword(tt, String(take!(buff.io)))
    push_token_from_transform!(buff.tokens, s)
end

"""
    flush_qgram!(buff::TextSearchBuffer, tt::AbstractTokenTransformation, mark_token_type)

Pushes the qgram inside the buffer to the token list; it discards empty strings.
"""
function flush_qgram!(buff::TextSearchBuffer, tt::AbstractTokenTransformation, mark_token_type)
    buff.io.size == 0 && return nothing
    mark_token_type && write(buff.io, '\t', 'q')
    s = transform_qgram(tt, String(take!(buff.io)))
    push_token_from_transform!(buff.tokens, s)
end

"""
    flush_skipgram!(buff::TextSearchBuffer, tt::AbstractTokenTransformation, mark_token_type)

Pushes the skipgram inside the buffer to the token list; it discards empty strings.
"""
function flush_skipgram!(buff::TextSearchBuffer, tt::AbstractTokenTransformation, mark_token_type)
    buff.io.size == 0 && return nothing
    mark_token_type && write(buff.io, '\t', 's')
    s = transform_skipgram(tt, String(take!(buff.io)))
    push_token_from_transform!(buff.tokens, s)
end

"""
    flush_collocations!(buff::TextSearchBuffer, tt::AbstractTokenTransformation, mark_token_type)

Pushes a collocation inside the buffer to the token list; it discards empty strings.
"""
function flush_collocation!(buff::TextSearchBuffer, tt::AbstractTokenTransformation, mark_token_type)
    buff.io.size == 0 && return nothing
    mark_token_type && write(buff.io, '\t', 'c')
    s = transform_collocation(tt, String(take!(buff.io)))
    push_token_from_transform!(buff.tokens, s)
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
    mfirst = length(buff.tokens) + 1
    # @info buff.normtext
    @inbounds for i in 2:n  # normtext[1] is BLANK
        c = buff.normtext[i]
        p = buff.normtext[i-1]

        if c == BLANK
            flush_unigram!(buff, tt)
        elseif isemoji(c)
            # emoji
            flush_unigram!(buff, tt)
            write(buff.io, c)
            flush_unigram!(buff, tt)
        elseif ispunct2(p)
            # previous char is punct
            if ispunct2(c)
                # a punctuaction string
                buff.io.size >= 3 && flush_unigram!(buff, tt)  # a bit large, so we flush and restart the punc string (3 is for most emojis and ...)
                write(buff.io, c)
            else
                !(p in ('#', '@', '_')) && flush_unigram!(buff, tt)  # current is not punctuaction so we flush if not a meta word
                write(buff.io, c)
            end
        elseif ispunct2(c) && p !== BLANK
            ## single punctuaction alone
            flush_unigram!(buff, tt)
            write(buff.io, c)
        else
            write(buff.io, c)
        end
    end

    flush_unigram!(buff, tt)
    mlast = length(buff.tokens)

    for i in mfirst:mlast
        push!(buff.unigrams, buff.tokens[i])
    end

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
    collocations(q, buff::TextSearchBuffer, tt::AbstractTokenTransformation, mark_token_type)

Computes a kind of collocations of the given text
"""
function collocations(q::Integer, buff::TextSearchBuffer, tt::AbstractTokenTransformation, mark_token_type)
    tokens = buff.unigrams
    n = length(tokens)

    for i in 1:n-1 # the upper limit is an implementation detail to discard some entries 
        for j in i+1:min(i+1+q, n)
            write(buff.io, buff.unigrams[i])
            write(buff.io, BLANK)
            write(buff.io, buff.unigrams[j])
            flush_collocation!(buff, tt, mark_token_type)
        end
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

