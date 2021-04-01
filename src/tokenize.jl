# This file is a part of TextSearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

export TokenizerBuffer, tokenize, qgrams, unigrams

struct TokenizerBuffer
    normtext::Vector{Char}
    tokens::Vector{Symbol}
    unigrams::Vector{Symbol}
    currtoken::Vector{UInt8}
    io::IOBuffer

    function TokenizerBuffer(; n=128)
        normtext = Vector{Char}(undef, n)
        tokens = Vector{Symbol}(undef, n)
        unigrams = Vector{Symbol}(undef, n)
        currtoken = Vector{UInt8}(undef, 16)
        resize!(normtext, 0)
        resize!(tokens, 0)
        resize!(unigrams, 0)
        resize!(currtoken, 0)

        new(
            normtext,
            tokens,
            unigrams,
            currtoken,
            IOBuffer(),
        )
    end
end

function Base.empty!(buff::TokenizerBuffer)
    empty!(buff.normtext)
    empty!(buff.tokens)
    empty!(buff.unigrams)
    empty!(buff.currtoken)
end

function tokenize__(config::TextConfig, textlist::AbstractVector, buff::TokenizerBuffer=TokenizerBuffer())
    n = length(textlist) * length(first(textlist))

    for text in textlist
        empty!(buff)
        if text isa AbstractString
            normalize_text(config, text, buff.normtext)
            tokenize_(config, buff)
        else
            tokenize(config, text, buff)
        end
    end

    buff.tokens
end

"""
    tokenize(config::TextConfig, text::AbstractString)

Tokenizes `text` using the given configuration
"""
function tokenize(config::TextConfig, text::AbstractString, buff::TokenizerBuffer=TokenizerBuffer())
    normalize_text(config, text, buff.normtext)
    tokenize_(config, buff)
end

function tokenize_(config::TextConfig, buff::TokenizerBuffer)
    for q in config.qlist
        qgrams(q, buff)
    end
    
    if length(config.nlist) > 0 || length(config.slist) > 0
        n1 = length(buff.tokens)
        unigrams(buff)  # unigrams are always activated if any |nlist| > 0 or |slist| > 0
        word_list = @view buff.tokens[n1+1:length(buff.tokens)]

        if length(config.nlist) == 0 || config.nlist[1] != 1 # always sorted
            append!(buff.unigrams, word_list)
            word_list = buff.unigrams
            #word_list = copy(word_list)
            resize!(buff.tokens, n1)
        end

        for q in config.nlist 
            q != 1 && nwords(word_list, q, buff)
        end

        for q in config.slist
            skipgrams(word_list, q, buff)
        end
    end

    buff.tokens
end

"""
    qgrams(q::Integer, buff)

Computes character q-grams for the given input
"""
function qgrams(q::Integer, buff::TokenizerBuffer)
    n = length(buff.normtext)

    for i in 1:(n - q + 1)
        for j in i:i+q-1
            @inbounds write(buff.io, buff.normtext[j])
        end

        flush_token!(buff)
    end

    buff.tokens
end

"""
    flush_token!(buff::IOBuffer)

Pushes the word inside buffer into token list; it discards empty strings.

"""
function flush_token!(buff::TokenizerBuffer)
    io = buff.io
    #= if b.size > 0
        resize!(buff.currtoken, b.size)
        seekstart(b)
        read!(b, buff.currtoken)
        push!(buff.tokens, Symbol(buff.currtoken))
        truncate(b, 0)
    end=#

    if io.size > 0
        # WARNING this is IOBuffer implementation dependant
        resize!(buff.currtoken, io.size)
        for i in 1:io.size
            @inbounds buff.currtoken[i] = io.data[i]
        end

        
        ss = Symbol(buff.currtoken)
        push!(buff.tokens, ss)
        truncate(io, 0)
    end
end

"""
    unigrams(buff::TokenizerBuffer)

Performs the word tokenization
"""
function unigrams(buff::TokenizerBuffer)
    n = length(buff.normtext)
    ## write(buff, '~')
    @inbounds for i in 1:n
        c = buff.normtext[i]

        if c == BLANK
            flush_token!(buff)
            ## write(buff, '~')
#        elseif i > 1
#            if c != '_' && ispunct(buff.normtext[i-1]) && !ispunct(c)
#                # flushing from punctuaction to non punctuaction
#                flush_token!(buff)
#                ## write(buff, '~')
#                write(buff.io, c)
#                continue
#            elseif !ispunct(buff.normtext[i-1]) && ispunct(c)
#                # flushing from neither punctuaction nor blank to some punctuaction symbol
#                flush_token!(buff)
#                ## write(buff.io, '~')
#                write(buff.io, c)
#                continue
#            else
#                write(buff.io, c)
#            end
        else
            write(buff.io, c)
        end
    end

    flush_token!(buff)
    buff.tokens
end

"""
    nwords(word_list::AbstractVector, q::Integer, buff::TokenizerBuffer)

"""
function nwords(word_list::AbstractVector, q::Integer, buff::TokenizerBuffer)
    n = length(word_list)

    @inbounds for i in 1:(n - q + 1)
        _last = i + q - 1
        for j in i:_last-1
            write(buff.io, word_list[j])
            write(buff.io, BLANK)
        end

        write(buff.io, word_list[_last])
        flush_token!(buff)
    end

    buff.tokens
end

"""
    skipgrams(word_list::AbstractVector, q::Skipgram, buff::TokenizerBuffer)

Tokenizes using skipgrams
"""
function skipgrams(word_list::AbstractVector, q::Skipgram, buff::TokenizerBuffer)
    n = length(word_list)

    for start in 1:(n - (q.qsize + (q.qsize - 1) * q.skip) + 1)
        if q.qsize == 2
            write(buff.io, word_list[start])
            write(buff.io, BLANK)
            write(buff.io, word_list[start + 1 + q.skip])
        else
            ep = q.qsize-2
            for i in 0:ep
                write(buff.io, word_list[start + i * (1+q.skip)])
                write(buff.io, BLANK)
            end
            ep += 1
            write(buff.io, word_list[start + ep * (1+q.skip)])
        end

        flush_token!(buff)
    end

    buff.tokens
end
