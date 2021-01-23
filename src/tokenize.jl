# This file is a part of TextSearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

export tokenize, qgrams, unigrams

"""
    tokenize(config::TextConfig, arr::AbstractVector)::Vector{Symbol}

Tokenizes an array of strings
"""
function tokenize(config::TextConfig, arr::AbstractVector{S})::Vector{Symbol} where S <: AbstractString
    L = Symbol[]
    n = length(arr)
    sizehint!(L, (length(config.nlist) + length(config.slist)) * (div(n, 2) + 1) + length(config.qlist) * n)
    buff = IOBuffer(Vector{UInt8}(undef, 16), write=true)
    for text in arr
        if config.group_url
            text = replace(text, RE_URL => "_url")
        end

        t = normalize(config, text)
        tokenize_(config, t, L, buff)
    end

    L
end

tokenize(config::TextConfig, text::AbstractString) = tokenize(config, Vector{Char}(text))

function tokenize(config::TextConfig, text::Vector{Char})
    L = String[]
    n = length(text)
    sizehint!(L, (length(config.nlist) + length(config.slist)) * (div(n, 2) + 1) + length(config.qlist) * n)
    buff = IOBuffer(Vector{UInt8}(undef, 16), write=true)
    tokenize(config, Vector{Char}(text), L, buff)
end

function tokenize(config::TextConfig, textlist::AbstractVector)
    L = String[]
    n = length(textlist) * length(first(textlist))
    sizehint!(L, (length(config.nlist) + length(config.slist)) * (div(n, 2) + 1) + length(config.qlist) * n)
    buff = IOBuffer(Vector{UInt8}(undef, 16), write=true)
    for text in textlist
        if text isa AbstractString
            tokenize(config, Vector{Char}(text), L, buff)
        else
            tokenize(config, text, L, buff)
        end
    end
    
    L
end

function tokenize(config::TextConfig, text::Vector{Char}, L::Vector{String}, buff::IOBuffer)
    for q in config.qlist
        qgrams(text, q, L)
    end

    if length(config.nlist) > 0 || length(config.slist) > 0
        n1 = length(L)
        unigrams(text, L, buff)  # unigrams are always activated if any |nlist| > 0 or |slist| > 0
        word_list = @view L[n1+1:length(L)]

        for q in config.nlist
            q != 1 && nwords(word_list, q, L, buff)
        end

        for q in config.slist
            skipgrams(word_list, q, L)
        end
    end

    L
end

qgrams(text::AbstractString, q::Integer) = qgrams(Vector{Char}(text), q, String[])

function qgrams(text::Vector{Char}, q::Integer, L::Vector{String})
    n = length(text)

    @inbounds for i in 1:(n - q + 1)
        last = i + q - 1
        push!(L, String(text[i:last]))
    end

    L
end


"""
    push_token!(output::Vector{Symbol}, buff::IOBuffer)

Pushes the word inside buffer into token list; it discards empty strings.
"""
function push_token!(output::AbstractVector, buff::IOBuffer)
    token = take!(buff) #::Vector{UInt8}
    length(token) > 0 && push!(output, String(token))
end

function unigrams(text::AbstractString)
    buff = IOBuffer(Vector{UInt8}(undef, 16), write=true)
    unigrams(Vector{Char}(text), String[], buff)
end

"""
    unigrams(text::Vector{Char}, L::Vector{String}, buff::IOBuffer)

Performs the word tokenization
"""
function unigrams(text::Vector{Char}, L::Vector{String}, buff::IOBuffer)
    n = length(text)
    @inbounds for i in 1:n
        c = text[i]

        if c == BLANK
            push_token!(L, buff)
        elseif i > 1
            if text[i-1] in PUNCTUACTION && !(c in PUNCTUACTION) 
                # flushing from punctuaction to non punctuaction
                push_token!(L, buff)
                write(buff, c)
                continue
            elseif !(text[i-1] in PUNCTUACTION_BLANK) && c in PUNCTUACTION
                # flushing from neither punctuaction nor blank to some punctuaction symbol
                push_token!(L, buff)
                write(buff, c)
                continue
            else
                write(buff, c)
            end
        else
            write(buff, c)
        end
    end

    push_token!(L, buff)

    L
end


"""
    tokenize_(config::TextConfig, text::Vector{Char}, L::Vector{Symbol}, buff)::Vector{Symbol}

Tokenizes a vector of characters (internal method)
"""
function nwords(word_list::AbstractVector, q, L::Vector{String}, buff)
    n = length(word_list)

    @inbounds for i in 1:(n - q + 1)
        _last = i + q - 1
        for j in i:_last-1
            # for w in @view word_list[i:i+q-1]
            write(buff, word_list[j])
            write(buff, BLANK)
        end

        write(buff, word_list[_last])
        push_token!(L, buff)
    end

    L
end


function skipgrams(word_list::AbstractVector, q::Skipgram, L::Vector{String})
    n = length(word_list)

    for start in 1:(n - (q.qsize + (q.qsize - 1) * q.skip) + 1)
        if q.qsize == 2
            t = string(word_list[start], BLANK, word_list[start + 1 + q.skip])
        else
            t = string(join([word_list[start + i * (1+q.skip)] for i in 0:(q.qsize-1)], BLANK))
        end
        
        push!(L, t)
    end


    L
end
