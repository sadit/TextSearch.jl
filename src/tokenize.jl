# This file is a part of TextSearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

export tokenize_multimessage, tokenize, qgrams, unigrams

"""
    tokenize(config::TextConfig, text::AbstractString)

Tokenizes `text` using the given configuration
"""
tokenize(config::TextConfig, text::AbstractString) = tokenize(config, normalize_text(config, text))

function tokenize_multimessage(config::TextConfig, textlist::AbstractVector;
        output=String[],
        normalizebuffer=Char[],
        buff=IOBuffer(Vector{UInt8}(undef, 16), write=true)
    )
    n = length(textlist) * length(first(textlist))
    sizehint!(output, (length(config.nlist) + length(config.slist)) * (div(n, 2) + 1) + length(config.qlist) * n)

    for text in textlist
        if text isa AbstractString
            empty!(normalizebuffer)
            normalize_text(config, text, output)
            tokenize(config, normalizebuffer; output=output, buff=buff)
        else
            tokenize(config, text, output=output, buff=buff)
        end
    end

    output
end

function tokenize(config::TextConfig, text::Vector{Char};
        output::Vector{String}=String[],
        buff::IOBuffer=IOBuffer(Vector{UInt8}(undef, 16), write=true)
    )
    for q in config.qlist
        qgrams(text, q, output, buff)
    end
    
    if length(config.nlist) > 0 || length(config.slist) > 0
        n1 = length(output)
        unigrams(text, output, buff)  # unigrams are always activated if any |nlist| > 0 or |slist| > 0
        word_list = @view output[n1+1:length(output)]

        if length(config.nlist) == 0 || config.nlist[1] != 1 # always sorted
            word_list = copy(word_list)
            resize!(output, n1)
        end

        for q in config.nlist 
            q != 1 && nwords(word_list, q, output, buff)
        end

        for q in config.slist
            skipgrams(word_list, q, output, buff)
        end
    end

    output
end

"""
    qgrams(text::AbstractString, q::Integer)
    qgrams(text::Vector{Char}, q::Integer, L::Vector{String}, buff)

Computes character q-grams for the given input
"""
qgrams(text::AbstractString, q::Integer) = qgrams(Vector{Char}(text), q, String[], IOBuffer(Vector{UInt8}(undef, 16), write=true))

function qgrams(text::Vector{Char}, q::Integer, L::Vector{String}, buff)
    n = length(text)

    @inbounds for i in 1:(n - q + 1)
        for j in i:i+q-1
            write(buff, text[j])
        end

        push_token!(L, buff)
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
    ## write(buff, '~')
    @inbounds for i in 1:n
        c = text[i]

        if c == BLANK
            push_token!(L, buff)
            ## write(buff, '~')
        elseif i > 1
            if text[i-1] in PUNCTUACTION && !(c in PUNCTUACTION) 
                # flushing from punctuaction to non punctuaction
                push_token!(L, buff)
                ## write(buff, '~')
                write(buff, c)
                continue
            elseif !(text[i-1] in PUNCTUACTION_BLANK) && c in PUNCTUACTION
                # flushing from neither punctuaction nor blank to some punctuaction symbol
                push_token!(L, buff)
                ## write(buff, '~')
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
    nwords(word_list::AbstractVector, q, L::Vector{String}, buff)

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

"""
    skipgrams(word_list::AbstractVector, q::Skipgram, L::Vector{String}, buff)

Tokenizes using skipgrams
"""
function skipgrams(word_list::AbstractVector, q::Skipgram, L::Vector{String}, buff)
    n = length(word_list)

    for start in 1:(n - (q.qsize + (q.qsize - 1) * q.skip) + 1)
        if q.qsize == 2
            write(buff, word_list[start])
            write(buff, BLANK)
            write(buff, word_list[start + 1 + q.skip])
        else
            ep = q.qsize-2
            for i in 0:ep
                write(buff, word_list[start + i * (1+q.skip)])
                write(buff, BLANK)
            end
            ep += 1
            write(buff, word_list[start + ep * (1+q.skip)])
        end

        push_token!(L, buff)
    end


    L
end
