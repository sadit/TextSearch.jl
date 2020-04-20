# This file is a part of TextSearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

export TextConfig, save, load, normalize_text, tokenize
using Unicode

#, language!
# using Languages
# using SnowballStemmer

# const _PUNCTUACTION = """;:,.@#&\\-\"'/:*"""
const _PUNCTUACTION = """;:,.&\\-\"'/:*"️“”«»"""
const _SYMBOLS = "()[]¿?¡!{}~<>|"
const PUNCTUACTION  = _PUNCTUACTION * _SYMBOLS
# A symbol s in this list will be expanded to BLANK*s if the predecesor of s is neither s nor BLANK
# On changes from s to BLANK or [^s] it will produce also produce an extra BLANK
# Note that enabled del_punc will delete all these symbols without any of the previous expansions

const BLANK_LIST = string(' ', '\t', '\n', '\v', '\r')
const RE_USER = r"""@[^;:,.@#&\\\-\"'/:\*\(\)\[\]\¿\?\¡\!\{\}~\<\>\|\s]+"""
const RE_URL = r"(http|ftp|https)://\S+"
const RE_NUM = r"\d+"
const BLANK = ' '
const PUNCTUACTION_BLANK = string(PUNCTUACTION, BLANK)
const EMOJIS = Set([l[1] for l in readlines(joinpath(@__DIR__, "emojis.txt"))])

# SKIP_WORDS = set(["…", "..", "...", "...."])

mutable struct TextConfig
    del_diac::Bool
    del_dup::Bool
    del_punc::Bool
    group_num::Bool
    group_url::Bool
    group_usr::Bool
    group_emo::Bool
    lc::Bool
    qlist::Vector{Int}
    nlist::Vector{Int}
    slist::Vector{Tuple{Int,Int}}
    normalize_words::Function

    """
    Initializes a `TextConfig` structure
    """
    function TextConfig(;
        del_diac=true,
        del_dup=false,
        del_punc=false,
        group_num=true,
        group_url=true,
        group_usr=false,
        group_emo=false,
        lc=true,
        qlist=Int[],
        nlist=Int[1],
        slist=Tuple{Int,Int}[],
        normalize_words::Function=identity
    )
        new(del_diac, del_dup, del_punc, group_num, group_url, group_usr, group_emo, lc,
            qlist, nlist, slist, normalize_words)
    end
end

"""
    normalize_text(config::TextConfig, text::AbstractString)::Vector{Char}

Normalizes a given text using the specified transformations of `config`

"""
function normalize_text(config::TextConfig, text::AbstractString)::Vector{Char}
    L = Char[BLANK]
    prev = BLANK

    @inbounds for u in Unicode.normalize(text, :NFD)
        if config.del_diac
            o = Int(u)
            0x300 <= o && o <= 0x036F && continue
        end

        if u in BLANK_LIST
            u = BLANK
        elseif config.del_dup && prev == u
            continue
        elseif config.del_punc && u in PUNCTUACTION
            L[end] !== BLANK && push!(L, BLANK)
            prev = u
            continue
        end

        if u in EMOJIS
            if prev != BLANK
                push!(L, BLANK)
            end
            if config.group_emo
                push!(L, '_');push!(L, 'e');push!(L, 'm');push!(L, 'o')
                prev = u
                continue
            end
        end

        prev = u
        push!(L, u)
    end

    push!(L, BLANK)

    L
end

function _apply_preprocessing(config::TextConfig, text)
    if config.lc
        text = lowercase(text)
    end

    if config.group_usr
        text = replace(text, RE_USER => "_usr")
    end

    if config.group_num
        text = replace(text, RE_NUM => "_num")
    end

    text
end

"""
    push_word!(config::TextConfig, output::Vector{Symbol}, token::Vector{UInt8}, normalize_words::Function)

Pushes a word into token list after applying the `normalize_words` function; it discards empty strings.
"""
function push_word!(config::TextConfig, output::AbstractVector, token::Vector{UInt8}, normalize_words::Function)
    if length(token) > 0
        w = _apply_preprocessing(config, String(token))
        t = normalize_words(w)

        if t isa AbstractString && length(t) > 0
            push!(output, t)
        end
    end
end

"""
    tokenize_words(config::TextConfig, text::Vector{Char}, normalize_words::Function, buff::IOBuffer)

Performs the word tokenization
"""
function tokenize_words(config::TextConfig, text::Vector{Char}, normalize_words::Function, buff::IOBuffer)
    n = length(text)
    L = String[]
    @inbounds for i in 1:n
        c = text[i]

        if c == BLANK
            push_word!(config, L, take!(buff), normalize_words)
        elseif i > 1
            if text[i-1] in PUNCTUACTION && !(c in PUNCTUACTION) 
                # flushing from punctuaction to non punctuaction.e
                push_word!(config, L, take!(buff), normalize_words)
                write(buff, c)
                continue
            elseif !(text[i-1] in PUNCTUACTION_BLANK) && c in PUNCTUACTION
                # flushing from neither punctuaction nor blank to some punctuaction symbol
                push_word!(config, L, take!(buff), normalize_words)
                write(buff, c)
                continue
            else
                write(buff, c)
            end
        else
            write(buff, c)
        end
    end

    push_word!(config, L, take!(buff), normalize_words)

    L
end

"""
    tokenize(config::TextConfig, arr::AbstractVector)::Vector{Symbol}

Tokenizes an array of strings
"""
function tokenize(config::TextConfig, arr::AbstractVector{S})::Vector{Symbol} where S <: AbstractString
    L = Symbol[]
    n = length(arr)
    sizehint!(L, (length(config.nlist) + length(config.slist)) * (div(n, 2) + 1) + length(config.qlist) * n)
    buff = IOBuffer(Vector{UInt8}(undef, 64), write=true)
    for text in arr
        if config.group_url
            text = replace(text, RE_URL => "_url")
        end

        t = normalize_text(config, text)
        tokenize_(config, t, L, buff)
    end

    L
end

"""
    tokenize(config::TextConfig, text::AbstractString)::Vector{Symbol}

Tokenizes a string
"""
function tokenize(config::TextConfig, text::AbstractString)::Vector{Symbol}
    if config.group_url
        text = replace(text, RE_URL => "_url")
    end

    t = normalize_text(config, text)
    n = length(text)
    L = Symbol[]
    sizehint!(L, (length(config.nlist) + length(config.slist)) * (div(n, 2) + 1) + length(config.qlist) * n)
    buff = IOBuffer(Vector{UInt8}(undef, 64), write=true)
    tokenize_(config, t, L, buff)
end


"""
    tokenize_(config::TextConfig, text::Vector{Char}, L::Vector{Symbol}, buff)::Vector{Symbol}

Tokenizes a vector of characters (internal method)
"""
function tokenize_(config::TextConfig, text::Vector{Char}, L::Vector{Symbol}, buff)::Vector{Symbol}
    word_list = tokenize_words(config, text, config.normalize_words, buff)
    text = [c for c in join(word_list, BLANK)]
    n = length(text)

    @inbounds for q in config.qlist
        for i in 1:(n - q + 1)
            last = i + q - 1
            push!(L, Symbol(String(text[i:last])))
        end
    end

    if length(config.nlist) > 0 || length(config.slist) > 0
        n = length(word_list)

        @inbounds for q in config.nlist
            for i in 1:(n - q + 1)
                last = i + q - 1
                for j in i:last-1
                    # for w in @view word_list[i:i+q-1]
                    write(buff, word_list[j])
                    write(buff, BLANK)
                end

                write(buff, word_list[last])
                push!(L, Symbol(take!(buff)))
            end
        end

        @inbounds for (qsize, skip) in config.slist
            for start in 1:(n - (qsize + (qsize - 1) * skip) + 1)
                if qsize == 2
                    t = Symbol(word_list[start], BLANK, word_list[start + 1 + skip])
                else
                    t = Symbol(join([word_list[start + i * (1+skip)] for i in 0:(qsize-1)], BLANK))
                end
                
                push!(L, t)
            end
        end
    end

    L
end
