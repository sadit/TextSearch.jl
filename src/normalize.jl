# This file is a part of TextSearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

export normalize_text
using Unicode

#, language!
# using Languages
# using SnowballStemmer

# const _PUNCTUACTION = """;:,.@#&\\-\"'/:*"""
const _PUNCTUACTION = """;:,.&\\-\"'/:*"️“”«»"""
const _SYMBOLS = "()[]¿?¡!{}~<>|^"
const PUNCTUACTION  = Set(_PUNCTUACTION * _SYMBOLS)
# A symbol s in this list will be expanded to BLANK*s if the predecesor of s is neither s nor BLANK
# On changes from s to BLANK or [^s] it will produce also produce an extra BLANK
# Note that enabled del_punc will delete all these symbols without any of the previous expansions

const BLANK_LIST = string(' ', '\t', '\n', '\v', '\r')
const RE_USER = r"""@[^;:,.@#&\\\-\"'/:\*\(\)\[\]\¿\?\¡\!\{\}~\<\>\|\s]+"""
const RE_URL = r"(http|ftp|https)://\S+"
const RE_NUM = r"\d+"
const BLANK = ' '
const PUNCTUACTION_BLANK = Set(_PUNCTUACTION * _SYMBOLS * BLANK)
const EMOJIS = Set([l[1] for l in readlines(joinpath(@__DIR__, "emojis.txt"))])

# SKIP_WORDS = set(["…", "..", "...", "...."])


function _preprocessing(config::TextConfig, text)
    if config.lc
        text = lowercase(text)
    end

    if config.group_url
        text = replace(text, RE_URL => "_url")
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
    normalize_text(config::TextConfig, text::AbstractString, L::Vector{Char}=Vector{Char}())::Vector{Char}

Normalizes a given text using the specified transformations of `config`

"""
function normalize_text(config::TextConfig, text::AbstractString, L::Vector{Char}=Vector{Char}())::Vector{Char}
    text = _preprocessing(config, text)
    push!(L, BLANK)
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

