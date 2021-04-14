# This file is a part of TextSearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

export normalize_text
using Base.Unicode

#, language!
# using Languages
# using SnowballStemmer

# A symbol s in this list will be expanded to BLANK*s if the predecesor of s is neither s nor BLANK
# On changes from s to BLANK or [^s] it will produce also produce an extra BLANK
# Note that enabled del_punc will delete all these symbols without any of the previous expansions

const BLANK = ' '
const EMOJIS = Set([l[1] for l in readlines(joinpath(@__DIR__, "emojis.txt"))])

const RE_USER = r"""@[^;:,.@#&\\\-\"'/:\*\(\)\[\]\¿\?\¡\!\{\}~\<\>\|\s]+"""
const RE_URL = r"(http|ftp|https)://\S+"
const RE_NUM = r"\d+"

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
    normalize_text(config::TextConfig, text::AbstractString, output::Vector{Char})

Normalizes a given text using the specified transformations of `config`
"""
function normalize_text(config::TextConfig, text::AbstractString, output::Vector{Char})
    text = _preprocessing(config, text)
    push!(output, BLANK)
    
    @inbounds for u in Unicode.normalize(text, casefold=config.lc, stripmark=config.del_diac, stripcc=true, compat=true)
        isspace(u) && (u = BLANK)
        config.del_punc && ispunct(u) && !(u in ('@', '#', '_')) && (u = BLANK)
        config.group_emo && u in EMOJIS && (u = '😂')
        config.del_dup && u === output[end] && continue

        push!(output, u)
    end

    push!(output, BLANK)
    output
end