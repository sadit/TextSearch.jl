# This file is a part of TextSearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

export normalize_text
using Base.Unicode

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
const BLANK = ' '
const PUNCTUACTION_BLANK = Set(_PUNCTUACTION * _SYMBOLS * BLANK)
const EMOJIS = Set([l[1] for l in readlines(joinpath(@__DIR__, "emojis.txt"))])


"""
    normalize_text(config::TextConfig, text::AbstractString, output::Vector{Char})

Normalizes a given text using the specified transformations of `config`

"""
function normalize_text(config::TextConfig, text::AbstractString, output::Vector{Char})
    push!(output, BLANK)
    prev = BLANK
    user = false
    url = 0

    function f(u)
        if isspace(u)
            u = BLANK
        end

        if config.group_num && isnumeric(u)
            u = '0'
            u != prev && push!(output, u)
            prev = u
            return
        end

        if config.group_usr
            if u === '@'
                user = true
                prev = u
                push!(output, '_'); push!(output, 'u'); push!(output, 's'); push!(output, 'r'); 
                return
            elseif user # consumes until BLANK is found
                if u === BLANK
                    user = false
                else
                    return
                end
            end
        end

        if config.group_url
             if url == 0 && u === 'h'
                url = length(output)
                prev = u
                push!(output, u)
                return
            elseif url > 0 # consumes until BLANK is found
                if u === BLANK
                    if (length(output) - url) < 10  # http(s)://a.b
                        url = 0
                    elseif output[url+1] == 't' && output[url+2] == 't' && output[url+1] == 'p' && output[url+1] == ':'
                        output[url] = '_'; output[url+1] = 'u'; output[url+2] = 'r'; output[url + 3] = 'l'
                        resize!(output, url + 4)
                        url = 0
                    else
                        url = 0
                    end
                else
                    push!(output, u)
                    return
                end
            end
        end

        if config.del_dup && prev === u
            return
        elseif config.del_punc && ispunct(u) && u != '#'
            output[end] !== BLANK && push!(output, BLANK)
            prev = u
            return
        end

        if u in EMOJIS
            if output[end] !== BLANK
                push!(output, BLANK)
            end
            if config.group_emo
                push!(output, '_');push!(output, 'e');push!(output, 'm');push!(output, 'o')
                prev = u
                return
            end
        end

        prev = u
        push!(output, u)
    end

    @inbounds for u in Unicode.normalize(text, casefold=config.lc, stripmark=config.del_diac, stripcc=true, compat=true)
        f(u)
    end

    f(BLANK)

    output
end

