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

"""
    normalize_text(config::TextConfig, text::AbstractString, output::Vector{Char})

Normalizes a given text using the specified transformations of `config`

"""
function normalize_text(config::TextConfig, text::AbstractString, output::Vector{Char})
    push!(output, BLANK)
    user = false
    url = 0
    function f(u)
        if isspace(u)
            u = BLANK
        end

        # group user tokens
        if config.group_usr
            if u === '@'
                user = true
                push!(output, '_'); push!(output, 'u'); push!(output, 's'); push!(output, 'r'); 
                return
            elseif user # consumes until BLANK is found
                if isletter(u) || isnumeric(u) # consumes
                    return
                else # stop consuming chars
                    push!(output, BLANK)
                    user = false
                end
            end
        end

        # group url tokens
        if config.group_url
            if url == 0 && u === 'h'
                url = length(output)
                push!(output, u)
                return
            elseif url > 0 # consumes until BLANK is found
                if u === BLANK
                    url = 0
                    if (length(output) - url) > 10 && output[url+1] == 't' && output[url+2] == 't' && output[url+1] == 'p' && output[url+1] == ':'
                        output[url] = '_'; output[url+1] = 'u'; output[url+2] = 'r'; output[url + 3] = 'l'
                        resize!(output, url + 4)
                    end
                else
                    push!(output, u)
                    return
                end
            end
        end

        #  group numeric tokens
        if config.group_num && isnumeric(u)
            u = '0'
            output[end] != u && push!(output, u)
            return
        end


        # delete punctuactions
        if config.del_punc && ispunct(u)
            output[end] !== BLANK && push!(output, BLANK)
            return
        end
        
        # manage proper spaces on sequences  "punctuaction-alphanumeric" and "alphanumeric-punctuaction"
        # with special handling of '#' and '@' as part of tokens
        if ispunct(u) && !ispunct(output[end])
            output[end] !== BLANK && push!(output, BLANK)
        elseif ispunct(output[end]) && !ispunct(u) && !(output[end] in ('#', '@'))
            push!(output, BLANK)
        end

        # emojis grouping
        if u in EMOJIS
            output[end] !== BLANK && push!(output, BLANK)

            if config.group_emo
                push!(output, '_');push!(output, 'e');push!(output, 'm');push!(output, 'o')
                return
            end
        end

         # delete duplications
        config.del_dup && output[end] === u && return
        u == BLANK && output[end] == BLANK && return
        push!(output, u)
    end

    @inbounds for u in Unicode.normalize(text, casefold=config.lc, stripmark=config.del_diac, stripcc=true, compat=true)
        f(u)
    end

    f(BLANK)

    output
end
