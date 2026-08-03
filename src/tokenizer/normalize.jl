# This file is a part of TextSearch.jl

export normalize_text
using Base.Unicode

#, language!
# using Languages
# using SnowballStemmer

function _preprocessing(config::TextConfig, text)
    if config.lc
        text = lowercase(text)
    end

    if config.group_url
        text = replace(text, config.re_url => "_url ")
    end

    if config.group_usr
        text = replace(text, config.re_user => "_usr ")
    end

    if config.group_num
        text = replace(text, config.re_num => "0 ")
    end

    text
end

"""
    normalize_text(config::TextConfig, text::AbstractString, output::Vector{Char}; limits::Bool=true)

Normalizes a given text using the specified transformations of `config`

# Example

```julia
julia> buff = Char[];

julia> normalize_text(TextConfig(), "Café", buff);

julia> String(buff)
" cafe "
```
"""
function normalize_text(config::TextConfig, text::AbstractString, output::Vector{Char}; limits::Bool=true)
    text = _preprocessing(config, text)
    limits && push!(output, BLANK)
    rep = 0

    @inbounds for u in Unicode.normalize(text, casefold=config.lc, stripmark=config.del_diac, stripcc=true, compat=true)
        isspace(u) && (u = BLANK)
        config.del_punc && ispunct(u) && !(u in ('@', '#', '_')) && (u = BLANK)
        config.group_emo && isemoji(u, config.emojis) && (u = '👾')
        rep = u === output[end] ? rep + 1 : 0
        config.del_dup && rep > 1 && continue
        push!(output, u)
    end

    limits && push!(output, BLANK)
    output
end
