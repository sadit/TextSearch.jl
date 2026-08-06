# This file is a part of TextSearch.jl

export normalize_text
using Base.Unicode

function _preprocessing(config::TextConfig, text)
    norm = config.normalization

    if norm.lc
        text = lowercase(text)
    end

    if norm.group_url
        text = replace(text, norm.re_url => "_url ")
    end

    if norm.group_usr
        text = replace(text, norm.re_user => "_usr ")
    end

    if norm.group_num
        text = replace(text, norm.re_num => "0 ")
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
    norm = config.normalization
    text = _preprocessing(config, text)
    limits && push!(output, BLANK)
    rep = 0

    @inbounds for u in Unicode.normalize(text, casefold=norm.lc, stripmark=norm.del_diac, stripcc=true, compat=true)
        isspace(u) && (u = BLANK)
        norm.del_punc && ispunct(u) && !(u in ('@', '#', '_')) && (u = BLANK)
        norm.group_emo && isemoji(u, norm.emojis) && (u = '👾')
        rep = u === output[end] ? rep + 1 : 0
        norm.del_dup && rep > 1 && continue
        push!(output, u)
    end

    limits && push!(output, BLANK)
    output
end
