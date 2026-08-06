# This file is a part of TextSearch.jl

export NormalizationConfig

"""
    NormalizationConfig(;
        del_diac::Bool=true,
        del_dup::Bool=false,
        del_punc::Bool=false,
        group_num::Bool=true,
        group_url::Bool=true,
        group_usr::Bool=false,
        group_emo::Bool=false,
        lc::Bool=true,
        re_user::Regex=DEFAULT_RE_USER,
        re_url::Regex=DEFAULT_RE_URL,
        re_num::Regex=DEFAULT_RE_NUM,
        emojis::Set{Char}=DEFAULT_EMOJIS
    )

Defines the text normalization stage of a [`TextConfig`](@ref) (see its `normalization`
field): utf8 normalization, character removal, whitespace normalization, casing, etc.
Consumed by [`normalize_text`](@ref).

- `del_diac`: indicates if diacritic symbols should be removed
- `del_dup`: indicates if duplicate contiguous symbols must be replaced for a single symbol
- `del_punc`: indicates if punctuaction symbols must be removed
- `group_num`: indicates if numbers should be grouped _num
- `group_url`: indicates if urls should be grouped as _url
- `group_usr`: indicates if users (@usr) should be grouped as _usr
- `group_emo`: indicates if emojis should be grouped as _emo
- `lc`: indicates if the text should be normalized to lower case
- `re_user`, `re_url`, `re_num`: the regexes used to detect `@user` mentions, URLs, and
  numbers when their corresponding `group_*` flag is set (see [`normalize_text`](@ref)).
  Override them to customize detection (e.g. for a different language or domain).
- `emojis`: the set of emoji characters grouped when `group_emo` is set (see [`isemoji`](@ref)).

# Example

```julia
julia> buff = Char[];

julia> normalize_text(TextConfig(normalization=NormalizationConfig()), "Café", buff);

julia> String(buff)
" cafe "
```
"""
Base.@kwdef struct NormalizationConfig
    del_diac::Bool  = true
    del_dup::Bool   = false
    del_punc::Bool  = false
    group_num::Bool = true
    group_url::Bool = true
    group_usr::Bool = false
    group_emo::Bool = false
    lc::Bool        = true
    re_user::Regex = DEFAULT_RE_USER
    re_url::Regex  = DEFAULT_RE_URL
    re_num::Regex  = DEFAULT_RE_NUM
    emojis::Set{Char} = DEFAULT_EMOJIS
end

function NormalizationConfig(c::NormalizationConfig;
        del_diac::Bool=c.del_diac,
        del_dup::Bool=c.del_dup,
        del_punc::Bool=c.del_punc,
        group_num::Bool=c.group_num,
        group_url::Bool=c.group_url,
        group_usr::Bool=c.group_usr,
        group_emo::Bool=c.group_emo,
        lc::Bool=c.lc,
        re_user::Regex=c.re_user,
        re_url::Regex=c.re_url,
        re_num::Regex=c.re_num,
        emojis::Set{Char}=c.emojis
    )
    NormalizationConfig(; del_diac, del_dup, del_punc, group_num, group_url, group_usr, group_emo, lc, re_user, re_url, re_num, emojis)
end

function Base.show(io::IO, c::NormalizationConfig; prefix="", indent="  ")
    println(io, prefix, "NormalizationConfig:")
    prefix = indent * prefix
    for f in fieldnames(NormalizationConfig)
        print(io, prefix, indent)
        _show_field(io, f, getfield(c, f))
    end
end
