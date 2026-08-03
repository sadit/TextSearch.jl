# This file is a part of TextSearch.jl

export isemoji

# A symbol s in this list will be expanded to BLANK*s if the predecesor of s is neither s nor BLANK
# On changes from s to BLANK or [^s] it will produce also produce an extra BLANK
# Note that enabled del_punc will delete all these symbols without any of the previous expansions
const BLANK = ' '

"""
    DEFAULT_EMOJIS::Set{Char}

The built-in emoji set (loaded from `emojis.txt`), used as [`TextConfig`](@ref)'s default
`emojis` field. Pass a different `Set{Char}` to `TextConfig(; emojis=...)` to override it.
"""
const DEFAULT_EMOJIS = Set([l[1] for l in readlines(joinpath(@__DIR__, "emojis.txt"))])

"""
    DEFAULT_RE_USER::Regex

The built-in `@user`-mention regex, used as [`TextConfig`](@ref)'s default `re_user`
field. Pass a different `Regex` to `TextConfig(; re_user=...)` to override it.
"""
const DEFAULT_RE_USER = r"""@[^;:,.@#&\\\-\"'/:\*\(\)\[\]\¿\?\¡\!\{\}~\<\>\|\s]+"""

"""
    DEFAULT_RE_URL::Regex

The built-in URL regex, used as [`TextConfig`](@ref)'s default `re_url` field. Pass a
different `Regex` to `TextConfig(; re_url=...)` to override it.
"""
const DEFAULT_RE_URL = r"(http|ftp|https)://\S+"

"""
    DEFAULT_RE_NUM::Regex

The built-in number regex, used as [`TextConfig`](@ref)'s default `re_num` field. Pass a
different `Regex` to `TextConfig(; re_num=...)` to override it.
"""
const DEFAULT_RE_NUM = r"[-+]?(\d+\.?\d*)|(\.\d+)"

"""
    isemoji(c::Char, emojis::Set{Char}=DEFAULT_EMOJIS)::Bool

Tests whether `c` is one of the emoji characters in `emojis` (by default,
[`DEFAULT_EMOJIS`](@ref), the set known to TextSearch, loaded from `emojis.txt`). Used
by [`normalize_text`](@ref) when `TextConfig`'s `group_emo` option is set.

# Example

```julia
julia> isemoji('😀')
true

julia> isemoji('a')
false
```
"""
isemoji(c::Char, emojis::Set{Char}=DEFAULT_EMOJIS) = c in emojis
