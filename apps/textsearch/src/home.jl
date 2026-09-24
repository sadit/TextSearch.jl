"""
    textsearch_home() -> String

The `textsearch` app's home directory, `~/.textsearch/` by default; override with the
`TEXTSEARCH_HOME` environment variable (used by tests to sandbox away from the real
user home directory).
"""
textsearch_home() = get(ENV, "TEXTSEARCH_HOME", joinpath(homedir(), ".textsearch"))

"""
    profiles_dir() -> String

`textsearch_home()`'s `profiles/` subdirectory, created if missing.
"""
profiles_dir() = mkpath(joinpath(textsearch_home(), "profiles"))

"""
    default_nickname(path::AbstractString) -> String

Derives a nickname from a profile zip's filename: strips the directory and the trailing
`.zip` extension. E.g. `"/tmp/foo.zip"` -> `"foo"`.
"""
default_nickname(path::AbstractString) = first(splitext(basename(path)))

"""
    profile_path(nickname::AbstractString) -> String

The installed path an installed profile named `nickname` would live at (whether or not
it actually exists yet).
"""
profile_path(nickname::AbstractString) = joinpath(profiles_dir(), nickname * ".zip")

"""
    list_nicknames() -> Vector{String}

Nicknames of every installed profile, sorted alphabetically.
"""
list_nicknames() = sort([first(splitext(f)) for f in readdir(profiles_dir()) if endswith(f, ".zip")])

"""
    lsi_dir() -> String

`textsearch_home()`'s `lsi/` subdirectory, created if missing. LSI projections live apart from
`profiles/` because they are not profiles: `list` would otherwise show each one as a nickname
that `load_profile` refuses.
"""
lsi_dir() = mkpath(joinpath(textsearch_home(), "lsi"))

"""
    lsi_path(nickname::AbstractString) -> String

Where the LSI projection of installed profile `nickname` lives (whether or not it exists).
"""
lsi_path(nickname::AbstractString) = joinpath(lsi_dir(), nickname * ".zip")

"""
    list_lsi_nicknames() -> Vector{String}

Nicknames of the profiles that have an LSI projection installed, sorted alphabetically.
"""
list_lsi_nicknames() = sort([first(splitext(f)) for f in readdir(lsi_dir()) if endswith(f, ".zip")])

"""
    lsi_binding(nickname) -> Union{Nothing,NamedTuple}

`nothing` when no LSI is installed for `nickname`; otherwise its [`lsi_summary`](@ref) plus
`bound`, whether it names the installed profile of that nickname (`false` also when the
profile itself is not installed).
"""
function lsi_binding(nickname::AbstractString)
    path = lsi_path(nickname)
    isfile(path) || return nothing
    s = lsi_summary(path)
    ppath = profile_path(nickname)
    bound = isfile(ppath) && s.profile_id == profile_id(load_profile(ppath))
    merge(s, (; path, bound))
end
