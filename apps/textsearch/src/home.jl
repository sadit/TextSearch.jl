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
