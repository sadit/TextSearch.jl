using Test, TextSearch, SimilaritySearch

# Runs the README's examples.
#
# This exists because the README rotted without anyone noticing: the TextConfig
# reorganization landed ten days before v1.0.0 was tagged, the README was not updated, and
# the release shipped with examples that could not run (issue #31). The test suite was green
# the whole time, because nothing executed the README.
#
# Every fenced `julia` block that begins with `using TextSearch` is extracted and evaluated,
# each in its own module so blocks cannot lean on each other's state. Blocks that are not
# runnable Julia -- the `] add TextSearch` pkg-REPL snippets -- are skipped by that same rule
# rather than by a marker someone has to remember to add.
#
# Each runnable block must also declare the version it targets, as a `# TextSearch vX.Y`
# first line, and that version must match `Project.toml`. A reader needs to know which
# version an example was written for, and a marker nobody checks would drift exactly the way
# the examples themselves did.

"""
    readme_julia_blocks(path) -> Vector{Tuple{Int,String}}

Extracts fenced ```julia blocks from `path`, returning `(starting_line, code)` for the ones
that open with `using TextSearch` -- i.e. the self-contained, runnable examples.
"""
function readme_julia_blocks(path::AbstractString)
    blocks = Tuple{Int,String}[]
    inblock = false
    startline = 0
    buf = String[]

    for (i, line) in enumerate(eachline(path))
        if !inblock && startswith(line, "```julia")
            inblock = true
            startline = i
            empty!(buf)
        elseif inblock && startswith(line, "```")
            inblock = false
            code = join(buf, "\n")
            occursin(r"^\s*using TextSearch"m, code) && push!(blocks, (startline, code))
        elseif inblock
            push!(buf, line)
        end
    end

    blocks
end

"""
    declared_version() -> String

The `major.minor` of `Project.toml`'s version, which is what README examples must declare.
Read with a regex rather than the TOML stdlib to avoid adding a test dependency for one line.
"""
function declared_version()
    txt = read(joinpath(@__DIR__, "..", "Project.toml"), String)
    m = match(r"^version\s*=\s*\"(\d+)\.(\d+)"m, txt)
    m === nothing && error("could not read the package version from Project.toml")
    "$(m[1]).$(m[2])"
end

@testset "README examples run" begin
    path = joinpath(@__DIR__, "..", "README.md")
    @test isfile(path)
    version = declared_version()

    blocks = readme_julia_blocks(path)
    # a README with no runnable example would pass vacuously, which is the failure mode this
    # test exists to prevent
    @test length(blocks) >= 2

    for (line, code) in blocks
        @testset "README.md:$line" begin
            # the version marker travels with copy-pasted code, so it goes inside the block
            @test occursin(Regex("^# TextSearch v" * version * "\\s*\$", "m"), code)

            mod = Module(Symbol("ReadmeBlock", line))
            # `include_string` so a syntax error is reported against the README's own text
            @test (Base.include_string(mod, code, "README.md:$line"); true)
        end
    end
end
