using Documenter, TextSearch

makedocs(;
    modules=[TextSearch],
    authors="Eric S. Tellez",
    repo="https://github.com/sadit/TextSearch.jl/blob/{commit}{path}#L{line}",
    sitename="TextSearch.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", nothing) == "true",
        canonical="https://sadit.github.io/TextSearch.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "API" => "api.md"
    ],
    warnonly=true
)

deploydocs(;
    repo="github.com/sadit/TextSearch.jl",
    devbranch=nothing,
    branch = "gh-pages",
    versions = ["stable" => "v^", "v#.#.#", "dev" => "dev"]
)
