using TextSearch
using Documenter

makedocs(;
    modules=[TextSearch],
    authors="Eric S. Tellez",
    repo="https://github.com/sadit/TextSearch.jl/blob/{commit}{path}#L{line}",
    sitename="TextSearch.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://sadit.github.io/TextSearch.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Searching" => "searching.md",
        "Preprocessing" => "preprocessing.md",
        "Models" => "modeling.md",
        "API" => "api.md"
    ],
)

deploydocs(;
    repo="github.com/sadit/TextSearch.jl",
    devbranch="main",
    branch = "gh-pages",
    versions = ["stable" => "v^", "v#.#.#"]
)
