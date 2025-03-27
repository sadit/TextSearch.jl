
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://sadit.github.io/TextSearch.jl/dev)
[![Build Status](https://github.com/sadit/TextSearch.jl/workflows/CI/badge.svg)](https://github.com/sadit/TextSearch.jl/actions)
[![Coverage](https://codecov.io/gh/sadit/TextSearch.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/sadit/TextSearch.jl)

# TextSearch

`TextSearch.jl` is a package to create vector representations of text, mostly, independently of the language. It is intended to be used with [SimilaritySearch.jl](https://github.com/sadit/SimilaritySearch.jl), but can be used independetly if needed.
`TextSearch.jl` was renamed from `TextModel.jl` to reflect its capabilities and mission.

For generic text analysis you should use other packages like [TextAnalysis.jl](https://github.com/johnmyleswhite/TextAnalysis.jl).

It supports a number of simple text preprocessing functions, and three different kinds of tokenizers, i.e., word n-grams, character q-grams, and skip-grams. It supports creating multisets of tokens, commonly named bag of words (BOW).
`TextSearch.jl` can produce sparse vector representations based on term-weighting schemes like TF, IDF, and TFIDF. It also supports term-weighting schemes designed to cope text classification tasks, mostly based on distributional representations.

# Installing 

You may install the package as follows
```julia
] add TextSearch
```
also, you can run the set of tests as follows
```julia
] test TextSearch
```

## Using the library

The directory [examples](https://github.com/sadit/TextSearch.jl/tree/master/src) contains a few examples of how to use it, based on [Pluto.jl](https://github.com/fonsp/Pluto.jl)


After cloning the repository, you must intantiate the directory. 

```julia
using Pkg
pkg"instantiate"
```

once you instantiated your environment, just run Pluto notebook and explore the examples
```julia
using Pluto
Pluto.run()
```

## What is new in 0.19
Basically, this is a breaking release since it supports SimilaritySearch v0.12 which has some API changes; indexing and search functions now require an InvertedFileContext object
