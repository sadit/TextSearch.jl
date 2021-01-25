# TextSearch.jl

[![Build Status](https://travis-ci.org/sadit/TextSearch.jl.svg?branch=master)](https://travis-ci.org/sadit/TextSearch.jl)
[![Coverage Status](https://coveralls.io/repos/github/sadit/TextSearch.jl/badge.svg?branch=master)](https://coveralls.io/github/sadit/TextSearch.jl?branch=master)
[![codecov](https://codecov.io/gh/sadit/TextSearch.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/sadit/TextSearch.jl)


`TextSearch.jl` is a package to create vector representations of text, mostly, independently of the language. It is intended to be used with [SimilaritySearch.jl](https://github.com/sadit/SimilaritySearch.jl), but can be used independetly if needed.
`TextSearch.jl` was renamed from `TextModel.jl` to reflect its capabilities and mission.

For generic text analysis you should use other packages like [TextAnalysis.jl](https://github.com/johnmyleswhite/TextAnalysis.jl).

It supports a number of simple text preprocessing functions, and three different kinds of tokenizers, i.e., word n-grams, character q-grams, and skip-grams. It supports creating multisets of tokens, commonly named bag of words (BOW).
`TextSearch.jl` can produce sparse vector representations based on term-weighting schemes like TF, IDF, and TFIDF. It also supports term-weighting schemes designed to cope text classification tasks, mostly based on distributional representations.

# Installing SimilaritySearch

You may install the package as follows
```bash
julia -e 'using Pkg; pkg"add TextSearch.jl"'
```
also, you can run the set of tests as follows
```bash
julia -e 'using Pkg; pkg"test TextSearch"'
```

## Using the library

The directory [examples](https://github.com/sadit/TextSearch.jl/tree/master/src) contains a few examples of how to use it, based on [Pluto.jl](https://github.com/fonsp/Pluto.jl)


After cloning the repository, you must intantiate the directory. 

```julia
using Pkg
pkg"instantiate"
```

once you instantiated your environment, just run Pluto notebook and explore the exampless.
```julia
using Pluto
Pluto.run()
```

