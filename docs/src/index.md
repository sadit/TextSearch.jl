```@meta
CurrentModule = TextSearch
```

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
