# TextModel.jl

[![Build Status](https://travis-ci.org/sadit/TextModel.jl.svg?branch=master)](https://travis-ci.org/sadit/TextModel.jl)
[![Coverage Status](https://coveralls.io/repos/github/sadit/TextModel.jl/badge.svg?branch=master)](https://coveralls.io/github/sadit/TextModel.jl?branch=master)
[![codecov.io](http://codecov.io/github/sadit/TextModel.jl/coverage.svg?branch=master)](http://codecov.io/github/sadit/TextModel.jl?branch=master)


TextModel.jl is a package to create vector representations of text, mostly, independently of the language. It is intended to be used with [SimilaritySearch.jl](https://github.com/sadit/SimilaritySearch.jl), but can be used independetly if needed.

For generic text analysis you should use other packages like [TextAnalysis.jl](https://github.com/johnmyleswhite/TextAnalysis.jl).

It supports a number of simple text preprocessing functions, and three different kinds of tokenizers, i.e., word n-grams, character q-grams, and skip-grams. It supports creating multisets of tokens, commonly named bag of words (BOW). TextModel.jl can produce sparse vector representations based on term-weighting schemes like TF, IDF, and TFIDF. It also supports term-weighting schemes designed to cope text classification tasks, mostly based on distributional representations.
