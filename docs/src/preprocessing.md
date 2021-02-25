```@meta

CurrentModule = TextSearch
DocTestSetup = quote
    using TextSearch
end
```

This package focus on solving similarity search queries for text collections. The first stage is to modelate the collection of textual documents, such that each document converts in a vector, and therefore, the collection of vectors create an structure that can be efficiently lookup to solve queries.


## Preprocessing and tokenizing text

The processesing and tokenization is defined through a text configuration [`TextConfig`](@ref).

```@repl Preprocessing
using TextSearch

compute_bow(TextConfig(nlist=[1]), "HI!! this is fun!! http://something")
compute_bow(TextConfig(del_punc=true), "HI!! this is fun!! http://something")
compute_bow(TextConfig(nlist=[2]), "HI!! this is fun!! http://something")
```

`TextSearch.jl` relies on the concept of bag of words, implemented as a dictionary of `Symbol => Int` (with a convenient type alias of `BOW`). Please observe that it is possible to indicate combination of tokenizers.

This set may be manipulated to remove, conflated or whatever is needed for a task. Internally `compute_bow` preprocess and tokenizes the text and then creates the bag of words.

```@repl Preprocessing

tokens = tokenize(TextConfig(), "HI!! this is fun!! http://something")
compute_bow(tokens)
```
