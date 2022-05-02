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
tokenize(Tokenizer(TextConfig(del_punc=true)), "HI!! this is fun!! http://something")
tokenize(Tokenizer(TextConfig(nlist=[2])), "HI!! this is fun!! http://something")
```


The configuration may be manipulated to remove, conflated or whatever is needed for a task. 
