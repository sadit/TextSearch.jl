```@meta

CurrentModule = TextSearch
DocTestSetup = quote
    using TextSearch
end
```


# Modeling the text collection

The processesing and tokenization is defined through a text configuration [`TextConfig`](@ref).

```@setup Model
using TextSearch, CodecZlib, JSON3, CategoricalArrays
filename = "emo50k.json.gz"
!isfile(filename) && download("https://github.com/sadit/TextClassificationTutorial/raw/main/data/emo50k.json.gz", filename)

corpus = String[]
labels = []
open(filename) do f
    for line in eachline(GzipDecompressorStream(f))
        t = JSON3.read(line, Dict)
        push!(corpus, t["text"])
        push!(labels, t["klass"])
    end
end

labels = categorical(labels)
tok = Tokenizer(TextConfig(group_emo=false, group_num=false, group_url=false, group_usr=false, nlist=[1]))
```

Once a `Tokenizer` is defined, we need to create a model for the text, we select a typical vector model. The model constructor needs to know the weighthing scheme and some stats about the corpus' vocabulary:
```@repl Model
model = VectorModel(IdfWeighting(), TfWeighting(), compute_bow_corpus(tok, corpus))
```

Now we can vectorize a text
```@repl Model
vectorize(model, tok, "la mejor música, la música de siempre!"; normalize=false)
```

Note: by default `normalize=true` normalizes the vector.


## Entropy models

`TextSearch` supports a family of text models based on labeled data called Entropy-based models; which use the empirical distribution of symbols along labels to determine its importance; the entropy of this distribution is used to compose the symbol weight. In this example, the variable `labels` is a categorical array containing emojis associated to each text in the corpus, see [`CategoricalArrays.jl`](https://github.com/JuliaData/CategoricalArrays.jl).

```@repl Model
labels
vcorpus = compute_bow_corpus(tok, corpus)
model = VectorModel(EntropyWeighting(), BinaryLocalWeighting(), vcorpus, labels)
```

Now we can vectorize a text
```@repl Model
vec = vectorize(tok, model, "la mejor música, la música de siempre!"; normalize=false)
decode(tok, vec)
```

# Inspecting models
Models have a `id2token` dictionary to map identifiers to symbols
```@repl Model
model.tokens
```