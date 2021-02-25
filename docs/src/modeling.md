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
config = TextConfig(group_emo=false, group_num=false, group_url=false, group_usr=false, nlist=[1])
```

Once a `TextConfig` is define, we need to create 

We need to create a model for the text, we select a typical vector model. The model constructor needs to know the weighthing scheme and some stats about the corpus' vocabulary:
```@repl Model
corpus_bow = compute_bow(config, corpus)
modeltp = VectorModel(TpWeighting(), corpus_bow)
modelfreq = VectorModel(FreqWeighting(), corpus_bow)
modeltf = VectorModel(TfWeighting(), corpus_bow)
modelidf = VectorModel(IdfWeighting(), corpus_bow)
modeltfidf = VectorModel(TfidfWeighting(), corpus_bow)
```

Now we can vectorize a text
```@repl Model
text = "las mejor música, la música de siempre!"
b = compute_bow(config, text)
vectorize(modeltp, b; normalize=false)
vectorize(modelfreq, b; normalize=false)
vectorize(modeltf, b; normalize=false)
vectorize(modelidf, b; normalize=false)
vectorize(modeltfidf, b; normalize=false)
```

Note: typically, you may to set `normalize=true` to allow the vector normalization.


## Entropy models

`TextSearch` supports a family of text models based on labeled data called Entropy-based models; which use the empirical distribution of symbols along labels to determine its importance; the entropy of this distribution is used to compose the symbol weight. In this example, the variable `labels` is a categorical array containing emojis associated to each text in the corpus, see [`CategoricalArrays.jl`](https://github.com/JuliaData/CategoricalArrays.jl).

```@repl Model
labels
vcorpus = compute_bow.(config, corpus)
modelent = EntModel(EntWeighting(), vcorpus, labels)
modelenttp = EntModel(EntTpWeighting(), vcorpus, labels)
modelenttf = EntModel(EntTfWeighting(), vcorpus, labels)
```

Now we can vectorize a text
```@repl Model
text = "las mejor música, la música de siempre!"
b = compute_bow(config, text)
vectorize(modelent, b; normalize=false)
vectorize(modelenttp, b; normalize=false)
vec = vectorize(modelenttf, b; normalize=false)

```

# Inspecting models
Models have a `id2token` dictionary to map identifiers to symbols
```@repl Model
Dict(modelenttf.id2token[k] => v for (k,v) in vec)
```