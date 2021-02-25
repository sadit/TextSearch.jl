```@meta

CurrentModule = TextSearch
DocTestSetup = quote
    using TextSearch
end
```


# Modeling the text collection

The processesing and tokenization is defined through a text configuration [`TextConfig`](@ref).

```@setup Model

using TextSearch, CodecZlib, JSON3
filename = "emo50k.json.gz"
!isfile(filename) && download("https://github.com/sadit/TextClassificationTutorial/raw/main/data/emo50k.json.gz", filename)
function gettext(line)
    t = JSON3.read(line, Dict)
    replace(t["text"], "_emo" => t["klass"])
end

corpus = open(filename) do f
    [gettext(line) for line in eachline(GzipDecompressorStream(f))]
end;

config = TextConfig(group_emo=false, group_num=false, group_url=false, group_usr=false, nlist=[1])
```

Once a `TextConfig` is define, we need to create 

We need to create a model for the text, we select a typical vector model. The model constructor needs to know the weighthing scheme and some stats about the corpus' vocabulary:
```@repl Model
corpus_bow = compute_bow(config, corpus)
modelfreq = VectorModel(FreqWeighting(), corpus_bow)
modeltf = VectorModel(TfWeighting(), corpus_bow)
modelidf = VectorModel(IdfWeighting(), corpus_bow)
modeltfidf = VectorModel(TfidfWeighting(), corpus_bow)
```

Now we can vectorize a text
```@repl Model
text = "las mejor música, la música de siempre!"
b = compute_bow(config, text)
vectorize(modelfreq, b; normalize=false)
vectorize(modeltf, b; normalize=false)
vectorize(modelidf, b; normalize=false)
vectorize(modeltfidf, b; normalize=false)
```

Note: typically, you may to set `normalize=true` to allow the vector normalization.

