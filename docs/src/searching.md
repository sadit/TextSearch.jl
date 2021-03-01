```@meta

CurrentModule = TextSearch
DocTestSetup = quote
    using TextSearch
end
```
This package focus on solving similarity search queries for text collections. 
The general flow is:

 _proprocess and normalize text_ ``\rightarrow`` _tokenize_ ``\rightarrow`` _vectorize_ ``\rightarrow`` index ``\rightarrow`` solve queries


First of all, we need a collection of documents. We will retrieve a small corpus of anonymized tweets; each tweet contains an emoji that is used as a label, in this example, we put that emoji in to the text for dispaying

```@repl Search
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
```

Now, we need to define the preprocessing step and tokenizer

```@repl Search
config = TextConfig(group_emo=false, group_num=false, group_url=false, group_usr=false, nlist=[1])
```

We need to create a model for the text, we select a typical vector model. The model constructor needs to know the weighthing scheme and some stats about the corpus' vocabulary:
```@repl Search
model = VectorModel(TfWeighting, IdfWeighting(), compute_bow(config, corpus))
```

This model is used to vectorize the corpus, and then, create the Inverted Index search structure.
```@repl Search
invindex = InvIndex(vectorize.(model, compute_bow.(config, corpus)))
```

The index can be used for solving queries efficiently
```@repl Search
q = "que buena música"
for p in search(invindex, vectorize(model, compute_bow(config, q)), 5)
    @info (dist=p.dist, tweet=corpus[p.id])
end
```
