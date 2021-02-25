var documenterSearchIndex = {"docs":
[{"location":"searching/","page":"Searching","title":"Searching","text":"\nCurrentModule = TextSearch\nDocTestSetup = quote\n    using TextSearch\nend","category":"page"},{"location":"searching/","page":"Searching","title":"Searching","text":"This package focus on solving similarity search queries for text collections.  The general flow is:","category":"page"},{"location":"searching/","page":"Searching","title":"Searching","text":"proprocess and normalize text rightarrow tokenize rightarrow vectorize rightarrow index rightarrow solve queries","category":"page"},{"location":"searching/","page":"Searching","title":"Searching","text":"First of all, we need a collection of documents. We will retrieve a small corpus of anonymized tweets; each tweet contains an emoji that is used as a label, in this example, we put that emoji in to the text for dispaying","category":"page"},{"location":"searching/","page":"Searching","title":"Searching","text":"using TextSearch, CodecZlib, JSON3\nfilename = \"emo50k.json.gz\"\n!isfile(filename) && download(\"https://github.com/sadit/TextClassificationTutorial/raw/main/data/emo50k.json.gz\", filename)\nfunction gettext(line)\n    t = JSON3.read(line, Dict)\n    replace(t[\"text\"], \"_emo\" => t[\"klass\"])\nend\n\ncorpus = open(filename) do f\n    [gettext(line) for line in eachline(GzipDecompressorStream(f))]\nend;","category":"page"},{"location":"searching/","page":"Searching","title":"Searching","text":"Now, we need to define the preprocessing step and tokenizer","category":"page"},{"location":"searching/","page":"Searching","title":"Searching","text":"config = TextConfig(group_emo=false, group_num=false, group_url=false, group_usr=false, nlist=[1])","category":"page"},{"location":"searching/","page":"Searching","title":"Searching","text":"We need to create a model for the text, we select a typical vector model. The model constructor needs to know the weighthing scheme and some stats about the corpus' vocabulary:","category":"page"},{"location":"searching/","page":"Searching","title":"Searching","text":"model = VectorModel(TfidfWeighting(), compute_bow(config, corpus))","category":"page"},{"location":"searching/","page":"Searching","title":"Searching","text":"This model is used to vectorize the corpus, and then, create the Inverted Index search structure.","category":"page"},{"location":"searching/","page":"Searching","title":"Searching","text":"invindex = InvIndex(vectorize.(model, compute_bow.(config, corpus)))","category":"page"},{"location":"searching/","page":"Searching","title":"Searching","text":"The index can be used for solving queries efficiently","category":"page"},{"location":"searching/","page":"Searching","title":"Searching","text":"q = \"que buena música\"\nfor p in search(invindex, vectorize(model, compute_bow(config, q)), 5)\n    @info (dist=p.dist, tweet=corpus[p.id])\nend","category":"page"},{"location":"apimodels/","page":"Weighting schemes","title":"Weighting schemes","text":"\nCurrentModule = TextSearch\nDocTestSetup = quote\n    using TextSearch\nend","category":"page"},{"location":"apimodels/#VectorModel","page":"Weighting schemes","title":"VectorModel","text":"","category":"section"},{"location":"apimodels/","page":"Weighting schemes","title":"Weighting schemes","text":"\nIdFreq\nVectorModel\nvectorize(::VectorModel, a, b)\nprune(::VectorModel, a, b)\nprune_select_top(::VectorModel, f)","category":"page"},{"location":"apimodels/#TextSearch.IdFreq","page":"Weighting schemes","title":"TextSearch.IdFreq","text":"IdFreq(id, freq)\n\nStores a document identifier and its frequency\n\n\n\n\n\n","category":"type"},{"location":"apimodels/#TextSearch.VectorModel","page":"Weighting schemes","title":"TextSearch.VectorModel","text":"VectorModel(weighting::WeightingType, corpus::BOW; minocc::Integer=1)\n\nTrains a vector model using the input corpus. \n\n\n\n\n\n","category":"type"},{"location":"apimodels/#Weighting-methods-for-VectorModel","page":"Weighting schemes","title":"Weighting methods for VectorModel","text":"","category":"section"},{"location":"apimodels/","page":"Weighting schemes","title":"Weighting schemes","text":"TfWeighting\nIdfWeighting\nTfidfWeighting\nFreqWeighthing","category":"page"},{"location":"apimodels/#TextSearch.TfWeighting","page":"Weighting schemes","title":"TextSearch.TfWeighting","text":"TfWeighting()\n\nTerm frequency weighting\n\n\n\n\n\n","category":"type"},{"location":"apimodels/#TextSearch.IdfWeighting","page":"Weighting schemes","title":"TextSearch.IdfWeighting","text":"IdfWeighting()\n\nInverse document frequency weighting\n\n\n\n\n\n","category":"type"},{"location":"apimodels/#TextSearch.TfidfWeighting","page":"Weighting schemes","title":"TextSearch.TfidfWeighting","text":"TfidfWeighting()\n\nTFIDF weighting\n\n\n\n\n\n","category":"type"},{"location":"apimodels/#EntModel","page":"Weighting schemes","title":"EntModel","text":"","category":"section"},{"location":"apimodels/","page":"Weighting schemes","title":"Weighting schemes","text":"\nEntModel\nvectorize(::EntModel, a, b)\nprune(::EntModel, a, b)\nprune_select_top(::EntModel, f)","category":"page"},{"location":"apimodels/#TextSearch.EntModel","page":"Weighting schemes","title":"TextSearch.EntModel","text":"EntModel(model::DistModel, weighting::WeightingType; lower=0.0)\nEntModel(config::TextConfig, weighting::WeightingType, corpus, y; smooth=3, minocc=1, weights=:balance, lower=0.0, nclasses=0)\n\nFits an EntModel using the already fitted DistModel. It accepts only symbols with a final weight higher or equal than lower. Parameters:     - corpus is text collection     - y the set of associated labels (one-to-one with corpus)     - smooth is a smoothing factor for the histogram.     - weights accepts a list of weights (one per class) to be applied to the histogram     - lower controls the minimum weight to be accepted     - nclasses specifies the number of classes \t- minocc: minimum population to consider a token (without considering the smoothing factor).\n\n\n\n\n\n","category":"type"},{"location":"apimodels/#Weighting-methods-for-EntModel","page":"Weighting schemes","title":"Weighting methods for EntModel","text":"","category":"section"},{"location":"apimodels/","page":"Weighting schemes","title":"Weighting schemes","text":"EntTfWeighting\nEntTpWeighting\nEntWeighting","category":"page"},{"location":"apitextconfig/","page":"Preprocessing","title":"Preprocessing","text":"\nCurrentModule = TextSearch\nDocTestSetup = quote\n    using TextSearch\nend","category":"page"},{"location":"apitextconfig/","page":"Preprocessing","title":"Preprocessing","text":"TextConfig\ntokenize\nnormalize_text\nSkipgram\ncompute_bow","category":"page"},{"location":"apitextconfig/#TextSearch.TextConfig","page":"Preprocessing","title":"TextSearch.TextConfig","text":"TextConfig(;\n    del_diac::Bool=true,\n    del_dup::Bool=false,\n    del_punc::Bool=false,\n    group_num::Bool=true,\n    group_url::Bool=true,\n    group_usr::Bool=false,\n    group_emo::Bool=false,\n    lc::Bool=true,\n    qlist::Vector=Int8[],\n    nlist::Vector=Int8[1],\n    slist::Vector{Skipgram}=Skipgram[]\n)\n\nDefines a preprocessing and tokenization pipeline\n\ndel_diac: indicates if diacritic symbols should be removed\ndel_dup: indicates if duplicate contiguous symbols must be replaced for a single symbol\ndel_punc: indicates if punctuaction symbols must be removed\ngroup_num: indicates if numbers should be grouped _num\ngroup_url: indicates if urls should be grouped as _url\ngroup_usr: indicates if users (@usr) should be grouped as _usr\ngroup_emo: indicates if emojis should be grouped as _emo\nlc: indicates if the text should be normalized to lower case\nqlist: a list of character q-grams to use\nnlist: a list of words n-grams to use\nslist: a list of skip-grams tokenizers to use\n\nNote that if at least one tokenizer must be specified and that if nlist and slist are not empty, unigrams are also forced if they are not included in the list\n\n\n\n\n\n","category":"type"},{"location":"apitextconfig/#TextSearch.tokenize","page":"Preprocessing","title":"TextSearch.tokenize","text":"tokenize(config::TextConfig, text::AbstractString)\n\nTokenizes text using the given configuration\n\n\n\n\n\n","category":"function"},{"location":"apitextconfig/#TextSearch.normalize_text","page":"Preprocessing","title":"TextSearch.normalize_text","text":"normalize_text(config::TextConfig, text::AbstractString, L::Vector{Char}=Vector{Char}())::Vector{Char}\n\nNormalizes a given text using the specified transformations of config\n\n\n\n\n\n","category":"function"},{"location":"apitextconfig/#TextSearch.Skipgram","page":"Preprocessing","title":"TextSearch.Skipgram","text":"Skipgram(qsize, skip)\n\nA skipgram is a kind of tokenization where qsize words having skip separation are used as a single token.\n\n\n\n\n\n","category":"type"},{"location":"apitextconfig/#TextSearch.compute_bow","page":"Preprocessing","title":"TextSearch.compute_bow","text":"compute_bow(tokenlist::AbstractVector{S}, bow::BOW=BOW()) where {S<:AbstractString}\ncompute_bow(config::TextConfig, text::AbstractString, bow::BOW=BOW())\ncompute_bow(config::TextConfig, text::AbstractVector, bow::BOW=BOW())\n\nCreates a bag of words from the given text (a string or a list of strings). If bow is given then updates the bag with the text. When config is given, the text is parsed according to it.\n\n\n\n\n\n","category":"function"},{"location":"modeling/","page":"Models","title":"Models","text":"\nCurrentModule = TextSearch\nDocTestSetup = quote\n    using TextSearch\nend","category":"page"},{"location":"modeling/#Modeling-the-text-collection","page":"Models","title":"Modeling the text collection","text":"","category":"section"},{"location":"modeling/","page":"Models","title":"Models","text":"The processesing and tokenization is defined through a text configuration TextConfig.","category":"page"},{"location":"modeling/","page":"Models","title":"Models","text":"\nusing TextSearch, CodecZlib, JSON3\nfilename = \"emo50k.json.gz\"\n!isfile(filename) && download(\"https://github.com/sadit/TextClassificationTutorial/raw/main/data/emo50k.json.gz\", filename)\nfunction gettext(line)\n    t = JSON3.read(line, Dict)\n    replace(t[\"text\"], \"_emo\" => t[\"klass\"])\nend\n\ncorpus = open(filename) do f\n    [gettext(line) for line in eachline(GzipDecompressorStream(f))]\nend;\n\nconfig = TextConfig(group_emo=false, group_num=false, group_url=false, group_usr=false, nlist=[1])","category":"page"},{"location":"modeling/","page":"Models","title":"Models","text":"Once a TextConfig is define, we need to create ","category":"page"},{"location":"modeling/","page":"Models","title":"Models","text":"We need to create a model for the text, we select a typical vector model. The model constructor needs to know the weighthing scheme and some stats about the corpus' vocabulary:","category":"page"},{"location":"modeling/","page":"Models","title":"Models","text":"corpus_bow = compute_bow(config, corpus)\nmodelfreq = VectorModel(FreqWeighting(), corpus_bow)\nmodeltf = VectorModel(TfWeighting(), corpus_bow)\nmodelidf = VectorModel(IdfWeighting(), corpus_bow)\nmodeltfidf = VectorModel(TfidfWeighting(), corpus_bow)","category":"page"},{"location":"modeling/","page":"Models","title":"Models","text":"Now we can vectorize a text","category":"page"},{"location":"modeling/","page":"Models","title":"Models","text":"text = \"las mejor música, la música de siempre!\"\nb = compute_bow(config, text)\nvectorize(modelfreq, b; normalize=false)\nvectorize(modeltf, b; normalize=false)\nvectorize(modelidf, b; normalize=false)\nvectorize(modeltfidf, b; normalize=false)","category":"page"},{"location":"modeling/","page":"Models","title":"Models","text":"Note: typically, you may to set normalize=true to allow the vector normalization.","category":"page"},{"location":"preprocessing/","page":"Preprocessing","title":"Preprocessing","text":"\nCurrentModule = TextSearch\nDocTestSetup = quote\n    using TextSearch\nend","category":"page"},{"location":"preprocessing/","page":"Preprocessing","title":"Preprocessing","text":"This package focus on solving similarity search queries for text collections. The first stage is to modelate the collection of textual documents, such that each document converts in a vector, and therefore, the collection of vectors create an structure that can be efficiently lookup to solve queries.","category":"page"},{"location":"preprocessing/#Preprocessing-and-tokenizing-text","page":"Preprocessing","title":"Preprocessing and tokenizing text","text":"","category":"section"},{"location":"preprocessing/","page":"Preprocessing","title":"Preprocessing","text":"The processesing and tokenization is defined through a text configuration TextConfig.","category":"page"},{"location":"preprocessing/","page":"Preprocessing","title":"Preprocessing","text":"using TextSearch\n\ncompute_bow(TextConfig(nlist=[1]), \"HI!! this is fun!! http://something\")\ncompute_bow(TextConfig(del_punc=true), \"HI!! this is fun!! http://something\")\ncompute_bow(TextConfig(nlist=[2]), \"HI!! this is fun!! http://something\")","category":"page"},{"location":"preprocessing/","page":"Preprocessing","title":"Preprocessing","text":"TextSearch.jl relies on the concept of bag of words, implemented as a dictionary of Symbol => Int (with a convenient type alias of BOW). Please observe that it is possible to indicate combination of tokenizers.","category":"page"},{"location":"preprocessing/","page":"Preprocessing","title":"Preprocessing","text":"This set may be manipulated to remove, conflated or whatever is needed for a task. Internally compute_bow preprocess and tokenizes the text and then creates the bag of words.","category":"page"},{"location":"preprocessing/","page":"Preprocessing","title":"Preprocessing","text":"\ntokens = tokenize(TextConfig(), \"HI!! this is fun!! http://something\")\ncompute_bow(tokens)","category":"page"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = TextSearch","category":"page"},{"location":"#TextSearch","page":"Home","title":"TextSearch","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"TextSearch.jl is a package to create vector representations of text, mostly, independently of the language. It is intended to be used with SimilaritySearch.jl, but can be used independetly if needed. TextSearch.jl was renamed from TextModel.jl to reflect its capabilities and mission.","category":"page"},{"location":"","page":"Home","title":"Home","text":"For generic text analysis you should use other packages like TextAnalysis.jl.","category":"page"},{"location":"","page":"Home","title":"Home","text":"It supports a number of simple text preprocessing functions, and three different kinds of tokenizers, i.e., word n-grams, character q-grams, and skip-grams. It supports creating multisets of tokens, commonly named bag of words (BOW). TextSearch.jl can produce sparse vector representations based on term-weighting schemes like TF, IDF, and TFIDF. It also supports term-weighting schemes designed to cope text classification tasks, mostly based on distributional representations.","category":"page"},{"location":"#Installing","page":"Home","title":"Installing","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"You may install the package as follows","category":"page"},{"location":"","page":"Home","title":"Home","text":"] add TextSearch","category":"page"},{"location":"","page":"Home","title":"Home","text":"also, you can run the set of tests as follows","category":"page"},{"location":"","page":"Home","title":"Home","text":"] test TextSearch","category":"page"}]
}
