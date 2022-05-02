var documenterSearchIndex = {"docs":
[{"location":"api/","page":"API","title":"API","text":"\nCurrentModule = TextSearch\nDocTestSetup = quote\n    using TextSearch\nend","category":"page"},{"location":"api/#TextSearch-API","page":"API","title":"TextSearch API","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"Modules = [TextSearch]\nOrder   = [:function, :type]","category":"page"},{"location":"api/#Base.:*-Union{Tuple{Tv}, Tuple{Ti}, Tuple{Dict{Ti, Tv}, Dict{Ti, Tv}}} where {Ti, Tv<:Real}","page":"API","title":"Base.:*","text":"*(a::DVEC{Ti,Tv}, b::DVEC{Ti,Tv}) where {Ti,Tv<:Real}\n*(a::DVEC{K, V}, b::F) where K where {V<:Real} where {F<:Real}\n\nComputes the element-wise product of a and b\n\n\n\n\n\n","category":"method"},{"location":"api/#Base.:+-Union{Tuple{Tv}, Tuple{Ti}, Tuple{Dict{Ti, Tv}, Dict{Ti, Tv}}} where {Ti, Tv<:Real}","page":"API","title":"Base.:+","text":"+(a::DVEC{Ti,Tv}, b::DVEC{Ti,Tv}) where {Ti,Tv<:Real}\n+(a::DVEC, b::Pair)\n\nComputes the sum of a and b\n\n\n\n\n\n","category":"method"},{"location":"api/#Base.:--Union{Tuple{Tv}, Tuple{Ti}, Tuple{Dict{Ti, Tv}, Dict{Ti, Tv}}} where {Ti, Tv<:Real}","page":"API","title":"Base.:-","text":"-(a::DVEC{Ti,Tv}, b::DVEC{Ti,Tv}) where {Ti,Tv<:Real}\n\nSubstracts of b of a\n\n\n\n\n\n","category":"method"},{"location":"api/#Base.:/-Union{Tuple{K}, Tuple{V}, Tuple{F}, Tuple{Dict{K, V}, F}} where {F<:Real, V<:Real, K}","page":"API","title":"Base.:/","text":"/(a::DVEC{K, V}, b::F) where K where {V<:Real} where {F<:Real}\n\nComputes the element-wise division of a and b\n\n\n\n\n\n","category":"method"},{"location":"api/#Base.sum-Tuple{AbstractVector{<:Dict{Ti, Tv} where {Ti, Tv<:Number}}}","page":"API","title":"Base.sum","text":"Base.sum(col::AbstractVector{<:DVEC})\n\nComputes the sum of the given list of vectors\n\n\n\n\n\n","category":"method"},{"location":"api/#Base.zero-Union{Tuple{Type{Dict{Ti, Tv}}}, Tuple{Tv}, Tuple{Ti}} where {Ti, Tv}","page":"API","title":"Base.zero","text":"zero(::Type{DVEC{Ti,Tv}}) where {Ti,Tv}\n\nCreates an empty DVEC vector\n\n\n\n\n\n","category":"method"},{"location":"api/#Distances.evaluate-Tuple{AngleDistance, Dict{Ti, Tv} where {Ti, Tv<:Number}, Dict{Ti, Tv} where {Ti, Tv<:Number}}","page":"API","title":"Distances.evaluate","text":"evaluate(::AngleDistance, a::DVEC, b::DVEC)::Float64\n\nComputes the angle between two DVEC sparse vectors\n\n\n\n\n\n","category":"method"},{"location":"api/#Distances.evaluate-Tuple{CosineDistance, Dict{Ti, Tv} where {Ti, Tv<:Number}, Dict{Ti, Tv} where {Ti, Tv<:Number}}","page":"API","title":"Distances.evaluate","text":"evaluate(::CosineDistance, a::DVEC, b::DVEC)::Float64\n\nComputes the cosine distance between two DVEC sparse vectors\n\n\n\n\n\n","category":"method"},{"location":"api/#Distances.evaluate-Tuple{NormalizedAngleDistance, Dict{Ti, Tv} where {Ti, Tv<:Number}, Dict{Ti, Tv} where {Ti, Tv<:Number}}","page":"API","title":"Distances.evaluate","text":"evaluate(::NormalizedAngleDistance, a::DVEC, b::DVEC)::Float64\n\nComputes the angle  between two DVEC sparse vectors\n\nIt supposes that all bags are normalized (see normalize! function)\n\n\n\n\n\n","category":"method"},{"location":"api/#Distances.evaluate-Tuple{NormalizedCosineDistance, Dict{Ti, Tv} where {Ti, Tv<:Number}, Dict{Ti, Tv} where {Ti, Tv<:Number}}","page":"API","title":"Distances.evaluate","text":"evaluate(::NormalizedCosineDistance, a::DVEC, b::DVEC)::Float64\n\nComputes the cosine distance between two DVEC sparse vectors\n\nIt supposes that bags are normalized (see normalize! function)\n\n\n\n\n\n","category":"method"},{"location":"api/#LinearAlgebra.dot-Tuple{Dict{Ti, Tv} where {Ti, Tv<:Number}, Dict{Ti, Tv} where {Ti, Tv<:Number}}","page":"API","title":"LinearAlgebra.dot","text":"dot(a::DVEC, b::DVEC)::Float64\n\nComputes the dot product for two DVEC vectors\n\n\n\n\n\n","category":"method"},{"location":"api/#LinearAlgebra.norm-Tuple{Dict{Ti, Tv} where {Ti, Tv<:Number}}","page":"API","title":"LinearAlgebra.norm","text":"norm(a::DVEC)::Float64\n\nComputes a normalized DVEC vector\n\n\n\n\n\n","category":"method"},{"location":"api/#LinearAlgebra.normalize!-Union{Tuple{Dict{Ti, Tv}}, Tuple{Tv}, Tuple{Ti}} where {Ti, Tv<:AbstractFloat}","page":"API","title":"LinearAlgebra.normalize!","text":"normalize!(bow::DVEC)\n\nInplace normalization of bow\n\n\n\n\n\n","category":"method"},{"location":"api/#SparseArrays.sparse-Union{Tuple{AbstractVector{S}}, Tuple{S}, Tuple{Tv}, Tuple{Ti}, Tuple{AbstractVector{S}, Any}} where {Ti<:Integer, Tv<:Number, S<:Dict{Ti, Tv}}","page":"API","title":"SparseArrays.sparse","text":"sparse(cols::AbstractVector{S}, m=0) where S<:DVEC{Ti,Tv} where {Ti<:Integer,Tv<:Number}\n\nCreates a sparse matrix from an array of DVEC sparse vectors.\n\n\n\n\n\n","category":"method"},{"location":"api/#SparseArrays.sparsevec-Union{Tuple{Dict{Ti, Tv}}, Tuple{Tv}, Tuple{Ti}, Tuple{Dict{Ti, Tv}, Any}} where {Ti<:Integer, Tv<:Number}","page":"API","title":"SparseArrays.sparsevec","text":"sparsevec(vec::DVEC{Ti,Tv}, m=0) where {Ti<:Integer,Tv<:Number}\n\nCreates a sparse vector from a DVEC sparse vector\n\n\n\n\n\n","category":"method"},{"location":"api/#TextSearch.add!-Union{Tuple{Tv}, Tuple{Ti}, Tuple{Dict{Ti, Tv}, Dict{Ti, Tv}}} where {Ti, Tv<:Real}","page":"API","title":"TextSearch.add!","text":"add!(a::DVEC{Ti,Tv}, b::DVEC{Ti,Tv}) where {Ti,Tv<:Real}\nadd!(a::DVEC{Ti,Tv}, b::AbstractSparseArray) where {Ti,Tv<:Real}\nadd!(a::DVEC{Ti,Tv}, b::Pair{Ti,Tv}) where {Ti,Tv<:Real}\n\nUpdates a to the sum of a+b\n\n\n\n\n\n","category":"method"},{"location":"api/#TextSearch.centroid-Tuple{AbstractVector{<:Dict{Ti, Tv} where {Ti, Tv<:Number}}}","page":"API","title":"TextSearch.centroid","text":"centroid(cluster::AbstractVector{<:DVEC})\n\nComputes a centroid of the given list of DVEC vectors\n\n\n\n\n\n","category":"method"},{"location":"api/#TextSearch.dvec-Tuple{SparseArrays.AbstractSparseVector}","page":"API","title":"TextSearch.dvec","text":"dvec(x::AbstractSparseVector)\n\nConverts an sparse vector into a DVEC sparse vector\n\n\n\n\n\n","category":"method"},{"location":"api/#TextSearch.flush_token!-Tuple{Tokenizer}","page":"API","title":"TextSearch.flush_token!","text":"flush_token!(tok::Tokenizer)\n\nPushes the word inside buffer into token list; it discards empty strings.\n\n\n\n\n\n","category":"method"},{"location":"api/#TextSearch.normalize_text-Tuple{TextConfig, AbstractString, Vector{Char}}","page":"API","title":"TextSearch.normalize_text","text":"normalize_text(config::TextConfig, text::AbstractString, output::Vector{Char})\n\nNormalizes a given text using the specified transformations of config\n\n\n\n\n\n","category":"method"},{"location":"api/#TextSearch.nwords-Tuple{Tokenizer, Integer}","page":"API","title":"TextSearch.nwords","text":"nwords(tok::Tokenizer, q::Integer)\n\n\n\n\n\n","category":"method"},{"location":"api/#TextSearch.prune-Tuple{VectorModel, AbstractFloat}","page":"API","title":"TextSearch.prune","text":"prune(model::VectorModel, lowerweight::AbstractFloat)\n\nCreates a new vector model without terms with smaller global weights than lowerweight.\n\n\n\n\n\n","category":"method"},{"location":"api/#TextSearch.prune_select_top-Tuple{VectorModel, Integer}","page":"API","title":"TextSearch.prune_select_top","text":"prune_select_top(model::VectorModel, k::Integer)\nprune_select_top(model::VectorModel, ratio::AbstractFloat)\n\nCreates a new vector model with the best k tokens from model based on global weighting. ratio is a floating point between 0 and 1 indicating the ratio of the vocabulary to be kept\n\n\n\n\n\n","category":"method"},{"location":"api/#TextSearch.qgrams-Tuple{Tokenizer, Integer}","page":"API","title":"TextSearch.qgrams","text":"qgrams(tok::Tokenizer, q::Integer)\n\nComputes character q-grams for the given input\n\n\n\n\n\n","category":"method"},{"location":"api/#TextSearch.skipgrams-Tuple{Tokenizer, Skipgram}","page":"API","title":"TextSearch.skipgrams","text":"skipgrams(tok::Tokenizer, q::Skipgram)\n\nTokenizes using skipgrams\n\n\n\n\n\n","category":"method"},{"location":"api/#TextSearch.tokenize-Tuple{Tokenizer, AbstractString}","page":"API","title":"TextSearch.tokenize","text":"tokenize(tok::Tokenizer, text::AbstractString)\n\nTokenizes text using the given configuration\n\n\n\n\n\n","category":"method"},{"location":"api/#TextSearch.tokenize_corpus-Tuple{Tokenizer, Any}","page":"API","title":"TextSearch.tokenize_corpus","text":"tokenize_corpus(tok::Tokenizer, arr)\n\nTokenize a list of texts.\n\n\n\n\n\n","category":"method"},{"location":"api/#TextSearch.unigrams-Tuple{Tokenizer}","page":"API","title":"TextSearch.unigrams","text":"unigrams(tok::Tokenizer)\n\nPerforms the word tokenization\n\n\n\n\n\n","category":"method"},{"location":"api/#TextSearch.vectorize","page":"API","title":"TextSearch.vectorize","text":"vectorize(messages::AbstractVector, bow::BOW=BOW())\n\nComputes a bag of words from messages\n\n\n\n\n\n","category":"function"},{"location":"api/#TextSearch.vectorize-2","page":"API","title":"TextSearch.vectorize","text":"vectorize(voc::Vocabulary, tokenlist::AbstractVector, bow::BOW=BOW())\nvectorize(voc::Vocabulary, tok::Tokenizer, text::AbstractString, bow::BOW=BOW())\n\nCreates a bag of words from the given text (a string or a list of strings). If bow is given then updates the bag with the text. When config is given, the text is parsed according to it.\n\n\n\n\n\n","category":"function"},{"location":"api/#TextSearch.vectorize-Union{Tuple{_L}, Tuple{_G}, Tuple{VectorModel{_G, _L}, Dict{UInt32, Int32}}} where {_G, _L}","page":"API","title":"TextSearch.vectorize","text":"vectorize(model::VectorModel, bow::BOW; normalize=true, mindocs=model.mindocs, minweight=1e-6) where Tv<:Real\n\nComputes a weighted vector using the given bag of words and the specified weighting scheme.\n\n\n\n\n\n","category":"method"},{"location":"api/#TextSearch.vectorize_corpus","page":"API","title":"TextSearch.vectorize_corpus","text":"vectorize_corpus(tok::Tokenizer, corpus::AbstractVector)\n\nComputes a list of bag of words from a corpus\n\n\n\n\n\n","category":"function"},{"location":"api/#TextSearch.BinaryGlobalWeighting","page":"API","title":"TextSearch.BinaryGlobalWeighting","text":"BinaryGlobalWeighting()\n\nThe weight is 1 for known tokens, 0 for out of vocabulary tokens\n\n\n\n\n\n","category":"type"},{"location":"api/#TextSearch.BinaryLocalWeighting","page":"API","title":"TextSearch.BinaryLocalWeighting","text":"BinaryLocalWeighting()\n\nThe weight is 1 for known tokens, 0 for out of vocabulary tokens\n\n\n\n\n\n","category":"type"},{"location":"api/#TextSearch.EntropyWeighting","page":"API","title":"TextSearch.EntropyWeighting","text":"EntropyWeighting(; smooth=0.0, lowerweight=0.0, weights=:balance)\n\nEntropy weighting uses the empirical entropy of the vocabulary along classes to produce a notion of importance for each token\n\n\n\n\n\n","category":"type"},{"location":"api/#TextSearch.FreqWeighting","page":"API","title":"TextSearch.FreqWeighting","text":"FreqWeighting()\n\nFrequency weighting\n\n\n\n\n\n","category":"type"},{"location":"api/#TextSearch.GlobalWeighting","page":"API","title":"TextSearch.GlobalWeighting","text":"GlobalWeighting\n\nAbstract type for global weighting\n\n\n\n\n\n","category":"type"},{"location":"api/#TextSearch.IdfWeighting","page":"API","title":"TextSearch.IdfWeighting","text":"IdfWeighting()\n\nInverse document frequency weighting\n\n\n\n\n\n","category":"type"},{"location":"api/#TextSearch.LocalWeighting","page":"API","title":"TextSearch.LocalWeighting","text":"LocalWeighting\n\nAbstract type for local weighting\n\n\n\n\n\n","category":"type"},{"location":"api/#TextSearch.Skipgram","page":"API","title":"TextSearch.Skipgram","text":"Skipgram(qsize, skip)\n\nA skipgram is a kind of tokenization where qsize words having skip separation are used as a single token.\n\n\n\n\n\n","category":"type"},{"location":"api/#TextSearch.TextConfig","page":"API","title":"TextSearch.TextConfig","text":"TextConfig(;\n    del_diac::Bool=true,\n    del_dup::Bool=false,\n    del_punc::Bool=false,\n    group_num::Bool=true,\n    group_url::Bool=true,\n    group_usr::Bool=false,\n    group_emo::Bool=false,\n    lc::Bool=true,\n    qlist::Vector=Int8[],\n    nlist::Vector=Int8[],\n    slist::Vector{Skipgram}=Skipgram[]\n)\n\nDefines a preprocessing and tokenization pipeline\n\ndel_diac: indicates if diacritic symbols should be removed\ndel_dup: indicates if duplicate contiguous symbols must be replaced for a single symbol\ndel_punc: indicates if punctuaction symbols must be removed\ngroup_num: indicates if numbers should be grouped _num\ngroup_url: indicates if urls should be grouped as _url\ngroup_usr: indicates if users (@usr) should be grouped as _usr\ngroup_emo: indicates if emojis should be grouped as _emo\nlc: indicates if the text should be normalized to lower case\nqlist: a list of character q-grams to use\nnlist: a list of words n-grams to use\nslist: a list of skip-grams tokenizers to use\n\nNote: If qlist, nlist, and slists are all empty arrays, then it defaults to nlist=[1]\n\n\n\n\n\n","category":"type"},{"location":"api/#TextSearch.TextModel","page":"API","title":"TextSearch.TextModel","text":"Model\n\nAn abstract type that represents a weighting model\n\n\n\n\n\n","category":"type"},{"location":"api/#TextSearch.TfWeighting","page":"API","title":"TextSearch.TfWeighting","text":"TfWeighting()\n\nTerm frequency weighting\n\n\n\n\n\n","category":"type"},{"location":"api/#TextSearch.Tokenizer","page":"API","title":"TextSearch.Tokenizer","text":"struct Tokenizer\n\nA tokenizer converts a text into a set of tokens, and in particular, each token in this implementation is represented as the hash code of the corresponding string token. This methods also normalize and preprocess the text following instructions in the given TextConfig object. The structure has several fields:\n\nthe text config object\nthe rest of the fields are used as buffers (multithreaded applications need independent copies of tokenizers)\n\nNote: non-thread safe, make a copy of this structure for each thread.\n\n\n\n\n\n","category":"type"},{"location":"api/#TextSearch.TpWeighting","page":"API","title":"TextSearch.TpWeighting","text":"TpWeighting()\n\nTerm probability weighting\n\n\n\n\n\n","category":"type"},{"location":"api/#TextSearch.VectorModel-Tuple{EntropyWeighting, LocalWeighting, Vocabulary, AbstractVector{Dict{UInt32, Int32}}, CategoricalArrays.CategoricalArray}","page":"API","title":"TextSearch.VectorModel","text":"VectorModel(ent::EntropyWeighting, lw::LocalWeighting, corpus::BOW, labels;\n    mindocs::Integer=1,\n    smooth::Float64=0.0,\n    weights=:balance,\n    lowerweight=0.0\n)\n\nCreates a vector model using the input corpus. \n\n\n\n\n\n","category":"method"},{"location":"api/#TextSearch.VectorModel-Tuple{GlobalWeighting, LocalWeighting, Vocabulary}","page":"API","title":"TextSearch.VectorModel","text":"VectorModel(global_weighting::GlobalWeighting, local_weighting::LocalWeighting, voc::Vocabulary; mindocs=1)\n\nCreates a vector model using the input corpus. \n\n\n\n\n\n","category":"method"},{"location":"api/#TextSearch.Vocabulary-Tuple{Any}","page":"API","title":"TextSearch.Vocabulary","text":"Vocabulary(corpus)\n\nComputes a vocabulary from an already tokenized corpus\n\n\n\n\n\n","category":"method"},{"location":"api/#TextSearch.Vocabulary-Tuple{Integer}","page":"API","title":"TextSearch.Vocabulary","text":"Vocabulary(n::Integer)\n\nCreates a Vocabulary struct\n\n\n\n\n\n","category":"method"},{"location":"api/#TextSearch.Vocabulary-Tuple{Tokenizer, Any}","page":"API","title":"TextSearch.Vocabulary","text":"Vocabulary(tok, corpus)\n\nComputes a vocabulary from a corpus using the tokenizer tok\n\n\n\n\n\n","category":"method"},{"location":"searching/","page":"Searching","title":"Searching","text":"\nCurrentModule = TextSearch\nDocTestSetup = quote\n    using TextSearch\nend","category":"page"},{"location":"searching/","page":"Searching","title":"Searching","text":"This package focus on solving similarity search queries for text collections.  The general flow is:","category":"page"},{"location":"searching/","page":"Searching","title":"Searching","text":"proprocess and normalize text rightarrow tokenize rightarrow vectorize rightarrow index rightarrow solve queries","category":"page"},{"location":"searching/","page":"Searching","title":"Searching","text":"First of all, we need a collection of documents. We will retrieve a small corpus of anonymized tweets; each tweet contains an emoji that is used as a label, in this example, we put that emoji in to the text for dispaying","category":"page"},{"location":"searching/","page":"Searching","title":"Searching","text":"using TextSearch, CodecZlib, JSON3\nfilename = \"emo50k.json.gz\"\n!isfile(filename) && download(\"https://github.com/sadit/TextClassificationTutorial/raw/main/data/emo50k.json.gz\", filename)\nfunction gettext(line)\n    t = JSON3.read(line, Dict)\n    replace(t[\"text\"], \"_emo\" => t[\"klass\"])\nend\n\ncorpus = open(filename) do f\n    [gettext(line) for line in eachline(GzipDecompressorStream(f))]\nend;","category":"page"},{"location":"searching/","page":"Searching","title":"Searching","text":"Now, we need to define the preprocessing step and the tokenizer","category":"page"},{"location":"searching/","page":"Searching","title":"Searching","text":"tok = Tokenizer(TextConfig(group_emo=false, group_num=false, group_url=false, group_usr=false, nlist=[1]))","category":"page"},{"location":"searching/","page":"Searching","title":"Searching","text":"We need to create a model for the text, we select a typical vector model. The model constructor needs to know the weighthing scheme and some stats about the corpus' vocabulary:","category":"page"},{"location":"searching/","page":"Searching","title":"Searching","text":"model = VectorModel(IdfWeighting(), TfWeighting(), tok, corpus)","category":"page"},{"location":"searching/","page":"Searching","title":"Searching","text":"This model is used to vectorize the corpus, and then, create the Inverted Index search structure.","category":"page"},{"location":"searching/","page":"Searching","title":"Searching","text":"invindex = InvIndex(vocsize(model))\nappend!(VectorDatabase(vectorize_corpus(model, tok, corpus)))","category":"page"},{"location":"searching/","page":"Searching","title":"Searching","text":"The index can be used for solving queries efficiently","category":"page"},{"location":"searching/","page":"Searching","title":"Searching","text":"q = \"que buena música!!\"\nfor p in search(invindex, vectorize(model, tok, q), 5)\n    @info (dist=p.dist, tweet=corpus[p.id])\nend","category":"page"},{"location":"preprocessing/","page":"Preprocessing","title":"Preprocessing","text":"\nCurrentModule = TextSearch\nDocTestSetup = quote\n    using TextSearch\nend","category":"page"},{"location":"preprocessing/","page":"Preprocessing","title":"Preprocessing","text":"This package focus on solving similarity search queries for text collections. The first stage is to modelate the collection of textual documents, such that each document converts in a vector, and therefore, the collection of vectors create an structure that can be efficiently lookup to solve queries.","category":"page"},{"location":"preprocessing/#Preprocessing-and-tokenizing-text","page":"Preprocessing","title":"Preprocessing and tokenizing text","text":"","category":"section"},{"location":"preprocessing/","page":"Preprocessing","title":"Preprocessing","text":"The processesing and tokenization is defined through a text configuration TextConfig.","category":"page"},{"location":"preprocessing/","page":"Preprocessing","title":"Preprocessing","text":"using TextSearch\ntokenize(Tokenizer(TextConfig(del_punc=true)), \"HI!! this is fun!! http://something\")\ntokenize(Tokenizer(TextConfig(nlist=[2])), \"HI!! this is fun!! http://something\")","category":"page"},{"location":"preprocessing/","page":"Preprocessing","title":"Preprocessing","text":"The configuration may be manipulated to remove, conflated or whatever is needed for a task. ","category":"page"},{"location":"modeling/","page":"Models","title":"Models","text":"\nCurrentModule = TextSearch\nDocTestSetup = quote\n    using TextSearch\nend","category":"page"},{"location":"modeling/#Modeling-the-text-collection","page":"Models","title":"Modeling the text collection","text":"","category":"section"},{"location":"modeling/","page":"Models","title":"Models","text":"The processesing and tokenization is defined through a text configuration TextConfig.","category":"page"},{"location":"modeling/","page":"Models","title":"Models","text":"using TextSearch, CodecZlib, JSON3, CategoricalArrays\nfilename = \"emo50k.json.gz\"\n!isfile(filename) && download(\"https://github.com/sadit/TextClassificationTutorial/raw/main/data/emo50k.json.gz\", filename)\n\ncorpus = String[]\nlabels = []\nopen(filename) do f\n    for line in eachline(GzipDecompressorStream(f))\n        t = JSON3.read(line, Dict)\n        push!(corpus, t[\"text\"])\n        push!(labels, t[\"klass\"])\n    end\nend\n\nlabels = categorical(labels)\ntok = Tokenizer(TextConfig(group_emo=false, group_num=false, group_url=false, group_usr=false, nlist=[1]))","category":"page"},{"location":"modeling/","page":"Models","title":"Models","text":"Once a Tokenizer is defined, we need to create a model for the text, we select a typical vector model. The model constructor needs to know the weighthing scheme and some stats about the corpus' vocabulary:","category":"page"},{"location":"modeling/","page":"Models","title":"Models","text":"model = VectorModel(IdfWeighting(), TfWeighting(), tok, corpus)","category":"page"},{"location":"modeling/","page":"Models","title":"Models","text":"Now we can vectorize a text","category":"page"},{"location":"modeling/","page":"Models","title":"Models","text":"vectorize(model, tok, \"la mejor música, la música de siempre!\"; normalize=false)","category":"page"},{"location":"modeling/","page":"Models","title":"Models","text":"Note: by default normalize=true normalizes the vector.","category":"page"},{"location":"modeling/#Entropy-models","page":"Models","title":"Entropy models","text":"","category":"section"},{"location":"modeling/","page":"Models","title":"Models","text":"TextSearch supports a family of text models based on labeled data called Entropy-based models; which use the empirical distribution of symbols along labels to determine its importance; the entropy of this distribution is used to compose the symbol weight. In this example, the variable labels is a categorical array containing emojis associated to each text in the corpus, see CategoricalArrays.jl.","category":"page"},{"location":"modeling/","page":"Models","title":"Models","text":"labels\nmodel = VectorModel(EntropyWeighting(), BinaryLocalWeighting(), tok, corpus, labels)","category":"page"},{"location":"modeling/","page":"Models","title":"Models","text":"Now we can vectorize a text","category":"page"},{"location":"modeling/","page":"Models","title":"Models","text":"vec = vectorize(tok, model, \"la mejor música, la música de siempre!\"; normalize=false)\ndecode(tok, vec)","category":"page"},{"location":"sparse/","page":"-","title":"-","text":"\nCurrentModule = TextSearch\nDocTestSetup = quote\n    using TextSearch\nend","category":"page"},{"location":"sparse/#Sparse-matrices","page":"-","title":"Sparse matrices","text":"","category":"section"},{"location":"sparse/","page":"-","title":"-","text":"Inverted indexes/files are representations of sparse matrices optimized for certain operations. We provide some functions to convert inverted files to sparse matrices.","category":"page"},{"location":"sparse/","page":"-","title":"-","text":"sparse\nsparsevec","category":"page"},{"location":"sparse/#SparseArrays.sparse","page":"-","title":"SparseArrays.sparse","text":"sparse(cols::AbstractVector{S}, m=0) where S<:DVEC{Ti,Tv} where {Ti<:Integer,Tv<:Number}\n\nCreates a sparse matrix from an array of DVEC sparse vectors.\n\n\n\n\n\n","category":"function"},{"location":"sparse/#SparseArrays.sparsevec","page":"-","title":"SparseArrays.sparsevec","text":"sparsevec(vec::DVEC{Ti,Tv}, m=0) where {Ti<:Integer,Tv<:Number}\n\nCreates a sparse vector from a DVEC sparse vector\n\n\n\n\n\n","category":"function"},{"location":"sparse/","page":"-","title":"-","text":"Inverted indexes constructors also support sparse matrices as input (wrapped on MatrixDatabase structs)","category":"page"},{"location":"sparse/#Dictionary-based-sparse-vectors","page":"-","title":"Dictionary-based sparse vectors","text":"","category":"section"},{"location":"sparse/","page":"-","title":"-","text":"Some application domains could take advantage of hash based sparse vectors, like text search and classification. Therefore, this package also provide a partial implementation of sparse vectors using Dict.","category":"page"},{"location":"sparse/","page":"-","title":"-","text":"dvec\nDVEC\nSVEC\nSVEC32\nSVEC64\nnnz\nfindmax\nargmax\nmaximum\nfindmin\nargmin\nminimum\nnormalize!\ndot\nnorm\nzero\nadd!\nsum\n+\n-\n*\n/\ncentroid\nevaluate\nNormalizedAngleDistance\nNormalizedCosineDistance\nAngleDistance\nCosineDistance\nevaluate","category":"page"},{"location":"sparse/#TextSearch.dvec","page":"-","title":"TextSearch.dvec","text":"dvec(x::AbstractSparseVector)\n\nConverts an sparse vector into a DVEC sparse vector\n\n\n\n\n\n","category":"function"},{"location":"sparse/#LinearAlgebra.normalize!","page":"-","title":"LinearAlgebra.normalize!","text":"normalize!(bow::DVEC)\n\nInplace normalization of bow\n\n\n\n\n\n","category":"function"},{"location":"sparse/#LinearAlgebra.dot","page":"-","title":"LinearAlgebra.dot","text":"dot(a::DVEC, b::DVEC)::Float64\n\nComputes the dot product for two DVEC vectors\n\n\n\n\n\n","category":"function"},{"location":"sparse/#LinearAlgebra.norm","page":"-","title":"LinearAlgebra.norm","text":"norm(a::DVEC)::Float64\n\nComputes a normalized DVEC vector\n\n\n\n\n\n","category":"function"},{"location":"sparse/#Base.zero","page":"-","title":"Base.zero","text":"zero(::Type{DVEC{Ti,Tv}}) where {Ti,Tv}\n\nCreates an empty DVEC vector\n\n\n\n\n\n","category":"function"},{"location":"sparse/#TextSearch.add!","page":"-","title":"TextSearch.add!","text":"add!(a::DVEC{Ti,Tv}, b::DVEC{Ti,Tv}) where {Ti,Tv<:Real}\nadd!(a::DVEC{Ti,Tv}, b::AbstractSparseArray) where {Ti,Tv<:Real}\nadd!(a::DVEC{Ti,Tv}, b::Pair{Ti,Tv}) where {Ti,Tv<:Real}\n\nUpdates a to the sum of a+b\n\n\n\n\n\n","category":"function"},{"location":"sparse/#Base.sum","page":"-","title":"Base.sum","text":"Base.sum(col::AbstractVector{<:DVEC})\n\nComputes the sum of the given list of vectors\n\n\n\n\n\n","category":"function"},{"location":"sparse/#Base.:+","page":"-","title":"Base.:+","text":"+(a::DVEC{Ti,Tv}, b::DVEC{Ti,Tv}) where {Ti,Tv<:Real}\n+(a::DVEC, b::Pair)\n\nComputes the sum of a and b\n\n\n\n\n\n","category":"function"},{"location":"sparse/#Base.:-","page":"-","title":"Base.:-","text":"-(a::DVEC{Ti,Tv}, b::DVEC{Ti,Tv}) where {Ti,Tv<:Real}\n\nSubstracts of b of a\n\n\n\n\n\n","category":"function"},{"location":"sparse/#Base.:*","page":"-","title":"Base.:*","text":"*(a::DVEC{Ti,Tv}, b::DVEC{Ti,Tv}) where {Ti,Tv<:Real}\n*(a::DVEC{K, V}, b::F) where K where {V<:Real} where {F<:Real}\n\nComputes the element-wise product of a and b\n\n\n\n\n\n","category":"function"},{"location":"sparse/#Base.:/","page":"-","title":"Base.:/","text":"/(a::DVEC{K, V}, b::F) where K where {V<:Real} where {F<:Real}\n\nComputes the element-wise division of a and b\n\n\n\n\n\n","category":"function"},{"location":"sparse/#TextSearch.centroid","page":"-","title":"TextSearch.centroid","text":"centroid(cluster::AbstractVector{<:DVEC})\n\nComputes a centroid of the given list of DVEC vectors\n\n\n\n\n\n","category":"function"},{"location":"sparse/#Distances.evaluate","page":"-","title":"Distances.evaluate","text":"evaluate(::NormalizedCosineDistance, a::DVEC, b::DVEC)::Float64\n\nComputes the cosine distance between two DVEC sparse vectors\n\nIt supposes that bags are normalized (see normalize! function)\n\n\n\n\n\nevaluate(::CosineDistance, a::DVEC, b::DVEC)::Float64\n\nComputes the cosine distance between two DVEC sparse vectors\n\n\n\n\n\nevaluate(::NormalizedAngleDistance, a::DVEC, b::DVEC)::Float64\n\nComputes the angle  between two DVEC sparse vectors\n\nIt supposes that all bags are normalized (see normalize! function)\n\n\n\n\n\nevaluate(::AngleDistance, a::DVEC, b::DVEC)::Float64\n\nComputes the angle between two DVEC sparse vectors\n\n\n\n\n\n","category":"function"},{"location":"sparse/","page":"-","title":"-","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = TextSearch","category":"page"},{"location":"#TextSearch","page":"Home","title":"TextSearch","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"TextSearch.jl is a package to create vector representations of text, mostly, independently of the language. It is intended to be used with SimilaritySearch.jl, but can be used independetly if needed. TextSearch.jl was renamed from TextModel.jl to reflect its capabilities and mission.","category":"page"},{"location":"","page":"Home","title":"Home","text":"For generic text analysis you should use other packages like TextAnalysis.jl.","category":"page"},{"location":"","page":"Home","title":"Home","text":"It supports a number of simple text preprocessing functions, and three different kinds of tokenizers, i.e., word n-grams, character q-grams, and skip-grams. It supports creating multisets of tokens, commonly named bag of words (BOW). TextSearch.jl can produce sparse vector representations based on term-weighting schemes like TF, IDF, and TFIDF. It also supports term-weighting schemes designed to cope text classification tasks, mostly based on distributional representations.","category":"page"},{"location":"#Installing","page":"Home","title":"Installing","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"You may install the package as follows","category":"page"},{"location":"","page":"Home","title":"Home","text":"] add TextSearch","category":"page"},{"location":"","page":"Home","title":"Home","text":"also, you can run the set of tests as follows","category":"page"},{"location":"","page":"Home","title":"Home","text":"] test TextSearch","category":"page"}]
}
