var documenterSearchIndex = {"docs":
[{"location":"api/","page":"API","title":"API","text":"\nCurrentModule = TextSearch\nDocTestSetup = quote\n    using TextSearch\nend","category":"page"},{"location":"api/#TextSearch-API","page":"API","title":"TextSearch API","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"Modules = [TextSearch]\nOrder   = [:function, :type]","category":"page"},{"location":"api/#Base.:*-Union{Tuple{Tv}, Tuple{Ti}, Tuple{Dict{Ti, Tv}, Dict{Ti, Tv}}} where {Ti, Tv<:Real}","page":"API","title":"Base.:*","text":"*(a::DVEC{Ti,Tv}, b::DVEC{Ti,Tv}) where {Ti,Tv<:Real}\n*(a::DVEC{K, V}, b::F) where K where {V<:Real} where {F<:Real}\n\nComputes the element-wise product of a and b\n\n\n\n\n\n","category":"method"},{"location":"api/#Base.:+-Union{Tuple{Tv}, Tuple{Ti}, Tuple{Dict{Ti, Tv}, Dict{Ti, Tv}}} where {Ti, Tv<:Real}","page":"API","title":"Base.:+","text":"+(a::DVEC{Ti,Tv}, b::DVEC{Ti,Tv}) where {Ti,Tv<:Real}\n+(a::DVEC, b::Pair)\n\nComputes the sum of a and b\n\n\n\n\n\n","category":"method"},{"location":"api/#Base.:--Union{Tuple{Tv}, Tuple{Ti}, Tuple{Dict{Ti, Tv}, Dict{Ti, Tv}}} where {Ti, Tv<:Real}","page":"API","title":"Base.:-","text":"-(a::DVEC{Ti,Tv}, b::DVEC{Ti,Tv}) where {Ti,Tv<:Real}\n\nSubstracts of b of a\n\n\n\n\n\n","category":"method"},{"location":"api/#Base.:/-Union{Tuple{K}, Tuple{V}, Tuple{F}, Tuple{Dict{K, V}, F}} where {F<:Real, V<:Real, K}","page":"API","title":"Base.:/","text":"/(a::DVEC{K, V}, b::F) where K where {V<:Real} where {F<:Real}\n\nComputes the element-wise division of a and b\n\n\n\n\n\n","category":"method"},{"location":"api/#Base.sum-Tuple{AbstractVector{var\"#s5\"} where var\"#s5\"<:(Dict{Ti, Tv} where {Ti, Tv<:Number})}","page":"API","title":"Base.sum","text":"Base.sum(col::AbstractVector{<:DVEC})\n\nComputes the sum of the given list of vectors\n\n\n\n\n\n","category":"method"},{"location":"api/#Base.zero-Union{Tuple{Type{Dict{Ti, Tv}}}, Tuple{Tv}, Tuple{Ti}} where {Ti, Tv}","page":"API","title":"Base.zero","text":"zero(::Type{DVEC{Ti,Tv}}) where {Ti,Tv}\n\nCreates an empty DVEC vector\n\n\n\n\n\n","category":"method"},{"location":"api/#Distances.evaluate-Tuple{AngleDistance, Dict{Ti, Tv} where {Ti, Tv<:Number}, Dict{Ti, Tv} where {Ti, Tv<:Number}}","page":"API","title":"Distances.evaluate","text":"evaluate(::AngleDistance, a::DVEC, b::DVEC)::Float64\n\nComputes the angle between two DVEC sparse vectors\n\n\n\n\n\n","category":"method"},{"location":"api/#Distances.evaluate-Tuple{CosineDistance, Dict{Ti, Tv} where {Ti, Tv<:Number}, Dict{Ti, Tv} where {Ti, Tv<:Number}}","page":"API","title":"Distances.evaluate","text":"evaluate(::CosineDistance, a::DVEC, b::DVEC)::Float64\n\nComputes the cosine distance between two DVEC sparse vectors\n\n\n\n\n\n","category":"method"},{"location":"api/#Distances.evaluate-Tuple{NormalizedAngleDistance, Dict{Ti, Tv} where {Ti, Tv<:Number}, Dict{Ti, Tv} where {Ti, Tv<:Number}}","page":"API","title":"Distances.evaluate","text":"evaluate(::NormalizedAngleDistance, a::DVEC, b::DVEC)::Float64\n\nComputes the angle  between two DVEC sparse vectors\n\nIt supposes that all bags are normalized (see normalize! function)\n\n\n\n\n\n","category":"method"},{"location":"api/#Distances.evaluate-Tuple{NormalizedCosineDistance, Dict{Ti, Tv} where {Ti, Tv<:Number}, Dict{Ti, Tv} where {Ti, Tv<:Number}}","page":"API","title":"Distances.evaluate","text":"evaluate(::NormalizedCosineDistance, a::DVEC, b::DVEC)::Float64\n\nComputes the cosine distance between two DVEC sparse vectors\n\nIt supposes that bags are normalized (see normalize! function)\n\n\n\n\n\n","category":"method"},{"location":"api/#LinearAlgebra.dot-Tuple{Dict{Ti, Tv} where {Ti, Tv<:Number}, Dict{Ti, Tv} where {Ti, Tv<:Number}}","page":"API","title":"LinearAlgebra.dot","text":"dot(a::DVEC, b::DVEC)::Float64\n\nComputes the dot product for two DVEC vectors\n\n\n\n\n\n","category":"method"},{"location":"api/#LinearAlgebra.norm-Tuple{Dict{Ti, Tv} where {Ti, Tv<:Number}}","page":"API","title":"LinearAlgebra.norm","text":"norm(a::DVEC)::Float64\n\nComputes a normalized DVEC vector\n\n\n\n\n\n","category":"method"},{"location":"api/#LinearAlgebra.normalize!-Union{Tuple{Dict{Ti, Tv}}, Tuple{Tv}, Tuple{Ti}} where {Ti, Tv<:AbstractFloat}","page":"API","title":"LinearAlgebra.normalize!","text":"normalize!(bow::DVEC)\n\nInplace normalization of bow\n\n\n\n\n\n","category":"method"},{"location":"api/#SparseArrays.sparse-Union{Tuple{AbstractArray{Dict{Ti, Tv}, 1}}, Tuple{Tv}, Tuple{Ti}, Tuple{AbstractArray{Dict{Ti, Tv}, 1}, Any}} where {Ti<:Integer, Tv<:Number}","page":"API","title":"SparseArrays.sparse","text":"sparse(cols::AbstractVector{S}, m=0; minweight=1e-9) where S<:DVEC{Ti,Tv} where {Ti<:Integer,Tv<:Number}\n\nCreates a sparse matrix from an array of DVEC sparse vectors.\n\n\n\n\n\n","category":"method"},{"location":"api/#SparseArrays.sparsevec-Union{Tuple{Dict{Ti, Tv}}, Tuple{Tv}, Tuple{Ti}, Tuple{Dict{Ti, Tv}, Integer}} where {Ti<:Integer, Tv<:Number}","page":"API","title":"SparseArrays.sparsevec","text":"sparsevec(vec::DVEC{Ti,Tv}, m=0) where {Ti<:Integer,Tv<:Number}\n\nCreates a sparse vector from a DVEC sparse vector\n\n\n\n\n\n","category":"method"},{"location":"api/#TextSearch.add!-Union{Tuple{Tv}, Tuple{Ti}, Tuple{Dict{Ti, Tv}, Dict{Ti, Tv}}} where {Ti, Tv<:Real}","page":"API","title":"TextSearch.add!","text":"add!(a::DVEC{Ti,Tv}, b::DVEC{Ti,Tv}) where {Ti,Tv<:Real}\nadd!(a::DVEC{Ti,Tv}, b::AbstractSparseArray) where {Ti,Tv<:Real}\nadd!(a::DVEC{Ti,Tv}, b::Pair{Ti,Tv}) where {Ti,Tv<:Real}\n\nUpdates a to the sum of a+b\n\n\n\n\n\n","category":"method"},{"location":"api/#TextSearch.centroid-Tuple{AbstractVector{var\"#s2\"} where var\"#s2\"<:(Dict{Ti, Tv} where {Ti, Tv<:Number})}","page":"API","title":"TextSearch.centroid","text":"centroid(cluster::AbstractVector{<:DVEC})\n\nComputes a centroid of the given list of DVEC vectors\n\n\n\n\n\n","category":"method"},{"location":"api/#TextSearch.dvec-Tuple{SparseArrays.AbstractSparseVector{Tv, Ti} where {Tv, Ti}}","page":"API","title":"TextSearch.dvec","text":"dvec(x::AbstractSparseVector)\n\nConverts an sparse vector into a DVEC sparse vector\n\n\n\n\n\n","category":"method"},{"location":"api/#TextSearch.filter_tokens-Tuple{Function, Vocabulary}","page":"API","title":"TextSearch.filter_tokens","text":"filter_tokens(pred::Function, voc::Vocabulary)\n\nReturns a copy of reduced vocabulary based on evaluating pred function for each entry in voc\n\n\n\n\n\n","category":"method"},{"location":"api/#TextSearch.flush_nword!-Tuple{TextSearch.TextSearchBuffer, AbstractTokenTransformation}","page":"API","title":"TextSearch.flush_nword!","text":"flush_nword!(buff::TextSearchBuffer, tt::AbstractTokenTransformation)\n\nPushes the nword inside the buffer to the token list; it discards empty strings.\n\n\n\n\n\n","category":"method"},{"location":"api/#TextSearch.flush_qgram!-Tuple{TextSearch.TextSearchBuffer, AbstractTokenTransformation}","page":"API","title":"TextSearch.flush_qgram!","text":"flush_qgram!(buff::TextSearchBuffer, tt::AbstractTokenTransformation)\n\nPushes the qgram inside the buffer to the token list; it discards empty strings.\n\n\n\n\n\n","category":"method"},{"location":"api/#TextSearch.flush_skipgram!-Tuple{TextSearch.TextSearchBuffer, AbstractTokenTransformation}","page":"API","title":"TextSearch.flush_skipgram!","text":"flush_skipgram!(buff::TextSearchBuffer, tt::AbstractTokenTransformation)\n\nPushes the skipgram inside the buffer to the token list; it discards empty strings.\n\n\n\n\n\n","category":"method"},{"location":"api/#TextSearch.flush_unigram!-Tuple{TextSearch.TextSearchBuffer, AbstractTokenTransformation}","page":"API","title":"TextSearch.flush_unigram!","text":"flush_unigram!(buff::TextSearchBuffer, tt::AbstractTokenTransformation)\n\nPushes the word inside the buffer to the token list; it discards empty strings.\n\n\n\n\n\n","category":"method"},{"location":"api/#TextSearch.normalize_text-Tuple{TextConfig, AbstractString, Vector{Char}}","page":"API","title":"TextSearch.normalize_text","text":"normalize_text(config::TextConfig, text::AbstractString, output::Vector{Char})\n\nNormalizes a given text using the specified transformations of config\n\n\n\n\n\n","category":"method"},{"location":"api/#TextSearch.nwords-Tuple{Integer, TextSearch.TextSearchBuffer, AbstractTokenTransformation}","page":"API","title":"TextSearch.nwords","text":"nwords(q::Integer, buff::TextSearchBuffer, tt::AbstractTokenTransformation)\n\n\n\n\n\n","category":"method"},{"location":"api/#TextSearch.qgrams-Tuple{Integer, TextSearch.TextSearchBuffer, AbstractTokenTransformation}","page":"API","title":"TextSearch.qgrams","text":"qgrams(q::Integer, buff::TextSearchBuffer, tt::AbstractTokenTransformation)\n\nComputes character q-grams for the given input\n\n\n\n\n\n","category":"method"},{"location":"api/#TextSearch.skipgrams-Tuple{Skipgram, TextSearch.TextSearchBuffer, AbstractTokenTransformation}","page":"API","title":"TextSearch.skipgrams","text":"skipgrams(q::Skipgram, buff::TextSearchBuffer, tt::AbstractTokenTransformation)\n\nTokenizes using skipgrams\n\n\n\n\n\n","category":"method"},{"location":"api/#TextSearch.tokenize-Tuple{Function, TextConfig, AbstractString, TextSearch.TextSearchBuffer}","page":"API","title":"TextSearch.tokenize","text":"tokenize(textconfig::TextConfig, text)\ntokenize(copy_::Function, textconfig::TextConfig, text)\n\ntokenize(textconfig::TextConfig, text, buff)\ntokenize(copy_::Function, textconfig::TextConfig, text, buff)\n\nTokenizes text using the given configuration. The tokenize makes heavy usage of buffers, and when these buffers are shared it is mandatory to create a copy of the result (buff.tokens).\n\nChange the default copy function to make an additional filtering of the tokens. You can also pass the identity function to avoid copying.\n\n\n\n\n\n","category":"method"},{"location":"api/#TextSearch.tokenize_and_append!-Tuple{Vocabulary, TextConfig, Any}","page":"API","title":"TextSearch.tokenize_and_append!","text":"tokenize_and_append!(voc::Vocabulary, textconfig::TextConfig, corpus; minbatch=0, ignore_new_tokens=false)\n\nParsers each document in the given corpus and appends each token in corpus to the vocabulary.\n\n\n\n\n\n","category":"method"},{"location":"api/#TextSearch.tokenize_corpus-Tuple{Function, TextConfig, Any}","page":"API","title":"TextSearch.tokenize_corpus","text":"tokenize_corpus(textconfig::TextConfig, arr; minbatch=0)\ntokenize_corpus(copy_::Function, textconfig::TextConfig, arr; minbatch=0)\n\nTokenize a list of texts. The copy_ function is passed to tokenize as first argument.\n\n\n\n\n\n","category":"method"},{"location":"api/#TextSearch.transform_nword-Tuple{AbstractTokenTransformation, Any}","page":"API","title":"TextSearch.transform_nword","text":"transform_nword(::AbstractTokenTransformation, tok)\n\nHook applied in the tokenization stage to change the input token tok if needed. For instance, it can be used to apply stemming or any other kind of normalization. Return nothing to ignore the tok occurence (e.g., stop words).\n\n\n\n\n\n","category":"method"},{"location":"api/#TextSearch.transform_qgram-Tuple{AbstractTokenTransformation, Any}","page":"API","title":"TextSearch.transform_qgram","text":"transform_qgram(::AbstractTokenTransformation, tok)\n\nHook applied in the tokenization stage to change the input token tok if needed. For instance, it can be used to apply stemming or any other kind of normalization. Return nothing to ignore the tok occurence (e.g., stop words).\n\n\n\n\n\n","category":"method"},{"location":"api/#TextSearch.transform_skipgram-Tuple{AbstractTokenTransformation, Any}","page":"API","title":"TextSearch.transform_skipgram","text":"transform_skipgram(::AbstractTokenTransformation, tok)\n\nHook applied in the tokenization stage to change the input token tok if needed. For instance, it can be used to apply stemming or any other kind of normalization. Return nothing to ignore the tok occurence (e.g., stop words).\n\n\n\n\n\n","category":"method"},{"location":"api/#TextSearch.transform_unigram-Tuple{AbstractTokenTransformation, Any}","page":"API","title":"TextSearch.transform_unigram","text":"transform_unigram(::AbstractTokenTransformation, tok)\n\nHook applied in the tokenization stage to change the input token tok if needed. For instance, it can be used to apply stemming or any other kind of normalization. Return nothing to ignore the tok occurence (e.g., stop words).\n\n\n\n\n\n","category":"method"},{"location":"api/#TextSearch.unigrams-Tuple{TextSearch.TextSearchBuffer, AbstractTokenTransformation}","page":"API","title":"TextSearch.unigrams","text":"unigrams(buff::TextSearchBuffer, tt::AbstractTokenTransformation)\n\nPerforms the word tokenization\n\n\n\n\n\n","category":"method"},{"location":"api/#TextSearch.vectorize","page":"API","title":"TextSearch.vectorize","text":"vectorize(voc::Vocabulary, tokenlist::AbstractVector, bow=BOW())\nvectorize(voc::Vocabulary, textconfig::TextConfig, text, buff)\nvectorize(copy_::Function, voc::Vocabulary, textconfig::TextConfig, text, buff)\nvectorize(copy_::Function, voc::Vocabulary, textconfig::TextConfig, text)\nvectorize(voc::Vocabulary, textconfig::TextConfig, text)\n\nCreates a bag of words from the given text (a string or a list of strings). If bow is given then updates the bag with the text. When config is given, the text is parsed according to it.\n\n\n\n\n\n","category":"function"},{"location":"api/#TextSearch.vectorize-Tuple{Function, Vocabulary, TextConfig, AbstractVector{T} where T, Any}","page":"API","title":"TextSearch.vectorize","text":"vectorize(voc::Vocabulary, textconfig::TextConfig, messages::AbstractVector)\n\nComputes a bag of words from messages\n\n\n\n\n\n","category":"method"},{"location":"api/#TextSearch.vectorize-Union{Tuple{L_}, Tuple{G_}, Tuple{Function, VectorModel{G_, L_}, Dict{UInt32, Int32}}, Tuple{Function, VectorModel{G_, L_}, Dict{UInt32, Int32}, Dict{UInt32, Float32}}} where {G_, L_}","page":"API","title":"TextSearch.vectorize","text":"vectorize(copy_::Function, model::VectorModel, bow::BOW, vec::SVEC=SVEC(); normalize=true, minweight=1e-9) where Tv<:Real\n\nComputes a weighted vector using the given bag of words and the specified weighting scheme.\n\n\n\n\n\n","category":"method"},{"location":"api/#TextSearch.vectorize_corpus-Tuple{Function, Vocabulary, TextConfig, AbstractVector{T} where T}","page":"API","title":"TextSearch.vectorize_corpus","text":"vectorize_corpus(textconfig::TextConfig, textconfig::TextConfig, corpus::AbstractVector; minbatch=0)\nvectorize_corpus(copy_::Function, voc::Vocabulary, textconfig::TextConfig, corpus::AbstractVector; minbatch=0)\n\nComputes a list of bag of words from a corpus\n\n\n\n\n\n","category":"method"},{"location":"api/#TextSearch.BinaryGlobalWeighting","page":"API","title":"TextSearch.BinaryGlobalWeighting","text":"BinaryGlobalWeighting()\n\nThe weight is 1 for known tokens, 0 for out of vocabulary tokens\n\n\n\n\n\n","category":"type"},{"location":"api/#TextSearch.BinaryLocalWeighting","page":"API","title":"TextSearch.BinaryLocalWeighting","text":"BinaryLocalWeighting()\n\nThe weight is 1 for known tokens, 0 for out of vocabulary tokens\n\n\n\n\n\n","category":"type"},{"location":"api/#TextSearch.EntropyWeighting","page":"API","title":"TextSearch.EntropyWeighting","text":"EntropyWeighting(; smooth=0.0, lowerweight=0.0, weights=:balance)\n\nEntropy weighting uses the empirical entropy of the vocabulary along classes to produce a notion of importance for each token\n\n\n\n\n\n","category":"type"},{"location":"api/#TextSearch.FreqWeighting","page":"API","title":"TextSearch.FreqWeighting","text":"FreqWeighting()\n\nFrequency weighting\n\n\n\n\n\n","category":"type"},{"location":"api/#TextSearch.GlobalWeighting","page":"API","title":"TextSearch.GlobalWeighting","text":"GlobalWeighting\n\nAbstract type for global weighting\n\n\n\n\n\n","category":"type"},{"location":"api/#TextSearch.IdfWeighting","page":"API","title":"TextSearch.IdfWeighting","text":"IdfWeighting()\n\nInverse document frequency weighting\n\n\n\n\n\n","category":"type"},{"location":"api/#TextSearch.LocalWeighting","page":"API","title":"TextSearch.LocalWeighting","text":"LocalWeighting\n\nAbstract type for local weighting\n\n\n\n\n\n","category":"type"},{"location":"api/#TextSearch.Skipgram","page":"API","title":"TextSearch.Skipgram","text":"Skipgram(qsize, skip)\n\nA skipgram is a kind of tokenization where qsize words having skip separation are used as a single token.\n\n\n\n\n\n","category":"type"},{"location":"api/#TextSearch.TextConfig","page":"API","title":"TextSearch.TextConfig","text":"TextConfig(;\n    del_diac::Bool=true,\n    del_dup::Bool=false,\n    del_punc::Bool=false,\n    group_num::Bool=true,\n    group_url::Bool=true,\n    group_usr::Bool=false,\n    group_emo::Bool=false,\n    lc::Bool=true,\n    qlist::Vector=Int8[],\n    nlist::Vector=Int8[],\n    slist::Vector{Skipgram}=Skipgram[],\n    tt=IdentityTokenTransformation()\n)\n\nDefines a preprocessing and tokenization pipeline\n\ndel_diac: indicates if diacritic symbols should be removed\ndel_dup: indicates if duplicate contiguous symbols must be replaced for a single symbol\ndel_punc: indicates if punctuaction symbols must be removed\ngroup_num: indicates if numbers should be grouped _num\ngroup_url: indicates if urls should be grouped as _url\ngroup_usr: indicates if users (@usr) should be grouped as _usr\ngroup_emo: indicates if emojis should be grouped as _emo\nlc: indicates if the text should be normalized to lower case\nqlist: a list of character q-grams to use\nnlist: a list of words n-grams to use\nslist: a list of skip-grams tokenizers to use\ntt: An AbstractTokenTransformation struct\n\nNote: If qlist, nlist, and slists are all empty arrays, then it defaults to nlist=[1]\n\n\n\n\n\n","category":"type"},{"location":"api/#TextSearch.TextModel","page":"API","title":"TextSearch.TextModel","text":"Model\n\nAn abstract type that represents a weighting model\n\n\n\n\n\n","category":"type"},{"location":"api/#TextSearch.TfWeighting","page":"API","title":"TextSearch.TfWeighting","text":"TfWeighting()\n\nTerm frequency weighting\n\n\n\n\n\n","category":"type"},{"location":"api/#TextSearch.TpWeighting","page":"API","title":"TextSearch.TpWeighting","text":"TpWeighting()\n\nTerm probability weighting\n\n\n\n\n\n","category":"type"},{"location":"api/#TextSearch.VectorModel-Tuple{EntropyWeighting, LocalWeighting, Vocabulary, AbstractVector{Dict{UInt32, Int32}}, Any}","page":"API","title":"TextSearch.VectorModel","text":"VectorModel(ent::EntropyWeighting, lw::LocalWeighting, corpus::BOW, labels;\n    mindocs::Integer=1,\n    smooth::Float64=0.0,\n    weights=:balance,\n    lowerweight=0.0\n)\n\nCreates a vector model using the input corpus. \n\n\n\n\n\n","category":"method"},{"location":"api/#TextSearch.Vocabulary-Tuple{Any}","page":"API","title":"TextSearch.Vocabulary","text":"Vocabulary(corpus)\n\nComputes a vocabulary from an already tokenized corpus\n\n\n\n\n\n","category":"method"},{"location":"api/#TextSearch.Vocabulary-Tuple{Integer}","page":"API","title":"TextSearch.Vocabulary","text":"Vocabulary(n::Integer)\n\nCreates a Vocabulary struct\n\n\n\n\n\n","category":"method"},{"location":"api/#TextSearch.Vocabulary-Tuple{TextConfig, AbstractVector{T} where T}","page":"API","title":"TextSearch.Vocabulary","text":"Vocabulary(textconfig, corpus; minbatch=0, thesaurus=nothing)\n\nComputes a vocabulary from a corpus using the TextConfig textconfig. If thesaurus is a collection of strings, then it is used as a fixed list of tokens.\n\n\n\n\n\n","category":"method"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = TextSearch","category":"page"},{"location":"#TextSearch","page":"Home","title":"TextSearch","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"TextSearch.jl is a package to create vector representations of text, mostly, independently of the language. It is intended to be used with SimilaritySearch.jl, but can be used independetly if needed. TextSearch.jl was renamed from TextModel.jl to reflect its capabilities and mission.","category":"page"},{"location":"","page":"Home","title":"Home","text":"For generic text analysis you should use other packages like TextAnalysis.jl.","category":"page"},{"location":"","page":"Home","title":"Home","text":"It supports a number of simple text preprocessing functions, and three different kinds of tokenizers, i.e., word n-grams, character q-grams, and skip-grams. It supports creating multisets of tokens, commonly named bag of words (BOW). TextSearch.jl can produce sparse vector representations based on term-weighting schemes like TF, IDF, and TFIDF. It also supports term-weighting schemes designed to cope text classification tasks, mostly based on distributional representations.","category":"page"},{"location":"#Installing","page":"Home","title":"Installing","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"You may install the package as follows","category":"page"},{"location":"","page":"Home","title":"Home","text":"] add TextSearch","category":"page"},{"location":"","page":"Home","title":"Home","text":"also, you can run the set of tests as follows","category":"page"},{"location":"","page":"Home","title":"Home","text":"] test TextSearch","category":"page"}]
}
