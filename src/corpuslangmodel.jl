# This file is a part of TextSearch.jl

export EncodedCorpus, CorpusLanguageModel

struct EncodedCorpus
    tc::TextConfig
    voc::Vocabulary{String}
    seq::Vector{UInt32}
    offset::Vector{UInt64}
end

function EncodedCorpus(
        corpus; kwargs...
    )
    tc = TextConfig(nlist=[1], mark_token_type=false)
    voc = Vocabulary(tc, corpus)
    EncodedCorpus(tc, voc, corpus; kwargs...)
end

function EncodedCorpus(
        tc::TextConfig,
        voc::Vocabulary,
        corpus;
        bsize::Int=10^4
    )
    
    tc.nlist == [1] && length(tc.qlist) == 0 && length(tc.slist) == 0 || throw(ArgumentError("only unigrams are supported for EncodedCorpus"))
    seq = UInt32[]
    offset = UInt64[]

    sizehint!(seq, bsize)
    for subcorpus in Iterators.partition(corpus, bsize)
        off = 0
        for tokdoc in tokenize_corpus(tc, subcorpus)
            off += length(tokdoc)
            push!(offset, off)
            
            for tok in tokdoc
                i = get(voc.token2id, tok, zero(UInt32))
                push!(seq, i)
            end
        end
    end
    
    EncodedCorpus(tc, voc, seq, offset)
end

@inline Base.length(ecorpus::EncodedCorpus) = length(ecorpus.offset)
@inline Base.eachindex(ecorpus::EncodedCorpus) = 1:length(ecorpus)
function Base.iterate(ecorpus::EncodedCorpus, i::Int=1)
    n = length(ecorpus)
    (n == 0 || i > n) && return nothing
    @inbounds ecorpus[i], i+1
end

function Base.getindex(ecorpus::EncodedCorpus, i::Integer)
    fetch(ecorpus, i)
end

function fetch(ecorpus::EncodedCorpus, i::Integer)
    sp, ep = i == 1 ? (UInt64(1), ecorpus.offset[1]) : (ecorpus.offset[i-1]+1, ecorpus.offset[i])
    view(ecorpus.seq, sp:ep)
end

function decode(ecorpus::EncodedCorpus, doc)
    [ecorpus.voc.token[id] for id in doc]
end

struct CorpusLanguageModel
    corpus::EncodedCorpus
    labels::Dict{UInt32,Vector{Pair{UInt32,Float32}}}  # labels to enrich corpus elements
    lexidx::BM25InvertedFile
    semidx::BinaryInvertedFile
end

function CorpusLanguageModel(corpus::EncodedCorpus, labels=nothing;
        k::Int=15,
        #bsize::Int=10^5,
        list_min_length_for_checking::Int=32,
        list_max_allowed_length::Int=128,
        doc_min_freq::Int=1,
        doc_max_ratio::AbstractFloat=0.8
    )

    labels = labels === nothing ? Dict{UInt32,Vector{Pair{UInt32,Float32}}}() : labels

    n = length(corpus)
    doclen = Int32[length(text) for text in corpus]
    avg_doc_len = mean(doclen)
    bm25 = BM25(avg_doc_len, n)
    voc = corpus.voc

    @info "lexidx"
    lexidx = BM25InvertedFile(
        nothing,
        corpus.tc,
        voc,
        bm25,
        AdjacencyList(IdIntWeight; n=vocsize(voc)),
        Vector{Int32}(undef, 0)
    )

    #=@time for part in Iterators.partition(1:n, bsize)
        append_items!(lexidx, VectorDatabase([corpus[i] for i in part]))
    end=#
    DB = VectorDatabase([corpus[i] for i in 1:n])
    @info "append_items"
    append_items!(lexidx, DB)

    doc_max_freq = ceil(Int, vocsize(voc) * doc_max_ratio)
    @info "filter lists!"
    filter_lists!(lexidx;
                  list_min_length_for_checking,
                  list_max_allowed_length,
                  doc_min_freq,
                  doc_max_freq
                 )


    @info "searchbatch"
    @time knns, _ = searchbatch(lexidx, DB, k)
    semidx = BinaryInvertedFile(n, JaccardDistance())
    @time append_items!(semidx, MatrixDatabase(knns))

    CorpusLanguageModel(corpus, labels, lexidx, semidx)
    #corpus = readlines("data/StackOverflow.txt")
    #E = EncodedCorpus(corpus)
end

enrich_bow!(v::Dict, l::Nothing) = v
function enrich_bow!(v::Dict, l)
    for (k, w) in l
        v[k] = w
    end

    v
end

function lexicalsearch(model::CorpusLanguageModel, text, res::KnnResult; tc=model.lexidx.textconfig)
    search(model.lexidx, text, res).res
end


function lexicalvectorize(model::CorpusLanguageModel, text, res::KnnResult; normalize=true)
    D = DVEC{UInt32,Float32}()
    res = lexicalsearch(model, text, res)
    for p in res
        D[p.id] = abs(p.weight)
    end

    for tok in keys(D)
        enrich_bow!(D, get(model.labels, tok, nothing))
    end

    normalize && normalize!(D)
    D
end

function lexicalvectorize(model::CorpusLanguageModel, text; k::Int=15, normalize=true)
    lexicalvectorize(model, text, getknnresult(k); normalize)
end

function semanticsearch(model::CorpusLanguageModel, text, res::KnnResult)
    D = lexicalvectorize(model, text, res)
    res = reuse!(res)
    search(model.semidx, D, res).res
end

function decode(model::CorpusLanguageModel, idlist)
    [model.corpus.voc[i] for i in itertokenid(idlist)]
end

function Base.getindex(model::CorpusLanguageModel, i::Int)
    model.corpus[i]
end

function semanticvectorize(
        model::CorpusLanguageModel,
        text;
        klex::Int=1,
        ksem=klex,
        normalize::Bool=true,
        keeplex::Bool=true
    )
    res = getknnresult(klex)
    D = lexicalvectorize(model, text, res)
    res = reuse!(res)
    search(model.semidx, D, res)

    if keeplex
        for p in res
            D[p.id] = get(D, p.id, 0f0) + abs(p.weight)
        end
    else
        empty!(D)
        for p in res
            D[p.id] = abs(p.weight)
        end
    end

    normalize && normalize!(D)
    D
end

function subvoc(model::CorpusLanguageModel, idlist, tc=model.corpus.tc; k=100)
    corpus = [model.vocngrams.token[i] for i in itertokenid(idlist)]
    Vocabulary(tc, corpus)
end

function context(model::CorpusLanguageModel, token::AbstractString, relativepos::Int; tc=model.corpus.tc, k=100)
    id = model.corpus.voc.token2id[token]
    C = Dict{String,Int}()

    for p in neighbors(model.lexidx.adj, id)
        ng = model.vocngrams.token[p.id]
        prev, next = split(ng, token)

        if relativepos < 0
            tokens = tokenize(tc, prev)
            #@show ng, token, next, tokens, relativepos
            abs(relativepos) <= length(tokens) || continue
            t = tokens[end+relativepos+1]
            C[t] = get(C, t, 0) + 1
        else
            tokens = tokenize(tc, next)
            relativepos <= length(tokens) || continue
            t = tokens[relativepos]
            C[t] = get(C, t, 0) + 1
        end
    end

    C
end

function context(model::CorpusLanguageModel, token::AbstractString; tc=model.corpus.tc, k=100)
    id = model.corpus.voc.token2id[token]
    counters = Vector{Tuple{Int,String}}()

    for p in neighbors(model.lexidx.adj, id)
        ng = model.vocngrams.token[p.id]
        prev, next = split(ng, token)

        tokens = tokenize(tc, prev)
        n = length(tokens)
        for (i, t) in enumerate(tokens)
            push!(counters, (i-n-1, t))
        end

        tokens = tokenize(tc, next)
        n = length(tokens)
        for (i, t) in enumerate(tokens)
            push!(counters, (i, t))
        end
    end

    counters
end

