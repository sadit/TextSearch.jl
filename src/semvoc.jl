# This file is a part of TextSearch.jl

export SemanticVocabulary

struct SemanticVocabulary{VocType<:Vocabulary}
    voc::VocType
    lexidx::BM25InvertedFile
    semidx::BinaryInvertedFile
    knns::Matrix{Int32}
end

SemanticVocabulary(C::SemanticVocabulary; 
                   voc=C.voc, lexidx=C.lexidx, semidx=C.semidx) =
    SemanticVocabulary(voc, lexidx, semidx)


function SemanticVocabulary(voc::Vocabulary;
        tc::TextConfig=TextConfig(nlist=[1], qlist=[4]),
        k::Int=33,
        list_min_length_for_checking::Int=32,
        list_max_allowed_length::Int=128,
        doc_min_freq::Int=1,  # any hard vocabulary pruning are expected to be made in `voc`
        doc_max_ratio::AbstractFloat=0.3 # popular tokens are likely to be thrash
    )
    doc_max_freq = ceil(Int, doc_max_ratio * vocsize(voc))
    C = tokenize_corpus(tc, voc.token)

    lexidx = BM25InvertedFile(tc, C) do t
        doc_min_freq <= t.ndocs <= doc_max_freq
    end

    #=@time for part in Iterators.partition(1:n, bsize)
        append_items!(lexidx, VectorDatabase([corpus[i] for i in part]))
    end=#
    @info "append_items"
    @time append_items!(lexidx, C; sort=false)

    #doc_max_freq = ceil(Int, vocsize(voc) * doc_max_ratio)
    @info "filter lists!"
    @time filter_lists!(lexidx;
                  list_min_length_for_checking,
                  list_max_allowed_length,
                  doc_min_freq,
                  doc_max_freq,
                  always_sort=true # we need this since we call append_items! without sorting
                 )
    @info "searchbatch"
    @time knns, _ = searchbatch(lexidx, VectorDatabase(C), k)
    semidx = BinaryInvertedFile(size(knns, 2), JaccardDistance())
    @time append_items!(semidx, MatrixDatabase(knns))

    SemanticVocabulary(voc, lexidx, semidx, knns)
end

enrich_bow!(v::Dict, l::Nothing) = v
function enrich_bow!(v::Dict, l)
    for (k, w) in l
        v[k] = w
    end

    v
end

function lexicalsearch(model::SemanticVocabulary, text, res::KnnResult; tc=model.lexidx.textconfig)
    search(model.lexidx, text, res).res
end


function lexicalvectorize(model::SemanticVocabulary, text, res::KnnResult; normalize=true)
    D = DVEC{UInt32,Float32}()
    res = lexicalsearch(model, text, res)
    for p in res
        D[p.id] = abs(p.weight)
    end

    normalize && normalize!(D)
    D
end

function lexicalvectorize(model::SemanticVocabulary, text; k::Int=15, normalize=true)
    lexicalvectorize(model, text, getknnresult(k); normalize)
end

function semanticsearch(model::SemanticVocabulary, text, res::KnnResult)
    D = lexicalvectorize(model, text, res)
    res = reuse!(res)
    search(model.semidx, D, res).res
end

function decode(model::SemanticVocabulary, idlist)
    [model[i] for i in itertokenid(idlist)]
end

Base.getindex(model::SemanticVocabulary, i::Integer) = model.voc[i]

function semanticvectorize(
        model::SemanticVocabulary,
        text;
        klex::Int=33,
        ksem=klex,
        normalize::Bool=true,
        keeplex::Bool=true
    )
    res = getknnresult(klex)
    D = lexicalvectorize(model, text, res)
    res = reuse!(res, ksem)
    search(model.semidx, D, res)

    if keeplex
        for p in res
            D[p.id] = get(D, p.id, 0f0) + 1f0 - p.weight # works for jaccard
        end
    else
        empty!(D)
        for p in res
            D[p.id] = 1f0 - p.weight # jaccard 
        end
    end

    normalize && normalize!(D)
    D
end


function subvoc(model::SemanticVocabulary, idlist, tc=model.lexidx.textconfig; k=100)
    corpus = [model.voc.token[i] for i in itertokenid(idlist)]
    Vocabulary(tc, corpus)
end
