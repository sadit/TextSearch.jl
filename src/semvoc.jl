# This file is a part of TextSearch.jl

export SemanticVocabulary

struct SemanticVocabulary
    voc::Vocabulary
    lexidx::BM25InvertedFile
    knns::Matrix{Int32}
    klex::Int64
    ksem::Int64
    wsem::Float32
end

SemanticVocabulary(C::SemanticVocabulary; 
                   voc=C.voc, lexidx=C.lexidx, semidx=C.semidx, klex=C.klex, ksem=C.ksem, wsem=C.wsem) =
    SemanticVocabulary(voc, lexidx, semidx, klex, ksem, wsem)

vocsize(model::SemanticVocabulary) = vocsize(voc)

function token2id(model::SemanticVocabulary, tok::AbstractString; ksem::Int=model.ksem)
    id = token2id(model.voc, tok)

    if id == 0
        klex=model.klex
        res = getknnresult(klex)
        search(model, tok, res)

        if ksem == 0
            id = argmin(res)
        else
            buff = take!(TEXT_SEARCH_CACHES)
            empty!(buff.vec)
            D = buff.vec
            sizehint!(D, length(res) * (1 + ksem))
            try
                for p in res
                    D[p.id] = get(D, p.id, 0f0) + 1f0
                    for i in view(model.knns, 1:ksem, p.id)
                        D[i] = get(D, i, 0f0) + 1f0
                    end
                end

                id = argmax(D)
            finally
                put!(TEXT_SEARCH_CACHES, buff)
            end
        end
    end

    id
end

function SemanticVocabulary(voc::Vocabulary;
        tc::TextConfig=TextConfig(nlist=[1], qlist=[4]),
        ksem::Int=32,
        klex::Int=ksem,
        wsem::Float32=1f0,
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
    @time knns, _ = searchbatch(lexidx, VectorDatabase(C), ksem)

    SemanticVocabulary(voc, lexidx, knns, klex, ksem, wsem)
end

enrich_bow!(v::Dict, l::Nothing) = v
function enrich_bow!(v::Dict, l)
    for (k, w) in l
        v[k] = w
    end

    v
end

function search(model::SemanticVocabulary, text, res::KnnResult; tc=model.lexidx.textconfig)
    search(model.lexidx, text, res)
end

function vectorize(model::SemanticVocabulary, text; klex::Int=model.klex, normalize=true, ksem::Int=model.ksem, wsem::Float32=model.wsem)
    res = getknnresult(klex)
    search(model, text, res)
    D = DVEC{UInt32,Float32}()
    sizehint!(D, length(res) * (1 + ksem))
    if ksem == 0 || w == 0
        for p in res
            D[p.id] = abs(p.weight)
        end 
    else
        for p in res
            D[p.id] = get(D, p.id, 0f0) + abs(p.weight)
            for i in view(model.knns, 1:ksem, p.id)
                D[i] = get(D, i, 0f0) + wsem
            end
        end 
    end

    normalize && normalize!(D)
    D
end

function decode(model::SemanticVocabulary, idlist)
    [model[i] for i in itertokenid(idlist)]
end

Base.getindex(model::SemanticVocabulary, i::Integer) = model.voc[i]

function subvoc(model::SemanticVocabulary, idlist, tc=model.lexidx.textconfig)
    corpus = [model.voc.token[i] for i in itertokenid(idlist)]
    Vocabulary(tc, corpus)
end
