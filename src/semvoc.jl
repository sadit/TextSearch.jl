# This file is a part of TextSearch.jl

export SemanticVocabulary

struct SemanticVocabulary{VocType<:Vocabulary}
    voc::VocType
    lexidx::BM25InvertedFile
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

    SemanticVocabulary(voc, lexidx, knns)
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

function vectorize(model::SemanticVocabulary, text; k::Int=15, normalize=true, ksem::Int=k, w::Float32=1f0)
    res = getknnresult(k)
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
                D[i] = get(D, i, 0f0) + w
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
