# This file is a part of TextSearch.jl

export SemanticVocabulary, AbstractTokenSelection, SelectCentralToken, SelectAllTokens, subvoc, decode, encode

struct SemanticVocabulary{SelType}
    voc::Vocabulary
    lexidx::BM25InvertedFile
    knns::Matrix{Int32}
    sel::SelType
end

abstract type AbstractTokenSelection end

struct SelectCentralToken <: AbstractTokenSelection
    klex::Int32
    ksem::Int32
end

struct SelectAllTokens <: AbstractTokenSelection
    klex::Int32
    ksem::Int32
end

SemanticVocabulary(C::SemanticVocabulary;
                   voc=C.voc, lexidx=C.lexidx, semidx=C.semidx, sel=C.sel) =
    SemanticVocabulary(voc, lexidx, semidx, C.sel)

vocsize(model::SemanticVocabulary) = vocsize(voc)

function SemanticVocabulary(voc::Vocabulary, sel::AbstractTokenSelection=SelectCentralToken(16, 8);
        textconfig::TextConfig=TextConfig(nlist=[1], qlist=[4]),
        list_min_length_for_checking::Int=32,
        list_max_allowed_length::Int=128,
        doc_min_freq::Int=1,  # any hard vocabulary pruning are expected to be made in `voc`
        doc_max_ratio::AbstractFloat=0.3 # popular tokens are likely to be thrash
    )
    doc_max_freq = ceil(Int, doc_max_ratio * vocsize(voc))
    C = tokenize_corpus(textconfig, voc.token)
    lexidx = BM25InvertedFile(textconfig, C) do t
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
    @time knns, _ = searchbatch(lexidx, VectorDatabase(C), sel.ksem)
    SemanticVocabulary(voc, lexidx, knns, sel)
end

enrich_bow!(v::Dict, l::Nothing) = v
function enrich_bow!(v::Dict, l)
    for (k, w) in l
        v[k] = w
    end

    v
end

function search(model::SemanticVocabulary, text, res::KnnResult)
    search(model.lexidx, text, res)
end

function decode(model::SemanticVocabulary, idlist)
    [model.voc.token[i] for i in itertokenid(idlist) if i > 0]
end

Base.getindex(model::SemanticVocabulary, i::Integer) = model.voc[i]

function subvoc(model::SemanticVocabulary, idlist, tc=model.lexidx.voc.textconfig)
    corpus = [model.voc.token[i] for i in itertokenid(idlist)]
    Vocabulary(tc, corpus)
end
