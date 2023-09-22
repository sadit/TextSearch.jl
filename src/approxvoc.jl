# This file is a part of TextSearch.jl

export QgramsLookup

struct QgramsLookup <: AbstractTokenLookup
    voc::Vocabulary{TokenLookup}
    idx::BinaryInvertedFile
    maxdist::Float32
end

"""
    QgramsLookup(
        voc::Vocabulary,
        dist::SemiMetric=JaccardDistance();
        maxdist::Real = 0.7,
        textconfig=TextConfig(qlist=[3]),
        doc_min_freq::Integer=1,  # any hard vocabulary pruning are expected to be made in `voc`
        doc_max_ratio::AbstractFloat=0.4 # popular tokens are likely to be thrash
    )
"""
function QgramsLookup(
        voc::Vocabulary,
        dist::SemiMetric=JaccardDistance();
        maxdist::Real = 0.7,
        textconfig=TextConfig(qlist=[3]),
        doc_min_freq::Integer=1,  # any hard vocabulary pruning are expected to be made in `voc`
        doc_max_ratio::AbstractFloat=0.4 # popular tokens are likely to be thrash
    )
    
    voc_ = Vocabulary(textconfig, token(voc))
    voc_ = filter_tokens(voc_) do t
        doc_min_freq <= t.ndocs <= doc_max_ratio * vocsize(voc_)
    end

    invfile = BinaryInvertedFile(vocsize(voc_), dist)
    append_items!(invfile, VectorDatabase(bagofwords_corpus(voc_, token(voc))))
    QgramsLookup(voc_, invfile, maxdist)
end

function token2id(voc::Vocabulary{QgramsLookup}, tok)::UInt32
    lookup = voc.lookup
    i = get(voc.token2id, tok, zero(UInt32))
    i > 0 && return i
    tok == "" && return 0
    res = KnnResult(1)
    search(lookup.idx, bagofwords(lookup.voc, tok), res)
    p = res[1]
    p.weight > lookup.maxdist ? 0 : p.id
end

