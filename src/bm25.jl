# This file is a part of TextSearch.jl

struct BM25
    #k1::Float32
    #b::Float32
    k1_plus_1::Float32
    k1_mult_1_min_b::Float32
    k1_mult_b_div_avg_doc_len::Float32
    δ::Float32
    collection_size::UInt32
end

function BM25(avg_doc_len::AbstractFloat, collection_size::Integer; k1=1.2f0, b=0.75f0, δ=1f0)
    BM25( #k1, b,
        convert(Float32, k1 + 1f0),
        convert(Float32, k1 * (1f0 - b)),
        convert(Float32, k1 * b / avg_doc_len),
        δ,
        convert(UInt32, collection_size)
    )
end

function bm25score(bm25::BM25, voc::Vocabulary, query::DVEC, doc::DVEC)
    s = 0.0

    doclen = sum(f for f in values(doc))
    for tokenID in keys(query)
        s += tokenscore(bm25, ndocs(voc, tokenID), doclen, doc[tokenID])
    end

    s
end

function tokenscore(bm25::BM25, toknumdocs, doclen, tokfreqindoc)
    idf = log(1f0 + (bm25.collection_size - toknumdocs + 0.5f0) / (toknumdocs + 0.5f0))
    num = tokfreqindoc * bm25.k1_plus_1 
    den = tokfreqindoc + bm25.k1_mult_1_min_b + doclen * bm25.k1_mult_b_div_avg_doc_len 
    convert(Float32, idf * (num / den + bm25.δ))
end
