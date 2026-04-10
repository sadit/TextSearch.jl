# This file is a part of TextSearch.jl
export BM25, bm25score, tokenscore

struct BM25
    #k1::Float32
    #b::Float32
    k1_plus_1::Float32
    k1_mult_1_min_b::Float32
    k1_mult_b_div_avg_doc_len::Float32
    δ::Float32
    trainsize::Int32
end


function Base.show(io::IO, bm::BM25; prefix="", indent="  ")
    println(io, prefix, "BM25:")
    prefix = indent * prefix
    k1 = bm.k1_plus_1 - 1
    b = - (bm.k1_mult_1_min_b / k1 - 1)
    avgdoclen = 1 / (bm.k1_mult_b_div_avg_doc_len / b / k1)
    println(io, prefix, "k1: ", k1)
    println(io, prefix, "b: ", b)
    println(io, prefix, "avgdoclen: ", avgdoclen)
    println(io, prefix, "δ: ", bm.δ)
    println(io, prefix, "trainsize: ", bm.trainsize)
end


function BM25(trainsize::Integer, avgdoclen::AbstractFloat; k1=1.2f0, b=0.75f0, δ=1f0)
    BM25( #k1, b,
        convert(Float32, k1 + 1f0),
        convert(Float32, k1 * (1f0 - b)),
        convert(Float32, k1 * b / avgdoclen),
        δ,
        convert(Int32, trainsize)
    )
end

BM25(voc::Vocabulary; k1=1.2f0, b=0.75f0, δ=1f0) = BM25(trainsize(voc), avgdoclen(voc); k1, b, δ)

function bm25score(bm25::BM25, voc::Vocabulary, query::Dict, doc::Dict)::Float32
    s = 0f0

    doclen = sum(f for f in values(doc))
    for tokenID in keys(query)
        w = get(doc, tokenID, 0f0)
        if w > 0f0
            s += tokenscore(bm25, ndocs(voc, tokenID), doclen, w)
        end
    end

    s
end

function tokenscore(bm25::BM25, toknumdocs, doclen, tokfreqindoc)
    idf = log(1f0 + (bm25.trainsize - toknumdocs + 0.5f0) / (toknumdocs + 0.5f0))
    num = tokfreqindoc * bm25.k1_plus_1 
    den = tokfreqindoc + bm25.k1_mult_1_min_b + doclen * bm25.k1_mult_b_div_avg_doc_len 
    convert(Float32, idf * (num / den + bm25.δ))
end
