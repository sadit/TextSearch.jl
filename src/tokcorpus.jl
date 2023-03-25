# This file is part of TextSearch.jl

struct EncodedCorpus
    tc::TextConfig
    voc::Vocabulary
    seq::Vector{UInt32}
    offset::Vector{UInt64}
end

EncodedCorpus(C::EncodedCorpus; tc=C.tc, voc=C.voc, seq=C.seq, offset=C.offset) =
    Encoded(tc, voc, seq, offset)

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
    
    #tc.nlist == [1] && length(tc.qlist) == 0 && length(tc.slist) == 0 || throw(ArgumentError("only unigrams are supported for EncodedCorpus"))
    seq = UInt32[]
    offset = UInt64[]

    sizehint!(seq, bsize)
    for subcorpus in Iterators.partition(corpus, bsize)
        off = 0
        for tokdoc in tokenize_corpus(tc, subcorpus)
            for tok in tokdoc
                i = get(voc.token2id, tok, zero(UInt32))
                if i > 0
                    push!(seq, i)
                    off += 1
                end                
            end
            push!(offset, off)
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
