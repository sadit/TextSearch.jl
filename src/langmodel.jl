# This file is a part of TextSearch.jl

export LanguageModel, lexicalsearch, semanticsearch,
    lexicalvectorize, semanticvectorize, subvoc, context, decode 
"""
    struct LanguageModel{LexIndexType<:BM25InvertedFile, SemIndexType<:AbstractInvertedFile}

"""
struct LanguageModel{LexIndexType<:BM25InvertedFile, SemIndexType<:AbstractInvertedFile}
    tc::TextConfig
    vocngrams::Vocabulary
    lexidx::LexIndexType
    semidx::SemIndexType
end

function LanguageModel(
        corpus;
        tc1=TextConfig(nlist=[4]),
        tc2=TextConfig(nlist=[1]),
        min_ndocs1::Int=2,
        min_ndocs2::Int=3,
        k::Int=33,
        list_min_length_for_checking::Int=30,
        list_max_allowed_length::Int=256,
        doc_min_freq::Int=1,
        doc_max_ratio::AbstractFloat=0.8
    )

    ngrams = Vocabulary(tc1, corpus)
    if min_ndocs1 > 1
        ngrams = filter_tokens(ngrams) do t
            min_ndocs1 <= t.ndocs # it depends highly on the size of the dataset and rarety of the examples
        end
    end

    lexidx = BM25InvertedFile(tc2, ngrams.token, VectorDatabase(SVEC[])) do t
        min_ndocs2 <= t.ndocs # again
    end

    @time append_items!(lexidx, ngrams.token)
    doc_max_freq = ceil(Int, vocsize(ngrams) * doc_max_ratio)
    filter_lists!(lexidx;
                  list_min_length_for_checking,
                  list_max_allowed_length,
                  doc_min_freq,
                  doc_max_freq
                 )

    semidx = BinaryInvertedFile(vocsize(ngrams), JaccardDistance())
    @time knns, _ = searchbatch(lexidx, lexidx.db, k)
    @time append_items!(semidx, MatrixDatabase(knns))
    LanguageModel(tc1, ngrams, lexidx, semidx)
end

function lexicalsearch(ngrams::LanguageModel, text::AbstractString, res::KnnResult)
    search(ngrams.lexidx, text, res).res
end

function lexicalvectorize(ngrams::LanguageModel, text::AbstractString, res::KnnResult; normalize=true)
    D = DVEC{UInt32,Float32}()
    res = lexicalsearch(ngrams, text, res)
    for p in res
        D[p.id] = abs(p.weight)
    end
    
    normalize && normalize!(D)
    D
end

function lexicalvectorize(ngrams::LanguageModel, text::AbstractString; k::Int=15, normalize=true)
    lexicalvectorize(ngrams, text, getknnresult(k); normalize)
end

function semanticsearch(ngrams::LanguageModel, text::AbstractString, res::KnnResult)
    D = lexicalvectorize(ngrams, text, res)
    res = reuse!(res)
    search(ngrams.semidx, D, res).res
    #ngrams.vocngrams.token[IdView(search(ngrams.semidx, d, getknnresult(33)).res) |> collect]
end

itertokenid(idlist::AbstractVector) = idlist 
itertokenid(idlist::Vector{IdWeight}) = (p.id for p in idlist) 
itertokenid(idlist::Vector{IdIntWeight}) = (p.id for p in idlist) 
itertokenid(idlist::Dict) = keys(idlist) 
itertokenid(idlist::KnnResult) = IdView(idlist) 

function decode(ngrams::LanguageModel, idlist)
    [ngrams.vocngrams[i] for i in itertokenid(idlist)]
end

function semanticvectorize(ngrams::LanguageModel, text::AbstractString, res::KnnResult; normalize=true)
    D = DVEC{UInt32,Float32}()
    res = semanticsearch(ngrams, text, res)
    for p in res
        D[p.id] = abs(p.weight)
    end
    
    normalize && normalize!(D)
    D
end

function semanticvectorize(ngrams::LanguageModel, text::AbstractString; k::Int=15, normalize=true)
    semanticvectorize(ngrams, text, getknnresult(k); normalize)
end

function subvoc(ngrams::LanguageModel, idlist, tc=TextConfig(nlist=[1]); k=100)
    corpus = [ngrams.vocngrams.token[i] for i in itertokenid(idlist)]
    Vocabulary(tc, corpus)
end

function context(ngrams::LanguageModel, token::AbstractString, relativepos::Int, tc=TextConfig(nlist=[1]); k=100)
    id = ngrams.lexidx.voc.token2id[token]
    C = Dict{String,Int}()

    for p in neighbors(ngrams.lexidx.adj, id)
        ng = ngrams.vocngrams.token[p.id]
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

function context(ngrams::LanguageModel, token::AbstractString, tc=TextConfig(nlist=[1]); k=100)
    id = ngrams.lexidx.voc.token2id[token]
    counters = Vector{Tuple{Int,String}}()

    for p in neighbors(ngrams.lexidx.adj, id)
        ng = ngrams.vocngrams.token[p.id]
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

