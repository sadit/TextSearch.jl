# This file is a part of TextSearch.jl

using JLD2: jldopen, JLDFile
export savemodel, loadmodel, loadindex

function serializeindex(file, parent::String, index::BM25InvertedFile, meta, options::Dict)
    adj = StaticAdjacencyList(index.adj)
    I = copy(index; adj)
    file[joinpath(parent, "index")] = I
end

"""
    loadindex(...; staticgraph=false, parent="/")
    restoreindex(file, parent::String, index, meta, options::Dict; staticgraph=false)

load the inverted index optionally making the postings lists static or dynamic 
"""
function restoreindex(file::JLDFile, parent::String, index::BM25InvertedFile, meta, options::Dict; staticgraph=false)
    adj = staticgraph ? index.adj : AdjacencyList(index.adj)
    copy(index; adj)
end

# lang model
function savemodel(filename::AbstractString, ngrams; meta=nothing, parent="/")
    jldopen(filename, "w") do f
        savemodel(f, ngrams; meta, parent)
    end
end

function savemodel(file::JLDFile, ngrams::LanguageModel; meta=nothing, parent="/")
    file[joinpath(parent, "meta")] = meta
    file[joinpath(parent, "tc")] = ngrams.vocngrams 
    file[joinpath(parent, "vocngrams")] = ngrams.semidx
    saveindex(file, ngrams.lexidx; parent=joinpath(parent, "lexidx"))
    saveindex(file, ngrams.semidx; parent=joinpath(parent, "semidx"))
end

function loadmodel(t::Type{LanguageModel}, filename::AbstractString; parent="/", staticgraph=false)
    jldopen(filename) do f
        loadmodel(t, f; staticgraph, parent)
    end
end

function loadmodel(::Type{LanguageModel}, file::JLDFile; parent="/", staticgraph=false)
    meta = file[joinpath(parent, "meta")]
    tc = file[joinpath(parent, "tc")]
    vocngrams = file[joinpath(parent, "vocngrams")]

    lexidx, _ = loadindex(file; parent=joinpath(parent, "lexidx"), staticgraph)
    semidx, _ = loadindex(file; parent=joinpath(parent, "semidx"), staticgraph)

    LanguageModel(tc, vocngrams, lexidx, semidx), meta
end


# corpus lang model

function savemodel(file::JLDFile, model::CorpusLanguageModel; meta=nothing, parent="/")
    file[joinpath(parent, "meta")] = meta
    file[joinpath(parent, "corpus")] = model.corpus 
    file[joinpath(parent, "labels")] = model.labels 

    saveindex(file, model.lexidx; parent=joinpath(parent, "lexidx"))
    saveindex(file, model.semidx; parent=joinpath(parent, "semidx"))
end

function loadmodel(::Type{CorpusLanguageModel}, file::JLDFile; parent="/", staticgraph=false)
    meta = file[joinpath(parent, "meta")]
    corpus = file[joinpath(parent, "corpus")]
    labels = file[joinpath(parent, "labels")]

    lexidx, _ = loadindex(file; parent=joinpath(parent, "lexidx"), staticgraph)
    semidx, _ = loadindex(file; parent=joinpath(parent, "semidx"), staticgraph)

    CorpusLanguageModel(corpus, labels, lexidx, semidx), meta
end
