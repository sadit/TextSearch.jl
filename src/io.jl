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
function savemodel(filename::AbstractString, model; meta=nothing, parent="/")
    jldopen(filename, "w") do f
        savemodel(f, model; meta, parent)
    end
end

function savemodel(file::JLDFile, model::SemanticVocabulary; meta=nothing, parent="/")
    file[joinpath(parent, "meta")] = meta
    file[joinpath(parent, "voc")] = model.voc
    file[joinpath(parent, "knns")] = model.knns
    file[joinpath(parent, "sel")] = model.sel
    saveindex(file, model.lexidx; parent=joinpath(parent, "lexidx"))
end

function loadmodel(t::Type, filename::AbstractString; parent="/", staticgraph=false)
    jldopen(filename) do f
        loadmodel(t, f; staticgraph, parent)
    end
end

function loadmodel(::Type{SemanticVocabulary}, file::JLDFile; parent="/", staticgraph=false)
    meta = file[joinpath(parent, "meta")]
    voc = file[joinpath(parent, "voc")]
    knns = file[joinpath(parent, "knns")]

    lexidx, _ = loadindex(file; parent=joinpath(parent, "lexidx"), staticgraph)

   SemanticVocabulary(tvoc, lexidx, knns, sel), meta
end

