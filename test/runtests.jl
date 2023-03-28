using SimilaritySearch, SimilaritySearch.AdjacencyLists, TextSearch, InvertedFiles
using Test, SparseArrays, LinearAlgebra, CategoricalArrays, StatsBase, Random

using Aqua
Aqua.test_all(TextSearch, ambiguities=false)
Aqua.test_ambiguities([TextSearch])

const fit = TextSearch.fit

const text0 = "@user;) #jello.world"
const text1 = "hello world!! @user;) #jello.world :)"
const text2 = "a b c d e f g h i j k l m n o p q"
const corpus = ["hello world :)", "@user;) excellent!!", "#jello world."]
const sentiment_corpus = ["me gusta", "me encanta", "lo lo odio", "odio esto", "me encanta esto LOL!"]
const sentiment_labels = categorical(["pos", "pos", "neg", "neg", "pos"])
const sentiment_msg = "lol, esto me encanta"
_corpus = [
    "la casa roja",
    "la casa verde",
    "la casa azul",
    "la manzana roja",
    "la pera verde esta rica",
    "la manzana verde esta rica",
    "la hoja verde",
]


function are_posting_lists_sorted(invindex)
    for (k, lists) in invindex.lists
        if issorted([p.id for p in lists]) == false
            return false
        end
    end

    true
end


#=include("tok.jl")
include("voc.jl")
include("vec.jl")
include("search.jl")
=#
include("semvoc.jl")

@info "FINISH"
