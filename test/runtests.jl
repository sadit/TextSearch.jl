using SimilaritySearch, TextSearch
using Test, SparseArrays, LinearAlgebra, StatsBase, Random

using Aqua
Aqua.test_all(TextSearch, ambiguities=false, piracies=false, stale_deps=false)
Aqua.test_ambiguities([TextSearch])
using LinearAlgebra, SparseArrays 
using SimilaritySearch
Aqua.test_piracies(TextSearch, treat_as_own=[Base.:*, Base.:/, Base.:+, Base.:-, argmin, argmax, findmin, findmax, sum, Dist.evaluate, dot, maximum, minimum, norm, sparse, sparse_coo, nnz, sparsevec, zero])

const fit = TextSearch.fit

const text0 = "@user;) #jello.world"
const text1 = "hello world!! @user;) #jello.world :)"
const text2 = "a b c d e f g h i j k l m n o p q"
const corpus = ["hello world :)", "@user;) excellent!!", "#jello world."]
const sentiment_corpus = ["me gusta", "me encanta", "lo lo odio", "odio esto", "me encanta esto LOL!"]
const sentiment_labels = ["pos", "pos", "neg", "neg", "pos"]
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


include("intersections.jl")
include("tok.jl")
include("voc.jl")
include("teststopwords.jl")
include("vec.jl")
include("svec.jl")
include("search.jl")
include("ext_snowball.jl")
include("testqueryexpansion.jl")
include("testprofile.jl")
include("testmergeprofiles.jl")
include("testrefit.jl")
include("fulltext.jl")
include("testlsi.jl")
include("testlemmas.jl")
include("testri.jl")
@info "FINISH"
