# using Languages
using SimilaritySearch, TextSearch
using Test, SparseArrays, LinearAlgebra, CategoricalArrays, StatsBase, Random
const fit = TextSearch.fit

const text0 = "@user;) #jello.world"
const text1 = "hello world!! @user;) #jello.world :)"
const text2 = "a b c d e f g h i j k l m n o p q"
const corpus = ["hello world :)", "@user;) excellent!!", "#jello world."]

@testset "Character q-grams" begin
    config = TextConfig()
    config.group_usr = false
    config.nlist = []
    config.qlist = [3]
    config.slist = []
    a = [(p.first, p.second) for p in compute_bow(tokenize(config, text0))]
    b = [(:rld, 1), (Symbol(" #j"), 1), (Symbol(" . "), 1), (Symbol(" ;)"), 1), (Symbol(" wo"), 1), (Symbol("#je"), 1), (Symbol(") #"), 1), (Symbol(". w"), 1), (Symbol(";) "), 1), (Symbol("@us"), 1), (:ell, 1), (Symbol("er "), 1), (:jel, 1), (:llo, 1), (Symbol("lo "), 1), (Symbol("o ."), 1), (:orl, 1), (Symbol("r ;"), 1), (:ser, 1), (:use, 1), (:wor, 1)]
    @info setdiff(a, b)
    @info setdiff(b, a)
    @test sort(a) == sort(b)
end

@testset "Word n-grams" begin
    config = TextConfig()
    config.group_usr = false
    config.nlist = [1, 2]
    config.qlist = []
    config.slist = []
    a = [(p.first, p.second) for p in compute_bow(tokenize(config, text0))]
    b = [(Symbol("#jello"), 1), (Symbol("#jello ."), 1), (:., 1), (Symbol(". world"), 1), (Symbol(";)"), 1), (Symbol(";) #jello"), 1), (Symbol("@user"), 1), (Symbol("@user ;)"), 1), (:world, 1)]
    @test sort(a) == sort(b)
 end

@testset "Skip-grams" begin
    config = TextConfig()
    config.nlist = []
    config.qlist = []
    config.del_punc = true
    config.slist = [(2,1), (2, 2), (3, 1), (3, 2)]
    #L = collect(compute_bow(text2, config))
    #sort!(L)
    a = [(p.first, p.second) for p in compute_bow(tokenize(config, text2))]
    b = [(Symbol("a c"), 1), (Symbol("a c e"), 1), (Symbol("a d"), 1), (Symbol("a d g"), 1), (Symbol("b d"), 1), (Symbol("b d f"), 1), (Symbol("b e"), 1), (Symbol("b e h"), 1), (Symbol("c e"), 1), (Symbol("c e g"), 1), (Symbol("c f"), 1), (Symbol("c f i"), 1), (Symbol("d f"), 1), (Symbol("d f h"), 1), (Symbol("d g"), 1), (Symbol("d g j"), 1), (Symbol("e g"), 1), (Symbol("e g i"), 1), (Symbol("e h"), 1), (Symbol("e h k"), 1), (Symbol("f h"), 1), (Symbol("f h j"), 1), (Symbol("f i"), 1), (Symbol("f i l"), 1), (Symbol("g i"), 1), (Symbol("g i k"), 1), (Symbol("g j"), 1), (Symbol("g j m"), 1), (Symbol("h j"), 1), (Symbol("h j l"), 1), (Symbol("h k"), 1), (Symbol("h k n"), 1), (Symbol("i k"), 1), (Symbol("i k m"), 1), (Symbol("i l"), 1), (Symbol("i l o"), 1), (Symbol("j l"), 1), (Symbol("j l n"), 1), (Symbol("j m"), 1), (Symbol("j m p"), 1), (Symbol("k m"), 1), (Symbol("k m o"), 1), (Symbol("k n"), 1), (Symbol("k n q"), 1), (Symbol("l n"), 1), (Symbol("l n p"), 1), (Symbol("l o"), 1), (Symbol("m o"), 1), (Symbol("m o q"), 1), (Symbol("m p"), 1), (Symbol("n p"), 1), (Symbol("n q"), 1), (Symbol("o q"), 1)]
    @test sort(a) == sort(b)
end

@testset "Tokenizer, DVEC, and vectorize" begin # test_vmodel
    config = TextConfig()
    config.nlist = [1]
    config.qlist = []
    config.slist = []
    config.group_usr = false

    @test tokenize(config, text1) == [Symbol(h) for h in ["hello", "world", "!!",  "@user", ";)", "#jello", ".", "world", ":)"]]
    model = fit(VectorModel, config, corpus)
    x = vectorize(model, TfModel, text1)
    @test nnz(x) == 8
    x = vectorize(model, TfModel, text2)
    @test nnz(x) == 0
end

    
const sentiment_corpus = ["me gusta", "me encanta", "lo odio", "odio esto", "me encanta esto LOL!"]
const sentiment_labels = categorical(["pos", "pos", "neg", "neg", "pos"])
const sentiment_msg = "lol, esto me encanta"

@testset "DistModel tests" begin
    config = TextConfig()
    config.nlist = [1]
    
    dmodel = fit(DistModel, config, sentiment_corpus, sentiment_labels)
    @show sentiment_msg
    @show dmodel
    #a = vectorize(dmodel, TfIdf, sentiment_msg)
    #b = [(:me1,1.0),(:me2,0.0),(:encanta1,1.0),(:encanta2,0.0),(:esto1,0.4),(:esto2,0.6),(:lol1,1.0),(:lol2,0.0)]
    #@test string(a) == string(b)

end

@testset "EntModel tests" begin
    config = TextConfig()
    config.nlist = [1]
    dmodel = fit(DistModel, config, sentiment_corpus, sentiment_labels, weights=:balance, smooth=1, minocc=1)
    emodel = fit(EntModel, dmodel)
    emodel_ = fit(EntModel, config, sentiment_corpus, sentiment_labels, weights=:balance, smooth=1, minocc=1)
    a = vectorize(emodel, sentiment_corpus)
    b = vectorize(emodel_, sentiment_corpus)
    @test 0.999 < dot(a, b)
 end


@testset "distances" begin
    u = Dict(:el => 0.9, :hola => 0.1, :mundo => 0.2) |> normalize!
    v = Dict(:el => 0.4, :hola => 0.2, :mundo => 0.4) |> normalize!
    w = Dict(:xel => 0.4, :xhola => 0.2, :xmundo => 0.4) |> normalize!

    dist = angle_distance
    @test dist(u, v) ≈ 0.5975474808029686
    @test dist(u, u) <= eps(Float32)
    @test dist(w, u) ≈ 1.5707963267948966
end

@testset "operations" begin
    u = Dict(:el => 0.1, :hola => 0.2, :mundo => 0.4)
    v = Dict(:el => 0.2, :hola => 0.4, :mundo => 0.8)
    w = Dict(:el => 0.1^2, :hola => 0.2^2, :mundo => 0.4^2)
    y = Dict(:el => 0.1/9, :hola => 0.2/9, :mundo => 0.4/9)
    @test u == u
    @test u != v
    @test u + u == v
    @test u * u == w
    @test u * (1/9) == y
    @test (1/9) * u == y
    @test dot(normalize!(u + v - v), normalize!(u)) > 0.99
end

_corpus = [
    "la casa roja",
    "la casa verde",
    "la casa azul",
    "la manzana roja",
    "la pera verde esta rica",
    "la manzana verde esta rica",
    "la hoja verde",
]

@testset "transpose bow" begin
    config = TextConfig()
    config.nlist = [1]
    config.qlist = []
    config.slist = []
    model = fit(VectorModel, config, _corpus)
    X = [vectorize(model, FreqModel, x) for x in _corpus]
    dX = transpose(X)
end

function are_posting_lists_sorted(invindex)
    for (k, lists) in invindex.lists
        if issorted([p.id for p in lists]) == false
            return false
        end
    end

    true
end

@testset "intersection" begin
    Random.seed!(1)
    for i in 1:100
        A = shuffle!(collect(1:1000))[1:10] |> sort!
        B = shuffle!(collect(1:1000))[1:100] |> sort!
        C = intersection([A, B])
        C_ = sort!(intersect(A, B))
        @test C == C_
        if C != C_
            @info A
            @info B
            @info C C_
            exit(-1)
        end
    end
end

@testset "invindex" begin
    config = TextConfig()
    #config.qlist = [3, 4]
    #config.nlist = [1, 2, 3]
    config.nlist = [1]

    model = fit(VectorModel, config, _corpus)
    invindex = fit(InvIndex, [vectorize(model, TfidfModel, text) for text in _corpus])
    @test are_posting_lists_sorted(invindex)
    begin # searching
        q = vectorize(model, TfidfModel, "la casa roja")
        res = search_with_union(invindex, cosine_distance, q, KnnResult(4))
        @test sort([r.id for r in res]) == [1, 2, 3, 4]

        res = search_with_one_error(invindex, cosine_distance, q, KnnResult(4))
        @info "ONE-ERROR" res
        res = search_with_intersection(invindex, cosine_distance, q, KnnResult(4))
        @test [r.id for r in res] == [1]

        q = vectorize(model, TfidfModel, "esta rica")
        res = search_with_intersection(invindex, cosine_distance, q, KnnResult(4))
        @test [5, 6] == sort!([r.id for r in res])
    end

    shortindex = prune(invindex, 3)
    @test are_posting_lists_sorted(invindex)
    q = vectorize(model, TfidfModel, "la casa roja")
    res = search_with_union(shortindex, cosine_distance, q, KnnResult(4))
    @test sort!([r.id for r in res]) == [1, 2, 3, 4]

    begin # searching with intersection
        res = search_with_intersection(shortindex, cosine_distance, q, KnnResult(4))
        @test [r.id for r in res] == [1]

        q = vectorize(model, TfidfModel, "esta rica")
        res = search_with_intersection(shortindex, cosine_distance, q, KnnResult(4))
        @info res
        @test [5, 6] == sort!([r.id for r in res])
    end
end

@testset "centroid computing" begin
    config = TextConfig()
    config.nlist = [1]
    config.qlist = []
    config.slist = []
    model = fit(VectorModel, config, _corpus)
    X = [vectorize(model, FreqModel, x) for x in _corpus]
    x = sum(X) |> normalize!
    vec = bow(model, x)
    expected = Dict(:la => 0.7366651330405098,:verde => 0.39921969741172364,:azul => 0.11248181187626208,:pera => 0.08712803682959973,:esta => 0.17425607365919946,:roja => 0.22496362375252416,:hoja => 0.11248181187626208,:casa => 0.33744543562878626,:rica => 0.17425607365919946,:manzana => 0.19960984870586182)
    @test 0.999 < dot(vec, expected)
end


@testset "neardup" begin
    config = TextConfig()
    config.nlist = [1]
    config.qlist = []
    config.slist = []

    function create_corpus()
        alphabet = Char.(97:100)
        corpus = String[]
        L = Char[]
        for i in 1:10000
            resize!(L, 0)
            for j in 1:7
                append!(L, rand(alphabet, 2))
                push!(L, ' ')
            end

            push!(corpus, join(L))
        end

        corpus
    end

    corpus = create_corpus()
    model = fit(VectorModel, config, corpus)
    @show corpus[1:10]
    X = [vectorize(model, TfModel, x) for x in corpus]
    L, D  = neardup(X, 0.2)
    @test length(X) > length(unique(L))
end
