# using Languages
using SimilaritySearch, TextSearch
using Test, SparseArrays, LinearAlgebra, CategoricalArrays, StatsBase, Random
const fit = TextSearch.fit

const text0 = "@user;) #jello.world"
const text1 = "hello world!! @user;) #jello.world :)"
const text2 = "a b c d e f g h i j k l m n o p q"
const corpus = ["hello world :)", "@user;) excellent!!", "#jello world."]

@testset "individual tokenizers" begin
    @show text0
    @test qgrams(text0, 1) == ["@", "u", "s", "e", "r", ";", ")", " ", "#", "j", "e", "l", "l", "o", ".", "w", "o", "r", "l", "d"] 
    @test qgrams(text0, 3) == ["@us", "use", "ser", "er;", "r;)", ";) ", ") #", " #j", "#je", "jel", "ell", "llo", "lo.", "o.w", ".wo", "wor", "orl", "rld"]
    @show text1
    @test unigrams(text0) == ["@user", ";)", "#jello", ".", "world"]
    @test unigrams(text1) == ["hello", "world", "!!", "@user", ";)", "#jello", ".", "world", ":)"]
    @test unigrams(corpus[1]) == ["hello", "world", ":)"]
    @test unigrams(corpus[2]) == ["@user", ";)", "excellent", "!!"]
    @test unigrams(corpus[3]) == ["#jello", "world", "."]
    @show corpus
end

@testset "Normalize and tokenize" begin
    config = TextConfig(del_punc=true, group_usr=true, nlist=Int8[1, 2, 3])
    t = normalize_text(config, text1)
    @test t == [' ', 'h', 'e', 'l', 'l', 'o', ' ', 'w', 'o', 'r', 'l', 'd', ' ', ' ', '_', 'u', 's', 'r', ' ', ' ', '#', 'j', 'e', 'l', 'l', 'o', ' ', 'w', 'o', 'r', 'l', 'd', ' ', ' ']
    @test tokenize(config, t) == ["hello", "world", "_usr", "#jello", "world", "hello world", "world _usr", "_usr #jello", "#jello world", "hello world _usr", "world _usr #jello", "_usr #jello world"]
end

@testset "Tokenize skipgrams" begin
    config = TextConfig(del_punc=true, group_usr=true, slist=[Skipgram(2,1)])
    t = normalize_text(config, text1)
    @test t == [' ', 'h', 'e', 'l', 'l', 'o', ' ', 'w', 'o', 'r', 'l', 'd', ' ', ' ', '_', 'u', 's', 'r', ' ', ' ', '#', 'j', 'e', 'l', 'l', 'o', ' ', 'w', 'o', 'r', 'l', 'd', ' ', ' ']
    @info join(t)
    @test tokenize(config, t) == ["hello", "world", "_usr", "#jello", "world", "hello _usr", "world #jello", "_usr world"]
end

@testset "Tokenizer, DVEC, and vectorize" begin # test_vmodel
    config = TextConfig(group_usr=true, nlist=Int8[1])
    #t = normalize_text(config, text1)
    #tokenize(config, t)
    model = VectorModel(FreqWeighting, compute_bow_multimessage(config, corpus))
    x = vectorize(model, compute_bow(config, text1))
    @show corpus
    @show text1
    @test nnz(x) == 8
    x = vectorize(model, compute_bow(config, text2))
    @test nnz(x) == 0
end

const sentiment_corpus = ["me gusta", "me encanta", "lo odio", "odio esto", "me encanta esto LOL!"]
const sentiment_labels = categorical(["pos", "pos", "neg", "neg", "pos"])
const sentiment_msg = "lol, esto me encanta"

@testset "DistModel tests" begin
    config = TextConfig(nlist=Int8[1])
    corpus_ = compute_bow_list(config, sentiment_corpus)
    @show corpus_
    dmodel = DistModel(corpus_, sentiment_labels)
    @show sentiment_msg
    @show dmodel
    # test?
end

@testset "EntModel tests" begin
    config = TextConfig(nlist=Int8[1])
    dmodel = DistModel(compute_bow_list(config, sentiment_corpus), sentiment_labels, weights=:balance, smooth=1, minocc=1)
    emodel = EntModel(dmodel, EntWeighting)
    emodel_ = EntModel(EntWeighting, compute_bow_list(config, sentiment_corpus), sentiment_labels, weights=:balance, smooth=1, minocc=1)
    a = vectorize(emodel, compute_bow_multimessage(config, sentiment_corpus))
    b = vectorize(emodel_, compute_bow_multimessage(config, sentiment_corpus))
    @test 0.999 < dot(a, b)
end


@testset "distances" begin
    u = Dict(:el => 0.9, :hola => 0.1, :mundo => 0.2) |> normalize!
    v = Dict(:el => 0.4, :hola => 0.2, :mundo => 0.4) |> normalize!
    w = Dict(:xel => 0.4, :xhola => 0.2, :xmundo => 0.4) |> normalize!

    dist = AngleDistance()
    @test evaluate(dist, u, v) ≈ 0.5975474808029686
    @test evaluate(dist, u, u) <= eps(Float32)
    @test evaluate(dist, w, u) ≈ 1.5707963267948966
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

exit(0)

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
    invindex = InvIndex([vectorize(model, TfidfModel, text) for text in _corpus])
    @test are_posting_lists_sorted(invindex)
    begin # searching
        q = vectorize(model, TfidfModel, "la casa roja")
        res = search_with_union(invindex, q, KnnResult(4))
        @test sort([r.id for r in res]) == [1, 2, 3, 4]

        res = search_with_one_error(invindex, q, KnnResult(4))
        @info "ONE-ERROR" res
        res = search_with_intersection(invindex, q, KnnResult(4))
        @test [r.id for r in res] == [1]

        q = vectorize(model, TfidfModel, "esta rica")
        res = search_with_intersection(invindex, q, KnnResult(4))
        @test [5, 6] == sort!([r.id for r in res])
    end

    shortindex = prune(invindex, 3)
    @test are_posting_lists_sorted(invindex)
    q = vectorize(model, TfidfModel, "la casa roja")
    res = search_with_union(shortindex, q, KnnResult(4))
    @test sort!([r.id for r in res]) == [1, 2, 3, 4]

    begin # searching with intersection
        res = search_with_intersection(shortindex, q, KnnResult(4))
        @test [r.id for r in res] == [1]

        q = vectorize(model, TfidfModel, "esta rica")
        res = search_with_intersection(shortindex, q, KnnResult(4))
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
