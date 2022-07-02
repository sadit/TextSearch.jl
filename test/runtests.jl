using SimilaritySearch, TextSearch, InvertedFiles
using Test, SparseArrays, LinearAlgebra, CategoricalArrays, StatsBase, Random
const fit = TextSearch.fit

const text0 = "@user;) #jello.world"
const text1 = "hello world!! @user;) #jello.world :)"
const text2 = "a b c d e f g h i j k l m n o p q"
const corpus = ["hello world :)", "@user;) excellent!!", "#jello world."]


@testset "DVEC" begin
    cmpex(u, v) = abs(u[1] - v[1]) < 1e-3 && u[2] == v[2]

    aL = []
    AL = []
    for i in 1:10
        A = rand(300)
        B = rand(300)
        a = SVEC(k => v for (k, v) in enumerate(A))
        b = SVEC(k => v for (k, v) in enumerate(B))

        @test abs(norm(A) - norm(a)) < 1e-3
        @test abs(norm(B) - norm(b)) < 1e-3
        normalize!(A); normalize!(a)
        normalize!(B); normalize!(b)
        @test abs(norm(a) - 1.0) < 1e-3
        @test abs(norm(b) - 1.0) < 1e-3

        @test abs(dot(a, b) - dot(A, B)) < 1e-3
        @test abs(maximum(a) - maximum(A)) < 1e-3
        @test abs(minimum(a) - minimum(A)) < 1e-3

        
        @test cmpex(findmax(a), findmax(A))
        @test cmpex(findmin(a), findmin(A))

        push!(aL, a)
        push!(AL, A)
    end

    @test (norm(sum(AL)) - norm(sum(aL))) < 1e-3

    adist = AngleDistance()
    cdist = CosineDistance()

    for i in 1:length(aL)-1
        @test abs(evaluate(adist, aL[i], aL[i+1]) - evaluate(adist, AL[i], AL[i+1])) < 1e-3
        @test abs(evaluate(cdist, aL[i], aL[i+1]) - evaluate(cdist, AL[i], AL[i+1])) < 1e-3
    end
end

@testset "individual tokenizers" begin
    m = TextConfig(nlist=[1])
    @test tokenize(m, text0) == ["@user", ";)", "#jello", ".", "world"]

    m = TextConfig(nlist=[2])
    @test tokenize(m, text0) == ["\tn@user ;)", "\tn;) #jello", "\tn#jello .", "\tn. world"]

    m = TextConfig(nlist=[3])
    @test tokenize(m, text0) == ["\tn@user ;) #jello", "\tn;) #jello .", "\tn#jello . world"]

    m = TextConfig(qlist=[3])
    @test tokenize(m, text0) == map(p -> "\tq" * p, [" @u", "@us", "use", "ser", "er;", "r;)", ";) ", ") #", " #j", "#je", "jel", "ell", "llo", "lo.", "o.w", ".wo", "wor", "orl", "rld", "ld "])
    m = TextConfig(nlist=[1])
    @test tokenize(m, text1) == ["hello", "world", "!!", "@user", ";)", "#jello", ".", "world", ":)"]

    m = TextConfig(slist=[Skipgram(2,1)])
    @test tokenize(m, text1) == map(p -> "\ts" * p, ["hello !!", "world @user", "!! ;)", "@user #jello", ";) .", "#jello world", ". :)"])

end

@testset "message vectors" begin
    m = TextConfig(nlist=[1, 2])
    A = tokenize(m, "hello ;) #jello world.")
    B = tokenize(m, ["hello ;)", "#jello world."])
    push!(B, "\tn;) #jello")
    @test sort(A) == sort(B)
    # @show sort(A) sort(B)
end

@testset "vocabulary of different kinds of docs" begin
    tok = TextConfig(nlist=[1])
    A = Vocabulary(tok, ["hello ;)", "#jello world."])
    B = Vocabulary(tok, [["hello ;)", "#jello world."]])
    @test A.occs == B.occs
    @test sort(A.token) == sort(B.token)
    @info A.corpuslen, B.corpuslen
    @test A.corpuslen == 2 && B.corpuslen == 1
end

@testset "Normalize and tokenize" begin
    tok = TextConfig(del_punc=true, group_usr=true, nlist=[1, 2, 3])
    @test tokenize(tok, text1) == ["hello", "world", "_usr", "#jello", "world", "\tnhello world", "\tnworld _usr", "\tn_usr #jello", "\tn#jello world", "\tnhello world _usr", "\tnworld _usr #jello", "\tn_usr #jello world"]
end

@testset "Normalize and tokenize bigrams and trigrams" begin
    tok = TextConfig(del_punc=true, group_usr=true, nlist=[2, 3])
    @test tokenize(tok, text1) == ["\tnhello world", "\tnworld _usr", "\tn_usr #jello", "\tn#jello world", "\tnhello world _usr", "\tnworld _usr #jello", "\tn_usr #jello world"]
end

@testset "Normalize and tokenize" begin
    tok = TextConfig(del_punc=false, group_usr=true, nlist=[1])
    text3 = "a ab __b @@c ..!d ''e \"!\"f +10 -20 30 40.00 .50 6.0 7.. ======= !()[]{}"
    @test tokenize(tok, text3) == ["a", "ab", "__b", "@_usr", "..!", "d", "''", "e", "\"!\"", "f", "0", "0", "0", "0", "0", "0", "0", ".", "=======", "!()", "[]{", "}"]
end

@testset "Tokenize skipgrams" begin
    tok = TextConfig(del_punc=false, group_usr=false, slist=[Skipgram(3,1)])
    tokens = tokenize(tok, text1)
    @show text1 tokens
    @test tokens == ["\tshello !! ;)", "\tsworld @user #jello", "\ts!! ;) .", "\ts@user #jello world", "\ts;) . :)"]

    config = TextConfig(del_punc=false, group_usr=false, nlist=[], slist=[Skipgram(3,1), Skipgram(2, 1)])
    tokens = tokenize(config, text1)
    @show text1 tokens
    @test tokens == ["\tshello !!", "\tsworld @user", "\ts!! ;)", "\ts@user #jello", "\ts;) .", "\ts#jello world", "\ts. :)", "\tshello !! ;)", "\tsworld @user #jello", "\ts!! ;) .", "\ts@user #jello world", "\ts;) . :)"]
end

@testset "Vocabulary and BOW" begin
    tok = TextConfig(nlist=[1])
    C = tokenize_corpus(tok, corpus)
    voc = Vocabulary(C)
    @test vectorize_corpus(voc, tok, C) == Dict{UInt32, Int32}[Dict(0x00000002 => 1, 0x00000003 => 1, 0x00000001 => 1), Dict(0x00000005 => 1, 0x00000004 => 1, 0x00000006 => 1, 0x00000007 => 1), Dict(0x00000002 => 1, 0x00000009 => 1, 0x00000008 => 1)]
end

@testset "Tokenizer, DVEC, and vectorize" begin
    tok = TextConfig(group_usr=true, nlist=[1])
    voc = Vocabulary(tok, corpus)
    model = VectorModel(BinaryGlobalWeighting(), FreqWeighting(), voc)
    x = vectorize(model, tok, text1)
    @show text1 => x
    @show corpus
    @show text1
    @show text2
    v = vectorize(model, tok, text2)
    @test 1 == length(v) && v[0] == 1 # empty vectors use 0 as centinel
end

@testset "tokenize list of strings as a single message" begin
    tok = TextConfig(nlist=[1])
    model = VectorModel(BinaryGlobalWeighting(), FreqWeighting(), tok, corpus)
    @test vectorize(model, tok, ["hello ;)", "#jello world."]) == vectorize(model, tok, "hello ;) #jello world.")
end

###########
###########
###########
###########

const sentiment_corpus = ["me gusta", "me encanta", "lo lo odio", "odio esto", "me encanta esto LOL!"]
const sentiment_labels = categorical(["pos", "pos", "neg", "neg", "pos"])
const sentiment_msg = "lol, esto me encanta"

@testset "Tokenizer, DVEC, and vectorize" begin
    tok = TextConfig(group_usr=true, nlist=[1])
    voc = Vocabulary(tok, sentiment_corpus)
    corpus_bows = vectorize_corpus(voc, tok, corpus)
    model = VectorModel(EntropyWeighting(), BinaryLocalWeighting(), voc, corpus_bows, sentiment_labels)
    @info sum(voc.weight)
    @test (7.059714 - sum(model.voc.weight)) < 1e-5
    model = VectorModel(EntropyWeighting(), BinaryLocalWeighting(), tok, corpus, sentiment_labels)
    @test (7.059714 - sum(model.voc.weight)) < 1e-5
end

@testset "Weighting schemes" begin
    tok = TextConfig(group_usr=true, nlist=[1])
    for (gw, lw, dot_) in [
            (BinaryGlobalWeighting(), FreqWeighting(), 0.3162),
            (BinaryGlobalWeighting(), TfWeighting(), 0.3162),
            (BinaryGlobalWeighting(), TpWeighting(), 0.3162),
            (IdfWeighting(), BinaryLocalWeighting(), 0.5000),
            (IdfWeighting(), TfWeighting(), 0.3162),

            (EntropyWeighting(), FreqWeighting(), 0.44456),
            (EntropyWeighting(), TfWeighting(), 0.44456),
            (EntropyWeighting(), TpWeighting(), 0.44456),
            (EntropyWeighting(), BinaryLocalWeighting(), 0.70290)
        ]

        if gw isa EntropyWeighting
            model = VectorModel(gw, lw, tok, sentiment_corpus, sentiment_labels)
        else
            model = VectorModel(gw, lw, tok, sentiment_corpus)
        end

        x = vectorize(model, tok, sentiment_corpus[3])
        y = vectorize(model, tok, sentiment_corpus[4])
        @show gw, lw, dot_, dot(x, y), x, y
        @test abs(dot(x, y) - dot_) < 1e-3
    end

    
    for (gw, lw, dot_, p) in [
            (EntropyWeighting(), BinaryLocalWeighting(), 0.7071, 0.9),
            (IdfWeighting(), TfWeighting(), 0.3162, 0.9),
        ]
        if gw isa EntropyWeighting
            model = VectorModel(gw, lw, tok, sentiment_corpus, sentiment_labels)
        else
            model = VectorModel(gw, lw, tok, sentiment_corpus)
        end

        model_ = prune_select_top(model, p)
        @test trainsize(model) == trainsize(model_)
        @test vocsize(model) > vocsize(model_)

        x = vectorize(model_, tok, sentiment_corpus[3])
        y = vectorize(model_, tok, sentiment_corpus[4])
        @show "=========", x, y, norm(x), norm(y)
        @show gw, lw, dot(x, y), dot_, x, y
        @test abs(dot(x, y) - dot_) < 1e-3
    end
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

@testset "invindex" begin
    tok = TextConfig(nlist=[1])
    model = VectorModel(IdfWeighting(), TfWeighting(), tok, _corpus)
    db = vectorize_corpus(model, tok, _corpus)
    invindex = WeightedInvertedFile(length(model.voc))
    append!(invindex, VectorDatabase(db))
    begin # searching
        q = vectorize(model, tok, "la casa roja")
        p = search(invindex, q, KnnResult(4))
        @test sort!(collect(idview(p.res))) == [1, 2, 3, 4]
    end
end

@testset "centroid computing" begin
    tok = TextConfig(nlist=[1])
    model = VectorModel(BinaryGlobalWeighting(), FreqWeighting(), tok, _corpus)
    X = vectorize_corpus(model, tok, _corpus)
    vec = sum(X) |> normalize!
    vec = Dict(model.voc.token[t] => w for (t, w) in vec)
    expected = Dict("la" => 0.7366651330405098, "verde" => 0.39921969741172364, "azul" => 0.11248181187626208, "pera" => 0.08712803682959973, "esta" => 0.17425607365919946, "roja" => 0.22496362375252416, "hoja" => 0.11248181187626208, "casa" => 0.33744543562878626, "rica" => 0.17425607365919946, "manzana" => 0.19960984870586182)
    @test 0.999 < dot(vec, expected)
end

@info "FINISH"