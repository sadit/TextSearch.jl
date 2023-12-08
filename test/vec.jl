
@testset "Tokenizer, DVEC, and vectorize" begin
    textconfig = TextConfig(group_usr=true, nlist=[1])
    voc = Vocabulary(textconfig, corpus)
    model = VectorModel(BinaryGlobalWeighting(), FreqWeighting(), voc)
    x = vectorize(model, text1)
    @show text1 => x
    @show corpus
    @show text1
    @show text2
    v = vectorize(model, text2)
    @show text2 => v
    @test 1 == length(v) && v[0] == 1 # empty vectors use 0 as centinel
end

@testset "tokenize list of strings as a single message" begin
    textconfig = TextConfig(nlist=[1], mark_token_type=false)
    model = VectorModel(BinaryGlobalWeighting(), FreqWeighting(), Vocabulary(textconfig, corpus))
    @test vectorize(model, ["hello ;)", "#jello world."]) == vectorize(model, "hello ;) #jello world.")
end


@testset "Tokenizer, DVEC, and vectorize" begin
    textconfig = TextConfig(group_usr=true, nlist=[1])
    voc = Vocabulary(textconfig, sentiment_corpus)
    corpus_bows = bagofwords_corpus(voc, sentiment_corpus)
    @show length(corpus), length(corpus_bows)
    model = VectorModel(EntropyWeighting(), BinaryLocalWeighting(), voc, sentiment_corpus, sentiment_labels; smooth=0, mindocs=1)
    @test (7.059714 - sum(model.weight)) < 1e-5
    model = VectorModel(EntropyWeighting(), BinaryLocalWeighting(), voc, corpus_bows, sentiment_labels; smooth=0, mindocs=1)
    @test (7.059714 - sum(model.weight)) < 1e-5
end

@testset "Weighting schemes" begin
    textconfig = TextConfig(group_usr=true, nlist=[1])
    for (gw, lw, dot_) in [
            (BinaryGlobalWeighting(), FreqWeighting(), 0.3162),
            (BinaryGlobalWeighting(), TfWeighting(), 0.3162),
            (BinaryGlobalWeighting(), TpWeighting(), 0.3162),
            (IdfWeighting(), BinaryLocalWeighting(), 0.3668),
            (IdfWeighting(), TfWeighting(), 0.2053),

            (EntropyWeighting(), FreqWeighting(), 0.44456),
            (EntropyWeighting(), TfWeighting(), 0.44456),
            (EntropyWeighting(), TpWeighting(), 0.44456),
            (EntropyWeighting(), BinaryLocalWeighting(), 0.7029)
        ]

        voc = Vocabulary(textconfig, sentiment_corpus)

        if gw isa EntropyWeighting
            model = VectorModel(gw, lw, voc, sentiment_corpus, sentiment_labels; smooth=0, mindocs=1)
        else
            model = VectorModel(gw, lw, voc)
        end

        x = vectorize(model, sentiment_corpus[3])
        y = vectorize(model, sentiment_corpus[4])
        @show gw, lw, dot_, dot(x, y), x, y
        @test abs(dot(x, y) - dot_) < 1e-3
    end

    for (gw, lw, dot_, p) in [
            (EntropyWeighting(), BinaryLocalWeighting(), 0.7071067690849304, 0.9),
            (IdfWeighting(), TfWeighting(), 0.0, 0.9),
        ]

        voc = Vocabulary(textconfig, sentiment_corpus)
        if gw isa EntropyWeighting
            model = VectorModel(gw, lw, voc, sentiment_corpus, sentiment_labels; smooth=0, mindocs=1)
        else
            model = VectorModel(gw, lw, voc)
        end
       
        q = quantile(model.weight, p)
        model_ = filter_tokens(t -> q <= t.weight, model)
        @info "====== weight:"
        @info model.weight
        @info model_.weight
        @test trainsize(model) == trainsize(model_)
        @test vocsize(model) > vocsize(model_)
        @info "====== token:", model_.voc.token
        @info sentiment_corpus[3], sentiment_corpus[4]
        x = vectorize(model_, sentiment_corpus[3])
        y = vectorize(model_, sentiment_corpus[4])
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

