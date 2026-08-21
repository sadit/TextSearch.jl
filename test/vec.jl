
@testset "Tokenizer, Dict-based vectors, and vectorize" begin
    textconfig = TextConfig(normalization=NormalizationConfig(group_usr=true), tokenization=TokenizationConfig(nlist=[1]))
    voc = Vocabulary(textconfig, corpus)
    model = VectorModel(BinaryGlobalWeighting(), FreqWeighting(), voc)
    x = vectorize(model, text1)
    @show text1 => x
    @show corpus
    @show text1
    @show text2
    v = vectorize(model, text2)
    @show text2 => v
    @info v
    @test 0 == nnz(v)
end

@testset "tokenize list of strings as a single message" begin
    textconfig = TextConfig(tokenization=TokenizationConfig(nlist=[1], mark_token_type=false))
    model = VectorModel(BinaryGlobalWeighting(), FreqWeighting(), Vocabulary(textconfig, corpus))
    @test vectorize(model, ["hello ;)", "#jello world."]) == vectorize(model, "hello ;) #jello world.")
end


@testset "Tokenizer, Dict-based vectors, and vectorize" begin
    textconfig = TextConfig(normalization=NormalizationConfig(group_usr=true), tokenization=TokenizationConfig(nlist=[1]))
    voc = Vocabulary(textconfig, sentiment_corpus)
    corpus_bows = bagofwords_corpus(voc, sentiment_corpus)
    @show length(corpus), length(corpus_bows)
    model = VectorModel(EntropyWeighting(), BinaryLocalWeighting(), voc, sentiment_corpus, sentiment_labels; smooth=0, mindocs=1)
    @test (7.059714 - sum(model.weight)) < 1e-5
    model = VectorModel(EntropyWeighting(), BinaryLocalWeighting(), voc, corpus_bows, sentiment_labels; smooth=0, mindocs=1)
    @test (7.059714 - sum(model.weight)) < 1e-5
end

@testset "Weighting schemes" begin
    textconfig = TextConfig(normalization=NormalizationConfig(group_usr=true), tokenization=TokenizationConfig(nlist=[1]))
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
        @test gettrainsize(model) == gettrainsize(model_)
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

