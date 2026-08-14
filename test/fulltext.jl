using Test, SimilaritySearch, TextSearch

@testset "FullText and TextInvertedFile" begin
    corpus = [
        "la casa roja",
        "la casa verde",
        "la casa azul",
        "la manzana roja",
        "la pera verde esta rica",
        "la manzana verde esta rica",
        "la hoja verde",
    ]

    textconfig = TextConfig(tokenization=TokenizationConfig(nlist=[1]))
    voc = Vocabulary(textconfig, corpus)

    @testset "Weighted TextInvertedFile (NormCosine)" begin
        model = VectorModel(IdfWeighting(), TfWeighting(), voc)
        idx = TextInvertedFile(model; dist=Dist.NormCosine())
        ctx = InvertedFileContext()
        
        append_items!(idx, ctx, corpus)
        @test length(idx) == length(corpus)

        push_item!(idx, ctx, "una manzana dulce")
        @test length(idx) == length(corpus) + 1

        res = search(idx, ctx, "la casa roja", knnqueue(KnnSorted, 3))
        ids = collect(IdView(res))
        @test ids[1] == 1  # exact match "la casa roja"

        # Search using pre-vectorized query
        qvec = vectorize(model, "la casa roja")
        res_vec = search(idx, ctx, qvec, knnqueue(KnnSorted, 3))
        @test collect(IdView(res_vec)) == ids
    end

    @testset "Set Metric TextInvertedFile (Jaccard)" begin
        model = VectorModel(BinaryGlobalWeighting(), BinaryLocalWeighting(), voc)
        idx = TextInvertedFile(model; dist=Dist.Sets.Jaccard())
        ctx = InvertedFileContext()

        append_items!(idx, ctx, corpus)
        @test length(idx) == length(corpus)

        res = search(idx, ctx, "la casa verde", knnqueue(KnnSorted, 3))
        ids = collect(IdView(res))
        @test ids[1] == 2  # exact match "la casa verde"
    end

    @testset "Entropy-based TextInvertedFile" begin
        labels = ["casa", "casa", "casa", "fruta", "fruta", "fruta", "planta"]
        model = VectorModel(EntropyWeighting(), TfWeighting(), voc, corpus, labels; mindocs=1, verbose=false)
        idx = TextInvertedFile(model; dist=Dist.NormCosine())
        ctx = InvertedFileContext()

        append_items!(idx, ctx, corpus)
        @test length(idx) == length(corpus)

        res = search(idx, ctx, "la manzana verde", knnqueue(KnnSorted, 3))
        ids = collect(IdView(res))
        @test 6 in ids || 4 in ids || 2 in ids
    end
end
