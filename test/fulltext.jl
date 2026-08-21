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

    @testset "Query-time synonym expansion (expand_query_synonyms)" begin
        # corpus[4] = "la manzana roja"; a query for "pera roja" should rank it higher
        # once "pera" is registered as a synonym of "manzana".
        synonyms = Dict("pera" => ["manzana"])

        expand_textconfig = TextConfig(tokenization=TokenizationConfig(nlist=[1]), expand_query_synonyms=true)
        expand_voc = Vocabulary(expand_textconfig, corpus)
        expand_model = VectorModel(IdfWeighting(), TfWeighting(), expand_voc)
        idx_expand = TextInvertedFile(expand_model; dist=Dist.NormCosine(), synonyms)
        ctx = InvertedFileContext()
        append_items!(idx_expand, ctx, corpus)

        plain_model = VectorModel(IdfWeighting(), TfWeighting(), voc)
        idx_plain = TextInvertedFile(plain_model; dist=Dist.NormCosine())
        append_items!(idx_plain, ctx, corpus)

        res_expand = search(idx_expand, ctx, "pera roja", knnqueue(KnnSorted, 3))
        res_plain = search(idx_plain, ctx, "pera roja", knnqueue(KnnSorted, 3))

        ids_expand = collect(IdView(res_expand))
        ids_plain = collect(IdView(res_plain))
        dists_expand = collect(DistView(res_expand))
        dists_plain = collect(DistView(res_plain))

        @test 4 in ids_plain && 4 in ids_expand
        # doc4 ("la manzana roja") ranks strictly better once "pera" expands into its synonym "manzana"
        @test findfirst(==(4), ids_expand) < findfirst(==(4), ids_plain)
        @test dists_expand[findfirst(==(4), ids_expand)] < dists_plain[findfirst(==(4), ids_plain)]

        # the flag alone, without an attached synonyms dict, must not error and must behave like plain search
        idx_flag_only = TextInvertedFile(expand_model; dist=Dist.NormCosine())
        append_items!(idx_flag_only, ctx, corpus)
        res_flag_only = search(idx_flag_only, ctx, "pera roja", knnqueue(KnnSorted, 3))
        @test collect(IdView(res_flag_only)) == ids_plain
    end
end
