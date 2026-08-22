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

    @testset "Query-time expansion" begin
        # corpus[4] = "la manzana roja"; a query for "pera roja" should rank it higher
        # once "pera" is registered as a query_expansion of "manzana".
        query_expansion = Dict("pera" => ["manzana"])

        # Handing a network over IS the request to expand with it; there is no separate flag.
        # Whether a profile wants that is recorded as its applied.query_expansion, not on the config.
        expand_model = VectorModel(IdfWeighting(), TfWeighting(), voc)
        idx_expand = TextInvertedFile(expand_model; dist=Dist.NormCosine(), query_expansion)
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
        # doc4 ("la manzana roja") ranks strictly better once "pera" expands into its query_expansion "manzana"
        @test findfirst(==(4), ids_expand) < findfirst(==(4), ids_plain)
        @test dists_expand[findfirst(==(4), ids_expand)] < dists_plain[findfirst(==(4), ids_plain)]

        # no network attached: plain search, no error
        idx_flag_only = TextInvertedFile(expand_model; dist=Dist.NormCosine())
        append_items!(idx_flag_only, ctx, corpus)
        res_flag_only = search(idx_flag_only, ctx, "pera roja", knnqueue(KnnSorted, 3))
        @test collect(IdView(res_flag_only)) == ids_plain
    end
end

@testset "avgdoclen agrees with the lengths BM25 actually measures" begin
    # BM25's length normalization is `doclen / avgdoclen`, with `doclen` measured at index time
    # as a document's total occurrences and `avgdoclen` read off the training vocabulary. The
    # two have to be the same notion of length or the ratio is not 1 for an average document:
    # when `numtokens` counted distinct tokens instead, this ratio was 3.47 on Spanish
    # Wikipedia, driving the normalization as if b were 2.6 rather than 0.75.
    tc = TextConfig(tokenization=TokenizationConfig(nlist=[1]))
    docs = ["la casa la casa roja tiene un jardin muy grande y bonito",
            "la pera verde verde esta muy rica",
            "el jardin de la casa verde",
            "casa"]
    voc = Vocabulary(tc, docs; verbose=false)
    idx = BM25InvertedFile(voc)
    append_items!(idx, docs)

    @test sum(idx.doclens) == getnumtokens(voc)
    @test sum(idx.doclens) / length(idx.doclens) ≈ avgdoclen(voc)
    for (i, d) in enumerate(docs)
        @test idx.doclens[i] == length(tokenize(tc, d))
    end
end
