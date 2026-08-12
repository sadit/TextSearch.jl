
@testset "invindex" begin
    textconfig = TextConfig(tokenization=TokenizationConfig(nlist=[1]))
    model = VectorModel(IdfWeighting(), TfWeighting(), Vocabulary(textconfig, _corpus))
    db = vectorize_corpus(model, _corpus)
    invindex = WeightedInvertedFile(length(model.voc))
    ctx = InvertedFileContext()
    append_items!(invindex, ctx, VectorDatabase(db))
    begin # searching
        q = vectorize(model, "la casa roja")
        R = search(invindex, ctx, q, knnqueue(KnnSorted, 4))
        @test sort!(collect(IdView(R))) == [1, 2, 3, 4]
    end
end


@testset "centroid computing" begin
    textconfig = TextConfig(tokenization=TokenizationConfig(nlist=[1]))
    model = VectorModel(BinaryGlobalWeighting(), FreqWeighting(), Vocabulary(textconfig, _corpus))
    X = vectorize_corpus(model, _corpus)
    vec = sum(X) |> normalize!
    vec = Dict(model.voc.token[t] => w for (t, w) in zip(SparseArrays.nonzeroinds(vec), SparseArrays.nonzeros(vec)))
    expected = Dict("la" => 0.7366651330405098, "verde" => 0.39921969741172364, "azul" => 0.11248181187626208, "pera" => 0.08712803682959973, "esta" => 0.17425607365919946, "roja" => 0.22496362375252416, "hoja" => 0.11248181187626208, "casa" => 0.33744543562878626, "rica" => 0.17425607365919946, "manzana" => 0.19960984870586182)
    dot_ = sum(w * get(expected, k, 0.0) for (k, w) in vec)
    @test 0.999 < dot_
end

@testset "bm25 invindex" begin
    for (i, m) in enumerate(_corpus)
        @info i => m
    end
    voc = Vocabulary(TextConfig(tokenization=TokenizationConfig(nlist=[1])), _corpus)
    voc = filter_tokens(voc) do t
        1 < t.ndocs < 5
    end
    invfile = BM25InvertedFile(voc)
    ctx = InvertedFileContext()
    append_items!(invfile, ctx, _corpus)
    R = search(invfile, ctx, "la casa de la manzana verde", knnqueue(KnnSorted, 3))
    @test collect(IdView(R)) == UInt32[0x00000006, 0x00000002, 0x00000004]
    @test Dist.evaluate(Dist.SqL2(), collect(Float32, DistView(R)), Float32[-3.3956785, -3.1118512, -2.5816276]) <= 1e-4
    @show invfile.voc
    @show invfile.bm25
end

@testset "bm25score matches BM25InvertedFile search, both for SparseVecView and SparseVector" begin
    voc = Vocabulary(TextConfig(tokenization=TokenizationConfig(nlist=[1])), _corpus)
    invfile = BM25InvertedFile(voc)
    ctx = InvertedFileContext()
    append_items!(invfile, ctx, _corpus)

    qtext = "la casa de la manzana verde"
    qvec = sparsevec(bagofwords(voc, qtext), vocsize(voc))
    R = search(invfile, ctx, qtext, knnqueue(KnnSorted, length(invfile)))
    ids = collect(IdView(R))
    dists = collect(DistView(R))
    @test length(ids) > 0

    for (docid, dist) in zip(ids, dists)
        expected = -dist  # `search` stores -bm25score as "distance"

        # doc as the SparseVecView `BM25InvertedFile` itself stores in `db`
        got_sparseview = bm25score(invfile.bm25, voc, qvec, database(invfile)[docid])
        @test got_sparseview ≈ expected atol=1f-4

        # doc rebuilt from scratch as a plain SparseVector, e.g. via bagofwords + sparsevec
        docvec = sparsevec(bagofwords(voc, _corpus[docid]), vocsize(voc))
        got_sparsevector = bm25score(invfile.bm25, voc, qvec, docvec)
        @test got_sparsevector ≈ expected atol=1f-4
    end
end

