
@testset "invindex" begin
    textconfig = TextConfig(nlist=[1])
    model = VectorModel(IdfWeighting(), TfWeighting(), Vocabulary(textconfig, _corpus))
    db = vectorize_corpus(model, _corpus)
    invindex = WeightedInvertedFile(length(model.voc))
    ctx = InvertedFileContext()
    append_items!(invindex, ctx, VectorDatabase(db))
    begin # searching
        q = vectorize(model, "la casa roja")
        R = search(invindex, ctx, q, KnnResult(4))
        @test sort!([p.id for p in R.res]) == [1, 2, 3, 4]
    end
end


@testset "centroid computing" begin
    textconfig = TextConfig(nlist=[1])
    model = VectorModel(BinaryGlobalWeighting(), FreqWeighting(), Vocabulary(textconfig, _corpus))
    X = vectorize_corpus(model, _corpus)
    vec = sum(X) |> normalize!
    vec = Dict(model.voc.token[t] => w for (t, w) in vec)
    expected = Dict("la" => 0.7366651330405098, "verde" => 0.39921969741172364, "azul" => 0.11248181187626208, "pera" => 0.08712803682959973, "esta" => 0.17425607365919946, "roja" => 0.22496362375252416, "hoja" => 0.11248181187626208, "casa" => 0.33744543562878626, "rica" => 0.17425607365919946, "manzana" => 0.19960984870586182)
    @test 0.999 < dot(vec, expected)
end

@testset "bm25 invindex" begin
    for (i, m) in enumerate(_corpus)
        @info i => m
    end
    invfile = BM25InvertedFile(TextConfig(nlist=[1]), _corpus) do t
        1 < t.ndocs < 5
    end
    ctx = InvertedFileContext()
    append_items!(invfile, ctx, _corpus)
    R = search(invfile, ctx, "la casa de la manzana verde", KnnResult(3))
    @test collect(IdView(R.res)) == UInt32[0x00000006, 0x00000002, 0x00000004]
    @test evaluate(SqL2Distance(), collect(DistView(R.res)), Float32[-3.3956785, -3.1118512, -2.5816276]) <= 1e-4
    @show invfile.voc
    @show invfile.bm25
end

@testset "bm25 invindex" begin
    invfile = BM25InvertedFile(TextConfig(nlist=[1]), _corpus)
    ctx = InvertedFileContext()
    append_items!(invfile, ctx, _corpus)
    filter_lists!(invfile;
                  list_min_length_for_checking=2,
                  list_max_allowed_length=3,
                  doc_min_freq=1,
                  doc_max_freq=3)
    R = search(invfile, ctx, "la casa de la manzana verde", KnnResult(3))
    @test collect(IdView(R.res)) == UInt32[0x00000006, 0x00000002, 0x00000004]
    @show collect(DistView(R.res))
    @show invfile.voc
    @show invfile.bm25

    @testset "saveindex and loadindex BM25InvertedFile" begin
        tmpfile = tempname()
        @info "--- load and save!!!"
        saveindex(tmpfile, invfile; meta=[1, 2, 4, 8], store_db=false)

            G, meta = loadindex(tmpfile, database(invfile); staticgraph=true)
            @test meta == [1, 2, 4, 8]
            @test G.adj isa StaticAdjacencyList
            R = search(G, ctx, "la casa de la manzana verde", KnnResult(3))
            @test collect(IdView(R.res)) == UInt32[0x00000006, 0x00000002, 0x00000004]
    end

end

