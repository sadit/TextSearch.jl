
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
    @test length(invfile) == length(_corpus)
    R = search(invfile, ctx, "la casa de la manzana verde", knnqueue(KnnSorted, 3))
    @test collect(IdView(R)) == UInt32[0x00000006, 0x00000002, 0x00000004]
    @test Dist.evaluate(Dist.SqL2(), collect(Float32, DistView(R)), Float32[-3.3956785, -3.1118512, -2.5816276]) <= 1e-4
    @show invfile.voc
    @show invfile.bm25
end

@testset "BM25InvertedFile query-time synonym expansion" begin
    synonyms = Dict("pera" => ["manzana"])

    textconfig = TextConfig(tokenization=TokenizationConfig(nlist=[1]))
    voc = Vocabulary(textconfig, _corpus)
    ctx = InvertedFileContext()

    idx_plain = BM25InvertedFile(voc)
    append_items!(idx_plain, ctx, _corpus)

    # Handing a network over IS the request to expand with it; there is no separate flag.
    idx_expand = BM25InvertedFile(voc; synonyms)
    append_items!(idx_expand, ctx, _corpus)

    res_plain = search(idx_plain, ctx, "pera roja", knnqueue(KnnSorted, 3))
    res_expand = search(idx_expand, ctx, "pera roja", knnqueue(KnnSorted, 3))

    ids_plain = collect(IdView(res_plain))
    ids_expand = collect(IdView(res_expand))
    dists_plain = collect(DistView(res_plain))
    dists_expand = collect(DistView(res_expand))

    @test 4 in ids_plain && 4 in ids_expand
    # doc4 ("la manzana roja") ranks strictly better once "pera" expands into its synonym "manzana"
    @test findfirst(==(4), ids_expand) < findfirst(==(4), ids_plain)
    @test dists_expand[findfirst(==(4), ids_expand)] < dists_plain[findfirst(==(4), ids_plain)]

    # no network attached: plain search, no error
    idx_flag_only = BM25InvertedFile(voc)
    append_items!(idx_flag_only, ctx, _corpus)
    res_flag_only = search(idx_flag_only, ctx, "pera roja", knnqueue(KnnSorted, 3))
    @test collect(IdView(res_flag_only)) == ids_plain
end

struct RecorderLog <: AbstractLog
    events::Vector{Tuple{Symbol,Int,Int}}
end
RecorderLog() = RecorderLog(Tuple{Symbol,Int,Int}[])

function SimilaritySearch.LOG(log::RecorderLog, event::Symbol, index::AbstractSearchIndex, ctx::AbstractContext, sp::Integer, ep::Integer)
    push!(log.events, (event, Int(sp), Int(ep)))
end

@testset "BM25InvertedFile LOG events: exactly-once :add!" begin
    voc = Vocabulary(TextConfig(tokenization=TokenizationConfig(nlist=[1])), _corpus)
    voc = filter_tokens(voc) do t
        1 < t.ndocs < 5
    end
    n = length(_corpus)

    # fused append_items! (raw text): exactly one :add! covering the whole freshly-built index
    rec = RecorderLog()
    invfile = BM25InvertedFile(voc)
    ctx = InvertedFileContext(logger=rec)
    append_items!(invfile, ctx, _corpus)
    @test rec.events == [(:add!, 1, n)]

    # single push_item!: exactly one :add! for just the new document
    empty!(rec.events)
    push_item!(invfile, ctx, "la casa verde")
    @test rec.events == [(:add!, n + 1, n + 1)]

    # decoupled path (db grown directly, then index!): exactly one :add! for the new block
    empty!(rec.events)
    invfile2 = BM25InvertedFile(voc)
    ctx2 = InvertedFileContext(logger=rec)
    append_items!(database(invfile2), database(invfile))
    index!(invfile2, ctx2)
    @test rec.events == [(:add!, 1, n + 1)]
end

@testset "BM25InvertedFile decoupled index!" begin
    voc = Vocabulary(TextConfig(tokenization=TokenizationConfig(nlist=[1])), _corpus)
    voc = filter_tokens(voc) do t
        1 < t.ndocs < 5
    end
    ctx = InvertedFileContext()

    invfile_fused = BM25InvertedFile(voc)
    append_items!(invfile_fused, ctx, _corpus)
    @test length(invfile_fused) == length(_corpus)

    # decoupled path: grow db directly with the already-encoded SparseVecViews from the fused
    # index, bypassing the raw-text encode+register+store fusion, then catch up with index!
    invfile_deferred = BM25InvertedFile(voc)
    append_items!(database(invfile_deferred), database(invfile_fused))
    @test length(invfile_deferred) == 0
    @test length(database(invfile_deferred)) == length(_corpus)
    index!(invfile_deferred, ctx)
    @test length(invfile_deferred) == length(_corpus) == length(database(invfile_deferred))

    q = "la casa de la manzana verde"
    R_fused = search(invfile_fused, ctx, q, knnqueue(KnnSorted, 3))
    R_deferred = search(invfile_deferred, ctx, q, knnqueue(KnnSorted, 3))
    @test collect(IdView(R_fused)) == collect(IdView(R_deferred))
    @test collect(DistView(R_fused)) ≈ collect(DistView(R_deferred))

    # no-op index!: nothing new appended, must not error or change state
    index!(invfile_deferred, ctx)
    @test length(invfile_deferred) == length(_corpus)
    R_deferred2 = search(invfile_deferred, ctx, q, knnqueue(KnnSorted, 3))
    @test collect(IdView(R_deferred)) == collect(IdView(R_deferred2))

    # push_item! directly on db also outgrows the index until index! catches up
    push_item!(database(invfile_deferred), database(invfile_fused)[1])
    @test length(invfile_deferred) == length(_corpus)
    @test length(database(invfile_deferred)) == length(_corpus) + 1
    index!(invfile_deferred, ctx)
    @test length(invfile_deferred) == length(_corpus) + 1
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

