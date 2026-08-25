using Test, TextSearch, SimilaritySearch, SparseArrays

@testset "query pipeline" begin
    corpus = ["la casa roja tiene jardin", "la casa verde tiene jardin",
              "una manzana roja y una pera", "la pera verde esta rica",
              "la manzana verde esta rica", "el jardin azul es grande"]
    voc = Vocabulary(TextConfig(), corpus; verbose=false)
    model = VectorModel(IdfWeighting(), TfWeighting(), voc)
    lsi = LatentSemanticIndexing(model, corpus; maxoutdim=4, verbose=false)
    net = query_expansion(lsi, 3; verbose=false)

    @testset "it reproduces expand_query! exactly, weight for weight" begin
        # The reason this matters: unifying the two query paths must not move any score. The
        # vector-level path could weight but not correct; this one does both, so it has to hit
        # the same numbers where correction has nothing to do.
        for q in ("casa", "pera verde", "manzana", "jardin azul")
            for distances in (nothing, net.distances)
                old = vectorize(model, q; normalize=false)
                expand_query!(old, voc, net.query_expansion; distances, normalize=true)

                qp = QueryPipeline(expansion=net.query_expansion, distances=distances)
                new = queryvector(model, query_tokens(voc, q, qp))

                @test old.nzind == new.nzind
                @test old.nzval ≈ new.nzval
            end
        end
    end

    @testset "a bag of words is presence only, as BM25 requires" begin
        # bm25score never reads the query side's frequencies, and BOW's counts are Int32, so a
        # weight here would be carried through a whole search and then ignored.
        for q in ("casa", "pera verde")
            old = bagofwords(voc, q)
            expand_query!(old, voc, net.query_expansion)
            new = querybow(voc, query_tokens(voc, q, QueryPipeline(expansion=net.query_expansion)))
            @test Set(keys(old)) == Set(keys(new))
            @test all(==(one(Int32)), values(new))
        end
    end

    @testset "contributions accumulate rather than being deduplicated" begin
        # A neighbour reachable from two query tokens is listed twice on purpose: `queryvector`
        # sums the two contributions, which is what `expand_query!` did by merging duplicate ids.
        # Deduplicating in the pipeline silently dropped the second one and moved scores.
        net2 = Dict("pera" => ["rica"], "verde" => ["rica"])
        q = query_tokens(voc, "pera verde", QueryPipeline(expansion=net2))
        rica = [t for t in q.terms if t.token == "rica"]
        @test length(rica) == 2
        @test Set(t.source for t in rica) == Set(["pera", "verde"])
        # ...and the set view collapses them, so a consumer that only matches tokens is unaffected
        @test length(querytokenset(q)) == length(unique(t.token for t in q.terms))
    end

    @testset "correction happens here, which the vector-level path could not do" begin
        # This is the whole point of the unification: `expand_query!` iterated ids, so a spelling
        # absent from the vocabulary never reached it. Measured on Portuguese Wikipedia, searching
        # `regiao` through the old path returned neighbours of a 33-document misspelling.
        cased = TextConfig(normalization=NormalizationConfig(lc=false, del_diac=false, del_punc=true))
        docs = vcat(fill("la música clásica", 30), ["musica italiana"], fill("el sol brilla", 4))
        cvoc = Vocabulary(cased, docs; verbose=false)
        variants = derive_variants(cvoc)

        qp = QueryPipeline(policy=QueryPolicy(negligible_ratio=2), variants=variants)
        q = query_tokens(cvoc, "musica", qp)
        @test "música" in querytokenset(q)
        @test !("musica" in querytokenset(q))            # corrected, therefore replaced
        @test occursin("misspelling", only(explain(q.resolution)))

        # and with correction off, the literal query -- the escape a consumer owes
        off = query_tokens(cvoc, "musica", QueryPipeline(policy=QueryPolicy(correction=:off),
                                                        variants=variants))
        @test querytokenset(off) == Set(["musica"])
        @test isempty(explain(off.resolution))
    end

    @testset "expansion draws from the corrected spelling, not the typed one" begin
        cased = TextConfig(normalization=NormalizationConfig(lc=false, del_diac=false, del_punc=true))
        docs = vcat(fill("la música clásica", 30), ["musica italiana"], fill("el sol brilla", 4))
        cvoc = Vocabulary(cased, docs; verbose=false)
        variants = derive_variants(cvoc)
        # `musica` (1 document) and `música` (30) have different neighbours; the corrected
        # spelling is the one whose list gets used
        net3 = Dict("musica" => ["italiana"], "música" => ["clásica"])
        q = query_tokens(cvoc, "musica", QueryPipeline(policy=QueryPolicy(negligible_ratio=2),
                                                      variants=variants, expansion=net3))
        @test "clásica" in querytokenset(q)
        @test !("italiana" in querytokenset(q))
    end

    @testset "policy gates each half independently" begin
        qp(; kw...) = QueryPipeline(policy=QueryPolicy(; kw...), expansion=net.query_expansion)
        @test length(querytokenset(query_tokens(voc, "casa", qp(expansion=false)))) == 1
        wide = querytokenset(query_tokens(voc, "casa", qp()))
        capped = querytokenset(query_tokens(voc, "casa", qp(expansion_k=1)))
        @test length(capped) < length(wide)
        # no network at all is the same as asking for none
        @test querytokenset(query_tokens(voc, "casa", QueryPipeline())) == Set(["casa"])
    end

    @testset "policy travels per call, and the maps stay where they were derived" begin
        # The reason this keyword exists: one index has to be able to answer both the corrected
        # reading of a query and the literal one, without a second index and without rederiving
        # the variant map (0.24s over a large vocabulary) on every search.
        cased = TextConfig(normalization=NormalizationConfig(lc=false, del_diac=false))
        # 60 accented against 1 bare: `QueryPolicy`'s default negligible_ratio is 50, so a
        # thinner margin is correctly left alone and would test nothing
        ccorpus = vcat([" la música clásica de la ciudad $i" for i in 1:60],
                       ["una musica rara sin acento"])
        cvoc = Vocabulary(cased, ccorpus; verbose=false)
        variants = derive_variants(cvoc; min_ndocs=1)
        pipe = QueryPipeline(policy=QueryPolicy(), variants=variants,
                             expansion=net.query_expansion)

        # the pipeline's own policy corrects
        @test "música" in querytokenset(query_tokens(cvoc, "musica", pipe))
        # ... and the override searches what was typed, off the very same pipeline
        literal = querytokenset(query_tokens(cvoc, "musica", pipe;
                                             policy=QueryPolicy(correction=:off)))
        @test literal == Set(["musica"])
        # the pipeline is unchanged by having been overridden: it is the caller's argument that
        # varies, not the index's state
        @test "música" in querytokenset(query_tokens(cvoc, "musica", pipe))

        # both halves of the policy respond to the override, and reusing the maps under a
        # policy that ignores them changes nothing
        wide = querytokenset(query_tokens(voc, "casa", pipe))
        @test length(querytokenset(query_tokens(voc, "casa", pipe;
                                                policy=QueryPolicy(expansion=false)))) == 1
        @test length(querytokenset(query_tokens(voc, "casa", pipe;
                                                policy=QueryPolicy(expansion_k=1)))) < length(wide)

        # omitting it is exactly the pipeline's own policy
        off = QueryPipeline(policy=QueryPolicy(correction=:off), variants=variants)
        @test querytokenset(query_tokens(cvoc, "musica", off)) ==
              querytokenset(query_tokens(cvoc, "musica", off; policy=QueryPolicy(correction=:off)))
    end

    @testset "search takes the same override, on both inverted files" begin
        cased = TextConfig(normalization=NormalizationConfig(lc=false, del_diac=false))
        # 60 accented against 1 bare: `QueryPolicy`'s default negligible_ratio is 50, so a
        # thinner margin is correctly left alone and would test nothing
        ccorpus = vcat([" la música clásica de la ciudad $i" for i in 1:60],
                       ["una musica rara sin acento"])
        cvoc = Vocabulary(cased, ccorpus; verbose=false)
        cmodel = VectorModel(IdfWeighting(), TfWeighting(), cvoc)
        qp = QueryPipeline(policy=QueryPolicy(), variants=derive_variants(cvoc; min_ndocs=1))
        ctx = InvertedFileContext()

        for idx in (BM25InvertedFile(cvoc; query=qp),
                    TextInvertedFile(cmodel; dist=Dist.NormCosine(), query=qp))
            append_items!(idx, ctx, ccorpus)
            hits(policy) = begin
                r = knnqueue(KnnSorted, 3)
                policy === nothing ? search(idx, ctx, "musica", r) :
                                     search(idx, ctx, "musica", r; policy)
                Set(Int.(r.ids[r.sp:r.ep]))
            end
            corrected = hits(nothing)                              # the index's own policy
            literal   = hits(QueryPolicy(correction=:off))
            # document 61 is the only one spelling it without the accent
            @test literal == Set([61])
            @test !(61 in corrected)
            # and the index is unchanged: asking again still corrects
            @test hits(nothing) == corrected
        end
    end

    @testset "an already-tokenized query skips tokenization" begin
        @test querytokenset(query_tokens(voc, ["casa", "jardin"])) == Set(["casa", "jardin"])
        @test QueryPipeline().variants === nothing
        @test_throws ArgumentError QueryPipeline(distances=Dict("a" => Float32[1.0]))
    end
end
