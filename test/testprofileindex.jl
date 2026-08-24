using Test, TextSearch, SimilaritySearch

# The path this file covers had no coverage at all: the four test files that touch `load_profile`
# never built an index, and the ones that build indexes never load a profile. So "does a
# pre-trained profile actually work when you search with it?" was answerable only by trying it by
# hand -- and when I did, on the real Portuguese Wikipedia profile, it did not: the index expanded
# a query without correcting it, so `regiao` (33 documents, a misspelling) returned the neighbours
# of that misspelling instead of anything about regions.
@testset "a pre-trained profile drives an index" begin
    quietctx() = InvertedFileContext(reporters=[])

    # A corpus that keeps case and diacritics, with one rare misspelling of a common word --
    # the shape that matters, since `min_ndocs` leaves such spellings in the vocabulary.
    corpus = vcat(
        ["la música clásica del siglo XIX", "un festival de música popular",
         "la música es un arte antiguo", "la música andina del altiplano"],
        fill("una canción de música tradicional", 20),
        ["musica italiana del renacimiento"],                 # the only unaccented spelling
        ["el sol brilla sobre el mar", "el sol de la mañana en la playa"],
    )
    cfg = TextConfig(normalization=NormalizationConfig(lc=false, del_diac=false, del_punc=true))
    voc = Vocabulary(cfg, corpus; verbose=false)
    model = VectorModel(IdfWeighting(), TfWeighting(), voc)
    profile = TextProfile(model;
                          query_expansion=Dict("música" => ["clásica", "canción"],
                                               "musica" => ["italiana", "renacimiento"]),
                          lineage=[LineageStep(:fit; trainsize=length(corpus))])

    dir = mktempdir()
    try
        save_profile(dir, profile)
        p = load_profile(dir)

        # Index only part of the corpus: the point of a profile is that the statistics do NOT
        # come from what happens to be indexed.
        indexed = corpus[1:4]

        @testset "the corpus's statistics, the index's document lengths" begin
            idx = BM25InvertedFile(p)
            ctx = quietctx()
            append_items!(idx, ctx, indexed)

            @test length(idx) == length(indexed)
            # the scorer speaks for the whole corpus, not for the four documents indexed
            @test idx.bm25.trainsize == gettrainsize(p.model.voc) == length(corpus)
            @test idx.bm25.trainsize != length(indexed)
            # and its idf reads the profile's document frequencies
            @test getndocs(idx.voc, token2id(idx.voc, "música")) == 24
            @test length(idx.doclens) == length(indexed)
        end

        @testset "correction reaches the index, which it never could before" begin
            # `musica` is in the vocabulary (one document), so this is exactly the case where
            # "it exists, therefore they meant it" fails.
            idx = BM25InvertedFile(p; policy=QueryPolicy(negligible_ratio=2))
            ctx = quietctx()
            append_items!(idx, ctx, indexed)

            res = knnqueue(KnnSorted, 4)
            search(idx, ctx, "musica", res)
            hits = collect(IdView(res))
            @test !isempty(hits)                       # none of the indexed documents holds `musica`
            @test all(occursin("música", indexed[id]) for id in hits)
        end

        @testset "correction off is the literal query, and it finds nothing here" begin
            idx = BM25InvertedFile(p; policy=QueryPolicy(correction=:off))
            ctx = quietctx()
            append_items!(idx, ctx, indexed)

            res = knnqueue(KnnSorted, 4)
            search(idx, ctx, "musica", res)
            @test isempty(collect(IdView(res)))        # `musica` occurs in no indexed document
        end

        @testset "expansion follows the profile's own marker" begin
            # carried but not applied, so a plain index does not expand...
            @test !p.applied.query_expansion
            @test BM25InvertedFile(p).query.expansion === nothing
            # ...and asking takes it anyway, which is what a base profile needs
            @test BM25InvertedFile(p; expansion=true).query.expansion == p.query_expansion

            applied = with_applied(p; query_expansion=true)
            @test BM25InvertedFile(applied).query.expansion == p.query_expansion
        end

        @testset "the vector-space index takes a profile the same way" begin
            idx = TextInvertedFile(p; policy=QueryPolicy(negligible_ratio=2))
            ctx = quietctx()
            append_items!(idx, ctx, indexed)
            @test length(idx) == length(indexed)
            @test idx.model === p.model              # the corpus's weights, not the subset's

            res = knnqueue(KnnSorted, 4)
            search(idx, ctx, "musica", res)
            hits = collect(IdView(res))
            @test !isempty(hits)
            @test all(occursin("música", indexed[id]) for id in hits)
        end
    finally
        rm(dir; recursive=true, force=true)
    end
end
