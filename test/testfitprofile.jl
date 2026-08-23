using Test, TextSearch, SimilaritySearch

@testset "fit_profile" begin
    corpus = ["la casa roja tiene un jardin grande", "la casa verde esta cerca del rio",
              "el jardin de la casa azul es pequeno", "una manzana roja y una pera verde",
              "la pera verde esta muy rica hoy", "la manzana verde esta rica tambien",
              "una hoja verde cayo en el jardin", "el rio azul cruza el jardin grande"]
    tc = TextConfig()

    @testset "one call produces a base profile with every artifact" begin
        p = fit_profile(tc, corpus; min_ndocs=2, stopwords=(; doc_freq_threshold=0.5),
                        encoder=(; outdim=4), expansion=(; k=3), verbose=false)
        @test isbase(p)
        @test gettrainsize(p.model.voc) == length(corpus)
        @test !isempty(p.stopwords) && p.applied.stopwords
        @test !isempty(p.query_expansion)
        @test !p.applied.lemmas                      # a base profile carries, does not apply
        @test gettextconfig(p).pipeline.lemmas === nothing
        # the threshold is recorded because `merge_profiles` needs it to impute what a batch
        # that removed a stopword never counted
        @test last(p.lineage).params["doc_freq_threshold"] == 0.5
    end

    @testset "stopwords are removed before the vocabulary the encoder trains on" begin
        p = fit_profile(tc, corpus; stopwords=(; doc_freq_threshold=0.5), encoder=(; outdim=4),
                        verbose=false)
        for w in p.stopwords
            @test token2id(p.model.voc, w) == 0      # never entered the counters at all
            @test !haskey(p.query_expansion, w)      # nor the network
        end
    end

    @testset "a reused stopword set makes batches agree exactly" begin
        # This is what makes a merge exact: identical sets across parts means no partial counts,
        # so there is nothing to impute.
        a = fit_profile(tc, corpus[1:4]; stopwords=(; doc_freq_threshold=0.5), encoder=(; outdim=3),
                        verbose=false)
        b = fit_profile(tc, corpus[5:8]; stopwords=(; reuse=a.stopwords), encoder=(; outdim=3),
                        verbose=false)
        @test b.stopwords == a.stopwords
        @test b.applied.stopwords
        m = merge_profiles([a, b])
        @test gettrainsize(m.model.voc) == length(corpus)
    end

    @testset "applying lemmas rebuilds the vocabulary and remaps the network" begin
        plain = fit_profile(tc, corpus; min_ndocs=2, encoder=(; outdim=4), expansion=(; k=3),
                            verbose=false)
        lem = fit_profile(tc, corpus; min_ndocs=2, encoder=(; outdim=4), expansion=(; k=3),
                          lemmas=(; apply=true), verbose=false)
        # meaningful either way: a map is applied exactly when there is one to apply
        @test lem.applied.lemmas == !isempty(lem.lemmas)
        if !isempty(lem.lemmas)
            @test lem.applied.lemmas
            @test gettextconfig(lem).pipeline.lemmas !== nothing
            # the network was rewritten onto lemmas: no entry names a token the vocabulary lost
            for (k, vs) in lem.query_expansion
                @test token2id(lem.model.voc, k) != 0
                for v in vs
                    @test token2id(lem.model.voc, v) != 0
                end
            end
            @test vocsize(lem.model.voc) <= vocsize(plain.model.voc)
        end
    end

    @testset "external word vectors replace the encoder" begin
        voc = Vocabulary(tc, corpus; verbose=false)
        # deterministic stand-in for real embeddings: no LSI runs at all
        W = MatrixDatabase(Float32[i * j for i in 1:4, j in 1:vocsize(voc)])
        p = fit_profile(tc, corpus; encoder=(; wordvectors=W, source_path="vectors.bin"),
                        expansion=(; k=2), verbose=false)
        @test last(p.lineage).params["encoder"] == "external"
        @test last(p.lineage).params["source_path"] == "vectors.bin"
    end

    @testset "pruning everything is an error, not an empty model" begin
        @test_throws ErrorException fit_profile(tc, corpus; min_ndocs=1000, verbose=false)
    end

    @testset "how reproducible a fit is, exactly" begin
        a = fit_profile(tc, corpus; min_ndocs=2, encoder=(; outdim=4), expansion=(; k=3), verbose=false)
        b = fit_profile(tc, corpus; min_ndocs=2, encoder=(; outdim=4), expansion=(; k=3), verbose=false)

        # These parts are stable: they are counted, not factorized.
        @test a.stopwords == b.stopwords
        @test vocsize(a.model.voc) == vocsize(b.model.voc)
        @test gettrainsize(a.model.voc) == gettrainsize(b.model.voc)
        @test Set(keys(a.query_expansion)) == Set(keys(b.query_expansion))

        # The network's contents are NOT, and cannot be made so by sorting. The factorization
        # under it runs threaded, so its reductions sum in a nondeterministic order and the
        # embeddings differ around the seventh significant digit (0.05991161 against
        # 0.05991143 for the same pair). That is enough to flip a near-tie across the top-k
        # boundary: `rica` came out ["esta","verde","pera"] one run and ["esta","verde",
        # "manzana"] the next. So a published profile cannot be verified by checksum, and two
        # fits of one corpus are equivalent rather than equal.
        for (k, da) in a.query_expansion_distances
            db = get(b.query_expansion_distances, k, nothing)
            db === nothing && continue
            length(da) == length(db) || continue     # a near-tie crossed the top-k boundary
            @test da ≈ db atol=1f-4
        end
    end

    @testset "exact ties are ordered deterministically" begin
        # What sorting *can* fix, and what it was not doing: among neighbours at exactly equal
        # distance, `allknn` returns no defined order, and it varied run to run. Tested on
        # synthetic vectors so no factorization noise is involved -- these ties are exact.
        docs = ["alfa beta gama", "alfa beta gama", "delta alfa beta gama"]
        voc = Vocabulary(TextConfig(), docs; verbose=false)
        n = vocsize(voc)
        # every token at the same distance from every other: all ties, nothing but ties
        W = MatrixDatabase(zeros(Float32, 3, n))
        for j in 1:n; W.matrix[(j - 1) % 3 + 1, j] = 1f0; end
        first = query_expansion(voc, W, min(3, n - 1); verbose=false).query_expansion
        for _ in 1:4
            @test query_expansion(voc, W, min(3, n - 1); verbose=false).query_expansion == first
        end
        # what is pinned is that repeated fits agree, not any particular order: the tie-break
        # is (distance, token), and asserting the alphabetical half of that here would only
        # restate the implementation
    end

    @testset "config equality is by value, which it was not" begin
        # Every field of these is compared, and several are heap objects (`nlist`, the emoji
        # table, the compiled regexes), so Julia's default field-wise `===` said two configs
        # built from the same settings were different. `merge_profiles` had its own private
        # comparison because of it.
        @test TextConfig() == TextConfig()
        @test NormalizationConfig() == NormalizationConfig()
        @test TokenizationConfig(nlist=[1]) == TokenizationConfig(nlist=[1])
        @test TokenizationConfig(nlist=[1]) != TokenizationConfig(nlist=[1, 2])
        @test TextConfig(normalization=NormalizationConfig(lc=false)) != TextConfig()
    end
end
