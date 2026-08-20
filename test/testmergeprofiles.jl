using Test, TextSearch, SimilaritySearch

@testset "merge_profiles" begin
    docs = [
        "la casa roja tiene jardin",
        "la casa verde tiene jardin",
        "la casa azul es pequena",
        "una manzana roja y una pera",
        "la pera verde esta rica",
        "la manzana verde esta rica",
    ]
    tc = TextConfig(tokenization=TokenizationConfig(nlist=[1]))

    "save a profile built from `d` (plus optional extras) and load it straight back"
    function roundtrip(d; textconfig=tc, gw=IdfWeighting(), lw=TfWeighting(), kwargs...)
        voc = Vocabulary(textconfig, d; verbose=false)
        model = VectorModel(gw, lw, voc)
        dir = tempname()
        save_profile(dir, model; kwargs...)
        p = load_profile(dir)
        rm(dir; recursive=true, force=true)
        p
    end

    @testset "vocabulary counts and weights merge EXACTLY" begin
        a = roundtrip(docs[1:3])
        b = roundtrip(docs[4:6])
        merged = merge_profiles([a, b])

        whole = Vocabulary(tc, docs; verbose=false)
        wholemodel = VectorModel(IdfWeighting(), TfWeighting(), whole)
        mvoc = merged.model.voc

        @test trainsize(mvoc) == trainsize(whole) == 6
        @test numtokens(mvoc) == numtokens(whole)
        @test vocsize(mvoc) == vocsize(whole)

        # token order need not match (merged order follows insertion), so compare by token
        for id in eachindex(whole)
            t = token(whole, id)
            mid = token2id(mvoc, t)
            @test mid != 0
            @test occs(mvoc, mid) == occs(whole, id)
            @test ndocs(mvoc, mid) == ndocs(whole, id)
        end

        # weights are RECOMPUTED from the merged counters, so they equal a single global fit
        for id in eachindex(whole)
            mid = token2id(mvoc, token(whole, id))
            @test merged.model.weight[mid] ≈ wholemodel.weight[id]
        end
        @test merged.model.maxoccs == wholemodel.maxoccs
    end

    @testset "encoder records the merge" begin
        merged = merge_profiles([roundtrip(docs[1:3]; encoder=(; kind=:lsi, outdim=4)),
                                 roundtrip(docs[4:6]; encoder=(; kind=:lsi, outdim=4))])
        @test merged.encoder.kind === :merged
        @test merged.encoder.n_sources == 2
        @test merged.encoder.source_kinds == "lsi"
    end

    @testset "synonyms fuse by rank consensus" begin
        # every token here is in the corpus vocabulary, so nothing is dropped as OOV.
        # "jardin" is ranked #1 by BOTH inputs; "roja" and "pera" are ranked #2 by one each.
        # Consensus must put jardin first even though one input reported a closer raw
        # distance for another candidate -- that is the whole point of fusing ranks.
        a = roundtrip(docs; synonyms=Dict("casa" => ["jardin" => 0.30f0, "roja" => 0.40f0]))
        b = roundtrip(docs; synonyms=Dict("casa" => ["jardin" => 0.10f0, "pera" => 0.20f0]))
        merged = merge_profiles([a, b])

        got = merged.synonyms["casa"]
        @test first(got).first == "jardin"
        # the kept Float32 is the mean of the distances the contributors reported
        @test first(got).second ≈ 0.20f0
        # synonyms_k=0 keeps as many neighbors as the richest input had (2 here), so the
        # fused pool of 3 candidates is truncated -- the merged list does not grow with the
        # number of inputs. Single-support ties break by mean distance, so pera (0.20)
        # beats roja (0.40) for the remaining slot.
        @test [p.first for p in got] == ["jardin", "pera"]

        @testset "synonyms_k overrides the default cap" begin
            m1 = merge_profiles([a, b]; synonyms_k=1)
            @test [p.first for p in m1.synonyms["casa"]] == ["jardin"]

            m3 = merge_profiles([a, b]; synonyms_k=3)
            @test [p.first for p in m3.synonyms["casa"]] == ["jardin", "pera", "roja"]
        end
    end

    @testset "OOV synonyms/lemmas are dropped, not carried over" begin
        a = roundtrip(docs; synonyms=Dict("casa" => ["noexisteenvocab" => 0.1f0]),
                            lemmas=Dict("casa" => "tampocoexiste"))
        merged = merge_profiles([a, roundtrip(docs)])
        @test !haskey(merged.synonyms, "casa") || isempty(merged.synonyms["casa"])
        @test !haskey(merged.lemmas, "casa")
    end

    @testset "lemmas merge by plurality vote" begin
        a = roundtrip(docs; lemmas=Dict("casas" => "casa"))
        b = roundtrip(docs; lemmas=Dict("casas" => "casa"))
        c = roundtrip(docs; lemmas=Dict("casas" => "jardin"))
        # "casas" isn't in the corpus vocabulary, so use tokens that are
        a2 = roundtrip(docs; lemmas=Dict("roja" => "casa"))
        b2 = roundtrip(docs; lemmas=Dict("roja" => "casa"))
        c2 = roundtrip(docs; lemmas=Dict("roja" => "jardin"))
        merged = merge_profiles([a2, b2, c2])
        @test merged.lemmas["roja"] == "casa"   # 2 votes vs 1
    end

    @testset "conflicting lemma votes cannot produce a cycle" begin
        # one input says roja => casa, the other says casa => roja: a naive merge would keep
        # both edges and make lemma lookup non-terminating
        a = roundtrip(docs; lemmas=Dict("roja" => "casa"))
        b = roundtrip(docs; lemmas=Dict("casa" => "roja"))
        merged = merge_profiles([a, b])

        # exactly one direction survives, and following it terminates immediately
        @test !(haskey(merged.lemmas, "roja") && haskey(merged.lemmas, "casa"))
        for (tok, lemma) in merged.lemmas
            @test !haskey(merged.lemmas, lemma)   # the target is itself canonical
            @test tok != lemma
        end
        # "casa" occurs more often than "roja" in this corpus, so it is the canonical one
        @test merged.lemmas["roja"] == "casa"
    end

    @testset "stopword candidates: recomputed globally, unioned with the inputs'" begin
        a = roundtrip(docs[1:3]; stopword_candidates=["previamente_detectada"])
        b = roundtrip(docs[4:6])
        merged = merge_profiles([a, b]; doc_freq_threshold=0.5)
        # kept even though it cannot be re-derived (it is absent from the merged vocabulary)
        @test "previamente_detectada" in merged.stopword_candidates
        # and "la", in 5 of 6 documents, is re-derived from the merged counters
        @test "la" in merged.stopword_candidates
        @test issorted(merged.stopword_candidates)
    end

    @testset "differing stopword transformations union" begin
        tc_a = TextConfig(tc; transformation=IgnoreStopwords(Set(["la"])))
        tc_b = TextConfig(tc; transformation=IgnoreStopwords(Set(["una"])))
        merged = merge_profiles([roundtrip(docs[1:3]; textconfig=tc_a),
                                 roundtrip(docs[4:6]; textconfig=tc_b)])
        tt = merged.model.voc.textconfig.transformation
        @test tt isa IgnoreStopwords
        @test tt.stopwords == Set(["la", "una"])
    end

    @testset "incompatible inputs are rejected" begin
        @test_throws ArgumentError merge_profiles([])

        # different tokenization
        other_tok = TextConfig(tokenization=TokenizationConfig(nlist=[1, 2]))
        @test_throws ErrorException merge_profiles([roundtrip(docs[1:3]),
                                                    roundtrip(docs[4:6]; textconfig=other_tok)])

        # different normalization
        other_norm = TextConfig(tc; normalization=NormalizationConfig(lc=false))
        @test_throws ErrorException merge_profiles([roundtrip(docs[1:3]),
                                                    roundtrip(docs[4:6]; textconfig=other_norm)])

        # different weighting scheme
        @test_throws ErrorException merge_profiles([roundtrip(docs[1:3]),
                                                    roundtrip(docs[4:6]; lw=BinaryLocalWeighting())])
    end

    @testset "structurally identical configs built separately are accepted" begin
        # regression guard: `==` on TokenizationConfig is false for equal-but-distinct
        # configs (its nlist is a fresh Vector), so merge must compare fields by meaning
        tc2 = TextConfig(tokenization=TokenizationConfig(nlist=[1]))
        @test tc.tokenization != tc2.tokenization          # documents the trap
        merged = merge_profiles([roundtrip(docs[1:3]), roundtrip(docs[4:6]; textconfig=tc2)])
        @test trainsize(merged.model.voc) == 6
    end
end
