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
        save_profile(dir, TextProfile(model; kwargs...))
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

        @test gettrainsize(mvoc) == gettrainsize(whole) == 6
        @test getnumtokens(mvoc) == getnumtokens(whole)
        @test vocsize(mvoc) == vocsize(whole)

        # token order need not match (merged order follows insertion), so compare by token
        for id in eachindex(whole)
            t = gettoken(whole, id)
            mid = token2id(mvoc, t)
            @test mid != 0
            @test getoccs(mvoc, mid) == getoccs(whole, id)
            @test getndocs(mvoc, mid) == getndocs(whole, id)
        end

        # weights are RECOMPUTED from the merged counters, so they equal a single global fit
        for id in eachindex(whole)
            mid = token2id(mvoc, gettoken(whole, id))
            @test merged.model.weight[mid] ≈ wholemodel.weight[id]
        end
        @test merged.model.maxoccs == wholemodel.maxoccs
    end

    @testset "lineage records the merge" begin
        fit = [LineageStep(:fit; outdim=4)]
        merged = merge_profiles([roundtrip(docs[1:3]; lineage=fit),
                                 roundtrip(docs[4:6]; lineage=fit)])
        @test last(merged.lineage).stage === :merge
        @test last(merged.lineage).params["n_sources"] == 2
        # merging batches of one corpus does not make a tuned model
        @test isbase(merged)
    end

    @testset "query_expansion fuse by rank consensus" begin
        # every token here is in the corpus vocabulary, so nothing is dropped as OOV.
        # "jardin" is ranked #1 by BOTH inputs; "roja" and "pera" are ranked #2 by one each.
        # Consensus must put jardin first even though one input reported a closer raw
        # distance for another candidate -- that is the whole point of fusing ranks.
        a = roundtrip(docs; query_expansion=Dict("casa" => ["jardin", "roja"]), query_expansion_distances=Dict("casa" => Float32[0.30, 0.40]))
        b = roundtrip(docs; query_expansion=Dict("casa" => ["jardin", "pera"]), query_expansion_distances=Dict("casa" => Float32[0.10, 0.20]))
        merged = merge_profiles([a, b])

        got = merged.query_expansion["casa"]
        @test first(got) == "jardin"
        # the distances that come back are the mean of what the contributors reported, in a
        # parallel list rather than interleaved with the words
        @test first(merged.query_expansion_distances["casa"]) ≈ 0.20f0
        # query_expansion_k=0 keeps as many neighbors as the richest input had (2 here), so the
        # fused pool of 3 candidates is truncated -- the merged list does not grow with the
        # number of inputs. Single-support ties break by mean distance, so pera (0.20)
        # beats roja (0.40) for the remaining slot.
        @test got == ["jardin", "pera"]

        @testset "query_expansion_k overrides the default cap" begin
            m1 = merge_profiles([a, b]; query_expansion_k=1)
            @test m1.query_expansion["casa"] == ["jardin"]

            m3 = merge_profiles([a, b]; query_expansion_k=3)
            @test m3.query_expansion["casa"] == ["jardin", "pera", "roja"]
        end
    end

    @testset "OOV query_expansion/lemmas are dropped, not carried over" begin
        a = roundtrip(docs; query_expansion=Dict("casa" => ["noexisteenvocab"]),
                            lemmas=Dict("casa" => "tampocoexiste"))
        merged = merge_profiles([a, roundtrip(docs)])
        @test !haskey(merged.query_expansion, "casa") || isempty(merged.query_expansion["casa"])
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

    @testset "stopwords: recomputed globally, unioned with the inputs'" begin
        a = roundtrip(docs[1:3]; stopwords=Set(["previamente_detectada"]))
        b = roundtrip(docs[4:6])
        merged = merge_profiles([a, b]; doc_freq_threshold=0.5)
        # kept even though it cannot be re-derived (it is absent from the merged vocabulary)
        @test "previamente_detectada" in merged.stopwords
        # and "la", in 5 of 6 documents, is re-derived from the merged counters
        @test "la" in merged.stopwords
    end

    @testset "differing artifacts combine instead of being rejected" begin
        # This used to be the sharpest edge of the old design: artifacts lived inside the
        # TextConfig, so merging compared them for EQUALITY and needed a special case to
        # union differing stopword sets -- and it rejected two profiles whose lemma maps were
        # merely different, even though voting on them is exactly what a merge should do.
        # Now policy is compared and artifacts combine, so nothing about them has to match.
        applied_lem = AppliedArtifacts(lemmas=true)
        a = roundtrip(docs[1:3]; stopwords=Set(["la"]), lemmas=Dict("roja" => "casa"),
                                 applied=applied_lem)
        b = roundtrip(docs[4:6]; stopwords=Set(["una"]), lemmas=Dict("roja" => "casa"),
                                 applied=applied_lem)
        merged = merge_profiles([a, b])

        @test gettrainsize(merged.model.voc) == 6
        # Stopword sets no longer simply union: each one is re-judged on the merged counters
        # (with whatever a batch destroyed imputed back -- see `_impute_removed_stopwords`,
        # which does not apply here since neither input APPLIED its set, so both vocabularies
        # hold exact counts). "la" is in 5 of the 6 documents and is a corpus stopword; "una"
        # is in 1 and is not, however ubiquitous b found it. Union would have taken both.
        @test "la" in merged.stopwords
        @test !("una" in merged.stopwords)
        @test token2id(merged.model.voc, "la") != 0 && token2id(merged.model.voc, "una") != 0
        @test merged.lemmas["roja"] == "casa"                # maps vote
        @test merged.applied.lemmas                          # applied if any input applied
        # and the merged profile applies exactly the map it carries
        @test gettextconfig(merged).pipeline.lemmas !== nothing
    end

    @testset "a disagreeing lemma map votes rather than erroring" begin
        applied_lem = AppliedArtifacts(lemmas=true)
        a = roundtrip(docs; lemmas=Dict("roja" => "casa"), applied=applied_lem)
        b = roundtrip(docs; lemmas=Dict("roja" => "casa"), applied=applied_lem)
        c = roundtrip(docs; lemmas=Dict("roja" => "jardin"), applied=applied_lem)
        merged = merge_profiles([a, b, c])
        @test merged.lemmas["roja"] == "casa"    # 2 votes vs 1, no error
    end

    @testset "profiles of different languages cannot be merged" begin
        # Nothing else can tell them apart: two Wikipedia profiles in Spanish and Portuguese
        # have identical normalization and tokenization, so before the language was recorded
        # this merge succeeded silently and produced a model of neither language. Measured on
        # the real 10k probes: 94,330 + 67,868 -> 129,940 tokens, no complaint.
        es = roundtrip(docs[1:3]; textconfig=TextConfig(tc; language=:es))
        pt = roundtrip(docs[4:6]; textconfig=TextConfig(tc; language=:pt))
        @test_throws ErrorException merge_profiles([es, pt])

        # the same language merges as before
        es2 = roundtrip(docs[4:6]; textconfig=TextConfig(tc; language=:es))
        merged = merge_profiles([es, es2])
        @test gettrainsize(merged.model.voc) == 6
        @test getpolicy(merged).language === :es

        # `:unknown` cannot contradict anything, so it stays permissive -- a hand-built
        # config that never declared a language must not become unmergeable
        plain = roundtrip(docs[4:6])
        @test getpolicy(plain).language === :unknown
        @test gettrainsize(merge_profiles([es, plain]).model.voc) == 6
    end

    @testset "language survives the round-trip" begin
        p = roundtrip(docs; textconfig=TextConfig(tc; language=:pt))
        @test getpolicy(p).language === :pt
        @test gettextconfig(p).language === :pt
    end

    @testset "incompatible POLICY is rejected" begin
        # policy is the only thing that has to match, and it has to match exactly
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
        @test gettrainsize(merged.model.voc) == 6
    end

    @testset "a stopword only SOME inputs removed gets its counts imputed" begin
        # `fit` applies stopwords by putting them in the pipeline, so a flagged token never
        # enters that batch's vocabulary and the merged counters hold only the batches that did
        # not flag it. Merging used to keep such a token with those partial counts, which made
        # its idf far too high while the profile's own stopword list called it a stopword.
        # Measured on Portuguese Wikipedia: 18 of 35, with `como` at df=0.049 against a true
        # corpus df above 0.5.
        # built the way `fit` builds it: the stopword is in the pipeline, so "la" is genuinely
        # absent from a's vocabulary rather than merely listed
        a = roundtrip(docs[1:3]; textconfig=TextConfig(tc; pipeline=TokenPipeline(stopwords=Set(["la"]))),
                      stopwords=Set(["la"]), applied=AppliedArtifacts(stopwords=true))
        @test !("la" in gettoken.(Ref(a.model.voc), eachindex(a.model.voc)))
        # "la" is NOT a stopword for b, so b's vocabulary contains it and the merged counters
        # carry b's count alone -- a fraction of the truth
        b = roundtrip(docs[4:6])
        @test "la" in gettoken.(Ref(b.model.voc), eachindex(b.model.voc))

        # a's batch recorded no count for "la", but it did record that its document frequency
        # there exceeded the threshold -- so threshold*trainsize is a real lower bound, and
        # the merge imputes it instead of trusting b's partial count or deleting the token
        merged = merge_profiles([a, b]; doc_freq_threshold=0.9)
        mvoc = merged.model.voc
        bnd = token2id(mvoc, "la")
        @test bnd != 0                        # not deleted
        @test !("la" in merged.stopwords)     # 3/6 documents is under 0.9
        # b's exact count plus the bound imputed for a
        @test getndocs(mvoc, bnd) ==
              getndocs(b.model.voc, token2id(b.model.voc, "la")) +
              round(Int, 0.9 * gettrainsize(a.model.voc))
        # the imputed occurrences enter numtokens too, or avgdoclen stays short by that much
        @test getnumtokens(mvoc) > getnumtokens(a.model.voc) + getnumtokens(b.model.voc)

        # and when the imputed frequency DOES cross the threshold it is listed as a stopword.
        # The vocabulary still records it -- with the imputed count, not a fraction of one --
        # because the vocabulary is the record of the corpus and the artifact is the policy
        # over it; `gettextconfig` is what actually filters the token out.
        hi = merge_profiles([a, b]; doc_freq_threshold=0.3)
        @test "la" in hi.stopwords
        @test "la" in gettextconfig(hi).pipeline.stopwords
        @test getndocs(hi.model.voc, token2id(hi.model.voc, "la")) ==
              getndocs(b.model.voc, token2id(b.model.voc, "la")) +
              round(Int, 0.3 * gettrainsize(a.model.voc))
    end

    @testset "query_expansion fusion counts agreeing inputs, and is not normalized" begin
        # Pins the rule against a plausible-looking change that was measured and rejected: see
        # `_fuse_query_expansion`. Normalizing by the inputs that could have voted rewrote 64.9% of a
        # Portuguese Wikipedia network and cost `cidade` the neighbour `cidades`, because being
        # in a top-8 out of 150k tokens is selective enough that two agreeing embeddings beat one.
        #
        # `ubicuo` is in all 8 inputs and reaches rank 3 in 7 of them; `raro` is in one input and
        # is rank 1 there. Summed, `ubicuo` wins on consensus. Per opportunity, `raro` would.
        function synp(d, syn)
            voc = Vocabulary(tc, d; verbose=false)
            TextProfile(VectorModel(IdfWeighting(), TfWeighting(), voc); query_expansion=syn)
        end
        withraro = ["casa raro z ubicuo", "casa raro z ubicuo"]
        without  = ["casa z ubicuo", "casa z ubicuo"]
        inputs = [synp(withraro, Dict("casa" => ["raro", "z", "ubicuo"]))]
        append!(inputs, [synp(without, Dict("casa" => ["z", "ubicuo"])) for _ in 1:7])

        merged = merge_profiles(inputs)
        cands = merged.query_expansion["casa"]
        @test findfirst(==("ubicuo"), cands) < findfirst(==("raro"), cands)
        @test token2id(inputs[2].model.voc, "raro") == 0   # the absence is real
    end

    @testset "a token EVERY input removed stays a stopword" begin
        # nothing left to impute from, so it cannot be re-derived -- but it is still a stopword
        tcsw = TextConfig(tc; pipeline=TokenPipeline(stopwords=Set(["la"])))
        a = roundtrip(docs[1:3]; textconfig=tcsw, stopwords=Set(["la"]),
                      applied=AppliedArtifacts(stopwords=true))
        b = roundtrip(docs[4:6]; textconfig=tcsw, stopwords=Set(["la"]),
                      applied=AppliedArtifacts(stopwords=true))
        merged = merge_profiles([a, b])
        @test "la" in merged.stopwords
        @test token2id(merged.model.voc, "la") == 0
    end

    @testset "the merge keeps the inputs' lineage" begin
        # istuned reads nothing but the lineage, so dropping the inputs' steps made a merge of
        # refitted profiles report itself as a base model
        a = roundtrip(docs[1:3]; lineage=[LineageStep(:fit; trainsize=3)])
        b = roundtrip(docs[4:6]; lineage=[LineageStep(:fit; trainsize=3),
                                         LineageStep(:refit; kappa=2.0)])
        merged = merge_profiles([a, b])
        stages = [s.stage for s in merged.lineage]
        @test stages == [:fit, :refit, :merge]
        @test istuned(merged)
        @test !isbase(merged)
        # each distinct stage carries how many inputs contributed it
        @test merged.lineage[1].params["n_sources"] == 2
        @test merged.lineage[2].params["n_sources"] == 1

        # a merge of plain fits stays a base
        plain = merge_profiles([a, roundtrip(docs[4:6]; lineage=[LineageStep(:fit; trainsize=3)])])
        @test [s.stage for s in plain.lineage] == [:fit, :merge]
        @test isbase(plain)
    end
end
