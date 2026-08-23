using Test, TextSearch, SimilaritySearch

@testset "stopword_candidates" begin
    # "la" appears in every document (5/5 = 1.0 doc-freq ratio); "roja"/"verde" in 2/5;
    # everything else in 1/5.
    corpus = [
        "la casa roja",
        "la casa verde",
        "la manzana roja",
        "la pera verde",
        "la hoja",
    ]
    textconfig = TextConfig(tokenization=TokenizationConfig(nlist=[1]))
    voc = Vocabulary(textconfig, corpus)
    model = VectorModel(IdfWeighting(), TfWeighting(), voc)

    @testset "default threshold (0.5): only tokens above the ratio are flagged" begin
        candidates = stopword_candidates(voc)
        @test "la" in candidates
        @test !("roja" in candidates)   # 2/5 = 0.4, at/under threshold
        @test !("casa" in candidates)
    end

    @testset "sorted by decreasing document-frequency ratio" begin
        candidates = stopword_candidates(voc, 0.1)
        ratios = [getndocs(voc, token2id(voc, t)) / gettrainsize(voc) for t in candidates]
        @test issorted(ratios; rev=true)
    end

    @testset "VectorModel overload matches the Vocabulary one" begin
        @test stopword_candidates(model) == stopword_candidates(voc)
    end

    @testset "threshold out of (0,1] errors clearly" begin
        @test_throws ArgumentError stopword_candidates(voc, 0.0)
        @test_throws ArgumentError stopword_candidates(voc, 1.5)
    end

    @testset "under lc=false, a flagged spelling takes its other casings with it" begin
        # `la` is in 4 of 6 documents (0.667, flagged); `La` in 2 (0.333, below any threshold
        # that flags `la`). Detection is per spelling, so without the casing extension `La`
        # would survive as an ordinary content token -- the leak measured on Spanish
        # Wikipedia paragraphs, where `Las` (df 0.081) stayed while `las` was removed.
        corpus2 = ["la casa roja", "La casa verde", "la manzana roja",
                   "la pera Verde", "La hoja", "la nube"]
        tc = TextConfig(normalization=NormalizationConfig(lc=false),
                        tokenization=TokenizationConfig(nlist=[1]))
        voc2 = Vocabulary(tc, corpus2)
        candidates = stopword_candidates(voc2, 0.5)
        @test "la" in candidates
        @test "La" in candidates
        # and the extension does not lower the threshold for a group nothing flagged:
        # `Verde` and `verde` are one document each and neither is a candidate
        @test !("Verde" in candidates)
        @test !("verde" in candidates)
        @test !("casa" in candidates)
        ratios = [getndocs(voc2, token2id(voc2, t)) / gettrainsize(voc2) for t in candidates]
        @test issorted(ratios; rev=true)     # siblings carry their own ratio, not the flagged one

        # the same corpus under lc=true has one spelling per word, so the pass is a no-op
        tclc = TextConfig(normalization=NormalizationConfig(lc=true),
                          tokenization=TokenizationConfig(nlist=[1]))
        lowered = stopword_candidates(Vocabulary(tclc, corpus2), 0.5)
        @test lowered == ["la"]
    end

    @testset "empty vocabulary returns an empty list" begin
        empty_voc = Vocabulary(textconfig, String[])
        @test stopword_candidates(empty_voc) == String[]
    end
end
