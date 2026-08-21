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

    @testset "empty vocabulary returns an empty list" begin
        empty_voc = Vocabulary(textconfig, String[])
        @test stopword_candidates(empty_voc) == String[]
    end
end
