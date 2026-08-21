using Test, TextSearch, SimilaritySearch

@testset "lemma_clusters" begin
    corpus = ["cat", "cats", "dog", "dogs", "fish"]
    textconfig = TextConfig(tokenization=TokenizationConfig(nlist=[1]))
    voc = Vocabulary(textconfig, corpus)
    m = vocsize(voc)
    @test m == 5

    # Hand-placed 2D embeddings: {cat,cats} and {dog,dogs} are tight pairs (~0.01 apart),
    # mutually ~10 apart from each other and from fish -- a well-separated 3-cluster
    # scenario whose GROUPING outcome is invariant to fft's random starting point (see
    # reasoning in the PR/commit that added this test).
    coords = Dict(
        "cat" => Float32[0.0, 0.0], "cats" => Float32[0.01, 0.0],
        "dog" => Float32[10.0, 0.0], "dogs" => Float32[10.01, 0.0],
        "fish" => Float32[0.0, 10.0],
    )
    X = Matrix{Float32}(undef, 2, m)
    for tid in 1:m
        X[:, tid] = coords[token(voc, tid)]
    end
    wordvecs = MatrixDatabase(X)

    @testset "shortest selector groups tight pairs, singleton left alone" begin
        lemmas = lemma_clusters(voc, wordvecs; algorithm=:fft, num_clusters=3, selector=:shortest, dist=Dist.L2(), morphology=:none)
        @test lemmas["cats"] == "cat"
        @test lemmas["dogs"] == "dog"
        @test !haskey(lemmas, "cat")
        @test !haskey(lemmas, "dog")
        @test !haskey(lemmas, "fish")
    end

    @testset "num_clusters=0 defaults to sqrt(vocsize) heuristic and still runs" begin
        lemmas = lemma_clusters(voc, wordvecs; dist=Dist.L2(), morphology=:none)
        @test lemmas isa Dict{String,String}
    end

    @testset "most_frequent selector picks the higher-occs token" begin
        corpus2 = ["cat", "cats", "cats", "cats"]
        voc2 = Vocabulary(textconfig, corpus2)
        X2 = Matrix{Float32}(undef, 2, vocsize(voc2))
        for tid in 1:vocsize(voc2)
            X2[:, tid] = coords[token(voc2, tid)]
        end
        lemmas = lemma_clusters(voc2, MatrixDatabase(X2); num_clusters=1, selector=:most_frequent, dist=Dist.L2(), morphology=:none)
        @test lemmas["cat"] == "cats"  # occs("cats")=3 > occs("cat")=1
    end

    @testset "shortest_then_most_frequent breaks length ties by occs" begin
        corpus3 = ["ax", "ax", "ax", "by"]
        voc3 = Vocabulary(textconfig, corpus3)
        coords3 = Dict("ax" => Float32[0.0, 0.0], "by" => Float32[0.01, 0.0])
        X3 = Matrix{Float32}(undef, 2, vocsize(voc3))
        for tid in 1:vocsize(voc3)
            X3[:, tid] = coords3[token(voc3, tid)]
        end
        lemmas = lemma_clusters(voc3, MatrixDatabase(X3); num_clusters=1, selector=:shortest_then_most_frequent, dist=Dist.L2(), morphology=:none)
        @test lemmas["by"] == "ax"  # same length, occs("ax")=3 > occs("by")=1
    end

    @testset "order=:morphology_first groups families the semantic partition splits apart" begin
        # two inflection families whose embeddings are deliberately far apart: a
        # semantic-first pass partitions them by embedding and can never reunite them,
        # while morphology-first groups by surface form and only then consults embeddings.
        words = ["cantar", "cantara", "cantaba", "volar", "volara"]
        voc6 = Vocabulary(textconfig, words)
        X6 = Matrix{Float32}(undef, 2, vocsize(voc6))
        for tid in 1:vocsize(voc6)
            t = token(voc6, tid)
            # spread members of the same family far apart in embedding space
            X6[:, tid] = startswith(t, "cant") ? Float32[10.0 * length(t), 0.0] :
                                                  Float32[0.0, 10.0 * length(t)]
        end
        wv6 = MatrixDatabase(X6)

        L = lemma_clusters(voc6, wv6; order=:morphology_first, semantic_threshold=99.0,
                            morphology=:jaccard, morphology_threshold=0.5, min_common_prefix=3,
                            selector=:shortest, dist=Dist.L2())
        @test L["cantara"] == "cantar"
        @test L["cantaba"] == "cantar"
        @test L["volara"] == "volar"
        # the two families must stay separate: no "cant*" maps to a "vol*" lemma
        @test !any(startswith(l, "vol") for (t, l) in L if startswith(t, "cant"))

        @testset "a tight semantic threshold splits a morphological family" begin
            # embeddings are far apart by construction, so requiring closeness must prevent
            # the merges above -- this is the knob that keeps homographs apart
            L2 = lemma_clusters(voc6, wv6; order=:morphology_first, semantic_threshold=0.001,
                                 morphology=:jaccard, morphology_threshold=0.5,
                                 min_common_prefix=3, dist=Dist.L2())
            @test length(L2) < length(L)
        end

        @test_throws ErrorException lemma_clusters(voc6, wv6; order=:bogus)
        @test_throws ErrorException lemma_clusters(voc6, wv6; order=:morphology_first, morphology=:none)
    end

    @testset "unknown algorithm/selector/morphology error clearly" begin
        @test_throws ErrorException lemma_clusters(voc, wordvecs; algorithm=:bogus)
        @test_throws ErrorException lemma_clusters(voc, wordvecs; selector=:bogus)
        @test_throws ErrorException lemma_clusters(voc, wordvecs; morphology=:bogus)
    end

    @testset "morphology splits semantically-close but unrelated words apart" begin
        # one semantic cluster holding two distinct inflection families plus a look-alike
        # that shares almost every character bigram but no prefix
        words = ["ciudad", "ciudades", "guerra", "guerras", "abioticos", "bioticos"]
        voc4 = Vocabulary(textconfig, words)
        X4 = zeros(Float32, 2, vocsize(voc4))          # all identical -> one cluster
        wv4 = MatrixDatabase(X4)

        # without morphology the whole cluster collapses onto a single representative
        plain = lemma_clusters(voc4, wv4; num_clusters=1, morphology=:none, dist=Dist.L2())
        @test length(unique(values(plain))) == 1

        # with it, each family keeps its own lemma and the look-alike is left alone
        L = lemma_clusters(voc4, wv4; num_clusters=1, morphology=:jaccard,
                            morphology_threshold=0.5, min_common_prefix=3, dist=Dist.L2())
        @test L["ciudades"] == "ciudad"
        @test L["guerras"] == "guerra"
        @test !haskey(L, "guerra") && !haskey(L, "ciudad")
        # "abioticos"/"bioticos" share no 3-char prefix, so they must not be merged
        @test !haskey(L, "abioticos") && !haskey(L, "bioticos")

        @testset "min_common_prefix=0 lets the position-blind match through" begin
            L0 = lemma_clusters(voc4, wv4; num_clusters=1, morphology=:jaccard,
                                 morphology_threshold=0.5, min_common_prefix=0, dist=Dist.L2())
            @test haskey(L0, "abioticos") || haskey(L0, "bioticos")
        end

        @testset "levenshtein is available and handles non-ASCII tokens" begin
            # "»"/"—" are multi-byte: a byte-indexed edit distance (which is what
            # Dist.Seqs.Levenshtein does on a String) throws StringIndexError on them, so
            # this also guards the Char-vector conversion in _morphology_metric.
            voc5 = Vocabulary(textconfig, ["mañana", "mañanas", "»", "—"])
            @test "»" in [token(voc5, i) for i in eachindex(voc5)]
            X5 = zeros(Float32, 2, vocsize(voc5))
            L5 = lemma_clusters(voc5, MatrixDatabase(X5); num_clusters=1,
                                 morphology=:levenshtein, morphology_threshold=0.3,
                                 min_common_prefix=3, dist=Dist.L2())
            @test L5 isa Dict{String,String}
            # the default normalization strips diacritics, so the vocabulary holds "manana"
            @test get(L5, "mananas", "") == "manana"
        end
    end
end
