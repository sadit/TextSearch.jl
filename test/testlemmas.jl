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
        lemmas = lemma_clusters(voc, wordvecs; algorithm=:fft, num_clusters=3, selector=:shortest, dist=Dist.L2())
        @test lemmas["cats"] == "cat"
        @test lemmas["dogs"] == "dog"
        @test !haskey(lemmas, "cat")
        @test !haskey(lemmas, "dog")
        @test !haskey(lemmas, "fish")
    end

    @testset "num_clusters=0 defaults to sqrt(vocsize) heuristic and still runs" begin
        lemmas = lemma_clusters(voc, wordvecs; dist=Dist.L2())
        @test lemmas isa Dict{String,String}
    end

    @testset "most_frequent selector picks the higher-occs token" begin
        corpus2 = ["cat", "cats", "cats", "cats"]
        voc2 = Vocabulary(textconfig, corpus2)
        X2 = Matrix{Float32}(undef, 2, vocsize(voc2))
        for tid in 1:vocsize(voc2)
            X2[:, tid] = coords[token(voc2, tid)]
        end
        lemmas = lemma_clusters(voc2, MatrixDatabase(X2); num_clusters=1, selector=:most_frequent, dist=Dist.L2())
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
        lemmas = lemma_clusters(voc3, MatrixDatabase(X3); num_clusters=1, selector=:shortest_then_most_frequent, dist=Dist.L2())
        @test lemmas["by"] == "ax"  # same length, occs("ax")=3 > occs("by")=1
    end

    @testset "unknown algorithm/selector error clearly" begin
        @test_throws ErrorException lemma_clusters(voc, wordvecs; algorithm=:bogus)
        @test_throws ErrorException lemma_clusters(voc, wordvecs; selector=:bogus)
    end
end
