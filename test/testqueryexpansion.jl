using Test, TextSearch, SimilaritySearch, LinearAlgebra, SparseArrays

@testset "expand_synonyms!" begin
    corpus = [
        "la casa roja",
        "la casa verde",
        "la manzana roja",
        "la pera verde esta rica",
    ]
    textconfig = TextConfig(tokenization=TokenizationConfig(nlist=[1]))
    voc = Vocabulary(textconfig, corpus)
    model = VectorModel(IdfWeighting(), TfWeighting(), voc)

    @testset "exact weight injection (normalize=false)" begin
        q = vectorize(model, "pera"; normalize=false)
        pera_w = q.nzval[1]
        synonyms = Dict("pera" => [("manzana", 0.1f0)])

        expand_synonyms!(q, voc, synonyms; normalize=false)

        manzana_id = token2id(voc, "manzana")
        j = findfirst(==(manzana_id), q.nzind)
        @test j !== nothing
        @test isapprox(q.nzval[j], pera_w * exp(-0.1f0), atol=1e-6)
    end

    @testset "OOV synonym silently skipped" begin
        q = vectorize(model, "pera"; normalize=false)
        n0 = nnz(q)
        synonyms = Dict("pera" => [("nonexistentwordxyz", 0.1f0)])

        expand_synonyms!(q, voc, synonyms; normalize=false)
        @test nnz(q) == n0  # nothing added
    end

    @testset "empty synonyms dict is a no-op (besides normalization)" begin
        q1 = vectorize(model, "la casa roja"; normalize=false)
        q2 = copy(q1)

        expand_synonyms!(q1, voc, Dict{String,Vector{Pair{String,Float32}}}())
        normalize!(q2)

        @test q1.nzind == q2.nzind
        @test isapprox(q1.nzval, q2.nzval, atol=1e-6)
    end

    @testset "normalized output has unit norm" begin
        q = vectorize(model, "pera"; normalize=false)
        synonyms = Dict("pera" => [("manzana", 0.1f0), ("roja", 0.5f0)])
        expand_synonyms!(q, voc, synonyms)  # normalize=true by default
        @test isapprox(norm(q), 1f0, atol=1e-5)
    end

    @testset "duplicate synonym ids are merged, not left as separate entries" begin
        # two original tokens both point to the same synonym -> must be combined, not duplicated
        q = vectorize(model, "casa roja"; normalize=false)
        synonyms = Dict(
            "casa" => [("verde", 0.2f0)],
            "roja" => [("verde", 0.3f0)],
        )
        expand_synonyms!(q, voc, synonyms; normalize=false)

        verde_id = token2id(voc, "verde")
        @test count(==(verde_id), q.nzind) == 1
        @test issorted(q.nzind)  # SparseVector invariant preserved
    end

    @testset "mutates its argument in place" begin
        q = vectorize(model, "pera"; normalize=false)
        synonyms = Dict("pera" => [("manzana", 0.1f0)])
        out = expand_synonyms!(q, voc, synonyms)
        @test out === q
    end
end
