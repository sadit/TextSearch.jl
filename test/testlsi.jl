using Test, TextSearch, SimilaritySearch, LinearAlgebra, SparseArrays

@testset "LatentSemanticIndexing (LSI)" begin
    corpus = [
        "the quick brown fox jumps over the lazy dog",
        "quick brown dogs and foxes are running in the park",
        "lazy dogs like to sleep all day in the sun",
        "artificial intelligence and machine learning in Julia",
        "deep neural networks and natural language processing with Julia",
        "high performance computing and numerical algorithms",
        "similarity search and nearest neighbor graphs in metric spaces",
        "vector databases and dense semantic embeddings for search"
    ]

    config = TextConfig(tokenization=TokenizationConfig(nlist=[1]))
    voc = Vocabulary(config, corpus; verbose=false)
    vmodel = VectorModel(IdfWeighting(), TfWeighting(), voc)

    @testset "Constructors and Dimensions" begin
        # 1. From VectorModel
        lsi1 = LatentSemanticIndexing(vmodel, corpus; maxoutdim=16, verbose=false)
        @test lsi1 isa LatentSemanticIndexing
        @test lsi1 isa LSIModel
        @test indim(lsi1) == vocsize(vmodel)
        @test outdim(lsi1) <= 16
        @test outdim(lsi1) <= length(corpus)
        @test length(lsi1.s) == outdim(lsi1)
        @test size(lsi1.P) == (outdim(lsi1), indim(lsi1))

        # 2. From TextConfig
        lsi2 = LatentSemanticIndexing(config, corpus; maxoutdim=8, verbose=false)
        @test lsi2 isa LatentSemanticIndexing
        @test outdim(lsi2) <= 8

        # 3. From raw corpus
        lsi3 = LatentSemanticIndexing(corpus; maxoutdim=128, verbose=false)
        @test lsi3 isa LatentSemanticIndexing
        @test outdim(lsi3) <= length(corpus)

        # 4. From pre-vectorized corpus
        X_sparse = vectorize_corpus(vmodel, corpus; verbose=false)
        lsi4 = LatentSemanticIndexing(vmodel, X_sparse; maxoutdim=8, verbose=false)
        @test lsi4 isa LatentSemanticIndexing
        @test outdim(lsi4) == outdim(lsi2)

        # 5. Accessors and show
        @test vocsize(lsi1) == vocsize(vmodel)
        @test trainsize(lsi1) == trainsize(vmodel)
        io = IOBuffer()
        show(io, lsi1)
        str = String(take!(io))
        @test occursin("LatentSemanticIndexing", str)
        @test occursin("scaling", str)
    end

    @testset "vectorize and vectorize!" begin
        lsi = LatentSemanticIndexing(vmodel, corpus; maxoutdim=8, verbose=false)
        k = outdim(lsi)

        # Single string vectorize
        q = "quick brown fox in the park"
        v1 = vectorize(lsi, q)
        @test v1 isa Vector{Float32}
        @test length(v1) == k
        @test isapprox(norm(v1), 1f0, atol=1e-5)

        # In-place vectorize!
        v2 = zeros(Float32, k)
        vectorize!(v2, lsi, q)
        @test isapprox(v1, v2, atol=1e-6)

        # Unnormalized vectorize
        v_unnorm = vectorize(lsi, q; normalize=false)
        @test isapprox(v_unnorm / norm(v_unnorm), v1, atol=1e-5)

        # From SparseVector
        svec = vectorize(vmodel, q)
        v3 = vectorize(lsi, svec)
        @test isapprox(v1, v3, atol=1e-5)

        # Out-of-vocabulary query
        q_oov = "nonexistentwordxyz abcdefgh"
        v_oov = vectorize(lsi, q_oov)
        @test length(v_oov) == k
        @test all(v_oov .== 0f0)
    end

    @testset "vectorize_corpus and Search Integration" begin
        lsi = LatentSemanticIndexing(vmodel, corpus; maxoutdim=8, verbose=false)
        k = outdim(lsi)

        # vectorize_corpus
        X_dense = vectorize_corpus(lsi, corpus; verbose=false)
        @test X_dense isa MatrixDatabase{Matrix{Float32}}
        @test length(X_dense) == length(corpus)
        @test size(X_dense.matrix) == (k, length(corpus))

        for i in 1:length(corpus)
            col = X_dense[i]
            @test isapprox(norm(col), 1f0, atol=1e-5)
            v_single = vectorize(lsi, corpus[i])
            @test isapprox(col, v_single, atol=1e-5)
        end

        # Indexing in ExhaustiveSearch
        index_ex = ExhaustiveSearch(Dist.Cosine(), X_dense)
        ectx = GenericContext()
        query = "Julia machine learning and neural networks"
        qvec = vectorize(lsi, query)
        res_ex = search(index_ex, ectx, qvec, knnqueue(KnnSorted, 2))
        @test length(res_ex) == 2
        # Documents 4 and 5 discuss machine learning in Julia
        @test IdView(res_ex)[1] in (4, 5)

        # Indexing in SearchGraph
        G = SearchGraph(Dist.Cosine(), X_dense)
        ctx = SearchGraphContext()
        index!(G, ctx)
        res_g = search(G, ctx, qvec, knnqueue(KnnSorted, 2))
        @test length(res_g) == 2
        @test IdView(res_g)[1] in (4, 5)
    end

    @testset "wordvectors and synonyms" begin
        lsi = LatentSemanticIndexing(vmodel, corpus; maxoutdim=8, verbose=false)
        k = outdim(lsi)
        m = vocsize(lsi)

        X = wordvectors(lsi)
        @test X isa MatrixDatabase{Matrix{Float32}}
        @test length(X) == m
        @test size(X.matrix) == (k, m)
        for t in 1:m
            @test isapprox(norm(X[t]), 1f0, atol=1e-5)
        end

        # normalize=false keeps the raw (scaling-adjusted) P columns, unnormalized
        X_raw = wordvectors(lsi; normalize=false)
        @test isapprox(X_raw.matrix, lsi.P, atol=1e-6)

        # a single in-vocabulary word reduces, after L2-normalization, to exactly its column of P
        # (a one-nonzero sparse vector projects to a positive scalar multiple of that column)
        tokenID = token2id(voc, "quick")
        @test tokenID > 0
        @test isapprox(X[tokenID], vectorize(lsi, "quick"), atol=1e-5)

        net = synonyms(lsi, 3; verbose=false)
        @test net isa Dict{String,Vector{Pair{String,Float32}}}
        @test length(net) == m
        for (tok, neighbors) in net
            @test length(neighbors) <= 3
            @test all(nb != tok for (nb, _) in neighbors)  # never its own synonym
            @test issorted(last.(neighbors))                # increasing distance
        end
    end

    @testset "factorization: lanczos / full" begin
        # this corpus is far below the auto threshold, so :auto must take the exact path
        @test min(vocsize(vmodel), length(corpus)) <= TextSearch.LSI.LSI_FULL_FACTORIZATION_MAX
        L_auto = LatentSemanticIndexing(vmodel, corpus; maxoutdim=4, verbose=false)
        L_full = LatentSemanticIndexing(vmodel, corpus; maxoutdim=4, factorization=:full, verbose=false)
        @test L_auto.s == L_full.s

        # lanczos (ARPACK) must agree with the dense path to working precision
        L_lcz = LatentSemanticIndexing(vmodel, corpus; maxoutdim=4, factorization=:lanczos, verbose=false)
        @test outdim(L_lcz) == outdim(L_full)
        @test isapprox(L_lcz.s, L_full.s; rtol=1e-5)
        v = vectorize(L_lcz, corpus[1])
        @test length(v) == outdim(L_lcz)
        @test isapprox(norm(v), 1f0, atol=1e-5)

        # asking for as many dimensions as the smaller side leaves ARPACK no room, so the
        # lanczos path must degrade instead of throwing
        L_deg = LatentSemanticIndexing(vmodel, corpus; maxoutdim=length(corpus), factorization=:lanczos, verbose=false)
        @test outdim(L_deg) == min(length(corpus), vocsize(vmodel))

        @test_throws ArgumentError LatentSemanticIndexing(vmodel, corpus; factorization=:bogus, verbose=false)
        # the removed sketch must not linger as a silently-accepted option
        @test_throws ArgumentError LatentSemanticIndexing(vmodel, corpus; factorization=:randomized, verbose=false)
    end

    @testset "synonyms: approximate vs exhaustive search" begin
        lsi = LatentSemanticIndexing(vmodel, corpus; maxoutdim=8, verbose=false)
        m = vocsize(lsi)

        # this vocabulary is far below the auto threshold, so :auto must pick the exact
        # path -- that is what keeps small-corpus results (and these tests) deterministic
        @test m < TextSearch.LSI.SYNONYMS_APPROX_THRESHOLD
        @test synonyms(lsi, 3; verbose=false) == synonyms(lsi, 3; approx=false, verbose=false)

        # forcing the approximate path must still produce a well-formed network
        net = synonyms(lsi, 3; approx=true, verbose=false)
        @test net isa Dict{String,Vector{Pair{String,Float32}}}
        @test length(net) == m
        for (tok, neighbors) in net
            @test length(neighbors) <= 3
            @test all(nb != tok for (nb, _) in neighbors)
            @test issorted(last.(neighbors))
        end

        # recall targets are forwarded, not silently ignored
        net2 = synonyms(lsi, 3; approx=true, construction_recall=0.99, search_recall=0.95, verbose=false)
        @test net2 isa Dict{String,Vector{Pair{String,Float32}}}

        @test_throws ArgumentError synonyms(lsi, 3; approx=:bogus, verbose=false)
    end

    @testset "Scaling options" begin
        for scaling in (:none, :inv_singular_values, :singular_values)
            lsi_s = LatentSemanticIndexing(vmodel, corpus; maxoutdim=4, scaling, verbose=false)
            @test lsi_s.scaling === scaling
            v = vectorize(lsi_s, corpus[1])
            @test length(v) == outdim(lsi_s)
            @test isapprox(norm(v), 1f0, atol=1e-5)
        end

        @test_throws ArgumentError LatentSemanticIndexing(vmodel, corpus; scaling=:invalid_scaling, verbose=false)
    end
end
