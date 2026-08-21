using Test, TextSearch, SimilaritySearch, LinearAlgebra, SparseArrays
using SimilaritySearch.ScalarQuant: SQu8, SQgu8

@testset "RandomIndexing (RI)" begin
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

    @testset "Constructors, Methods and Dimensions" begin
        # 1. Default constructor (maxoutdim=1024, :gaussian)
        ri1 = RandomIndexing(vmodel)
        @test ri1 isa RandomIndexing
        @test ri1 isa RIModel
        @test indim(ri1) == vocsize(vmodel)
        @test outdim(ri1) == 1024
        @test ri1.method === :gaussian
        @test size(ri1.P) == (1024, vocsize(vmodel))

        # 2. QR method
        ri_qr = RandomIndexing(vmodel; maxoutdim=512, method=:qr)
        @test outdim(ri_qr) == 512
        @test ri_qr.method === :qr

        # 3. Sparse random / ternary method
        ri_sp = RandomIndexing(vmodel; maxoutdim=256, method=:sparse_random)
        @test outdim(ri_sp) == 256
        @test ri_sp.method === :sparse_random

        # 4. Convenience from TextConfig & corpus
        ri_cfg = RandomIndexing(config, corpus; maxoutdim=128, verbose=false)
        @test ri_cfg isa RandomIndexing
        @test outdim(ri_cfg) == 128

        # 5. Convenience from raw corpus
        ri_raw = RandomIndexing(corpus; maxoutdim=64, verbose=false)
        @test ri_raw isa RandomIndexing
        @test outdim(ri_raw) == 64

        # 6. Accessors and show
        @test vocsize(ri1) == vocsize(vmodel)
        @test gettrainsize(ri1) == gettrainsize(vmodel)
        io = IOBuffer()
        show(io, ri1)
        str = String(take!(io))
        @test occursin("RandomIndexing", str)
        @test occursin("gaussian", str)
    end

    @testset "Dense Float32 Vectorization & Search" begin
        ri = RandomIndexing(vmodel; maxoutdim=128, method=:gaussian)
        k = outdim(ri)

        # Single string vectorize
        q = "Julia machine learning and neural networks"
        v1 = vectorize(ri, q)
        @test v1 isa Vector{Float32}
        @test length(v1) == k
        @test isapprox(norm(v1), 1f0, atol=1e-5)

        # In-place vectorize!
        v2 = zeros(Float32, k)
        vectorize!(v2, ri, q)
        @test isapprox(v1, v2, atol=1e-6)

        # vectorize_corpus
        X_dense = vectorize_corpus(ri, corpus; verbose=false)
        @test X_dense isa MatrixDatabase{Matrix{Float32}}
        @test length(X_dense) == length(corpus)
        @test size(X_dense.matrix) == (k, length(corpus))

        # Search with ExhaustiveSearch
        index_ex = ExhaustiveSearch(Dist.Cosine(), X_dense)
        ectx = GenericContext()
        res_ex = search(index_ex, ectx, v1, knnqueue(KnnSorted, 2))
        @test length(res_ex) == 2
        @test IdView(res_ex)[1] in (4, 5)

        # Search with SearchGraph
        G = SearchGraph(Dist.Cosine(), X_dense)
        ctx = SearchGraphContext()
        index!(G, ctx)
        res_g = search(G, ctx, v1, knnqueue(KnnSorted, 2))
        @test length(res_g) == 2
        @test IdView(res_g)[1] in (4, 5)
    end

    @testset "SQu8 Scalar Quantization Pipeline" begin
        ri = RandomIndexing(vmodel; maxoutdim=128)

        # Single document SQu8
        q = "Julia machine learning and neural networks"
        q_squ8 = vectorize(SQu8, ri, q)
        @test q_squ8 isa SQu8.SQu8Vec

        # Corpus SQu8Database
        db_squ8 = vectorize_corpus(SQu8, ri, corpus; verbose=false)
        @test db_squ8 isa SQu8.SQu8Database
        @test length(db_squ8) == length(corpus)

        # Search against SQu8Database
        index_squ8 = ExhaustiveSearch(SQu8.NormCosine(), db_squ8)
        ectx = GenericContext()
        res_squ8 = search(index_squ8, ectx, q_squ8, knnqueue(KnnSorted, 2))
        @test length(res_squ8) == 2
        @test length(res_squ8) == 2 && DistView(res_squ8)[1] >= 0f0
    end

    @testset "SQgu8 Global Quantization Pipeline" begin
        ri = RandomIndexing(vmodel; maxoutdim=128)

        # Corpus SQgu8 Database
        db_gu8 = vectorize_corpus(SQgu8, ri, corpus; verbose=false)
        @test db_gu8 isa MatrixDatabase{Matrix{UInt8}}
        @test length(db_gu8) == length(corpus)

        # Single document SQgu8 with shared minmax
        q = "Julia machine learning and neural networks"
        q_gu8 = vectorize(SQgu8, ri, q; minmax=(-1f0, 1f0))
        @test q_gu8 isa Vector{UInt8}
        @test length(q_gu8) == outdim(ri)

        # Search against SQgu8
        index_gu8 = ExhaustiveSearch(Dist.NormCosine(), db_gu8)
        ectx = GenericContext()
        res_gu8 = search(index_gu8, ectx, q_gu8, knnqueue(KnnSorted, 2))
        @test length(res_gu8) == 2
    end

    @testset "BitSketch Pipeline" begin
        ri = RandomIndexing(vmodel; maxoutdim=256)

        # Single document BitSketch
        q = "Julia machine learning and neural networks"
        b_q = bitsketch(ri, q)
        @test b_q isa Vector{UInt64}
        @test length(b_q) == cld(256, 64)

        # Dispatched vectorize with BitSketch tag
        b_q2 = vectorize(BitSketch, ri, q)
        @test b_q == b_q2

        # Corpus BitSketch Database
        db_bits = bitsketch(ri, corpus; verbose=false)
        @test db_bits isa MatrixDatabase{Matrix{UInt64}}
        @test length(db_bits) == length(corpus)
        @test size(db_bits.matrix) == (cld(256, 64), length(corpus))

        # Dispatched vectorize_corpus with BitSketch tag
        db_bits2 = vectorize_corpus(BitSketch, ri, corpus; verbose=false)
        @test db_bits.matrix == db_bits2.matrix

        # Hamming search with ExhaustiveSearch
        index_bits = ExhaustiveSearch(Dist.Bits.Hamming(), db_bits)
        ectx = GenericContext()
        res_bits = search(index_bits, ectx, b_q, knnqueue(KnnSorted, 2))
        @test length(res_bits) == 2
        @test IdView(res_bits)[1] in (4, 5)

        # Hamming search with SearchGraph
        G_bits = SearchGraph(Dist.Bits.Hamming(), db_bits)
        ctx = SearchGraphContext()
        index!(G_bits, ctx)
        res_gbits = search(G_bits, ctx, b_q, knnqueue(KnnSorted, 2))
        @test length(res_gbits) == 2
        @test IdView(res_gbits)[1] in (4, 5)
    end
end
