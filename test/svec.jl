
@testset "SparseVector-based vectors" begin
    cmpex(u, v) = abs(u[1] - v[1]) < 1e-3 && u[2] == v[2]

    aL = []
    AL = []
    n = 300
    for i in 1:10
        A = rand(Float32, n)
        B = rand(Float32, n)
        a = sparsevec(collect(1:n), A, n)
        b = sparsevec(collect(1:n), B, n)

        @test abs(norm(A) - norm(a)) < 1e-3
        @test abs(norm(B) - norm(b)) < 1e-3
        normalize!(A); normalize!(a)
        normalize!(B); normalize!(b)
        @test abs(norm(a) - 1.0) < 1e-3
        @test abs(norm(b) - 1.0) < 1e-3

        @test abs(sparsedot(a, b) - dot(A, B)) < 1e-3
        @test abs(dot(a, b) - dot(A, B)) < 1e-3  # native SparseArrays dot, sanity check
        @test abs(maximum(a) - maximum(A)) < 1e-3
        @test abs(minimum(a) - minimum(A)) < 1e-3

        push!(aL, a)
        push!(AL, A)
    end

    @test (norm(sum(AL)) - norm(sum(aL))) < 1e-3

    adist = Dist.Angle()
    cdist = Dist.Cosine()

    for i in 1:length(aL)-1
        @test abs(evaluate(adist, aL[i], aL[i+1]) - evaluate(adist, AL[i], AL[i+1])) < 1e-3
        @test abs(evaluate(cdist, aL[i], aL[i+1]) - evaluate(cdist, AL[i], AL[i+1])) < 1e-3
    end
end

@testset "SparseVector adaptive dot: small/large/ratio thresholds agree with native dot" begin
    Random.seed!(17)
    n = 5000
    for (ka, kb) in [(3, 3), (3, 500), (500, 500), (10, 40), (40, 10), (2000, 5)]
        ia = randperm(n)[1:ka]; ib = randperm(n)[1:kb]
        a = sparsevec(ia, rand(Float32, ka), n)
        b = sparsevec(ib, rand(Float32, kb), n)
        @test isapprox(sparsedot(a, b), dot(a, b); atol=1e-4)
        @test isapprox(sparsedot(a, b; small_threshold=1, ratio_threshold=1.0), dot(a, b); atol=1e-4)
        @test isapprox(sparsedot(b, a), sparsedot(a, b); atol=1e-4)  # symmetry
    end
    a0 = spzeros(Float32, n)
    b0 = sparsevec([1, 2], Float32[1, 2], n)
    @test sparsedot(a0, b0) == 0f0
end

@testset "SparseVector centroid/sum matches Dict-based centroid/sum" begin
    Random.seed!(23)
    n = 400
    dicts = Dict{UInt32,Float32}[]
    svecs = SparseVector{Float32,Int}[]
    for _ in 1:15
        k = rand(1:30)
        idx = randperm(n)[1:k]
        vals = rand(Float32, k)
        push!(dicts, Dict{UInt32,Float32}(UInt32(i) => v for (i, v) in zip(idx, vals)))
        push!(svecs, sparsevec(idx, vals, n))
    end

    cd = centroid(dicts)
    cs = centroid(svecs)
    @test nnz(cs) == length(cd)
    for i in cs.nzind
        @test isapprox(cs[i], get(cd, UInt32(i), 0f0); atol=1e-4)
    end
end
