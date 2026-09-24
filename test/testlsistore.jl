using Test, TextSearch, SimilaritySearch, LinearAlgebra
using SimilaritySearch.ScalarQuant: SQu8, SQMinC

@testset "save_lsi / load_lsi" begin
    corpus = ["la casa roja tiene jardin", "la casa verde tiene jardin",
              "una manzana roja y una pera", "la pera verde esta rica",
              "la manzana verde esta rica", "el jardin azul es grande",
              "el cielo azul sobre el jardin", "una pera y una manzana verdes"]
    tc = TextConfig(tokenization=TokenizationConfig(nlist=[1]))
    p = fit_profile(tc, corpus; min_ndocs=1, encoder=(; outdim=6), expansion=(; k=2), verbose=false)
    lsi = LatentSemanticIndexing(p.model, corpus; maxoutdim=6, scaling=:none, verbose=false)
    m = vocsize(p.model)

    column(P, j) = Float32[P[i, j] for i in 1:size(P, 1)]
    cosine(a, b) = dot(a, b) / (norm(a) * norm(b))

    @testset "the projection round-trips within a quantization step" begin
        mktempdir() do dir
            d = joinpath(dir, "art")
            save_lsi(d, lsi, p; name="test", repo="sadit/TextSearch.jl", tag="v1.2.0")
            @test Set(readdir(d)) == Set(["manifest.json", "projection_codes.bin",
                                          "projection_mins.bin", "projection_scales.bin",
                                          "singular_values.bin"])

            back = load_lsi(d, p)
            @test outdim(back) == outdim(lsi)
            @test back.scaling === lsi.scaling
            @test back.s == lsi.s

            # per column, which is the geometry the token-level uses depend on and the reason
            # this is quantized per column rather than over the whole matrix
            @test minimum(cosine(column(lsi.P, j), column(back.P, j)) for j in 1:m) > 0.999
            # and the document projection, which is a weighted sum of those columns
            v1, v2 = vectorize(lsi, "la casa con jardin"), vectorize(back, "la casa con jardin")
            @test cosine(v1, v2) > 0.9999
        end
    end

    @testset "it reads back from a zip as well as a directory" begin
        mktempdir() do dir
            d = joinpath(dir, "art")
            save_lsi(d, lsi, p)
            z = joinpath(dir, "art.zip")
            zip_profile(d, z)
            back = load_lsi(z, p)
            @test outdim(back) == outdim(lsi)
            @test column(back.P, 1) == column(load_lsi(d, p).P, 1)
        end
    end

    @testset "outdim truncates exactly, so one artifact serves every smaller one" begin
        mktempdir() do dir
            d = joinpath(dir, "art")
            save_lsi(d, lsi, p)
            full = load_lsi(d, p)
            cut = load_lsi(d, p; outdim=3)

            @test outdim(cut) == 3
            @test cut.s == full.s[1:3]
            # exact, not approximate: the singular values come out ordered and the
            # quantization is per column, so dropping rows changes nothing that remains
            @test all(cut.P[i, j] === full.P[i, j] for i in 1:3, j in 1:m)

            @test_throws ArgumentError load_lsi(d, p; outdim=0)
            @test_throws ArgumentError load_lsi(d, p; outdim=outdim(lsi) + 1)
        end
    end

    @testset "the wrong profile is refused, and named" begin
        other = fit_profile(tc, vcat(corpus, "un documento totalmente distinto"); min_ndocs=1,
                            encoder=(; outdim=6), expansion=(; k=2), verbose=false)
        mktempdir() do dir
            d = joinpath(dir, "art")
            save_lsi(d, lsi, p; name="test", repo="sadit/TextSearch.jl", tag="v1.2.0")

            err = try (load_lsi(d, other); nothing) catch e; e end
            @test err isa ErrorException
            @test occursin(profile_id(p), err.msg)      # the one it wants
            @test occursin(profile_id(other), err.msg)  # the one it got
            @test occursin("test", err.msg)             # and how to find it
            @test occursin("sadit/TextSearch.jl", err.msg)

            # the same profile, re-saved, is still the same profile
            mktempdir() do pdir
                save_profile(pdir, p)
                @test outdim(load_lsi(d, load_profile(pdir))) == outdim(lsi)
            end
        end
    end

    @testset "binding an LSI to a vocabulary that is not its own is refused at save time" begin
        other = fit_profile(tc, vcat(corpus, "un documento totalmente distinto"); min_ndocs=1,
                            encoder=(; outdim=6), expansion=(; k=2), verbose=false)
        mktempdir() do dir
            @test_throws ErrorException save_lsi(joinpath(dir, "art"), lsi, other)
        end
    end

    @testset "quantized_wordvectors searches the codes without expanding them" begin
        mktempdir() do dir
            d = joinpath(dir, "art")
            save_lsi(d, lsi, p)
            back = load_lsi(d, p)

            q = quantized_wordvectors(back)
            @test q isa SQu8.SQu8Database
            @test length(q) == m
            # the stored codes themselves, not a re-quantization of anything
            @test q.Q === back.P.codes

            ctx = GenericContext()
            W = wordvectors(back)
            dense = ExhaustiveSearch(Dist.NormCosine(), W)
            quant = ExhaustiveSearch(SQu8.NormCosine(), q)
            ids(R) = collect(IdView(R))

            # As SETS, not as ordered lists. This corpus is small enough that six pairs of
            # tokens occur in exactly the same documents -- casa/tiene, una/y, manzana/pera,
            # esta/rica, es/grande, cielo/sobre -- so LSI gives each pair one identical column
            # and their order within a result is an arbitrary tie-break that neither path
            # promises. The dense reference does not even return every token as its own
            # nearest neighbour here, for the same reason.
            @test all(Set(ids(search(quant, ctx, q[j], knnqueue(KnnSorted, 3)))) ==
                      Set(ids(search(dense, ctx, W[j], knnqueue(KnnSorted, 3)))) for j in 1:m)

            @testset "and the dense route gives the same thing" begin
                qd = quantized_wordvectors(lsi)
                @test qd isa SQu8.SQu8Database
                @test length(qd) == m
                @test ids(search(ExhaustiveSearch(SQu8.NormCosine(), qd), ctx, qd[1],
                                 knnqueue(KnnSorted, 3))) ==
                      ids(search(dense, ctx, W[1], knnqueue(KnnSorted, 3)))
            end

            @testset "the normalization is load-bearing, not decoration" begin
                # Handing `NormCosine` the RAW codes is the shortcut this function exists to
                # prevent: that distance is `1 - dot`, a cosine only for unit vectors, and LSI
                # columns are not unit vectors. Measured over 4,273 real columns the top-10
                # overlap against the dense answer drops to 0.25, with a token its own nearest
                # neighbour 7% of the time. Here it is enough to pin that the shortcut is not
                # equivalent: some token stops being its own nearest neighbour.
                raw = SQu8.SQu8Database([SQMinC(back.P.mins[j], back.P.scales[j]) for j in 1:m],
                                        back.P.codes)
                rawidx = ExhaustiveSearch(SQu8.NormCosine(), raw)
                nbrs(idx, db, j) = Set(ids(search(idx, ctx, db[j], knnqueue(KnnSorted, 3))))
                ref(j) = Set(ids(search(dense, ctx, W[j], knnqueue(KnnSorted, 3))))

                # The normalized route reproduces the dense neighbourhood for EVERY token;
                # the raw-code shortcut disagrees on a large share of them (8 of 19 as this
                # fixture stands). The bound is loose on purpose: 19 tokens, six pairs of them
                # sharing a column, and only three neighbours asked for leaves little room to
                # diverge, so this understates the effect badly. Measured on a real 4,273-token
                # vocabulary the shortcut's top-10 overlap against the dense answer is 0.25.
                @test count(nbrs(quant, q, j) != ref(j) for j in 1:m) == 0
                @test count(nbrs(rawidx, raw, j) != ref(j) for j in 1:m) >= m ÷ 4
            end
        end
    end

    @testset "the codes are SQu8's own, byte for byte" begin
        # The point of quantizing this way rather than inventing an encoding: the stored codes
        # ARE a SimilaritySearch quantized database's contents, so the same artifact can be
        # read as a dense projection or handed to a search pipeline that never dequantizes.
        X = randn(Float32, 16, 500)
        codes, mins, scales = TextSearch._quantize_columns(X)
        db = SQu8.quantize(X)
        @test codes == db.Q
        @test all(db.E[j].min === mins[j] && db.E[j].c === scales[j] for j in 1:size(X, 2))
    end

    @testset "QuantizedProjection checks it has one parameter pair per column" begin
        @test_throws DimensionMismatch QuantizedProjection(zeros(UInt8, 4, 3),
                                                           zeros(Float32, 2), zeros(Float32, 3))
    end
end
