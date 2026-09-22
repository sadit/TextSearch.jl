using Test, TextSearch, JSON3

@testset "binary array members" begin
    TS = TextSearch

    @testset "round trip at every supported dtype and shape" begin
        mktempdir() do dir
            read_bytes(name) = read(joinpath(dir, name))
            for A in (UInt8[1, 2, 250], UInt32[1, 70000, 3], Int32[-5, 0, 7],
                      Float32[1.5, -2.25, 0.0])
                e = TS._save_array(dir, "a.bin", A)
                # the manifest fully describes the bytes; nothing is self-describing on disk
                @test e["shape"] == [length(A)]
                got = TS._load_array(read_bytes, e)
                @test got == A
                @test eltype(got) === eltype(A)
            end

            # a matrix comes back a matrix, not a flat vector
            M = Float32[1 2 3; 4 5 6]
            e = TS._save_array(dir, "m.bin", M)
            @test e["shape"] == [2, 3]
            @test TS._load_array(read_bytes, e) == M
        end
    end

    @testset "u8 quantization is bounded by its own step" begin
        x = Float32[0.0, 0.25, 0.5, 0.75, 0.938]
        q, lo, hi = TS._quantize_u8(x)
        @test eltype(q) === UInt8
        # exactly the observed extremes, in Float64: the range is data-derived, not assumed
        @test lo == Float64(minimum(x)) && hi == Float64(maximum(x))
        back = TS._dequantize_u8(q, lo, hi)
        # half a step of a 255-level grid over the observed range
        @test maximum(abs.(back .- x)) <= (hi - lo) / 255 / 2 + 1e-6
        @test eltype(back) === Float32

        # the endpoints are exact, which is what keeps a range from drifting on a round trip
        @test back[1] ≈ x[1]
        @test back[end] ≈ x[end]

        # a constant array is the degenerate case and is exact rather than special-cased
        c = Float32[0.7, 0.7, 0.7]
        qc, lc, hc = TS._quantize_u8(c)
        @test all(==(0x00), qc)
        @test TS._dequantize_u8(qc, lc, hc) ≈ c
    end

    @testset "a truncated or mismatched member is refused, not reinterpreted" begin
        mktempdir() do dir
            A = Float32[1, 2, 3, 4]
            e = TS._save_array(dir, "a.bin", A)
            trunc_bytes(name) = read(joinpath(dir, name))[1:end-4]
            # would otherwise reinterpret into plausible-looking garbage
            @test_throws ErrorException TS._load_array(trunc_bytes, e)

            bad = copy(e); bad["dtype"] = "f64"
            @test_throws ErrorException TS._load_array(name -> read(joinpath(dir, name)), bad)
        end
    end

    @testset "an unsupported element type is refused at save time" begin
        mktempdir() do dir
            @test_throws ErrorException TS._save_array(dir, "a.bin", Float64[1.0, 2.0])
        end
    end

    @testset "an older profile is refused by version, not silently emptied" begin
        # Found the hard way: the first cut of the binary-distances change simply did not look
        # for the old JSON key, so an older profile loaded fine and came back with NO distances
        # -- `expand_query!` would quietly fall back to rank weighting and nothing would say
        # why. That was patched with an ad-hoc check on the moved key; the format version does
        # the job properly now that it is allowed to move, and catches every other way an older
        # file differs at the same time.
        corpus = ["la casa roja", "la casa verde", "la pera verde esta rica"]
        p = fit_profile(TextConfig(), corpus; min_ndocs=1,
                        encoder=(; outdim=2), expansion=(; k=2), verbose=false)
        mktempdir() do dir
            save_profile(dir, p)
            mpath = joinpath(dir, "manifest.json")
            m = JSON3.read(read(mpath), Dict{String,Any})
            m["format_version"] = "1.0"
            m["artifacts"]["query_expansion"]["distances_file"] = "query_expansion_distances.json"
            delete!(m["artifacts"]["query_expansion"], "distances")
            open(io -> JSON3.write(io, m), mpath, "w")
            err = try (load_profile(dir); nothing) catch e; e end
            @test err isa ErrorException
            @test occursin("format_version", err.msg)
            @test occursin("Refit", err.msg)
        end
    end

    @testset "a profile's distances survive the binary round trip" begin
        corpus = ["la casa roja tiene jardin", "la casa verde tiene jardin",
                  "una manzana roja y una pera", "la pera verde esta rica",
                  "la manzana verde esta rica", "el jardin azul es grande"]
        p = fit_profile(TextConfig(), corpus; min_ndocs=1,
                        encoder=(; outdim=4), expansion=(; k=3), verbose=false)
        @test p.query_expansion_distances !== nothing

        mktempdir() do dir
            d = joinpath(dir, "prof")
            save_profile(d, p)
            # the binary member is there, and the old JSON one is not
            @test isfile(joinpath(d, "query_expansion_distances.bin"))
            @test !isfile(joinpath(d, "query_expansion_distances.json"))

            for src in (d, zip_profile(d, joinpath(dir, "prof.zip")))   # directory and zip alike
                q = load_profile(src)
                @test keys(q.query_expansion_distances) == keys(p.query_expansion_distances)
                for (tok, ds) in p.query_expansion_distances
                    got = q.query_expansion_distances[tok]
                    @test length(got) == length(ds)
                    # quantized, so equal only to within the step -- the ranking is what matters
                    # and it is preserved exactly
                    @test maximum(abs.(got .- ds); init=0f0) < 0.01
                    @test sortperm(got) == sortperm(ds)
                end
                @test q.query_expansion == p.query_expansion
            end
        end
    end
end
