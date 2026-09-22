using Test, TextSearch, SimilaritySearch, JSON3

@testset "save_profile / load_profile / zip_profile" begin
    corpus = [
        "la casa roja",
        "la casa verde",
        "la manzana roja",
        "la pera verde esta rica",
    ]
    # Every key and neighbour is a vocabulary token: the network is stored as ids now, and
    # a name the vocabulary lacks could never have matched anything anyway.
    query_expansion = Dict("casa" => ["roja", "verde"], "pera" => ["manzana"])
    query_expansion_distances = Dict("casa" => Float32[0.12, 0.20], "pera" => Float32[0.1])
    lemmas = Dict("casas" => "casa", "peras" => "pera")
    stopwords = Set(["la", "esta"])
    lineage = [LineageStep(:fit; trainsize=4, outdim=8)]

    tc = TextConfig(tokenization=TokenizationConfig(nlist=[1]))
    mkmodel(docs=corpus; textconfig=tc) =
        VectorModel(IdfWeighting(), TfWeighting(), Vocabulary(textconfig, docs; verbose=false))

    @testset "directory layout and round-trip" begin
        p = TextProfile(mkmodel(); stopwords, lemmas, query_expansion, query_expansion_distances, lineage,
                        applied=AppliedArtifacts(stopwords=true, query_expansion=true))

        dir = tempname()
        try
            save_profile(dir, p)

            # one file per "large" variable, not a single big JSON blob -- and each artifact
            # appears exactly ONCE, which is the point of the layout
            for f in ("manifest.json", "vocabulary.json", "weights.json", "stopwords.json",
                      "lemmas.json", "query_expansion_counts.bin",
                      "query_expansion_neighbors.bin", "query_expansion_distances.bin")
                @test isfile(joinpath(dir, f))
            end
            # the network is the binary part: as strings it was 86% of a real profile, mostly a
            # permuted second copy of the vocabulary (see `_save_expansion`)
            @test !isfile(joinpath(dir, "query_expansion.json"))
            @test !isfile(joinpath(dir, "query_expansion_distances.json"))
            @test !isfile(joinpath(dir, "lemma_map.json"))            # no second lemma copy
            @test !isfile(joinpath(dir, "stopword_candidates.json"))  # no second stopword copy

            q = load_profile(dir)

            @test q.model.voc.token == p.model.voc.token
            @test q.model.voc.occs == p.model.voc.occs
            @test q.model.voc.ndocs == p.model.voc.ndocs
            @test q.model.voc.trainsize[] == p.model.voc.trainsize[]
            @test q.model.voc.numtokens[] == p.model.voc.numtokens[]
            @test q.model.global_weighting isa IdfWeighting
            @test q.model.local_weighting isa TfWeighting
            @test q.model.maxoccs == p.model.maxoccs
            @test q.model.weight == p.model.weight

            @test q.stopwords == stopwords
            @test q.lemmas == lemmas
            @test q.query_expansion == query_expansion
            # Quantized, so this is the one field that does NOT round-trip exactly. What has to
            # survive is the ranking it induces -- that is all `expand_query!` reads it for --
            # plus a value within the u8 step of the original.
            @test keys(q.query_expansion_distances) == keys(query_expansion_distances)
            for (tok, ds) in query_expansion_distances
                got = q.query_expansion_distances[tok]
                @test length(got) == length(ds)
                @test maximum(abs.(got .- ds); init=0f0) < 0.01
                @test sortperm(got) == sortperm(ds)
            end
            @test q.applied == p.applied
            @test length(q.lineage) == 1
            @test q.lineage[1].stage === :fit
            @test q.lineage[1].params["trainsize"] == 4

            @test vectorize(q.model, "la casa roja") == vectorize(p.model, "la casa roja")
        finally
            rm(dir; force=true, recursive=true)
        end
    end

    @testset "no artifacts: files and manifest keys omitted" begin
        p = TextProfile(mkmodel())
        dir = tempname()
        try
            save_profile(dir, p)
            for f in ("stopwords.json", "lemmas.json", "query_expansion_counts.bin",
                      "query_expansion_neighbors.bin", "query_expansion_distances.bin")
                @test !isfile(joinpath(dir, f))
            end

            q = load_profile(dir)
            @test isempty(q.stopwords)
            @test isempty(q.lemmas)
            @test isempty(q.query_expansion)
            @test q.query_expansion_distances === nothing
            @test q.applied == AppliedArtifacts()
            @test isempty(q.lineage)
        finally
            rm(dir; force=true, recursive=true)
        end
    end

    @testset "the applied marker survives the round-trip, per artifact" begin
        # what makes a base profile a base: artifacts carried but not in the pipeline
        for (sw, lem, syn) in Iterators.product((false, true), (false, true), (false, true))
            p = TextProfile(mkmodel(); stopwords, lemmas, query_expansion,
                            applied=AppliedArtifacts(stopwords=sw, lemmas=lem, query_expansion=syn))
            dir = tempname()
            try
                save_profile(dir, p)
                q = load_profile(dir)
                @test q.applied == AppliedArtifacts(stopwords=sw, lemmas=lem, query_expansion=syn)
                # and the config it tokenizes with follows the marker, not the mere presence
                # of the artifact
                @test (gettextconfig(q).pipeline.lemmas !== nothing) == lem
            finally
                rm(dir; force=true, recursive=true)
            end
        end
    end

    @testset "a network naming a token the vocabulary lacks is refused at save time" begin
        # The layout stores ids, so it cannot express one; and the query path could not have
        # used it either, since `bagofwords!` and `expand_query!` both skip an id of 0. Better
        # to say so than to drop it on the way out.
        p = TextProfile(mkmodel(); query_expansion=Dict("casa" => ["hogar"]))
        dir = tempname()
        try
            err = try (save_profile(dir, p); nothing) catch e; e end
            @test err isa ErrorException
            @test occursin("not in the vocabulary", err.msg)
            @test occursin("hogar", err.msg)
        finally
            rm(dir; force=true, recursive=true)
        end
    end

    @testset "query_expansion distances are optional and can be dropped" begin
        p = TextProfile(mkmodel(); query_expansion)   # ranking only
        dir = tempname()
        try
            save_profile(dir, p)
            @test isfile(joinpath(dir, "query_expansion_neighbors.bin"))
            @test !isfile(joinpath(dir, "query_expansion_distances.bin"))
            q = load_profile(dir)
            @test q.query_expansion == query_expansion
            @test q.query_expansion_distances === nothing
        finally
            rm(dir; force=true, recursive=true)
        end
    end

    @testset "an older format version is refused by name, not half-parsed" begin
        p = TextProfile(mkmodel())
        dir = tempname()
        try
            save_profile(dir, p)
            man = JSON3.read(read(joinpath(dir, "manifest.json"), String), Dict{String,Any})
            # any version that is not the current one; "1.0" would silently stop testing the
            # refusal the day the current version became 1.0, which is exactly what happened
            man["format_version"] = "0.9"
            open(io -> JSON3.write(io, man), joinpath(dir, "manifest.json"), "w")

            err = try
                load_profile(dir); nothing
            catch e
                sprint(showerror, e)
            end
            @test err !== nothing
            @test occursin("0.9", err)          # says which version it found
            @test occursin("Refit", err)        # and what to do about it
        finally
            rm(dir; force=true, recursive=true)
        end
    end

    @testset "custom tokenization generators are refused rather than mis-saved" begin
        cfg = TextConfig(tokenization=TokenizationConfig(nlist=[1],
                                                         generators=[UnigramGenerator()]))
        p = TextProfile(mkmodel(; textconfig=cfg))
        dir = tempname()
        try
            @test_throws ErrorException save_profile(dir, p)
        finally
            rm(dir; force=true, recursive=true)
        end
    end

    @testset "zip_profile packages a directory, load_profile reads it back directly" begin
        p = TextProfile(mkmodel(); stopwords, lemmas, query_expansion, query_expansion_distances,
                        applied=AppliedArtifacts(stopwords=true, lemmas=true))
        dir = tempname()
        try
            save_profile(dir, p)
            zippath = zip_profile(dir)
            try
                @test isfile(zippath)
                q = load_profile(zippath)
                @test q.model.voc.token == p.model.voc.token
                @test q.stopwords == stopwords
                @test q.lemmas == lemmas
                @test q.applied.lemmas
                # the zip and the directory are the same profile
                d = load_profile(dir)
                @test q.model.weight == d.model.weight
            finally
                rm(zippath; force=true)
            end
        finally
            rm(dir; force=true, recursive=true)
        end
    end

    @testset "zip_profile deflates, and a stored archive still loads" begin
        # It stored uncompressed for a long time without anyone noticing, because a profile zip
        # lands within a kilobyte of the sum of its members and that reads like framing overhead.
        # On a real profile this was 766,486 bytes against 277,874 -- the largest single saving
        # this format has available.
        p = TextProfile(mkmodel(); stopwords, lemmas, query_expansion, query_expansion_distances,
                        lineage)
        mktempdir() do dir
            d = joinpath(dir, "prof")
            save_profile(d, p)
            members = sum(filesize(joinpath(d, f)) for f in readdir(d))

            deflated = zip_profile(d, joinpath(dir, "c.zip"))
            stored = zip_profile(d, joinpath(dir, "s.zip"); compress=false)
            @test filesize(deflated) < filesize(stored)
            @test filesize(stored) >= members          # stored: members plus framing
            # both forms read back, and to the same thing -- compression is not a format change
            a, b = load_profile(deflated), load_profile(stored)
            @test a.model.voc.token == b.model.voc.token == p.model.voc.token
            @test a.query_expansion == b.query_expansion == p.query_expansion
            @test keys(a.query_expansion_distances) == keys(b.query_expansion_distances)
        end
    end

    @testset "zip_profile default zippath is dir * \".zip\"" begin
        p = TextProfile(mkmodel())
        dir = tempname()
        try
            save_profile(dir, p)
            @test zip_profile(dir) == dir * ".zip"
            rm(dir * ".zip"; force=true)
        finally
            rm(dir; force=true, recursive=true)
        end
    end

    @testset "list_remote_profiles and download_profile" begin
        # Querying GitHub release assets
        remotes = list_remote_profiles(; repo="sadit/TextSearch.jl", tag="v1.1.0")
        @test !isempty(remotes)
        @test any(r -> r.name == "en", remotes)
        @test any(r -> r.name == "es", remotes)

        # Custom mock URL returning json release structure
        mock_json = joinpath(tempname() * ".json")
        try
            open(mock_json, "w") do io
                println(io, """
                {
                    "assets": [
                        {"name": "mock_es.zip", "size": 1024, "browser_download_url": "https://example.com/mock_es.zip"}
                    ]
                }
                """)
            end
            mock_remotes = list_remote_profiles(; url="file://" * mock_json)
            @test length(mock_remotes) == 1
            @test mock_remotes[1].name == "mock_es"
            @test mock_remotes[1].size == 1024
        finally
            rm(mock_json; force=true)
        end
    end
end

