using Test, TextSearch, SimilaritySearch, JSON3

@testset "save_profile / load_profile / zip_profile" begin
    corpus = [
        "la casa roja",
        "la casa verde",
        "la manzana roja",
        "la pera verde esta rica",
    ]
    synonyms = Dict("casa" => ["hogar", "vivienda"], "pera" => ["manzana"])
    synonym_distances = Dict("casa" => Float32[0.12, 0.20], "pera" => Float32[0.1])
    lemmas = Dict("casas" => "casa", "peras" => "pera")
    stopwords = Set(["la", "esta"])
    lineage = [LineageStep(:fit; trainsize=4, outdim=8)]

    tc = TextConfig(tokenization=TokenizationConfig(nlist=[1]))
    mkmodel(docs=corpus; textconfig=tc) =
        VectorModel(IdfWeighting(), TfWeighting(), Vocabulary(textconfig, docs; verbose=false))

    @testset "directory layout and round-trip" begin
        p = TextProfile(mkmodel(); stopwords, lemmas, synonyms, synonym_distances, lineage,
                        applied=AppliedArtifacts(stopwords=true, synonyms=true))

        dir = tempname()
        try
            save_profile(dir, p)

            # one file per "large" variable, not a single big JSON blob -- and each artifact
            # appears exactly ONCE, which is the point of the layout
            for f in ("manifest.json", "vocabulary.json", "weights.json", "stopwords.json",
                      "lemmas.json", "synonyms.json", "synonym_distances.json")
                @test isfile(joinpath(dir, f))
            end
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
            @test q.synonyms == synonyms
            @test q.synonym_distances == synonym_distances
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
            for f in ("stopwords.json", "lemmas.json", "synonyms.json", "synonym_distances.json")
                @test !isfile(joinpath(dir, f))
            end

            q = load_profile(dir)
            @test isempty(q.stopwords)
            @test isempty(q.lemmas)
            @test isempty(q.synonyms)
            @test q.synonym_distances === nothing
            @test q.applied == AppliedArtifacts()
            @test isempty(q.lineage)
        finally
            rm(dir; force=true, recursive=true)
        end
    end

    @testset "the applied marker survives the round-trip, per artifact" begin
        # what makes a base profile a base: artifacts carried but not in the pipeline
        for (sw, lem, syn) in Iterators.product((false, true), (false, true), (false, true))
            p = TextProfile(mkmodel(); stopwords, lemmas, synonyms,
                            applied=AppliedArtifacts(stopwords=sw, lemmas=lem, synonyms=syn))
            dir = tempname()
            try
                save_profile(dir, p)
                q = load_profile(dir)
                @test q.applied == AppliedArtifacts(stopwords=sw, lemmas=lem, synonyms=syn)
                # and the config it tokenizes with follows the marker, not the mere presence
                # of the artifact
                @test has_lemma_transformation(textconfig(q).transformation) == lem
            finally
                rm(dir; force=true, recursive=true)
            end
        end
    end

    @testset "synonym distances are optional and can be dropped" begin
        p = TextProfile(mkmodel(); synonyms)   # ranking only
        dir = tempname()
        try
            save_profile(dir, p)
            @test isfile(joinpath(dir, "synonyms.json"))
            @test !isfile(joinpath(dir, "synonym_distances.json"))
            q = load_profile(dir)
            @test q.synonyms == synonyms
            @test q.synonym_distances === nothing
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
            man["format_version"] = "1.0"
            open(io -> JSON3.write(io, man), joinpath(dir, "manifest.json"), "w")

            err = try
                load_profile(dir); nothing
            catch e
                sprint(showerror, e)
            end
            @test err !== nothing
            @test occursin("1.0", err)          # says which version it found
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
        p = TextProfile(mkmodel(); stopwords, lemmas, synonyms, synonym_distances,
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
end
