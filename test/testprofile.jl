using Test, TextSearch, SimilaritySearch
using Snowball, Languages

@testset "save_profile / load_profile / zip_profile" begin
    corpus = [
        "la casa roja",
        "la casa verde",
        "la manzana roja",
        "la pera verde esta rica",
    ]
    synonyms = Dict(
        "casa" => [("hogar" => 0.12f0), ("vivienda" => 0.20f0)],
        "pera" => [("manzana" => 0.1f0)],
    )

    @testset "default TextConfig: directory layout and round-trip" begin
        textconfig = TextConfig(tokenization=TokenizationConfig(nlist=[1]))
        voc = Vocabulary(textconfig, corpus)
        model = VectorModel(IdfWeighting(), TfWeighting(), voc)

        dir = tempname()
        try
            save_profile(dir, model; synonyms)

            # one file per "large" variable, not a single big JSON blob
            @test isfile(joinpath(dir, "manifest.json"))
            @test isfile(joinpath(dir, "vocabulary.json"))
            @test isfile(joinpath(dir, "weights.json"))
            @test isfile(joinpath(dir, "synonyms.json"))
            @test !isfile(joinpath(dir, "stopwords.json"))  # no stopwords transformation here

            reloaded_model, reloaded_synonyms = load_profile(dir)

            @test reloaded_model.voc.token == model.voc.token
            @test reloaded_model.voc.occs == model.voc.occs
            @test reloaded_model.voc.ndocs == model.voc.ndocs
            @test reloaded_model.voc.trainsize[] == model.voc.trainsize[]
            @test reloaded_model.voc.numtokens[] == model.voc.numtokens[]
            @test reloaded_model.global_weighting isa IdfWeighting
            @test reloaded_model.local_weighting isa TfWeighting
            @test reloaded_model.maxoccs == model.maxoccs
            @test reloaded_model.weight == model.weight
            @test reloaded_synonyms == synonyms

            q = "la casa roja"
            @test vectorize(reloaded_model, q) == vectorize(model, q)
        finally
            rm(dir; force=true, recursive=true)
        end
    end

    @testset "no synonyms: synonyms.json is omitted, load_profile returns an empty Dict" begin
        textconfig = TextConfig(tokenization=TokenizationConfig(nlist=[1]))
        voc = Vocabulary(textconfig, corpus)
        model = VectorModel(IdfWeighting(), TfWeighting(), voc)

        dir = tempname()
        try
            save_profile(dir, model)
            @test !isfile(joinpath(dir, "synonyms.json"))

            _, reloaded_synonyms = load_profile(dir)
            @test reloaded_synonyms isa Dict{String,Vector{Pair{String,Float32}}}
            @test isempty(reloaded_synonyms)
        finally
            rm(dir; force=true, recursive=true)
        end
    end

    @testset "ChainTransformation with Snowball + IgnoreStopwords: stopwords.json + round-trip" begin
        lang = Languages.Spanish()
        textconfig = TextConfig(
            tokenization=TokenizationConfig(nlist=[1]),
            transformation=ChainTransformation([IgnoreStopwords(lang), SnowballTokenTransformation(lang)]),
        )
        voc = Vocabulary(textconfig, corpus)
        model = VectorModel(IdfWeighting(), TfWeighting(), voc)

        dir = tempname()
        try
            save_profile(dir, model)
            @test isfile(joinpath(dir, "stopwords.json"))

            reloaded_model, reloaded_synonyms = load_profile(dir)

            @test isempty(reloaded_synonyms)
            @test reloaded_model.voc.token == model.voc.token

            q = "la casa roja"
            @test collect(tokenize(reloaded_model.voc.textconfig, q)) == collect(tokenize(textconfig, q))
            @test vectorize(reloaded_model, q) == vectorize(model, q)
        finally
            rm(dir; force=true, recursive=true)
        end
    end

    @testset "zip_profile packages a directory, load_profile reads it back directly" begin
        textconfig = TextConfig(tokenization=TokenizationConfig(nlist=[1]))
        voc = Vocabulary(textconfig, corpus)
        model = VectorModel(IdfWeighting(), TfWeighting(), voc)

        dir = tempname()
        zippath = dir * ".zip"
        try
            save_profile(dir, model; synonyms)
            out = zip_profile(dir, zippath)
            @test out == zippath
            @test isfile(zippath)

            reloaded_model, reloaded_synonyms = load_profile(zippath)
            @test reloaded_model.voc.token == model.voc.token
            @test reloaded_model.weight == model.weight
            @test reloaded_synonyms == synonyms

            q = "la casa roja"
            @test vectorize(reloaded_model, q) == vectorize(model, q)
        finally
            rm(dir; force=true, recursive=true)
            rm(zippath; force=true)
        end
    end

    @testset "zip_profile default zippath is dir * \".zip\"" begin
        textconfig = TextConfig(tokenization=TokenizationConfig(nlist=[1]))
        voc = Vocabulary(textconfig, corpus)
        model = VectorModel(IdfWeighting(), TfWeighting(), voc)

        dir = tempname()
        try
            save_profile(dir, model)
            out = zip_profile(dir)
            try
                @test out == dir * ".zip"
                @test isfile(out)
            finally
                rm(out; force=true)
            end
        finally
            rm(dir; force=true, recursive=true)
        end
    end
end
