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
    lemmas = Dict("casas" => "casa", "peras" => "pera")
    stopword_candidates = ["la", "esta"]
    encoder = (; kind=:lsi, outdim=8, scaling=:none, source_path="")

    @testset "default TextConfig: directory layout and round-trip" begin
        textconfig = TextConfig(tokenization=TokenizationConfig(nlist=[1]))
        voc = Vocabulary(textconfig, corpus)
        model = VectorModel(IdfWeighting(), TfWeighting(), voc)

        dir = tempname()
        try
            save_profile(dir, model; synonyms, lemmas, stopword_candidates, encoder)

            # one file per "large" variable, not a single big JSON blob
            @test isfile(joinpath(dir, "manifest.json"))
            @test isfile(joinpath(dir, "vocabulary.json"))
            @test isfile(joinpath(dir, "weights.json"))
            @test isfile(joinpath(dir, "synonyms.json"))
            @test isfile(joinpath(dir, "lemmas.json"))
            @test isfile(joinpath(dir, "stopword_candidates.json"))
            @test !isfile(joinpath(dir, "stopwords.json"))  # no stopwords transformation here

            p = load_profile(dir)

            @test p.model.voc.token == model.voc.token
            @test p.model.voc.occs == model.voc.occs
            @test p.model.voc.ndocs == model.voc.ndocs
            @test p.model.voc.trainsize[] == model.voc.trainsize[]
            @test p.model.voc.numtokens[] == model.voc.numtokens[]
            @test p.model.global_weighting isa IdfWeighting
            @test p.model.local_weighting isa TfWeighting
            @test p.model.maxoccs == model.maxoccs
            @test p.model.weight == model.weight
            @test p.synonyms == synonyms
            @test p.lemmas == lemmas
            @test sort(p.stopword_candidates) == sort(stopword_candidates)
            @test p.encoder["kind"] == "lsi"
            @test p.encoder["outdim"] == 8
            @test p.encoder["scaling"] == "none"

            q = "la casa roja"
            @test vectorize(p.model, q) == vectorize(model, q)
        finally
            rm(dir; force=true, recursive=true)
        end
    end

    @testset "none of synonyms/lemmas/stopword_candidates/encoder: files+keys omitted" begin
        textconfig = TextConfig(tokenization=TokenizationConfig(nlist=[1]))
        voc = Vocabulary(textconfig, corpus)
        model = VectorModel(IdfWeighting(), TfWeighting(), voc)

        dir = tempname()
        try
            save_profile(dir, model)
            @test !isfile(joinpath(dir, "synonyms.json"))
            @test !isfile(joinpath(dir, "lemmas.json"))
            @test !isfile(joinpath(dir, "stopword_candidates.json"))

            p = load_profile(dir)
            @test p.synonyms isa Dict{String,Vector{Pair{String,Float32}}} && isempty(p.synonyms)
            @test p.lemmas isa Dict{String,String} && isempty(p.lemmas)
            @test p.stopword_candidates isa Vector{String} && isempty(p.stopword_candidates)
            @test p.encoder === nothing
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

            p = load_profile(dir)

            @test isempty(p.synonyms)
            @test p.model.voc.token == model.voc.token

            q = "la casa roja"
            @test collect(tokenize(p.model.voc.textconfig, q)) == collect(tokenize(textconfig, q))
            @test vectorize(p.model, q) == vectorize(model, q)
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
            save_profile(dir, model; synonyms, lemmas, stopword_candidates, encoder)
            out = zip_profile(dir, zippath)
            @test out == zippath
            @test isfile(zippath)

            p = load_profile(zippath)
            @test p.model.voc.token == model.voc.token
            @test p.model.weight == model.weight
            @test p.synonyms == synonyms
            @test p.lemmas == lemmas
            @test sort(p.stopword_candidates) == sort(stopword_candidates)
            @test p.encoder["kind"] == "lsi"

            q = "la casa roja"
            @test vectorize(p.model, q) == vectorize(model, q)
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
