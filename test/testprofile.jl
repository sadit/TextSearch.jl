using Test, TextSearch, SimilaritySearch
using Snowball, Languages

@testset "save_profile / load_profile" begin
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

    @testset "default TextConfig round-trip" begin
        textconfig = TextConfig(tokenization=TokenizationConfig(nlist=[1]))
        voc = Vocabulary(textconfig, corpus)
        model = VectorModel(IdfWeighting(), TfWeighting(), voc)

        path = tempname() * ".json"
        try
            save_profile(path, model; synonyms)
            reloaded_model, reloaded_synonyms = load_profile(path)

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
            rm(path; force=true)
        end
    end

    @testset "ChainTransformation with Snowball + IgnoreStopwords round-trip" begin
        lang = Languages.Spanish()
        textconfig = TextConfig(
            tokenization=TokenizationConfig(nlist=[1]),
            transformation=ChainTransformation([IgnoreStopwords(lang), SnowballTokenTransformation(lang)]),
        )
        voc = Vocabulary(textconfig, corpus)
        model = VectorModel(IdfWeighting(), TfWeighting(), voc)

        path = tempname() * ".json"
        try
            save_profile(path, model)
            reloaded_model, reloaded_synonyms = load_profile(path)

            @test isempty(reloaded_synonyms)
            @test reloaded_model.voc.token == model.voc.token

            q = "la casa roja"
            @test collect(tokenize(reloaded_model.voc.textconfig, q)) == collect(tokenize(textconfig, q))
            @test vectorize(reloaded_model, q) == vectorize(model, q)
        finally
            rm(path; force=true)
        end
    end
end
