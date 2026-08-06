using Snowball, Languages

@testset "SnowballTokenTransformation stems unigrams" begin
    tt = SnowballTokenTransformation(Languages.Spanish())
    cfg = TextConfig(transformation=tt)
    toks = collect(tokenize(cfg, "las casas rojas"))
    @test toks == [Snowball.stem(Snowball.Stemmer("spanish"), w) for w in ["las", "casas", "rojas"]]

    tt_en = SnowballTokenTransformation(Languages.English())
    cfg_en = TextConfig(transformation=tt_en)
    @test collect(tokenize(cfg_en, "fishing fishes fisher")) == ["fish", "fish", "fisher"]
end

@testset "language-aware IgnoreStopwords" begin
    cfg = TextConfig(transformation=IgnoreStopwords(Languages.Spanish()))
    @test collect(tokenize(cfg, "la casa roja")) == ["casa", "roja"]
end

@testset "SnowballTokenTransformation + IgnoreStopwords via ChainTransformation" begin
    lang = Languages.Spanish()
    cfg = TextConfig(transformation=ChainTransformation([
        IgnoreStopwords(lang),
        SnowballTokenTransformation(lang),
    ]))
    stemmer = Snowball.Stemmer("spanish")
    expected = [Snowball.stem(stemmer, w) for w in ["casa", "roja"]]
    @test collect(tokenize(cfg, "la casa roja")) == expected
end

@testset "unsupported language errors clearly" begin
    @test_throws ErrorException SnowballTokenTransformation(Languages.Mandarin())
end
