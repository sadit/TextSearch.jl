using Test, TextSearch, SimilaritySearch, JSON3
using Snowball, Languages

@testset "save_profile / load_profile / zip_profile" begin
    corpus = [
        "la casa roja",
        "la casa verde",
        "la manzana roja",
        "la pera verde esta rica",
    ]
    synonyms = Dict(
        "casa" => ["hogar", "vivienda"],
        "pera" => ["manzana"],
    )
    synonym_distances = Dict(
        "casa" => Float32[0.12, 0.20],
        "pera" => Float32[0.1],
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
            save_profile(dir, model; synonyms, synonym_distances, lemmas, stopword_candidates, encoder)

            # one file per "large" variable, not a single big JSON blob
            @test isfile(joinpath(dir, "manifest.json"))
            @test isfile(joinpath(dir, "vocabulary.json"))
            @test isfile(joinpath(dir, "weights.json"))
            @test isfile(joinpath(dir, "synonyms.json"))
            @test isfile(joinpath(dir, "synonym_distances.json"))
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
            @test p.synonym_distances == synonym_distances
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
            @test !isfile(joinpath(dir, "synonym_distances.json"))
            @test !isfile(joinpath(dir, "lemmas.json"))
            @test !isfile(joinpath(dir, "stopword_candidates.json"))

            p = load_profile(dir)
            @test p.synonyms isa Dict{String,Vector{String}} && isempty(p.synonyms)
            @test p.synonym_distances === nothing
            @test p.lemmas isa Dict{String,String} && isempty(p.lemmas)
            @test p.stopword_candidates isa Vector{String} && isempty(p.stopword_candidates)
            @test p.encoder === nothing
        finally
            rm(dir; force=true, recursive=true)
        end
    end

    @testset "legacy interleaved synonyms.json still loads" begin
        # Profiles written before synonyms were split store [[neighbor, distance], ...] in
        # one file. They must keep loading, distances included, without a format bump --
        # the payload identifies itself by shape. This is what keeps already-published
        # profiles usable.
        textconfig = TextConfig(tokenization=TokenizationConfig(nlist=[1]))
        voc = Vocabulary(textconfig, corpus)
        model = VectorModel(IdfWeighting(), TfWeighting(), voc)

        dir = tempname()
        try
            save_profile(dir, model; synonyms, synonym_distances)
            # rewrite synonyms.json in the old layout and drop the separate distances file
            open(joinpath(dir, "synonyms.json"), "w") do io
                JSON3.write(io, Dict(
                    tok => [[syn, synonym_distances[tok][i]] for (i, syn) in enumerate(syns)]
                    for (tok, syns) in synonyms))
            end
            rm(joinpath(dir, "synonym_distances.json"))
            man = JSON3.read(read(joinpath(dir, "manifest.json"), String), Dict{String,Any})
            delete!(man, "synonym_distances_file")
            open(joinpath(dir, "manifest.json"), "w") do io
                JSON3.write(io, man)
            end

            p = load_profile(dir)
            @test p.synonyms == synonyms
            @test p.synonym_distances == synonym_distances   # recovered from the interleaved file
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

    @testset "ChainTransformation with LemmaTransformation + IgnoreStopwords" begin
        # A lemma map baked into the TextConfig, which is what makes the mapping apply to
        # documents and queries alike without either side having to remember to do it.
        lemmacorpus = ["la casa roja", "las casas rojas", "la pera verde"]
        lt = LemmaTransformation(Dict("casas" => "casa", "rojas" => "roja", "las" => "la"))
        textconfig = TextConfig(
            tokenization=TokenizationConfig(nlist=[1]),
            transformation=ChainTransformation([lt, IgnoreStopwords(Set(["la"]))]),
        )
        voc = Vocabulary(textconfig, lemmacorpus)
        model = VectorModel(IdfWeighting(), TfWeighting(), voc)

        # the vocabulary itself is lemmatized: the inflected forms are gone, and the lemma
        # carries the whole family's counts (this is the point of doing it here rather than
        # in each consumer -- idf now covers the family, not one of its forms)
        @test token2id(voc, "casas") == 0
        @test token2id(voc, "rojas") == 0
        @test ndocs(voc, token2id(voc, "casa")) == 2
        @test ndocs(voc, token2id(voc, "roja")) == 2
        # lemmas run BEFORE the stopword filter, so a form that lemmatizes into a stopword
        # is dropped rather than smuggling it back in ("las" -> "la", filtered)
        @test token2id(voc, "las") == 0
        @test token2id(voc, "la") == 0

        # the reverse order is what that guards against, and it is observably wrong
        wrong = TextConfig(
            tokenization=TokenizationConfig(nlist=[1]),
            transformation=ChainTransformation([IgnoreStopwords(Set(["la"])), lt]),
        )
        @test collect(tokenize(wrong, "las casas rojas")) == ["la", "casa", "roja"]
        @test collect(tokenize(textconfig, "las casas rojas")) == ["casa", "roja"]

        dir = tempname()
        try
            save_profile(dir, model)
            # per-kind file naming: both steps carry bulk data, and neither gets a "_2"
            # suffix that would imply a second map of its own kind
            @test isfile(joinpath(dir, "stopwords.json"))
            @test isfile(joinpath(dir, "lemma_map.json"))
            @test !isfile(joinpath(dir, "lemma_map_2.json"))

            p = load_profile(dir)
            @test p.model.voc.token == model.voc.token

            for q in ("las casas rojas", "la casa roja", "una pera")
                @test collect(tokenize(p.model.voc.textconfig, q)) == collect(tokenize(textconfig, q))
                @test vectorize(p.model, q) == vectorize(model, q)
            end
            # a query in the inflected form reaches the lemma, on both sides
            @test collect(tokenize(p.model.voc.textconfig, "casas")) == ["casa"]
        finally
            rm(dir; force=true, recursive=true)
        end
    end

    @testset "LemmaTransformation propagates into n-grams" begin
        lt = LemmaTransformation(Dict("casas" => "casa"))
        @test TextSearch.Tokenizer.transform_unigram(lt, "casas") == "casa"
        @test TextSearch.Tokenizer.transform_unigram(lt, "perro") == "perro"

        # Only transform_unigram is defined, but n-gram generators consume the already
        # transformed word stream, so the lemma reaches each n-gram word by word rather
        # than leaving n-grams on unlemmatized forms.
        cfg = TextConfig(tokenization=TokenizationConfig(nlist=[2]), transformation=lt)
        bigrams = collect(tokenize(cfg, "las casas rojas"))
        @test any(t -> occursin("las casa", t), bigrams)
        @test any(t -> occursin("casa rojas", t), bigrams)
        @test !any(t -> occursin("casas", t), bigrams)

        # constructing from a non-concrete dict type works (the JSON decode path)
        @test LemmaTransformation(Dict{SubString{String},SubString{String}}()).lemmas isa Dict{String,String}
    end

    @testset "zip_profile packages a directory, load_profile reads it back directly" begin
        textconfig = TextConfig(tokenization=TokenizationConfig(nlist=[1]))
        voc = Vocabulary(textconfig, corpus)
        model = VectorModel(IdfWeighting(), TfWeighting(), voc)

        dir = tempname()
        zippath = dir * ".zip"
        try
            save_profile(dir, model; synonyms, synonym_distances, lemmas, stopword_candidates, encoder)
            out = zip_profile(dir, zippath)
            @test out == zippath
            @test isfile(zippath)

            p = load_profile(zippath)
            @test p.model.voc.token == model.voc.token
            @test p.model.weight == model.weight
            @test p.synonyms == synonyms
            @test p.synonym_distances == synonym_distances
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
