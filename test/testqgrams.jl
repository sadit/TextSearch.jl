# This file is a part of TextSearch.jl

# a generator `save_profile` does not know how to write
struct _NotSavedGenerator <: AbstractTokenGenerator end
TextSearch.Tokenizer.generate!(::_NotSavedGenerator, buff, pipe, mark) = nothing

qgramconfig(qs...; nlist=Int8[], kwargs...) =
    TextConfig(tokenization=TokenizationConfig(; nlist, generators=[QgramGenerator(q) for q in qs], kwargs...))

@testset "QgramGenerator" begin
    @testset "tokens" begin
        tc = qgramconfig(3)
        # boundary blanks included, and windows cross the word boundary
        @test collect(tokenize(tc, "ab cd")) == [" ab\tq", "ab \tq", "b c\tq", " cd\tq", "cd \tq"]
        # a run of blanks reads as one; layout alone makes no q-grams
        @test collect(tokenize(tc, "ab \t  cd  ")) == collect(tokenize(tc, "ab cd"))
        # each field is its own text: nothing spans "ab" and "cd"
        @test collect(tokenize(tc, ["ab", "cd"])) == [" ab\tq", "ab \tq", " cd\tq", "cd \tq"]
        # normalization applies first, as for words
        @test "ane\tq" in collect(tokenize(tc, "Añejo"))
        # shorter than q: none
        @test isempty(collect(tokenize(qgramconfig(5), "ab")))
        @test length(collect(tokenize(qgramconfig(1), "ab"))) == 4

        @test_throws ArgumentError QgramGenerator(0)
    end

    @testset "several lengths, and with words" begin
        # " abcdef " is 8 characters: 6 windows of 3 and 4 of 5
        toks = collect(tokenize(qgramconfig(3, 5), "abcdef"))
        @test length(toks) == 6 + 4
        @test "abc\tq" in toks && "abcde\tq" in toks

        mixed = collect(tokenize(qgramconfig(3; nlist=[1]), "porque que"))
        @test "que" in mixed && "que\tq" in mixed   # the word and the 3-gram stay apart
        @test count(==("que\tq"), mixed) == 2       # inside "porque", and the word itself
        # untagged, the two are the same token
        untagged = collect(tokenize(qgramconfig(3; nlist=[1], mark_token_type=false), "porque que"))
        @test count(==("que"), untagged) == 3
    end

    @testset "word stages leave tagged q-grams alone" begin
        tc = TextConfig(qgramconfig(3; nlist=[1]); pipeline=TokenPipeline(stopwords=["que"]))
        toks = collect(tokenize(tc, "porque que"))
        @test !("que" in toks)
        @test "que\tq" in toks
    end

    corpus = ["la casa roja", "la casa verde", "la casa azul", "el perro verde", "un gato negro"]
    labels = [1, 1, 1, 2, 2]

    @testset "vocabulary and weighting" begin
        voc = Vocabulary(qgramconfig(3, 5), corpus; verbose=false)
        @test getndocs(voc, token2id(voc, "cas\tq")) == 3
        @test getndocs(voc, token2id(voc, "a ca\tq")) == 0   # a 4-gram: not generated

        model = VectorModel(voc)
        V = vectorize_corpus(model, corpus; verbose=false)
        @test dot(V[1], V[2]) > dot(V[1], V[5])
        @test nnz(vectorize(model, "xyzw")) == 0

        # the Zipfian tail, pruned with the existing token filter
        pruned = filter_tokens(t -> t.ndocs >= 2, voc)
        @test all(>=(2), pruned.ndocs)
        @test token2id(pruned, "cas\tq") != 0 && token2id(pruned, "gat\tq") == 0

        # words and q-grams side by side, each pruned by its own threshold via the tag
        both = Vocabulary(qgramconfig(3; nlist=[1]), corpus; verbose=false)
        isq(t) = endswith(t.token, "\tq")
        kept = filter_tokens(t -> isq(t) ? t.ndocs >= 2 : true, both)
        @test token2id(kept, "gato") != 0 && token2id(kept, "gat\tq") == 0

        em = VectorModel(EntropyWeighting(), TfWeighting(), voc, corpus, labels; mindocs=1, verbose=false)
        # " casa" is only in class 1; "verde" is in both
        @test em.weight[token2id(voc, " casa\tq")] > em.weight[token2id(voc, "verde\tq")]

        ri = RandomIndexing(model; maxoutdim=32)
        @test indim(ri) == vocsize(voc)
        x = vectorize(ri, "la casa")
        y = ri.P * Vector(vectorize(model, "la casa"))
        @test x ≈ y ./ norm(y)
        @test length(bitsketch(ri, "la casa")) == 1
    end

    @testset "saving" begin
        model = VectorModel(Vocabulary(qgramconfig(3, 5; nlist=[1]), corpus; verbose=false))
        p = TextProfile(model)
        mktempdir() do dir
            save_profile(dir, p)
            man = TextSearch.JSON3.read(read(joinpath(dir, "manifest.json"), String))
            # a profile with generators is "1.2", which a "1.1"-only build refuses outright
            @test man.format_version == "1.2"
            q = load_profile(dir)
            @test gettextconfig(q) == gettextconfig(p)
            @test profile_id(q) == profile_id(p)
            @test vectorize(q.model, "la casa verde") == vectorize(model, "la casa verde")

            zipped = joinpath(mktempdir(), "p.zip")
            zip_profile(dir, zipped)
            @test gettextconfig(load_profile(zipped)) == gettextconfig(p)
        end

        # without generators, nothing changes: still "1.1", no generators key
        plain = TextProfile(VectorModel(Vocabulary(TextConfig(), corpus; verbose=false)))
        mktempdir() do dir
            save_profile(dir, plain)
            man = TextSearch.JSON3.read(read(joinpath(dir, "manifest.json"), String))
            @test man.format_version == "1.1"
            @test !haskey(man.policy.tokenization, :generators)
        end

        # any other custom generator still refuses to save
        custom = TextProfile(VectorModel(Vocabulary(
            TextConfig(tokenization=TokenizationConfig(nlist=[1], generators=[_NotSavedGenerator()])), corpus; verbose=false)))
        mktempdir() do dir
            @test_throws ErrorException save_profile(dir, custom)
        end
    end
end
