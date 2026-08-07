
function test_equals(a, b)
    a = a isa TokenizedText ? a.tokens : a
    b = b isa TokenizedText ? b.tokens : b
    if a != b
        @info :diff => setdiff(a, b)
        @info :intersection => intersect(a, b)
        @info :evaluated => a
        @info :correct => b
        error("a difference was found")
    end

    @test a == b
end

@testset "individual tokenizers" begin
    m = TextConfig(tokenization=TokenizationConfig(nlist=[1]))
    test_equals(tokenize(m, text0), ["@user", ";)", "#jello", ".", "world"])

    m = TextConfig(tokenization=TokenizationConfig(nlist=[2]))
    test_equals(tokenize(m, text0), ["@user ;)\tn", ";) #jello\tn", "#jello .\tn", ". world\tn"])

    m = TextConfig(tokenization=TokenizationConfig(nlist=[3]))
    test_equals(tokenize(m, text0), ["@user ;) #jello\tn", ";) #jello .\tn", "#jello . world\tn"])

    m = TextConfig(tokenization=TokenizationConfig(nlist=[1]))
    test_equals(tokenize(m, text1), ["hello", "world", "!!", "@user", ";)", "#jello", ".", "world", ":)"])
end

@testset "message vectors" begin
    m = TextConfig(tokenization=TokenizationConfig(nlist=[1, 2]))
    A = tokenize(m, "hello ;) #jello world.")
    B = tokenize(m, ["hello ;)", "#jello world."])
    push!(B, ";) #jello\tn")
    test_equals(sort(A.tokens), sort(B.tokens))
    # @show sort(A) sort(B)
end

@testset "vocabulary of different kinds of docs" begin
    textconfig = TextConfig(tokenization=TokenizationConfig(nlist=[1]))
    A = Vocabulary(textconfig, ["hello ;)", "#jello world."])
    B = Vocabulary(textconfig, [["hello ;)", "#jello world."]])
    @show token(A)
    @show token(B)
    @test occs(A) == occs(B)
    @test sort(token(A)) == sort(token(B))
    @info trainsize(A), trainsize(B)
    @test trainsize(A) == 2 && trainsize(B) == 1
    C = merge_voc(A, B)
    @test token(C) == token(A)
    @test occs(C) == 2 .* occs(A)
    @test trainsize(C) == 3
    @test vocsize(C) == vocsize(A)
    @show trainsize(A), trainsize(B), trainsize(C)
end

@testset "Normalize and tokenize" begin
    textconfig = TextConfig(
        normalization=NormalizationConfig(del_punc=true, group_usr=true),
        tokenization=TokenizationConfig(nlist=[1, 2, 3], mark_token_type=false)
    )
    test_equals(tokenize(textconfig, text1),
                      ["hello", "world", "_usr", "#jello", "world", "hello world", "world _usr", "_usr #jello", "#jello world", "hello world _usr", "world _usr #jello", "_usr #jello world"]
                     )
end

@testset "Normalize and tokenize bigrams and trigrams" begin
    textconfig = TextConfig(
        normalization=NormalizationConfig(del_punc=true, group_usr=true),
        tokenization=TokenizationConfig(nlist=[2, 3], mark_token_type=false)
    )
    test_equals(
                    tokenize(textconfig, text1),
                      ["hello world", "world _usr", "_usr #jello", "#jello world", "hello world _usr", "world _usr #jello", "_usr #jello world"]
                     )
end

@testset "Normalize and tokenize" begin
    textconfig = TextConfig(
        normalization=NormalizationConfig(del_punc=false, group_usr=true),
        tokenization=TokenizationConfig(nlist=[1], mark_token_type=false)
    )
    text3 = "a ab __b @@c ..!d ''e \"!\"f +10 -20 30 40.00 .50 6.0 7.. ======= !()[]{}"
     test_equals(tokenize(textconfig, text3),
                      ["a", "ab", "__b", "@_usr", "..!", "d", "''", "e", "\"!\"", "f", "0", "0", "0", "0", "0", "0", "0", ".", "=======", "!()", "[]{", "}"]
                     )
end

@testset "ChainTransformation actually chains" begin
    ct = ChainTransformation([IgnoreStopwords(Set(["the"])), IgnoreStopwords(Set(["a"]))])
    test_equals(tokenize(TextConfig(tokenization=TokenizationConfig(nlist=[1]), transformation=ct), "the cat sat on a mat"), ["cat", "sat", "on", "mat"])
end

@testset "custom AbstractTokenGenerator extends tokenize_ without editing TextConfig" begin
    struct FirstCharGenerator <: AbstractTokenGenerator end
    TextSearch.needs_unigrams(::FirstCharGenerator) = true
    TextSearch.tokentag(::FirstCharGenerator) = 'i'

    function TextSearch.generate!(gen::FirstCharGenerator, buff, tt, mark_token_type)
        isempty(buff.unigrams) && return nothing
        write(buff.io, first(buff.unigrams[1]))
        TextSearch.flush_token!(buff, tt, gen, mark_token_type)
    end

    cfg = TextConfig(tokenization=TokenizationConfig(nlist=[1], generators=[FirstCharGenerator()]))
    test_equals(tokenize(cfg, "cat sat"), ["cat", "sat", "c\ti"])

    # an existing AbstractTokenTransformation (only overriding the legacy per-kind hooks)
    # keeps working unmodified against the new custom generator (defaults to identity);
    # "cat" is filtered from buff.unigrams too, since it feeds from the post-transform
    # token list, so the surviving first unigram is "sat".
    cfg2 = TextConfig(tokenization=TokenizationConfig(nlist=[1], generators=[FirstCharGenerator()]), transformation=IgnoreStopwords(Set(["cat"])))
    test_equals(tokenize(cfg2, "cat sat"), ["sat", "s\ti"])
end

@testset "overridable regex/emoji set" begin
    cfg = TextConfig(normalization=NormalizationConfig(group_url=true, re_url=r"myurl"), tokenization=TokenizationConfig(nlist=[1]))
    test_equals(tokenize(cfg, "visit myurl now"), ["visit", "_url", "now"])

    cfg2 = TextConfig(normalization=NormalizationConfig(group_emo=true, emojis=Set(['x'])), tokenization=TokenizationConfig(nlist=[1]))
    test_equals(tokenize(cfg2, "a x b"), ["a", "👾", "b"])
end

@testset "nested partial config updates" begin
    base = TextConfig(tokenization=TokenizationConfig(nlist=[1]))
    updated = TextConfig(base; tokenization=TokenizationConfig(base.tokenization; nlist=[1, 2]))
    test_equals(tokenize(updated, "cat sat"), ["cat", "sat", "cat sat\tn"])
    @test base.tokenization.nlist == Int8[1]
end

@testset "paragraph and sentence tokenizers" begin
    doc = "First paragraph line 1.\nFirst paragraph line 2.\n\nSecond paragraph here.\n\nThird paragraph."
    paragraphs = tokenize_paragraphs(doc)
    @test length(paragraphs) == 3
    @test paragraphs[1] == "First paragraph line 1.\nFirst paragraph line 2."
    @test paragraphs[2] == "Second paragraph here."
    @test paragraphs[3] == "Third paragraph."

    sentences = tokenize_sentences(doc)
    @test length(sentences) == 4
    @test sentences[1] == "First paragraph line 1."
    @test sentences[2] == "First paragraph line 2."
    @test sentences[3] == "Second paragraph here."
    @test sentences[4] == "Third paragraph."

    cfg = TextConfig(normalization=NormalizationConfig(lc=true, del_punc=true))
    norm_paragraphs = tokenize_paragraphs(cfg, doc)
    @test length(norm_paragraphs) == 3
    @test norm_paragraphs[2] == "second paragraph here"

    norm_sentences = tokenize_sentences(cfg, doc)
    @test length(norm_sentences) == 4
    @test norm_sentences[3] == "second paragraph here"
end

