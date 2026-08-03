
@testset "Dict-based vectors" begin
    cmpex(u, v) = abs(u[1] - v[1]) < 1e-3 && u[2] == v[2]

    aL = []
    AL = []
    for i in 1:10
        A = rand(300)
        B = rand(300)
        a = SVEC(k => v for (k, v) in enumerate(A))
        b = SVEC(k => v for (k, v) in enumerate(B))

        @test abs(norm(A) - norm(a)) < 1e-3
        @test abs(norm(B) - norm(b)) < 1e-3
        normalize!(A); normalize!(a)
        normalize!(B); normalize!(b)
        @test abs(norm(a) - 1.0) < 1e-3
        @test abs(norm(b) - 1.0) < 1e-3

        @test abs(dot(a, b) - dot(A, B)) < 1e-3
        @test abs(maximum(a) - maximum(A)) < 1e-3
        @test abs(minimum(a) - minimum(A)) < 1e-3

        
        @test cmpex(findmax(a), findmax(A))
        @test cmpex(findmin(a), findmin(A))

        push!(aL, a)
        push!(AL, A)
    end

    @test (norm(sum(AL)) - norm(sum(aL))) < 1e-3

    adist = Dist.Angle()
    cdist = Dist.Cosine()

    for i in 1:length(aL)-1
        @test abs(evaluate(adist, aL[i], aL[i+1]) - evaluate(adist, AL[i], AL[i+1])) < 1e-3
        @test abs(evaluate(cdist, aL[i], aL[i+1]) - evaluate(cdist, AL[i], AL[i+1])) < 1e-3
    end
end

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
    m = TextConfig(nlist=[1])
    test_equals(tokenize(m, text0), ["@user", ";)", "#jello", ".", "world"])

    m = TextConfig(nlist=[2])
    test_equals(tokenize(m, text0), ["@user ;)\tn", ";) #jello\tn", "#jello .\tn", ". world\tn"])

    m = TextConfig(nlist=[3])
    test_equals(tokenize(m, text0), ["@user ;) #jello\tn", ";) #jello .\tn", "#jello . world\tn"])

    m = TextConfig(qlist=[3])
    test_equals(tokenize(m, text0), map(p -> p*"\tq", [" @u", "@us", "use", "ser", "er;", "r;)", ";) ", ") #", " #j", "#je", "jel", "ell", "llo", "lo.", "o.w", ".wo", "wor", "orl", "rld", "ld "]))
    
    m = TextConfig(nlist=[1])
    test_equals(tokenize(m, text1), ["hello", "world", "!!", "@user", ";)", "#jello", ".", "world", ":)"])

    m = TextConfig(slist=[Skipgram(2,1)])
    test_equals(tokenize(m, text1), map(p -> "$p\ts", ["hello !!", "world @user", "!! ;)", "@user #jello", ";) .", "#jello world", ". :)"]))

end

@testset "message vectors" begin
    m = TextConfig(nlist=[1, 2])
    A = tokenize(m, "hello ;) #jello world.")
    B = tokenize(m, ["hello ;)", "#jello world."])
    push!(B, ";) #jello\tn")
    test_equals(sort(A.tokens), sort(B.tokens))
    # @show sort(A) sort(B)
end

@testset "vocabulary of different kinds of docs" begin
    textconfig = TextConfig(nlist=[1])
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
    textconfig = TextConfig(del_punc=true, group_usr=true, nlist=[1, 2, 3], mark_token_type=false)
    test_equals(tokenize(textconfig, text1), 
                      ["hello", "world", "_usr", "#jello", "world", "hello world", "world _usr", "_usr #jello", "#jello world", "hello world _usr", "world _usr #jello", "_usr #jello world"]
                     )
end

@testset "Normalize and tokenize bigrams and trigrams" begin
    textconfig = TextConfig(del_punc=true, group_usr=true, nlist=[2, 3], mark_token_type=false)
    test_equals(
                    tokenize(textconfig, text1),
                      ["hello world", "world _usr", "_usr #jello", "#jello world", "hello world _usr", "world _usr #jello", "_usr #jello world"]
                     )
end

@testset "Normalize and tokenize" begin
    textconfig = TextConfig(del_punc=false, group_usr=true, nlist=[1], mark_token_type=false)
    text3 = "a ab __b @@c ..!d ''e \"!\"f +10 -20 30 40.00 .50 6.0 7.. ======= !()[]{}"
     test_equals(tokenize(textconfig, text3),
                      ["a", "ab", "__b", "@_usr", "..!", "d", "''", "e", "\"!\"", "f", "0", "0", "0", "0", "0", "0", "0", ".", "=======", "!()", "[]{", "}"]
                     )
end

@testset "Tokenize skipgrams" begin
    textconfig = TextConfig(del_punc=false, group_usr=false, slist=[Skipgram(3,1)])
    tokens = tokenize(textconfig, text1)
    @show text1 tokens
    test_equals(tokens,
                      ["hello !! ;)\ts", "world @user #jello\ts", "!! ;) .\ts", "@user #jello world\ts", ";) . :)\ts"]
                     )

    config = TextConfig(del_punc=false, group_usr=false, nlist=[], slist=[Skipgram(3,1), Skipgram(2, 1)], mark_token_type=false)
    tokens = tokenize(config, text1)
    @show text1 tokens
    test_equals(tokens,
                      ["hello !!", "world @user", "!! ;)", "@user #jello", ";) .", "#jello world", ". :)", "hello !! ;)", "world @user #jello", "!! ;) .", "@user #jello world", ";) . :)"]
                     )
end

@testset "ChainTransformation actually chains" begin
    ct = ChainTransformation([IgnoreStopwords(Set(["the"])), IgnoreStopwords(Set(["a"]))])
    test_equals(tokenize(TextConfig(nlist=[1], tt=ct), "the cat sat on a mat"), ["cat", "sat", "on", "mat"])
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

    cfg = TextConfig(nlist=[1], generators=[FirstCharGenerator()])
    test_equals(tokenize(cfg, "cat sat"), ["cat", "sat", "c\ti"])

    # an existing AbstractTokenTransformation (only overriding the legacy per-kind hooks)
    # keeps working unmodified against the new custom generator (defaults to identity);
    # "cat" is filtered from buff.unigrams too, since it feeds from the post-transform
    # token list, so the surviving first unigram is "sat".
    cfg2 = TextConfig(nlist=[1], generators=[FirstCharGenerator()], tt=IgnoreStopwords(Set(["cat"])))
    test_equals(tokenize(cfg2, "cat sat"), ["sat", "s\ti"])
end

@testset "overridable regex/emoji set" begin
    cfg = TextConfig(nlist=[1], group_url=true, re_url=r"myurl")
    test_equals(tokenize(cfg, "visit myurl now"), ["visit", "_url", "now"])

    cfg2 = TextConfig(nlist=[1], group_emo=true, emojis=Set(['x']))
    test_equals(tokenize(cfg2, "a x b"), ["a", "👾", "b"])
end

