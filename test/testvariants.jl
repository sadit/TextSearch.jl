using Test, TextSearch, SimilaritySearch

@testset "derive_variants" begin
    mkcfg(; lc, diac) = TextConfig(
        normalization=NormalizationConfig(lc=lc, del_diac=diac, del_punc=true),
        tokenization=TokenizationConfig(nlist=[1]))

    # `Leon` is written capitalized and accented; the folded spellings a person types are absent
    docs = ["León es una ciudad", "el león rugió", "León y Castilla", "un león viejo"]

    # `León` is written capitalized and accented; `rugió` is accented
    docs = ["León es una ciudad", "el león rugió", "León y Castilla", "un león viejo"]

    @testset "only the non-derivable spellings are stored" begin
        voc = Vocabulary(mkcfg(lc=false, diac=false), docs; verbose=false)
        v = derive_variants(voc; min_ndocs=1)
        # accents cannot be computed from the folded form, so they are stored
        @test Set(v["leon"]) == Set(["león", "León"])
        @test v["rugio"] == ["rugió"]
        # capitalization CAN be computed, so `Castilla` is not stored: `castilla` reaches it
        # through `_derivable_forms` at query time
        @test !haskey(v, "castilla")
        @test !haskey(v, "ciudad")            # folds to itself
    end

    @testset "the min_ndocs floor keeps the map to plausible query targets" begin
        voc = Vocabulary(mkcfg(lc=false, diac=false), docs; verbose=false)
        @test haskey(derive_variants(voc; min_ndocs=1), "rugio")
        @test !haskey(derive_variants(voc; min_ndocs=2), "rugio")   # only one document
        @test haskey(derive_variants(voc; min_ndocs=2), "leon")     # several
    end

    @testset "what the profile already folded produces nothing" begin
        @test isempty(derive_variants(Vocabulary(mkcfg(lc=true, diac=true), docs; verbose=false);
                                      min_ndocs=1))
    end

    @testset "resolve_query_tokens: :auto corrects only where the evidence says so" begin
        d = ["Sol brilla", "el sol calienta", "practico deporte", "practicó ayer",
             "Madrid capital", "USA grande", "León ciudad", "el león"]
        cfg = mkcfg(lc=false, diac=false)
        voc = Vocabulary(cfg, d; verbose=false)
        v = derive_variants(voc; min_ndocs=1)
        res(q, pol=QueryPolicy()) = resolve_query_tokens(voc, collect(tokenize(cfg, q)), v, pol)

        # in the vocabulary and not negligible: used as typed, nothing added -- writing carefully
        # is not penalized
        r = res("sol")
        @test r.tokens == ["sol"]
        @test r.resolved[1].kept && isempty(r.resolved[1].added)
        @test r.resolved[1].dominant == "sol"       # its own group, so it carries the expansion
        @test isempty(explain(r))

        # not in the vocabulary: corrected, so the typed form is REPLACED rather than kept
        r = res("madrid")
        @test r.tokens == ["Madrid"]
        @test !r.resolved[1].kept
        @test r.resolved[1].added == ["Madrid" => :derived]
        @test !haskey(v, "madrid")                  # computed, nothing was stored for it
        @test occursin("not found", only(explain(r)))
        @test occursin("instead", only(explain(r)))

        # corrected through the stored map, and to every spelling that clears the floor
        r = res("leon")
        @test Set(r.tokens) == Set(["león", "León"])
        @test all(p -> last(p) === :variant, r.resolved[1].added)

        # unbridgeable: passed through, matching nothing, exactly as before
        r = res("inexistente")
        @test r.tokens == ["inexistente"]
        @test r.resolved[1].kept
        @test isempty(r.resolved[1].dominant)    # nothing in the vocabulary, so nothing expands

        # :off is the "search instead for ..." escape: no bridging at all, whatever the evidence
        r = res("madrid", QueryPolicy(correction=:off))
        @test r.tokens == ["madrid"]
        @test r.resolved[1].kept && isempty(explain(r))
    end

    @testset "the ratio rule: presence is not evidence, and a bridge does not drag in noise" begin
        # 60 documents write `música`, one writes `musica` -- the shape measured on Spanish
        # Wikipedia, where `musica` (9 documents, Italian-language paragraphs) sat beside
        # `música` (4,404) and searching stopped at the former, returning nothing at all.
        corpus = vcat(fill("la música suena", 60), ["musica italiana"],
                      fill("el sol calienta", 60), ["Sol brilla"], ["SOL memoria"])
        cfg = mkcfg(lc=false, diac=false)
        voc = Vocabulary(cfg, corpus; verbose=false)
        v = derive_variants(voc; min_ndocs=1)
        @test v == Dict("musica" => ["música"])
        res(q, pol=QueryPolicy()) = resolve_query_tokens(voc, [q], v, pol)

        # present but negligible: corrected, and the report says how many documents decided it
        r = res("musica")
        @test r.tokens == ["música"]
        @test !r.resolved[1].kept
        @test r.resolved[1].dominant == "música"
        @test occursin("only 1 document", only(explain(r)))
        # the ratio is what decides it, so lifting it leaves the typed form alone
        @test res("musica", QueryPolicy(negligible_ratio=Inf)).tokens == ["musica"]
        # ...and :off reaches the same place by intent rather than by evidence
        @test res("musica", QueryPolicy(correction=:off)).tokens == ["musica"]

        # the other direction: a bridge does not reach a spelling the corpus barely has.
        # `SOL` (1 document) is what gave `digitalizada máx chip flash SDRAM` in the real profile
        @test res("sol", QueryPolicy(correction=:always)).tokens == ["sol"]
        @test res("sol", QueryPolicy(correction=:always,
                                     negligible_ratio=Inf)).tokens == ["sol", "Sol", "SOL"]
    end

    @testset "expansion_sources: one spelling per typed token, the commonest" begin
        corpus = vcat(fill("la música suena", 60), ["musica italiana"], fill("el sol brilla", 4))
        cfg = mkcfg(lc=false, diac=false)
        voc = Vocabulary(cfg, corpus; verbose=false)
        v = derive_variants(voc; min_ndocs=1)

        # corrected: the neighbours come from `música`, not from the 1-document `musica` whose
        # list would be built out of a single Italian paragraph
        @test expansion_sources(resolve_query_tokens(voc, ["musica"], v)) == ["música"]
        # untouched: exactly the token list, so a well-typed query expands as it always did
        r = resolve_query_tokens(voc, ["sol", "brilla"], v)
        @test r.tokens == ["sol", "brilla"]
        @test expansion_sources(r) == ["sol", "brilla"]
        # and with correction off, the literal query gets the literal query's neighbours
        @test expansion_sources(resolve_query_tokens(voc, ["musica"], v,
                                    QueryPolicy(correction=:off))) == ["musica"]
        # nothing in the vocabulary contributes nothing to expand
        @test isempty(expansion_sources(resolve_query_tokens(voc, ["inexistente"], v)))
    end

    @testset ":always reaches what :auto has no reason to" begin
        d = ["Sol brilla", "el sol calienta", "practico deporte", "practicó ayer"]
        cfg = mkcfg(lc=false, diac=false)
        voc = Vocabulary(cfg, d; verbose=false)
        v = derive_variants(voc; min_ndocs=1)
        res(q, pol) = resolve_query_tokens(voc, collect(tokenize(cfg, q)), v, pol)

        # a spelling that is itself a healthy vocabulary token gives :auto no evidence to act on,
        # so its alternatives are unreachable; :always is the only way to them, and because
        # nothing says the typed form is wrong it is kept rather than replaced
        @test res("practico", QueryPolicy()).tokens == ["practico"]
        @test res("practico", QueryPolicy(correction=:always)).tokens == ["practico", "practicó"]
        @test res("sol", QueryPolicy()).tokens == ["sol"]
        @test res("sol", QueryPolicy(correction=:always)).tokens == ["sol", "Sol"]
        @test res("sol", QueryPolicy(correction=:always)).resolved[1].kept

        # the report distinguishes an enrichment from a correction
        @test occursin("also searched as",
                       only(explain(res("sol", QueryPolicy(correction=:always)))))
        # a token that bridges to nothing produces no line at all, in any mode
        for c in (:off, :auto, :always)
            @test isempty(explain(res("Solx", QueryPolicy(correction=c))))
        end
    end

    @testset "QueryPolicy validates what it is given" begin
        @test_throws ArgumentError QueryPolicy(correction=:nonsense)
        @test_throws ArgumentError QueryPolicy(negligible_ratio=0.5)
        @test_throws ArgumentError QueryPolicy(expansion_k=-1)
        @test QueryPolicy().correction === :auto
        @test QueryPolicy().expansion
        @test occursin("correction=:auto", string(QueryPolicy()))
    end
end
