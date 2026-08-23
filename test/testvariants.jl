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

    @testset "resolve_query_tokens: strict treats bridging as a fallback" begin
        d = ["Sol brilla", "el sol calienta", "practico deporte", "practicó ayer",
             "Madrid capital", "USA grande", "León ciudad", "el león"]
        cfg = mkcfg(lc=false, diac=false)
        voc = Vocabulary(cfg, d; verbose=false)
        v = derive_variants(voc; min_ndocs=1)
        res(q; kw...) = resolve_query_tokens(voc, collect(tokenize(cfg, q)), v; kw...)

        # in the vocabulary: used as typed, nothing added -- writing carefully is not penalized
        r = res("sol")
        @test r.tokens == ["sol"]
        @test r.resolved[1].ndocs > 0 && isempty(r.resolved[1].added)
        @test r.resolved[1].dominant == "sol"       # its own group, so it carries the expansion
        @test isempty(explain(r))

        # out of vocabulary, bridged by a COMPUTED spelling: nothing is stored for this
        r = res("madrid")
        @test r.tokens == ["madrid", "Madrid"]   # the typed form is never dropped
        @test r.resolved[1].added == ["Madrid" => :derived]
        @test !haskey(v, "madrid")
        @test occursin("not found", only(explain(r)))

        # out of vocabulary, bridged by the stored map
        r = res("leon")
        @test Set(r.tokens) == Set(["leon", "león", "León"])
        @test all(p -> last(p) === :variant, r.resolved[1].added)

        # unbridgeable: passed through, matching nothing, exactly as before
        r = res("inexistente")
        @test r.tokens == ["inexistente"]
        @test isempty(r.resolved[1].dominant)    # nothing in the vocabulary, so nothing expands
    end

    @testset "the ratio rule: presence is not enough, and a bridge does not drag in noise" begin
        # 60 documents write `música`, one writes `musica` -- the shape measured on Spanish
        # Wikipedia, where `musica` (9 documents, Italian-language paragraphs) sat beside
        # `música` (4,404) and `:strict` stopped at the former, returning nothing.
        corpus = vcat(fill("la música suena", 60), ["musica italiana"],
                      fill("el sol calienta", 60), ["Sol brilla"], ["SOL memoria"])
        cfg = mkcfg(lc=false, diac=false)
        voc = Vocabulary(cfg, corpus; verbose=false)
        v = derive_variants(voc; min_ndocs=1)
        @test v == Dict("musica" => ["música"])

        # present but negligible: strict bridges anyway, and says how many documents decided it
        r = resolve_query_tokens(voc, ["musica"], v)
        @test r.tokens == ["musica", "música"]      # typed form kept: this only ever adds
        @test r.resolved[1].rare
        @test r.resolved[1].dominant == "música"
        @test occursin("only 1 document", only(explain(r)))
        # the ratio is what decides it, so disabling it restores stop-at-step-1
        @test resolve_query_tokens(voc, ["musica"], v; negligible_ratio=Inf).tokens == ["musica"]

        # and the other direction: a bridge does not reach a spelling the corpus barely has.
        # `SOL` (1 document) is what gave `digitalizada máx chip flash SDRAM` in the real profile
        @test resolve_query_tokens(voc, ["sol"], v; policy=:aggressive).tokens == ["sol"]
        @test resolve_query_tokens(voc, ["sol"], v; policy=:aggressive,
                                   negligible_ratio=Inf).tokens == ["sol", "Sol", "SOL"]

        @test_throws ArgumentError resolve_query_tokens(voc, ["sol"], v; negligible_ratio=0)
    end

    @testset "expansion_sources: one spelling per typed token, the commonest" begin
        corpus = vcat(fill("la música suena", 60), ["musica italiana"], fill("el sol brilla", 4))
        cfg = mkcfg(lc=false, diac=false)
        voc = Vocabulary(cfg, corpus; verbose=false)
        v = derive_variants(voc; min_ndocs=1)

        # bridged: the neighbours come from `música`, not from the 1-document `musica` whose
        # list would be built out of a single Italian paragraph
        @test expansion_sources(resolve_query_tokens(voc, ["musica"], v)) == ["música"]
        # unbridged: exactly the token list, so a well-typed query expands as it always did
        r = resolve_query_tokens(voc, ["sol", "brilla"], v)
        @test r.tokens == ["sol", "brilla"]
        @test expansion_sources(r) == ["sol", "brilla"]
        # nothing in the vocabulary contributes nothing to expand
        @test isempty(expansion_sources(resolve_query_tokens(voc, ["inexistente"], v)))
    end

    @testset "aggressive reaches what strict cannot" begin
        d = ["Sol brilla", "el sol calienta", "practico deporte", "practicó ayer"]
        cfg = mkcfg(lc=false, diac=false)
        voc = Vocabulary(cfg, d; verbose=false)
        v = derive_variants(voc; min_ndocs=1)
        res(q, pol) = resolve_query_tokens(voc, collect(tokenize(cfg, q)), v; policy=pol)

        # a folded form that is ITSELF a vocabulary token stops at step 1 under :strict, so its
        # alternatives are unreachable; :aggressive is the only way to them
        @test res("practico", :strict).tokens == ["practico"]
        @test res("practico", :aggressive).tokens == ["practico", "practicó"]
        @test res("sol", :strict).tokens == ["sol"]
        @test res("sol", :aggressive).tokens == ["sol", "Sol"]   # via the computed path, unstored

        # and the report distinguishes a correction from an enrichment
        @test occursin("also searched as", only(explain(res("sol", :aggressive))))
        # a token that bridges to nothing produces no line at all, in either policy
        @test isempty(explain(res("Solx", :strict)))
        @test isempty(explain(res("Solx", :aggressive)))

        @test_throws ArgumentError res("sol", :nonsense)
    end
end
