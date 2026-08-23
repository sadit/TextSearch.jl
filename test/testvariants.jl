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
        @test r.resolved[1].invocabulary && isempty(r.resolved[1].added)
        @test isempty(explain(r))

        # out of vocabulary, bridged by a COMPUTED spelling: nothing is stored for this
        r = res("madrid")
        @test r.tokens == ["madrid", "Madrid"]
        @test r.resolved[1].added == ["Madrid" => :derived]
        @test !haskey(v, "madrid")
        @test occursin("not found", only(explain(r)))

        # out of vocabulary, bridged by the stored map
        r = res("leon")
        @test Set(r.tokens) == Set(["leon", "león", "León"])
        @test all(p -> last(p) === :variant, r.resolved[1].added)

        # unbridgeable: passed through, matching nothing, exactly as before
        @test res("inexistente").tokens == ["inexistente"]
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
