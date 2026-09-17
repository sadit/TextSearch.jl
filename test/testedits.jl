using Test, TextSearch, SimilaritySearch

@testset "edit-distance correction" begin
    mkcfg(; lc=false, diac=false) = TextConfig(
        normalization=NormalizationConfig(lc=lc, del_diac=diac, del_punc=true),
        tokenization=TokenizationConfig(nlist=[1]))

    @testset "derive_edits indexes what it is told to" begin
        cfg = mkcfg()
        docs = vcat(fill("la guerra civil", 5), ["un enemigo solitario"])
        voc = Vocabulary(cfg, docs; verbose=false)

        ei = derive_edits(voc)
        @test length(ei) == vocsize(voc)
        @test occursin("EditIndex", string(ei))

        # min_ndocs is a floor on what may be corrected TO
        few = derive_edits(voc; min_ndocs=5)
        @test length(few) < vocsize(voc)
        @test isempty(edit_candidates(few, "enemig"))        # `enemigo` has one document

        @test_throws ArgumentError derive_edits(voc; minlength=0)
    end

    @testset "edit_candidates returns the distance-1 neighbourhood, ascending and complete" begin
        cfg = mkcfg()
        # `casa`/`cosa`/`caza` are mutually one substitution apart
        voc = Vocabulary(cfg, ["la casa", "la cosa", "la caza", "el arbol"]; verbose=false)
        ei = derive_edits(voc; minlength=1)

        c = edit_candidates(ei, "cesa")
        @test Set(gettoken(voc, i) for i in c) == Set(["casa", "cosa"])   # not `caza`: two edits
        @test issorted(c)                                                  # deterministic

        # a transposition is one edit, which is the whole point of Damerau over Levenshtein
        @test Set(gettoken(voc, i) for i in edit_candidates(ei, "acsa")) == Set(["casa"])

        # nothing within one edit
        @test isempty(edit_candidates(ei, "zzzzzz"))
    end

    @testset "minlength is a cost gate, not a quality one" begin
        # It exists to skip the band where a lookup is nearly always wasted -- at length 1 every
        # single-character token is one substitution from every other. It is NOT what makes the
        # correction safe; uniqueness is (see the table in `edit_candidates`).
        cfg = mkcfg()
        voc = Vocabulary(cfg, ["la casa", "la cosa"]; verbose=false)
        @test isempty(edit_candidates(derive_edits(voc; minlength=6), "cesa"))
        @test !isempty(edit_candidates(derive_edits(voc; minlength=4), "cesa"))
    end

    @testset "multibyte tokens do not throw" begin
        # `Dist.Seqs.*` index positionally, which on a String means BYTE offsets and throws
        # StringIndexError on any multi-byte character. Tokens are indexed as Char vectors for
        # exactly this reason, and a real Spanish vocabulary is full of them.
        cfg = mkcfg()
        voc = Vocabulary(cfg, ["la canción alemana", "el corazón roto", "una canción"]; verbose=false)
        ei = derive_edits(voc; minlength=1)
        @test Set(gettoken(voc, i) for i in edit_candidates(ei, "cancion")) == Set(["canción"])
        @test Set(gettoken(voc, i) for i in edit_candidates(ei, "corazóon")) == Set(["corazón"])
    end

    @testset "resolve_query_tokens: a unique neighbour corrects, an ambiguous one does not" begin
        cfg = mkcfg()
        docs = vcat(fill("la guerra civil", 10), fill("la casa roja", 10),
                    fill("la cosa rara", 10), fill("la caza mayor", 10))
        voc = Vocabulary(cfg, docs; verbose=false)
        ei = derive_edits(voc; minlength=1)
        res(q; kwargs...) = resolve_query_tokens(voc, [q], nothing, QueryPolicy(); kwargs...)

        # one candidate: corrected, and the typed form is REPLACED, as every correction is
        r = res("guerar"; edits=ei)
        @test r.tokens == ["guerra"]
        @test !r.resolved[1].kept
        @test r.resolved[1].added == ["guerra" => :edit]
        @test occursin("not found", only(explain(r)))

        # three candidates one edit away: declined outright, rather than picking the commonest.
        # This is the rule that buys 0.999 precision against 0.868 for "most frequent candidate".
        r = res("caaa"; edits=ei)
        @test r.tokens == ["caaa"]
        @test r.resolved[1].kept && isempty(r.resolved[1].added)

        # without an index nothing happens at all, which is how a consumer turns this off
        @test res("guerar").tokens == ["guerar"]

        # ...and so does :off, by intent rather than by absence
        @test resolve_query_tokens(voc, ["guerar"], nothing,
                                   QueryPolicy(correction=:off); edits=ei).tokens == ["guerar"]
    end

    @testset "a token the vocabulary holds is never guessed at" begin
        # Unlike an accent, where presence is weak evidence (`musica` at 9 documents against
        # `música`'s 4,404), an arbitrary edit is well evidenced against by the corpus simply
        # holding the word: `casa` is a word, so it is not a typo for `cosa`.
        cfg = mkcfg()
        voc = Vocabulary(cfg, vcat(["la casa"], fill("la cosa", 50)); verbose=false)
        ei = derive_edits(voc; minlength=1)
        for correction in (:auto, :always)
            r = resolve_query_tokens(voc, ["casa"], nothing,
                                     QueryPolicy(; correction); edits=ei)
            @test "casa" in r.tokens
            @test !any(p -> last(p) === :edit, r.resolved[1].added)
        end
    end

    @testset "the folds are tried first, and a guess only fills what they leave empty" begin
        cfg = mkcfg()
        # `leon` reaches `León` through the stored map (an accent cannot be computed from the
        # folded form) -- and it is ALSO one edit from `leona`, a different, real word. The
        # certain claim has to win without the guess being consulted, which is what makes the
        # ordering in `_candidate_group` load-bearing rather than cosmetic.
        docs = vcat(fill("León ciudad", 10), fill("el leona grande", 10))
        voc = Vocabulary(cfg, docs; verbose=false)
        v = derive_variants(voc; min_ndocs=1)
        ei = derive_edits(voc; minlength=1)
        @test gettoken(voc, only(edit_candidates(ei, "leon"))) == "leona"   # the guess on offer

        r = resolve_query_tokens(voc, ["leon"], v, QueryPolicy(); edits=ei)
        @test r.resolved[1].added == ["León" => :variant]
        @test !any(p -> last(p) === :edit, r.resolved[1].added)

        # the same for a fold that IS computable: `madrid` reaches `Madrid` with nothing stored
        voc2 = Vocabulary(cfg, vcat(fill("Madrid capital", 10), fill("madrig raro", 10)); verbose=false)
        r2 = resolve_query_tokens(voc2, ["madrid"], nothing, QueryPolicy();
                                  edits=derive_edits(voc2; minlength=1))
        @test r2.resolved[1].added == ["Madrid" => :derived]
    end

    @testset "QueryPipeline carries it, and query_tokens reports it" begin
        cfg = mkcfg()
        docs = vcat(fill("la guerra civil", 10), fill("el jardin azul", 10))
        voc = Vocabulary(cfg, docs; verbose=false)
        model = VectorModel(IdfWeighting(), TfWeighting(), voc)
        ei = derive_edits(voc; minlength=1)

        qp = QueryPipeline(edits=ei)
        @test occursin("edits=", string(qp))

        q = query_tokens(voc, "guerar", qp)
        @test [t.token for t in q.terms] == ["guerra"]
        @test only(q.terms).reason === :edit
        # a correction is the word the person meant, so it enters at full strength
        @test only(q.terms).factor == 1f0

        # it reaches the representations, which is the point: the typed form scored nothing
        @test !isempty(querybow(voc, q))
        @test nnz(queryvector(model, q)) == 1
        @test querytokenset(q) == Set(["guerra"])

        # an uncorrected OOV token still produces an empty representation, as before
        q0 = query_tokens(voc, "guerar", QueryPipeline())
        @test isempty(querybow(voc, q0))
    end
end
