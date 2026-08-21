using Test, TextSearch, SimilaritySearch

# These are library tests on purpose: a refit is an operation of TextSearch itself, so the
# whole thing must be exercisable without the CLI app.
# Digs the lemma mapping out of a transformation pipeline, so a test can compare what a
# profile APPLIES against what it saves.
_find_lemma_map(tt::LemmaTransformation) = tt.lemmas
_find_lemma_map(::AbstractTokenTransformation) = nothing
function _find_lemma_map(tt::ChainTransformation)
    for s in tt.list
        m = _find_lemma_map(s)
        m === nothing || return m
    end
    nothing
end

@testset "refit_profile" begin
    tc = TextConfig(tokenization=TokenizationConfig(nlist=[1]))

    # A "base" standing in for a large generic corpus: "wikipedia" is everywhere, "gato" is
    # solidly attested, "abstruso" is attested but rare, and "typoxyz" appears once.
    basedocs = String[]
    for i in 1:100
        push!(basedocs, "wikipedia articulo numero $i sobre algo")
    end
    for i in 1:40
        basedocs[i] *= " gato"
    end
    for i in 1:5
        basedocs[i] *= " abstruso"
    end
    basedocs[1] *= " typoxyz"
    # Enough distinct base tokens that the blended vocabulary ends up LARGER than the corpus a
    # pinned avgdoclen describes. Without that, the fixture cannot catch a `max(vocsize, ...)`
    # floor swallowing the avgdoclen override -- which is exactly what happened: the floor read
    # as a sane "one occurrence per token" bound and silently made the knob a no-op on any real
    # base. Each token lands in ~11 of the 100 documents, enough that the default kappa carries
    # it rather than rounding it away. The names are alphabetic on purpose: a numeric suffix
    # would be collapsed by group_num into a single token.
    xname(k) = "tok" * string(Char('a' + div(k, 26))) * string(Char('a' + mod(k, 26)))
    for i in 1:100, j in 1:6
        basedocs[i] *= " " * xname(mod(i * 6 + j, 54))
    end

    function mkprofile(docs; textconfig=tc, kwargs...)
        voc = Vocabulary(textconfig, docs; verbose=false)
        model = VectorModel(IdfWeighting(), TfWeighting(), voc)
        dir = tempname()
        save_profile(dir, model; kwargs...)
        p = load_profile(dir)
        rm(dir; recursive=true, force=true)
        p
    end

    base = mkprofile(basedocs)
    bvoc = base.model.voc

    # the sample is a different dataset: it talks about perros, never about gatos
    sampledocs = ["el perro ladra", "otro perro corre", "el perro y el perro",
                  "un perro mas", "perro perro perro"]

    @testset "the sample dominates, and the base is not thrown away" begin
        r = refit_profile(base, sampledocs; verbose=false)
        voc = r.model.voc

        # sample tokens are present, and so is the base's well-attested vocabulary
        @test token2id(voc, "perro") != 0
        @test token2id(voc, "wikipedia") != 0
        @test token2id(voc, "gato") != 0

        # trainsize is sample-sized plus the prior's weight in documents, NOT base-sized:
        # this is what makes a refitted profile lighter than the generic one
        @test trainsize(voc) == 2 * length(sampledocs)
        # the result is base-survivors plus the sample, so it is smaller than their union:
        # some of the base was pruned rather than carried
        svoc = Vocabulary(refit_textconfig(base), sampledocs; verbose=false)
        @test vocsize(voc) < vocsize(bvoc) + vocsize(svoc)

        # every count stays within its corpus: this is the guard against a negative idf and
        # a negative BM25 numerator, both of which a folded/blended ndocs could produce
        @test all(id -> ndocs(voc, id) <= trainsize(voc), eachindex(voc))
        @test all(w -> w >= 0, r.model.weight)

        # numtokens describes what actually shipped
        @test numtokens(voc) == sum(voc.occs)
        @test avgdoclen(voc) > 0

        # The result must describe a POSSIBLE corpus: a token present in n documents occurs
        # at least n times. This caught a real bug -- scaling occs by the base's share of
        # total tokens while scaling ndocs per document made the two round against different
        # denominators, so carried tokens landed with ndocs >= 1 and occs == 0 (and a
        # numtokens below the vocabulary size).
        @test all(id -> occs(voc, id) >= ndocs(voc, id), eachindex(voc))
        @test numtokens(voc) >= vocsize(voc)
    end

    @testset "the verbose report runs" begin
        # Every other test here passes verbose=false, which left the reporting branch
        # unexercised -- and it contained a MethodError: the `avgdoclen` keyword shadows the
        # function of the same name, so calling it bare threw only when verbose was on. The
        # CLI found it, not the library tests. Assert the branch executes and says something.
        for adl in (:blend, :sample)
            msg = mktemp() do path, io
                redirect_stderr(io) do
                    refit_profile(base, sampledocs; avgdoclen=adl, verbose=true)
                end
                flush(io)
                read(path, String)
            end
            @test occursin("kappa=", msg)
            @test occursin("avgdoclen=", msg)
        end
    end

    @testset "a base-important token absent from the sample survives, with LESS weight" begin
        # the core requirement: do not throw away important words the sample never shows,
        # but do reduce their importance
        r = refit_profile(base, sampledocs; verbose=false)
        voc = r.model.voc

        for tok in ("wikipedia", "gato")
            @test token2id(voc, tok) != 0
            rate_base = ndocs(bvoc, token2id(bvoc, tok)) / trainsize(bvoc)
            rate_refit = ndocs(voc, token2id(voc, tok)) / trainsize(voc)
            @test rate_refit < rate_base    # importance reduced, presence kept
        end
    end

    @testset "a base-unimportant token absent from the sample is dropped" begin
        r = refit_profile(base, sampledocs; keep_floor=3, verbose=false)
        # seen in exactly 1 of 100 base documents and never in the sample
        @test token2id(base.model.voc, "typoxyz") != 0
        @test token2id(r.model.voc, "typoxyz") == 0
    end

    @testset "kappa controls how much the base counts for" begin
        small = refit_profile(base, sampledocs; kappa=1, verbose=false)
        big = refit_profile(base, sampledocs; kappa=10_000, verbose=false)

        # with a huge prior the blended rates approach the base's; with a tiny one they do not
        rate(v, t) = ndocs(v, token2id(v, t)) / trainsize(v)
        base_rate = rate(bvoc, "wikipedia")
        @test abs(rate(big.model.voc, "wikipedia") - base_rate) <
              abs(rate(small.model.voc, "wikipedia") - base_rate)

        # a tiny prior cannot keep base-only tokens: they round to zero and fall out
        @test vocsize(small.model.voc) < vocsize(big.model.voc)
        @test trainsize(small.model.voc) < trainsize(big.model.voc)
        @test big.encoder.kappa == 10_000
    end

    # Vocabulary construction is threaded and merges per-thread partials, so token ORDER is
    # not part of its contract -- two identical corpora can yield the same counts in a
    # different order. Compare vocabularies by content.
    counts(v) = Dict(token(v, id) => (occs(v, id), ndocs(v, id)) for id in eachindex(v))
    weights(m) = Dict(token(m.voc, id) => m.weight[id] for id in eachindex(m.voc))

    @testset "layer 3 (caller-built Vocabulary) == layer 4 (corpus)" begin
        # the seam that makes a refit usable from any program: build the sample vocabulary
        # however you like, as long as it uses refit_textconfig
        rtc = refit_textconfig(base)
        svoc = Vocabulary(rtc, sampledocs; verbose=false)
        a = refit_profile(base, svoc; verbose=false)
        b = refit_profile(base, sampledocs; verbose=false)

        @test counts(a.model.voc) == counts(b.model.voc)
        @test weights(a.model) == weights(b.model)

        @testset "a sample accumulated incrementally matches one built in a pass" begin
            # this is the streaming / grow-over-time path
            v1 = Vocabulary(rtc, sampledocs[1:2]; verbose=false)
            v2 = Vocabulary(rtc, sampledocs[3:end]; verbose=false)
            acc = Vocabulary(rtc, Int64(trainsize(v1) + trainsize(v2)),
                             Int64(numtokens(v1) + numtokens(v2)))
            update_voc!(acc, v1)
            update_voc!(acc, v2)

            c = refit_profile(base, acc; verbose=false)
            @test counts(c.model.voc) == counts(a.model.voc)
        end
    end

    @testset "a sample built under the wrong TextConfig is rejected" begin
        # the failure this guards against is silent: mismatched tokens would be interpolated
        # against each other and the numbers would simply be wrong
        wrong = Vocabulary(TextConfig(tokenization=TokenizationConfig(nlist=[1, 2])),
                           sampledocs; verbose=false)
        @test_throws ErrorException refit_profile(base, wrong; verbose=false)

        wrongnorm = Vocabulary(TextConfig(tc; normalization=NormalizationConfig(lc=false)),
                               sampledocs; verbose=false)
        @test_throws ErrorException refit_profile(base, wrongnorm; verbose=false)
    end

    @testset "lemmas: carried by the base, applied by the refit" begin
        lemmas = Dict("perros" => "perro", "gatos" => "gato")
        lbase = mkprofile(basedocs; lemmas)
        @test !has_lemma_transformation(lbase.model.voc.textconfig.transformation)

        lsample = ["el perro ladra", "los perros ladran", "perros y perros",
                   "un perro mas", "perro perro perros"]

        @testset "apply_lemmas=true lemmatizes both sides" begin
            r = refit_profile(lbase, lsample; verbose=false)
            voc = r.model.voc
            @test has_lemma_transformation(voc.textconfig.transformation)
            @test r.encoder.lemmas_applied
            # the inflected form is gone; the lemma carries the family
            @test token2id(voc, "perros") == 0
            @test token2id(voc, "perro") != 0
            # and a query in the inflected form reaches it through the TextConfig alone
            @test collect(tokenize(voc.textconfig, "perros")) == ["perro"]
        end

        @testset "the applied map and the saved map are the same map" begin
            # The TextConfig starts out carrying the base's FULL map (it must, to tokenize
            # the sample before the vocabulary exists), but the prune then removes most of
            # its targets. Shipping the full map while saving and reporting the restricted
            # one means the profile applies something other than what it says it does -- and
            # on the real Wikipedia-es profile the dead entries were 47% of the output file.
            r = refit_profile(lbase, lsample; verbose=false)
            voc = r.model.voc
            applied = _find_lemma_map(voc.textconfig.transformation)

            @test applied == r.lemmas
            # nothing in the applied map can point at a token the vocabulary lacks, so no
            # entry is dead weight
            @test all(lemma -> token2id(voc, lemma) != 0, values(applied))
        end

        @testset "apply_lemmas=false leaves the map as an artifact only" begin
            r = refit_profile(lbase, lsample; apply_lemmas=false, verbose=false)
            @test !has_lemma_transformation(r.model.voc.textconfig.transformation)
            @test !r.encoder.lemmas_applied
            @test token2id(r.model.voc, "perros") != 0   # still its own token
            @test !isempty(r.lemmas)                      # still carried
        end
    end

    @testset "fold_lemmas: occs exact, ndocs capped" begin
        # two forms of one family, deliberately co-occurring in a document so the ndocs
        # overestimate is real and the cap has to catch it
        docs = ["casa casas", "casa", "casas", "otro"]
        voc = Vocabulary(tc, docs; verbose=false)
        occs_before = occs(voc, token2id(voc, "casa")) + occs(voc, token2id(voc, "casas"))

        f = fold_lemmas(voc, Dict("casas" => "casa"))
        @test f.folded == 1
        @test token2id(f.voc, "casas") == 0
        # occurrences are additive, so this is exact
        @test occs(f.voc, token2id(f.voc, "casa")) == occs_before
        # documents are not: 2 + 2 = 4 would exceed the 4-document corpus only if all four
        # matched, so assert the invariant that matters rather than a magic number
        @test all(id -> ndocs(f.voc, id) <= trainsize(f.voc), eachindex(f.voc))
        @test numtokens(f.voc) == numtokens(voc)

        @testset "a lemma missing from the vocabulary drops rather than resurrecting" begin
            # "casa" is deliberately not a token here, so folding onto it must not create it
            v2 = Vocabulary(tc, ["casas rojas", "casas"]; verbose=false)
            f2 = fold_lemmas(v2, Dict("casas" => "casa"))
            @test f2.dropped == 1
            @test token2id(f2.voc, "casa") == 0
            @test token2id(f2.voc, "casas") == 0
        end

        @testset "the cap fires when a family really oversubscribes" begin
            # every document contains both forms, so summing ndocs doubles the corpus size
            v3 = Vocabulary(tc, ["casa casas", "casa casas", "casa casas"]; verbose=false)
            f3 = fold_lemmas(v3, Dict("casas" => "casa"))
            @test f3.capped >= 1
            @test ndocs(f3.voc, token2id(f3.voc, "casa")) == trainsize(v3)
        end
    end

    @testset "extend_lemmas: families the base never saw" begin
        # the base knows "audifono" but has never seen its plural; the sample uses both
        extdocs = copy(basedocs)
        for i in 1:30
            extdocs[i] *= " audifono"
        end
        ebase = mkprofile(extdocs; lemmas=Dict("gatos" => "gato"))
        esample = ["compre audifonos nuevos", "los audifonos suenan bien",
                   "audifonos y audifono", "un audifono roto", "audifonos otra vez"]

        @testset "without the extension they stay unmerged" begin
            r = refit_profile(ebase, esample; verbose=false)
            voc = r.model.voc
            # two tokens, so the family's document frequency is split across forms
            @test token2id(voc, "audifono") != 0
            @test token2id(voc, "audifonos") != 0
            @test !haskey(r.lemmas, "audifonos")
        end

        @testset "with it, the new form elects the base's established one" begin
            r = refit_profile(ebase, esample; extend_lemmas=true, verbose=false)
            voc = r.model.voc
            # the base form wins the election because the merged counts favour it, which is
            # the point of grouping over base+sample rather than the sample alone
            @test r.lemmas["audifonos"] == "audifono"
            @test token2id(voc, "audifonos") == 0
            @test token2id(voc, "audifono") != 0
            # and the family's counts are now together
            @test ndocs(voc, token2id(voc, "audifono")) >=
                  ndocs(refit_profile(ebase, esample; verbose=false).model.voc,
                        token2id(refit_profile(ebase, esample; verbose=false).model.voc, "audifono"))
            # the base's own map is carried through untouched
            @test r.lemmas["gatos"] == "gato"
        end

        @testset "the base's own clustering decisions are not overruled" begin
            # "abstruso" and "algo" are base-only tokens; whatever the base decided about
            # them must stand, so no entry may appear for a token the sample never brought
            r = refit_profile(ebase, esample; extend_lemmas=true, verbose=false)
            newtokens = Set(["audifonos", "compre", "nuevos", "suenan", "bien", "roto",
                             "otra", "vez", "los", "y", "un"])
            for (tok, _) in r.lemmas
                haskey(ebase.lemmas, tok) || @test tok in newtokens
            end
        end

        @testset "extend_lemmas needs lemmas applied" begin
            # nothing to extend into: the map would not be in the pipeline at all
            r = refit_profile(ebase, esample; extend_lemmas=true, apply_lemmas=false, verbose=false)
            @test !has_lemma_transformation(r.model.voc.textconfig.transformation)
            @test !haskey(r.lemmas, "audifonos")
        end
    end

    @testset "extend_lemmas_morphological on its own" begin
        voc = Vocabulary(tc, ["gato gatos gatito", "perro perros", "casa"]; verbose=false)

        # no candidates: everything is considered
        all_ext = extend_lemmas_morphological(voc, Dict{String,String}())
        @test !isempty(all_ext)
        @test all(t -> token2id(voc, t) != 0, keys(all_ext))
        @test all(l -> token2id(voc, l) != 0, values(all_ext))
        # a lemma is never itself a key, so lookup terminates in one step
        @test !any(l -> haskey(all_ext, l), values(all_ext))

        # candidates restrict the OUTPUT, not just the cost
        only_perros = extend_lemmas_morphological(voc, Dict{String,String}(); candidates=["perros"])
        @test collect(keys(only_perros)) == ["perros"]

        # tokens already mapped are left alone rather than re-grouped or chained
        pre = Dict("gatos" => "gato")
        ext = extend_lemmas_morphological(voc, pre)
        @test !haskey(ext, "gatos")

        # morphology is the whole signal here, so :none is an error rather than a silent no-op
        @test_throws ErrorException extend_lemmas_morphological(voc, Dict{String,String}();
                                                               morphology=:none)
    end

    @testset "avgdoclen: blend vs pinned to the sample" begin
        # the sample's documents are much shorter than the base's, so the two disagree
        blended = refit_profile(base, sampledocs; verbose=false)
        pinned = refit_profile(base, sampledocs; avgdoclen=:sample, verbose=false)
        svoc = Vocabulary(refit_textconfig(base), sampledocs; verbose=false)

        # the base fixture is deliberately token-rich, so the pinned corpus has FEWER
        # occurrences than the vocabulary has tokens -- the case a floor would have masked
        @test numtokens(pinned.model.voc) < vocsize(pinned.model.voc)
        @test avgdoclen(blended.model.voc) > avgdoclen(pinned.model.voc)
        @test isapprox(avgdoclen(pinned.model.voc), avgdoclen(svoc); rtol=0.05)

        # only numtokens moves: the knob exists to steer BM25's length normalization, and
        # must not touch the counts the weights come from. Compared by content, since token
        # order is not part of Vocabulary's contract.
        @test counts(blended.model.voc) == counts(pinned.model.voc)
        @test weights(blended.model) == weights(pinned.model)

        @testset "an explicit average is accepted, nonsense is not" begin
            fixed = refit_profile(base, sampledocs; avgdoclen=12.5, verbose=false)
            @test isapprox(avgdoclen(fixed.model.voc), 12.5; rtol=0.05)
            @test_throws ArgumentError refit_profile(base, sampledocs; avgdoclen=0, verbose=false)
            @test_throws ArgumentError refit_profile(base, sampledocs; avgdoclen=:bogus, verbose=false)
        end
    end

    @testset "synonyms are inherited and restricted to survivors" begin
        syn = Dict("gato" => ["wikipedia", "typoxyz", "noexisteenvocab"])
        sdist = Dict("gato" => Float32[0.1, 0.2, 0.3])
        sbase = mkprofile(basedocs; synonyms=syn, synonym_distances=sdist)

        r = refit_profile(sbase, sampledocs; verbose=false)
        @test haskey(r.synonyms, "gato")
        # "typoxyz" was pruned and "noexisteenvocab" never existed: both must go, and the
        # distances must stay aligned with what remains
        @test r.synonyms["gato"] == ["wikipedia"]
        @test r.synonym_distances["gato"] == Float32[0.1]

        @testset "a network with no distances survives the restriction" begin
            nbase = mkprofile(basedocs; synonyms=syn)
            r2 = refit_profile(nbase, sampledocs; verbose=false)
            @test r2.synonyms["gato"] == ["wikipedia"]
            @test r2.synonym_distances === nothing
        end
    end

    @testset "the result is self-contained and saveable" begin
        r = refit_profile(base, sampledocs; verbose=false)
        dir = tempname()
        try
            save_profile(dir, r.model; r.synonyms, r.synonym_distances, r.lemmas,
                         r.stopword_candidates, r.encoder)
            q = load_profile(dir)
            @test q.model.voc.token == r.model.voc.token
            @test q.model.weight == r.model.weight
            @test q.encoder["kind"] == "refit"
            # nothing in the saved profile refers back to the base
            @test vectorize(q.model, "perro") == vectorize(r.model, "perro")
        finally
            rm(dir; force=true, recursive=true)
        end
    end

    @testset "EntropyWeighting is rejected" begin
        # its weights are supervised, so they cannot be re-derived from a profile
        voc = Vocabulary(tc, basedocs; verbose=false)
        w = ones(Float32, vocsize(voc))
        emodel = VectorModel(EntropyWeighting(), TfWeighting(), voc; weight=w)
        ebase = (; model=emodel, synonyms=Dict{String,Vector{String}}(),
                   synonym_distances=nothing, lemmas=Dict{String,String}(),
                   stopword_candidates=String[], encoder=nothing)
        @test_throws ErrorException refit_profile(ebase, sampledocs; verbose=false)
    end
end
