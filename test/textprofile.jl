using Test, TextSearch, SimilaritySearch

# The invariants the old design could not state.
#
# Artifacts used to live in two places: inside `TextConfig.transformation`, where the
# tokenizer read them, and at the profile's top level, where they were saved. Nothing tied the
# copies together, and both drifted -- a refitted profile once applied a 110,393-entry lemma
# map while saving and reporting the 40,320-entry one. With one home and a derived
# `TextConfig`, that class of bug is unrepresentable, and these tests say so out loud.

@testset "TextProfile: policy / artifact split" begin
    corpus = ["la casa roja", "las casas rojas", "la pera verde", "las peras verdes"]
    tc = TextConfig(tokenization=TokenizationConfig(nlist=[1]))
    lemmas = Dict("casas" => "casa", "rojas" => "roja", "las" => "la")
    stopwords = Set(["la"])

    mkmodel(docs=corpus; textconfig=tc) =
        VectorModel(IdfWeighting(), TfWeighting(), Vocabulary(textconfig, docs; verbose=false))

    """
        applied_lemma_map(p) -> Dict

    Digs the lemma mapping out of what the profile actually tokenizes with, so a test can
    compare it against what the profile carries.
    """
    function applied_lemma_map(p)
        find(tt::LemmaTransformation) = tt.lemmas
        find(::AbstractTokenTransformation) = nothing
        function find(tt::ChainTransformation)
            for s in tt.list
                m = find(s)
                m === nothing || return m
            end
            nothing
        end
        find(textconfig(p).transformation)
    end

    @testset "what a profile applies IS what it carries" begin
        p = TextProfile(mkmodel(); stopwords, lemmas,
                        applied=AppliedArtifacts(stopwords=true, lemmas=true))

        @test applied_lemma_map(p) == p.lemmas
        # and there is no way to construct one where they differ: the config is derived, not
        # accepted as an input
        @test textconfig(p) === p.model.voc.textconfig
    end

    @testset "carried but not applied leaves the pipeline alone" begin
        p = TextProfile(mkmodel(); stopwords, lemmas)   # applied defaults to all false
        @test !isempty(p.lemmas)                        # carried
        @test applied_lemma_map(p) === nothing          # but not in the pipeline
        @test textconfig(p).transformation isa IdentityTokenTransformation
        # so the vocabulary is NOT lemmatized
        @test token2id(p.model.voc, "casas") != 0
    end

    @testset "with_applied re-materializes rather than editing a pipeline by hand" begin
        base = TextProfile(mkmodel(); stopwords, lemmas)
        on = with_applied(base; lemmas=true)

        @test applied_lemma_map(on) == lemmas
        @test applied_lemma_map(with_applied(on; lemmas=false)) === nothing
        # artifacts are untouched by toggling; only the derived config moves
        @test on.lemmas == base.lemmas
        @test on.stopwords == base.stopwords
    end

    @testset "lemmas chain BEFORE stopwords, whichever path built the profile" begin
        # Not cosmetic: with the filter first, "las" is not itself a stopword, survives it,
        # and is only then rewritten to "la" -- putting a stopword into the vocabulary through
        # the back door. One function knows this order now.
        p = TextProfile(mkmodel(); stopwords, lemmas,
                        applied=AppliedArtifacts(stopwords=true, lemmas=true))
        @test collect(tokenize(textconfig(p), "las casas rojas")) == ["casa", "roja"]

        # and the vocabulary built under it agrees
        voc = Vocabulary(textconfig(p), corpus; verbose=false)
        @test token2id(voc, "la") == 0
        @test token2id(voc, "las") == 0
        @test token2id(voc, "casas") == 0
        @test ndocs(voc, token2id(voc, "casa")) == 2
    end

    @testset "policy is the corpus-independent half" begin
        p = TextProfile(mkmodel(); stopwords, lemmas,
                        applied=AppliedArtifacts(stopwords=true, lemmas=true))
        pol = policy(p)

        @test pol.transformation isa IdentityTokenTransformation
        @test pol.normalization === textconfig(p).normalization
        @test pol.tokenization === textconfig(p).tokenization
        # a policy tokenizes like a bare config: no artifact leaks into it
        @test collect(tokenize(pol, "las casas rojas")) == ["las", "casas", "rojas"]
    end

    @testset "base versus tuned is derived from the lineage" begin
        fit = TextProfile(mkmodel(); lineage=[LineageStep(:fit; trainsize=4)])
        @test isbase(fit)
        @test !istuned(fit)

        merged = TextProfile(mkmodel();
                             lineage=[LineageStep(:fit), LineageStep(:merge; n_sources=3)])
        @test isbase(merged)          # merging batches of one corpus is still a base

        tuned = TextProfile(mkmodel();
                            lineage=[LineageStep(:fit), LineageStep(:refit; kappa=400.0)])
        @test istuned(tuned)
        @test !isbase(tuned)

        # a refit of a refit stays tuned, with no rule needed for it
        again = TextProfile(mkmodel(); lineage=[tuned.lineage..., LineageStep(:refit; kappa=10.0)])
        @test istuned(again)

        # a profile with no recorded lineage is a base by default, not an error
        @test isbase(TextProfile(mkmodel()))

        @test occursin("fit", lineage_summary(merged))
        @test occursin("merge", lineage_summary(merged))
        @test occursin("no lineage", lineage_summary(TextProfile(mkmodel())))
    end

    @testset "an empty artifact never enters the pipeline, even marked applied" begin
        # asking to apply nothing is not an error, it is a no-op
        p = TextProfile(mkmodel(); applied=AppliedArtifacts(stopwords=true, lemmas=true))
        @test textconfig(p).transformation isa IdentityTokenTransformation
    end
end
