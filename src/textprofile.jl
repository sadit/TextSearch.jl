# This file is a part of TextSearch.jl

export TextProfile, AppliedArtifacts, LineageStep, textconfig, policy, with_applied, isbase, istuned,
       lineage_summary

# ── the policy / artifact cut ────────────────────────────────────────────────
#
# `TextConfig` and a profile are the same idea at two lifecycle stages, but the useful line
# between them is not "in memory" versus "on disk". It is whether a corpus produced the thing:
#
#   policy     -- normalization flags, nlist, mark_token_type. Hand-authorable with no corpus.
#                 Two profiles merge only if their policies are IDENTICAL.
#   artifacts  -- stopword set, lemma map, synonym network, vocabulary counters, weights.
#                 Estimated from data. Two profiles merge by COMBINING these (union, rank
#                 fusion, plurality vote, addition).
#
# Those are opposite operations, which is why keeping artifacts inside `TextConfig`'s
# `transformation` field hurt: the merge path had to special-case "these differ only in their
# stopword set" inside a check that otherwise demanded equality, and each artifact ended up
# stored twice -- once where the tokenizer reads it, once where the profile saves it -- with
# nothing tying the copies together. Both places drifted, and both bugs were real.
#
# Here the artifacts have one home, this struct, and the `TextConfig` the tokenizer uses is
# *derived* from them by `textconfig`. The applied map cannot disagree with the saved map
# because there is only one map.

"""
    LineageStep(stage::Symbol, params::Dict{String,Any})

One step in how a profile came to be: `:fit` from a corpus, `:merge` of several profiles, or
`:refit` against a dataset sample. `params` carries the stage's own details (a fit's encoder
and corpus size, a merge's source count, a refit's `kappa`), as JSON-serializable scalars.

This replaces the `encoder` field, which had drifted into recording lineage anyway -- a merge
wrote `kind=:merged` and a refit `kind=:refit` into a field named for the encoder.
"""
struct LineageStep
    stage::Symbol
    params::Dict{String,Any}
end

LineageStep(stage::Symbol; kwargs...) =
    LineageStep(stage, Dict{String,Any}(String(k) => v for (k, v) in kwargs))

Base.show(io::IO, s::LineageStep) =
    print(io, s.stage, isempty(s.params) ? "" :
              "(" * join(("$k=$v" for (k, v) in sort(collect(s.params); by=first)), ", ") * ")")

"""
    AppliedArtifacts(; stopwords=false, lemmas=false, synonyms=false)

Which of a profile's artifacts are in play, as opposed to merely carried.

The distinction is the point of a *base* profile: a generic model computes a lemma map and a
synonym network, but whether to apply them belongs to the model being tuned from it. A tuned
profile that declines lemmatization simply does not apply the map, and one that never needed
it does not carry it either.

`stopwords` and `lemmas` are tokenization-time and enter the [`textconfig`](@ref).
`synonyms` is query-time only -- documents are never expanded, see
[`expand_synonyms!`](@ref) -- so it does not enter the config at all; it tells a consumer
building an index whether to hand the network over.
"""
Base.@kwdef struct AppliedArtifacts
    stopwords::Bool = false
    lemmas::Bool = false
    synonyms::Bool = false
end

Base.show(io::IO, a::AppliedArtifacts) = print(io, "applied(",
    join((n for n in (:stopwords, :lemmas, :synonyms) if getfield(a, n)), ", "), ")")

"""
    TextProfile(model; stopwords, lemmas, synonyms, synonym_distances, applied, lineage)

A finished, portable text model: the vocabulary and weights in `model`, plus the artifacts a
corpus produced, plus the lineage that says how it got here.

Each artifact is stored **once**, and `model.voc.textconfig` is rebuilt by the constructor as
the materialization of the profile's policy plus whichever artifacts `applied` selects. That
is a structural guarantee rather than a convention: there is no way to hold a profile whose
tokenizer applies a different lemma map than the one it saves.

Whether a profile is a base or a tuned model is read off the lineage rather than declared --
see [`isbase`](@ref)/[`istuned`](@ref) -- so it cannot contradict the facts, and it answers
"where did this come from?" at the same time.

Save and load with [`save_profile`](@ref)/[`load_profile`](@ref), combine batches of one
corpus with [`merge_profiles`](@ref), and adapt one to a dataset with
[`refit_profile`](@ref).
"""
struct TextProfile
    model::VectorModel
    stopwords::Set{String}
    lemmas::Dict{String,String}
    synonyms::Dict{String,Vector{String}}
    synonym_distances::Union{Nothing,Dict{String,Vector{Float32}}}
    applied::AppliedArtifacts
    lineage::Vector{LineageStep}

    function TextProfile(model::VectorModel,
                          stopwords::Set{String},
                          lemmas::Dict{String,String},
                          synonyms::Dict{String,Vector{String}},
                          synonym_distances::Union{Nothing,Dict{String,Vector{Float32}}},
                          applied::AppliedArtifacts,
                          lineage::Vector{LineageStep})
        # Materialize here, so the config the tokenizer sees is always this profile's own
        # artifacts. A caller cannot pass a mismatched one, because it is not an input.
        tc = _materialize(_policy(model.voc.textconfig), stopwords, lemmas, applied)
        new(_with_textconfig(model, tc), stopwords, lemmas, synonyms, synonym_distances,
            applied, lineage)
    end
end

function TextProfile(model::VectorModel;
                      stopwords=Set{String}(),
                      lemmas=Dict{String,String}(),
                      synonyms=Dict{String,Vector{String}}(),
                      synonym_distances=nothing,
                      applied::AppliedArtifacts=AppliedArtifacts(),
                      lineage::AbstractVector{LineageStep}=LineageStep[])
    TextProfile(model,
                Set{String}(String(w) for w in stopwords),
                Dict{String,String}(String(k) => String(v) for (k, v) in lemmas),
                Dict{String,Vector{String}}(String(k) => String[String(s) for s in v]
                                            for (k, v) in synonyms),
                synonym_distances === nothing ? nothing :
                    Dict{String,Vector{Float32}}(String(k) => Float32[Float32(d) for d in v]
                                                 for (k, v) in synonym_distances),
                applied, collect(LineageStep, lineage))
end

"""
    policy(p::TextProfile) -> TextConfig
    policy(tc::TextConfig) -> TextConfig

The corpus-independent half of a text configuration: normalization and tokenization, with no
transformation. This is what two profiles must share exactly to be merged, and what a user can
write by hand without any data.
"""
policy(p::TextProfile) = _policy(p.model.voc.textconfig)
policy(tc::TextConfig) = _policy(tc)
_policy(tc::TextConfig) = TextConfig(tc; transformation=IdentityTokenTransformation())

"""
    textconfig(p::TextProfile) -> TextConfig

The `TextConfig` this profile tokenizes with: its policy plus the artifacts it applies.

The chain order is not cosmetic. [`LemmaTransformation`](@ref) runs **before**
[`IgnoreStopwords`](@ref), because with the filter first a form that is not itself a stopword
survives it and is only then rewritten into one (`"las"` → `"la"`), smuggling the stopword
back into the vocabulary. Having exactly one function that knows this is most of the reason
the artifacts moved out of `TextConfig`.
"""
textconfig(p::TextProfile) = p.model.voc.textconfig

function _materialize(pol::TextConfig, stopwords, lemmas, applied::AppliedArtifacts)
    tt = AbstractTokenTransformation[]
    applied.lemmas && !isempty(lemmas) && push!(tt, LemmaTransformation(lemmas))
    applied.stopwords && !isempty(stopwords) && push!(tt, IgnoreStopwords(stopwords))

    transformation = isempty(tt) ? IdentityTokenTransformation() :
                     length(tt) == 1 ? only(tt) : ChainTransformation(tt)
    TextConfig(pol; transformation)
end

"""
    _with_textconfig(model::VectorModel, tc::TextConfig) -> VectorModel

Copies `model` with `tc` as its vocabulary's config, sharing the counter arrays rather than
copying them -- only the config field differs.
"""
function _with_textconfig(model::VectorModel, tc::TextConfig)
    v = model.voc
    voc = Vocabulary(tc, v.token, v.occs, v.ndocs, v.token2id, v.trainsize, v.numtokens)
    VectorModel(model.global_weighting, model.local_weighting, voc, model.maxoccs, model.weight)
end

"""
    with_applied(p::TextProfile; stopwords, lemmas, synonyms) -> TextProfile

`p` with different artifacts applied, rematerializing the `TextConfig`. This is how a
consumer turns lemmatization off (`textsearch search --no-lemmas`) or how a refit decides to
apply a base's carried map: change the marker, not the pipeline by hand.
"""
function with_applied(p::TextProfile;
                       stopwords::Bool=p.applied.stopwords,
                       lemmas::Bool=p.applied.lemmas,
                       synonyms::Bool=p.applied.synonyms)
    TextProfile(p.model, p.stopwords, p.lemmas, p.synonyms, p.synonym_distances,
                AppliedArtifacts(; stopwords, lemmas, synonyms), p.lineage)
end

"""
    isbase(p::TextProfile) -> Bool
    istuned(p::TextProfile) -> Bool

Whether `p` is a bootstrap model or one adapted to a dataset, **derived** from its lineage: a
profile with no `:refit` step is a base, one with a refit is tuned. Reading it off the lineage
rather than storing a label means it cannot contradict what actually happened, and a refit of
an already-tuned profile stays tuned without a rule for it.
"""
isbase(p::TextProfile) = !istuned(p)
istuned(p::TextProfile) = any(s -> s.stage === :refit, p.lineage)

"""
    lineage_summary(p::TextProfile) -> String

One line reading how `p` was produced, e.g. `"fit(trainsize=20000) -> merge(n_sources=16) ->
refit(kappa=400.0)"`. Used by `textsearch info`.
"""
lineage_summary(p::TextProfile) =
    isempty(p.lineage) ? "(no lineage recorded)" : join(string.(p.lineage), " -> ")

function Base.show(io::IO, p::TextProfile)
    println(io, "TextProfile: ", istuned(p) ? "tuned" : "base")
    println(io, "  ", lineage_summary(p))
    println(io, "  stopwords: ", length(p.stopwords), p.applied.stopwords ? " (applied)" : " (carried)")
    println(io, "  lemmas: ", length(p.lemmas), p.applied.lemmas ? " (applied)" : " (carried)")
    println(io, "  synonyms: ", length(p.synonyms), p.applied.synonyms ? " (applied)" : " (carried)",
            p.synonym_distances === nothing ? "" : ", with distances")
    show(io, p.model; prefix="  ")
end
