# This file is a part of TextSearch.jl

export QueryPipeline, QueryTerm, ResolvedQuery, query_tokens, querytokenset, querybow, queryvector

"""
    QueryPipeline(; policy=QueryPolicy(), variants=nothing, expansion=nothing, distances=nothing)

Everything the query side of a model needs, as plain data: the [`QueryPolicy`](@ref) to answer a
query under, the orthographic variant map correction bridges with, and the expansion network with
its optional distances.

There is one query pipeline and it lives here, which is the point of this type. Before it, the
work existed twice with each copy able to do something the other could not:

- The **vector-level** path (`expand_query!`, called from both inverted files) weighted the terms
  it added, by rank or by distance. But it iterated a query vector's nonzeros -- already token
  ids -- so a typed spelling absent from the vocabulary never reached it and correction was
  impossible by construction.
- The **string-level** path (the `textsearch` CLI's own) corrected first and expanded only from
  the commonest spelling of each corrected group, but added its terms unweighted, because it fed
  a `Set` for grep-like matching.

Unifying them is therefore not a choice between the two: [`query_tokens`](@ref) runs on strings,
where correction is possible, and emits weights, so nothing is lost. What each consumer does with
those weights is its own business -- see [`querybow`](@ref), [`queryvector`](@ref) and
[`querytokenset`](@ref).

`variants` may be `nothing`, which disables correction as surely as `policy.correction = :off`;
derive one with [`derive_variants`](@ref) once per model rather than once per query, since it is a
pure function of the vocabulary and costs 0.6s over 479,245 tokens.
"""
struct QueryPipeline
    policy::QueryPolicy
    variants::Union{Nothing,Dict{String,Vector{String}}}
    expansion::Union{Nothing,Dict{String,Vector{String}}}
    distances::Union{Nothing,Dict{String,Vector{Float32}}}

    function QueryPipeline(; policy::QueryPolicy=QueryPolicy(),
                             variants=nothing, expansion=nothing, distances=nothing)
        distances === nothing || expansion !== nothing ||
            throw(ArgumentError("distances were given without an expansion network to align them with"))
        new(policy, variants, expansion, distances)
    end
end

function Base.show(io::IO, qp::QueryPipeline)
    print(io, "QueryPipeline(", qp.policy)
    qp.variants === nothing || print(io, ", variants=", length(qp.variants))
    qp.expansion === nothing || print(io, ", expansion=", length(qp.expansion))
    qp.distances === nothing || print(io, "+distances")
    print(io, ")")
end

"""
    QueryTerm(token, source, factor, reason)

One term to search for, where it came from, and how much of that source's weight it carries.

`reason` is `:typed` for a spelling the person wrote, `:derived` or `:variant` for a correction
(see [`ResolvedToken`](@ref)), and `:expansion` for a neighbour the network contributed. For the
first three `source == token` and `factor == 1`: they are the *same word* differently spelled, and
nothing about a corrected spelling makes it a weaker match than the typo it replaced.

An expansion term names the query token whose list it came from, and `factor` is `exp(-d)` or
`1/rank`. It is a factor rather than an absolute weight because that is what a weighted
representation needs: `queryvector` gives the neighbour `factor` times **the source term's own
weight in the query**, which is what `expand_query!` did and what its tests pin. A neighbour of a
rare, high-idf query word should enter heavier than a neighbour of a common one.
"""
struct QueryTerm
    token::String
    source::String
    factor::Float32
    reason::Symbol
end

Base.show(io::IO, t::QueryTerm) = print(io, t.token, ":", t.reason,
                                        t.factor == 1 ? "" : "@" * string(round(t.factor; digits=3)))

"""
    ResolvedQuery(terms, resolution)

What [`query_tokens`](@ref) produces: the [`QueryTerm`](@ref)s to search for, and the
[`QueryResolution`](@ref) recording what correction did to each spelling that was typed -- so a
consumer can render [`explain`](@ref) and offer the literal query back.
"""
struct ResolvedQuery
    terms::Vector{QueryTerm}
    resolution::QueryResolution
end

Base.show(io::IO, q::ResolvedQuery) = print(io, "ResolvedQuery(", length(q.terms), " terms, ",
                                            q.resolution, ")")

"""
    query_tokens(voc::Vocabulary, query, qp::QueryPipeline=QueryPipeline()) -> ResolvedQuery

The query pipeline: turns what a person typed into the terms to search for, and records why.

`query` is raw text, a [`TokenizedText`](@ref), or an already-tokenized vector of strings. Text is
tokenized under `voc`'s own `TextConfig` -- the same one the documents went through, which it must
be, since the vocabulary's ids and counts came from it.

Then, in order:

1. **Correction.** [`resolve_query_tokens`](@ref) replaces spellings the evidence says are wrong
   and leaves the rest alone. Every spelling it produces weighs `1`.
2. **Expansion.** For each typed token, the network is looked up under *one* spelling -- the
   commonest of its corrected group, per [`expansion_sources`](@ref) -- and its neighbours are
   added with a weight: `exp(-d)` when `qp.distances` covers them, `1/rank` otherwise. Both are
   the weightings `expand_query!` used, kept so the numbers do not move.

A neighbour reachable from two query tokens appears twice, and one the person also typed appears
alongside the typed term: those are contributions, and it is the representation that decides what
to do with them -- [`queryvector`](@ref) adds them up, [`querybow`](@ref) and
[`querytokenset`](@ref) collapse them. Neighbours absent from `voc` are dropped, matching what the
vector-level path did with an id of `0`.
"""
function query_tokens(voc::Vocabulary, query, qp::QueryPipeline=QueryPipeline())
    tokens = query isa AbstractVector{<:AbstractString} ? query :
             collect(tokenize(voc.textconfig, query))
    res = resolve_query_tokens(voc, tokens, qp.variants, qp.policy)

    terms = QueryTerm[]
    seen = Set{String}()
    # Only the words the person meant are deduplicated. Expansion contributions are not: a
    # neighbour reachable from two query tokens is listed twice on purpose, because a weighted
    # representation adds the two contributions together (`queryvector`) while a set or a
    # presence-only bag collapses them anyway. Deduplicating here would silently drop the
    # second contribution and change scores.
    add!(tok, src, f, why) = begin
        if why === :expansion
            push!(terms, QueryTerm(tok, src, Float32(f), why))
        elseif !(tok in seen)
            push!(seen, tok)
            push!(terms, QueryTerm(tok, src, Float32(f), why))
        end
    end

    # a correction is the same word respelled, so it enters at full strength
    for r in res.resolved
        r.kept && add!(r.typed, r.typed, 1, :typed)
        for (form, why) in r.added
            add!(form, form, 1, why)
        end
    end
    # `res.tokens` is authoritative about what to search; a token contributed by another token's
    # group is already there, and this catches anything the loop above did not name
    for tok in res.tokens
        add!(tok, tok, 1, :typed)
    end

    if qp.policy.expansion && qp.expansion !== nothing
        for src in expansion_sources(res)
            neighbors = get(qp.expansion, src, nothing)
            neighbors === nothing && continue
            dl = qp.distances === nothing ? nothing : get(qp.distances, src, nothing)
            for (rank, syn) in enumerate(neighbors)
                qp.policy.expansion_k > 0 && rank > qp.policy.expansion_k && break
                token2id(voc, syn) == 0 && continue
                # rank weighting also covers a neighbour the distance list does not reach, so a
                # partially-populated `distances` degrades instead of erroring
                f = (dl !== nothing && rank <= length(dl)) ? _dist_weight(dl[rank]) :
                                                             _rank_weight(rank)
                add!(syn, src, f, :expansion)
            end
        end
    end

    ResolvedQuery(terms, res)
end

"""
    querytokenset(q::ResolvedQuery) -> Set{String}

The terms as a plain set, for a consumer that matches by token intersection and has no use for
weights -- `textsearch search`'s grep-like matching, for one.
"""
querytokenset(q::ResolvedQuery) = Set{String}(t.token for t in q.terms)

"""
    querybow(voc::Vocabulary, q::ResolvedQuery) -> BOW

The terms as a [`BOW`](@ref) of vocabulary ids, **presence only**: every term gets a count of `1`
and the weights are discarded.

That is not a shortcut. BM25 scoring never reads the query side's frequencies -- only which ids
are present -- so a weight there would be carried through the whole search and then ignored, and
`BOW`'s counts are `Int32` anyway. See `bm25score`.
"""
function querybow(voc::Vocabulary, q::ResolvedQuery)
    bow = BOW()
    for t in q.terms
        id = token2id(voc, t.token)
        id == 0 && continue
        bow[convert(UInt32, id)] = one(Int32)
    end
    bow
end

"""
    queryvector(model::VectorModel, q::ResolvedQuery; normalize=true) -> SparseVector

The terms as a weighted sparse vector under `model`: each term is weighted by the model as usual
and then scaled by its [`QueryTerm`](@ref) weight, so expansion neighbours enter attenuated by
rank or distance while typed and corrected spellings enter at full strength.

`normalize` (default `true`) is the last step, as it was in `expand_query!` -- a cosine index
needs it and doing it before scaling would undo the attenuation.
"""
function queryvector(model::VectorModel, q::ResolvedQuery; normalize::Bool=true, minweight=1e-6)
    voc = model.voc
    # The model weights the words the person meant; it is asked once, for exactly those, as a
    # `TokenizedText` so the ids are looked up rather than the strings re-tokenized.
    typed = TokenizedText([t.token for t in q.terms if t.reason !== :expansion])
    vec = vectorize(model, typed; normalize=false, minweight)

    isempty(q.terms) && return vec
    base = Dict{UInt32,Float32}(zip(vec.nzind, vec.nzval))
    I, F = collect(vec.nzind), collect(vec.nzval)
    for t in q.terms
        t.reason === :expansion || continue
        id = token2id(voc, t.token)
        id == 0 && continue
        sid = token2id(voc, t.source)
        sid == 0 && continue
        w = get(base, convert(UInt32, sid), 0f0) * t.factor
        w > 0 || continue
        push!(I, convert(UInt32, id))
        push!(F, w)
    end

    # duplicate ids sum, which is how a neighbour reached from two query tokens gets both
    # contributions and how one that was also typed keeps its own weight plus them
    vec = sparsevec(I, F, vocsize(voc))
    normalize && normalize!(vec)
    vec
end
