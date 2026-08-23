# This file is a part of TextSearch.jl

export derive_variants, resolve_query_tokens, QueryResolution, ResolvedToken, explain

using Base.Unicode

"""
    _fold(tok; lc::Bool, diac::Bool) -> String

One folded spelling of `tok`: case folded when `lc`, marks stripped when `diac`.

Mirrors what the normalization stage would have produced for this word had the profile been
configured that way -- `lowercase` first and then `Unicode.normalize`, in that order and for the
reason recorded in [`_preprocessing`](@ref): they disagree on the Turkish dotted capital I, where
`lowercase` gives `i` and case folding alone gives `i` plus a combining dot, and the second is a
token nobody can type.
"""
function _fold(tok::AbstractString; lc::Bool, diac::Bool)
    s = lc ? lowercase(tok) : tok
    Unicode.normalize(s; casefold=lc, stripmark=diac, stripcc=true, compat=true)
end

"""
    _derivable_forms(folded) -> Tuple

The corpus spellings a folded token can be *computed* back into: the token itself, its
first letter capitalized, and its upper-case form. These need no storage, which is most of what a
variant map would otherwise hold -- measured on 272,466 Spanish paragraphs, 46,200 of 68,693
folded-to-token pairs are of this shape, so leaving them out cuts the stored map by 70%.

Accents are the opposite case and are why the map exists at all: from `practico` there is no way
to compute whether the corpus writes `práctico` or `practicó`, and both are real words.
"""
_derivable_forms(folded::AbstractString) = (folded, uppercasefirst(folded), uppercase(folded))

"""
    derive_variants(voc::Vocabulary; min_ndocs=20, maxforms=8) -> Dict{String,Vector{String}}

Builds the query-side variant map for `voc`: for a folded spelling, the vocabulary tokens it
should reach that **cannot be computed from it**.

This is what lets a profile keep case and diacritics without becoming unsearchable. Preserving
them separates senses that folding destroys -- measured on Spanish Wikipedia, `granada` folded
returns only the city, while unfolded it also returns the heraldic charge (`gules azur bordura`),
and likewise for `cuba` (the barrel), `concepción` (the concept) and `león` (the animal). The cost
is that a query typed `leon` matches nothing, since the corpus writes `León`. This map is that
bridge, and it is query-only: applying it while indexing would blur the very distinctions it
exists to make searchable.

It cannot lean on the query-expansion network, and that is measured rather than assumed: across
16,376 case-twinned Spanish tokens the twin appears in its counterpart's expansion list only 13.6%
of the time and at rank 1 in 6.3%. Where it does appear, the pair are function words whose capital
is merely sentence-initial; where it does not, the two forms have genuinely different senses and
the network is right to keep them apart.

# What it leaves out, and why the map is small

**Derivable capitalization**, per [`_derivable_forms`](@ref): `madrid -> Madrid` is not stored
because it is computed at query time. Two thirds of the pairs are of that shape.

**Anything below `min_ndocs`.** A token appearing in a handful of documents is not a plausible
query target, so bridging to it buys nothing and the entries are pure size. Measured, the two
together take the map from 62,825 keys to 8,153 -- 87% smaller -- and the part that survives is
precisely the non-derivable part: the derivable fraction falls from 67% of pairs at a floor of 5
documents to 53% at 100, because rare tokens are disproportionately proper nouns whose only
variation is a capital, while frequent ones carry real accent alternatives.

Values are ordered by document frequency, most frequent first, and capped at `maxforms`.
"""
function derive_variants(voc::Vocabulary; min_ndocs::Integer=20, maxforms::Integer=8)
    norm = voc.textconfig.normalization
    # only folds the profile did not already apply can produce anything
    (norm.lc && norm.del_diac) && return Dict{String,Vector{String}}()

    acc = Dict{String,Vector{Tuple{Int,String}}}()
    for i in eachindex(voc)
        nd = Int(getndocs(voc, i))
        nd >= min_ndocs || continue
        tok = gettoken(voc, i)
        f = _fold(tok; lc=!norm.lc, diac=!norm.del_diac)
        (f == tok || isempty(f)) && continue
        tok in _derivable_forms(f) && continue
        push!(get!(() -> Tuple{Int,String}[], acc, f), (nd, tok))
    end

    out = Dict{String,Vector{String}}()
    for (f, forms) in acc
        sort!(forms; rev=true)
        keep = String[]
        for (_, t) in forms
            t in keep || push!(keep, t)
            length(keep) >= maxforms && break
        end
        out[f] = keep
    end
    out
end

"""
    ResolvedToken(typed, invocabulary, added)

What happened to one token of a query: the form as `typed`, whether that form was itself a
vocabulary token, and every form that was `added` for it as `form => reason`.

Reasons currently produced:

- `:derived` -- a spelling *computed* from the folded form, per [`_derivable_forms`](@ref):
  `madrid` reaching `Madrid`, `usa` reaching `USA`. Nothing is stored for these.
- `:variant` -- a spelling that had to be stored, because it cannot be computed: `practico`
  reaching `practicó`, `leon` reaching `León`.

The reason is a symbol rather than a Bool so that a future mechanism reports through the same
channel without changing this signature. The obvious next one is edit-distance correction --
`guerar` -> `guerra`, a transposition no fold can reach -- which SimilaritySearch can index the
vocabulary's strings for; it would report as `:edit`. Keeping the reasons open matters because a
deterministic fold and a distance guess are different kinds of claim, and a consumer telling the
user what was searched should be able to distinguish them. Deliberately not built yet.
"""
struct ResolvedToken
    typed::String
    invocabulary::Bool
    added::Vector{Pair{String,Symbol}}
end

"""
    QueryResolution(tokens, resolved)

The outcome of [`resolve_query_tokens`](@ref): `tokens` is what to search with, and `resolved`
records how each typed token got there.

Reportability is the point of the second field. What this does *is* spelling correction -- a
simple, deterministic kind, covering case and diacritics but not transpositions or wrong letters
-- and a search that silently substitutes what the user asked for owes them a way to see it.
"Showing results for X" needs this structure; so does deciding not to correct at all.
"""
struct QueryResolution
    tokens::Vector{String}
    resolved::Vector{ResolvedToken}
end

"""
    explain(r::QueryResolution) -> Vector{String}

One human-readable line per typed token that gained something, for a consumer that wants to tell
the user what was actually searched.
"""
function explain(r::QueryResolution)
    out = String[]
    for t in r.resolved
        isempty(t.added) && continue
        by = join(("$f ($why)" for (f, why) in t.added), ", ")
        push!(out, t.invocabulary ? "$(t.typed) also searched as $by" :
                                    "$(t.typed) not found, searched as $by")
    end
    out
end

Base.show(io::IO, t::ResolvedToken) = print(io, t.typed, t.invocabulary ? "" : "*",
    isempty(t.added) ? "" : " +[" * join(("$f:$w" for (f, w) in t.added), " ") * "]")

function Base.show(io::IO, r::QueryResolution)
    print(io, "QueryResolution(", length(r.tokens), " tokens")
    n = count(t -> !isempty(t.added), r.resolved)
    n == 0 || print(io, ", ", n, " bridged")
    print(io, ")")
end

"""
    resolve_query_tokens(voc::Vocabulary, tokens, variants=nothing;
                         policy::Symbol=:strict) -> QueryResolution

Turns the tokens of a query into the tokens to search with, recording why.

# Policies

`:strict` (default) treats bridging as a **fallback**, per token:

1. If the typed form is a vocabulary token, use it and stop. Someone who wrote `Sol` or
   `práctico` said something specific and it exists, so nothing is added -- writing carefully is
   not penalized, and this is what makes the whole approach safe to have on by default.
2. Otherwise fold it and add the **derivable** spellings that exist: `madrid` -> `Madrid`.
3. Then add whatever the stored `variants` map holds for the folded form: `leon` -> `León`.

`:aggressive` skips step 1: every token is folded and bridged whether or not it was found. This
trades precision for reach and is the right choice when recall matters more -- and it is the only
way to reach an accented alternative of a token that is *itself* in the vocabulary, since under
`:strict` a typed `practico` stops at step 1 and never sees `practicó`. Both are legitimate; the
caller knows which it wants.

Both policies only ever *add*. The typed form is always kept, so a token that is out of
vocabulary and unbridgeable passes through and matches nothing, exactly as it would have before.

# On ambiguity

`practico` typed without an accent is genuinely ambiguous between an adjective and a conjugated
verb, and under `:aggressive` it reaches both. That is the mirror image of what `del_diac=false`
buys on the document side, and the split is the point: the corpus keeps the distinction, so idf
and embeddings stay per-sense, while the query bridges it.
"""
function resolve_query_tokens(voc::Vocabulary, tokens, variants=nothing;
                              policy::Symbol=:strict)
    policy in (:strict, :aggressive) ||
        throw(ArgumentError("policy must be :strict or :aggressive; got $(repr(policy))"))
    norm = voc.textconfig.normalization
    out = String[]
    resolved = ResolvedToken[]

    for tok in tokens
        found = token2id(voc, tok) != 0
        tok in out || push!(out, tok)
        added = Pair{String,Symbol}[]

        if !(found && policy === :strict)
            f = _fold(tok; lc=!norm.lc, diac=!norm.del_diac)
            for cand in _derivable_forms(f)
                (cand in out || token2id(voc, cand) == 0) && continue
                push!(out, cand); push!(added, cand => :derived)
            end
            if variants !== nothing
                forms = get(variants, f, nothing)
                if forms !== nothing
                    for t in forms
                        (t in out || token2id(voc, t) == 0) && continue
                        push!(out, t); push!(added, t => :variant)
                    end
                end
            end
        end

        push!(resolved, ResolvedToken(tok, found, added))
    end

    QueryResolution(out, resolved)
end
