# This file is a part of TextSearch.jl

export derive_variants, resolve_query_tokens, QueryResolution, ResolvedToken, explain,
       expansion_sources

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
    derive_variants(voc::Vocabulary; min_ndocs=1, maxforms=8) -> Dict{String,Vector{String}}

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

# This is computed, never stored

A profile does not carry a variant map: it is a pure function of the vocabulary the profile
already holds, so storing one is a second copy of the same information -- and a copy that goes
wrong, because per-part maps cannot be combined into the map the combined vocabulary yields.
Measured on 9 parts of Portuguese Wikipedia, unioning them gave 21,646 keys against the 30,968
the merged vocabulary itself produces: a strict subset missing 30%, `tropecar -> tropeçar` among
them at 132 documents corpus-wide and about 15 per part, under any per-part floor. Deriving from
the merged counters instead costs 0.24s over 479,245 tokens.

# What it leaves out

**Derivable capitalization**, per [`_derivable_forms`](@ref): `madrid -> Madrid` is not included
because it is computed at query time from the folded form. Two thirds of the pairs are of that
shape, and the fraction falls with frequency -- 67% of pairs at 5 documents against 53% at 100 --
because rare tokens are disproportionately proper nouns whose only variation is a capital, while
frequent ones carry real accent alternatives.

**Anything below `min_ndocs`**, which defaults to no filtering at all. The floor was worth having
while the map was an artifact on disk; now that it is transient, its only remaining job is cost,
and there is little to buy: on the 479,245-token Portuguese vocabulary a floor of 1 gives 61,925
keys in 0.62s and 10.3 MB against 30,968 in 0.27s and 4.1 MB at a floor of 20. The 62k map has
twice the coverage of exactly the long tail a person is most likely to mistype and least likely to
find otherwise. Note also that a vocabulary pruned at fit time already imposes its own floor:
these profiles use `min_ndocs=5` there, so 1, 2 and 5 here produce identical maps.

Query-time quality is not this function's job either. A bridged spelling is admitted only if it is
not negligible beside the commonest spelling of its group -- see `negligible_ratio` in
[`QueryPolicy`](@ref) -- which is a relative test and a better one than any absolute count.

Values are ordered by document frequency, most frequent first, and capped at `maxforms`.
"""
function derive_variants(voc::Vocabulary; min_ndocs::Integer=1, maxforms::Integer=8)
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
    ResolvedToken(typed, ndocs, kept, added, dominant, dominantdocs)

What happened to one token of a query: the form as `typed`, how many documents hold that exact
spelling (`ndocs`, zero when it is not a vocabulary token), whether that spelling was `kept` in
the search set, every form that was `added` for it as `form => reason`, `dominant` -- the
commonest spelling of its group, which is the one allowed to contribute query expansion (see
[`expansion_sources`](@ref)) -- and `dominantdocs`, how many documents hold *that*. `dominant` is
empty only when no spelling of the group is in the vocabulary at all.

Both counts are carried because a correction is only explicable as a comparison. "appears in only
1,020 documents" is not a reason at corpus scale, where 1,020 documents is a perfectly ordinary
word; "1,020 against `música`'s 219,000" is.

`kept` is false exactly when the token was **corrected**: something was bridged for it *and* the
evidence said the typed spelling was wrong -- absent from the vocabulary, or negligible beside a
commoner spelling of the same word. Together with `ndocs` it tells a consumer which of three
things to report: a spelling that was not there at all, one that was there but too rare to be
what was meant, or an enrichment that left the typed form standing.

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
    ndocs::Int
    kept::Bool
    added::Vector{Pair{String,Symbol}}
    dominant::String
    dominantdocs::Int
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
        push!(out,
            t.kept       ? "$(t.typed) also searched as $by" :
            t.ndocs == 0 ? "$(t.typed) not found, searched as $by instead" :
                           "$(t.typed) is in $(t.ndocs) document$(t.ndocs == 1 ? "" : "s") against " *
                           "$(t.dominant)'s $(t.dominantdocs), so it reads as a misspelling; " *
                           "searched as $by instead")
    end
    out
end

"""
    expansion_sources(r::QueryResolution) -> Vector{String}

The tokens a consumer should look up in a query-expansion network: one per typed token, its
group's commonest spelling.

Not every token that was searched, and this is measured rather than stylistic. Expansion over the
whole bridged set mixes senses, because bridging deliberately reaches spellings the corpus barely
holds and their neighbour lists come from a handful of documents: on Spanish Wikipedia paragraphs
`SOL` (5 documents) gives `digitalizada máx chip flash SDRAM`, `Ano` (5) gives the Annobón islands,
`rio` (12, the verb *reír*) gives `llorar tiró Nazgûl`, and `musica` (9, Italian-language
paragraphs) gives `libreto Puccini Verdi Semiramide`. A search for `musica de leon` returned
Antonio Vivaldi and a chess article.

Expanding only what the user typed is not the fix either -- it fails in exactly those cases, since
the typed form *is* the rare one. The dominant spelling is: `sol` bridged gives `Sol` (1,659
documents) and `afelio perihelio eclipses eclíptica`, `rio` gives `río` and `afluente confluencia
cauce desemboca`. The rarer spellings stay in the search set as matching terms, where a wrong one
costs a handful of false positives instead of eight high-idf junk terms.

When nothing was bridged, the group is the typed token alone and this is exactly the token list --
so an unbridged query expands as it always did.
"""
function expansion_sources(r::QueryResolution)
    out = String[]
    for t in r.resolved
        isempty(t.dominant) && continue
        t.dominant in out || push!(out, t.dominant)
    end
    out
end

Base.show(io::IO, t::ResolvedToken) = print(io, t.typed, t.kept ? "" : "*",
    isempty(t.added) ? "" : " +[" * join(("$f:$w" for (f, w) in t.added), " ") * "]")

function Base.show(io::IO, r::QueryResolution)
    print(io, "QueryResolution(", length(r.tokens), " tokens")
    n = count(t -> !isempty(t.added), r.resolved)
    n == 0 || print(io, ", ", n, " bridged")
    print(io, ")")
end

"""
    resolve_query_tokens(voc::Vocabulary, tokens, variants=nothing,
                         policy::QueryPolicy=QueryPolicy()) -> QueryResolution

Turns the tokens of a query into the tokens to search with, recording why.

Each typed token defines a **group**: the vocabulary spellings it could be searched as -- itself,
the spellings *computed* from its folded form per [`_derivable_forms`](@ref), and whatever the
stored `variants` map holds for that folded form. The group's commonest spelling is its
`dominant`, and a spelling holding less than `1 / policy.negligible_ratio` of the dominant's
documents is negligible. `policy.correction` decides what to do with that; see
[`QueryPolicy`](@ref) for the three modes.

# Correcting replaces; enriching adds

A typed spelling is dropped from the result exactly when something was bridged for it **and** the
evidence says it was wrong: absent from the vocabulary, or negligible. That is what a correction
is, and it is why `:off` has to exist -- a consumer that corrects by default owes the person the
same query answered literally, the way a commercial engine offers "search instead for …".
[`explain`](@ref) phrases the two cases differently so a consumer can render that offer.

Where no evidence says otherwise the typed spelling stays and bridging only adds: under `:always`
a healthy `sol` reaches `Sol` while remaining itself.

# Presence is not evidence of intent

`min_ndocs=5` on the vocabulary means unaccented misspellings and foreign-language fragments *are*
tokens, so "it exists, therefore they meant it" fails. On 272,466 Spanish Wikipedia paragraphs
`ingles` holds 7 documents against `inglés`'s 5,188, `dia` 10 against `día`'s 7,093, `musica` 9
against `música`'s 4,404 -- and of the 518 map keys that are themselves vocabulary tokens, 111
have a spelling ten times commoner and 36 have one fifty times commoner. End to end, `search
musica` returned **0 paragraphs** while stopping at the typed form and 314 after correcting it.

The ratio is also what keeps a bridge from dragging in spellings the corpus barely holds, closing
a gap where resolution admitted any spelling merely present while [`derive_variants`](@ref)
applied `min_ndocs` when building the map: `sol` does not reach `SOL` (5 documents against 1,659),
whose neighbours were `digitalizada máx chip flash SDRAM`.

# On ambiguity

`practico` typed without an accent is genuinely ambiguous between an adjective and a conjugated
verb, and under `:always` it reaches both. That is the mirror image of what `del_diac=false` buys
on the document side, and the split is the point: the corpus keeps the distinction, so idf and
embeddings stay per-sense, while the query bridges it.
"""
function resolve_query_tokens(voc::Vocabulary, tokens, variants=nothing,
                              policy::QueryPolicy=QueryPolicy())
    norm = voc.textconfig.normalization
    out = String[]
    resolved = ResolvedToken[]

    for tok in tokens
        id = token2id(voc, tok)
        typedn = id == 0 ? 0 : Int(getndocs(voc, id))

        cands = policy.correction === :off ? Tuple{String,Symbol,Int}[] :
                                             _candidate_group(voc, tok, variants, norm)
        if isempty(cands)
            tok in out || push!(out, tok)
            push!(resolved, ResolvedToken(tok, typedn, true, Pair{String,Symbol}[],
                                          id == 0 ? "" : tok, typedn))
            continue
        end

        # the dominant spelling, and the floor every other one has to clear
        dominant, best = (id == 0 ? "" : tok), typedn
        for (c, _, n) in cands
            n > best && ((dominant, best) = (c, n))
        end
        wrong = id == 0 || typedn < best / policy.negligible_ratio

        if policy.correction === :auto && !wrong
            tok in out || push!(out, tok)
            push!(resolved, ResolvedToken(tok, typedn, true, Pair{String,Symbol}[], tok, typedn))
            continue
        end

        # a correction replaces what was typed; an enrichment leaves it in place. `wrong` implies
        # the dominant is one of the candidates, and it always clears its own floor, so a
        # correction never empties the search set.
        wrong || tok in out || push!(out, tok)
        added = Pair{String,Symbol}[]
        for (c, why, n) in cands
            n >= best / policy.negligible_ratio || continue
            c in out || push!(out, c)
            push!(added, c => why)
        end
        push!(resolved, ResolvedToken(tok, typedn, !wrong, added, dominant, best))
    end

    QueryResolution(out, resolved)
end

"""
    _candidate_group(voc, tok, variants, norm) -> Vector{Tuple{String,Symbol,Int}}

The vocabulary spellings `tok` could be searched as besides itself, each with the reason it was
reached and its document count. Computed spellings come first, then stored ones, so the order a
caller sees follows how much had to be assumed.
"""
function _candidate_group(voc::Vocabulary, tok, variants, norm)
    cands = Tuple{String,Symbol,Int}[]
    f = _fold(tok; lc=!norm.lc, diac=!norm.del_diac)

    function offer(cand, why)
        cand == tok && return
        i = token2id(voc, cand)
        i == 0 && return
        any(c -> c[1] == cand, cands) && return
        push!(cands, (String(cand), why, Int(getndocs(voc, i))))
    end

    for cand in _derivable_forms(f)
        offer(cand, :derived)
    end
    if variants !== nothing
        forms = get(variants, f, nothing)
        forms === nothing || for t in forms
            offer(t, :variant)
        end
    end
    cands
end
