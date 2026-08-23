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
    ResolvedToken(typed, ndocs, rare, added, dominant)

What happened to one token of a query: the form as `typed`, how many documents hold that exact
spelling (`ndocs`, zero when it is not a vocabulary token), whether it is `rare` next to the
commonest spelling of its group, every form that was `added` for it as `form => reason`, and
`dominant` -- that commonest spelling, which is the one allowed to contribute query expansion
(see [`expansion_sources`](@ref)). `dominant` is empty only when no spelling of the group is in
the vocabulary at all.

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
    rare::Bool
    added::Vector{Pair{String,Symbol}}
    dominant::String
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
            t.ndocs == 0 ? "$(t.typed) not found, searched as $by" :
            t.rare       ? "$(t.typed) appears in only $(t.ndocs) document$(t.ndocs == 1 ? "" : "s"), also searched as $by" :
                           "$(t.typed) also searched as $by")
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

Base.show(io::IO, t::ResolvedToken) = print(io, t.typed, t.ndocs == 0 ? "*" : "",
    isempty(t.added) ? "" : " +[" * join(("$f:$w" for (f, w) in t.added), " ") * "]")

function Base.show(io::IO, r::QueryResolution)
    print(io, "QueryResolution(", length(r.tokens), " tokens")
    n = count(t -> !isempty(t.added), r.resolved)
    n == 0 || print(io, ", ", n, " bridged")
    print(io, ")")
end

"""
    resolve_query_tokens(voc::Vocabulary, tokens, variants=nothing;
                         policy::Symbol=:strict, negligible_ratio::Real=50) -> QueryResolution

Turns the tokens of a query into the tokens to search with, recording why.

Each typed token defines a **group**: the vocabulary spellings it could be searched as -- itself,
the spellings *computed* from its folded form per [`_derivable_forms`](@ref), and whatever the
stored `variants` map holds for that folded form. The group's commonest spelling is its
`dominant`, and `negligible_ratio` is what makes a spelling count: a spelling holding fewer than
`1 / negligible_ratio` of the dominant's documents is negligible.

# Policies

`:strict` (default) treats bridging as a **fallback**: if the typed form is a vocabulary token
and is not negligible, it is used as typed and nothing is added. Someone who wrote `Sol` or
`práctico` said something specific and it exists, so writing carefully is not penalized, and this
is what makes the whole approach safe to have on by default.

Being in the vocabulary is not by itself enough, and that is measured. `min_ndocs=5` on the
vocabulary means unaccented misspellings and foreign-language fragments *are* tokens, so the
premise "it exists, therefore they meant it" fails: on 272,466 Spanish Wikipedia paragraphs
`ingles` holds 7 documents against `inglés`'s 5,188, `dia` 10 against `día`'s 7,093, `musica` 9
against `música`'s 4,404. Of the 518 map keys that are themselves vocabulary tokens, 111 have a
spelling ten times commoner and 36 have one fifty times commoner. End to end, `search musica`
returned **0 paragraphs** before this rule and 314 after. So `:strict` bridges when the typed form
is present but negligible, and [`explain`](@ref) reports it with the document count that decided
it.

`:aggressive` skips the check entirely: every token is bridged whether or not it was found. It is
the only way to reach an accented alternative of a token that is itself common -- typed `practico`
is not negligible next to `práctico`, so `:strict` leaves it alone. Both are legitimate; the
caller knows which it wants.

# What the ratio prunes, and what it never touches

Negligible **bridged** spellings are dropped, which is what keeps a bridge from dragging in
tokens the corpus barely has: `sol` no longer reaches `SOL` (5 documents against 1,659), `ano` no
longer reaches `Ano` (5 against 18,285). This closes a gap where the query side was looser than
the artifact -- [`derive_variants`](@ref) applies `min_ndocs` when *building* the map, while
resolution used to admit any spelling merely present.

**The typed form is never dropped.** It is always in the result, negligible or not, so this only
ever adds. That is deliberate and it is where the ratio stops: `cuba` holds 18 documents against
`Cuba`'s thousands, and someone typing lowercase almost certainly means the country -- but the
barrel is what they typed, it costs a handful of false positives to keep it, and silently
discarding a person's own word to search something else instead is not a trade this should make on
its own. The sense separation that `lc=false` buys is preserved where it does the work anyway: in
idf, in the embeddings, and in the per-sense expansion lists.

Pass `negligible_ratio=Inf` to disable the pruning entirely and admit every spelling present.

# On ambiguity

`practico` typed without an accent is genuinely ambiguous between an adjective and a conjugated
verb, and under `:aggressive` it reaches both. That is the mirror image of what `del_diac=false`
buys on the document side, and the split is the point: the corpus keeps the distinction, so idf
and embeddings stay per-sense, while the query bridges it.
"""
function resolve_query_tokens(voc::Vocabulary, tokens, variants=nothing;
                              policy::Symbol=:strict, negligible_ratio::Real=50)
    policy in (:strict, :aggressive) ||
        throw(ArgumentError("policy must be :strict or :aggressive; got $(repr(policy))"))
    negligible_ratio > 0 ||
        throw(ArgumentError("negligible_ratio must be positive; got $negligible_ratio"))
    norm = voc.textconfig.normalization
    out = String[]
    resolved = ResolvedToken[]

    for tok in tokens
        id = token2id(voc, tok)
        typedn = id == 0 ? 0 : Int(getndocs(voc, id))
        tok in out || push!(out, tok)        # never dropped: this only ever adds

        cands = _candidate_group(voc, tok, variants, norm)
        if isempty(cands)
            push!(resolved, ResolvedToken(tok, typedn, false, Pair{String,Symbol}[],
                                          id == 0 ? "" : tok))
            continue
        end

        # the dominant spelling and the floor every other one has to clear
        dominant, best = (id == 0 ? "" : tok), typedn
        for (c, _, n) in cands
            n > best && ((dominant, best) = (c, n))
        end
        floor = best / negligible_ratio
        rare = id != 0 && typedn < floor

        if policy === :strict && id != 0 && !rare
            push!(resolved, ResolvedToken(tok, typedn, false, Pair{String,Symbol}[], tok))
            continue
        end

        added = Pair{String,Symbol}[]
        for (c, why, n) in cands
            n >= floor || continue
            c in out || push!(out, c)
            push!(added, c => why)
        end
        push!(resolved, ResolvedToken(tok, typedn, rare, added, dominant))
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
