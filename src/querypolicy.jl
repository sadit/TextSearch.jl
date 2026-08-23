# This file is a part of TextSearch.jl

export QueryPolicy

"""
    QueryPolicy(; correction=:auto, expansion=true, expansion_k=0, negligible_ratio=50)

How a query should be treated, as plain data travelling beside the query text.

The shape follows what commercial search does: a query is answered with the most probable
reading of it, and the person is always offered the same query answered literally. Neither
correcting nor expanding is a property of the profile -- the same profile serves both -- so it is
not baked into the artifact; it is a mark on the query, and every consumer (the CLI, an
application, a service endpoint) passes one of these instead of inventing its own flags.

# `correction` -- orthographic bridging, see [`resolve_query_tokens`](@ref)

- `:auto` (default) -- bridge only where the evidence says the typed spelling is wrong: it is not
  in the vocabulary, or it is negligible beside a commoner spelling of the same word. Where it
  bridges it **replaces**, because that is what correcting means.
- `:off` -- search exactly what was typed. This is the "search instead for …" escape, and a
  consumer that corrects by default owes the person a way to reach it.
- `:always` -- bridge every token whether or not anything suggests it is wrong. Trades precision
  for reach, and it is the only way to reach an accented alternative of a spelling that is itself
  common (typed `practico` is not negligible beside `práctico`, so `:auto` leaves it alone).

# `negligible_ratio`

What "negligible" means: a spelling holding less than `1 / negligible_ratio` of the documents of
its group's commonest spelling. `1` makes every spelling but the commonest negligible, and `Inf`
makes none of them so. Measured on 272,466 Spanish Wikipedia paragraphs, the ratio
between a typed spelling and its commonest alternative decays smoothly -- of the 517 map keys that
are themselves vocabulary tokens, 266 sit in [1,2) and the counts fall through 87, 50, 39, 23 and
15 to [35,50), then 6, 5, 11 and 15 above -- so there is no gap to snap to, but the region around
the default is sparse and the choice is not delicate. At 50, `musica` (9 documents against
`música`'s 4,404) is corrected while `granada` (43 against `Granada`'s 1,438, ratio 33) is not.

# `expansion`

Whether to widen the query with the profile's expansion network, and `expansion_k` how many
neighbours per token (`0` = all the profile stored). On by default and turned off on request, the
same way as correction: both are guesses about intent, so both are answerable literally.
"""
struct QueryPolicy
    correction::Symbol
    expansion::Bool
    expansion_k::Int
    negligible_ratio::Float64

    function QueryPolicy(; correction::Symbol=:auto, expansion::Bool=true,
                           expansion_k::Integer=0, negligible_ratio::Real=50)
        correction in (:off, :auto, :always) ||
            throw(ArgumentError("correction must be :off, :auto or :always; got $(repr(correction))"))
        negligible_ratio >= 1 ||
            throw(ArgumentError("negligible_ratio must be at least 1; got $negligible_ratio"))
        expansion_k >= 0 ||
            throw(ArgumentError("expansion_k must be non-negative; got $expansion_k"))
        new(correction, expansion, Int(expansion_k), Float64(negligible_ratio))
    end
end

function Base.show(io::IO, p::QueryPolicy)
    print(io, "QueryPolicy(correction=:", p.correction)
    p.correction === :off || p.negligible_ratio == 50 ||
        print(io, ", negligible_ratio=", p.negligible_ratio)
    p.expansion || print(io, ", expansion=false")
    p.expansion && p.expansion_k > 0 && print(io, ", expansion_k=", p.expansion_k)
    print(io, ")")
end
