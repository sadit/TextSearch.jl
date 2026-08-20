# This file is a part of TextSearch.jl

export lemma_clusters

function _lemma_pick(selector::Symbol)
    if selector === :shortest
        (voc, group) -> begin
            best, best_len = group[1], length(token(voc, group[1]))
            for tid in group
                len = length(token(voc, tid))
                len < best_len && ((best, best_len) = (tid, len))
            end
            best
        end
    elseif selector === :most_frequent
        (voc, group) -> begin
            best, best_occs = group[1], occs(voc, group[1])
            for tid in group
                o = occs(voc, tid)
                o > best_occs && ((best, best_occs) = (tid, o))
            end
            best
        end
    elseif selector === :shortest_then_most_frequent
        (voc, group) -> begin
            best = group[1]
            best_key = (length(token(voc, best)), -occs(voc, best))
            for tid in group
                key = (length(token(voc, tid)), -occs(voc, tid))
                key < best_key && ((best, best_key) = (tid, key))
            end
            best
        end
    else
        error("unknown lemma selector: $selector; supported: shortest, most_frequent, shortest_then_most_frequent")
    end
end

"""
    lemma_clusters(voc::Vocabulary, wordvecs::AbstractDatabase;
                   algorithm::Symbol=:fft, num_clusters::Integer=0,
                   selector::Symbol=:shortest, dist=Dist.Cosine()) -> Dict{String,String}

Clusters `voc`'s tokens by their embeddings in `wordvecs` (column `t` = embedding of
token `t`, e.g. from [`LSI.wordvectors`](@ref) or an externally supplied matrix) via one
of `SimilaritySearch`'s [`fft`](@ref)/[`dnet`](@ref)/[`randsel`](@ref)/
[`multirandsel`](@ref), then picks one canonical "lemma" per cluster:

- `:shortest`: the token with the fewest characters.
- `:most_frequent`: the token with the highest `occs(voc, id)`.
- `:shortest_then_most_frequent`: shortest first, `occs` breaks ties.

`num_clusters=0` defaults to `ceil(sqrt(vocsize(voc)))`. Singleton clusters are never
remapped. Returns `{token => lemma}` containing only non-identity entries -- a lookup
miss means the token is its own lemma.

# Example
```julia
lemmas = lemma_clusters(voc, wordvectors(lsi))
lemmas["cats"]   # e.g. "cat", if "cats" and "cat" landed in the same cluster
```
"""
function lemma_clusters(voc::Vocabulary, wordvecs::AbstractDatabase;
                         algorithm::Symbol=:fft, num_clusters::Integer=0,
                         selector::Symbol=:shortest, dist=Dist.Cosine())
    m = vocsize(voc)
    k = num_clusters > 0 ? num_clusters : max(1, ceil(Int, sqrt(m)))

    R = if algorithm === :fft
        fft(dist, wordvecs, k; verbose=false)
    elseif algorithm === :dnet
        dnet(dist, wordvecs, k; verbose=false)
    elseif algorithm === :randsel
        randsel(dist, wordvecs, k; verbose=false)
    elseif algorithm === :multirandsel
        multirandsel(dist, wordvecs, k; verbose=false)
    else
        error("unknown lemma clustering algorithm: $algorithm; supported: fft, dnet, randsel, multirandsel")
    end

    groups = Dict{UInt32,Vector{UInt32}}()
    for tid in 1:m
        push!(get!(groups, R.nn[tid], UInt32[]), UInt32(tid))
    end

    pick = _lemma_pick(selector)
    lemmas = Dict{String,String}()
    for group in values(groups)
        length(group) <= 1 && continue
        chosen_tok = token(voc, pick(voc, group))
        for tid in group
            tok = token(voc, tid)
            tok == chosen_tok || (lemmas[tok] = chosen_tok)
        end
    end

    lemmas
end
