# This file is a part of TextSearch.jl

export stopword_candidates

"""
    stopword_candidates(voc::Vocabulary, threshold::Real=0.5) -> Vector{String}
    stopword_candidates(model::VectorModel, threshold::Real=0.5) -> Vector{String}

Flags tokens whose document-frequency ratio `ndocs(voc, id) / trainsize(voc)` exceeds
`threshold` as stopword candidates, sorted by decreasing ratio (most extreme first). A
frequency heuristic only -- it does not inspect token semantics -- so results should be
reviewed before being wired into an [`IgnoreStopwords`](@ref) transformation.

# Example
```julia
candidates = stopword_candidates(voc, 0.5)
textconfig = TextConfig(voc.textconfig; transformation=IgnoreStopwords(Set(candidates)))
```
"""
function stopword_candidates(voc::Vocabulary, threshold::Real=0.5)
    0 < threshold <= 1 || throw(ArgumentError("threshold must be in (0, 1], got $threshold"))
    n = trainsize(voc)
    n > 0 || return String[]

    scored = Tuple{Float64,String}[]
    for id in eachindex(voc)
        ratio = ndocs(voc, id) / n
        ratio > threshold && push!(scored, (ratio, token(voc, id)))
    end

    sort!(scored; rev=true)
    [tok for (_, tok) in scored]
end

stopword_candidates(model::VectorModel, threshold::Real=0.5) = stopword_candidates(model.voc, threshold)
