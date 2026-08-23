# This file is a part of TextSearch.jl

export stopword_candidates

"""
    stopword_candidates(voc::Vocabulary, threshold::Real=0.5) -> Vector{String}
    stopword_candidates(model::VectorModel, threshold::Real=0.5) -> Vector{String}

Flags tokens whose document-frequency ratio `getndocs(voc, id) / gettrainsize(voc)` exceeds
`threshold` as stopword candidates, sorted by decreasing ratio (most extreme first). A
frequency heuristic only -- it does not inspect token semantics -- so results should be
reviewed before being wired into a [`TokenPipeline`](@ref)'s `stopwords` stage.

**Detection is per spelling; removal is per word.** Under a profile that keeps case
(`lc=false`) a function word is several vocabulary tokens, and the threshold measures each
separately -- so it sees the fraction of documents containing a *spelling*, not a word.
Measured on 272,466 Spanish Wikipedia paragraphs at `threshold=0.1`, detection caught the
four commonest paragraph-initial forms (`El`, `En`, `La`, `Los`) and let 52 twins through,
including `Las` (22,040 documents, df 0.081), `A` (19,701), `Se` (17,315), `Por` (12,891) and
`De` (9,800) -- the same function word filtered in one casing and indexed as content in the
other. So once a spelling is flagged, every other casing of it that the vocabulary holds is
flagged with it.

The alternative -- pooling document frequencies across casings before comparing to the
threshold -- was rejected: the pooled value is not observable from these counters (a document
containing both `de` and `De` is counted twice, and the sum can exceed 1), and it would shift
the calibration of a threshold that was measured per spelling.

Casing is folded; diacritics are not. Folding diacritics would merge `té`/`te`, `más`/`mas`
and `sí`/`si`, deleting content words -- which is what a profile with `del_diac=false` exists
to prevent. The cost of folding case is small and worth naming: an acronym colliding with a
function word goes too (`ES`, 41 documents, follows `es`), which lowercasing profiles already
did.

# Example
```julia
candidates = stopword_candidates(voc, 0.5)
textconfig = TextConfig(voc.textconfig; pipeline=TokenPipeline(stopwords=Set(candidates)))
```
"""
function stopword_candidates(voc::Vocabulary, threshold::Real=0.5)
    0 < threshold <= 1 || throw(ArgumentError("threshold must be in (0, 1], got $threshold"))
    n = gettrainsize(voc)
    n > 0 || return String[]

    flagged = Int[]
    for id in eachindex(voc)
        getndocs(voc, id) / n > threshold && push!(flagged, id)
    end
    isempty(flagged) && return String[]

    # A profile that already lowercases has one spelling per word, so there is nothing to
    # extend and the grouping pass is skipped entirely.
    if !voc.textconfig.normalization.lc
        bycase = Dict{String,Vector{Int}}()
        for id in eachindex(voc)
            push!(get!(() -> Int[], bycase, lowercase(gettoken(voc, id))), id)
        end
        seen = Set(flagged)
        for id in copy(flagged), sib in get(bycase, lowercase(gettoken(voc, id)), Int[])
            sib in seen || (push!(flagged, sib); push!(seen, sib))
        end
    end

    scored = [(getndocs(voc, id) / n, gettoken(voc, id)) for id in flagged]
    sort!(scored; rev=true)
    [tok for (_, tok) in scored]
end

stopword_candidates(model::VectorModel, threshold::Real=0.5) = stopword_candidates(model.voc, threshold)
