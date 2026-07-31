"""
    update_voc!(voc::Vocabulary, another::Vocabulary)
    update_voc!(pred::Function, voc::Vocabulary, another::Vocabulary)

Update `voc` vocabulary using another vocabulary. Optionally a predicate can be given to filter vocabularies.

Note 1: `corpuslen` remains unchanged (the structure is immutable and a new `Vocabulary` should be created to update this field).
Note 2: Both `voc` and `another` vocabularies should had been created with a _compatible_ [`TextConfig`](@ref) to be able to work on them.

# Example

```julia
julia> cfg = TextConfig();

julia> voc = Vocabulary(cfg, 0, 0);

julia> update_voc!(voc, Vocabulary(cfg, ["hello world"]; verbose=false));

julia> vocsize(voc)
2
```
"""
update_voc!(voc::Vocabulary, another::Vocabulary) = update_voc!(t->true, voc, another)

function update_voc!(pred::Function, voc::Vocabulary, another::Vocabulary)
    for i in eachindex(another)
        v = another[i]
        if pred(v)
            push_token!(voc, v.token, v.occs, v.ndocs)
        end
    end

    voc
end


# filtering functions
"""
    filter_tokens!(voc::Vocabulary, text::TokenizedText)

Removes tokens from a given tokenized text based using the valid vocabulary

# Example

```julia
julia> voc = Vocabulary(TextConfig(), ["hello world"]; verbose=false);

julia> tks = tokenize(TextConfig(), "hello unknownword world");

julia> filter_tokens!(voc, tks); collect(tks)
["hello", "world"]
```
"""
function filter_tokens!(voc::Vocabulary, text::TokenizedText)
    j = 0
    for i in eachindex(text.tokens)
        t = text.tokens[i]
        if haskey(voc.token2id, t)
            j += 1
            text.tokens[j] = t
        end
    end

    resize!(text.tokens, j)
    text
end

"""
    filter_tokens!(voc::Vocabulary, arr::AbstractVector{TokenizedText})

Applies [`filter_tokens!(voc, text)`](@ref filter_tokens!) to every tokenized document in `arr`.
"""
function filter_tokens!(voc::Vocabulary, arr::AbstractVector{TokenizedText})
    for t in arr
        filter_tokens!(voc, t)
    end

    arr
end

"""
    merge_voc(voc1::Vocabulary, voc2::Vocabulary[, ...])
    merge_voc(pred::Function, voc1::Vocabulary, voc2::Vocabulary[, ...])

Merges two or more vocabularies into a new one. A predicate function can be used to filter token entries.

Note: All vocabularies should had been created with a _compatible_ [`TextConfig`](@ref) to be able to work on them.

# Example

```julia
julia> cfg = TextConfig();

julia> voc1 = Vocabulary(cfg, ["hello world"]; verbose=false);

julia> voc2 = Vocabulary(cfg, ["hello there"]; verbose=false);

julia> vocsize(merge_voc(voc1, voc2))
3
```
"""
merge_voc(voc1::Vocabulary, voc2::Vocabulary, voclist...) = merge_voc(x->true, voc1, voc2, voclist...)

function merge_voc(pred::Function, voc1::Vocabulary, voc2::Vocabulary, voclist...)
    #all(v -> v isa Vocabulary, voclist) || throw(ArgumentError("arguments should be of type `Vocabulary`"))
    
    L = [voc1, voc2]
    for v in voclist
        push!(L, v)
    end

    sort!(L, by=vocsize, rev=true)
    voc = Vocabulary(voc1.textconfig, sum(trainsize(v) for v in L), sum(numtokens(v) for v in L))

    for v in L
        update_voc!(pred, voc, v)
    end

    voc
end

"""
    filter_tokens(pred::Function, voc::Vocabulary)

Returns a copy of reduced vocabulary based on evaluating `pred` function for each entry in `voc`

# Example

```julia
julia> voc = Vocabulary(TextConfig(), ["hello world", "hello there"]; verbose=false);

julia> voc2 = filter_tokens(t -> t.ndocs >= 2, voc);

julia> vocsize(voc2)
1
```
"""
function filter_tokens(pred::Function, voc::Vocabulary)
    V = Vocabulary(voc.textconfig, trainsize(voc), numtokens(voc))

    for i in eachindex(voc)
        v = voc[i]
        if pred(v)
            push_token!(V, v.token, v.occs, v.ndocs)
        end
    end

    V
end
