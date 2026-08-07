# This file is a part of TextSearch.jl

export TokenizedText, tokenize, tokenize_corpus, unigrams, generate!, flush_token!, tokenizerbuffer

"""
    TokenizedText(tokens::AbstractVector{String})

Wraps the token list produced by [`tokenize`](@ref) for a single document. Behaves like
an `AbstractVector{String}` (it supports indexing, iteration, `push!`, `append!`, etc.)
and is the type consumed by `bagofwords`/`bagofwords!` and by `Vocabulary`-building
functions.

# Example

```julia
julia> collect(tokenize(TextConfig(), "Hello world!!"))
["hello", "world", "!!"]
```
"""
struct TokenizedText{StringVector<:AbstractVector{String}}
    tokens::StringVector
end

@inline Base.getindex(T::TokenizedText, i::Integer) = T.tokens[i]
@inline Base.setindex!(T::TokenizedText, v, i::Integer) = (T.tokens[i] = v)
@inline Base.firstindex(T::TokenizedText) = 1
@inline Base.lastindex(T::TokenizedText) = length(T)
@inline Base.eachindex(T::TokenizedText) = firstindex(T):lastindex(T)
@inline Base.length(T::TokenizedText) = length(T.tokens)
@inline Base.iterate(T::TokenizedText, s::Int=1) = iterate(T.tokens, s)
@inline Base.eltype(T::TokenizedText) = eltype(T.tokens)
@inline Base.push!(T::TokenizedText, a) = push!(T.tokens, a)
@inline Base.append!(T::TokenizedText, a) = append!(T.tokens, a)

tokenizedtext(s) = TokenizedText(Vector(s))
borrowtokenizedtext(s) = TokenizedText(s)

tokenize(copy_::Function, textconfig::TextConfig, text::TokenizedText, buff::TokenizerBuffer) = text
tokenize(copy_::Function, textconfig::TextConfig, text::TokenizedText) = text
tokenize(copy_::Function, textconfig::TextConfig, arr::AbstractVector{T}, buff::TokenizerBuffer) where {T<:TokenizedText} = arr
tokenize_corpus(copy_::Function, textconfig::TextConfig, arr::AbstractVector{T}) where {T<:TokenizedText} = arr

const EXTRA_PUNCT = Set(['~', '+', '^', '$', '|', '<', '>'])

"""
    tokenize(textconfig::TextConfig, text)
    tokenize(copy_::Function, textconfig::TextConfig, text)

    tokenize(textconfig::TextConfig, text, buff)
    tokenize(copy_::Function, textconfig::TextConfig, text, buff)

Tokenizes `text` using the given configuration. The `tokenize` makes heavy usage of buffers,
and when these buffers are shared it is mandatory to create a copy of the result (`buff.tokens`).

Change the default `copy` function to make an additional filtering of the tokens.
You can also pass the `identity` function to avoid copying.

# Example

```julia
julia> collect(tokenize(TextConfig(), "Hello world!!"))
["hello", "world", "!!"]
```
"""
function tokenize(copy_::Function, textconfig::TextConfig, text::AbstractString, buff::TokenizerBuffer)
    normalize_text(textconfig, text, buff.normtext)
    t = tokenize_(textconfig, buff)
    copy_(t)
end

function tokenize(copy_::Function, textconfig::TextConfig, arr, buff::TokenizerBuffer)
    normalize_text(textconfig, arr[1], buff.normtext)
    tokenize_(textconfig, buff)

    for i in 2:length(arr)
        empty!(buff.normtext); empty!(buff.unigrams)
        normalize_text(textconfig, arr[i], buff.normtext)
        tokenize_(textconfig, buff)
    end

    copy_(buff.tokens)
end

tokenize(textconfig::TextConfig, text) = tokenize(tokenizedtext, textconfig, text)

function tokenize(copy_::Function, textconfig::TextConfig, text)
    buff = take!(TOKENIZER_CACHES)
    empty!(buff)
    try
        tokenize(copy_, textconfig, text, buff)
    finally
        put!(TOKENIZER_CACHES, buff)
    end
end

"""
    tokenizerbuffer(f)

Borrows a [`TokenizerBuffer`](@ref) from `Tokenizer`'s own pool, passes it to `f`, and
returns it to the pool afterwards. Unlike the buffer-less [`tokenize`](@ref) methods
(which release their borrowed buffer before returning), the buffer stays borrowed for
the whole extent of `f`, so it is safe to alias its contents (e.g. via
[`borrowtokenizedtext`](@ref)) as long as they are consumed inside `f`.

# Example

```julia
julia> TextSearch.Tokenizer.tokenizerbuffer() do buff
           tokenize(borrowtokenizedtext, TextConfig(), "hello world", buff) |> collect
       end
["hello", "world"]
```
"""
@inline function tokenizerbuffer(f)
    buff = take!(TOKENIZER_CACHES)
    empty!(buff)
    try
        f(buff)
    finally
        put!(TOKENIZER_CACHES, buff)
    end
end

"""
    normalize_text(textconfig::TextConfig, text; limits::Bool=false)

Convenience method that normalizes `text` under `textconfig` (see
[`normalize_text(config, text, output; limits)`](@ref normalize_text)) and returns the
result as a `String` instead of writing into a caller-provided buffer.

# Example

```julia
julia> normalize_text(TextConfig(), "Café!!")
"cafe!!"
```
"""
function normalize_text(textconfig::TextConfig, text; limits::Bool=false)
    buff = take!(TOKENIZER_CACHES)
    empty!(buff)
    try
        String(normalize_text(textconfig, text, buff.normtext; limits))
    finally
        put!(TOKENIZER_CACHES, buff)
    end
end


"""
    tokenize_corpus(textconfig::TextConfig, arr; verbose=true)
    tokenize_corpus(copy_::Function, textconfig::TextConfig, arr; verbose=true)

Tokenize a list of texts. The `copy_` function is passed to [`tokenize`](@ref) as first argument.

# Example

```julia
julia> corpus = ["hello world", "the cat sat"];

julia> toks = tokenize_corpus(TextConfig(), corpus; verbose=false);

julia> collect(toks[1])
["hello", "world"]
```
"""
function tokenize_corpus(copy_::Function, textconfig::TextConfig, arr; verbose::Bool=true)
    n = length(arr)
    L = Vector{TokenizedText}(undef, n)
    minbatch = getminbatch(n)
    prog = Progress(n; dt=1, enabled=verbose, desc="tokenizing")

    @BATCHES minbatch for i in 1:n
        L[i] = tokenize(copy_, textconfig, arr[i])
        next!(prog)
    end

    L
end

tokenize_corpus(textconfig::TextConfig, arr; verbose::Bool=true) = tokenize_corpus(tokenizedtext, textconfig, arr; verbose)

function tokenize_(config::TextConfig, buff::TokenizerBuffer)
    gens = alltokengenerators(config.tokenization)
    tt = config.transformation
    mark_token_type = config.tokenization.mark_token_type

    for gen in gens
        needs_unigrams(gen) || generate!(gen, buff, tt, mark_token_type)
    end

    if any(needs_unigrams, gens)
        n1 = length(buff.tokens)
        unigrams(buff, tt)  # always populates buff.unigrams; also emits unigram tokens to buff.tokens

        any(g -> g isa UnigramGenerator, gens) || resize!(buff.tokens, n1)

        for gen in gens
            needs_unigrams(gen) && generate!(gen, buff, tt, mark_token_type)
        end
    end

    buff.tokens
end

"""
    generate!(gen::AbstractTokenGenerator, buff::TokenizerBuffer, tt::AbstractTokenTransformation, mark_token_type::Bool)

Runs `gen` over `buff`, appending its produced tokens to `buff.tokens`. Called by
[`tokenize`](@ref) for every generator in [`alltokengenerators`](@ref); a new
[`AbstractTokenGenerator`](@ref) subtype implements this method to define what it does.
[`UnigramGenerator`](@ref)'s tokens are emitted as a side effect of the shared
[`unigrams`](@ref) pass, so its own `generate!` is a no-op.
"""
generate!(::UnigramGenerator, buff::TokenizerBuffer, tt::AbstractTokenTransformation, mark_token_type::Bool) = nothing
generate!(gen::NWordGenerator, buff::TokenizerBuffer, tt::AbstractTokenTransformation, mark_token_type::Bool) = nwords(gen, buff, tt, mark_token_type)

function push_token_from_transform!(tokens, s::Nothing)
end

function push_token_from_transform!(tokens, s::AbstractString)
    push!(tokens, s)
end

function push_token_from_transform!(tokens, slist::AbstractVector)
    for s in slist
        push!(tokens, s)
    end
end

"""
    flush_token!(buff::TokenizerBuffer, tt::AbstractTokenTransformation, gen::AbstractTokenGenerator, mark_token_type::Bool)

Pushes the token accumulated in `buff.io` to the token list, applying `gen`'s
[`tokentag`](@ref) (when `mark_token_type`) and [`transform`](@ref) hook; discards empty
strings and tokens the transformation drops (returns `nothing` for).
"""
function flush_token!(buff::TokenizerBuffer, tt::AbstractTokenTransformation, gen::AbstractTokenGenerator, mark_token_type::Bool)
    buff.io.size == 0 && return nothing

    if mark_token_type
        tag = tokentag(gen)
        tag !== nothing && write(buff.io, '\t', tag)
    end

    s = transform(tt, gen, String(take!(buff.io)))
    push_token_from_transform!(buff.tokens, s)
end

ispunct2(c) = ispunct(c) || c in EXTRA_PUNCT

const UNIGRAM_GENERATOR = UnigramGenerator()

"""
    unigrams(buff::TokenizerBuffer, tt::AbstractTokenTransformation)

Performs the word tokenization
"""
function unigrams(buff::TokenizerBuffer, tt::AbstractTokenTransformation)
    n = length(buff.normtext)
    mfirst = length(buff.tokens) + 1
    # @info buff.normtext
    @inbounds for i in 2:n  # normtext[1] is BLANK
        c = buff.normtext[i]
        p = buff.normtext[i-1]

        if c == BLANK
            flush_token!(buff, tt, UNIGRAM_GENERATOR, false)
        elseif isemoji(c)
            # emoji
            flush_token!(buff, tt, UNIGRAM_GENERATOR, false)
            write(buff.io, c)
            flush_token!(buff, tt, UNIGRAM_GENERATOR, false)
        elseif ispunct2(p)
            # previous char is punct
            if ispunct2(c)
                # a punctuaction string
                buff.io.size >= 3 && flush_token!(buff, tt, UNIGRAM_GENERATOR, false)  # a bit large, so we flush and restart the punc string (3 is for most emojis and ...)
                write(buff.io, c)
            else
                !(p in ('#', '@', '_')) && flush_token!(buff, tt, UNIGRAM_GENERATOR, false)  # current is not punctuaction so we flush if not a meta word
                write(buff.io, c)
            end
        elseif ispunct2(c) && p !== BLANK
            ## single punctuaction alone
            flush_token!(buff, tt, UNIGRAM_GENERATOR, false)
            write(buff.io, c)
        else
            write(buff.io, c)
        end
    end

    flush_token!(buff, tt, UNIGRAM_GENERATOR, false)
    mlast = length(buff.tokens)

    for i in mfirst:mlast
        push!(buff.unigrams, buff.tokens[i])
    end

    buff.tokens
end

"""
    nwords(gen::NWordGenerator, buff::TokenizerBuffer, tt::AbstractTokenTransformation, mark_token_type)
"""
function nwords(gen::NWordGenerator, buff::TokenizerBuffer, tt::AbstractTokenTransformation, mark_token_type)
    q = gen.q
    n = length(buff.unigrams)

    @inbounds for i in 1:(n - q + 1)
        _last = i + q - 1
        for j in i:_last-1
            write(buff.io, buff.unigrams[j])
            write(buff.io, BLANK)
        end

        write(buff.io, buff.unigrams[_last])
        flush_token!(buff, tt, gen, mark_token_type)
    end

    buff.tokens
end

"""
    tokenize_paragraphs(text::AbstractString)::Vector{String}
    tokenize_paragraphs(textconfig::TextConfig, text::AbstractString)::Vector{String}
    tokenize_paragraphs([textconfig::TextConfig,] arr::AbstractVector)::Vector{String}

Splits `text` into paragraphs separated by two or more newlines (`\\n\\n+` or `\\r\\n\\r\\n+`).
Trims leading and trailing whitespace from each paragraph and filters out empty paragraphs.
When `textconfig` is provided, normalizes the text before paragraph splitting.
"""
function tokenize_paragraphs(text::AbstractString)
    paragraphs = String[]
    for p in split(text, r"(\r?\n){2,}")
        tp = strip(p)
        !isempty(tp) && push!(paragraphs, tp)
    end
    paragraphs
end

function tokenize_paragraphs(textconfig::TextConfig, text::AbstractString)
    res = String[]
    for p in tokenize_paragraphs(text)
        np = strip(normalize_text(textconfig, p))
        !isempty(np) && push!(res, np)
    end
    res
end

function tokenize_paragraphs(arr::AbstractVector)
    paragraphs = String[]
    for item in arr
        append!(paragraphs, tokenize_paragraphs(item))
    end
    paragraphs
end

function tokenize_paragraphs(textconfig::TextConfig, arr::AbstractVector)
    paragraphs = String[]
    for item in arr
        append!(paragraphs, tokenize_paragraphs(textconfig, item))
    end
    paragraphs
end

"""
    tokenize_sentences(text::AbstractString)::Vector{String}
    tokenize_sentences(textconfig::TextConfig, text::AbstractString)::Vector{String}
    tokenize_sentences([textconfig::TextConfig,] arr::AbstractVector)::Vector{String}

Splits `text` into sentences using sentence-ending punctuation (`.`, `!`, `?`) followed by whitespace
or newlines. Trims leading and trailing whitespace from each sentence and filters out empty sentences.
When `textconfig` is provided, normalizes each sentence after splitting.
"""
function tokenize_sentences(text::AbstractString)
    sentences = String[]
    for s in split(text, r"(?<=[.!?])\s+|[\r\n]+")
        ts = strip(s)
        !isempty(ts) && push!(sentences, ts)
    end
    sentences
end

function tokenize_sentences(textconfig::TextConfig, text::AbstractString)
    res = String[]
    for s in tokenize_sentences(text)
        ns = strip(normalize_text(textconfig, s))
        !isempty(ns) && push!(res, ns)
    end
    res
end

function tokenize_sentences(arr::AbstractVector)
    sentences = String[]
    for item in arr
        append!(sentences, tokenize_sentences(item))
    end
    sentences
end

function tokenize_sentences(textconfig::TextConfig, arr::AbstractVector)
    sentences = String[]
    for item in arr
        append!(sentences, tokenize_sentences(textconfig, item))
    end
    sentences
end

