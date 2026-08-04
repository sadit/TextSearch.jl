# This file is a part of TextSearch.jl

export TokenizerBuffer

"""
    TokenizerBuffer(n=128)

Self-contained scratch space reused across tokenization calls to avoid reallocating on
every call: `normtext` holds the normalized text, `tokens` accumulates the produced
tokens, `unigrams` holds the word-level basis used by n-word/skip-gram/collocation
generators, and `io` is scratch space for building individual token strings.

`Tokenizer` pools these internally for its own buffer-less convenience API (see
[`tokenize`](@ref)); callers that need to hold a buffer across several calls (e.g. to
safely alias its contents via `borrowtokenizedtext`) should borrow one from the same
pool via [`tokenizerbuffer`](@ref) instead of constructing their own.
"""
struct TokenizerBuffer
    normtext::Vector{Char}
    tokens::Vector{String}
    unigrams::Vector{String}
    io::IOBuffer

    function TokenizerBuffer(n::Integer=128)
        normtext = Char[]
        tokens = String[]
        unigrams = String[]
        io = IOBuffer()

        sizehint!(normtext, n)
        sizehint!(tokens, n)
        sizehint!(unigrams, n)

        new(normtext, tokens, unigrams, io)
    end
end

"""
    empty!(buff::TokenizerBuffer; normtext=true, tokens=true, unigrams=true)

Clears the requested scratch fields of `buff` in place. Returns `buff`.
"""
function Base.empty!(buff::TokenizerBuffer; normtext::Bool=true, tokens::Bool=true, unigrams::Bool=true)
    normtext && empty!(buff.normtext)
    tokens && empty!(buff.tokens)
    unigrams && empty!(buff.unigrams)
    buff
end
