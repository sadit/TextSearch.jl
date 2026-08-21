# This file is part of TextSearch.jl

function append_items!(idx::BM25InvertedFile, corpus::AbstractVector{T}; kwargs...) where {T<:AbstractString}
    append_items!(idx, getcontext(idx), VectorDatabase(bagofwords_corpus(idx.voc, corpus)); kwargs...)
end

function append_items!(idx::BM25InvertedFile, corpus::AbstractVector{T}; kwargs...) where {T<:TokenizedText}
    append_items!(idx, getcontext(idx), VectorDatabase(bagofwords_corpus(idx.voc, corpus)); kwargs...)
end


# ── accessor renaming: bare names -> get<field> ──────────────────────────────
#
# Accessors that share a name with the field they read (`Vocabulary.token`,
# `.occs`, `.ndocs`, `.trainsize`, `.numtokens`, `.textconfig`, `VectorModel.weight`) are now
# `get<field>`. The bare names were shadowable: a local or keyword argument called `trainsize`
# or `textconfig` silently hid the function, and the failure surfaced far away as
# "objects of type X are not callable". That happened twice in one sitting -- once with an
# `avgdoclen` keyword, once with a `textconfig` local -- so the collision is worth designing
# out rather than remembering.
#
# The old names keep working for one cycle. Note the deprecation does NOT remove the hazard
# for code that keeps using them; it only buys time to move. All of them were public before,
# `weight` included -- it sat on a continuation line of vmodel.jl's multi-line `export`, which
# is easy to miss when grepping export statements line by line.

@deprecate token(voc::Vocabulary, tokenID::Integer) gettoken(voc, tokenID)
@deprecate token(voc::Vocabulary) gettoken(voc)
@deprecate occs(voc::Vocabulary, tokenID::Integer) getoccs(voc, tokenID)
@deprecate occs(voc::Vocabulary) getoccs(voc)
@deprecate ndocs(voc::Vocabulary, tokenID::Integer) getndocs(voc, tokenID)
@deprecate ndocs(voc::Vocabulary) getndocs(voc)
@deprecate trainsize(voc::Vocabulary) gettrainsize(voc)
@deprecate numtokens(voc::Vocabulary) getnumtokens(voc)

@deprecate token(model::VectorModel, tokenID::Integer) gettoken(model, tokenID)
@deprecate token(model::VectorModel) gettoken(model)
@deprecate occs(model::VectorModel, tokenID::Integer) getoccs(model, tokenID)
@deprecate occs(model::VectorModel) getoccs(model)
@deprecate ndocs(model::VectorModel, tokenID::Integer) getndocs(model, tokenID)
@deprecate ndocs(model::VectorModel) getndocs(model)
@deprecate trainsize(model::VectorModel) gettrainsize(model)
@deprecate weight(model::VectorModel, tokenID::Integer) getweight(model, tokenID)
@deprecate weight(model::VectorModel) getweight(model)
