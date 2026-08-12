# This file is part of TextSearch.jl

"""
    IdWeight(id, weight)

Stores a pair of entries of the posting lists

"""
struct IdWeight
    id::UInt32
    weight::Float32
end

"""
    IdIntWeight(id, weight)

Stores a pair of objects to be accessed. Similar to [`IdWeight`](@ref) but it stores an integer weight
"""
struct IdIntWeight
    id::UInt32
    weight::Int32
end

Base.zero(::Type{IdWeight}) = IdWeight(zero(UInt32), zero(Float32))
Base.zero(::Type{IdIntWeight}) = IdWeight(zero(UInt32), zero(Int32))

# Comparisons are defined purely on `id` so that `IdWeight`/`IdIntWeight` posting-list
# entries can be merged directly by `SimilaritySearch.Intersections`'s generic algorithms
# (`umerge!`/`bk!`/`bkt!`/`xmerge!`), which compare list entries with `<`/`==`/`!=`.
Base.isless(a::IdWeight, b::IdWeight) = a.id < b.id
Base.:(==)(a::IdWeight, b::IdWeight) = a.id == b.id
Base.isless(a::IdIntWeight, b::IdIntWeight) = a.id < b.id
Base.:(==)(a::IdIntWeight, b::IdIntWeight) = a.id == b.id

using Base.Order
import Base.Order: lt

struct IdOrderingType <: Ordering end
struct WeightOrderingType <: Ordering end
struct RevWeightOrderingType <: Ordering end
const IdOrder = IdOrderingType()
const WeightOrder = WeightOrderingType()
const RevWeightOrder = RevWeightOrderingType()

@inline lt(::IdOrderingType, a, b) = a.id < b.id
@inline lt(::WeightOrderingType, a, b) = a.weight < b.weight
@inline lt(::RevWeightOrderingType, a, b) = b.weight < a.weight
@inline lt(::IdOrderingType, a::Number, b::Number) = a < b
@inline lt(::WeightOrderingType, a::Number, b::Number) = a < b
@inline lt(::RevWeightOrderingType, a::Number, b::Number) = b < a
