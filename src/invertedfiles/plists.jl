# This file is part of InvertedFiles.jl

import ..Intersections: getkey

"""
    struct PostingList

A paired list of identifiers and weights
"""
struct PostingList{ListType<:AbstractVector}
    list::ListType
    tokenID::UInt32
    weight::Float32  # useful for search time and saving global data
end

@inline Base.length(plist::PostingList) = length(plist.list)

@inline Base.@propagate_inbounds getkey(plist::PostingList{Vector{UInt32}}, i::Integer)::UInt32 = plist.list[i]
@inline Base.@propagate_inbounds getkey(plist::PostingList{Vector{IdWeight}}, i::Integer)::UInt32 = plist.list[i].id
@inline Base.@propagate_inbounds getkey(plist::PostingList{Vector{IdIntWeight}}, i::Integer)::UInt32 = plist.list[i].id
@inline Base.@propagate_inbounds getkey(plist::PostingList{<:SubArray{UInt32}}, i::Integer)::UInt32 = plist.list[i]
@inline Base.@propagate_inbounds getkey(plist::PostingList{<:SubArray{IdWeight}}, i::Integer)::UInt32 = plist.list[i].id
@inline Base.@propagate_inbounds getkey(plist::PostingList{<:SubArray{IdIntWeight}}, i::Integer)::UInt32 = plist.list[i].id
