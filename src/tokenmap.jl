# This file is a part of TextSearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

export TokenMap, TokenHash, TokenTable, encode, decode

abstract type TokenMap end
const InvMapTypeD = Union{Dict{UInt64,String},Nothing}
const InvMapTypeV = Union{Vector{String},Nothing}

struct TokenHash
struct TokenHash{InvMap<:InvMapTypeD} <: TokenMap
    invmap::InvMap
    isconstruction::Bool
end

TokenHash(isconstruction=true; storetokens=false) = storetokens ? TokenHash(Dict{UInt64,String}(), isconstruction) : TokenHash(nothing, isconstruction)
TokenHash(tmap::TokenHash, isconstruction=false) = TokenHash(tmap.invmap, isconstruction)

StructTypes.StructType(::Type{<:TokenHash}) = StructTypes.Struct()

function encode(m::TokenHash, token::String)
    # note that mixing String and Vector{UInt8} produce different hashes for eq. data
    h = hash(token)
    m.isconstruction && m.invmap !== nothing && (m.invmap[h] = token)
    h
end

function decode(m::TokenHash, id::UInt64)
    m.invmap === nothing ? nothing : m.invmap[id]
end

Base.broadcastable(m::TokenHash) = (m,)

struct TokenTable{InvMap<:InvMapTypeV} <: TokenMap
    map::Dict{String,UInt64}
    invmap::InvMap
    isconstruction::Bool
end

TokenTable(isconstruction=true; storetokens=false) = TokenTable(Dict{String,UInt64}(), storetokens ? String[] : nothing, isconstruction)
TokenTable(tmap::TokenTable, isconstruction=false) = TokenTable(tmap.map, tmap.invmap, isconstruction)

function encode(m::TokenTable, token::AbstractString)
    !m.isconstruction && return get(m.map, token, one(UInt64)<<63 | rand(UInt64))
    h = get!(m.map, token, length(m.map)+1)
    m.invmap !== nothing && h == length(m.map) && push!(m.invmap, token)
    h
end

function decode(m::TokenTable, id::UInt64)
    m.invmap[id]
end

Base.broadcastable(m::TokenTable) = (m,)

