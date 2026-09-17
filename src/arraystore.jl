# This file is a part of TextSearch.jl

# ── binary array members of a profile ────────────────────────────────────────
#
# A profile is a directory (or zip) of plain JSON, and `save_profile`'s docstring promises that
# every file is "fully inspectable/diffable/portable". That promise is worth keeping for anything
# a person might read: a vocabulary, a lemma map, a stopword list, the expansion network's shape.
#
# It is worth breaking for one kind of content -- a large array of numbers nobody reads by eye --
# and the numbers say how much:
#
#   query_expansion_distances, 47,501 cosine distances     569,045 B as JSON text
#                                             as Float32       190,004 B  (33.4%)
#                                             as u8 quantized   47,501 B  ( 8.3%)
#
# and the u8 round trip is retrieval-identical (top-1 agreement 1.0000, top-10 overlap 0.9995
# over 400 queries against a 16,640-document index; mean round-trip error 0.00092 on a
# [0, 0.938] range). JSON is costing 12x for digits no one will ever look at.
#
# This file is that escape hatch, and it is deliberately ONE mechanism rather than one per
# artifact. The expansion network's distances need it now; a stored LSI projection needs exactly
# the same thing at a different shape and dtype (a `(k, vocsize)` matrix of Float32, quantized
# the same way), and having two ad-hoc encodings of "an array in a zip member" is how the applied
# and saved copies of an artifact drifted apart the last time this format grew.
#
# What a caller gets is: hand `_save_array` an array, get back a manifest entry that fully
# describes it; hand `_load_array` that entry and a byte reader, get the array back. Everything
# needed to interpret the bytes -- dtype, shape, and the quantization range if any -- lives in the
# manifest beside the file name, so the binary member is never self-describing and never has to
# be.

"""
    _ARRAY_DTYPES

The element types a binary profile member may hold, by their manifest tag. Deliberately short:
an unknown tag is an error rather than a guess, and widening this is a format change.
"""
const _ARRAY_DTYPES = Dict{String,DataType}(
    "u8" => UInt8, "u32" => UInt32, "i32" => Int32, "f32" => Float32,
)

const _ARRAY_TAGS = Dict{DataType,String}(v => k for (k, v) in _ARRAY_DTYPES)

"Whether this host stores integers little-endian, which is the on-disk order (see [`_write_le`](@ref))."
@inline _islittle() = Base.ENDIAN_BOM == 0x04030201

"""
    _write_le(io, A)

Writes `A`'s elements in **little-endian** order, whatever the host's own order is.

Fixing the byte order rather than inheriting it is what makes a profile portable: these files are
published as release attachments and downloaded by whoever, so a big-endian reader must get the
same numbers. On the overwhelmingly common little-endian host this is a single bulk `write` and
costs nothing; elsewhere it converts element by element.
"""
function _write_le(io::IO, A::AbstractArray{T}) where {T}
    if T === UInt8 || _islittle()
        write(io, A)
    else
        for x in A
            write(io, htol(x))
        end
    end
    nothing
end

"""
    _read_le(bytes, T, n) -> Vector{T}

Inverse of [`_write_le`](@ref): `n` elements of type `T` out of `bytes`.

The length check is not a formality. A truncated or mismatched member would otherwise
`reinterpret` into plausible-looking garbage, and a profile is meant to be verified.
"""
function _read_le(bytes::AbstractVector{UInt8}, ::Type{T}, n::Integer) where {T}
    expected = n * sizeof(T)
    length(bytes) == expected ||
        error("binary profile member has $(length(bytes)) bytes, expected $expected " *
              "($n elements of $T); the file and the manifest disagree")
    A = collect(reinterpret(T, bytes))
    _islittle() || (A = ltoh.(A))
    A
end

"""
    _quantize_u8(x) -> (q::Vector{UInt8}, lo, hi)

Linearly quantizes `x` onto `0:255` over its own observed range.

The range is taken from the data rather than fixed, because that is what makes the step size
follow what is actually stored: the expansion network's cosine distances occupy `[0, 0.938]`
rather than the `[0, 2]` cosine allows, and assuming the theoretical range would throw away a
bit for nothing.

A constant array (`hi == lo`) quantizes to all zeros and dequantizes back to that constant, which
is exact -- so the degenerate case needs no special handling by callers.
"""
function _quantize_u8(x::AbstractArray{<:Real})
    lo, hi = Float64(minimum(x)), Float64(maximum(x))
    span = hi - lo
    q = if span <= 0
        fill(0x00, length(x))
    else
        UInt8[round(UInt8, clamp((Float64(v) - lo) / span, 0, 1) * 255) for v in x]
    end
    q, lo, hi
end

"""
    _dequantize_u8(q, lo, hi) -> Vector{Float32}

Inverse of [`_quantize_u8`](@ref), to `Float32` because that is what every consumer of these
arrays wants and storing more precision than `u8` carried would be a lie about the content.
"""
_dequantize_u8(q::AbstractArray{UInt8}, lo::Real, hi::Real) =
    Float32[Float32(lo + (Float64(b) / 255) * (Float64(hi) - Float64(lo))) for b in q]

"""
    _save_array(dir, name, A; quantize=false) -> Dict

Writes `A` as a binary member `name` of the profile at `dir`, and returns the **manifest entry**
that describes it: file name, dtype tag, shape, and the quantization range when `quantize` is
set.

`quantize` turns a real-valued array into `u8` over its own range. Use it where the consumer's
tolerance is known to exceed the step size -- which is a thing to measure, not to assume; see the
note at the top of this file for the measurement that justified it for expansion distances.

The entry is returned rather than written because the caller owns the manifest's layout: an array
belongs to some artifact, and only the caller knows where under `artifacts` it hangs.
"""
function _save_array(dir::AbstractString, name::AbstractString, A::AbstractArray;
                     quantize::Bool=false)
    entry = Dict{String,Any}("file" => name, "shape" => collect(Int, size(A)))

    if quantize
        q, lo, hi = _quantize_u8(A)
        open(io -> _write_le(io, q), joinpath(dir, name), "w")
        entry["dtype"] = "u8"
        entry["quant"] = Dict{String,Any}("lo" => lo, "hi" => hi)
    else
        T = eltype(A)
        haskey(_ARRAY_TAGS, T) ||
            error("cannot store an array of $T as a binary profile member; supported: " *
                  join(sort(collect(keys(_ARRAY_DTYPES))), ", "))
        open(io -> _write_le(io, A), joinpath(dir, name), "w")
        entry["dtype"] = _ARRAY_TAGS[T]
    end

    entry
end

"""
    _entry(e, key) -> value
    _entry(e, key, default) -> value

Reads `key` out of a manifest entry whichever way it is keyed.

[`_save_array`](@ref) builds a `Dict{String,Any}` and a round trip through JSON3 hands the same
entry back keyed by `Symbol`, so a reader that assumed one of the two worked in production and
failed in a test -- or would have failed the first time anything decoded an entry it had just
built. Accepting both is one line here against a conversion at every call site.
"""
function _entry(e, key::Symbol)
    haskey(e, key) && return e[key]
    haskey(e, String(key)) && return e[String(key)]
    error("binary profile member entry has no '$key' field")
end

_entry(e, key::Symbol, default) =
    haskey(e, key) ? e[key] : (haskey(e, String(key)) ? e[String(key)] : default)

_haskey(e, key::Symbol) = haskey(e, key) || haskey(e, String(key))

"""
    _load_array(read_bytes, entry) -> Array

Reads back what [`_save_array`](@ref) wrote. `read_bytes(name)` fetches a member's raw bytes --
the closure [`_profile_reader`](@ref) returns, which serves a directory and a zip alike.

A quantized member comes back as `Float32`, dequantized; an unquantized one comes back at its
stored type. Either way the shape recorded in the manifest is restored, so a matrix is a matrix.
"""
function _load_array(read_bytes, entry)
    tag = String(_entry(entry, :dtype))
    haskey(_ARRAY_DTYPES, tag) ||
        error("unknown dtype '$tag' in a binary profile member; this build supports " *
              join(sort(collect(keys(_ARRAY_DTYPES))), ", "))
    shape = Int[Int(d) for d in _entry(entry, :shape)]
    A = _read_le(read_bytes(String(_entry(entry, :file))), _ARRAY_DTYPES[tag], prod(shape))

    if _haskey(entry, :quant)
        q = _entry(entry, :quant)
        A = _dequantize_u8(A, Float64(_entry(q, :lo)), Float64(_entry(q, :hi)))
    end

    length(shape) <= 1 ? A : reshape(A, shape...)
end
