export Rocchio, RocchioBagging, fit, predict, transform
import SimilaritySearch: KnnResult
using StatsBase: countmap

mutable struct Rocchio
    protos::Vector{SVEC}  # prototypes
    pops::Vector{Int} # population per class
end

const RocchioBagging = Vector{Rocchio}

function fit(::Type{Rocchio}, X::AbstractVector{A}, y::AbstractVector{I}; nclasses=0) where {I<:Integer,A<:DVEC}
    if nclasses == 0
        nclasses = unique(y) |> length
    end

    prototypes = [SVEC() for i in 1:nclasses]
    populations = zeros(Int, nclasses)
    for i in 1:length(X)
        c = y[i]
        add!(prototypes[c], X[i])
        populations[c] += 1
    end

    n = length(X[1])
    for p in prototypes
        normalize!(p)
    end
    Rocchio(prototypes, populations)
end

function predict(rocchio::Rocchio, x::SVEC)
    res = KnnResult(1)
    for i in 1:length(rocchio.protos)
        d = cosine_distance(rocchio.protos[i], x)
        push!(res, i, d)
    end

    first(res).objID
end

function transform(rocchio::Rocchio, x::SVEC)
    [dot(rocchio.protos[i], x) for i in 1:length(rocchio.protos)]
end

function broadcastable(rocchio::Rocchio)
    (rocchio,)
end

function fit(::Type{RocchioBagging}, X::AbstractVector, y::AbstractVector{Int}, nrocchios=5; nclasses=0)::RocchioBagging
    if nclasses == 0
        nclasses = unique(y) |> length
    end

    s = ceil(Int, length(X) / 2)
    L = RocchioBagging()
    for i in 1:nrocchios
        I = rand(1:length(X), s)
        rocchio = fit(Rocchio, X[I], y[I], nclasses=nclasses)
        push!(L, rocchio)
    end

    L
end

function predict(rocchio::RocchioBagging, x)
    P = [predict(r, x) for r in rocchio] |> countmap |> collect
    sort!(P, by=x->x[end])
    P[end][1]
end

function transform(rocchio::RocchioBagging, x)
    sum(transform(r, x) for r in rocchio) |> normalize!
end

function broadcastable(rocchio::RocchioBagging)
    (rocchio,)
end
