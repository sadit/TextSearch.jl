# This file is a part of TextSearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

export neardup

function neardup(X::AbstractVector{T}, α=0.1) where T
    idx = InvertedFile()
    res = KnnResult(1)
    L = zeros(Int, length(X))
    D = zeros(Float64, length(X))
    L[1] = 1
    D[1] = 0.0
    push!(idx, 1 => X[1])
    @inbounds for i in 2:length(X)
        empty!(res)
        x = X[i]
        search(idx, x, res)
        if length(res) == 0 || minimum(res) > α
            push!(idx, i => x)
            L[i] = i
            D[i] = 0.0
        else
            L[i], D[i] = first(res)
        end
    end

    (idx=idx, nn=L, dist=D)
end