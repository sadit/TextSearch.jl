export neardup

function neardup(X::AbstractVector{T}, epsilon=0.1) where T
    invindex = InvIndex()
    res = KnnResult(1)
    L = zeros(Int, length(X))
    push!(invindex, 1, X[1])
    for i in 2:length(X)
        empty!(res)
        x = X[i]
        search(invindex, cosine_distance, x, res)
        if length(res) == 0 || first(res).dist > epsilon
            push!(invindex, i, x)
            L[i] = i
        else
            L[i] = first(res).objID
        end
    end

    L
end