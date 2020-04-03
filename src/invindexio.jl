# This file is a part of TextSearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

"""
    save(f::IO, invindex::InvIndex)

Writes `invindex` to the given stream
"""
function save(f::IO, invindex::InvIndex)
    write(f, invindex.n)
    write(f, length(invindex.lists))
    for (termID, lst) in invindex.lists
        println(f, string(termID))
        write(f, length(lst))
        for p in lst
            write(f, p.id, p.weight)
        end
    end
end

"""
    load(f::IO, ::Type{InvIndex})

Reads an inverted index from the given file
"""
function load(f::IO, ::Type{InvIndex})
    invindex = InvIndex()
    invindex.n = read(f, Int)
    m = read(f, Int)
    for i in 1:m
        termID = parse(Int, readline(f))
        l = read(f, Int)
        lst = Vector{IdWeight}(undef, l)
        invindex.lists[termID] = lst
        for j in 1:l
            id = read(f, Int)
            weight = read(f, Float64)
            lst[j] = IdWeight(id, weight)
        end
    end

    invindex
end
