# This file is part of Intersections.jl

export binarysearch, doublingsearch, doublingsearchrev, seqsearch, seqsearchrev

"""
 	binarysearch(A, x, sp=1, ep=length(A))

Finds the insertion position of `x` in `A` in the range `sp:ep`
"""
function binarysearch(A, x, sp = 1, ep = length(A))::Int
    while sp < ep
        mid = div(sp + ep, 2)
        if x <= getkey(A, mid)
            ep = mid
        else
            sp = mid + 1
        end
    end

    x <= getkey(A, sp) ? sp : sp + 1
end

"""
 	doublingsearch(A, x, sp=1, ep=length(A))

Finds the insertion position of `x` in `A`, starting at `sp`
"""
function doublingsearch(A, x, sp = 1, ep = length(A))::Int
    p = 0
    i = 1

    while sp + i <= ep && getkey(A, sp + i) < x
        p = i
        i += i
    end

    binarysearch(A, x, sp + p, min(ep, sp + i))
end


"""
 	doublingsearchrev(A, x, sp=1, ep=length(A))

Finds the insertion position of `x` in `A`, starting at the end
"""
function doublingsearchrev(A, x, sp = 1, ep = length(A))::Int
    x > getkey(A, ep) && return ep + 1

    i = 1
    p = ep
    while p >= sp && x <= getkey(A, p)
        ep = p
        p -= i
        i += i
    end

    binarysearch(A, x, max(p, sp), ep)
end

"""
 	seqsearchrev(A, x, sp=1, ep=length(A))

Reverse sequential search, i.e., it starts from `ep` to `sp`
"""
function seqsearchrev(A, x, sp = 1, ep = length(A))::Int
    pos = ep + 1
    while pos > sp && x <= getkey(A, pos - 1)
        pos -= 1
    end

    pos
end

"""
 	searchrev(A, x, sp=1, ep=length(A))

Sequential search, i.e., it starts from `sp` to `ep`
"""
function seqsearch(A, x, sp = 1, ep = length(A))::Int
    x > getkey(A, ep) && return ep + 1
    while sp <= ep && getkey(A, sp) < x
        sp += 1
    end

    sp > ep ? ep : sp
end

