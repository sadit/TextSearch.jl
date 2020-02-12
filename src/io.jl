# This file is a part of TextSearch.jl
# License is Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

export iterlines, itertweets, loadtweets

import GZip
import JSON

"""
   iterlines(fun, filename; maxlines=typemax(Int))

Calls `fun` over each line of the given filename, skipping empty lines. It supports gzip compressed inputs.

"""
function iterlines(fun, filename::String; maxlines=typemax(Int))
    if endswith(filename, ".gz")
        f = GZip.open(filename)
        i = 0
        while !eof(f) && i < maxlines
            line = readline(f)
            i += 1
            if length(line) == 0
                continue
            end
            fun(line)
        end
        close(f)
    else
        open(filename) do f
            i = 0
            while !eof(f) && i < maxlines
                line = readline(f)
                i += 1
                fun(line)
            end
        end
    end
end


"""
   parsetweet(line) -> Dict{String,Any}

Parses a json-dictionary string, which is the assumed object format.
It supports two formats:

- a raw json-dictionaries
- a dictionary prefixed by some prefix key, that is, key<tab>json-dictionary

"""
function parsetweet(line)
    JSON.parse(line)
end

"""
   itertweets(fun, filename::String; maxlines=typemax(Int))
   itertweets(fun, file; maxlines=typemax(Int))

Calls `fun` for each object in the file. One object per line using
the format supported by `parsetweet`

"""
function itertweets(fun, filename::String; maxlines=typemax(Int))
    iterlines(filename, maxlines=maxlines) do line
        tweet = parsetweet(line)
        fun(tweet)
    end
end

function itertweets(fun, file; maxlines=typemax(Int))
    i = 0
    while !eof(file) && i < maxlines
        line = readline(file)
        i += 1
        try
            tweet = parsetweet(line)
            fun(tweet)
        catch
            continue
        end
    end
end

"""
    loadtweets(f; maxlines=typemax(Int))

loads a list of tweets from a file using `itertweets`
"""
function loadtweets(f; maxlines=typemax(Int))
    L = Dict[]
    itertweets(f, maxlines=maxlines) do t
        push!(L, t)
    end
    L
end
