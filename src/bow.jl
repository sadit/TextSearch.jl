export TokenData, compute_dict_bow

mutable struct TokenData
    id::Int32
    freq::Int32
    TokenData() = new(0, 0)
    TokenData(a, b) = new(a, b)
end

const UNKNOWN_TOKEN = TokenData(0, 0)

## function compute_bow(config::TextConfig, text::String)
##     voc = Dict{Symbol,TokenData}()
##     voc = compute_bow(config, text, voc)
##     X = [(Symbol(token), idfreq.freq) for (token, idfreq) in voc]
##     sort!(X, by=x->x[1])
##     X
## end
## 
## function compute_bow(config::TextConfig, arr::AbstractVector{String})
##     voc = Dict{Symbol,TokenData}()
## 
## 	for text in arr
## 		compute_bow(config, text, voc)
##     end
##     
## 	X = [(Symbol(token), idfreq.freq) for (token, idfreq) in voc]
##     sort!(X, by=x->x[1])
## 	voc
## end

function compute_dict_bow(config::TextConfig, text::String, voc::Dict{Symbol,TokenData})
    for token in tokenize(config, text)
        sym = Symbol(token)
        h = get(voc, sym, UNKNOWN_TOKEN)
        if h.freq == 0
            voc[sym] = TokenData(length(voc), 1)
        else
            h.freq += 1
        end
    end

    voc
end

compute_dict_bow(config::TextConfig, text::String) = compute_dict_bow(config, text, Dict{Symbol,TokenData}())

function compute_dict_bow(config::TextConfig, arr::AbstractVector{String})
    D = Dict{Symbol,TokenData}()
    for text in arr
       compute_dict_bow(config, text, D)
    end

    D
end