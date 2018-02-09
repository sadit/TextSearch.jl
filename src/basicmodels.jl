export TextModel, VectorModel, fit!, inverse_vbow
    compute_bow, vectorize, vectorize_tfidf,
    vectorize_tf, vectorize_idf, vectorize_rawfreq, id2token

abstract type Model end

mutable struct VectorModel <: Model
    voc::Dict{String,WeightedToken}
    size::Int64
    filter_low::Int
    filter_high::Float64
    config::TextConfig
end

#
# function save(ostream, model::VectorModel)
#     M = Dict{String,Any}(
#         "length" => length(model.voc),
#         "size" => model.size,
#         "filter_low" => model.filter_low,
#         "filter_high" => model.filter_high,
#     )
#     write(ostream, JSON.json(M), "\n")
#     save(ostream, model.config)
#     for (k, v) in model.voc
#         write(ostream, JSON.json((k, v.id, v.freq)), "\n")
#     end
# end
#
# function load(istream, ::Type{VectorModel})
#     m = JSON.parse(readline(istream))
#     config = load(istream, TextConfig)
#     voc = Dict{String, WeightedToken}()
#     for i in 1:m["length"]
#         k, id, freq = JSON.parse(readline(istream))
#         voc[k] = WeightedToken(id, freq)
#     end
#     VectorModel(voc, m["size"], m["filter_low"], m["filter_high"], config)
# end

function id2token(model::VectorModel)
    m = Dict{Int,String}()
    for (word, token) in model.voc
        m[token.id] = word
    end

    m
end

function inverse_vbow(vec, vocmap)
    s = collect(vec.tokens)
    sort!(s, by=x -> -x.weight)
    [(vocmap[token.id], token.weight) for token in s]
end

VectorModel() = VectorModel(Dict{String, WeightedToken}(), 0, 1, 1.0, TextConfig())

function VectorModel(config::TextConfig)
    model = VectorModel()
    model.config = config
    model
end

function fit!(model::VectorModel, corpus; get_text::Function=identity)
    V = Dict{String,Int}()
    for item in corpus
        text = get_text(item)
        voc = compute_bow(text, model.config)
        for (token, freq) in voc
            V[token] = get(V, token, 0) + freq
        end
        model.size += 1
        if model.size % 10000 == 1
            info("advance VectorModel: $(model.size) processed items")
        end
    end

    info("finished VectorModel: $(model.size) processed items")

    for (token, freq) in V
        if freq < model.filter_low || freq > model.filter_high * model.size
            continue
        end

        model.voc[token] = WeightedToken(length(model.voc)+1, freq)
    end
end

function compute_bow(text::String, config::TextConfig)
    voc = Dict{String,Int}()
    for token in tokenize(text, config)
        freq = get(voc, token, 0) + 1
        voc[token] = freq
    end

    voc
end

function maxfreq(bow)
    maxfreq_ = 0
    for freq in values(bow)
        if maxfreq_ < freq
            maxfreq_ = freq
        end
    end

    maxfreq_
end

function weighted_vector(weighting_function::Function, text::String, model::VectorModel; corrector::Function=identity, maxlength=typemax(Int))
    vec = WeightedToken[]
    bow = compute_bow(text, model.config)
    maxfreq_ = maxfreq(bow)

    for (token, freq) in bow
        tokendata = try
            token = corrector(token)
            model.voc[token]
        catch err
            continue
        end

        w = weighting_function(freq, maxfreq_, model.size, tokendata.weight)
        push!(vec, WeightedToken(tokendata.id, w))
    end

    if length(vec) > maxlength
        sort!(vec, by=(x) -> -x.weight)
        vec = vec[1:maxlength]
    end

    sort!(vec, by=(x) -> x.id)
    VBOW(vec)
end

function _tfidf(freq, maxfreq, N, freqToken)
    (freq / maxfreq) * log(N / freqToken)
end

function _tf(freq, maxfreq, N, freqToken)
    (freq / maxfreq)
end

function _idf(freq, maxfreq, N, freqToken)
    log(N / freqToken)
end

function _rawfreq(freq, maxfreq, N, freqToken)
    freq
end

function vectorize_tfidf(text::String, model::VectorModel; corrector::Function=identity, maxlength=typemax(Int))
    weighted_vector(_tfidf, text, model, maxlength=maxlength, corrector=corrector)
end

function vectorize_tf(text::String, model::VectorModel; corrector::Function=identity, maxlength=typemax(Int))
    weighted_vector(_tf, text, model, maxlength=maxlength, corrector=corrector)
end

function vectorize_idf(text::String, model::VectorModel; corrector::Function=identity, maxlength=typemax(Int))
    weighted_vector(_idf, text, model, maxlength=maxlength, corrector=corrector)
end

function vectorize_rawfreq(text::String, model::VectorModel; corrector::Function=identity, maxlength=typemax(Int))
    weighted_vector(_rawfreq, text, model, maxlength=maxlength, corrector=corrector)
end

function vectorize(text::String, model::VectorModel; corrector::Function=identity, maxlength=typemax(Int))
    vectorize_tfidf(text, model, corrector=corrector, maxlength=maxlength)
end

function vectorize(textlist::AbstractVector{String}, model::Model; corrector::Function=identity)
    [vectorize(text, model, corrector=corrector) for text in textlist]
end
