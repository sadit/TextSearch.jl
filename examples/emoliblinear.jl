using Pkg
pkg"activate ."
pkg"add https://github.com/sadit/SimilaritySearch.jl https://github.com/sadit/TextSearch.jl https://github.com/sadit/KernelMethods.jl LIBLINEAR Random StatsBase"
using SimilaritySearch, TextSearch, LIBLINEAR, Random, StatsBase, KernelMethods
url = "http://ingeotec.mx/~sadit/emospace50k.json.gz"
!isfile(basename(url)) && download(url, basename(url))
db = loadtweets(basename(url))
n = length(db)
G = shuffle(1:n)

const P1 = G[1:div(length(G), 2)]
const P2 = G[div(length(G), 2)+1:end]

corpus = get.(db, "text", "")
labels = get.(db, "klass", "")
config = TextConfig(qlist=[3,5], nlist=[1], group_emo=true)  # emoticons must be removed

function tfidf_vectors(corpus, labels)
    model = fit(VectorModel, config, corpus[P1])
    X = [vectorize(model, TfidfModel, text) for text in corpus]
    X[P1], X[P2], labels[P1], labels[P2]
end

function entropy_vectors(corpus, labels)
    le = fit(LabelEncoder, labels)
    model = fit(EntModel, config, corpus[P1], KernelMethods.transform.(le, labels[P1]),smooth=9)
    model = prune_select_top(model, 0.2)
    @info "number-of-tokens:" length(model.tokens)

    # X = [vectorize(model, TfidfModel, text) for text in corpus]
    X = [vectorize(model, text) for text in corpus]
    X[P1], X[P2], labels[P1], labels[P2]
end

Xtrain, Xtest, ytrain, ytest = entropy_vectors(corpus, labels)
# Xtrain, Xtest, ytrain, ytest = tfidf_vectors(corpus, labels)

model = linear_train(labels[P1], hcat(Xtrain...), C=0.1)
predictions, decision_values = linear_predict(model, hcat(Xtest...))
@info "Accuracy:" mean(ytest .== predictions)
