# TextSearch.jl

[![Build Status](https://travis-ci.org/sadit/TextSearch.jl.svg?branch=master)](https://travis-ci.org/sadit/TextSearch.jl)
[![Coverage Status](https://coveralls.io/repos/github/sadit/TextSearch.jl/badge.svg?branch=master)](https://coveralls.io/github/sadit/TextSearch.jl?branch=master)
[![codecov.io](http://codecov.io/github/sadit/TextSearch.jl/coverage.svg?branch=master)](http://codecov.io/github/sadit/TextSearch.jl?branch=master)


TextSearch.jl is a package to create vector representations of text, mostly, independently of the language. It is intended to be used with [SimilaritySearch.jl](https://github.com/sadit/SimilaritySearch.jl), but can be used independetly if needed.

For generic text analysis you should use other packages like [TextAnalysis.jl](https://github.com/johnmyleswhite/TextAnalysis.jl).

It supports a number of simple text preprocessing functions, and three different kinds of tokenizers, i.e., word n-grams, character q-grams, and skip-grams. It supports creating multisets of tokens, commonly named bag of words (BOW). TextSearch.jl can produce sparse vector representations based on term-weighting schemes like TF, IDF, and TFIDF. It also supports term-weighting schemes designed to cope text classification tasks, mostly based on distributional representations.

# Installing SimilaritySearch


You may install the package as follows
```bash
julia -e 'using Pkg; pkg"add https://github.com/sadit/TextSearch.jl"'
```
also, you can run the set of tests as follows
```bash
julia -e 'using Pkg; pkg"test TextSearch"'
```

## Using the library
```julia
julia> using SimilaritySearch, TextSearch
julia> url = "http://ingeotec.mx/~sadit/emospace50k.json.gz"
julia> !isfile(basename(url)) && download(url, basename(url))
julia> db = loadtweets(basename(url))
# you can use a number of tokenizers, here we use character q-grams to improve support for informal writing
julia> config = TextConfig(qlist=[4], nlist=[])
julia> corpus = [t["text"] for t in db]
julia> model = fit(VectorModel, config, corpus)
julia> invindex = InvIndex()
julia> for (i, text) in enumerate(corpus)
        push!(invindex, i, weighted_bow(model, TfidfModel, text, norm=true))
    end
```

queries are made as follows
```julia
julia> q = weighted_bow(model, TfidfModel, "que chida musica!!!", norm=true)
julia> db[[p.objID for p in search(invindex, q, KnnResult(11))]]
```

you can save memory by pruning large lists, as follows
```julia
julia> invindex = prune(invindex, 100)
julia> for p in search(invindex, weighted_bow(model, TfidfModel, "que chida musica!!!", norm=true), KnnResult(11))
    println(db[p.objID]["klass"], "\t", db[p.objID]["text"])
end
```
in some cases this can improve results since it keeps the most weighted items per list.

It is also simple to modify the bag of words to apply query expansion, downsampling, error correction, etc.
```julia
julia> function randomsample!(bow)
        Dict(rand(bow, div(length(bow), 2)))
    end
julia> for p in search(invindex, weighted_bow(model, TfidfModel, "que chida musica!!!", randomsample!, norm=true), KnnResult(11))
    println(db[p.objID]["klass"], "\t", db[p.objID]["text"])
end
😎	No me toquen ando chida! 😎 https://t.co/39OKexhGFT
🙏	Díganme películas chidas para ver 🙏🏼
😋	Me cae bien mi vecino por que siempre pone canciones chidas😋
😉	Esta si esta chida para ir a la alameda los domingos 😉 https://t.co/vRExWJhOGH
😐	Me va a quedar bien chida la falda ... 😐 https://t.co/YV3sfBAjqD
😒	De chiquito cantaba chido😒
🤓	Se ve que se va a poner muy chida la Jornada. 🤓
💙	¡Qué chido está Pachuca! 💙
😢	Siento que en MARCO una chava me tomó una foto chida y nunca la subieron 😢
😥	El problema de ponerle fin a las relaciones es que también te separas de personas bien chidas que valen la pena 😥
😜	#BuenMartes #gentechida a darle con todo que ya sólo falta un día después de pasado mañana para que llegue el viernes!! 😜
```