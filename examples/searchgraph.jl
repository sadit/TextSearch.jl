### A Pluto.jl notebook ###
# v0.14.2

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ 11f6ef4a-5f08-11eb-0ba1-05a1a1ff205a
using SimilaritySearch, TextSearch, JSON, CodecZlib, PlutoUI, StatsBase

# ╔═╡ aeb6bb3a-5f4e-11eb-3b36-070b3356e64d
md"""
# Creating a generic similarity search index with text based vectors

This example shows how to perform knn searches using a `SearchGraph` index (defined in `SimilaritySearch.jl`) over a text dataset.
"""

# ╔═╡ 547e44b2-5f08-11eb-128c-61c32cdb2ee7
begin
	url = "https://github.com/sadit/TextClassificationTutorial/raw/main/data/emo50k.json.gz"
	!isfile(basename(url)) && download(url, basename(url))
	db = open(basename(url)) do stream
		[JSON.parse(line) for line in eachline(GzipDecompressorStream(stream))]
	end
	
	# you can use a number of tokenizers, here we use character q-grams to improve support for informal writing
	tok = Tokenizer(TextConfig(qlist=[4], nlist=[1], del_diac=true, group_usr=true, group_url=true))
	corpus = [t["text"] for t in db]
	model = VectorModel(IdfWeighting(), TfWeighting(), compute_bow_corpus(tok, corpus))
	X = vectorize_corpus(tok, model, corpus)
	nothing
end

# ╔═╡ 67fa1b66-5f4d-11eb-01b8-0d9b99c60386
md"""
## Creating the index

This example uses a different index, `SearchGraph` which has an slower construction but it is much more versatile than InvIndex. Here, we use it just as an example.
"""

# ╔═╡ f8bb66c4-5f4c-11eb-1f2e-4100b9d49eb1
index =	SearchGraph(CosineDistance(), X;
		search_algo=BeamSearch(4),
		neighborhood_algo=LogNeighborhood(3),
		automatic_optimization=false
	)


# ╔═╡ a873c5de-5f4d-11eb-3b7c-69b56d88d7ef
md"""
In my computer, an icore 7 processor, it requires close to 3 min. (single core construction).
"""

# ╔═╡ 010ec74a-5f0b-11eb-1c3a-f327bad5b02f
begin
	inputquery = @bind querytext TextField(default="la mejor música!!!")  # best music
	inputk = @bind ksearch Slider(1:31, 3, true)

	md"""
search for:
$(inputquery) k: $(inputk)
"""
end

# ╔═╡ 92d0c64e-5f09-11eb-0d65-05a484984b63
begin
	q = vectorize(model, compute_bow(tok, querytext))
	
	with_terminal() do
		res = search(index, q, ksearch)
		freqs = countmap([db[p.id]["klass"] for p in res]) |> collect
		sort!(freqs, by=x -> x[2], rev=true)
		println("emojis: ", join(["$(p[1]):$(p[2])" for p in freqs], ", "), "\n")
		for i in 1:max(1, length(res))
			p = res[i]
			u = db[p.id]
			text = u["text"] # join(normalize_text(config, u["text"]))
			println("$i (", u["klass"], ", ", round(p.dist, digits=3) , ") => ", text, "\n")
		end
	end
end

# ╔═╡ Cell order:
# ╠═aeb6bb3a-5f4e-11eb-3b36-070b3356e64d
# ╠═11f6ef4a-5f08-11eb-0ba1-05a1a1ff205a
# ╠═547e44b2-5f08-11eb-128c-61c32cdb2ee7
# ╠═67fa1b66-5f4d-11eb-01b8-0d9b99c60386
# ╠═f8bb66c4-5f4c-11eb-1f2e-4100b9d49eb1
# ╠═a873c5de-5f4d-11eb-3b7c-69b56d88d7ef
# ╠═010ec74a-5f0b-11eb-1c3a-f327bad5b02f
# ╠═92d0c64e-5f09-11eb-0d65-05a484984b63
