### A Pluto.jl notebook ###
# v0.12.18

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

# ╔═╡ 547e44b2-5f08-11eb-128c-61c32cdb2ee7
begin
	url = "http://ingeotec.mx/~sadit/emospace50k.json.gz"
	!isfile(basename(url)) && download(url, basename(url))
	db = open(basename(url)) do stream
		[JSON.parse(line) for line in eachline(GzipDecompressorStream(stream))]
	end
	
	# you can use a number of tokenizers, here we use character q-grams to improve support for informal writing
	config = TextConfig(qlist=[3], nlist=[1], group_usr=true, group_url=true)
	corpus = [t["text"] for t in db]
	model = VectorModel(TfidfWeighting, compute_bow_multimessage(config, corpus))
	X = [vectorize(model, compute_bow(config, text)) for text in corpus]
	invindex = InvIndex(X)
end

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
	q = vectorize(model, compute_bow(config, querytext))
	
	with_terminal() do
		res = search(invindex, q, ksearch)
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
# ╠═11f6ef4a-5f08-11eb-0ba1-05a1a1ff205a
# ╠═547e44b2-5f08-11eb-128c-61c32cdb2ee7
# ╠═010ec74a-5f0b-11eb-1c3a-f327bad5b02f
# ╠═92d0c64e-5f09-11eb-0d65-05a484984b63
