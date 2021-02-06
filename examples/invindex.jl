### A Pluto.jl notebook ###
# v0.12.20

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

# ╔═╡ fb25ee14-5f4e-11eb-1c8e-6d292c20388a
md"""
# Construction of an inverted index for searching in a message dataset


Firstly, we need to add the required packages, SimilaritySearch and TextSearch are the core for use, but others are required to show a full demostration.
"""

# ╔═╡ 0c25b8a2-5f4f-11eb-3410-f5560725ed17
md"""
## Loading thre dataset
"""

# ╔═╡ 547e44b2-5f08-11eb-128c-61c32cdb2ee7
begin
	url = "http://ingeotec.mx/~sadit/emospace50k.json.gz"
	!isfile(basename(url)) && download(url, basename(url))
	db = open(basename(url)) do stream
		[JSON.parse(line) for line in eachline(GzipDecompressorStream(stream))]
	end
end


# ╔═╡ 545269a4-5f4f-11eb-0f7d-01adcaf452a0
md"""
## Preprocessing and creating a text model
"""

# ╔═╡ 52bbc1a8-5f4f-11eb-1308-39c8a8a651b0
begin
	
	# you can use a number of tokenizers, here we use character q-grams to improve support for informal writing
	config = TextConfig(qlist=[3], nlist=[1], group_usr=true, group_url=true)
	corpus = [t["text"] for t in db]
	model = VectorModel(TfidfWeighting(), compute_bow_multimessage(config, corpus))
end

# ╔═╡ 7cf28d6c-5f4f-11eb-0101-63bc0ff84e94
md"""
## Encoding messages as vectors
"""

# ╔═╡ 6cceec0a-5f4f-11eb-0727-6d333d16196c
X = [vectorize(model, compute_bow(config, text)) for text in corpus];

# ╔═╡ 95ea6094-5f4f-11eb-0d62-5b0407fb63a3
md"## Construction of the dataset"

# ╔═╡ 8dc9b872-5f4f-11eb-254d-2ddb4b4b53e7
invindex = InvIndex(X);

# ╔═╡ 9f6f6568-5f4f-11eb-2a25-8d5c2aa35f76
md"## Searching (with some Pluto controls)"

# ╔═╡ 010ec74a-5f0b-11eb-1c3a-f327bad5b02f
begin
	inputquery = @bind querytext TextField(default="la mejor música!!!")  # best music
	inputk = @bind ksearch Slider(1:31, 3, true)

	md"""
search for:
$(inputquery) k: $(inputk)
"""
end

# ╔═╡ b32bfc10-5f4f-11eb-3e9c-05de9cc7d34f
md"## Results "

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
# ╠═fb25ee14-5f4e-11eb-1c8e-6d292c20388a
# ╠═11f6ef4a-5f08-11eb-0ba1-05a1a1ff205a
# ╠═0c25b8a2-5f4f-11eb-3410-f5560725ed17
# ╠═547e44b2-5f08-11eb-128c-61c32cdb2ee7
# ╠═545269a4-5f4f-11eb-0f7d-01adcaf452a0
# ╠═52bbc1a8-5f4f-11eb-1308-39c8a8a651b0
# ╠═7cf28d6c-5f4f-11eb-0101-63bc0ff84e94
# ╠═6cceec0a-5f4f-11eb-0727-6d333d16196c
# ╠═95ea6094-5f4f-11eb-0d62-5b0407fb63a3
# ╠═8dc9b872-5f4f-11eb-254d-2ddb4b4b53e7
# ╠═9f6f6568-5f4f-11eb-2a25-8d5c2aa35f76
# ╠═010ec74a-5f0b-11eb-1c3a-f327bad5b02f
# ╠═b32bfc10-5f4f-11eb-3e9c-05de9cc7d34f
# ╠═92d0c64e-5f09-11eb-0d65-05a484984b63
