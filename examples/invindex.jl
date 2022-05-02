### A Pluto.jl notebook ###
# v0.17.7

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 11f6ef4a-5f08-11eb-0ba1-05a1a1ff205a
using SimilaritySearch, TextSearch, InvertedFiles, JSON, CodecZlib, PlutoUI, StatsBase

# ╔═╡ fb25ee14-5f4e-11eb-1c8e-6d292c20388a
md"""
# Construction of an inverted index for searching in a message dataset


Firstly, we need to add the required packages, SimilaritySearch and TextSearch are the core for use, but others are required to show a full demonstration.
"""

# ╔═╡ 1721b373-e893-4fbc-8878-f6e14d96ec7c


# ╔═╡ 0c25b8a2-5f4f-11eb-3410-f5560725ed17
md"""
## Loading the dataset
"""

# ╔═╡ 547e44b2-5f08-11eb-128c-61c32cdb2ee7
begin
	url = "https://github.com/sadit/TextClassificationTutorial/raw/main/data/emo50k.json.gz"
	!isfile(basename(url)) && download(url, basename(url))
	db = open(basename(url)) do stream
		[JSON.parse(line) for line in eachline(GzipDecompressorStream(stream))]
	end
	
	md"""
	loaded $(length(db)) messages
	"""
end


# ╔═╡ 545269a4-5f4f-11eb-0f7d-01adcaf452a0
md"""
## Preprocessing and creating a text model
"""

# ╔═╡ 52bbc1a8-5f4f-11eb-1308-39c8a8a651b0
begin
	
	# you can use a number of tokenizers, here we use character q-grams to improve support for informal writing
	tok = Tokenizer(TextConfig(qlist=[3], nlist=[1], group_usr=true, group_url=true, del_diac=true, del_punc=true, group_emo=false))
	corpus = [t["text"] for t in db]
	model = VectorModel(IdfWeighting(), TfWeighting(), tok, corpus; mindocs=3)
end

# ╔═╡ 6dc5265a-f5ae-410d-b18b-5433f6248ae9


# ╔═╡ 7cf28d6c-5f4f-11eb-0101-63bc0ff84e94
md"""
## Encoding messages as vectors
"""

# ╔═╡ 6cceec0a-5f4f-11eb-0727-6d333d16196c
X = vectorize_corpus(model, tok, corpus);

# ╔═╡ 95ea6094-5f4f-11eb-0d62-5b0407fb63a3
md"## Construction of the dataset"

# ╔═╡ 8dc9b872-5f4f-11eb-254d-2ddb4b4b53e7
begin
	invindex = WeightedInvertedFile(vocsize(model))
	append!(invindex, VectorDatabase(X))
end

# ╔═╡ 9f6f6568-5f4f-11eb-2a25-8d5c2aa35f76
md"## Searching (with some Pluto controls)"

# ╔═╡ 010ec74a-5f0b-11eb-1c3a-f327bad5b02f
begin
	inputquery = @bind querytext TextField(default="me queda perfecta la canción!!!")  # best music
	inputk = @bind ksearch Slider(1:31, 7, true)

	md"""
search for:
$(inputquery) k: $(inputk)
"""
end

# ╔═╡ b32bfc10-5f4f-11eb-3e9c-05de9cc7d34f
md"## Results "

# ╔═╡ 92d0c64e-5f09-11eb-0d65-05a484984b63
begin
	q = vectorize(model, tok, querytext)
	
	with_terminal() do
		res = KnnResult(ksearch)
		search(invindex, q, res)
		freqs = countmap([db[id_]["klass"] for id_ in res.id]) |> collect
		sort!(freqs, by=x -> x[2], rev=true)
		println("emojis: ", join(["$(p[1]):$(p[2])" for p in freqs], ", "), "\n")
		for i in 1:length(res)
			u = db[res.id[i]]
			text = u["text"] # join(normalize_text(config, u["text"]))
			println("$i (", u["klass"], ", ", round(res.dist[i], digits=3) , ") => ", text, "\n")
		end
	end
end

# ╔═╡ 82647ddd-2a98-4edd-8649-79db649e0540


# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CodecZlib = "944b1d66-785c-5afd-91f1-9de20f533193"
InvertedFiles = "b20bd276-2c70-11ec-161a-3d1e1109a1c3"
JSON = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
SimilaritySearch = "053f045d-5466-53fd-b400-a066f88fe02a"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
TextSearch = "7f6f6c8a-3b03-11e9-223d-e7d88259bd6c"

[compat]
CodecZlib = "~0.7.0"
InvertedFiles = "~0.3.0"
JSON = "~0.21.3"
PlutoUI = "~0.7.38"
SimilaritySearch = "~0.8.16"
StatsBase = "~0.33.16"
TextSearch = "~0.11.1"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[CategoricalArrays]]
deps = ["DataAPI", "Future", "Missings", "Printf", "Requires", "Statistics", "Unicode"]
git-tree-sha1 = "109664d3a6f2202b1225478335ea8fea3cd8706b"
uuid = "324d7699-5711-5eae-9e2f-1d82baa6b597"
version = "0.10.5"

[[ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "9950387274246d08af38f6eef8cb5480862a435f"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.14.0"

[[ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "bf98fa45a0a4cee295de98d4c1462be26345b9a1"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.2"

[[CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "ded953804d019afa9a3f98981d99b33e3db7b6da"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.0"

[[ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "024fe24d83e4a5bf5fc80501a314ce0d1aa35597"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.0"

[[Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "b153278a25dd42c65abbf4e62344f9d22e59191b"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.43.0"

[[CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[DataAPI]]
git-tree-sha1 = "fb5f5316dd3fd4c5e7c30a24d50643b73e37cd40"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.10.0"

[[DataFrames]]
deps = ["Compat", "DataAPI", "Future", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrettyTables", "Printf", "REPL", "Reexport", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "6c19003824cbebd804a51211fd3bbd81bf1ecad5"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.3.3"

[[DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "3daef5523dd2e769dad2365274f760ff5f282c7d"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.11"

[[DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[Distances]]
deps = ["LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "3258d0659f812acde79e8a74b11f17ac06d0ca04"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.7"

[[Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "b19534d1895d702889b219c382a6e18010797f0b"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.6"

[[Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[HypertextLiteral]]
git-tree-sha1 = "2b078b5a615c6c0396c77810d92ee8c6f470d238"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.3"

[[IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[Intersections]]
deps = ["Random"]
git-tree-sha1 = "f15f3331c09e83c5d086b41b8b1bf987331a76a1"
uuid = "6d0fbc77-481e-44fe-9ace-1653ddcdea34"
version = "0.2.8"

[[InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "91b5dcf362c5add98049e6c29ee756910b03051d"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.3"

[[InvertedFiles]]
deps = ["Dates", "Intersections", "KCenters", "LinearAlgebra", "Parameters", "Random", "SearchModels", "SimilaritySearch", "SparseArrays", "StatsBase"]
git-tree-sha1 = "f9fd88e14769f58d0bad3fef6c6be9dd1661ed93"
uuid = "b20bd276-2c70-11ec-161a-3d1e1109a1c3"
version = "0.3.0"

[[InvertedIndices]]
git-tree-sha1 = "bee5f1ef5bf65df56bdd2e40447590b272a5471f"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.1.0"

[[IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

[[KCenters]]
deps = ["CategoricalArrays", "DelimitedFiles", "Distances", "LinearAlgebra", "MLDataUtils", "Random", "SimilaritySearch", "StatsBase", "Test"]
git-tree-sha1 = "80e3cee59c8637f327cace1c4d6ffb104c4452e3"
uuid = "5d8de97f-65f8-4dd6-a15b-0f89c36a43ce"
version = "0.6.1"

[[LearnBase]]
deps = ["LinearAlgebra", "StatsBase"]
git-tree-sha1 = "47e6f4623c1db88570c7a7fa66c6528b92ba4725"
uuid = "7f8f8fb0-2700-5f03-b4bd-41f8cfc144b6"
version = "0.3.0"

[[LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "76c987446e8d555677f064aaac1145c4c17662f8"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.14"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[MLDataPattern]]
deps = ["LearnBase", "MLLabelUtils", "Random", "SparseArrays", "StatsBase"]
git-tree-sha1 = "e99514e96e8b8129bb333c69e063a56ab6402b5b"
uuid = "9920b226-0b2a-5f5f-9153-9aa70a013f8b"
version = "0.5.4"

[[MLDataUtils]]
deps = ["DataFrames", "DelimitedFiles", "LearnBase", "MLDataPattern", "MLLabelUtils", "Statistics", "StatsBase"]
git-tree-sha1 = "ee54803aea12b9c8ee972e78ece11ac6023715e6"
uuid = "cc2ba9b6-d476-5e6d-8eaf-a92d5412d41d"
version = "0.5.4"

[[MLLabelUtils]]
deps = ["LearnBase", "MappedArrays", "StatsBase"]
git-tree-sha1 = "fd75d4b0c4016e047bbb6263eecf7ae3891af522"
uuid = "66a33bbf-0c2b-5fc8-a008-9da813334f0a"
version = "0.5.7"

[[MappedArrays]]
git-tree-sha1 = "e8b359ef06ec72e8c030463fe02efe5527ee5142"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.1"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

[[OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[Parsers]]
deps = ["Dates"]
git-tree-sha1 = "1285416549ccfcdf0c50d4997a94331e88d68413"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.3.1"

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "670e559e5c8e191ded66fa9ea89c97f10376bb4c"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.38"

[[PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "a6062fe4063cdafe78f4a0a81cfffb89721b30e7"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.2"

[[PrettyTables]]
deps = ["Crayons", "Formatting", "Markdown", "Reexport", "Tables"]
git-tree-sha1 = "dfb54c4e414caa595a1f2ed759b160f5a3ddcba5"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "1.3.1"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[SearchModels]]
deps = ["Distributed", "Parameters", "Random", "StatsBase"]
git-tree-sha1 = "1134625e07755b1d00f8c722cdffa884c66d6ca3"
uuid = "0e966ebe-b704-4a65-8279-db954bfe5da0"
version = "0.3.0"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[SimilaritySearch]]
deps = ["Dates", "Distances", "Intersections", "LinearAlgebra", "Parameters", "Random", "SearchModels", "StatsBase"]
git-tree-sha1 = "0720be12e7fdcb53fe802e9472e124e5ae0600a2"
uuid = "053f045d-5466-53fd-b400-a066f88fe02a"
version = "0.8.16"

[[Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "c82aaa13b44ea00134f8c9c89819477bd3986ecd"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.3.0"

[[StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "8977b17906b0a1cc74ab2e3a05faa16cf08a8291"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.16"

[[TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "5ce79ce186cc678bbb5c5681ca3379d1ddae11a1"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.7.0"

[[Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[TextSearch]]
deps = ["CategoricalArrays", "Intersections", "InvertedFiles", "LinearAlgebra", "Random", "SimilaritySearch", "SparseArrays", "StatsBase", "Test", "Unicode"]
git-tree-sha1 = "ea8afda8d5aeabcd5e9b96b69cd46a15e0b7d952"
uuid = "7f6f6c8a-3b03-11e9-223d-e7d88259bd6c"
version = "0.11.1"

[[TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "216b95ea110b5972db65aa90f88d8d89dcb8851c"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.6"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"

[[nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
"""

# ╔═╡ Cell order:
# ╟─fb25ee14-5f4e-11eb-1c8e-6d292c20388a
# ╠═11f6ef4a-5f08-11eb-0ba1-05a1a1ff205a
# ╠═1721b373-e893-4fbc-8878-f6e14d96ec7c
# ╟─0c25b8a2-5f4f-11eb-3410-f5560725ed17
# ╠═547e44b2-5f08-11eb-128c-61c32cdb2ee7
# ╠═545269a4-5f4f-11eb-0f7d-01adcaf452a0
# ╠═52bbc1a8-5f4f-11eb-1308-39c8a8a651b0
# ╠═6dc5265a-f5ae-410d-b18b-5433f6248ae9
# ╠═7cf28d6c-5f4f-11eb-0101-63bc0ff84e94
# ╠═6cceec0a-5f4f-11eb-0727-6d333d16196c
# ╠═95ea6094-5f4f-11eb-0d62-5b0407fb63a3
# ╠═8dc9b872-5f4f-11eb-254d-2ddb4b4b53e7
# ╠═9f6f6568-5f4f-11eb-2a25-8d5c2aa35f76
# ╠═010ec74a-5f0b-11eb-1c3a-f327bad5b02f
# ╟─b32bfc10-5f4f-11eb-3e9c-05de9cc7d34f
# ╠═92d0c64e-5f09-11eb-0d65-05a484984b63
# ╠═82647ddd-2a98-4edd-8649-79db649e0540
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
