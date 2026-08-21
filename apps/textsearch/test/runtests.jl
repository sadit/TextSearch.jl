using Test
using JSON3, CSV
using TextSearch

include(joinpath(@__DIR__, "..", "src", "TextSearchApp.jl"))
using .TextSearchApp

const FIT_CONFIG = """
[input]
format = "jsonl"
path = "%CORPUS%"
text_key = "text"

[output]
dir = "%OUTDIR%"
prefix = "corpus"
batch_size = %BATCH_SIZE%
resume = %RESUME%

[normalization]
del_diac = true
del_dup = false
del_punc = false
group_num = true
group_url = true
group_usr = false
group_emo = false
lc = true

[tokenization]
nlist = [1]
mark_token_type = true

[vocabulary]
min_ndocs = %MIN_NDOCS%

[stopwords]
enabled = %STOPWORDS%
doc_freq_threshold = 0.5

[encoder]
kind = "lsi"
outdim = 8
scaling = "none"
external_path = ""

[synonyms]
k = 3

[lemmas]
algorithm = "fft"
num_clusters = 0
selector = "shortest"
apply = %LEMMA_APPLY%
"""

function write_jsonl_corpus(path, docs)
    open(path, "w") do io
        for d in docs
            println(io, JSON3.write((; text=d)))
        end
    end
end

function write_fit_config(path; corpus, outdir, batch_size=0, stopwords=false, min_ndocs=1,
                          resume=false, lemma_apply=true)
    cfg = replace(FIT_CONFIG,
        "%CORPUS%" => corpus, "%OUTDIR%" => outdir,
        "%BATCH_SIZE%" => string(batch_size), "%STOPWORDS%" => string(stopwords),
        "%MIN_NDOCS%" => string(min_ndocs), "%RESUME%" => string(resume),
        "%LEMMA_APPLY%" => string(lemma_apply))
    write(path, cfg)
    path
end

"""
    capture_stdout(f) -> String

Runs `f()` with `stdout` redirected to a temp file and returns everything it printed.
`redirect_stdout` needs a real OS-backed stream (not a plain in-memory `IOBuffer`), hence
the temp file.
"""
function capture_stdout(f)
    mktemp() do path, io
        redirect_stdout(f, io)
        flush(io)
        read(path, String)
    end
end

@testset "textsearch CLI" begin
    mktempdir() do dir
        withenv("TEXTSEARCH_HOME" => joinpath(dir, "home")) do

            docs = [
                "la casa roja", "la casa verde", "la casa azul",
                "la manzana roja", "la pera verde esta rica",
                "la manzana verde esta rica", "la hoja verde",
            ]
            corpus_path = joinpath(dir, "corpus.jsonl")
            write_jsonl_corpus(corpus_path, docs)

            @testset "fit: single batch (batch_size=0)" begin
                outdir = joinpath(dir, "profiles1")
                cfgpath = write_fit_config(joinpath(dir, "fit1.toml"); corpus=corpus_path, outdir)
                TextSearchApp.cmd_fit(["--config", cfgpath])
                @test isfile(joinpath(outdir, "corpus-0001.zip"))
                @test !isfile(joinpath(outdir, "corpus-0002.zip"))
            end

            @testset "fit exits 0 (a batch count must not leak into the exit status)" begin
                outdir = joinpath(dir, "profiles_exit")
                cfgpath = write_fit_config(joinpath(dir, "fit_exit.toml"); corpus=corpus_path, outdir, batch_size=3)
                # 7 docs / batch_size 3 = 3 batches: a count-as-exit-code bug shows up here
                @test TextSearchApp.main(["fit", "--config", cfgpath]) == 0
            end

            @testset "fit: batching splits output into multiple zips" begin
                outdir = joinpath(dir, "profiles2")
                cfgpath = write_fit_config(joinpath(dir, "fit2.toml"); corpus=corpus_path, outdir, batch_size=3)
                TextSearchApp.cmd_fit(["--config", cfgpath])
                @test isfile(joinpath(outdir, "corpus-0001.zip"))
                @test isfile(joinpath(outdir, "corpus-0002.zip"))
                @test isfile(joinpath(outdir, "corpus-0003.zip"))
                @test !isfile(joinpath(outdir, "corpus-0004.zip"))
            end

            @testset "fit: resume skips completed parts without shifting boundaries" begin
                outdir = joinpath(dir, "profiles_resume")
                cfgpath = write_fit_config(joinpath(dir, "fit_resume.toml");
                                           corpus=corpus_path, outdir, batch_size=3, resume=true)
                TextSearchApp.cmd_fit(["--config", cfgpath])
                zips = sort(filter(f -> endswith(f, ".zip"), readdir(outdir)))
                @test length(zips) == 3

                # record what part 3 contained, drop it, and resume
                p3 = TextSearch.load_profile(joinpath(outdir, "corpus-0003.zip"))
                trainsize3, vocsize3 = TextSearch.trainsize(p3.model.voc), TextSearch.vocsize(p3.model.voc)
                mtime1 = mtime(joinpath(outdir, "corpus-0001.zip"))
                rm(joinpath(outdir, "corpus-0003.zip"))

                out = capture_stdout() do
                    TextSearchApp.cmd_fit(["--config", cfgpath])
                end
                @test occursin("skipping fit", out)
                @test isfile(joinpath(outdir, "corpus-0003.zip"))
                @test mtime(joinpath(outdir, "corpus-0001.zip")) == mtime1   # untouched

                # the refitted part must cover the same documents as before: skipping a part
                # still has to consume its inputs or every later boundary shifts
                p3b = TextSearch.load_profile(joinpath(outdir, "corpus-0003.zip"))
                @test TextSearch.trainsize(p3b.model.voc) == trainsize3
                @test TextSearch.vocsize(p3b.model.voc) == vocsize3
            end

            @testset "fit: stopwords enabled -- stopwords structurally absent from the profile's vocabulary" begin
                outdir = joinpath(dir, "profiles3")
                cfgpath = write_fit_config(joinpath(dir, "fit3.toml"); corpus=corpus_path, outdir, stopwords=true)
                TextSearchApp.cmd_fit(["--config", cfgpath])
                p = TextSearch.load_profile(joinpath(outdir, "corpus-0001.zip"))
                @test TextSearch.token2id(p.model.voc, "la") == 0  # "la" appears in every doc -> flagged and excluded
                @test !isempty(p.stopwords)
                @test p.applied.stopwords
            end

            @testset "fit: vocabulary.min_ndocs prunes rare tokens" begin
                outdir = joinpath(dir, "profiles4")
                cfgpath = write_fit_config(joinpath(dir, "fit4.toml"); corpus=corpus_path, outdir, min_ndocs=3)
                TextSearchApp.cmd_fit(["--config", cfgpath])
                p = TextSearch.load_profile(joinpath(outdir, "corpus-0001.zip"))
                # "casa" is in 3 of the 7 docs and survives; "azul" is in exactly 1 and does not
                @test TextSearch.token2id(p.model.voc, "casa") != 0
                @test TextSearch.token2id(p.model.voc, "azul") == 0
                @test all(id -> TextSearch.ndocs(p.model.voc, id) >= 3, eachindex(p.model.voc))

                # pruning everything is an error, not a silently empty profile
                bad = write_fit_config(joinpath(dir, "fit4bad.toml");
                                       corpus=corpus_path, outdir=joinpath(dir, "profiles4bad"), min_ndocs=999)
                @test_throws Exception TextSearchApp.cmd_fit(["--config", bad])
            end

            @testset "fit: [lemmas] apply bakes the lemma map into the TextConfig" begin
                # The 7-document fixture above yields no lemma families, so the third fit
                # pass would never run on it. This corpus has explicit inflection pairs.
                lemmadocs = [
                    "la casa grande tiene puertas", "las casas grandes tienen puertas",
                    "el gato negro duerme", "los gatos negros duermen",
                    "el perro corre rapido", "los perros corren rapido",
                    "la puerta abierta", "las puertas abiertas",
                    "casa casas gato gatos perro perros puerta puertas",
                ]
                lemmapath = joinpath(dir, "lemmacorpus.jsonl")
                write_jsonl_corpus(lemmapath, lemmadocs)

                outdir = joinpath(dir, "profiles_lemma")
                cfgpath = write_fit_config(joinpath(dir, "fit_lemma.toml");
                                           corpus=lemmapath, outdir)
                TextSearchApp.cmd_fit(["--config", cfgpath])
                p = TextSearch.load_profile(joinpath(outdir, "corpus-0001.zip"))

                @test !isempty(p.lemmas)
                # applied: the map is part of the pipeline, not just a saved artifact
                @test p.applied.lemmas
                @test has_lemma_transformation(gettextconfig(p).transformation)

                # the vocabulary is lemmatized, so the inflected forms are gone and the
                # lemma carries the family's counts
                for (inflected, lemma) in p.lemmas
                    @test TextSearch.token2id(p.model.voc, inflected) == 0
                    @test TextSearch.token2id(p.model.voc, lemma) != 0
                end

                # the synonym network was realigned onto lemmas: no entry may name a form
                # the vocabulary no longer has, or expand_synonyms! would drop it silently
                for (tok, syns) in p.synonyms
                    @test !haskey(p.lemmas, tok)
                    for (syn, _) in syns
                        @test !haskey(p.lemmas, syn)
                    end
                end

                # querying an inflected form reaches the lemma, through the TextConfig alone
                infl = first(keys(p.lemmas))
                @test collect(tokenize(p.model.voc.textconfig, infl)) == [p.lemmas[infl]]

                @testset "apply = false keeps the map as a reviewable artifact only" begin
                    outdir2 = joinpath(dir, "profiles_nolemma")
                    cfg2 = write_fit_config(joinpath(dir, "fit_nolemma.toml");
                                            corpus=lemmapath, outdir=outdir2, lemma_apply=false)
                    TextSearchApp.cmd_fit(["--config", cfg2])
                    q = TextSearch.load_profile(joinpath(outdir2, "corpus-0001.zip"))

                    @test !isempty(q.lemmas)                                   # still saved
                    @test !q.applied.lemmas
                    @test !has_lemma_transformation(gettextconfig(q).transformation)
                    # unlemmatized: the inflected forms are still their own tokens
                    @test TextSearch.token2id(q.model.voc, first(keys(q.lemmas))) != 0
                end
            end

            zippath = joinpath(dir, "profiles1", "corpus-0001.zip")

            # Runs cmd_search and returns the matched documents' texts, in output order.
            function search_texts(args...)
                out = capture_stdout() do
                    TextSearchApp.cmd_search([zippath, args...,
                                              "--collection", corpus_path, "--format", "jsonl"])
                end
                [JSON3.read(l)[:text] for l in filter(!isempty, split(out, '\n'))]
            end

            @testset "search: token-intersection matching" begin
                # The threshold is about set intersection, so it is asserted with both
                # artifacts off: this 7-document corpus has ~10 tokens, and an LSI over
                # that few tokens makes everything everything else's synonym, which would
                # swamp the very thing under test. The full pipe gets its own testset.
                texts = search_texts("casa roja", "--no-synonyms", "--no-lemmas")
                @test "la casa roja" in texts   # shares both "casa" and "roja"
                @test "la casa verde" in texts  # shares "casa" (t=1 default: any shared token)
                @test !("la pera verde esta rica" in texts)

                texts2 = search_texts("casa roja", "--no-synonyms", "--no-lemmas", "-t", "2")
                @test texts2 == ["la casa roja"]  # t=2: must share BOTH tokens
            end

            @testset "search: the full pipe expands the query" begin
                # What the artifacts contribute is corpus-dependent (and on a corpus this
                # small, arbitrary), so assert the invariant instead of specific synonyms:
                # expansion only ever adds query tokens, so at t=1 it can only add hits.
                narrow = search_texts("casa", "--no-synonyms", "--no-lemmas")
                wide = search_texts("casa")
                @test narrow ⊆ wide
                @test length(wide) > length(narrow)   # this corpus does expand "casa"

                # --synonyms-k bounds the expansion, so it sits between the two.
                capped = search_texts("casa", "--synonyms-k", "1")
                @test narrow ⊆ capped ⊆ wide
            end

            @testset "search: output is in corpus order, independent of chunking" begin
                # The matching loop is threaded but each task writes only its own slot and
                # printing happens afterwards in index order, so neither the thread count
                # nor the chunk boundaries may affect the output. Chunk 1 forces a flush
                # per document; 999 puts the whole corpus in one chunk.
                reference = search_texts("casa roja", "--no-synonyms", "--no-lemmas")
                for chunk in ("1", "2", "3", "999")
                    @test search_texts("casa roja", "--no-synonyms", "--no-lemmas",
                                       "--chunk", chunk) == reference
                end
                # hits come out as a subsequence of the corpus, never reordered
                @test reference == filter(in(Set(reference)), docs)
                Threads.nthreads() > 1 ||
                    @info "single-threaded run: the ordering assertions above cannot fail here"
            end

            @testset "search: presentation is serialized at volume" begin
                # The testset above runs over 7 documents, which is far too few to ever
                # expose a torn line or a reordering. This one builds a corpus big enough
                # that every chunk holds many documents per thread, and pins down the two
                # properties that parallel printing could break:
                #
                #   1. serialization -- no two tasks may write stdout concurrently, or a
                #      line would tear mid-JSON. Every line must parse.
                #   2. order -- hits must come out in corpus order, exactly.
                #
                # Matches are seeded at irregular positions so a chunk boundary landing on
                # one is not a special case, and each carries its index so the expected
                # output is known exactly rather than merely counted.
                big = String[]
                expected = String[]
                for i in 1:5000
                    if i % 7 == 3
                        push!(big, "documento numero $i con casa")
                        push!(expected, "documento numero $i con casa")
                    else
                        push!(big, "texto irrelevante numero $i")
                    end
                end
                bigpath = joinpath(dir, "big.jsonl")
                write_jsonl_corpus(bigpath, big)

                for chunk in ("64", "512", "4096", "9999")
                    out = capture_stdout() do
                        TextSearchApp.cmd_search([zippath, "casa", "--no-synonyms", "--no-lemmas",
                                                  "--collection", bigpath, "--format", "jsonl",
                                                  "--chunk", chunk])
                    end
                    lines = filter(!isempty, split(out, '\n'))
                    # every line is intact JSON: a concurrent write would tear one
                    parsed = [JSON3.read(l) for l in lines]
                    @test [String(r[:text]) for r in parsed] == expected
                end
            end

            @testset "install / list / info / uninstall" begin
                TextSearchApp.cmd_install([zippath, "mynick"])
                @test TextSearchApp.list_nicknames() == ["mynick"]

                info_text = capture_stdout() do
                    TextSearchApp.cmd_info(["mynick"])
                end
                @test occursin("trainsize", info_text)
                @test occursin("vocsize", info_text)
                @test occursin(TextSearchApp.profile_path("mynick"), info_text)

                uninstall_text = capture_stdout() do
                    TextSearchApp.cmd_uninstall(["mynick"])
                end
                @test occursin(TextSearchApp.profile_path("mynick"), uninstall_text)
                @test isfile(TextSearchApp.profile_path("mynick"))  # NOT deleted

                @test_throws Exception TextSearchApp.cmd_install([zippath, "mynick"])  # no --force -> errors
                TextSearchApp.cmd_install([zippath, "mynick", "--force"])              # --force -> ok
            end

            @testset "merge: folds batched profiles back into one" begin
                # profiles2/ holds the 3 profiles fit above with batch_size=3 over 7 docs
                batchdir = joinpath(dir, "profiles2")
                out = joinpath(dir, "merged.zip")
                @test TextSearchApp.cmd_merge([batchdir, "--out", out]) == 0
                @test isfile(out)

                m = TextSearch.load_profile(out)
                # the whole point: merging the batches recovers corpus-wide statistics
                @test TextSearch.trainsize(m.model.voc) == length(docs)

                whole = TextSearch.load_profile(zippath)   # single unbatched fit of the same docs
                @test TextSearch.vocsize(m.model.voc) == TextSearch.vocsize(whole.model.voc)
                @test TextSearch.numtokens(m.model.voc) == TextSearch.numtokens(whole.model.voc)
                for id in eachindex(whole.model.voc)
                    t = TextSearch.token(whole.model.voc, id)
                    mid = TextSearch.token2id(m.model.voc, t)
                    @test mid != 0
                    @test TextSearch.ndocs(m.model.voc, mid) == TextSearch.ndocs(whole.model.voc, id)
                    @test m.model.weight[mid] ≈ whole.model.weight[id]
                end
                @test last(m.lineage).stage === :merge
                @test last(m.lineage).params["n_sources"] == 3
                @test isbase(m)

                # a directory of profiles expands; a single profile is not a merge
                @test_throws Exception TextSearchApp.cmd_merge([zippath, "--out", joinpath(dir, "x.zip")])
                # output must be a .zip
                @test_throws Exception TextSearchApp.cmd_merge([batchdir, "--out", joinpath(dir, "nope.tar")])
            end

            @testset "refit: adapts a base profile to a sample" begin
                # a sample about a topic the base corpus never covers
                samplepath = joinpath(dir, "sample.jsonl")
                write_jsonl_corpus(samplepath, [
                    "el perro ladra fuerte", "otro perro corre rapido",
                    "el perro y el perro juegan", "un perro mas en el parque",
                ])
                outpath = joinpath(dir, "refitted.zip")

                @test TextSearchApp.cmd_refit([zippath, "--sample", samplepath,
                                               "--out", outpath]) == 0
                @test isfile(outpath)

                r = TextSearch.load_profile(outpath)
                @test TextSearch.token2id(r.model.voc, "perro") != 0
                @test istuned(r)
                @test last(r.lineage).stage === :refit
                # sample-sized, not base-sized: kappa defaults to the sample's document count
                @test TextSearch.trainsize(r.model.voc) == 8
                # the guard against a negative idf / negative BM25 numerator
                @test all(id -> TextSearch.ndocs(r.model.voc, id) <= TextSearch.trainsize(r.model.voc),
                          eachindex(r.model.voc))
                @test all(w -> w >= 0, r.model.weight)

                @testset "the output is self-contained" begin
                    # the whole point of emitting a new profile: it must not need the base
                    isolated = joinpath(dir, "isolated.zip")
                    cp(outpath, isolated)
                    @test TextSearchApp.cmd_search([isolated, "perro",
                                                    "--collection", samplepath,
                                                    "--format", "jsonl"]) == 0
                end

                @testset "--drop-distances omits the distances file" begin
                    lean = joinpath(dir, "lean.zip")
                    TextSearchApp.cmd_refit([zippath, "--sample", samplepath,
                                             "--out", lean, "--drop-distances"])
                    q = TextSearch.load_profile(lean)
                    @test q.synonym_distances === nothing
                    @test filesize(lean) <= filesize(outpath)
                end

                @testset "--kappa moves how much the base counts for" begin
                    big = joinpath(dir, "bigprior.zip")
                    TextSearchApp.cmd_refit([zippath, "--sample", samplepath,
                                             "--out", big, "--kappa", "1000"])
                    q = TextSearch.load_profile(big)
                    @test TextSearch.trainsize(q.model.voc) == 1004
                    # a large prior keeps base vocabulary the default kappa prunes away
                    @test vocsize(q.model.voc) >= vocsize(r.model.voc)
                end

                @testset "--avgdoclen pins BM25's length normalization" begin
                    pinned = joinpath(dir, "pinned.zip")
                    TextSearchApp.cmd_refit([zippath, "--sample", samplepath,
                                             "--out", pinned, "--avgdoclen", "sample"])
                    q = TextSearch.load_profile(pinned)
                    # the contract is "matches the sample", not a direction: on this fixture
                    # the sample's documents are LONGER than the base's, so pinning raises it
                    svoc = TextSearch.Vocabulary(
                        TextSearch.refit_textconfig(TextSearch.load_profile(zippath)),
                        [JSON3.read(l)[:text] for l in eachline(samplepath)]; verbose=false)
                    @test isapprox(TextSearch.avgdoclen(q.model.voc),
                                   TextSearch.avgdoclen(svoc); rtol=0.05)
                    @test TextSearch.avgdoclen(q.model.voc) != TextSearch.avgdoclen(r.model.voc)
                    # only numtokens moves; the counts the weights come from must not.
                    # Compared by content: Vocabulary construction is threaded and its token
                    # ORDER is not part of the contract.
                    vcounts(v) = Dict(TextSearch.token(v, i) =>
                                          (TextSearch.occs(v, i), TextSearch.ndocs(v, i))
                                      for i in eachindex(v))
                    @test vcounts(q.model.voc) == vcounts(r.model.voc)
                    @test Dict(TextSearch.token(q.model.voc, i) => q.model.weight[i]
                               for i in eachindex(q.model.voc)) ==
                          Dict(TextSearch.token(r.model.voc, i) => r.model.weight[i]
                               for i in eachindex(r.model.voc))
                end

                @testset "--extend-lemmas recovers families the base never saw" begin
                    ext = joinpath(dir, "extended.zip")
                    @test TextSearchApp.cmd_refit([zippath, "--sample", samplepath,
                                                   "--out", ext, "--extend-lemmas"]) == 0
                    q = TextSearch.load_profile(ext)
                    # this fixture may or may not yield a family, so assert the invariants
                    # rather than a specific pairing: whatever is mapped must be applied and
                    # must point into the vocabulary
                    for (_, lemma) in q.lemmas
                        @test TextSearch.token2id(q.model.voc, lemma) != 0
                    end

                    @test_throws Exception TextSearchApp.cmd_refit(
                        [zippath, "--sample", samplepath, "--out", joinpath(dir, "w.zip"),
                         "--extend-lemmas", "--no-lemmas"])
                end

                @testset "invalid arguments are rejected" begin
                    @test_throws Exception TextSearchApp.cmd_refit(
                        [zippath, "--sample", samplepath, "--out", joinpath(dir, "x.tar")])
                    # kappa and base-weight say the same thing two ways
                    @test_throws Exception TextSearchApp.cmd_refit(
                        [zippath, "--sample", samplepath, "--out", joinpath(dir, "y.zip"),
                         "--kappa", "10", "--base-weight", "0.5"])
                    @test_throws Exception TextSearchApp.cmd_refit(
                        [zippath, "--sample", samplepath, "--out", joinpath(dir, "z.zip"),
                         "--base-weight", "1.5"])
                    @test_throws Exception TextSearchApp.cmd_refit(
                        [zippath, "--sample", samplepath, "--out", joinpath(dir, "v.zip"),
                         "--avgdoclen", "nonsense"])
                end
            end

            @testset "corpusio.each_record: jsonl / csv / json" begin
                jsonl_path = joinpath(dir, "rec.jsonl")
                write_jsonl_corpus(jsonl_path, ["hello", "world"])
                recs = collect(TextSearchApp.each_record(:jsonl, jsonl_path, "text"))
                @test [r.first for r in recs] == ["hello", "world"]

                csv_path = joinpath(dir, "rec.csv")
                open(csv_path, "w") do io
                    println(io, "text,extra")
                    println(io, "hello,1")
                    println(io, "world,2")
                end
                recs_csv = collect(TextSearchApp.each_record(:csv, csv_path, "text"))
                @test [r.first for r in recs_csv] == ["hello", "world"]
                @test string(recs_csv[1].second["extra"]) == "1"

                json_path = joinpath(dir, "rec.json")
                write(json_path, JSON3.write([(; text="hello"), (; text="world")]))
                recs_json = collect(TextSearchApp.each_record(:json, json_path, "text"))
                @test [r.first for r in recs_json] == ["hello", "world"]
            end
        end
    end
end
