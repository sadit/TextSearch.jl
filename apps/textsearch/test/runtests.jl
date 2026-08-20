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
"""

function write_jsonl_corpus(path, docs)
    open(path, "w") do io
        for d in docs
            println(io, JSON3.write((; text=d)))
        end
    end
end

function write_fit_config(path; corpus, outdir, batch_size=0, stopwords=false, min_ndocs=1)
    cfg = replace(FIT_CONFIG,
        "%CORPUS%" => corpus, "%OUTDIR%" => outdir,
        "%BATCH_SIZE%" => string(batch_size), "%STOPWORDS%" => string(stopwords),
        "%MIN_NDOCS%" => string(min_ndocs))
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

            @testset "fit: stopwords enabled -- stopwords structurally absent from the profile's vocabulary" begin
                outdir = joinpath(dir, "profiles3")
                cfgpath = write_fit_config(joinpath(dir, "fit3.toml"); corpus=corpus_path, outdir, stopwords=true)
                TextSearchApp.cmd_fit(["--config", cfgpath])
                p = TextSearch.load_profile(joinpath(outdir, "corpus-0001.zip"))
                @test TextSearch.token2id(p.model.voc, "la") == 0  # "la" appears in every doc -> flagged and excluded
                @test !isempty(p.stopword_candidates)
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

            zippath = joinpath(dir, "profiles1", "corpus-0001.zip")

            @testset "search: token-intersection matching" begin
                out = capture_stdout() do
                    TextSearchApp.cmd_search([zippath, "casa roja", "--collection", corpus_path, "--format", "jsonl"])
                end
                lines = filter(!isempty, split(out, '\n'))
                texts = [JSON3.read(l)[:text] for l in lines]
                @test "la casa roja" in texts   # shares both "casa" and "roja"
                @test "la casa verde" in texts  # shares "casa" (t=1 default: any shared token)
                @test !("la pera verde esta rica" in texts)

                out2 = capture_stdout() do
                    TextSearchApp.cmd_search([zippath, "casa roja", "--collection", corpus_path, "--format", "jsonl", "-t", "2"])
                end
                lines2 = filter(!isempty, split(out2, '\n'))
                texts2 = [JSON3.read(l)[:text] for l in lines2]
                @test texts2 == ["la casa roja"]  # t=2: must share BOTH tokens
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
                @test m.encoder["kind"] == "merged"
                @test m.encoder["n_sources"] == 3

                # a directory of profiles expands; a single profile is not a merge
                @test_throws Exception TextSearchApp.cmd_merge([zippath, "--out", joinpath(dir, "x.zip")])
                # output must be a .zip
                @test_throws Exception TextSearchApp.cmd_merge([batchdir, "--out", joinpath(dir, "nope.tar")])
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
