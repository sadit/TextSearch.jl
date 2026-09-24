#!/usr/bin/env julia
#
# Fits the LSI projection that ships beside a profile, over the profile's own vocabulary.
#
#   julia --project=apps/textsearch lib/fit_lsi.jl PROFILE.zip CORPUS.jsonl OUT.zip \
#         [--outdim 256] [--scaling none] [--factorization auto] [--text-key text] \
#         [--group-key id] [--limit N] [--name NICK] [--repo owner/repo] [--tag TAG]
#
# Writes OUT.zip with `save_lsi`, bound to PROFILE by `profile_id`, then loads it back against
# the profile as a check.
#
# Why a separate fit rather than keeping the LSI that `fit` computes: that one is computed per
# part, over the part's vocabulary, and `merge` does not produce an LSI at all. `save_lsi`
# requires exactly the profile's vocabulary -- the columns are vocabulary ids -- so a projection
# for the merged profile has to be fitted against the merged profile's model.
#
# Why the FIRST unit of each document, with `--group-key`: consecutive records sharing a group
# key are one source document (the paragraphs of one Wikipedia article, as `parquet_to_jsonl.jl
# --split-paragraphs` writes them), and only the first record of each group is kept. A Wikipedia
# article's first paragraph is its summary, so this is one short, topical document per article:
# every article gets a say without long articles outweighing short ones, and the corpus is a
# fraction of the paragraph count (one record per article instead of ~20). Pass
# `--group-key ""` to use every record.

using JSON3, TextSearch

function parse_args(argv)
    argv = copy(argv)
    opts = Dict{String,Any}(
        "outdim" => 256, "scaling" => "none", "factorization" => "auto",
        "text-key" => "text", "group-key" => "id", "limit" => 0,
        "name" => "", "repo" => "sadit/TextSearch.jl", "tag" => PROFILES_RELEASE_TAG,
    )
    positional = String[]
    while !isempty(argv)
        a = popfirst!(argv)
        if a in ("--outdim", "--limit")
            opts[a[3:end]] = parse(Int, popfirst!(argv))
        elseif a in ("--scaling", "--factorization", "--text-key", "--group-key", "--name",
                     "--repo", "--tag")
            opts[a[3:end]] = popfirst!(argv)
        elseif startswith(a, "--")
            error("unknown option: $a")
        else
            push!(positional, a)
        end
    end
    length(positional) == 3 ||
        error("usage: fit_lsi.jl PROFILE.zip CORPUS.jsonl OUT.zip [--outdim N] [--scaling S] " *
              "[--factorization F] [--text-key K] [--group-key K] [--limit N] [--name NICK] " *
              "[--repo R] [--tag T]")
    opts["profile"], opts["corpus"], opts["out"] = positional
    opts
end

"""
    first_units(path; text_key, group_key, limit) -> Vector{String}

Streams a JSONL file and returns the text of the first record of each run of consecutive
records sharing `group_key` (every record when `group_key` is empty). `limit` caps the number
of texts returned, 0 meaning all.
"""
function first_units(path::AbstractString; text_key::AbstractString, group_key::AbstractString,
                     limit::Int)
    tk, gk = Symbol(text_key), Symbol(group_key)
    texts = String[]
    last = nothing
    nrecords = 0
    t0 = time()
    for line in eachline(path)
        isempty(line) && continue
        nrecords += 1
        rec = JSON3.read(line)
        if !isempty(group_key)
            g = get(rec, gk, nothing)
            g === nothing && error("record $nrecords has no \"$group_key\" key; pass --group-key \"\" to use every record")
            g == last && continue
            last = g
        end
        push!(texts, String(rec[tk]))
        nrecords % 1_000_000 == 0 &&
            @info "reading" nrecords kept=length(texts) elapsed_s=round(time() - t0; digits=1)
        limit > 0 && length(texts) >= limit && break
    end
    @info "corpus read" path nrecords kept=length(texts) elapsed_s=round(time() - t0; digits=1)
    texts
end

function main(argv)
    opts = parse_args(argv)
    profile = load_profile(opts["profile"])
    @info "profile" path=opts["profile"] id=profile_id(profile) vocsize=vocsize(profile.model)

    corpus = first_units(opts["corpus"]; text_key=opts["text-key"], group_key=opts["group-key"],
                         limit=opts["limit"])
    isempty(corpus) && error("no documents read from $(opts["corpus"])")

    t0 = time()
    lsi = LatentSemanticIndexing(profile.model, corpus; maxoutdim=opts["outdim"],
                                 scaling=Symbol(opts["scaling"]),
                                 factorization=Symbol(opts["factorization"]), verbose=true)
    @info "LSI fitted" outdim=outdim(lsi) ndocs=length(corpus) elapsed_s=round(time() - t0; digits=1)

    out = opts["out"]
    endswith(out, ".zip") || error("OUT must be a .zip path, got $out")
    dir = first(splitext(out))
    ispath(dir) && error("$dir already exists; remove it first (save_lsi writes there before zipping)")
    save_lsi(dir, lsi, profile; name=opts["name"], repo=opts["repo"], tag=opts["tag"])
    zip_profile(dir, out)
    rm(dir; recursive=true)

    # what a consumer will do with it: load against the profile, and project the same text
    back = load_lsi(out, profile)
    probe = corpus[1]
    a, b = vectorize(lsi, probe), vectorize(back, probe)
    cosine = sum(a .* b) / sqrt(sum(abs2, a) * sum(abs2, b))
    @info "wrote LSI artifact" out size_mb=round(filesize(out) / 2^20; digits=1) outdim=outdim(back) roundtrip_cosine=round(cosine; digits=5)
    cosine > 0.999 || error("round-trip cosine $cosine is below 0.999; the stored projection does not match the fitted one")
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main(ARGS)
