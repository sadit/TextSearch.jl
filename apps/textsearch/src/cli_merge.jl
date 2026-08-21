function parse_merge_args(args::Vector{String})
    s = ArgParseSettings(prog="textsearch merge",
        description="Merge several profiles into one. This is what makes 'fit's batching " *
                     "usable: batching a large corpus produces one independent profile per " *
                     "batch, and merging folds them back into a single corpus-wide profile. " *
                     "Vocabulary counts and weights are merged exactly (the merged IDF is the " *
                     "true corpus-wide IDF); synonyms are fused by rank consensus and lemmas " *
                     "by plurality vote, since each input encoded in its own embedding space. " *
                     "Inputs must share normalization/tokenization and weighting scheme.")
    @add_arg_table! s begin
        "profiles"
            help = "installed nicknames, profile paths, and/or directories/globs of profile .zips (2 or more)"
            nargs = '+'
            required = true
        "--out"
            help = "output profile .zip path"
            required = true
        "--doc-freq-threshold"
            help = "document-frequency cutoff for recomputing stopword candidates on the merged counters"
            arg_type = Float64
            default = 0.5
        "--synonyms-k"
            help = "neighbors to keep per token after fusion (0 = as many as the richest input had)"
            arg_type = Int
            default = 0
    end
    parse_args(args, s)
end

"""
    _expand_profile_specs(specs) -> Vector{String}

Resolves each `merge` input to concrete profile paths. A spec may be an installed nickname,
a path to a profile `.zip`/directory, or a directory *containing* profile `.zip`s -- the
last case is the common one, since `fit` writes a whole batch of them into one output
directory (`--batch-size` producing `prefix-0001.zip`, `prefix-0002.zip`, ...).
"""
function _expand_profile_specs(specs)
    paths = String[]
    for spec in specs
        if isdir(spec) && !isfile(joinpath(spec, "manifest.json"))
            zips = sort!(filter(f -> endswith(f, ".zip"), readdir(spec; join=true)))
            isempty(zips) && error("no profile .zip files found in directory '$spec'")
            append!(paths, zips)
        else
            push!(paths, _resolve_profile_path(spec))
        end
    end
    paths
end

function cmd_merge(args::Vector{String})
    o = parse_merge_args(args)
    paths = _expand_profile_specs(o["profiles"])
    length(paths) >= 2 ||
        error("merge needs at least 2 profiles, got $(length(paths)): $(join(paths, ", "))")

    println("merging $(length(paths)) profiles:")
    profiles = map(paths) do path
        p = load_profile(path)
        println("  $path  (trainsize=$(trainsize(p.model.voc)), vocsize=$(vocsize(p.model.voc)))")
        flush(stdout)
        p
    end

    merged = merge_profiles(profiles;
        doc_freq_threshold=o["doc-freq-threshold"], synonyms_k=o["synonyms-k"])

    out = o["out"]
    endswith(out, ".zip") || error("--out must end in .zip, got '$out'")
    mkpath(dirname(abspath(out)))
    tmpdir = out * ".tmpdir"
    try
        save_profile(tmpdir, merged)
        zip_profile(tmpdir, out)
    finally
        rm(tmpdir; recursive=true, force=true)
    end

    voc = merged.model.voc
    println("merged -> $out")
    println("  trainsize=$(trainsize(voc))  vocsize=$(vocsize(voc))  numtokens=$(numtokens(voc))")
    println("  synonyms=$(length(merged.synonyms)) tokens  lemmas=$(length(merged.lemmas)) remapped  " *
            "stopwords=$(length(merged.stopwords))")
    println("  lineage: ", lineage_summary(merged))
    0
end
