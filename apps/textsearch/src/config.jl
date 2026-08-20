const FIT_CONFIG_TEMPLATE = """
# textsearch fit configuration -- edit, save, and exit your editor to proceed.

[input]
format = "jsonl"          # "plaintext" | "csv" | "jsonl" | "json" | "parquet"
path = "corpus.jsonl"
text_key = "text"          # ignored for format="plaintext"

[output]
dir = "./profiles"
prefix = "corpus"
batch_size = 10000          # max docs per output profile zip; 0 = unbounded (single profile)

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

[stopwords]
enabled = true
doc_freq_threshold = 0.5

[encoder]
kind = "lsi"                # "lsi" | "external"
outdim = 128
scaling = "none"            # "none" | "inv_singular_values" | "singular_values"
external_path = ""          # kind="external": path to a token->vector JSON mapping

[synonyms]
k = 8

[lemmas]
algorithm = "fft"           # "fft" | "dnet" | "randsel" | "multirandsel"
num_clusters = 0             # 0 = auto (sqrt(vocsize))
selector = "shortest"        # "shortest" | "most_frequent" | "shortest_then_most_frequent"
"""

"""
    edit_toml_config(; template::AbstractString=FIT_CONFIG_TEMPLATE) -> Dict

Writes `template` to a temporary file, launches `\$EDITOR` on it (falling back to `vi`
with a printed notice if `EDITOR` isn't set), blocks until the editor exits, then parses
and returns the edited file with `TOML.parsefile`. This is the "visudo-style" flow:
`fit`'s options are edited as a TOML document rather than passed as a wall of flags.
"""
function edit_toml_config(; template::AbstractString=FIT_CONFIG_TEMPLATE)
    path = tempname() * ".toml"
    write(path, template)
    editor = get(ENV, "EDITOR", "")
    if isempty(editor)
        editor = "vi"
        println(stderr, "EDITOR not set, falling back to 'vi'")
    end
    run(`$editor $path`)
    TOML.parsefile(path)
end

"""
    load_fit_config(config_path::Union{Nothing,AbstractString}) -> Dict

`config_path === nothing`: runs the visudo-style editor flow ([`edit_toml_config`](@ref)).
Otherwise reads `config_path` directly with `TOML.parsefile`, skipping the editor
entirely -- the non-interactive/scriptable escape hatch used by `--config` and by tests.
"""
function load_fit_config(config_path::Union{Nothing,AbstractString})
    config_path === nothing ? edit_toml_config() : TOML.parsefile(config_path)
end
