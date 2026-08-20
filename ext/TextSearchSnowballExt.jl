# This file is a part of TextSearch.jl

module TextSearchSnowballExt

using TextSearch
using Snowball
using Languages
using Unicode: normalize

_snowball_algorithm_name(lang::Languages.Language) = lowercase(Languages.english_name(lang))

"""
    SnowballTokenTransformation(lang::Languages.Language; charenc=Snowball.UTF_8)

Convenience constructor (added by the `TextSearchSnowballExt` package extension) for
the core [`SnowballTokenTransformation`](@ref) that builds a `Snowball.Stemmer` for
`lang`. Errors if `lang` has no matching Snowball algorithm (see
`Snowball.stemmer_types()` for the full list of supported languages).

# Example

```julia
julia> using TextSearch, Snowball, Languages

julia> tt = SnowballTokenTransformation(Languages.Spanish());

julia> cfg = TextConfig(transformation=tt);

julia> collect(tokenize(cfg, "las casas rojas"))
["las", "casas", "roj"]
```
"""
function TextSearch.SnowballTokenTransformation(lang::Languages.Language; charenc=Snowball.UTF_8)
    alg = _snowball_algorithm_name(lang)
    alg in Snowball.stemmer_types() || error(
        "no Snowball stemmer available for language $(Languages.english_name(lang)); " *
        "available algorithms: $(join(sort(Snowball.stemmer_types()), ", "))"
    )
    TextSearch.SnowballTokenTransformation(Snowball.Stemmer(alg, charenc))
end

TextSearch.Tokenizer.transform_unigram(tt::TextSearch.SnowballTokenTransformation, tok) = Snowball.stem(tt.stemmer, tok)

_construct_snowball_transformation(algorithm::AbstractString, charenc::AbstractString) =
    TextSearch.SnowballTokenTransformation(Snowball.Stemmer(algorithm, charenc))

"""
    IgnoreStopwords(lang::Languages.Language; lc::Bool=true, del_diac::Bool=true)

Convenience constructor for the core [`IgnoreStopwords`](@ref) that fetches the
stopword list for `lang` from Languages.jl. Each stopword is normalized (lowercased
and/or stripped of diacritics, matching `NormalizationConfig`'s `lc`/`del_diac`
defaults) so it matches the tokens seen by `transform_unigram` after normalization;
pass `lc=false`/`del_diac=false` if your `NormalizationConfig` disables those.

# Example

```julia
julia> using TextSearch, Languages

julia> cfg = TextConfig(transformation=IgnoreStopwords(Languages.Spanish()));

julia> collect(tokenize(cfg, "la casa roja"))
["casa", "roja"]
```
"""
function TextSearch.IgnoreStopwords(lang::Languages.Language; lc::Bool=true, del_diac::Bool=true)
    normword(w) = begin
        w = lc ? lowercase(w) : w
        del_diac ? normalize(w, stripmark=true) : w
    end
    TextSearch.IgnoreStopwords(Set(normword(w) for w in Languages.stopwords(lang)))
end

end # module
