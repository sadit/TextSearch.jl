```@meta

CurrentModule = TextSearch
DocTestSetup = quote
    using TextSearch
end
```
# VectorModel
```@docs

IdFreq
VectorModel
vectorize(::VectorModel, a, b)
prune(::VectorModel, a, b)
prune_select_top(::VectorModel, f)
```

## Weighting methods for VectorModel

### Local
```@docs
TfWeighting
TpWeighting
FreqWeighthing
BinaryLocalWeighting
```

### Global
```@docs
IdfWeighting
EntropyWeighting
BinaryGlobalWeighting
```
