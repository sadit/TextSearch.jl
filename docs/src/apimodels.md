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
```@docs
TfWeighting
TpWeighting
IdfWeighting
TfidfWeighting
FreqWeighthing
```

# EntModel
```@docs

EntModel
vectorize(::EntModel, a, b)
prune(::EntModel, a, b)
prune_select_top(::EntModel, f)
```

## Weighting methods for EntModel
```@docs
EntTfWeighting
EntTpWeighting
EntWeighting
```