```@meta

CurrentModule = TextSearch
DocTestSetup = quote
    using TextSearch
end
```

## Sparse matrices
Inverted indexes/files are representations of sparse matrices optimized for certain operations.
We provide some functions to convert inverted files to sparse matrices.
```@docs
sparse
sparsevec
```

Inverted indexes constructors also support sparse matrices as input (wrapped on `MatrixDatabase` structs)

### Dictionary-based sparse vectors
Some application domains could take advantage of hash based sparse vectors, like text search and classification.
Therefore, this package also provide a partial implementation of sparse vectors using `Dict`.

```@docs
dvec
DVEC
SVEC
SVEC32
SVEC64
nnz
findmax
argmax
maximum
findmin
argmin
minimum
normalize!
dot
norm
zero
add!
sum
+
-
*
/
centroid
evaluate
NormalizedAngleDistance
NormalizedCosineDistance
AngleDistance
CosineDistance
evaluate
``` 