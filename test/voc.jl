
@testset "vocabulary" begin
    textconfig = TextConfig(tokenization=TokenizationConfig(nlist=[1]))
    voc1 = Vocabulary(textconfig, corpus)
    voc2 = Vocabulary(textconfig, tokenize_corpus(textconfig, corpus))
    @test Set(voc1.token) == Set(voc2.token)
    @test sum(voc1.ndocs) == sum(voc2.ndocs)
    @test sum(voc1.occs) == sum(voc2.occs)
    @test gettrainsize(voc1) == gettrainsize(voc2)
end

@testset "Vocabulary and BOW" begin
    textconfig = TextConfig(tokenization=TokenizationConfig(nlist=[1]))
    voc = Vocabulary(textconfig, corpus)
    B = bagofwords_corpus(voc, corpus)
    @info "==================="
    @test decode.(Ref(voc), B) == decode.(Ref(voc), bagofwords_corpus(voc, corpus))
end

@testset "numtokens counts occurrences, not distinct tokens" begin
    # `avgdoclen` is a mean DOCUMENT LENGTH, and BM25 divides each indexed document's length
    # (its total occurrences) by it. Accumulating `length(bow)` instead made it the mean number
    # of distinct tokens: 3.47x too small on Spanish Wikipedia articles, which drove the length
    # normalization as if b were 2.6 rather than 0.75.
    tc = TextConfig(tokenization=TokenizationConfig(nlist=[1]))
    corpus = ["la casa la casa roja", "la pera verde verde verde", "casa"]
    voc = Vocabulary(tc, corpus; verbose=false)

    total    = sum(length(tokenize(tc, d)) for d in corpus)
    distinct = sum(length(Set(tokenize(tc, d))) for d in corpus)
    @test distinct < total                      # the corpus can tell the two apart
    @test getnumtokens(voc) == total
    @test getnumtokens(voc) != distinct         # regression: the old accumulation
    @test avgdoclen(voc) == total / length(corpus)

    # with nothing pruned, the per-token occurrence counts must add up to the same number
    @test sum(getoccs(voc, i) for i in eachindex(voc)) == getnumtokens(voc)
end
