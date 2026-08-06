
@testset "vocabulary" begin
    textconfig = TextConfig(tokenization=TokenizationConfig(nlist=[1]))
    voc1 = Vocabulary(textconfig, corpus)
    voc2 = Vocabulary(textconfig, tokenize_corpus(textconfig, corpus))
    @test Set(voc1.token) == Set(voc2.token)
    @test sum(voc1.ndocs) == sum(voc2.ndocs)
    @test sum(voc1.occs) == sum(voc2.occs)
    @test trainsize(voc1) == trainsize(voc2)
end

@testset "Vocabulary and BOW" begin
    textconfig = TextConfig(tokenization=TokenizationConfig(nlist=[1]))
    voc = Vocabulary(textconfig, corpus)
    B = bagofwords_corpus(voc, corpus)
    @info "==================="
    @test decode.(Ref(voc), B) == decode.(Ref(voc), bagofwords_corpus(voc, corpus))
end
