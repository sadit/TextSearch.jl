
@testset "vocabulary" begin
    textconfig = TextConfig(nlist=[1])
    voc1 = Vocabulary(textconfig, corpus)
    voc2 = Vocabulary(textconfig, tokenize_corpus(textconfig, corpus))
    @test Set(voc1.token) == Set(voc2.token)
    @test sum(voc1.ndocs) == sum(voc2.ndocs)
    @test sum(voc1.occs) == sum(voc2.occs)
    @test voc1.corpuslen == voc2.corpuslen
end

@testset "Vocabulary and BOW" begin
    textconfig = TextConfig(nlist=[1])
    voc = Vocabulary(textconfig, corpus)
    B = bagofwords_corpus(voc, corpus)
    C = bagofwords_corpus(voc, corpus; minbatch=10^6)
    @info "==================="
    @test decode.(Ref(voc), B) == decode.(Ref(voc), C)
end

@testset "Approximate vocabulary" begin
    textconfig = TextConfig(nlist=[1])
    voc = Vocabulary(textconfig, corpus)
    @info corpus
    V = Vocabulary(voc; lookup=QgramsLookup(voc))
    @info "==================="
    @info V.lookup
    #@test decode.(Ref(voc), B) == decode.(Ref(voc), C)
end

