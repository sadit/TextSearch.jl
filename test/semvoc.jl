
@testset "semvoc" begin
    textconfig = TextConfig(nlist=[2])
    voc = vocabulary_from_thesaurus(textconfig, _corpus)
    semvoc = SemanticVocabulary(voc; textconfig=TextConfig(qlist=[4]))
    q = "laz mansanas"
    d = vectorize(semvoc, q) 
    for (k, v) in d
        @info voc[k] => v
    end

    @test voc[token2id(semvoc, q)].token == "la manzana verde esta rica"
    @info bagofwords(semvoc, "la manzana roja es rica")
    @info tokenize(semvoc, "la manzana roja es rica, pero la pera es ")
end

