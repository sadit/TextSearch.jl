using Test, TextSearch, SimilaritySearch

# The accessor rename ships deprecated aliases, and an alias nobody exercises is an alias
# nobody notices breaking. These call each old name and check it still answers what the new
# one does.
@testset "deprecated accessor aliases" begin
    voc = Vocabulary(TextConfig(tokenization=TokenizationConfig(nlist=[1])),
                     ["la casa roja", "la casa verde"]; verbose=false)
    model = VectorModel(IdfWeighting(), TfWeighting(), voc)
    id = token2id(voc, "casa")

    @testset "Vocabulary" begin
        @test token(voc, id) == gettoken(voc, id)
        @test token(voc) == gettoken(voc)
        @test occs(voc, id) == getoccs(voc, id)
        @test occs(voc) == getoccs(voc)
        @test ndocs(voc, id) == getndocs(voc, id)
        @test ndocs(voc) == getndocs(voc)
        @test trainsize(voc) == gettrainsize(voc)
        @test numtokens(voc) == getnumtokens(voc)
    end

    @testset "VectorModel" begin
        @test token(model, id) == gettoken(model, id)
        @test occs(model, id) == getoccs(model, id)
        @test ndocs(model, id) == getndocs(model, id)
        @test trainsize(model) == gettrainsize(model)
        @test weight(model, id) == getweight(model, id)
        @test weight(model) == getweight(model)
    end

    @testset "the new names are the exported ones" begin
        exported = names(TextSearch)
        for n in (:gettoken, :getoccs, :getndocs, :gettrainsize, :getnumtokens, :gettextconfig,
                  :getpolicy, :getweight)
            @test n in exported
        end
    end
end
