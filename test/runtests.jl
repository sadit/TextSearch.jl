
using TextModel
using Base.Test

begin # test_vmodel
    config = TextConfig()
    config.nlist = [1,2,3]
    config.qlist = [3,5]
    config.skiplist = [(2,1)]
    const text1 = "abracadabra patas de cabra que; hello world!!"
    const text2 = "pepe pecas pica papas con un pico; con un pepe pico las papas"
    const text3 = "me encanta ir de paseo"
    const text4 = "odio ir de paseo"
    const corpus = ["pepe pecas", "pica papas", "con un pico"]

    @test tokenize(text1, config) == [" ab","abr","bra","rac","aca","cad","ada","dab","abr","bra","ra ","a p"," pa","pat","ata","tas","as ","s d"," de","de ","e c"," ca","cab","abr","bra","ra ","a q"," qu","que","ue;","e; ","; h"," he","hel","ell","llo","lo ","o w"," wo","wor","orl","rld","ld!","d!!","!! "," abra","abrac","braca","racad","acada","cadab","adabr","dabra","abra ","bra p","ra pa","a pat"," pata","patas","atas ","tas d","as de","s de "," de c","de ca","e cab"," cabr","cabra","abra ","bra q","ra qu","a que"," que;","que; ","ue; h","e; he","; hel"," hell","hello","ello ","llo w","lo wo","o wor"," worl","world","orld!","rld!!","ld!! ","abracadabra","patas","de","cabra","que","hello","world","abracadabra patas","patas de","de cabra","cabra que","que hello","hello world","abracadabra patas de","patas de cabra","de cabra que","cabra que hello","que hello world","abracadabra de","patas cabra","de que","cabra hello","que world"]
    @test compute_bow(text2, config) == Dict("apa"=>2,"pico las papas"=>1,"epe p"=>2,"pico"=>2,"o; co"=>1,"on "=>2,"co;"=>1,"papas con un"=>1,"s p"=>2,"as co"=>1,"un pepe"=>1,"con u"=>2,"un pico con"=>1,"o las"=>1,"a p"=>1,"a pap"=>1,"ca pa"=>1," pi"=>3,"pico con"=>1," un"=>2,"con pico"=>1," papa"=>2,"eca"=>1,"epe"=>2,"pecas"=>2," pico"=>2," pepe"=>2,"n pep"=>1,"pepe pecas pica"=>1,"pica "=>1,"papas un"=>1,"ico; "=>1,"pico las"=>1,"pecas pica"=>1,"n pic"=>1,"; c"=>1,"ica"=>1,"pe pi"=>1,"apas "=>2,"n un "=>2,"ico"=>2,"as pa"=>1,"pica papas con"=>1,"cas"=>1,"co "=>1,"ecas "=>1,"cas p"=>1,"las p"=>1,"con un"=>2," peca"=>1,"pe pe"=>1, "con un pepe"=>1,"e pec"=>1,"pico "=>1,"papas"=>4,"as pi"=>1,"co la"=>1,"e pic"=>1,"pico con un"=>1," co"=>2," la"=>1,"pas"=>2,"con"=>4," con "=>2,"on un"=>2," pa"=>2,"pepe pico"=>1,"papas con"=>1,"un con"=>1,"; con"=>1,"pecas papas"=>1,"as "=>4,"pep"=>2,"pec"=>1,"co; c"=>1," pica"=>1,"con un pico"=>1,"pepe pica"=>1,"pico;"=>1,"un pe"=>1,"e p"=>2,"un pepe pico"=>1,"s con"=>1,"pico un"=>1,"ca "=>1,"pic"=>3,"un pi"=>1,"pas c"=>1,"pica papas"=>1,"pepe las"=>1,"las papas"=>1,"un "=>2,"pico papas"=>1,"s c"=>1,"un"=>2,"n u"=>2," un p"=>2,"pepe "=>2,"s pic"=>1,"n p"=>2,"pap"=>2," pe"=>3,"o; "=>1,"pepe"=>2,"pecas pica papas"=>1,"pepe pico las"=>1,"pica con"=>1," las "=>1,"pepe pecas"=>1,"ico l"=>1,"las"=>2,"o l"=>1,"s pap"=>1,"un pico"=>2,"con pepe"=>1,"pe "=>2,"ica p"=>1,"pica"=>1)

    vmodel = VectorModel(config)
    fit!(vmodel, corpus)
    @test length(vectorize(text1, vmodel)) == 3
    @test length(vectorize(text2, vmodel)) == 62
    @show vectorize_tfidf(text1, vmodel)
    @show vectorize_tf(text1, vmodel)
    @show vectorize_idf(text1, vmodel)
end

begin # function test_dist()
    const labeled_corpus = [("me gusta", 1), ("me encanta", 1), ("lo odio", 2), ("odio esto", 2), ("me encanta esto LOL!", 1)]
    config = TextConfig()
    config.nlist = [1]
    dmodel = DistModel(config, 2)
    fit!(dmodel, labeled_corpus)
    dmap = id2token(dmodel)
    sentiment_text = "lol, esto me encanta"
    @show sentiment_text
    @show dmodel
    #TextModel.hist(dmodel)
    @test [(dmap[t.id], t.weight) for t in vectorize(sentiment_text, dmodel).tokens] == [("me<1>",1.0),("me<2>",0.0),("encanta<1>",1.0),("encanta<2>",0.0),("esto<1>",0.5),("esto<2>",0.5),("lol<1>",1.0),("lol<2>",0.0)]

    emodel = EntModel(dmodel)
    @show emodel
    emap = id2token(emodel)
    @test [(emap[t.id], t.weight) for t in vectorize(sentiment_text, emodel).tokens] == [("esto",0.0),("encanta",1.0),("me",1.0),("lol",1.0)]

    #@show [(maptoken[term.id], term.id, term.weight) for term in vectorize(sentiment_text, emodel).terms]
    # @show vectorize(text4, vmodel)
end
#@test
# @test TextConfig()


@testset "DocumentType and VBOW" begin
    u = Dict("el" => 0.9, "hola" => 0.1, "mundo" => 0.2)
    v = Dict("el" => 0.4, "hola" => 0.2, "mundo" => 0.4)
    w = Dict("xel" => 0.4, "xhola" => 0.2, "xmundo" => 0.4)

    u1 = VBOW(u)
    v1 = VBOW(v)
    w1 = VBOW(w)

    dist = angle_distance
    @test dist(u1, v1) == 0.5975474808029686
    @test dist(u1, u1) <= eps(Float32)
    @test dist(w1, u1) == 1.5707963267948966
end
