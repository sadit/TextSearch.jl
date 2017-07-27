using Languages
using TextModel
using Base.Test

const text0 = "@user;) #jello.world"
const text1 = "hello world!! @user;) #jello.world :)"
const text2 = "a b c d e f g h i j k l m n o p q"
const corpus = ["hello world :)", "@user;) excellent!!", "#jello world."]

@testset "Constructors" begin
    @test VBOW(Dict("hola" => 1, "mundo" => 1, "!" => 8)) == VBOW([("hola", 1), ("mundo", 1), ("!", 8)])
    @test cosine(VBOW(Dict("hola" => 1, "mundo" => 1, "!" => 8)), VBOW([("hola", 1), ("mundo", 1), ("!", 8)])) ≈ 1.0
end

@testset "Character q-grams" begin
    config = TextConfig()
    config.del_usr = false
    config.nlist = []
    config.qlist = [3]
    config.skiplist = []
    @test compute_bow(text0, config) == Dict(") #"=>1,"llo"=>1,".wo"=>1,"orl"=>1,"rld"=>1,"ld "=>1,"wor"=>1,"ell"=>1,"r;)"=>1,"#je"=>1," @u"=>1,"ser"=>1,";) "=>1," #j"=>1,"jel"=>1,"lo."=>1,"@us"=>1,"use"=>1,"er;"=>1,"o.w"=>1)
end

@testset "Word n-grams" begin
    config = TextConfig()
    config.del_usr = false
    config.nlist = [1, 2]
    config.qlist = []
    config.skiplist = []
    @test compute_bow(text0, config) == Dict("@user"=>1,";) #jello"=>1,"@user ;)"=>1,"#jello"=>1,". world"=>1,"#jello ."=>1,"."=>1,"world"=>1,";)"=>1)
    config = TextConfig()
    config.stem = true
    config.del_sw = true
    language!(config, SpanishLanguage)
    #@test compute_bow("cosas que tienen cositas; la locura de ser y estar", config) == Dict(";"=>1,"locur"=>1,"cos"=>1,"cosit"=>1)
    @test compute_bow("pepe pecas pica papas con un pico pepe pecas pica papas", config) ==  Dict("pic"=>3,"pep"=>2,"pec"=>2,"pap"=>2)
end

@testset "Skip-grams" begin
    config = TextConfig()
    config.nlist = []
    config.qlist = []
    config.del_punc = true
    config.skiplist = [(2,1), (2, 2), (3, 1), (3, 2)]
    #L = collect(compute_bow(text2, config))
    #sort!(L)
    @test compute_bow(text2, config) == Dict("f i"=>1,"j m"=>1,"b e h"=>1,"i k m"=>1,"c e"=>1,"h k n"=>1,"m o q"=>1,"j l"=>1,"j l n"=>1,"a d g"=>1,"c e g"=>1,"a c"=>1,"a d"=>1,"m p"=>1,"b d f"=>1,"f h j"=>1,"l o"=>1,"b d"=>1,"i l o"=>1,"c f"=>1,"e h"=>1,"b e"=>1,"j m p"=>1,"k m"=>1,"c f i"=>1,"g i"=>1,"e g"=>1,"i l"=>1,"k m o"=>1,"e h k"=>1,"d f h"=>1,"e g i"=>1,"l n p"=>1,"a c e"=>1,"f i l"=>1,"m o"=>1,"n p"=>1,"d g j"=>1,"d g"=>1,"n q"=>1,"g i k"=>1,"o q"=>1,"f h"=>1,"l n"=>1,"h j"=>1,"k n"=>1,"g j m"=>1,"g j"=>1,"h k"=>1,"i k"=>1,"h j l"=>1,"k n q"=>1,"d f"=>1)
end

@testset "Tokenizer, BOW, and vectorize" begin # test_vmodel
    config = TextConfig()
    config.nlist = [1]
    config.qlist = []
    config.skiplist = []

    @test tokenize(text1, config) == String["hello", "world", "!!",  "@user", ";)", "#jello", ".", "world", ":)"]
    vmodel = VectorModel(config)
    TextModel.fit!(vmodel, corpus)
    @test length(vectorize(text1, vmodel)) == 8
    @test length(vectorize(text2, vmodel)) == 0
    @show vectorize_tfidf(text1, vmodel)
end


const labeled_corpus = [("me gusta", 1), ("me encanta", 1), ("lo odio", 2), ("odio esto", 2), ("me encanta esto LOL!", 1)]
const sentiment_text = "lol, esto me encanta"

@testset "DistModel tests" begin
    config = TextConfig()
    config.nlist = [1]
    dmodel = DistModel(config, 2)
    TextModel.fit!(dmodel, labeled_corpus)
    dmap = id2token(dmodel)
    @show sentiment_text
    @show dmodel
    #TextModel.hist(dmodel)
    @test [(dmap[t.id], t.weight) for t in vectorize(sentiment_text, dmodel).tokens] == [("me<1>",1.0),("me<2>",0.0),("encanta<1>",1.0),("encanta<2>",0.0),("esto<1>",0.5),("esto<2>",0.5),("lol<1>",1.0),("lol<2>",0.0)]

end

@testset "DistModel-normalize! tests" begin
    config = TextConfig()
    config.nlist = [1]
    dmodel = DistModel(config, 2)
    TextModel.fit!(dmodel, labeled_corpus, normalize=minimum)
    dmap = id2token(dmodel)
    @show sentiment_text
    @show dmodel
    #TextModel.hist(dmodel)
    d1 = [(dmap[t.id], t.weight) for t in vectorize(sentiment_text, dmodel).tokens]
    d2 = [("me<1>", 1.0), ("me<2>", 0.0), ("encanta<1>", 1.0), ("encanta<2>", 0.0), ("esto<1>", 0.4), ("esto<2>", 0.6), ("lol<1>", 1.0), ("lol<2>", 0.0)]
    @test string(d1) == string(d2)
end

@testset "EntModel tests" begin
    config = TextConfig()
    config.nlist = [1]
    dmodel = DistModel(config, 2)
    feed!(dmodel, labeled_corpus)
    emodel = EntModel(dmodel, 0)
    @show emodel
    emap = id2token(emodel)
    @test [(emap[t.id], t.weight) for t in vectorize(sentiment_text, emodel).tokens] == [("esto",0.0),("encanta",1.0),("me",1.0),("lol",1.0)]

    # @show [(maptoken[term.id], term.id, term.weight) for term in vectorize(sentiment_text, emodel).terms]
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
    @test dist(u1, v1) ≈ 0.5975474808029686
    @test dist(u1, u1) <= eps(Float32)
    @test dist(w1, u1) ≈ 1.5707963267948966
end

@testset "operations" begin
    u = VBOW(Dict("el" => 0.1, "hola" => 0.2, "mundo" => 0.4))
    v = VBOW(Dict("el" => 0.2, "hola" => 0.4, "mundo" => 0.8))
    w = VBOW(Dict("el" => 0.1^2, "hola" => 0.2^2, "mundo" => 0.4^2))
    y = VBOW(Dict("el" => 0.1/9, "hola" => 0.2/9, "mundo" => 0.4/9))
    @test u == u
    @test u != v
    @test u + u == v
    @test u * u == w
    @test u * (1/9) == y
    @test (1/9) * u == y
end

@testset "io" begin
    buff = IOBuffer("""{"key1": "value1a", "key2c": "value2a"}
{"key1": "value1b", "key2c": "value2b"}
{"key1": "value1c", "key2b": "value2c"}
{"key1": "value1d", "key2a": "value2d"}""")
    for x in itertweets(buff)
        print(x)
    end
end
