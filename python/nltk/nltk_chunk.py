import nltk
from nltk import RegexpParser
from nltk.corpus import brown, conll2000
from nltk.chunk import conllstr2tree, tree2conlltags
from nltk import UnigramTagger

conll_train = conll2000.chunked_sents('train.txt', chunk_types=['NP'])
conll_val = conll2000.chunked_sents('test.txt', chunk_types=['NP'])


# NP chunking
# sentence = [('the','DT'), ('little','JJ'), ('yellow','JJ'), ('dog','NN'), ('barked','VBD'), ('at','IN'), ('the','DT'), ('cat','NN')]
sentence = [('Rapunzel','NNP'), ('let','VBD'), ('down','RP'), ('her','PP$'), ('long','JJ'), ('golden','JJ'), ('hair','NN')]
tmp1 = [
    r'NP: {<DT|PP\$>?<JJ>*<NN>}',
    '{<NNP>+}', #proper noun
]
_PATTERN_CHUNK = '\n'.join(tmp1)
regex_chunker = RegexpParser(_PATTERN_CHUNK)
ret = regex_chunker.parse(sentence)
print(ret)
ret.__repr__()
ret.draw()


# tree label
regex_chunker = RegexpParser('CHUNK: {<V.*> <TO> <V.*>}')
count = 0
for sentence in brown.tagged_sents():
    for x in regex_chunker.parse(sentence).subtrees():
        if x.label()=='CHUNK':
            print(x)
            count += 1
    if count>5: break


# chinking
tmp1 = [
    'NP: {<.*>+}',
    '}<VBD|IN>+{', #chink sequences of VBD and IN
]
_PATTERN = '\n'.join(tmp1)
regex_chunker = RegexpParser(_PATTERN)
sentence = [('the','DT'), ('little','JJ'), ('yellow','JJ'), ('dog','NN'), ('barked','VBD'), ('at','IN'), ('the','DT'), ('cat','NN')]
print(regex_chunker.parse(sentence))


# conllstr2tree
text = '''
he PRP B-NP
accepted VBD B-VP
the DT B-NP
position NN I-NP
of IN B-PP
vice NN B-NP
chairman NN I-NP
of IN B-PP
Carlyle NNP B-NP
Group NNP I-NP
, , O
a DT B-NP
merchant NN I-NP
banking NN I-NP
concern NN I-NP
. . O
'''
conllstr2tree(text, chunk_types=['NP']).draw()


# conll2000: NP + VP + PP
data = conll2000.chunked_sents('train.txt', chunk_types=['NP'])
z1 = sorted({tuple(y for y,_ in x.leaves())
    for s in conll2000.chunked_sents()
    for x in s.subtrees()
    if x.label()=='PP' and len(x)>1})


# train and evaluate
chunker = RegexpParser('')
print(chunker.evaluate(conll_val))

chunker = RegexpParser('NP: {<[CDJNP].*>+}')
print(chunker.evaluate(conll_val))
label = [x for s in conll_val for _,_,x in tree2conlltags(s)]
predict = [x for s in conll_val for _,_,x in tree2conlltags(chunker.parse(s.leaves()))]
print('acc: ', sum(x==y for x,y in zip(label,predict))/len(label))

class UnigramChunker(nltk.ChunkParserI):
    def __init__(self, train_sents): 
        self.tagger = nltk.UnigramTagger([
                [(pos,chunk) for _,pos,chunk in tree2conlltags(sentence)]
                for sentence in train_sents])
        # self.tagger = nltk.BigramTagger([
        #         [(pos,chunk) for _,pos,chunk in tree2conlltags(sentence)]
        #         for sentence in train_sents])
    def parse(self, sentence): 
        tmp1 = self.tagger.tag([pos for (_,pos) in sentence])
        ret = [(word,pos,chunk) for (word,pos),(_,chunk) in zip(sentence,tmp1)]
        return nltk.chunk.conlltags2tree(ret)
chunker = UnigramChunker(conll_train)
print(chunker.evaluate(conll_val))
tmp1 = sorted(set(p for s in conll_train for (_,p) in s.leaves()))
dict(chunker.tagger.tag(tmp1))


class ConsecutiveNPChunker(nltk.ChunkParserI): 
    def __init__(self, train_sents, sentence_feature):
        self.sentence_feature = sentence_feature
        tmp1 = [tree2conlltags(s) for s in train_sents]
        train_set = []
        for sentence in train_sents:
            tmp1 = tree2conlltags(sentence)
            tmp2 = [x[:2] for x in tmp1]
            label = [x[2] for x in tmp1]
            train_set += [(self.sentence_feature(tmp2,ind1),x) for ind1,x in enumerate(label)]
        self.clf = nltk.NaiveBayesClassifier.train(train_set)
    def parse(self, sentence):
        tmp1 = [self.sentence_feature(sentence,ind1) for ind1 in range(len(sentence))]
        tmp2 = [self.clf.classify(x) for x in tmp1]
        return nltk.chunk.conlltags2tree([(token,pos,chunk) for (token,pos),chunk in zip(sentence,tmp2)])
def sentence_feature(sentence, ind1):
    return {
        'pos':sentence[ind1][1],
        'word':sentence[ind1][0],
        'prev_pos':sentence[ind1-1][1] if ind1>0 else '<START>',
    }
chunker = ConsecutiveNPChunker(conll_train, sentence_feature)
print(chunker.evaluate(conll_val))


# recursion
# sentence = [('Mary','NN'), ('saw','VBD'), ('the','DT'), ('cat','NN'), ('sit','VB'), ('on','IN'), ('the','DT'), ('mat','NN')]
sentence = [('John','NNP'), ('thinks','VBZ'), ('Mary','NN'), ('saw','VBD'), ('the','DT'), ('cat','NN'), ('sit','VB'), ('on','IN'), ('the','DT'), ('mat','NN')]
tmp1 = [
    'NP: {<DT|JJ|NN.*>+}',
    'PP: {<IN><NP>}',
    'VP: {<VB.*><NP|PP|CLAUSE>+$}',
    'CLAUSE: {<NP><VP>}',
]
regex_parser = RegexpParser('\n'.join(tmp1), loop=1)
regex_parser.parse(sentence).draw()
