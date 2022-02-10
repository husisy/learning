'''
reference: http://www.nltk.org/book/ch02.html
data list: http://nltk.org/data
access nltk corpus: http://nltk.org/howto

```help(nltk.corpus.reader)``` to get more detail
.fileids()
.categories()
.raw()
.words()
.sents()
.abspath()
.encoding()
.open()
.root()
.read()
'''
import nltk
import random
from nltk.corpus import gutenberg, webtext, nps_chat, brown, reuters, inaugural
from nltk.corpus import cess_esp, floresta, indian, udhr #other language
from nltk.corpus import stopwords, names #word list
from nltk.corpus import cmudict #procouncing dictionary
from nltk.corpus import swadesh #comparative wordlist
from nltk.corpus import wordnet


# nltk.download()
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('maxent_ne_chunker')
# nltk.download('words')
# nltk.download('treebank')


# the Project Gutenberg electronic text archive: http://www.gutenberg.org/
str_fmt = '{:>15} {:>10} {:>10} {:>10}    {}'
print(str_fmt.format('#character', '#word', '#sentence', '#vocab', 'name'))
for fileid in gutenberg.fileids():
    num1 = len(gutenberg.raw(fileid))
    tmp1 = gutenberg.words(fileid)
    num2 = len(tmp1)
    num3 = len(gutenberg.sents(fileid))
    num4 = len(set(x.lower() for x in tmp1))
    print(str_fmt.format(num1,num2,num3,num4,fileid))

# z1 = nltk.Text(gutenberg.words('austen-emma.txt'))
# z1.concordance('surprise')


# webtext: firefox, grail, overhead, pirates, singles, wine
for x in webtext.fileids(): print(x)


# nps_chat collected by the Naval Postgraduate School for research on automatic detection of Internet predators
for x in nps_chat.fileids(): print(x)


# brown corpus: http://icame.uib.no/brown/bcm-los.html
for x in brown.categories(): print(x)
print(len(brown.fileids()))

cfd = nltk.ConditionalFreqDist((x,y) for x in brown.categories() for y in brown.words(categories=x))
tmp1 = ['news', 'religion', 'hobbies', 'science_fiction', 'romance', 'humor']
tmp2 = ['can', 'could', 'may', 'might', 'must', 'will']
cfd.tabulate(conditions=tmp1, samples=tmp2)
#                  can could  may might must will
#            news   93   86   66   38   50  389
#        religion   82   59   78   12   54   71
#         hobbies  268   58  131   22   83  264
# science_fiction   16   49    4   12    8   16
#         romance   74  193   11   51   45   43
#           humor   16   30    8    8    9   13


# reuters: train/test
for x in reuters.categories(): print(x)
print(len(reuters.fileids()))
print(reuters.categories('training/9865'))
print(reuters.categories(['training/9865', 'training/9880']))
print(len(reuters.fileids('barley')))
print(len(reuters.fileids(['barley', 'corn'])))


# inaugural
print(len(inaugural.fileids()))
cfd = nltk.ConditionalFreqDist(
          (target,fileid[:4])
          for fileid in inaugural.fileids()
          for word in inaugural.words(fileid)
          for target in ['america', 'citizen']
          if word.lower().startswith(target))
cfd.plot()


# full penn treebank corpus
# https://en.wikipedia.org/wiki/User:Alvations/NLTK_cheatsheet/CorporaReaders#Penn_Tree_Bank


# Universal Declaration of Human Rights (UDHR)
print(''.join(udhr.words('Chinese_Mandarin-GB2312'))[:1000])

languages = ['Chickasaw', 'English', 'German_Deutsch',
    'Greenlandic_Inuktikut', 'Hungarian_Magyar', 'Ibibio_Efik']
cfd = nltk.ConditionalFreqDist((x,len(y)) for x in languages for y in udhr.words(x+'-Latin1'))
cfd.tabulate(conditions=['English','German_Deutsch'], samples=range(10), cumulative=True)
cfd.plot(cumulative=True)

# load your own corpus
# from nltk.corpus import PlaintextCorpusReader, BracketParseCorpusReader


# generate text with bigram
text = nltk.corpus.genesis.words('english-kjv.txt')
bigrams = nltk.bigrams(text)
cfd = nltk.ConditionalFreqDist(bigrams)
word_list = ['living']
for _ in range(15):
    word_list.append(cfd[word_list[-1]].max())
print(' '.join(word_list))


'''lexicon
A lexical entry consists of a headword (lemma) along with additional information
such as the part of speech and the sense definition.
Two distinct words having the same spelling are called homonyms
'''


# unusual words
_ENGLISH_VOCAB = set(x.lower() for x in nltk.corpus.words.words())
def unusual_words(text, english_vocab=_ENGLISH_VOCAB):
    text_vocab = set(x.lower() for x in text if x.isalpha())
    return sorted(text_vocab - english_vocab)
print(', '.join(unusual_words(gutenberg.words('austen-sense.txt'))[:100]))
print(', '.join(unusual_words(nps_chat.words())[:100]))


# stopwords
print(', '.join(stopwords.words('english')))


# word puzzle
puzzle_letters = nltk.FreqDist('egivrvonl')
obligatory = 'r'
wordlist = nltk.corpus.words.words()
ret = [x for x in wordlist if (len(x)>=6) and (obligatory in x) and (nltk.FreqDist(x)<=puzzle_letters)]
print(', '.join(ret))


# name
male_names = set(names.words('male.txt'))
female_names = set(names.words('female.txt'))
ret = sorted(set.intersection(male_names, female_names))
print(', '.join(ret))

cfd = nltk.ConditionalFreqDist((x,y[-1]) for x in names.fileids() for y in names.words(x))
cfd.plot()


# cmudict: http://en.wikipedia.org/wiki/Arpabet
entries = list(cmudict.entries())
for x in random.sample(entries,10):
    print(x)


# swadesh
print(', '.join(swadesh.fileids()))
print(', '.join(swadesh.words('en')))

fr2en = dict(swadesh.entries(['fr', 'en']))
print(fr2en['chien'])
print(fr2en['jeter'])


# shoebox and toolbox lexicons: http://www.sil.org/computing/toolbox/
# from nltk.corpus import toolbox
# toolbox.entries('rotokas.dic')


# wordnet, Synset (synonym set)
wordnet.synsets('motorcar')
z1 = wordnet.synset('car.n.01')
z1.lemma_names()
z1.definition()
z1.examples()
z1.hyponyms()
z1.hypernym_paths()

z1 = wordnet.synset('tree.n.01')
z1.part_meronyms()
z1.substance_meronyms()
z1.member_holonyms()

for x in wordnet.synsets('mint', pos=wordnet.NOUN):
    print(x.name(), ':', x.definition())

wordnet.synset('walk.v.01').entailments()
wordnet.synset('eat.v.01').entailments()
wordnet.synset('tease.v.03').entailments()

wordnet.lemma('supply.n.02.supply').antonyms()
wordnet.lemma('rush.v.01.rush').antonyms()
wordnet.lemma('horizontal.a.01.horizontal').antonyms()
wordnet.lemma('staccato.r.01.staccato').antonyms()

z1 = [
    wordnet.synset('right_whale.n.01'),
    wordnet.synset('orca.n.01'),
    wordnet.synset('minke_whale.n.01'),
    wordnet.synset('tortoise.n.01'),
    wordnet.synset('novel.n.01'),
]
for x in z1:
    print(z1[0].lowest_common_hypernyms(x), z1[0].path_similarity(x))
for x in z1:
    print(x.min_depth())

