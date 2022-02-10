'''reference: http://www.nltk.org/book/ch01.html'''
from collections import Counter

import nltk
from nltk import word_tokenize
from nltk.text import Text
from nltk.corpus import stopwords
from nltk.book import sent1,sent2,sent3,sent4,sent5,sent6,sent7,sent8,sent9,sents
from nltk.book import text1,text2,text3,text4,text5,text6,text7,text8,text9,texts


sent_list = [sent1,sent2,sent3,sent4,sent5,sent6,sent7,sent8,sent9]
text_list = [text1,text2,text3,text4,text5,text6,text7,text8,text9]
for ind1,x in enumerate(sent_list):
    print('{}: '.format(ind1) + ' '.join(x))


z1 = text1
z1.similar('monstrous')
z1.concordance('monstrous')
z1.concordance('contemptible')
z1.common_contexts(['monstrous', 'contemptible'])
z1.dispersion_plot(['monstrous', 'contemptible'])


str_fmt = '{:>55} {:>12} {:>12} {:>12} {:>20}'
print(str_fmt.format('name', '#token', '#set(token)', ".count('a')", 'lexical richness'))
for x in text_list:
    tmp1 = len(x)
    tmp2 = len(set(x))
    print(str_fmt.format(x.name[:55], tmp1, tmp2, x.count('a'), round(tmp1/tmp2,3)))


z1 = text1
x1 = nltk.FreqDist(z1)
x1.most_common(50)
x1.plot(50, cumulative=True)
# x1.hapaxes() #appear only once
sorted(x for x in set(z1) if len(x)>7 and x1[x]>7)


z1 = text1
z1.collocations()
tmp1 = set(stopwords.words()) | set('.,"-;_!?I' + "'") | {',"', '."', '--', '!"', '?"'}
hf1 = lambda x,y,_stopwords=tmp1: (x not in _stopwords) and (y not in _stopwords)
Counter((x.lower(),y.lower()) for (x,y) in nltk.bigrams(z1) if hf1(x,y)).most_common(20)


raw = 'This is a story about a foo bar. Foo likes to go to the bar and his last name is also bar. At home, he kept a lot of gold chocolate bars. The apple and the windows'
token = word_tokenize(raw)
text = Text(token)
' '.join(text)
text.similar('bar')
text.concordance('to')
text.findall('<.*> <a> <.*>')
text.findall('<a.*>{2,}')
