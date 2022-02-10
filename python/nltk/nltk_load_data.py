# -*- coding: utf-8 -*-
'''
reference: http://www.nltk.org/book/ch03.html

TODO: process html
TODO: process search engine results
TODO: process RSS feeds
TODO: extract text from PDF, MSDOC, etc.
'''
import os
import feedparser
import urllib.request
from bs4 import BeautifulSoup

import nltk
from nltk.text import Text
from nltk import word_tokenize
from nltk.corpus import PlaintextCorpusReader


# read from plain text
hf_file = lambda *x: os.path.join('data', *x)
with open(hf_file('gutenberg','2554.txt'), encoding='utf-8-sig') as fid: #http://www.gutenberg.org/files/2554
    raw = fid.read()
    ind1 = raw.find('PART I')
    ind2 = raw.rfind("End of Project Gutenberg’s Crime")
    raw = raw[ind1:ind2]
token = word_tokenize(raw)
text = Text(token)


# read from plain text
hf_file = lambda *x: os.path.join('data', 'PlaintextCorpusReader_folder', *x)

if not os.path.exists(hf_file()):
    raw = [
        'This is a story about a foo bar. Foo likes to go to the bar and his last name is also bar. At home, he kept a lot of gold chocolate bars.',
        'One day, foo went to the bar in his neighborhood and was shot down by a sheep, a blah blah black sheep.',
    ]
    for ind1, x in enumerate(raw):
        with open(hf_file('text{}.txt'.format(ind1)), 'w', encoding='utf-8') as fid:
            fid.write(x)

text = Text(PlaintextCorpusReader(hf_file(), '.*').words())


# read from html
with urllib.request.urlopen('http://news.bbc.co.uk/2/hi/health/2284783.stm') as url_handle:
    raw = BeautifulSoup(url_handle).get_text()
tokens = word_tokenize(raw)
text = Text(tokens[110:390])


# read from blog
llog = feedparser.parse("http://languagelog.ldc.upenn.edu/nll/?feed=atom")
post = llog.entries[2]
content = post.content[0].value
raw = BeautifulSoup(content).get_text()
text = Text(word_tokenize(raw))

'锟斤铐'.encode('unicode_escape')
ord('锟')
int('{!a}'.format('锟')[3:-1], 16)
