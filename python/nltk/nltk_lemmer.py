import nltk
from nltk import word_tokenize
from nltk import PorterStemmer, LancasterStemmer, WordNetLemmatizer

tmp1 = [
    'DENNIS: Listen, strange women lying in ponds distributing swords ',
    'is no basis for a system of government. Supreme executive power derives from ',
    'a mandate from the masses, not from some farcical aquatic ceremony.',
]
raw = ''.join(tmp1)
token = word_tokenize(raw)

porter_stemmer = nltk.PorterStemmer()
[porter_stemmer.stem(x) for x in token]

lancaster_stemmer = nltk.LancasterStemmer()
[lancaster_stemmer.stem(x) for x in token]

wordnet_lemmer = nltk.WordNetLemmatizer()
[wordnet_lemmer.lemmatize(x) for x in token]
