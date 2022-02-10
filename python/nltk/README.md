# nltk

1. link
   * [documentation](http://www.nltk.org/book)
2. `nltk.download()`

## Stanford - coreNLP

1. reference
   * [official site](https://stanfordnlp.github.io/CoreNLP/)
   * [github-wiki](https://github.com/nltk/nltk/wiki/Stanford-CoreNLP-API-in-NLTK)
2. tools
   * [part-of-speech (POS) tagger](https://nlp.stanford.edu/software/tagger.html)
   * [named entity recognizer (NER)](https://nlp.stanford.edu/software/CRF-NER.html)
   * [parser](https://nlp.stanford.edu/software/lex-parser.html)
   * [coreference resolution system](https://nlp.stanford.edu/software/dcoref.html)
   * [sentiment analysis](https://nlp.stanford.edu/sentiment/)
   * [bootstrapped pattern learning](https://nlp.stanford.edu/software/patternslearning.html)
   * open information extraction tools
3. human language support
   * English
   * Arabic
   * Chinese
   * French
   * German
   * Spanish

```bash
# localhost:9000
# run server (linux)
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer \
-preload tokenize,ssplit,pos,lemma,ner,parse,depparse \
-status_port 9000 -port 9000 -timeout 15000 &

java -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLP -file input.txt

# run server (win-cmd)
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer ^
-preload tokenize,ssplit,pos,lemma,ner,parse,depparse ^
-status_port 9000 -port 9000 -timeout 15000 &

# run server (win-powershell)
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer `
-preload tokenize,ssplit,pos,lemma,ner,parse,depparse `
-status_port 9000 -port 9000 -timeout 15000 "&"
```

```Python
from nltk.parse import CoreNLPParser
parser = CoreNLPParser(url='http://localhost:9000')
list(parser.parse('What is the airspeed of an unladen swallow ?'.split()))
list(parser.raw_parse('What is the airspeed of an unladen swallow ?'))
ner_tagger = CoreNLPParser(url='http://localhost:9000', tagtype='ner')
list(ner_tagger.tag(('Rami Eid is studying at Stony Brook University in NY'.split())))
```

## chapter00 Preface

1. [link](http://www.nltk.org/book/ch00.html)
2. task and module
   * access corpora: `corpus`
   * string process: `tokenize`, `stem`, tokenizer, tokenizer, stemmer
   * collocation discovery: `collocations`, t-test, chi-squared, pointwise mutual information
   * part of speech tagging: `tag`, n-gram, backoff, brill, HMM, TnT
   * machine learning: `classify`, `cluster`, `tbl`, decision tree, maximum entropy, naive Bayes, EM, k-means
   * chunk: `chunk`, re, n-gram, named-entity
   * parse: `parse`, `ccg`, chart, feature-based, unification, probabilistic, dependency
   * semantic interpretation: `sem`, `inference`, lambda calculus, first-order logic, model checking
   * evaluation metrics: `metrics`, precision, recall, agreement coefficients
   * probability and estimation: `probability`, frequency distribution, smoothed probability distribution
   * application: `app`, `chat`, graphical concordancer, parser, wordnet browser, chatbot
   * linguistic fieldwork: `toolbox`, manipulate data in SIL Toolbox format

## chapter01 Language Processing and Python

1. book example data
   * `nltk.book.sent1`, ... `nltk.book.sent9`, `nltk.book.sents()`
   * `nltk.book.text1`, ... `nltk.book.sent9`, `nltk.book.sents()`
2. `nltk.text.Text`
   * `.concordance()`
   * `.similar()`
   * `.common_contexts()`
   * `.dispersion_plot()`
   * `.findall()`
3. word sense disambiguation
   * The lost children were found by the searchers. (agentive)
   * The lost children were found by the mountain. (locative)
   * The lost children were found by the afternoon. (temporal)
4. pronoun resolution
   * The thieves stole the paintings. They were subsequently sold.
   * The thieves stole the paintings. They were subsequently caught.
   * The thiesves stole the paintings. They were subsequently found.
5. generating language output - question answering
   > text: The Thieves stole the paintings. Theye were subsequently sold.
   > human: Who or what was sold?
   > Machine: The paintings.
6. generating language output - machine translation, text alignment
   > The thieves stole the paintings. They were subsequently found
   > 小偷偷了画。后来发现他们。
   > 小偷偷了画。后来发现它们。
7. Spoken Dialog Systems, Turing test
8. Recognizing Textual Entailment (RTE)

## chapter02 Accessing text corpura and Lexical Resources

1. `nltk.corpus` api, more help see `nltk.corpus.reader?`
   * `.fileids()`
   * `.categories()`
   * `.raw()`
   * `.words()`
   * `.sents()`
   * `.abspath()`
   * `.readme()`
2. [Gutenberg Corpus](www.gutenberg.org): `nltk.corpus.gutenberg`
   * `austen-emma.txt`
   * `austen-persuasion.txt`
   * `austen-sense.txt`
   * `bible-kjv.txt`
   * `blake-poems.txt`
   * `bryant-stories.txt`
   * `burgess-busterbrown.txt`
   * `carroll-alice.txt`
   * `chesterton-ball.txt`
   * `chesterton-brown.txt`
   * `chesterton-thursday.txt`
   * `edgeworth-parents.txt`
   * `melville-moby_dick.txt`
   * `milton-paradise.txt`
   * `shakespeare-caesar.txt`
   * `shakespeare-hamlet.txt`
   * `shakespeare-macbeth.txt`
   * `whitman-leaves.txt`
3. `mltk.corpus.webtext`
   * `firefox.txt`
   * `grail.txt`
   * `overheard.txt`
   * `pirates.txt`
   * `singles.txt`
   * `wine.txt`
4. `nltk.corpus.nps_chat`: collected by the Naval Postgraduate School for research on automatic detection of Internet predators
   * `10-19-20s_706posts.xml`
   * `10-19-30s_705posts.xml`
   * `10-19-40s_686posts.xml`
   * `10-19-adults_706posts.xml`
   * `10-24-40s_706posts.xml`
   * `10-26-teens_706posts.xml`
   * `11-06-adults_706posts.xml`
   * `11-08-20s_705posts.xml`
   * `11-08-40s_706posts.xml`
   * `11-08-adults_705posts.xml`
   * `11-08-teens_706posts.xml`
   * `11-09-20s_706posts.xml`
   * `11-09-40s_706posts.xml`
   * `11-09-adults_706posts.xml`
   * `11-09-teens_706posts.xml`
5. [brown corpus](http://icame.uib.no/brown/bcm-los.html): `nltk.corpus.brown`
   * category
   * `adventure`
   * `belles_lettres`
   * `editorial`
   * `fiction`
   * `government`
   * `hobbies`
   * `humor`
   * `learned`
   * `lore`
   * `mystery`
   * `news`
   * `religion`
   * `reviews`
   * `romance`
   * `science_fiction`
6. `nltk.corpus.reuters`
   * category
   * train / test
7. `nltk.corpus.inaugural`
8. annotated text corpus (see table below)
9. Universal Declaration of Human Rights (UDHR) `nltk.corpus.udhr`
10. load corpus see documentation
11. lexical
    * lexical entry: a headword (lemma) along with additional information such as the part of speech and the sense definition
    * homonyms: two distinct words have the same spelling
    * `nltk.corpus.words`
    * `nltk.corpus.stopwords`
    * `nltk.corpus.names`
    * pronouncing dictionary `nltk.corpus.cmudict`
    * comparative wordlist `nltk.corpus.swadesh`
    * shoebox and toolbox lexicon `nltk.corpus.toolbox`
12. `nltk.corpus.wordnet`
    * `155287` words, `117659` synonym sets
    * lexical relation: hypernym, hyponym, meronym, holonym, entail, antonym
    * semantic similarity
    * ![hierarchy](http://www.nltk.org/images/wordnet-hierarchy.png)
13. `nltk.corpus.verbnet`

| Corpus | Compiler | Contents |
| :---: | :---: | :---: |
| Brown Corpus | Francis, Kucera | 15 genres, 1.15M words, tagged, categorized |
| CESS Treebanks | CLiC-UB | 1M words, tagged and parsed (Catalan, Spanish) |
| Chat-80 Data Files | Pereira & Warren | World Geographic Database |
| CMU Pronouncing Dictionary | CMU | 127k entries |
| CoNLL 2000 Chunking Data | CoNLL | 270k words, tagged and chunked |
| CoNLL 2002 Named Entity | CoNLL | 700k words, pos- and named-entity-tagged (Dutch, Spanish) |
| CoNLL 2007 Dependency Treebanks (sel) | CoNLL | 150k words, dependency parsed (Basque, Catalan) |
| Dependency Treebank | Narad | Dependency parsed version of Penn Treebank sample |
| FrameNet | Fillmore, Baker et al | 10k word senses, 170k manually annotated sentences |
| Floresta Treebank | Diana Santos et al | 9k sentences, tagged and parsed (Portuguese) |
| Gazetteer Lists | Various | Lists of cities and countries |
| Genesis Corpus | Misc web sources | 6 texts, 200k words, 6 languages |
| Gutenberg (selections) | Hart, Newby, et al | 18 texts, 2M words |
| Inaugural Address Corpus | CSpan | US Presidential Inaugural Addresses (1789-present) |
| Indian POS-Tagged Corpus | Kumaran et al | 60k words, tagged (Bangla, Hindi, Marathi, Telugu) |
| MacMorpho Corpus | NILC, USP, Brazil | 1M words, tagged (Brazilian Portuguese) |
| Movie Reviews | Pang, Lee | 2k movie reviews with sentiment polarity classification |
| Names Corpus | Kantrowitz, Ross | 8k male and female names |
| NIST 1999 Info Extr (selections) | Garofolo | 63k words, newswire and named-entity SGML markup |
| Nombank | Meyers | 115k propositions, 1400 noun frames |
| NPS Chat Corpus | Forsyth, Martell | 10k IM chat posts, POS-tagged and dialogue-act tagged |
| Open Multilingual WordNet | Bond et al | 15 languages, aligned to English WordNet |
| PP Attachment Corpus | Ratnaparkhi | 28k prepositional phrases, tagged as noun or verb modifiers |
| Proposition Bank | Palmer | 113k propositions, 3300 verb frames |
| Question Classification | Li, Roth | 6k questions, categorized |
| Reuters Corpus | Reuters | 1.3M words, 10k news documents, categorized |
| Roget's Thesaurus | Project Gutenberg | 200k words, formatted text |
| RTE Textual Entailment | Dagan et al | 8k sentence pairs, categorized |
| SEMCOR | Rus, Mihalcea | 880k words, part-of-speech and sense tagged |
| Senseval 2 Corpus | Pedersen | 600k words, part-of-speech and sense tagged |
| SentiWordNet | Esuli, Sebastiani | sentiment scores for 145k WordNet synonym sets |
| Shakespeare texts (selections) | Bosak | 8 books in XML format |
| State of the Union Corpus | CSPAN | 485k words, formatted text |
| Stopwords Corpus | Porter et al | 2,400 stopwords for 11 languages |
| Swadesh Corpus | Wiktionary | comparative wordlists in 24 languages |
| Switchboard Corpus (selections) | LDC | 36 phonecalls, transcribed, parsed |
| Univ Decl of Human Rights | United Nations | 480k words, 300+ languages |
| Penn Treebank (selections) | LDC | 40k words, tagged and parsed |
| TIMIT Corpus (selections) | NIST/LDC | audio files and transcripts for 16 speakers |
| VerbNet 2.1 | Palmer et al | 5k verbs, hierarchically organized, linked to WordNet |
| Wordlist Corpus | OpenOffice.org et al | 960k words and 20k affixes for 8 languages |
| WordNet 3.0 (English) | Miller, Fellbaum | 145k synonym sets |

## chapter03 Processing Raw Text

1. [link](http://www.nltk.org/book/ch03.html)
2. load corpus from
   * plain text file
   * html: `from bs4 import BeautifulSoup`
   * search engine
   * RSS feeds: `import feedparser`
   * MSword, pdf
3. NLP pipeline
   * load raw text (sequence of character)
   * tokenize the text
   * normalize the words and build the vocabulary, `.lower()`
4. UTF-8
   * characters are abstract entities which can be realized as one or more glyphs
   * only glyph can appear on a screen or be printed on paper
   * font is a mapping from character to glyph
5. stemmer: strip off any affixes
   * `nltk.PorterStemmer()`: good choice if indexing some texts and want to support search using alternative forms of words
   * `nltk.LancasterStemmer()`
6. lemmer: strip off any affixes, resulting a known word, slower
   * `nltk.WordNetLemmatizer()`
7. identify non-standard words
   * all decimal numbers can be mapped to a single token `0.0`
   * all acronym could be mapped to `AAA`
8. tokenizer
   * `nltk.regexp_tokenize()`
   * `nltk.word_tokenize()`
9. sentence segmentation
   * [unsupervised multilingual sentence boundary detection](http://www.aclweb.org/anthology/J06-4003)
   * [Distributional Regularity and Phonotactic Constraints are Useful for Segmentation](https://pdfs.semanticscholar.org/3897/13f6b0d7906b42eefeca0f7987d06728e86b.pdf)
