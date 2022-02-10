import re
import spacy
from spacy.matcher import Matcher, PhraseMatcher


nlp_sm = spacy.load('en_core_web_sm')
# nlp_md = spacy.load('en_core_web_md')


# Matcher
doc = nlp_sm('This is a text about Google I/O 2015. Real Google I/O')
str_fmt = '{:>15} {:>5} {:>5} {:>15}'
print(0, str_fmt.format('id_', 'ind1', 'ind2', 'text'))
matcher = Matcher(nlp_sm.vocab)
def hf1(matcher, doc, i, matches):
    id_, ind1, ind2 = matches[i]
    print(1, str_fmt.format(nlp_sm.vocab.strings[id_], ind1, ind2, doc[ind1:ind2].text))
pattern = [{'ORTH':'Google'}, {'ORTH':'I'}, {'ORTH':'/'}, {'ORTH':'O'}, {'IS_DIGIT':True,'OP':'?'}]
matcher.add('GoogleIO', hf1, pattern)
for id_,ind1,ind2 in matcher(doc):
    print(2, str_fmt.format(nlp_sm.vocab.strings[id_], ind1, ind2, doc[ind1:ind2].text))


# phone number
doc = nlp_sm('Call me at (123) 456 789 or (123) 456 789!')
matcher = Matcher(nlp_sm.vocab)
pattern = [{'ORTH':'('}, {'SHAPE':'ddd'}, {'ORTH':')'}, {'SHAPE':'ddd'}, {'ORTH':'-','OP':'?'}, {'SHAPE':'ddd'}]
matcher.add('PHONE_NUMBER', None, pattern)
for _,ind1,ind2 in matcher(doc):
    print(doc[ind1:ind2].text)


# collect sentence
doc = nlp_sm("I'd say that Facebook is evil. – Facebook is pretty cool, right?")
ret = []
def collect_sents(matcher, doc, i, matches):
    _,ind1,ind2 = matches[i]
    span = doc[ind1:ind2]
    ret.append((span.text, span.sent.text))
pattern = [{'LOWER':'facebook'}, {'LEMMA':'be'}, {'POS':'ADV', 'OP':'*'}, {'POS':'ADJ'}]
matcher = Matcher(nlp_sm.vocab)
matcher.add('FacebookIs', collect_sents, pattern)
_ = matcher(doc)
for x,y in ret: print(x,'\t',y)


# PhraseMatcher
matcher = PhraseMatcher(nlp_sm.vocab)
tmp1 = ['Barack Obama', 'Angela Merkel', 'Washington, D.C.']
pattern = [nlp_sm(x) for x in tmp1]
matcher.add('TerminologyList', None, *pattern)
tmp1 = 'German Chancellor Angela Merkel and US President Barack Obama ' + \
        'converse in the Oval Office inside the White House in Washington, D.C.'
doc = nlp_sm(tmp1)
for _,ind1,ind2 in matcher(doc):
    print(doc[ind1:ind2].text)


# re
text = 'The spelling is "definitely", not "definately" or "deffinitely"'
re_pattern = re.compile(r'deff?in[ia]tely')
print('re.findall: ', re_pattern.findall(text))

doc = nlp_sm(text)
print('re.finditer(doc):')
for match in re.finditer(re_pattern, doc.text):
    ind1,ind2 = match.span()
    print('\t', doc.char_span(ind1, ind2).text)

definitely_flag = lambda text: bool(re_pattern.match(text))
IS_DEFINITELY = nlp_sm.vocab.add_flag(definitely_flag)
doc = nlp_sm(text)
matcher = Matcher(nlp_sm.vocab)
matcher.add('DEFINITELY', None, [{IS_DEFINITELY:True}])
print('nlp_sm.vocab.add_flag:')
for _,ind1,ind2 in matcher(doc):
    print('\t', doc[ind1:ind2].text)

nlp_sm = spacy.load('en_core_web_sm')

