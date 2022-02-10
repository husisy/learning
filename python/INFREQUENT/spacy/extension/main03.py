'''
reference: https://github.com/explosion/spacy/blob/master/examples/pipeline/custom_component_countries_api.py
REST Countries API: https://restcountries.eu (Mozilla Public License MPL 2.0)
'''
import spacy
import requests
from spacy.matcher import Matcher

class BadHTMLMerger(object):
    name = 'bad_html'
    def __init__(self, nlp):
        self.matcher = Matcher(nlp.vocab)
        tmp1 = [{'ORTH':'<'}, {'LOWER':'br'}, {'ORTH':'>'}]
        tmp2 = [{'ORTH':'<'}, {'LOWER':'br/'}, {'ORTH':'>'}]
        self.matcher.add('BAD_HTML', None, tmp1, tmp2)
        self.set_extension()
    
    def set_extension(self):
        if not spacy.tokens.Token.has_extension('bad_html'):
            spacy.tokens.Token.set_extension('bad_html', default=False)
    
    def __call__(self, doc):
        span_list = [doc[ind1:ind2] for _,ind1,ind2 in self.matcher(doc)]
        for span in span_list:
            span.merge()
            for token in span:
                token._.bad_html = True
                doc.vocab[token.text].is_stop = True #strange no idea
        return doc

nlp_sm = spacy.load('en_core_web_sm')
bad_html = BadHTMLMerger(nlp_sm)
nlp_sm.add_pipe(bad_html, last=True)
doc = nlp_sm('Hello<br>world! <br/> This is a test.')
str_fmt = '{:>10} {:>10} {:>10}'
print(str_fmt.format('text', 'bad_html', 'is_stop'))
for x in doc:
    print(str_fmt.format(x.text, x._.bad_html, x.is_stop))
