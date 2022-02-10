'''
reference: https://github.com/explosion/spacy/blob/master/examples/pipeline/custom_component_countries_api.py
REST Countries API: https://restcountries.eu (Mozilla Public License MPL 2.0)
'''
import requests
from spacy.lang.en import English
from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc, Span, Token


class RESTCountriesComponent(object):
    name = 'rest_countries'
    def __init__(self, nlp):
        r = requests.get('https://restcountries.eu/rest/v2/all')
        r.raise_for_status()  #raise an error if fail
        self.countries = {x['name']:x for x in r.json()}
        self.label = nlp.vocab.strings['GPE']

        patterns = [nlp(x) for x in self.countries.keys()]
        self.matcher = PhraseMatcher(nlp.vocab)
        self.matcher.add('COUNTRIES', None, *patterns)
        self.set_extension()

    def set_extension(self):
        def hf1(name, default):
            if not Token.has_extension(name):
                Token.set_extension(name, default=default)
        hf1('is_country', False)
        hf1('country_capital', False)
        hf1('country_latlng', False)
        hf1('country_flag', False)
        has_country = lambda tokens: any(t._.get('is_country') for t in tokens)
        if not Doc.has_extension('has_country'):
            Doc.set_extension('has_country', getter=has_country)
        if not Span.has_extension('has_country'):
            Span.set_extension('has_country', getter=has_country)

    def __call__(self, doc):
        spans = []
        for _, start, end in self.matcher(doc):
            entity = Span(doc, start, end, label=self.label)
            for token in entity:
                token._.set('is_country', True)
                token._.set('country_capital', self.countries[entity.text]['capital'])
                token._.set('country_latlng', self.countries[entity.text]['latlng'])
                token._.set('country_flag', self.countries[entity.text]['flag'])
            doc.ents = list(doc.ents) + [entity]
            spans.append(entity)
        for span in spans:
            span.merge() #do this after setting the entities, otherwise cause mismatched index
        return doc


nlp_en = English()
rest_countries = RESTCountriesComponent(nlp_en)
nlp_en.add_pipe(rest_countries)
doc = nlp_en('Some text about Colombia and the Czech Republic')

print('pipeline: ', nlp_en.pipe_names)
for token in doc:
    if token._.is_country:
        x = token._
        print(token.text, x.country_capital, x.country_latlng, x.country_flag)
print('entities: ', [(x.text, x.label_) for x in doc.ents])
