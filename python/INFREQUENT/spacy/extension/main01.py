'''
reference: https://github.com/explosion/spacy/blob/master/examples/pipeline/custom_component_entities.py
'''
from spacy.lang.en import English
from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc, Span, Token


class TechCompanyRecognizer(object):
    name = 'tech_companies'
    def __init__(self, nlp, companies):
        self.label = nlp.vocab.strings['ORG']
        patterns = [nlp(x) for x in companies]
        self.matcher = PhraseMatcher(nlp.vocab)
        self.matcher.add('TECH_ORGS', None, *patterns)
        self.set_extension()

    def set_extension(self):
        has_tech_org = lambda tokens: any(t._.get('is_tech_org') for t in tokens)
        if not Token.has_extension('is_tech_org'):
            Token.set_extension('is_tech_org', default=False)
        if not Doc.has_extension('has_tech_org'):
            Doc.set_extension('has_tech_org', getter=has_tech_org)
        if not Span.has_extension('has_tech_org'):
            Span.set_extension('has_tech_org', getter=has_tech_org)

    def __call__(self, doc):
        spans = []
        for _, start, end in self.matcher(doc):
            entity = Span(doc, start, end, label=self.label)
            for token in entity: token._.set('is_tech_org', True)
            doc.ents = list(doc.ents) + [entity]
            spans.append(entity)
        for span in spans:
            span.merge() #do this after setting the entities, otherwise cause mismatched index
        return doc


text = 'Alphabet Inc. is the company behind Google.'
companies = ('Alphabet Inc.', 'Google', 'Netflix', 'Apple')

nlp_en = English()
component = TechCompanyRecognizer(nlp_en, companies)
nlp_en.add_pipe(component, last=True)

doc = nlp_en(text)
print('pipeline:           ', nlp_en.pipe_names)
print('doc._.has_tech_org: ', doc._.has_tech_org)
print('entities:           ', [(e.text, e.label_) for e in doc.ents])
print('\n{:15}: {}'.format('token', 'is_tech_org'))
for x in doc:
    print('{:15}: {}'.format(x.text, x._.is_tech_org))
