'''
reference: https://github.com/explosion/spacy/blob/master/examples/information_extraction/entity_relations.py

extract money and currency values (entities labelled as MONEY) and then check the
dependency tree to find the noun phrase they are referring to – for example:
$9.4 million --> Net income.
'''
import spacy

def extract_currency_relations(doc):
    for span in list(doc.ents) + list(doc.noun_chunks):
        span.merge()

    relations = []
    for money in filter(lambda w: w.ent_type_ == 'MONEY', doc):
        if money.dep_ in ('attr', 'dobj'):
            subject = [w for w in money.head.lefts if w.dep_ == 'nsubj']
            if subject:
                subject = subject[0]
                relations.append((subject, money))
        elif money.dep_ == 'pobj' and money.head.dep_ == 'prep':
            relations.append((money.head.head, money))
    return relations


nlp_sm = spacy.load('en_core_web_sm')

doc = nlp_sm('Net income was $9.4 million compared to the prior year of $2.7 million.')
# doc = nlp_sm('Revenue exceeded twelve billion dollars, with a loss of $1b.')
for r1, r2 in extract_currency_relations(doc):
    print('{:<10}\t{}\t{}'.format(r1.text, r2.ent_type_, r2.text))
