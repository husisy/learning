import nltk
from nltk.sem import Expression

nltk.boolean_ops()


val = nltk.Valuation([('P', True), ('Q', True), ('R', False)])
dom = set()
g = nltk.Assignment(dom)
m = nltk.Model(dom, val)
m.evaluate('(P & Q)', g)
m.evaluate('-(P & Q)', g)
m.evaluate('(P & R)', g)
m.evaluate('(P | R)', g)
