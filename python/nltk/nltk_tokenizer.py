import re
import random

import nltk
from nltk import word_tokenize, regexp_tokenize


# recommand
raw = (
    "'When I'M a Duchess,' she said to herself, (not in a very hopeful tone though), "
    "'I won't have any pepper in my kitchen AT ALL. Soup does very well without--Maybe it's always pepper that makes people hot-tempered,'..."
)


word_tokenize(raw)


# regexp_tokenize
re.split(' ', raw)
re.split(r'\s+', raw)
re.split(r'\W+', raw)
re.findall(r'\w+|\S\w*',raw) #I'M
re.findall(r"\w+(?:[-']\w+)*|'|\S\w*", raw)
re.findall(r"(?:\w+(?:[-']\w+)*)|'|(?:[-.(]+)|(?:\S\w*)", raw)

raw = 'That U.S.A. poster-print costs $12.40...'
pattern = '(?x)\n' + \
r'''#(?x) set flag to allow verbose regexps, \n in the line above is necessary
    ([A-Z]\.)+        # abbreviations, e.g. U.S.A.
  | \w+(-\w+)*        # words with optional internal hyphens
  | \$?\d+(\.\d+)?%?  # currency and percentages, e.g. $12.40, 82%
  | \.\.\.            # ellipsis
  | [][.,;"'?():-_`]  # these are separate tokens; includes ], [
'''
regexp_tokenize(raw, pattern) #failed, different result from the documentation


# simulated annealing word tokenize (DO NOT USE IT)
# reference: distributed regularity and phonotactic constaints are useful for segmentation
#    https://pdfs.semanticscholar.org/3897/13f6b0d7906b42eefeca0f7987d06728e86b.pdf
def segment(text, segs):
    words = []
    last = 0
    for i in range(len(segs)):
        if segs[i] == '1':
            words.append(text[last:i+1])
            last = i+1
    words.append(text[last:])
    return words

def evaluate(text, segs):
    words = segment(text, segs)
    return len(words) + sum(len(x)+1 for x in set(words))

_MAP = {'0':'1','1':'0'}
def flip_n(segs, n):
    for _ in range(n):
        ind1 = random.randint(0, len(segs)-1)
        segs = segs[:ind1] + _MAP[segs[ind1]] + segs[ind1+1:]
    return segs

def anneal(text, iterations=1000, cooling_rate=1.1):
    segs = ''.join(random.choices('01', k=len(text)-1))
    temperature = float(len(segs))
    while temperature > 0.5:
        best_segs, best = segs, evaluate(text, segs)
        for _ in range(iterations):
            guess = flip_n(segs, round(temperature))
            score = evaluate(text, guess)
            if score < best:
                best, best_segs = score, guess
        score, segs = best, best_segs
        temperature = temperature / cooling_rate
        print(evaluate(text, segs), segment(text, segs))
    print()
    return segs

raw = 'doyouseethekittyseethedoggydoyoulikethekittylikethedoggy'
seg = [
    '0000000000000001000000000010000000000000000100000000000',
    '0100100100100001001001000010100100010010000100010010000',
    '0000100100000011001000000110000100010000001100010000001',
    '0000100000100001000001000010000100000010000100000010000',
]
for ind1, x in enumerate(seg):
    print('{}: {}'.format(ind1, evaluate(raw,x)))

anneal(raw)
