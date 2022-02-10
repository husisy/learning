import nltk
from nltk.tree import Tree
from nltk.corpus import treebank

# RecursiveDescentParsing: top-down, cannot handle left-recursive production X->XY
# ShiftReduceParsing: bottom-up: not guarantee to find a parse for a text even if exist
# LeftCornerParsing: top-down method with bottom-up filtering
# ChartParsing: dynamic programming

def generate_grammar_token_pair(index=1):
    tmp1 = [
        'S -> NP VP',
        'PP -> P NP',
        "NP -> Det N | Det N PP | 'I'",
        'VP -> V NP | VP PP',
        "Det -> 'an' | 'my'",
        "N -> 'elephant' | 'pajamas'",
        "V -> 'shot'",
        "P -> 'in'",
    ]
    grammar1 = nltk.CFG.fromstring('\n'.join(tmp1)) #include recursive production
    token1 = ['I', 'shot', 'an', 'elephant', 'in', 'my', 'pajamas']

    tmp1 = [
        'S -> NP VP',
        'VP -> V NP | V NP PP',
        'PP -> P NP',
        'V -> "saw" | "ate" | "walked"',
        'NP -> "John" | "Mary" | "Bob" | Det N | Det N PP',
        'Det -> "a" | "an" | "the" | "my"',
        'N -> "man" | "dog" | "cat" | "telescope" | "park"',
        'P -> "in" | "on" | "by" | "with"',
    ]
    grammar2 = nltk.CFG.fromstring('\n'.join(tmp1))
    token2 = ['Mary', 'saw', 'Bob']
    token3 = ['the', 'dog', 'saw', 'a', 'man', 'in', 'the', 'park']

    if index==0:
        return grammar1, token1
    elif index==1:
        return grammar2, token2
    elif index==2:
        return grammar2, token3
    else:
        raise Exception('invalid index')


# RecursiveDescent
grammar,token = generate_grammar_token_pair(2)
RD_parser = nltk.RecursiveDescentParser(grammar)
list(RD_parser.parse(token))


# shift-reduce parsing
# nltk.app.srparser()
grammar,token = generate_grammar_token_pair(1)
SR_parser = nltk.ShiftReduceParser(grammar)
list(SR_parser.parse(token))


# left-corner parser


# ChartParser
# well-formed substring table (WFST)
# nltk.app.chartparser()
grammar,token = generate_grammar_token_pair(0)
CP_parser = nltk.ChartParser(grammar)
list(CP_parser.parse(token))

def well_formed_substring_table(tokens, grammar):
    index = dict((p.rhs(), p.lhs()) for p in grammar.productions())

    numtokens = len(tokens)
    wfst = [[None for i in range(numtokens+1)] for j in range(numtokens+1)]
    for i in range(numtokens):
        wfst[i][i+1] = index[(tokens[i],)]

    for span in range(2, numtokens+1):
        for start in range(numtokens+1-span):
            end = start + span
            for mid in range(start+1, end):
                nt1, nt2 = wfst[start][mid], wfst[mid][end]
                if nt1 and nt2 and (nt1,nt2) in index:
                    wfst[start][end] = index[(nt1,nt2)]
    return wfst

def display(wfst):
    print('\nWFST ' + ' '.join(('%-4d' % i) for i in range(1, len(wfst))))
    for i in range(len(wfst)-1):
        print('%d   ' % i, end=' ')
        for j in range(1, len(wfst)):
            print('%-4s' % (wfst[i][j] or '.'), end=' ')
        print()

display(well_formed_substring_table(token, grammar))


tmp1 = [
    'S -> NP V NP',
    'NP -> NP Sbar',
    'Sbar -> NP V',
    "NP -> 'fish'",
    "V -> 'fish'",
]
grammar = nltk.CFG.fromstring('\n'.join(tmp1))
token = ['fish']*5
parser = nltk.ChartParser(grammar)
z1 = list(parser.parse(token))


# weighted grammar
hf1 = lambda t: t.label()=='VP' and len(t)>2 and isinstance(t[0],Tree) and isinstance(t[1],Tree) and isinstance(t[2],Tree) \
        and t[1].label()=='NP' and (t[2].label()=='PP-DTV' or t[2].label()=='NP') \
        and ('give' in t[0].leaves() or 'gave' in t[0].leaves())
z1 = [x for t in treebank.parsed_sents() for x in t.subtrees(hf1)]


# probabilistic context free grammar (PCFG)
tmp1 = [
    'S    -> NP VP              [1.0]',
    'VP   -> TV NP              [0.4]',
    'VP   -> IV                 [0.3]',
    'VP   -> DatV NP NP         [0.3]',
    "TV   -> 'saw'              [1.0]",
    "IV   -> 'ate'              [1.0]",
    "DatV -> 'gave'             [1.0]",
    "NP   -> 'telescopes'       [0.8]",
    "NP   -> 'Jack'             [0.2]",
]
grammar = nltk.PCFG.fromstring('\n'.join(tmp1))
token = ['Jack', 'saw', 'telescopes']
parser = nltk.ViterbiParser(grammar)
z1 = list(parser.parse(token))
