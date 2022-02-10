import nltk
from nltk.tree import Tree

z1 = Tree(1, [2, Tree(3, [4]), 5])


tmp1 = Tree('V', ['saw'])
tmp2 = Tree('NP', ['him'])
tmp3 = Tree('VP', [tmp1, tmp2])
z1 = Tree('S', [Tree('NP', ['I']), tmp3])
z1.draw()
print(z1)
z1[0], z1[1], z1[1,0], z1[1,1]
z1[1,1].label()
z1[1,1].set_label('233')
assert z1 == Tree.fromstring('(S (NP I) (VP (V saw) (NP him)))')


z1 = '''(S
   (NP Alice)
   (VP
      (V chased)
      (NP
         (Det the)
         (N rabbit))))'''
z2 = Tree.fromstring(z1)
z3 = z2.pformat() #str(z2)


def my_traverse(t):
    if isinstance(t, Tree):
        print('(', t.label(), sep='', end=' ')
        for x in t:
            my_traverse(x)
        print(')', end=' ')
    else:
        print(t, end=' ')
z1 = Tree.fromstring('(S (NP Alice) (VP chased (NP the rabbit)))')
my_traverse(z1)
