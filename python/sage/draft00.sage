# comment

# https://doc.sagemath.org/html/en/thematic_tutorials/group_theory.html
G = SymmetricGroup(5)
G.order()
sigma = G([(1,3), (2,5,4)])
sigma = G("(1,3)(2,5,4)")
# sigma^(-1)
sigma.order() #6
sigma.sign() #-1

G = DihedralGroup(6)
G.is_abelian() #False
G.list()
G.cayley_table()
G.cayley_table().row_keys()
G.center()
# show(G.cayley_graph()) #require x11-forwarding
G.subgroup([G.random_element()])
G.conjugacy_classes_subgroups()


K = DihedralGroup(12)
rho = K([(1,4), (2,3), (5,12), (6,11), (7,10), (8,9)])
z0 = PermutationGroup([rho])
z1 = K.subgroup([rho]) #equivalent

# Symmetries of a cube
G = PermutationGroup(["(3,2,6,7)(4,1,5,8)","(1,2,6,5)(4,3,7,8)", "(1,2,3,4)(5,6,7,8)"])
H = SymmetricGroup(4)
G.is_isomorphic(H) #True


# normal subgroup
G = AlternatingGroup(4)
H = G.subgroup([G("(1,2) (3,4)"), G("(1,3) (2,4)"), G("(1,4) (2,3)")])
H.is_normal(G) #True
K = G.quotient(H)


# simple group
AlternatingGroup(5).is_simple() #True


# composition series
# wiki/composition-series https://en.wikipedia.org/wiki/Composition_series


# conjugacy class
G = SymmetricGroup(6)
group_order = G.order()
reps = G.conjugacy_classes_representatives()
G.centralizer(reps[0])
for g in reps:
    print("class size:", group_order / G.centralizer(g).order())


## Sylow subgroups
G = SymmetricGroup(7)
subgroups = G.conjugacy_classes_subgroups()
list(map(order, subgroups))
