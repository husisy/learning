import pynauty

g = pynauty.Graph(5)
g.connect_vertex(0, [1, 2, 3])
g.connect_vertex(2, [1, 3, 4])
g.connect_vertex(4, [3])
print(g)
# Graph(number_of_vertices=5, directed=False,
#  adjacency_dict = {
#   0: [1, 2, 3],
#   2: [1, 3, 4],
#   4: [3],
#  },
#  vertex_coloring = [
#  ],
# )

pynauty.autgrp(g)
# ([[3, 4, 2, 0, 1]], 2.0, 0, [0, 1, 2, 0, 1], 3)
g.connect_vertex(1, [3])
pynauty.autgrp(g)
# ([[0, 1, 3, 2, 4], [1, 0, 2, 3, 4]], 4.0, 0, [0, 0, 2, 2, 4], 3)

g.set_vertex_coloring([set([3])])
print(g)
# Graph(number_of_vertices=5, directed=False,
#  adjacency_dict = {
#   0: [1, 2, 3],
#   1: [3],
#   2: [1, 3, 4],
#   4: [3],
#  },
#  vertex_coloring = [
#   set([3]),
#   set([0, 1, 2, 4]),
#  ],
# )
pynauty.autgrp(g)
# ([[1, 0, 2, 3, 4]], 2.0, 0, [0, 0, 2, 3, 4], 4)


tmp0 = {
    0: [6, 7, 8, 9],
    1: [6, 7, 8, 11],
    2: [7, 8, 10, 12],
    3: [7, 9, 11, 12],
    4: [8, 10, 11, 12],
    5: [9, 10, 11, 12],
    6: [0, 1, 9, 10],
    7: [0, 1, 2, 3],
    8: [0, 1, 2, 4],
    9: [0, 3, 5, 6],
    10: [2, 4, 5, 6],
    11: [1, 3, 4, 5],
    12: [2, 3, 4, 5],
}
g0 = pynauty.Graph(number_of_vertices=13, directed=False, adjacency_dict=tmp0, vertex_coloring=[])
g1 = g0.copy()
pynauty.delete_random_edge(g1)
pynauty.isomorphic(g0, g1) #always False
