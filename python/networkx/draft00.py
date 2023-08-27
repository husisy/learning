import networkx as nx
import matplotlib.pyplot as plt


G = nx.Graph()
G.add_node(1)
G.add_nodes_from([2, 3])
G.add_nodes_from([
    (4, {'color':'red'}),
    (5, {'color': 'green'}),
])
H = nx.path_graph(10)
G.add_nodes_from(H)
# G.add_node(H) #different here!! H is the node of G here
G.add_edge(1, 2)
G.add_edge(2, 3)
G.add_edges_from([(1,2), (1,3)])
G.add_edges_from(H.edges)
G.clear()


G = nx.Graph()
G.add_edge(1, 2)
G.add_edge(1, 2) #no warning/error, replace the existing edge silently
G.number_of_nodes() #2
G.number_of_edges() #1
G.edges
G.edges[1,2]
G[1][2] #equiv to G.edges[1,2]
G.nodes
G.adj #dict
G.adj[2]
list(G.neighbors(1))
G.degree(1)
G.degree([1,2])
# G.remove_node(1)
# G.remove_nodes_from(1,2)
# G.remove_edge(1,2)


DG = nx.DiGraph() #directed graph
DG.add_edge(2, 1)
DG.add_edge(1, 3)
DG.add_edge(2, 4)
DG.add_edge(1, 2)
assert tuple(DG.successors(2))==(1,4)
assert tuple(DG.edges)==((2,1), (2,4), (1,3), (1,2))
assert DG.degree(2)==3
G = DG.to_undirected()


x0 = nx.Graph()
x0.add_edge(1, 2)
x1 = nx.DiGraph(x0)
assert tuple(x1.edges)==((1,2), (2,1))

x0 = nx.Graph([(0,1), (1,2), (2,3)])
assert tuple(x0.edges)==((0,1),(1,2),(2,3))

adjacency_dict = {0:(1,2), 1:(0,2), 2:(0,1)}
x0 = nx.Graph(adjacency_dict)
assert tuple(x0.edges)==((0,1),(0,2),(1,2))


G = nx.Graph([(1, 2, {'color':'yellow'}), (2, 3, {'weight':8})])
G.adj[1]
G[1] #equiv to G.adj[1]
G.edges[1,2]
G[1][2] #equiv to G.edges[1,2]
G.add_edge(1, 3)
G[1][3]['color'] = 'blue'
G.edges[1,3]['weight'] = 3
list(G.adj.items())
list(G.edges.data())
list(G.edges.data('weight'))


MG = nx.MultiGraph()
MG.add_weighted_edges_from([(1, 2, 0.5), (1, 2, 0.75), (2, 3, 0.5)])


fig,(ax0,ax1) = plt.subplots(1, 2)
G = nx.petersen_graph()
nx.draw(G, with_labels=True, font_weight='bold', ax=ax0)
nx.draw_shell(G, nlist=[range(5, 10), range(5)], with_labels=True, font_weight='bold', ax=ax1)
fig.tight_layout()
fig.savefig('tbd00.png')

kwargs = dict(node_color='black', node_size=100, width=3)
fig,tmp0 = plt.subplots(2, 2)
ax_list = tmp0.flatten()
nx.draw_random(G, ax=ax_list[0], **kwargs)
nx.draw_circular(G, ax=ax_list[1], **kwargs)
nx.draw_spectral(G, ax=ax_list[2], **kwargs)
nx.draw_shell(G, nlist=[range(5,10),range(5)], ax=ax_list[3], **kwargs)
tmp0 = ['draw_random', 'draw_circular', 'draw_spectral', 'draw_shell']
for x in range(4):
    ax_list[x].set_title(tmp0[x])
fig.tight_layout()
fig.savefig('tbd00.png')
