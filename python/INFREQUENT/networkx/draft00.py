import networkx as nx

G = nx.Graph()

G.add_node('2')
G.add_nodes_from(['23','233'])
G.add_edge('2', '23')
G.add_edges_from([('2','233'), ('23','233')])
list(G)
G.node
G.nodes
G.edges
G.edges(['2','233'])
G.degree
G.adj
G.graph
G.number_of_nodes()
G.number_of_edges()
G.clear()


H = nx.path_graph(10)
G.add_nodes_from(H)
G.add_edges_from(H.edges)
G.clear()

G.add_edges_from([(1,2), (1,3)])
G.clear()

G.add_node(1)
G.add_node(1)
G.clear()

G.add_edge('2','23')
G.remove_edge('2','23')

G.add_node('233')
G.remove_node('233')

G.add_edge(1,2)
H = nx.DiGraph(G)
H.clear()
