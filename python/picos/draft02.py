# https://picos-api.gitlab.io/picos/graphs.html
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import picos

np_rng = np.random.default_rng(233)

N = 20 #number of nodes
G = nx.DiGraph(nx.LCF_graph(N, [1,3,14], 5)) #bidirected graph using LCF notation

# Generate edge capacities.
for e in sorted(G.edges(data=True)):
    e[2]['capacity'] = np_rng.integers(1, 21)
capacity = {(e[0], e[1]):e[2]['capacity'] for e in sorted(G.edges(data=True))}

# pos = nx.planar_layout(G)
# pos = nx.spring_layout(G, k=0.5)
pos = {
    0:  (0.07, 0.70), 1:  (0.18, 0.78), 2:  (0.26, 0.45), 3:  (0.27, 0.66),
    4:  (0.42, 0.79), 5:  (0.56, 0.95), 6:  (0.60, 0.80), 7:  (0.64, 0.65),
    8:  (0.55, 0.37), 9:  (0.65, 0.30), 10: (0.77, 0.46), 11: (0.83, 0.66),
    12: (0.90, 0.41), 13: (0.70, 0.10), 14: (0.56, 0.16), 15: (0.40, 0.17),
    16: (0.28, 0.05), 17: (0.03, 0.38), 18: (0.01, 0.66), 19: (0.00, 0.95)
}

index_source = 16
index_sink = 10

node_colors = ['lightgrey']*N
node_colors[index_source] = 'lightgreen' # Source is green.
node_colors[index_sink] = 'lightblue'  # Sink is blue.

fig,ax = plt.subplots(figsize=(8,6))
nx.draw_networkx(G, pos, node_color=node_colors, ax=ax)
labels = {(x,y): f'{capacity[(x,y)]} | {capacity[(y,x)]}' for x,y in G.edges if x < y}
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, ax=ax)
ax.axis('off')
fig.tight_layout()
fig.savefig('tbd00.png', dpi=200)


## maxflow
maxflow = picos.Problem()
f = {x:picos.RealVariable(f'f[{x}]') for x in G.edges()}
F = picos.RealVariable('F')
maxflow.add_list_of_constraints([f[e] <= capacity[e] for e in G.edges()]) #edge capacities
hf0 = lambda i: picos.sum([f[p,i] for p in G.predecessors(i)]) #incoming flow
hf1 = lambda i: picos.sum([f[i,q] for q in G.successors(i)]) #outgoing flow
maxflow.add_list_of_constraints([hf0(x)==hf1(x) for x in G.nodes() if x not in (index_source,index_sink)])
maxflow.add_constraint(hf0(index_source) + F == hf1(index_source)) #source flow
maxflow.add_constraint(hf0(index_sink) == hf1(index_sink) + F) #sink flow
maxflow.add_list_of_constraints([f[e] >= 0 for e in G.edges()]) #flow nonnegativity
maxflow.set_objective('max', F)
maxflow.solve(solver='glpk')

# equivalent formulation
maxflow2 = picos.Problem()
f = {x:picos.RealVariable(f'f[{x}]') for x in G.edges()}
F = picos.RealVariable('F')
maxflow2.add_constraint(picos.FlowConstraint(
  G, f, source=index_source, sink=index_sink, capacity='capacity', flow_value=F, graphName='G'))
maxflow2.set_objective('max', F)
maxflow2.solve(solver='glpk')

fig,ax = plt.subplots(figsize=(8,6))
flow_edges = [x for x in G.edges() if f[x].value > 1e-4]
# Draw the nodes and the edges that don't carry flow.
tmp0 = [e for e in G.edges if e not in flow_edges and (e[1], e[0]) not in flow_edges]
nx.draw_networkx(G, pos, edge_color='lightgrey', node_color=node_colors, edgelist=tmp0, ax=ax)
# Draw the edges that carry flow.
nx.draw_networkx_edges(G, pos, edgelist=flow_edges, ax=ax)
# Show flow values and capacities on these edges.
labels={e: f'{f[e]}/{capacity[e]}' for e in flow_edges}
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, ax=ax)
ax.set_title(f"Maximum flow value: {F.value}")
ax.axis('off')
fig.tight_layout()
fig.savefig('tbd00.png', dpi=200)



mincut = picos.Problem()
d = {x:picos.BinaryVariable(f'd[{x}]') for x in G.edges()} #cut indicator variables
p = picos.BinaryVariable('p', N) #potentials variable
mincut.add_list_of_constraints([d[i,j] >= p[i]-p[j] for (i,j) in G.edges()]) #potential inequalities
mincut.add_constraint(p[index_source] == 1)
mincut.add_constraint(p[index_sink] == 0)
mincut.add_constraint(p >= 0)
mincut.add_list_of_constraints([d[e] >= 0 for e in G.edges()])
mincut.set_objective('min', picos.sum([capacity[e]*d[e] for e in G.edges()]))
mincut.solve(solver='glpk')

fig,ax = plt.subplots(figsize=(8,6))
cut = [e for e in G.edges() if abs(d[e].value-1) < 1e-6]
S = [n for n in G.nodes() if abs(p[n].value-1) < 1e-6]
T = [n for n in G.nodes() if abs(p[n].value) < 1e-6]
# Draw the nodes and the edges that are not in the cut.
tmp0 = [e for e in G.edges() if e not in cut and (e[1], e[0]) not in cut]
nx.draw_networkx(G, pos, node_color=node_colors, edgelist=tmp0, ax=ax)
# Draw edges that are in the cut.
nx.draw_networkx_edges(G, pos, edgelist=cut, edge_color='r', ax=ax)
# Show capacities for cut edges.
labels = {e: '{}'.format(capacity[e]) for e in cut}
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_color='r', ax=ax)
ax.set_title(f"Minimum cut value: {mincut.value}")
ax.axis('off')
fig.tight_layout()
fig.savefig('tbd00.png', dpi=200)


# capacited flow constraint
capaflow = maxflow.get_constraint((0,))
dualcut = [e for i, e in enumerate(G.edges()) if abs(capaflow[i].dual - 1) < 1e-6]
consflow = maxflow.get_constraint((1,))
tmp0 = [n for n in G.nodes() if n not in (index_source,index_sink)]
Sdual = [index_source] + [n for i, n in enumerate(tmp0) if abs(consflow[i].dual - 1) < 1e-6]
Tdual = [index_sink] + [n for i, n in enumerate(tmp0) if abs(consflow[i].dual) < 1e-6]

fig,ax = plt.subplots(figsize=(8,6))
tmp0 = [e for e in G.edges() if e not in dualcut and (e[1], e[0]) not in dualcut]
nx.draw_networkx(G, pos, node_color=node_colors, edgelist=tmp0, ax=ax)
nx.draw_networkx_edges(G, pos, edgelist=dualcut, edge_color='r', ax=ax)
labels = {e: '{}'.format(capacity[e]) for e in dualcut}
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_color='r', ax=ax)
ax.set_title(f"Minimum cut value: {sum(capacity[e] for e in dualcut)}")
ax.axis('off')
fig.tight_layout()
fig.savefig('tbd00.png', dpi=200)


## multicut (MIP)
multicut = picos.Problem()
pairs=[(0,12),(1,5),(1,19),(2,11),(3,4),(3,9),(3,18),(6,15),(10,14)] #pairs to be separated
sources = set([p[0] for p in pairs])
sinks = set([p[1] for p in pairs])
y = {x:picos.BinaryVariable(f'y[{x}]') for x in G.edges()} #cut indicator variables
p = {x:picos.RealVariable(f'p[{x}]', N) for x in sources} #potentials variable
multicut.add_list_of_constraints([y[i,j] >= p[s][i]-p[s][j] for s in sources for (i,j) in G.edges()]) #the potential inequalities
multicut.add_list_of_constraints([p[s][s] == 1 for s in sources]) #Set the source potentials to one
multicut.add_list_of_constraints([p[s][t] == 0 for (s,t) in pairs]) #Set the sink potentials to zero
multicut.add_list_of_constraints([p[s] >= 0 for s in sources]) #Enforce nonnegativity
multicut.set_objective('min', picos.sum([capacity[e]*y[e] for e in G.edges()]))
multicut.solve(solver='glpk')
cut = [e for e in G.edges() if round(y[e]) == 1]

fig,ax = plt.subplots(figsize=(8,6))
colors=[
    ('#4CF3CE','#0FDDAF'), # turquoise
    ('#FF4D4D','#FF0000'), # red
    ('#FFA64D','#FF8000'), # orange
    ('#3ABEFE','#0198E1'), # topaz
    ('#FFDB58','#FFCC11'), # mustard
    ('#BCBC8F','#9F9F5F')  # khaki
]
node_colors=['lightgrey']*N
for i,s in enumerate(sources):
    node_colors[s]=colors[i][0]
    for t in [t for (s0,t) in pairs if s0==s]:
        node_colors[t]=colors[i][1]
# Draw the nodes and the edges that are not in the cut.
tmp0 = [e for e in G.edges() if e not in cut and (e[1], e[0]) not in cut]
nx.draw_networkx(G, pos, node_color=node_colors, edgelist=tmp0, ax=ax)
# Draw the edges that are in the cut.
nx.draw_networkx_edges(G, pos, edgelist=cut, edge_color='r', ax=ax)
# Show capacities for cut edges.
labels={e: '{}'.format(capacity[e]) for e in cut}
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_color='r', ax=ax)
ax.set_title(f"Multicut value: {multicut.value}")
ax.axis('off')
fig.tight_layout()
fig.savefig('tbd00.png', dpi=200)


## maximum cut
G = nx.Graph(G) # Make G undirected.
for (i,j) in G.edges():
    G[i][j]['weight']=capacity[i,j]+capacity[j,i]
maxcut = picos.Problem()
X = picos.SymmetricVariable('X', N)
# Retrieve the Laplacian of the graph.
L = 1/4.*nx.laplacian_matrix(G).todense()
maxcut.add_constraint(picos.maindiag(X) == 1)
maxcut.add_constraint(X >> 0) #PSD
maxcut.set_objective('max', L|X)
maxcut.solve(solver='cvxopt')


# Do up to 100 projections
V = np.linalg.cholesky(X.value) #V V^T = X
hf0 = lambda x: np.dot(x, L @ x)
tmp0 = [np.sign(V @ np_rng.normal(size=N)) for _ in range(100)]
x_cut = max(tmp0, key=hf0)
obj = hf0(x_cut)

# Extract the cut and the seperated node sets.
S1 = [n for n in range(N) if x_cut[n]<0]
S2 = [n for n in range(N) if x_cut[n]>0]
cut = [(i,j) for (i,j) in G.edges() if x_cut[i]*x_cut[j]<0]
leave = [e for e in G.edges if e not in cut]

fig,ax = plt.subplots(figsize=(8,6))
# Assign colors based on set membership.
node_colors = [('lightgreen' if n in S1 else 'lightblue') for n in range(N)]
# Draw the nodes and the edges that are not in the cut.
nx.draw_networkx(G, pos, node_color=node_colors, edgelist=leave, ax=ax)
labels={e: '{}'.format(G[e[0]][e[1]]['weight']) for e in leave}
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, ax=ax)
# Draw the edges that are in the cut.
nx.draw_networkx_edges(G, pos, edgelist=cut, edge_color='r', ax=ax)
labels={e: '{}'.format(G[e[0]][e[1]]['weight']) for e in cut}
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_color='r', ax=ax)
sval = sum(G[e[0]][e[1]]['weight'] for e in cut)
ax.set_title('SDP relaxation value: {0:.1f}\nCut value: {1:.1f} = {2:.3f}x{0:.1f}'.format(maxcut.value, sval, sval/maxcut.value))
ax.axis('off')
fig.tight_layout()
fig.savefig('tbd00.png', dpi=200)
