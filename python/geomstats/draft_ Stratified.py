# https://geomstats.github.io/notebooks/17_foundations__stratified_spaces.html
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

import geomstats
import geomstats.geometry.stratified.spider
import geomstats.geometry.stratified.graph_space

# geomstats.backend.random.seed(2020)

spider = geomstats.geometry.stratified.spider.Spider(n_rays=3, equip=True)
spider.n_rays #3
x0 = spider.random_point(n_samples=2)
x0[0].stratum #0
x0[0].stratum_coord
spider.metric.dist(x0[0], x0[1])
spider_geodesic_func = spider.metric.geodesic(x0[0], x0[1])
spider_geodesic_func(0)
spider_geodesic_func(0.5)
spider_geodesic_func(1)


## graph space
adj = geomstats.backend.array([[10, 3, 1], [3, 2, 4], [1, 4, 5]])
graph_point = geomstats.geometry.stratified.graph_space.GraphPoint(adj=adj)
graph_point.adj

# visualization
graph_point_nx = graph_point.to_networkx()
edges, weights = zip(*nx.get_edge_attributes(graph_point_nx, "weight").items())
pos = nx.spring_layout(graph_point_nx)
fig,ax = plt.subplots()
nx.draw(graph_point_nx, pos, node_color="b", edgelist=edges, ax=ax, edge_color=weights, width=5.0, edge_cmap=plt.cm.Blues)
# fig.savefig('tbd00.png')

graph_space = geomstats.geometry.stratified.graph_space.GraphSpace(n_nodes=4)
points = graph_space.random_point(2)

dist0 = graph_space.metric.dist(graph_a=points[0], graph_b=points[1])
graph_space.metric.set_aligner("FAQ") #Fast Quadratic Assignment Matching (FAQ)
dist1 = graph_space.metric.dist(graph_a=points[0], graph_b=points[1])
z0 = graph_space.metric.align_point_to_point(base_graph=points, graph_to_permute=points)
z1 = graph_space.metric.perm_

graph_space.metric.set_point_to_geodesic_aligner("default", s_min=-1.0, s_max=1.0)
geodesic_func = graph_space.metric.geodesic(points[0], points[1])
graph_space.metric.align_point_to_geodesic(geodesic=geodesic_func, graph_to_permute=points[1])
