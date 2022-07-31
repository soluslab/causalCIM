# An implementation of the RP-algorithm (Rebane and Pearl, 1987) for learning polytrees.

import numpy as np
import networkx as nx
from scipy.stats import chi2_contingency
from networkx.algorithms import tree
from DAGTreeGenerator import (
    gen_samples,
    to_igraph
)

# Calculating empirical mutual information.
def calc_MI(data, x, y, bins):
    c_xy = np.histogram2d(data[:, x], data[:, y], bins)[0] + np.full([bins, bins], 0.0000000001, float)
    g, p, dof, expected = chi2_contingency(c_xy, lambda_="log-likelihood")
    mi = 0.5 * g / c_xy.sum()
    return mi

# Function for producing minimum weight spanning tree. Valid methods are 'kruskal, 'prim', or 'boruvka'.
# Default method is 'kruskal'. Default number of bins is 5.
def MI_MWST(data, bins=5, method='kruskal'):
    n = data.shape[1]
    edge_weights = np.full([n, n], -1, float)
    G_data = nx.Graph()
    for i in range(n):
        for j in range(i):
            w = -calc_MI(data, j, i, bins)
            G_data.add_edge(j, i, weight=w)
            edge_weights[j][i] = -w
    min_weight_tree_data = tree.minimum_spanning_edges(G_data, algorithm=method, data=False)
    min_weight_tree_edge_list = [list(e) for e in min_weight_tree_data]
    ig_min_weight_tree = to_igraph(min_weight_tree_edge_list, n)
    return ig_min_weight_tree, edge_weights, min_weight_tree_edge_list

# RP-algorithm.  Outputs adjacency matrix of learned CPDAG of the polytree.
def rp(data, alpha=0.05):
    n = data.shape[1]
    learned_tree, weights, edges = MI_MWST(data)
    sorted_edges = sorted(sorted(e) for e in edges)
    rp_adj_mat = np.zeros([n, n], int)
    for e in sorted_edges:
        rp_adj_mat[e[0]][e[1]] = 1
        rp_adj_mat[e[1]][e[0]] = 1
    for v in range(n):
        v_edges = [e[1] for e in sorted_edges if e[0] == v] + [e[0] for e in sorted_edges if e[1] == v]
        for i in range(len(v_edges)):
            for j in range(i):
                if weights[j][i] < alpha:
                    rp_adj_mat[v][i] = 0
                    rp_adj_mat[v][j] = 0
    return rp_adj_mat


# # Example
# g = gen_samples(4, 250, 1312)
# print(g[0])
# learned_rp = rp(g[4])
# print(learned_rp)
