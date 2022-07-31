# Importing Packages
import numpy as np
import random
from numpy.random import default_rng, randint
import igraph


#Function for topologically sorting a DAG
def topological_sort(vertices, edges):
    EDGES = [e for e in edges]
    sources = []
    for i in vertices:
        t = 0
        for e in EDGES:
            if e[1] == i:
                t = t+1
        if t == 0:
            sources.append(i)
    order = []
    T = [i for i in sources]
    while len(T) != 0:
        v = T[0]
        order.append(v)
        T.remove(v)
        for j in vertices:
            for e in EDGES:
                if e == [v, j]:
                    EDGES.remove(e)
                    h = 0
                    if len(EDGES) != 0:
                        for e in EDGES:
                            if e[1] == j:
                                h = h + 1
                    if h == 0:
                        T.append(j)
    return order

# Function for Generating True Oriented Tree and Gaussian Samples with independent standard normal errors
def gen_samples(num_nodes, num_samples, seed = 57):
    random.seed(seed)
    rng = default_rng(seed)
    prufer = [random.randint(0,num_nodes-1) for i in range(num_nodes-2)]
    deg_seq = np.ones(num_nodes)
    for i in prufer:
        deg_seq[i] = deg_seq[i]+1
    edges = list()
    for i in range(num_nodes-2):
        for j in range(num_nodes):
            if deg_seq[j] == 1:
                edges.append([prufer[i], j])
                deg_seq[prufer[i]] = deg_seq[prufer[i]]-1
                deg_seq[j] = deg_seq[j]-1
                break
    final_edge = []
    for i in range(num_nodes):
        if deg_seq[i] == 1:
            final_edge.append(i)
    edges.append(final_edge)
    edges = [sorted(e) for e in edges]
    edge_turns = [random.randint(0,1) for i in range(len(edges))]
    for i in range(len(edges)):
        if edge_turns[i] == 1:
            edges[i] = [edges[i][1], edges[i][0]]
    order = topological_sort(range(num_nodes), edges)
    matrix = np.zeros((num_nodes, num_nodes), bool)
    edge_mask = [[order.index(e[0]), order.index(e[1])] for e in edges]
    for em in edge_mask:
        matrix[em[0]][em[1]] = True
    weights = np.zeros_like(matrix, float)
    weights[matrix] = rng.random(len(edges)) * 2 - 1
    means = 0
    st_dvs = 1
    samples = rng.normal(means, st_dvs, (num_samples, num_nodes))
    # print(samples[:, np.argsort(order)])
    for feature, parents in zip(samples.T, weights.T):
        feature += samples @ parents
    samples_perm = samples[:, np.argsort(order)]
    return edges, order, matrix, weights, samples_perm

def struct_hamming_sim(g, h):
    num_nodes = len(g)
    num_poss_edges = num_nodes**2 - num_nodes
    sim = (g == h).sum() - num_nodes
    return sim / (num_poss_edges)

def true_pos(g,h): # h is the true graph
    num_nodes = len(h)
    t = 0
    for r_idx in range(num_nodes):
        for c_idx in range(num_nodes):
            if h[r_idx][c_idx] == True:
                if g[r_idx][c_idx] == True:
                    t = t + 1
    return t

def false_pos(g,h): # h is the true graph
    num_nodes = len(h)
    t = 0
    for r_idx in range(num_nodes):
        for c_idx in range(num_nodes):
            if g[r_idx][c_idx] == True:
                if h[r_idx][c_idx] == False:
                    t = t + 1
    return t

def to_igraph(g, n): # g is a tree given as a list of edges and n is the number of vertices
    tuple_edges = [tuple(e) for e in g]
    ig = igraph.Graph()
    ig.add_vertices(range(n))
    ig.add_edges(tuple_edges)
    return ig

# Example on six nodes with 10 samples. Changing the seed changes the graph and the samples.
# Default seed is the Grothendeick prime.
# g = gen_samples(4,2,1312)
# print(g[0])
# print(g[1])
# print(g[2])
# print(g[3])
# print(g[4])
# print(to_igraph(g[0],4))
#
# h = to_igraph(g[0], 4)
# a = h.get_adjacency()
# print(a)

