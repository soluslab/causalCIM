# Importing Packages
import numpy as np
import random
from numpy.random import default_rng
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

# Permutation matrix for a given order
def permmat(order):
    n = len(order)
    id_mat = np.eye(n)
    pmat = np.zeros((n,n),float)

    for i in range(n):
        pmat[i, :] = id_mat[order[i], :]

    return np.transpose(pmat)

# Function for generating true oriented tree and Gaussian samples with independent standard normal errors
def gen_samples(num_nodes, num_samples, seed = 57):
    np.random.seed(seed)
    rng = default_rng(seed)
    prufer = [random.randint(0, num_nodes - 1) for i in range(num_nodes - 2)]
    deg_seq = np.ones(num_nodes)
    for i in prufer:
        deg_seq[i] = deg_seq[i]+1
    edges = list()
    for i in range(num_nodes-2):
        for j in range(num_nodes):
            if deg_seq[j] == 1:
                edges.append([prufer[i], j])
                deg_seq[prufer[i]] = deg_seq[prufer[i]] - 1
                deg_seq[j] = deg_seq[j] - 1
                break
    final_edge = []
    for i in range(num_nodes):
        if deg_seq[i] == 1:
            final_edge.append(i)
    edges.append(final_edge)
    edges = [sorted(e) for e in edges]
    edge_turns = [random.randint(0, 1) for i in range(len(edges))]
    for i in range(len(edges)):
        if edge_turns[i] == 1:
            edges[i] = [edges[i][1], edges[i][0]]
    order = topological_sort(range(num_nodes), edges)
    matrix = np.zeros((num_nodes, num_nodes), bool)
    edge_mask = [[order.index(e[0]), order.index(e[1])] for e in edges]
    for em in edge_mask:
        matrix[em[0]][em[1]] = True

    weights = np.zeros_like(matrix, float)
    signs = (-1)**(np.random.binomial(n=1, p=1/2, size=len(edges)))
    lambdas = np.random.uniform(size=len(edges), low=0.25, high=1)
    weights[matrix] = np.transpose(np.array([signs[i]*lambdas[i] for i in range(len(edges))]))
    permMatrix = permmat(order)
    w_mat = np.linalg.inv(np.eye(num_nodes) - np.transpose(np.matmul(np.matmul(permMatrix, weights), np.transpose(permMatrix))))
    means = 0
    st_dvs = 1

    samples = rng.normal(means, st_dvs, (num_samples, num_nodes))
    samples = np.transpose(np.matmul(w_mat, np.transpose(samples)))

    adjMat = np.matmul(np.matmul(permMatrix, matrix), np.transpose(permMatrix)) + np.eye(num_nodes)

    return edges, order, adjMat, weights, samples


#######################################################
### Functions for manipulating and analyzing graphs ###
#######################################################

# Convert a list of edges to an igraph object
def to_igraph(g, n): # g is a tree given as a list of edges and n is the number of vertices
    tuple_edges = [tuple(e) for e in g]
    ig = igraph.Graph()
    ig.add_vertices(range(n))
    ig.add_edges(tuple_edges)
    return ig

def getPatternFromAdjMat(graph):
    n = graph.shape[0]

    skel = graph + np.transpose(graph) - np.eye(n)
    v_structures = np.zeros([n, n, n], int)

    for i in range(n):
        Lgraph = [j for j in range(n) if graph[j][i] == 1]
        Lgraph.remove(i)

        if len(Lgraph) > 1:
            for k in Lgraph:
                for t in Lgraph:
                    if k == t:
                        continue
                    else:
                        v_structures[i][k][t] = 1

    return skel, v_structures

def getPatternFromCIM(CIMlinObj):
    vec_result, skeleton, coords = CIMlinObj
    n = skeleton.shape[0]

    v_structures = np.zeros([n, n, n], int)

    c = len(coords)
    for i in range(c):
        if vec_result.x[i] == 1 and len(coords[i]) == 3:
            [a,b,c] = coords[i]
            [ab, ac, bc] = [skeleton[a][b], skeleton[a][c], skeleton[b][c]]
            if ab == 0:
                v_structures[c][a][b] = 1
                v_structures[c][b][a] = 1
            elif ac == 0:
                v_structures[b][a][c] = 1
                v_structures[b][c][a] = 1
            elif bc == 0:
                v_structures[a][b][c] = 1
                v_structures[a][c][b] = 1

    return skeleton, v_structures

def getPatternAdjMat(pattern):
    adjMatrix = pattern[0]
    n = adjMatrix.shape[0]
    vs = pattern[1]

    for i in range(n):
        for j in range(n):
            for k in range(n):
                if vs[i][j][k] == 1:
                    adjMatrix[i][k] = 0
                    adjMatrix[i][j] = 0
    return adjMatrix

def acc(graph, true_graph): #input as patterns

    graph_skel, graph_v = graph
    true_skel, true_v = true_graph

    n = graph_skel.shape[0]

    skel_accuracy = 0
    for i in range(n):
        for j in range(n):
            if graph_skel[i][j] == true_skel[i][j]:
                skel_accuracy += 1
    skel_accuracy = (skel_accuracy - n)/(n*(n-1))

    v_accuracy = 0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if k <= j:
                    continue
                else:
                    if graph_v[i][j][k] == true_v[i][j][k]:
                        v_accuracy += 1

    v_accuracy = v_accuracy / ((n**2 * (n-1)) / 2)

    return skel_accuracy, v_accuracy


