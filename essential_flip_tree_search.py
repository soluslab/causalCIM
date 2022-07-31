#!/usr/bin/env python
# coding: utf-8    

import igraph
import ges
import numpy as np
import networkx as nx
from networkx.algorithms import tree
from scipy.stats import chi2_contingency


# Given a subtree (as a list of vertices)
# checks whether turning this subtree corresponds 
# to an essential flip. Optional check_input can 
# be set to false to improve speed if the user is 
# sure that the input is valid.

def validate_essential_flip(input_graph, input_subtree, check_input = True):
    if check_input:
        if not is_subtree(input_graph, input_subtree):
            return False
    if len(input_subtree) < 2:
        return False
    delta_st = delta_subtree(input_graph, input_subtree, check_input = False)
    if len(delta_st) < 1:
        return False
    for edge in input_graph.get_edgelist():
        if edge[0] in delta_st and edge[1] in delta_st:
            if not is_essential_edge_tree(input_graph, edge) or not is_essential_edge_tree(input_graph, edge, flip_subgraph=input_subtree):
                return False
    return True



# Checks whether a specific edge is essential in a tree
# additional option of considering the induced subgraph of a 
# set of nodes being reversed.

def is_essential_edge_tree(input_graph, input_edge, flip_subgraph = []):
    # If the edge is reversed
    if input_edge[0] in flip_subgraph and input_edge[1] in flip_subgraph:
        # Here we need to follow from input_edge[0] as 
        # opposed from input_edge[1]
        pa_size = len([i for i in input_graph.predecessors(input_edge[0]) if i not in flip_subgraph]) + len([i for i in input_graph.successors(input_edge[0]) if i in flip_subgraph]) 
        if pa_size > 1:
            return True
        if pa_size == 0:
            raise ValueError(str(input_edge) + " does not seem to be an edge of your graph.")
        return is_essential_edge_tree_help(input_graph, input_edge[1], flip_subgraph)
    # If the edge is not reversed
    else:
        if input_edge[1] in flip_subgraph:
            pa_size = len([i for i in input_graph.predecessors(input_edge[1]) if i not in flip_subgraph]) + len([i for i in input_graph.successors(input_edge[1]) if i in flip_subgraph]) 
        else:
            pa_size = len(input_graph.predecessors(input_edge[1]))
        if (pa_size > 1):
            return True
        if (pa_size == 0):
            raise ValueError(str(input_edge) + " does not seem to be an edge of your graph.")
        return is_essential_edge_tree_help(input_graph, input_edge[0], flip_subgraph)

def is_essential_edge_tree_help(input_graph, input_vertex, flip_subgraph):
    if input_vertex in flip_subgraph:
        pa = [i for i in input_graph.predecessors(input_vertex) if i not in flip_subgraph] + [i for i in input_graph.successors(input_vertex) if i in flip_subgraph]
    else:
        pa = input_graph.predecessors(input_vertex)
    if len(pa) > 1:
        return True
    if len(pa) == 0:
        return False
    return is_essential_edge_tree_help(input_graph, pa[0], flip_subgraph)



def span(input_graph, input_vertexset):
    subtree_ret = list(range(input_graph.vcount()))
    check = True
    while check:
        check = False
        for node in subtree_ret.copy():
            if (not node in input_vertexset) and len([i for i in input_graph.neighbors(node) if not i in subtree_ret]) < 2:
                subtree_ret.remove(node)
    return subtree_ret



# Checks whether the induced subgraph 
# is a subtree (or is disconnected)

def is_subtree(input_graph, input_vertexset):
    if not input_graph.is_tree(mode = 'all'):
        raise ValueError("is_subtree can only deal with trees")
    if input_graph.is_directed():
        graph_copy=input_graph.copy()
        graph_copy.to_undirected()
    else:
        graph_copy=input_graph.copy()
    if len(input_vertexset) == 0:
        return True
    if graph_copy.induced_subgraph(input_vertexset).ecount() == len(input_vertexset)-1:
        return True
    return False


# Finds all non-empty subtrees and returns them as 
# lists of the vertices they contain.

def all_subtrees(input_graph, check_input = True):
    if check_input:
        if not input_graph.is_tree() or input_graph.is_directed():
            raise ValueError("all_subtrees only implemented for undirected trees")
    v = 0
    subtrees_w_v = all_subtrees_w_node(input_graph, v, -1)
    subtrees_no_v = all_subtrees_no_node(input_graph, v, -1)
    return subtrees_w_v + subtrees_no_v
    
def all_subtrees_w_node(input_graph, start_node, ignore_node):
    subtrees_list = [[start_node]]
    order = input_graph.bfs(0)[0]
    for node in order:
        if node != ignore_node and node != start_node:
            ne = [i for i in input_graph.neighbors(node) if i != ignore_node]
            for s in subtrees_list.copy():
                if len([i for i in ne if i in s]) > 0:
                    temp = s.copy()
                    temp.append(node)
                    subtrees_list.append(temp)
                    
    return subtrees_list        

def all_subtrees_no_node(input_graph, start_node, ignore_node):
    ne_set = [i for i in input_graph.neighbors(start_node) if i != ignore_node]
    if len(ne_set) == 0:
        return []
    subtrees_w_v = []
    subtrees_no_v = []
    for node in ne_set:
        subtrees_w_v += all_subtrees_w_node(input_graph, node, start_node)
        subtrees_no_v += all_subtrees_no_node(input_graph, node, start_node)
    return subtrees_w_v + subtrees_no_v



# Finds the Delta set

def delta_set(input_graph, input_subtree, check_input = True):
    if check_input:
        if not is_subtree(input_graph, input_subtree):
            raise ValueError("Need a valid subtree for delta_set")
    delta = []
    for node in input_subtree.copy():
        paG = [i for i in input_graph.predecessors(node) if i in input_subtree]
        paH = [i for i in input_graph.successors(node) if i in input_subtree]
        pa = [i for i in input_graph.predecessors(node) if not i in input_subtree]
        if len(pa) > 0:
            if len(paG) + len(paH) > 0:
                delta.append(node)
        else:
            if len(paG) > 1 or len(paH) > 1:
                delta.append(node)
    return delta




# Finds the span of the Delta set inside a given subtree

def delta_subtree(input_graph, input_subtree, check_input = True):
    if check_input:
        if not is_subtree(input_graph, input_subtree):
            raise ValueError("delta_subtree only implemented if given a complete subtree")
    delta = delta_set(input_graph, input_subtree, check_input = False)          
    return span(input_graph, delta)

def reverse_subtree(input_graph, input_subtree, check_input = True):
    if check_input:
        if not is_subtree(input_graph, input_subtree):
            raise ValueError("reverse_subtree only implemented if given a complete subtree")
    return_graph = input_graph.copy()
    for edge in input_graph.get_edgelist():
        if edge[0] in input_subtree and edge[1] in input_subtree:
            return_graph.delete_edges([edge])
            return_graph.add_edges([(edge[1], edge[0])])
    return return_graph

# This function should be renamed
def essential_flip_search(input_skeleton, input_score):
    if input_skeleton.is_directed():
        ske = input_skeleton.copy()
        ske.to_undirected()
    else:
        ske = input_skeleton.copy()
    st = all_subtrees(ske)
    gd = ske.copy()
    gd.to_directed(mode='random')
    max_score = input_score.full_score(gd.get_adjacency_sparse().toarray())
    while True:
        potential_moves = [reverse_subtree(gd, i) for i in st if validate_essential_flip(gd, i)]
        scores = [input_score.full_score(t.get_adjacency_sparse().toarray()) for t in potential_moves]
        ind = np.argmax(scores)
        if scores[ind] > max_score:
            gd = potential_moves[ind]
            max_score = scores[ind]
        else:
            break 
    return gd, max_score

# Convert a graph given as a list of edges to an igraph object
def to_igraph(g, n): # g is a tree given as a list of edges and n is the number of vertices
    tuple_edges = [tuple(e) for e in g]
    ig = igraph.Graph()
    ig.add_vertices(range(n))
    ig.add_edges(tuple_edges)
    return ig

# Learning a tree skeleton from data:
# Function for conditional expectations
def phi_func(node1, node2, data, x):
    # x a realization of node1
    vals = data[data[:,node1] == x][:, node2].mean()
    return(vals)

# Function for computing edge weights from data:
def emp_edge_weight(node1, node2, data):
    df_nodes = data[:, [node1, node2]]
    n_rows = df_nodes.shape[0]
    devs = np.array([df_nodes[i, 1] - phi_func(node1, node2, data, df_nodes[i, 0]) for i in range(n_rows)])
    var = devs.var()
    var_node2 = data[:, node2].var()
    return(((1/2)*np.log(var/var_node2)))

# Function for computing empirical mutual information:
def calc_MI(data, x, y, bins):
    c_xy = np.histogram2d(data[:, x], data[:, y], bins)[0] + np.full([bins, bins], 0.0000000001, float)
    g, p, dof, expected = chi2_contingency(c_xy, lambda_="log-likelihood")
    mi = 0.5 * g / c_xy.sum()
    return mi

def MI_MWST(data, bins=5, method='kruskal'):
    n = data.shape[1]
    G_data = nx.Graph()
    for i in range(n):
        for j in range(i):
            G_data.add_edge(j, i, weight=-calc_MI(data, j, i, bins))
    min_weight_tree_data = tree.minimum_spanning_edges(G_data, algorithm=method, data=False)
    min_weight_tree_edge_list = [list(e) for e in min_weight_tree_data]
    ig_min_weight_tree = to_igraph(min_weight_tree_edge_list, n)
    return (ig_min_weight_tree)

# Function for producing minimum weight spanning tree. Valid methods are 'kruskal, 'prim', or 'boruvka'.
# Default method is 'kruskal'.
def min_weight_tree(data, method='kruskal'):
    edge_weights = np.empty((11, 11), float)
    n = data.shape[1]
    for node1 in range(n):
        for node2 in range(n):
            if node1 == node2:
                continue
            edge_weights[node1, node2] = emp_edge_weight(node1, node2, data)
    G_data = nx.MultiGraph()
    for node1 in range(n):
        for node2 in range(n):
            if node1 == node2:
                continue
            G_data.add_edge(node1, node2, weight=edge_weights[node1, node2])
    min_weight_tree_data = tree.minimum_spanning_edges(G_data, algorithm=method, data=False)
    min_weight_tree_edge_list = [list(e[:-1]) for e in min_weight_tree_data]
    ig_min_weight_tree = to_igraph(min_weight_tree_edge_list, n)
    return(ig_min_weight_tree)

# EFT causal discovery algorithm
def eft(data, skeleton = "none", bins = 5):
    bic = ges.scores.GaussObsL0Pen(data)
    if skeleton == "none":
        # learned_tree, learned_score = essential_flip_search(min_weight_tree(data), bic)
        learned_tree, learned_score = essential_flip_search(MI_MWST(data, bins), bic)
    else:
        learned_tree, learned_score = essential_flip_search(skeleton, bic)
    return learned_tree, learned_score

# # Here is some test code
#
# import sempler
# import numpy as np
#
# # Give us a random tree
# g = igraph.Graph.Tree_Game(n = 4)
# # Randomly direct this tree
# gd = g.copy()
# gd.to_directed(mode = 'random')
# # Print the tree
# print(gd)
#
# # Give us a random tree
# g = igraph.Graph.Tree_Game(n = 4)
#
# # Randomly direct this tree
# gd = g.copy()
# gd.to_directed(mode = 'random')
#
# # Print the tree
# print(gd)
#
# # Generate observational data from a Gaussian SCM using sempler
# # See documentation for the 'ges' package for source.
# A = gd.get_adjacency_sparse().toarray()
# W = A * np.random.uniform(1, 2, A.shape) # sample weights
# data = sempler.LGANM(W,(1,2), (1,2)).sample(n=5000)
#
# # Run GES with the Gaussian BIC score
# estimate, score = ges.fit_bic(data, A0 = g.get_adjacency_sparse().toarray(), phases = ['turning'])
#
# print(estimate)
#
# # Run essential_flip_search with the same BIC score
# # bic = ges.scores.GaussObsL0Pen(data)
# # essential_flip_estimate, essential_flip_score = essential_flip_search(g, bic)
# essential_flip_estimate, essential_flip_score = eft(data, g)
# est_wo_skel, score_wo_skel = eft(data)
# print(essential_flip_estimate)
# print(est_wo_skel)




