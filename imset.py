
# coding: utf-8

# ### Imset
# 
# This code is for calculating characteristic 
# imsets of DAG-models. Throughout the term imset
# is used for short as characteristic imset.
# This is a (0,1)-vector encoding of a Markov
# equivalence class of DAGs.
# 
# We implement the Greedy CIM algorithm for learning
# BIC-optimal imsets.  We also implement a hybrid
# version called skeletal greedy CIM which first learns
# an undirected graph via CI-tests and then searches for
# the BIC-optimal imset with the given skeleton.
# We also implement a couple of different variants.
# For example we have a breadth-first-search version
# as well as a depth-first-search.


# Import important modules

# For graph handling. 
# To install, visit: https://igraph.org/c/ and then https://igraph.org/python/
import igraph

# For math
import math
import scipy

# For testing the algorithms on random graphs.
import random

# For Warnings 
import warnings

# Because usual copies of list of lists does not work nice.
import copy



# Load rpy2 and relevant packages to score BIC-function,
# and to run different algorithms from the pcalg and bnlearn 
# library.

import rpy2

import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

from rpy2.robjects.packages import importr

rbase = importr("base")
bnlearn = importr("bnlearn")
pcalg = importr("pcalg")
# Rgraphviz = importr("Rgraphviz")
rgraph = importr("igraph")
rstats = importr("stats")
rutils = importr("utils")

import numpy;


# For running greedySP

import causaldag as cd
import numpy as np
from conditional_independence import MemoizedCI_Tester, partial_correlation_suffstat, partial_correlation_test


# For plotting the data

import matplotlib
import matplotlib.pyplot
import matplotlib.markers
import re

matplotlib.pyplot.style.use('classic')


# Import the non-essential modules

# For printing time
import time


# Produce all sets of size 2 or greater in a list.
# The return list is ordered in colexiographic order.
# INPUT: a set
# RETURNS: all sets of cardinality greater or equal to 2 as a list.
def coordinatelist(input_set):
    assert(isinstance(input_set, set))
    size = len(input_set)
    ret = powerset(input_set)
    
    # Can be rewritten to be more efficient.
    temp_list = ret.copy()
    for i in input_set:
        ret.remove({i})
    ret.pop(0)
     
    return ret

# Produce all sets of size 2 and 3 in a list.
# The return list is ordered in colexiographic order.
# INPUT: a set
# RETURNS: all sets of cardinality 2 or 3
def small_coordinatelist(input_set):
    assert(isinstance(input_set, set))
    
    size = len(input_set)
    ret = [set({})]
    for element in input_set:
        temp_list = ret.copy()
        for counter in range(len(temp_list)):
            if (len(temp_list[counter]) < 3):
                ret.append(temp_list[counter].union({element}));
    
    temp_list = ret.copy()
    for i in temp_list:
        if (len(i) < 2):
            ret.remove(i)
     
    return ret


# Produce the power set as a list
# The return list is ordered in colexiographic order.
# INPUT: a set.
# RETURNS: a list of all subsets.
def powerset(input_set):
    assert(isinstance(input_set, set))
    ret = [set()]
    for element in input_set:
        temp_list = ret.copy()
        for counter in range(len(temp_list)):
            ret.append(temp_list[counter].union({element}))
    return ret


# Get the parent set and the children as sets instead of lists
# INPUT: A directed graph and a node.
# RETURNS: The parent set or the children set.

def parents(input_graph, input_node):
    return set(input_graph.predecessors(input_node))

def children(input_graph, input_node):
    return set(input_graph.successors(input_node))


# Returns the value of the characteristic imset for the graph in the given coordinate.
# INPUT: a DAG and a set
# RETURNS: a 0/1 value

def imset_coordinate(input_graph, input_coordinate_set):
    copy_set = input_coordinate_set.copy()
    node = next(iter(copy_set))
    temp_set = children(input_graph, node).intersection(input_coordinate_set)
    while (len(temp_set) > 0):
        node = list(temp_set)[0]
        temp_set = children(input_graph, node).intersection(input_coordinate_set)
    copy_set.discard(node)
    if (parents(input_graph, node).issuperset(copy_set)):
        return 1
    return 0


# Calls the "coordinatelist" function and calculates the imset value for each element.
# INPUT: a dag
# RETURNS: a list of lists. The inner lists consists of [set, imset_coordinate(graph, set)]

def imset(input_graph):
    assert (input_graph.is_dag())
    ret = []
    vertex_set = set(range(input_graph.vcount()))
    coordinate_list = coordinatelist(vertex_set)
    for i in range(len(coordinate_list)):
        ret.append([coordinate_list[i], imset_coordinate(input_graph, coordinate_list[i])])
    return ret

# Calls the "smallcoordinatelist" function and calculates the imset value for all elements of size 2 and 3.
# Useful for checks where the full imset is not required ex. Markov equivalence.
# INPUT: a dag
# RETURNS: a list of lists. The inner lists concists of [set, imset_coordinate(graph, set)]

def small_imset(input_graph):
    assert (input_graph.is_dag())
    ret=[]
    vertex_set = set(range(input_graph.vcount()))
    coordinate_list = small_coordinatelist(vertex_set)
    for i in range(len(coordinate_list)):
        ret.append([coordinate_list[i], imset_coordinate(input_graph, coordinate_list[i])])
    return ret
    

## Conversion functions


# Functions for cropping imsets.
# INPUT: An imset and a size.
# RETURNS: The imset 

# Only takes full sized imsets
def imset2small_imset(input_imset):
    coordinate_list = small_coordinatelist(input_imset[-1][0])
    ret_imset = []
    for i in coordinate_list:
        ret_imset.append([i, input_imset[set2imset_pos(i)][1]])
    return ret_imset

def imset_cutof(input_imset, size = 3):
    ret = input_imset.copy()
    for i in input_imset:
        if (len(i[0]) > size):
            ret.remove(i)
    return ret


# Function for turning a small_imset to a full sized imset.
# Very slow, do not recomend using if it can be avoided.
# INPUT: A small imset
# RETURNS: A full sized imset

def small_imset2imset(input_imset):
    return imset(pdag2dag(small_imset2pdag(input_imset)))


# Separates the imset vector into two lists, preserving the order
# INPUT: a imset or smallimset
# RETURNS: two lists

def imset2vec(input_imset):
    cordinatevalue = []
    coordinatelist = []
    for i in range(len(input_imset)):
        cordinatevalue.append(input_imset[i][1])
        coordinatelist.append(input_imset[i][0])
    return coordinatelist, cordinatevalue


# Implementation of "A simple algorithm to construct a consistent
# extension of a partially oriented graph".
# https://ftp.cs.ucla.edu/pub/stat_ser/r185-dor-tarsi.pdf
# INPUT: A partially directed graph
# RETURNS: A DAG

def pdag2dag(input_graph):
    nodes = input_graph.vcount()
    graph_return = input_graph.copy()
    graph_temp = input_graph.copy()
    
    current_node = 0
    while (current_node < nodes):
        ch_set = children(graph_temp, current_node)
        pa_set = parents(graph_temp, current_node)
        ne_set = pa_set.intersection(ch_set)
        # If it is a sink
        if (len(ch_set) + len(pa_set) == 0):
            pass
        elif (len(ch_set.difference(pa_set)) == 0):
            condition_b = True
            for i in ne_set:
                if (not parents(graph_temp, i).union(children(graph_temp, i), set({i})).issuperset(pa_set)):
                    condition_b = False;
            if (condition_b):
                for i in ne_set:
                    graph_return.delete_edges([(current_node, i)])
                for i in ch_set:
                    graph_temp.delete_edges([(current_node, i)])
                for i in pa_set:
                    graph_temp.delete_edges([(i, current_node)])
                current_node = -1
        current_node += 1
            
    if (not graph_return.is_dag()):
        raise ValueError("In: pdag2dag. No extension exists.")
    return graph_return;


# Takes a full size imset and constructs a pDAG based on 
# the 2- and 3-sets and outputs said pDAG.
# That graph is graphically equivalent to all graphs in the MEC.
# Undirected edges are portrayed by mutual edges.
# Raises ValueError if the 2- and 3-sets are not consistent with a DAG
# INPUT: An imset
# RETURNS: A partially directed graph

def imset2pdag(input_imset):
    if input_imset == []:
        raise ValueError("Imset cannot be empty.")
    nodes = len(input_imset[-1][0])
    graph = igraph.Graph(nodes, directed = True)
    
    # Add edges according to the 2-sets
    for i in range(nodes):
        for j in range(i+1, nodes):
            if (input_imset[set2imset_pos({i,j})][1] == 1):
                graph.add_edge(i,j)
                graph.add_edge(j,i)
    
    # Add v-structures according to the 3-sets
    for i in range(nodes):
        for j in range(i+1, nodes):
            for k in range(j+1,nodes):
                if (input_imset[set2imset_pos({i,j,k})][1] == 1):
                    
                    edge_ij = input_imset[set2imset_pos({i,j})][1]
                    edge_ik = input_imset[set2imset_pos({i,k})][1]
                    edge_jk = input_imset[set2imset_pos({j,k})][1]
                    check = False
                    if edge_ij+edge_ik+edge_jk < 2:
                        pass
                    elif edge_ij+edge_ik+edge_jk == 3:
                        check = True
                    elif (edge_ij == 0):
                        check = True
                        try:
                            graph.delete_edges([(k,i)])
                        except ValueError:
                            pass
                        try:
                            graph.delete_edges([(k,j)])
                        except ValueError:
                            pass
                    elif (edge_ik == 0):
                        check = True
                        try:
                            graph.delete_edges([(j,i)])
                        except ValueError:
                            pass
                        try:
                            graph.delete_edges([(j,k)])
                        except ValueError:
                            pass
                    elif (edge_jk == 0):
                        check = True
                        try:
                            graph.delete_edges([(i,j)])
                        except ValueError:
                            pass
                        try:
                            graph.delete_edges([(i,k)])
                        except ValueError:
                            pass
                    
                    # If we have not created a v-structure or the imset is complete, somethings wrong.
                    if not check:
                        raise ValueError("The imset is not consistent with a pDAG")
                    
                else:
                    edge_ij = input_imset[set2imset_pos({i,j})][1]
                    edge_ik = input_imset[set2imset_pos({i,k})][1]
                    edge_jk = input_imset[set2imset_pos({j,k})][1]
                    if edge_ij+edge_ik+edge_jk == 3:
                        raise ValueError("The imset is not consistent with a pDAG")
    # We need to do a final check to see if everything is alright.
    # We need small_imset(pdag2dag(graph)) = imset2small_imset(input_imset)
    if not is_equal(small_imset(pdag2dag(graph)), imset2small_imset(input_imset)):
        raise ValueError("The imset is not consistent with a DAG")
    return graph

# Takes a small_imset and constructs a pDAG.
# That graph is graphically equivalent to all graphs in the MEC.
# Undirected edges are portrated by mutual edges.
# Raises ValueError if the small_imset is not consistent with a DAG
# INPUT: A small imset
# RETURNS: A partially directed graph

def small_imset2pdag(input_imset):
    nodes_set = set()
    for i in input_imset:
        nodes_set.update(i[0]);
    nodes = len(nodes_set);
    graph = igraph.Graph(nodes, directed = True)
    
    # Add edges according to the 2-sets
    for i in range(nodes):
        for j in range(i+1, nodes):
            if (input_imset[set2small_imset_pos({i,j})][1] == 1):
                graph.add_edge(i,j)
                graph.add_edge(j,i)
    
    # Add v-structures according to the 3-sets
    for i in range(nodes):
        for j in range(i+1, nodes):
            for k in range(j+1,nodes):
                if (input_imset[set2imset_pos({i,j,k})][1] == 1):
                    
                    edge_ij = input_imset[set2small_imset_pos({i,j})][1]
                    edge_ik = input_imset[set2small_imset_pos({i,k})][1]
                    edge_jk = input_imset[set2small_imset_pos({j,k})][1]
                    check = False
                    if edge_ij+edge_ik+edge_jk < 2:
                        pass
                    elif edge_ij+edge_ik+edge_jk == 3:
                        check = True
                    elif (edge_ij == 0):
                        check = True
                        try:
                            graph.delete_edges([(k,i)])
                        except ValueError:
                            pass
                        try:
                            graph.delete_edges([(k,j)])
                        except ValueError:
                            pass
                    elif (edge_ik == 0):
                        check = True
                        try:
                            graph.delete_edges([(j,i)])
                        except ValueError:
                            pass
                        try:
                            graph.delete_edges([(j,k)])
                        except ValueError:
                            pass
                    elif (edge_jk == 0):
                        check = True
                        try:
                            graph.delete_edges([(i,j)])
                        except ValueError:
                            pass
                        try:
                            graph.delete_edges([(i,k)])
                        except ValueError:
                            pass
                    
                    # If we have not created a v-structure or the imset is complete, something is wrong.
                    if not check:
                        raise ValueError("The imset is not consistent with a pDAG")
                    
                else:
                    edge_ij = input_imset[set2imset_pos({i,j})][1]
                    edge_ik = input_imset[set2imset_pos({i,k})][1]
                    edge_jk = input_imset[set2imset_pos({j,k})][1]
                    if edge_ij+edge_ik+edge_jk == 3:
                        raise ValueError("The imset is not consistent with a pDAG")
    # We need to do a final check to see if everything is alright.
    # We need small_imset(pdag2dag(graph)) = imset2small_imset(input_imset)
    if not is_equal(small_imset(pdag2dag(graph)), input_imset):
        raise ValueError("The imset is not consistent with a DAG")
    return graph


# Takes a full size imset and returns a directed graph
# in the MEC. For efficiency this should be merged with
# imset2pdag as imset2pdag calls pdag2dag towards the end.
# INPUT: A full size imset
# RETURNS: A DAG

def imset2dag(input_imset):
    return pdag2dag(imset2pdag(input_imset))


# Takes a small_imset and returns a directed graph
# in the MEC. For efficiency this should be merged with
# small_imset2pdag as small_imset2pdag calls pdag2dag 
# towards the end.
# INPUT: A small size imset
# RETURNS: A DAG

def small_imset2dag(input_imset):
    return pdag2dag(small_imset2pdag(input_imset))


# Takes a set and returns the position
# in a full-sized imset as a int.
# INPUT: A set
# RETURNS: An int

def set2imset_pos(input_set):
    if (len(input_set) < 2):
        raise ValueError("Trying to find a position in a imset with size less than 2.\nSet has size:" + str(len(input_set)))
    ret = 0;
    max_value = 0
    for i in input_set:
        if max_value < i:
            max_value = i
    
    coordinate_value = 0;
    for i in input_set:
        coordinate_value += 2**i
    coordinate_value = coordinate_value- max_value -2
    return coordinate_value

# Takes a set and returns the position
# in a small_imset as a int.
# INPUT: A set
# RETURNS: An int

def set2small_imset_pos(input_set):
    if (len(input_set) == 2):
        j = max(input_set)
        i = min(input_set)
        coordinate_value = math.comb(j, 2) + math.comb(j, 3) + i + math.comb(i, 2)
    elif (len(input_set) == 3):
        k = max(input_set)
        i = min(input_set)
        j = list(input_set.difference({k,i}))[0]
        coordinate_value = math.comb(k, 2) + math.comb(k, 3) + j + math.comb(j, 2) + i
    else:
        raise ValueError("Trying to find a position in a small_imset with size not 2 or 3.\nSet has size:" + len(input_set))
    return coordinate_value


# Calculates imset1 - imset2. Returns lists A and B of sets 
# such that imset1 - imset2 = \sum_{S\in A}e_S - \sum_{S\in B}e_S
# if passed two graphs, return the corresponding for their imsets.
# INPUT: Two graphs/imsets.
# RETURNS: Two lists.

def imsetdif(input_graph1, input_graph2):
    if (isinstance(input_graph1, igraph.Graph)):
        imset1 = imset(input_graph1)
    else:
        imset1 = input_graph1;
    if (isinstance(input_graph2, igraph.Graph)):
        imset2 = imset(input_graph2)
    else:
        imset2 = input_graph2
    A=[]
    B=[]
    if (len(imset1)!=len(imset2)):
        raise ValueError("Imsets/graphs must be of equal size.\nGot imsets of size" + len(imset1) + " and " + len(imset2) + "\n" )
    for i in range(len(imset1)):
        if (imset1[i][1] == 1):
            if (imset2[i][1] == 0):
                A.append(imset1[i][0])
        else:
            if (imset2[i][1] == 1):
                B.append(imset1[i][0])
    return A,B


# Checks the value of the imset in the given coordinate.
# If you are sure that the imset is full-size and correct,
# set full_imset = True. Then the algorithm is quicker, 
# but does not check correctness. Same for small_imset.
# If given an invalid coordinate, returns -1.
# INPUT: An imset and a set
# RETURNS: An int

def imset_value(input_imset, input_set, full_imset = False, small_imset = False):
    if full_imset:
        return input_imset[set2imset_pos(input_set)][1]
    elif small_imset:
        return input_imset[set2small_imset_pos(input_set)][1]
    else:
        for i in input_imset:
            if (i[0] == input_set):
                return i[1]
    warnings.warn("Imset coordinate not found, returning -1")
    return -1


# A check to see whether or not two lists contains 
# the same elements regardless of order. Use only 
# when elements are unhashable and unsortable 
# (for example, sets).
# INPUT: 2 lists.
# RETURNS: A boolean stating whether they contain the same elements.

def set_lists_equal(input_list1, input_list2):
    copy1 = input_list1.copy()
    copy2 = input_list2.copy()
    for i in copy1:
        try:
            copy2.remove(i)
        except ValueError:
            return False
        
    if (len(copy2)==0):
        return True
    return False


# Checks if two graphs (or imsets) are Markov equivalent
# Does this via checking if the two small_imset are the same 
# or if given two imsets, if they are the same
# INPUT: Two imsets
# RETURNS: A bool

def is_markovequiv(input_graph1, input_graph2):
    if isinstance(input_graph1, igraph.Graph):
        imset1 = small_imset(input_graph1)
    else:
        imset1 = input_graph1.copy()
    if isinstance(input_graph2, igraph.Graph):
        imset2 = small_imset(input_graph2)
    else: 
        imset2 = input_graph2.copy()
        
    return is_equal(imset1, imset2)


# Checks if two imsets are the same.
# INPUT: Two imsets
# RETURNS: A bool

def is_equal(input_imset1, input_imset2):
    if imset2vec(input_imset1)[1] == imset2vec(input_imset2)[1] and input_imset1[-1][0] == input_imset2[-1][0]:
        return True
    return False


# Returns whether or not the pair (graph1, graph2)
# is an addition. Can take both graphs and imsets 
# as inputs.
# INPUT: Two gaphs/imsets (full size imsets only)
# RETURNS: A boolean

def is_addition(input_graph1, input_graph2):
    if isinstance(input_graph1, igraph.Graph):
        imset1 = imset(input_graph1)
    else:
        imset1 = input_graph1.copy();
    if isinstance(input_graph2, igraph.Graph):
        imset2 = imset(input_graph2)
    else: 
        imset2 = input_graph2.copy()
    A, B=imsetdif(imset1, imset2)
    if len(A)+len(B) == 1:
        return True
    return False


# Returns whether or not the pair (graph1, graph2)
# is a budding. verbose option for debugging.
# Can take both graphs and imsets as inputs
# INPUT: Two gaphs/imsets (full size imsets only)
# RETURNS: A boolean

def is_budding(input_graph1, input_graph2, verbose = False):
    if isinstance(input_graph1, igraph.Graph):
        imset1 = imset(input_graph1)
    else:
        imset1 = input_graph1.copy()
    if isinstance(input_graph2, igraph.Graph):
        imset2 = imset(input_graph2)
    else: 
        imset2 = input_graph2.copy()
    A, B = imsetdif(imset1, imset2)
    # Check that exactly one of the sets are empty, 
    # we don't have an addition and set that A is 
    # the non-empty one
    if len(A)+len(B) < 2:
        if verbose:
            print("False because 1") 
        return False
    if len(A) > 0:
        if len(B) > 0:
            if verbose:
                print("False because 2")
            return False
    else:
        A = B
        B = []
    # Now make sure that A indeed is a budding set
    # Make sure they contain no sets of size 2.
    for i in A:
        if len(i) < 3:
            if verbose:
                print("False because 3")
            return False
    # Find i, j and S*
    union_set = set()
    for i in A:
        union_set.update(i)
    intersection_set = union_set
    for i in A:
        intersection_set = intersection_set.intersection(i)
    if len(intersection_set) < 2 or len(intersection_set) > 3:
        if verbose:
            print("False because 4")
        return False
    nodei = -1
    nodej = -1
    if len(intersection_set) == 3:
        temp_list = list(intersection_set)
        if imset_value(imset1, set({temp_list[0], temp_list[1]})) == 0:
            nodei = temp_list[2]
            temp_list[2] = temp_list[0]
            temp_list[0]=nodei
        elif imset_value(imset1, set({temp_list[0], temp_list[2]})) == 0:
            nodei = temp_list[1]
            temp_list[1] = temp_list[0]
            temp_list[0]=nodei
        elif imset_value(imset1, set({temp_list[1], temp_list[2]})) == 0:
            nodei = temp_list[0]
        else: 
            if verbose:
                print("False because 5")
            return False
        for i in union_set.difference(intersection_set):
            if imset_value(imset1, set({i, temp_list[1]})) == 0:
                nodej = temp_list[2]
        if nodej == -1:
            nodej = temp_list[1]
    else:
        node1 = list(intersection_set)[0]
        node2 = list(intersection_set)[1]
        
        for i in union_set.difference(intersection_set):
            if imset_value(imset1, set({node1, i})) == 0:
                nodej = node1
                nodei = node2
                break
        if nodei == -1:
            for i in union_set.difference(intersection_set):
                if imset_value(imset1, set({node2, i})) == 0:
                    nodej = node2
                    nodei = node1
                    break
        if nodei == -1:
            if verbose:
                print("False because 6")
            return False
    S = union_set.copy().difference({nodei, nodej})
    
    ne_set = set()
    for i in S:
        if imset_value(imset1, {nodej, i}) == 1:
            ne_set.add(i)
    # If this is a budding we now have i, j, S and ne(i)\cap ne(j) 
    if verbose:
        print("i =", nodei)
        print("j =", nodej)
        print("S =", S)
        print("ne =", ne_set)
    
    if not imset_value(imset1, {nodei, nodej}) == 1:
        return False
    
    pow_set = powerset(S)
    pow_set.remove(set())
    
    for i in pow_set:
        if not imset_value(imset1, i.union({nodei})) == 1:
            if verbose: 
                print("False because 7", i.union({nodei}))
            return False
    
    
    bud_set = []
    for i in pow_set:
        if not i.issubset(ne_set):
            bud_set.append(i.union({nodei, nodej}))
    if verbose: 
        print(sorted(bud_set))
        print(sorted(A))
        
    if set_lists_equal(A, bud_set):
        return True
    
    if verbose: 
        print("False because default")
    return False


# Returns whether the pair (graph1, graph2) is a budding.
# verbose option for debugging
# Can take both graphs and imsets as inputs
# INPUT: Two graphs/imsets (full size imsets only)
# RETURNS: A boolean

def is_ges(input_graph1, input_graph2, verbose = False):
    if isinstance(input_graph1, igraph.Graph):
        imset1 = imset(input_graph1)
    else:
        imset1 = input_graph1.copy()
    if isinstance(input_graph2, igraph.Graph):
        imset2 = imset(input_graph2)
    else: 
        imset2 = input_graph2.copy()
    A, B = imsetdif(imset1, imset2);
    if verbose: 
        print(A, B)
    # Check that exactly one of the sets are empty, 
    # we don't have an addition and set that A is 
    # the non-empty one
    if len(A)+len(B) < 2:
        if verbose:
            print("False because 1") 
        return False
    if len(A) > 0:
        if len(B) >0:
            if verbose:
                print("False because 2")
            return False
    else:
        A = B
        B = []
    # Now make sure that A indeed is a budding set
    # Make sure they contain no sets of size 2.
    edge = [-1, -1]
    for i in A:
        if len(i) == 2:
            if edge[0] != -1:
                if verbose:
                    print("False because differs by more than one 2-set")
                return False
            else:
                temp_list = list(i)
                edge[0] = temp_list[0]
                edge[1] = temp_list[1]
    if edge[0] == -1:
            if verbose:
                print("False because differs no 2-set")
            return False
    
    union_set = set()
    for i in A:
        union_set.update(i)
    pow_set = powerset(union_set.difference(set(edge)))
    ges_set = []
    for i in pow_set:
        ges_set.append(i.union(set(edge)))
    if not set_lists_equal(ges_set, A):
        if verbose:
            print("False because 3")
        return False
    
    check = True
    pow_set.remove(set())
    for i in pow_set:
        if imset_value(imset1, i.union({edge[0]})) == 0:
            check = False
    
    if check:
        return True
    
    for i in pow_set:
        if imset_value(imset1, i.union({edge[1]})) == 0:
            if verbose:
                print("False because 3")
            return False
    
    return True


# Returns whether or not the pair (graph1, graph2)
# is a flip. verbose option for debugging
# Can take both graphs and imsets as options
# INPUT: Two gaphs/imsets (full size imsets only)
# RETURNS: A boolean

def is_flip(input_graph1, input_graph2, verbose = False):
    if isinstance(input_graph1, igraph.Graph):
        imset1 = imset(input_graph1)
    else:
        imset1 = input_graph1.copy()
    if isinstance(input_graph2, igraph.Graph):
        imset2 = imset(input_graph2)
    else: 
        imset2 = input_graph2.copy()
    A, B = imsetdif(imset2, imset1)
    if verbose: 
        print(A, B)
    if len(A) == 0 or len(B) == 0:
        if verbose: 
            print("False because 1")
        return False
    for i in A:
        if len(i) == 2:
            if verbose: 
                print("False because 2")
            return False
    for i in B:
        if len(i) == 2:
            if verbose: 
                print("False because 2")
            return False
    
    union_set1 = set()
    union_set2 = set()
    for i in A:
        union_set1.update(i)
    for i in B:
        union_set2.update(i)
    intersection_set = union_set1.copy()
    for i in B:
        intersection_set.intersection_update(i)
    
    if len(intersection_set) != 2:
        if verbose: 
            print("False because 3")
        return False
    
    ne_set = set()
    intersection_list = list(intersection_set)
    for i in union_set1.union(union_set2).difference(intersection_set):
        if imset_value(imset1, set({i, intersection_list[0]})) == 1 and imset_value(imset1, set({i, intersection_list[1]})) == 1:
            ne_set.add(i)
    if verbose: 
        print("ne_set =", ne_set)
    
    nodei = -1
    nodej = -1
    
    for i in union_set1.difference(ne_set).difference(intersection_set):
        if imset_value(imset1, set({i, intersection_list[0]})) == 0:
            nodej = intersection_list[0]
            nodei = intersection_list[1]
            break
    if nodei == -1:
        nodej = intersection_list[1]
        nodei = intersection_list[0]
    
    Si=union_set1.difference(intersection_set)
    Sj=union_set2.difference(intersection_set)
    # If this is a budding we should have i, j, Si, and Sj now.
    if verbose:
        print("i =", nodei)
        print("j =", nodej)
        print("Si =", Si)
        print("Sj =", Sj)
    
    
    pow_set_i = powerset(Si)
    pow_set_i.remove(set())
    for i in pow_set_i:
        if imset_value(imset1, i.union({nodei})) == 0:
            if verbose: 
                print("False because 4")
            return False
    pow_set_j = powerset(Sj)
    pow_set_j.remove(set())
    for i in pow_set_j:
        if imset_value(imset1, i.union({nodej})) == 0:
            if verbose: 
                print("False because 4")
            return False
    
    flip_set_pos = []
    flip_set_neg = []
    
    for i in pow_set_i:
        if not i.issubset(ne_set):
            flip_set_pos.append(i.union({nodei, nodej}))
    for i in pow_set_j:
        if not i.issubset(ne_set):
            flip_set_neg.append(i.union({nodei, nodej}))
    
    if verbose: 
        print(flip_set_pos)
        print(A)
        print(flip_set_neg)
        print(B)
    
    if set_lists_equal(A, flip_set_pos) and set_lists_equal(B, flip_set_neg):
        return True
    
    if verbose: 
        print("False because default")
    return False


# Check whether or not the imset corresponds to a 
# characteristic imset of a DAG
# INPUT: A list of list
# RETURNS: A boolean

def is_imset(input_imset):
    # Check that the 2/3-sets encode 
    # for a valid imset/DAG.
    try:
        temp_dag = imset2dag(input_imset)
    except ValueError:
        return False
    # If the imset would be consistent, then 
    # it must be the imset of temp_dag
    if imset2vec(imset(temp_dag))[1] == imset2vec(input_imset)[1]:
        return True
    return False


# Try all acyclic orientations of an undirected graph.
# Does this naively and does nothing to ignore markov equivalent graphs.
# INPUT: An undirected or partially directed graph.
# RETURNS: A list of imsets values for all acyclic orientations 

def try_acyclic_orientation(input_graph):
    imsetlist = []
    if(input_graph.is_directed()):
        print("The input to 'try_acyclic_orientation' was directed.\nWe will only direct the mutual edges.")
        dir_graph = input_graph.copy()
    else:
        dir_graph = input_graph.copy()
        dir_graph.to_directed()
    try_acyclic_orientation_help(dir_graph, imsetlist)
    return imsetlist


def try_acyclic_orientation_help(input_graph, input_imsetlist):
    if input_graph.is_dag():
        imset_temp = imset(input_graph)
        input_imsetlist.append(imset_temp)
    else:
        check = False
        edgelist=input_graph.get_edgelist()
        edge = edgelist[0]
        for i in range(len(edgelist)):
            if input_graph.is_mutual(edgelist[i]):
                edge = edgelist[i]
                i = len(edgelist)
                check = True
        if check:
            graph_copy1=input_graph.copy()
            graph_copy1.delete_edges([edge])
            try_acyclic_orientation_help(graph_copy1, input_imsetlist)
            del graph_copy1
            graph_copy2=input_graph.copy()
            graph_copy2.delete_edges([(edge[1], edge[0])])
            try_acyclic_orientation_help(graph_copy2, input_imsetlist)
            del graph_copy2


# Runs over all DAGs with a fixed number 
# of nodes and calculates the imset for all of them. 
# Notice that it will probably produce replicas of 
# most imsets.
# INPUT: An integer, number of nodes
# RETURNS: A list of imsets values for all DAGs 

def try_all_dags(input_nodes):
    imsetlist = []
    graph = igraph.Graph(input_nodes, directed = True)
    try_all_dags_help(graph, imsetlist, [0, 0])
    return imsetlist

def try_all_dags_help(input_graph, input_imsetlist, input_edge):
    if (input_edge[0] == input_graph.vcount()-1) and (input_edge[1] ==  input_graph.vcount()-1):
        input_imsetlist.append(imset(input_graph))
    else:
        input_edge[0] = input_edge[0] + 1
        if (input_edge[0] == input_graph.vcount()):
            input_edge[0] = 0
            input_edge[1] = input_edge[1] + 1
        try_all_dags_help(input_graph.copy(), input_imsetlist, input_edge.copy())
        input_graph.add_edges([input_edge])
        if input_graph.is_dag():
            try_all_dags_help(input_graph.copy(), input_imsetlist, input_edge.copy())



# Try all acyclic orientations of an undirected graph.
# Does this naively and does nothing to ignore markov equivalent graphs.
# INPUT: An undirected or partially directed graph.
# RETURNS: A list of small_imsets values for all acyclic orientations 

def small_try_acyclic_orientation(input_graph):
    imsetlist = []
    if input_graph.is_directed():
        dir_graph = input_graph.copy()
    else:
        dir_graph = input_graph.copy()
        dir_graph.to_directed()
    small_try_acyclic_orientation_help(dir_graph, imsetlist)
    return imsetlist

def small_try_acyclic_orientation_help(input_graph, input_imsetlist):
    if (input_graph.is_dag()):
        imset_temp = imset2vec(small_imset(graph))[1]
        input_imsetlist.append(imset_temp)
    else:
        check = False
        edgelist=input_graph.get_edgelist()
        edge = edgelist[0]
        for i in range(len(edgelist)):
            if (input_graph.is_mutual(edgelist[i])):
                edge = edgelist[i]
                i = len(edgelist)
                check = True
        if (check):
            graph_copy1=input_graph.copy()
            graph_copy1.delete_edges([edge])
            small_try_acyclic_orientation_help(graph_copy1, imsetlist)
            del graph_copy1
            graph_copy2=input_graph.copy()
            graph_copy2.delete_edges([(edge[1], edge[0])])
            small_try_acyclicorientation_help(graph_copy2, imsetlist)
            del graph_copy2


# Reads a csv-file of an adjacency matrix of a 
# graph and returns the graph. The csv-file is 
# expected to be written in the same format as
# the R function write.csv writes in. That is,
# the first row and column are the names of the 
# vertices and a non-zero value represent an edge.

def read_dag(input_file):
    data_matrix = numpy.genfromtxt(input_file, delimiter = ',')
    data_matrix = data_matrix[1:, 1:]
    size = len(data_matrix[0])
    pmatrix_int = numpy.zeros((size, size))
    for i in range(size):
        for j in range(size):
            pmatrix_int[i,j] = 1 if data_matrix[i,j] != 0 else 0
    ret = igraph.Graph.Adjacency(list(pmatrix_int))
    return ret


# Run the implementation of greedySP from uhlerlab/causaldag

def greedySP(input_data, input_cut_off = 1e-3):
    data_matrix = numpy.genfromtxt(input_data, delimiter=',')
    data_matrix = data_matrix[1:, 1:]
    nnodes = len(data_matrix[0])
    suffstat = partial_correlation_suffstat(data_matrix)
    ci_tester = MemoizedCI_Tester(partial_correlation_test, suffstat, alpha=input_cut_off)
    est_dag = cd.structure_learning.gsp(set(range(nnodes)), ci_tester, None, 1)
    # est_dag = cd.sparsest_permutation(set(range(nnodes)), ci_tester)
    est_cpdag = est_dag.cpdag()
    return pdag2dag(igraph.Graph.Adjacency(est_cpdag.to_amat()[0].tolist()))


### R-code


# Copies the graph to an r-object
# INPUT: A python igraph
# RETURNS: A R igraph object

def igraph2rgraph(input_graph):
    pmatrix = numpy.transpose(numpy.array(list(input_graph.get_adjacency()))).flatten()
    return rgraph.graph_from_adjacency_matrix(rpy2.robjects.r.matrix(pmatrix , nrow=input_graph.vcount(), ncol=input_graph.vcount()))


# Initiates and creates a BIC-score function to 
# be used with the score_BIC function.
# INPUT: A data matrix
# RETURNS: An score object

def initiate_BIC(input_data):
    data_matrix = numpy.genfromtxt(input_data, delimiter = ',')
    data_matrix = data_matrix[1:, 1:]
    rows = len(data_matrix)
    cols = len(data_matrix[0])
    r_data_matrix = rpy2.robjects.r.matrix(data_matrix , nrow=rows, ncol=cols)
    return_BIC = rpy2.robjects.r['new']("GaussL0penObsScore", r_data_matrix)
    return return_BIC


# Score a graph on the BIC
# INPUT: A score object and a graph
# RETURNS: The score

def score_BIC(input_BIC, input_graph):
    internal_BIC_function = rbase.__dict__["$"](input_BIC, "global.score")
    graph_matrix = rpy2.robjects.r.matrix(numpy.transpose(numpy.array(list(input_graph.get_adjacency()))).flatten() , nrow=input_graph.vcount(), ncol=input_graph.vcount())
    ret = internal_BIC_function(rpy2.robjects.r['as'](graph_matrix, "GaussParDAG"))
    return ret[0]


# Run the 'pcalg' function 'skeleton'
# INPUT: A score object
# RETURNS: An undirected python igraph 

def skeleton(input_BIC, input_cut_off = 0.01):
    suff_stat = rbase.list(C = rstats.cor(rbase.__dict__["$"](rbase.__dict__["$"](input_BIC, "pp.dat"), "data")), n = rbase.__dict__["$"](rbase.__dict__["$"](input_BIC, "pp.dat"), "total.data.count"))
    size = rbase.__dict__["$"](input_BIC, "node.count")()[0]
    skeleton_ret = pcalg.skeleton(suff_stat, indepTest=pcalg.gaussCItest, p=size, alpha = input_cut_off)
    r_adj_matrix = rpy2.robjects.r['as'](skeleton_ret, "matrix")
    pmatrix = numpy.zeros((size, size))
    for i in range(size):
        for j in range(size):
            pmatrix[i,j] = 1 if r_adj_matrix[j*size+ i] else 0
    ret = igraph.Graph.Adjacency(list(pmatrix), mode = "UNDIRECTED")
    return ret


# Run the 'pcalg' function 'pc'.
# As the pc-algorithm does not always return 
# a pdag we try to direct it, but cannot always.
# INPUT: A score object
# RETURNS: A DAG

def pc(input_BIC, input_cut_off = 0.01):
    suff_stat = rbase.list(C = rstats.cor(rbase.__dict__["$"](rbase.__dict__["$"](input_BIC, "pp.dat"), "data")), n = rbase.__dict__["$"](rbase.__dict__["$"](input_BIC, "pp.dat"), "total.data.count"))
    size = rbase.__dict__["$"](input_BIC, "node.count")()[0]
    pc_ret = pcalg.pc(suff_stat, indepTest=pcalg.gaussCItest, p=size, alpha = input_cut_off, u2pd = rbase.c('retry'))
    r_adj_matrix = rpy2.robjects.r['as'](pc_ret, "matrix")
    pmatrix = numpy.zeros((size, size))
    for i in range(size):
        for j in range(size):
            pmatrix[i,j] = 1 if r_adj_matrix[i*size+ j] else 0
    ret = igraph.Graph.Adjacency(list(pmatrix))
    if not ret.is_dag():
        try:
            ret = pdag2dag(ret)
        except ValueError:
            raise ValueError("pc did not return a DAG on the given data.")
    return ret


# Run the 'bnlearn' function 'mmhc'
# INPUT: A score object
# RETURNS: A DAG

def mmhc(input_BIC):
    data_as_data_frame = rbase.as_data_frame(rbase.__dict__["$"](rbase.__dict__["$"](input_BIC, "pp.dat"), "data"))
    size = rbase.__dict__["$"](input_BIC, "node.count")()[0]
    mmhc_ret = bnlearn.mmhc(data_as_data_frame)
    r_adj_matrix = bnlearn.amat(mmhc_ret)
    pmatrix = numpy.zeros((size, size))
    for i in range(size):
        for j in range(size):
            pmatrix[i,j] = 1 if r_adj_matrix[j*size+ i] else 0
    ret = igraph.Graph.Adjacency(list(pmatrix))
    return ret


# Transform a python igraph to a R-igraph 
# object.
# INPUT: A python igraph
# RETURNS: An R igraph 

def rgraph2igraph(input_rgraph):
    r_adj_matrix = rgraph.as_adj(input_rgraph, sparse = False)
    size = int(rgraph.gorder(input_rgraph)[0])
    pmatrix = numpy.zeros((size, size))
    for i in range(size):
        for j in range(size):
            pmatrix[i,j] = 1 if r_adj_matrix[j*size+ i] else 0
    ret = igraph.Graph.Adjacency(list(pmatrix))
    if not ret.is_dag():
        ret = pdag2dag(ret)
    return ret


# Transform a gaussParDAG object as returned 
# from, for example, the pcalg ges function
# into a igraph object.
# INPUT: A gaussParDAG
# RETURNS: A python igraph 

def gaussParDAG2igraph(input_dag):
    r_adj_matrix = rpy2.robjects.r['as'](input_dag, "matrix")
    size = int(math.sqrt(len(r_adj_matrix)))
    pmatrix_bool = numpy.array(list(r_adj_matrix)).reshape((size, size))
    pmatrix = numpy.zeros((size, size))
    for i in range(size):
        for j in range(size):
            pmatrix[i,j] = 1 if r_adj_matrix[j*size+ i] else 0
    ret = igraph.Graph.Adjacency(list(pmatrix))
    if not ret.is_dag():
        ret = pdag2dag(ret)
    return ret


# Tranforms a igraph to a bn from the bnlearn
# library. 

def igraph2bn(input_graph):
    ret_string = ""
    for i in range(input_graph.vcount()):
        ret_string += "[" + str(i)
        if len(parents(input_graph, i)) != 0:
            ret_string += "|"
            for j in parents(input_graph, i):
                ret_string += str(j) + ":"
            ret_string = ret_string[:-1]
        ret_string += "]"
    ret = bnlearn.model2network(ret_string)
    return ret


# Calculates the Structural Hamming Distance.

def shd(input_graph_1, input_graph_2):
    ret = bnlearn.shd(igraph2bn(input_graph_1), igraph2bn(input_graph_2))
    return ret[0]


### The algorithms


# Given a imsetlist, score them against the 
# given BIC and return the imset with the maximum
# value and said value.

def naive_search(input_imsetlist, input_BIC):
    max_score = score_BIC(input_BIC, imset2dag(imsetlist[0]))
    opt_imset = input_imsetlist[0]
    for i in range(1,len(input_imsetlist)):
        temp_score = score_BIC(input_BIC, imset2dag(imsetlist[i]))
        if (temp_score > max_score):
            max_score = temp_score
            opt_imset = copy.deepcopy(imsetlist[i])
    return opt_imset, max_score


# ### Depth first search
# 
# All depth first searches. 


# The greedy CIM algorithm. 
# Has option showsteps to print the steps taken.
# Can turn off the edge and/or the turn phase of the algorithm. 

def gcim(input_BIC, input_graph = None, input_imset = None, showsteps = False, edge_phase = True, turn_phase = True):
    if isinstance(input_graph, igraph.Graph):
        nodes = input_graph.vcount()
        start_imset = imset(input_graph)
        start_score = score_BIC(input_BIC, input_graph)
    elif input_graph != None:
        raise ValueError("Input graph needs to be an instance of igraph.Graph")
    elif input_imset != None:
        start_imset = input_imset
        start_score = score_BIC(input_BIC, imset2dag(input_imset))
    else:
        nodes = len(rbase.__dict__["$"](input_BIC, ".nodes"))
        start_dag = igraph.Graph(nodes, directed = True)
        start_imset = imset(start_dag)
        start_score = score_BIC(input_BIC, start_dag)
    total_steps = 0
    temp_int = 1
    temp_steps = -1
    temp_imset = start_imset
    temp_score = start_score
    while (total_steps > temp_steps):
        temp_steps = total_steps
        while (temp_int > 0 and edge_phase):
            temp_imset, temp_int, temp_score = try_edges(temp_imset, temp_score, input_BIC, show_steps= showsteps)
            total_steps += temp_int
        while (temp_int > 0 and turn_phase):
            temp_imset, temp_int, temp_score = try_turns(temp_imset, temp_score, input_BIC, show_steps= showsteps)
            total_steps += temp_int
        
    
    return temp_imset, temp_score, total_steps  


# A phased version of the Greedy CIM. 
# First only adds in edges and does a 
# turning phase, then removes edges and 
# turning phases.

def gcim_phased(input_BIC, input_graph = None, input_imset = None, showsteps = False, edge_phase = True, turn_phase = True):
    if isinstance(input_graph, igraph.Graph):
        nodes = input_graph.vcount()
        start_imset = imset(input_graph)
        start_score = score_BIC(input_BIC, input_graph)
    elif input_graph != None:
        raise ValueError("Input graph needs to be an instance of igraph.Graph")
    elif input_imset != None:
        start_imset = input_imset
        start_score = score_BIC(input_BIC, imset2dag(input_imset))
    else:
        nodes = len(rbase.__dict__["$"](input_BIC, ".nodes"))
        start_dag = igraph.Graph(nodes, directed = True)
        start_imset = imset(start_dag)
        start_score = score_BIC(input_BIC, start_dag)
    total_steps = 0
    temp_steps = -1
    temp_int = 1
    temp_imset = start_imset
    temp_score = start_score
    while (total_steps > temp_steps):
        temp_steps = total_steps
        while (temp_int > 0 and edge_phase):
            temp_imset, temp_int, temp_score = try_edges_additions(temp_imset, temp_score, input_BIC, show_steps= showsteps)
            total_steps += temp_int
        while (temp_int > 0 and turn_phase):
            temp_imset, temp_int, temp_score = try_turns(temp_imset, temp_score, input_BIC, show_steps= showsteps)
            total_steps += temp_int
    temp_steps = -1
    temp_int = 1
    while (total_steps > temp_steps):
        temp_steps = total_steps
        while (temp_int > 0 and edge_phase):
            temp_imset, temp_int, temp_score = try_edges_removals(temp_imset, temp_score, input_BIC, show_steps= showsteps)
            total_steps += temp_int
        while (temp_int > 0 and turn_phase):
            temp_imset, temp_int, temp_score = try_turns(temp_imset, temp_score, input_BIC, show_steps= showsteps)
            total_steps += temp_int
    
    return temp_imset, temp_score, total_steps  


# Tries to do edge additions..
# Is not currently used for any algorithm.

def try_2additions(input_imset, input_score, input_BIC, show_steps = False):
    ret = 0
    nodes_set = set()
    for i in input_imset:
        nodes_set.update(i[0]);
    nodes = len(nodes_set);
    current_imset = copy.deepcopy(input_imset)
    current_score = input_score
    
    for i in reversed(range(nodes)):
        for j in reversed(range(i+1,nodes)):
            temp_imset = copy.deepcopy(current_imset)
            temp_imset[set2imset_pos({i,j})][1] = 1- temp_imset[set2imset_pos({i,j})][1]
            # According to a theorem, we only need to check that
            # the 2/3-set coordinates are ok. This we do with the
            # imset2dag function.
            try:
                temp_dag = imset2dag(temp_imset)
                temp_score = score_BIC(input_BIC, temp_dag)
                if (temp_score > current_score):
                    current_imset = temp_imset
                    current_score = temp_score
                    ret +=1
                    if show_steps:
                        print("2-addition:", current_score, "\n", imset2dag(current_imset))
            except ValueError:
                pass
    
    return current_imset, ret, current_score


# Depth first trying additions of 
# the second kind. only tries to 
# add in undirected edges.

def try_2additions_additions(input_imset, input_score, input_BIC, show_steps = False):
    ret = 0
    nodes_set = set()
    for i in input_imset:
        nodes_set.update(i[0]);
    nodes = len(nodes_set);
    current_imset = copy.deepcopy(input_imset)
    current_score = input_score
    
    for i in reversed(range(nodes)):
        for j in reversed(range(i+1,nodes)):
            temp_imset = copy.deepcopy(current_imset)
            if temp_imset[set2imset_pos({i,j})][1] == 0:
                temp_imset[set2imset_pos({i,j})][1] = 1
                # According to a theorem, we only need to check that
                # the 2/3-set coordinates are ok. This we do with the
                # imset2dag function.
                try:
                    temp_dag = imset2dag(temp_imset)
                    temp_score = score_BIC(input_BIC, temp_dag)
                    if (temp_score > current_score):
                        current_imset = temp_imset
                        current_score = temp_score
                        ret += 1
                        if show_steps:
                            print("2-addition addition:", current_score, "\n", imset2dag(current_imset))
                except ValueError:
                    pass
    
    return current_imset, ret, current_score

# Depth first trying additions of 
# the second kind. only tries to 
# remove undirected edges.

def try_2additions_removals(input_imset, input_score, input_BIC, show_steps = False):
    ret = 0
    nodes_set = set()
    for i in input_imset:
        nodes_set.update(i[0]);
    nodes = len(nodes_set);
    current_imset = copy.deepcopy(input_imset)
    current_score = input_score
    
    for i in reversed(range(nodes)):
        for j in reversed(range(i+1,nodes)):
            temp_imset = copy.deepcopy(current_imset)
            if temp_imset[set2imset_pos({i,j})][1] == 1:
                temp_imset[set2imset_pos({i,j})][1] = 0
                # According to a theorem, we only need to check that
                # the 2/3-set coordinates are ok. This we do with the
                # imset2dag function.
                try:
                    temp_dag = imset2dag(temp_imset)
                    temp_score = score_BIC(input_BIC, temp_dag)
                    if (temp_score > current_score):
                        current_imset = temp_imset
                        current_score = temp_score
                        ret +=1
                        if show_steps:
                            print("2-addition removal:", current_score, "\n", imset2dag(current_imset))
                except ValueError:
                    pass
    
    return current_imset, ret, current_score

# Tries to do buddings. 
# Is not currently used for any algorihtm.

def try_buddings(input_imset, input_score, input_BIC, show_steps = False):
    steps_taken = 0
    nodes = len(input_imset[-1][0])
    current_imset = copy.deepcopy(input_imset)
    current_score = input_score
    
    for i in range(nodes):
        for j in range(nodes):
            # Make sure i - j in G
            if (i != j) and (current_imset[set2imset_pos({i,j})][1] == 1):
                # Find ne_G(i) and ne_G(j)
                ne_i = set()
                ne_j = set()
                for k in range(nodes):
                    if k == i or k == j:
                        pass
                    else:
                        if (current_imset[set2imset_pos({i,k})][1] == 1):
                            ne_i.add(k)
                        if (current_imset[set2imset_pos({j,k})][1] == 1):
                            ne_j.add(k)
                # In a budding S* has (by def) size 2 or more, 
                # that is the coordinatelist.
                s_set_canditates = coordinatelist(ne_i)
                for S in s_set_canditates:
                    # Check to see if (i, j, S) can be a budding
                    check_break = False
                    perm = imset_value(current_imset, S.union({i}), full_imset=True)
                    # We have already checked all sets of size less than 2.
                    for k in coordinatelist(S):
                        if (imset_value(current_imset, k.union({i}), full_imset=True) != 1):
                            check_break = True
                            break
                    # Make a copy of current_imset and change it as 
                    # if {current_imset, temp_imset} was a budding
                    # with respect to (i,j,S)
                    temp_imset = copy.deepcopy(current_imset)
                    bud_set = powerset(S)
                    perm = imset_value(temp_imset, S.union({i,j}), full_imset=True)
                    for k in bud_set:
                        if not k.issubset(ne_j):
                            if (perm != imset_value(temp_imset, k.union({i, j}), full_imset=True)):
                                check_break = True
                                break
                            else:
                                temp_imset[set2imset_pos(k.union({i, j}))][1] = 1 - temp_imset[set2imset_pos(k.union({i, j}))][1] 
                    # Now if no breaks have happened, we should more or less have a budding, 
                    # up to the fact that temp_imset might not be an imset
                    if (not check_break) and is_imset(temp_imset): # These checks can be done in a better way.
                        temp_dag = imset2dag(temp_imset)
                        temp_score = score_BIC(input_BIC, temp_dag)
                        if (temp_score > current_score):
                            current_imset = temp_imset
                            current_score = temp_score
                            steps_taken += 1
                            if show_steps:
                                print("budding:", current_score, "\n", imset2dag(current_imset))
    # Return all relevant data.
    return current_imset, steps_taken, current_score


# Does one pass of the turn phase. Whenever
# a better score is found it moves on to that 
# imset.

def try_turns(input_imset, input_score, input_BIC, show_steps= False):
    steps_taken = 0
    nodes = len(input_imset[-1][0])
    current_imset = copy.deepcopy(input_imset)
    current_score = input_score
    
    for i in range(nodes):
        for j in range(i+1, nodes):
            # Make sure i - j in G
            if (current_imset[set2imset_pos({i,j})][1] == 1):
                # Find ne_G(i) and ne_G(j)
                ne_i = set()
                ne_j = set()
                for k in range(nodes):
                    if k == i or k == j:
                        pass
                    else:
                        if (current_imset[set2imset_pos({i,k})][1] == 1):
                            ne_i.add(k)
                        if (current_imset[set2imset_pos({j,k})][1] == 1):
                            ne_j.add(k)
                for S_i in powerset(ne_i):
                    check_i = True
                    # Check if S_i is reasonable
                    for temp_set in coordinatelist(S_i):
                        if (imset_value(current_imset, temp_set.union({i}), full_imset=True) != 1):
                            check_i = False
                            break
                    # So S_i is at least reasonable
                    if check_i:
                        for S_j in powerset(ne_j):
                            check_j = True
                            # Check if S_j is reasonable
                            for temp_set in coordinatelist(S_j):
                                if (imset_value(current_imset, temp_set.union({j}), full_imset=True) != 1):
                                    check_j = False
                                    break
                            # So S_j is at least reasonable
                            # Account for Markov equivalence
                            if S_j.issubset(ne_i) and S_i.issubset(ne_j):
                                check_j = False
                            if check_j:
                                temp_imset = copy.deepcopy(current_imset)
                                # Account for sign and order of (i,j)
                                if not S_i.issubset(ne_j):
                                    perm = imset_value(temp_imset, S_i.union({i, j}), full_imset=True)
                                elif not S_j.issubset(ne_i):
                                    perm = 1 - imset_value(temp_imset, S_j.union({i, j}), full_imset=True)
                                else:
                                    raise RuntimeError ("Logical fallacy in 'try_turns'. This error should never happen.")
                                check_flip = True
                                
                                flip_set_i = powerset(S_i)
                                flip_set_j = powerset(S_j)
                                for temp_coord_set in flip_set_i:
                                    if not temp_coord_set.issubset(ne_j):
                                        if (perm != imset_value(temp_imset, temp_coord_set.union({i, j}), full_imset=True)):
                                            check_flip = False
                                            break
                                        else:
                                            temp_imset[set2imset_pos(temp_coord_set.union({i, j}))][1] = 1 - temp_imset[set2imset_pos(temp_coord_set.union({i, j}))][1] 

                                for temp_coord_set in flip_set_j:
                                    if not temp_coord_set.issubset(ne_i):
                                        if (1-perm != imset_value(temp_imset, temp_coord_set.union({i, j}), full_imset=True)):
                                            check_flip = False
                                            break
                                        else:
                                            temp_imset[set2imset_pos(temp_coord_set.union({i, j}))][1] = 1 - temp_imset[set2imset_pos(temp_coord_set.union({i, j}))][1] 

                                if check_flip and is_imset(temp_imset):
                                    try:
                                        temp_dag = imset2dag(temp_imset)
                                        temp_score = score_BIC(input_BIC, temp_dag)
                                        if (temp_score > current_score):
                                            current_imset = temp_imset
                                            current_score = temp_score
                                            steps_taken += 1
                                            if show_steps:
                                                print("Turn:", current_score, "\n", imset2dag(current_imset))
                                    except ValueError:
                                        pass
    
    return current_imset, steps_taken, current_score


# Finds ges-edges and moves to them. Is 
# Currently not used in any algorithm.

def try_ges(input_imset, input_score, input_BIC, show_steps= False):
    no_steps = 0
    nodes = len(input_imset[-1][0])
    current_imset = copy.deepcopy(input_imset)
    current_score = input_score
    
    for i in range(nodes):
        for j in range(nodes):
            ne_i = set()
            if i != j:
                perm = current_imset[set2imset_pos({i,j})][1]
                for k in range(nodes):
                    if k == i:
                        pass
                    elif (current_imset[set2imset_pos({i,k})][1] == 1):
                        ne_i.add(k)
                ne_i.discard(j);
            
            loop_set = powerset(ne_i)
            loop_set.pop(0)
            for S in loop_set:
                check_S = True
                for k in coordinatelist(S):
                    if (current_imset[set2imset_pos(k.union({i}))][1] != 1):
                        check_S = False
                        break
                if check_S:
                    check = True
                    temp_imset = copy.deepcopy(current_imset)
                    for k in powerset(S):
                        if (current_imset[set2imset_pos(k.union({i, j}))][1] != perm):
                            check = False
                            break
                        else:
                            temp_imset[set2imset_pos(k.union({i, j}))][1] = 1 - temp_imset[set2imset_pos(k.union({i, j}))][1]
                    if check:
                        if is_imset(temp_imset):
                            try:
                                temp_dag = imset2dag(temp_imset)
                                temp_score = score_BIC(input_BIC, temp_dag)
                                if (temp_score > current_score):
                                    current_imset = temp_imset
                                    current_score = temp_score
                                    no_steps += 1
                                    if show_steps:
                                        print("GES:", current_score, "\n", imset2dag(current_imset))
                            except ValueError:
                                pass
        return current_imset, no_steps, current_score


# A version of 'try_ges' that only removes
# edges and does not remove them. Currently 
# not in use.

def try_ges_removals(input_imset, input_score, input_BIC, show_steps = False):
    no_steps = 0
    nodes = len(input_imset[-1][0])
    current_imset = copy.deepcopy(input_imset)
    current_score = input_score
    
    for i in range(nodes):
        for j in range(nodes):
            ne_i = set()
            if i != j and current_imset[set2imset_pos({i,j})][1] == 1:
                for k in range(nodes):
                    if k == i:
                        pass
                    elif (current_imset[set2imset_pos({i,k})][1] == 1):
                        ne_i.add(k)
                ne_i.discard(j);
            
            loop_set = powerset(ne_i)
            loop_set.pop(0)
            for S in loop_set:
                check_S = True
                for k in coordinatelist(S):
                    if (current_imset[set2imset_pos(k.union({i}))][1] != 1):
                        check_S = False
                        break
                if check_S:
                    check = True
                    temp_imset = copy.deepcopy(current_imset)
                    for k in powerset(S):
                        if (current_imset[set2imset_pos(k.union({i, j}))][1] == 0):
                            check = False
                            break
                        else:
                            temp_imset[set2imset_pos(k.union({i, j}))][1] = 1 - temp_imset[set2imset_pos(k.union({i, j}))][1]
                    if check:
                        if is_imset(temp_imset):
                            try:
                                temp_dag = imset2dag(temp_imset)
                                temp_score = score_BIC(input_BIC, temp_dag)
                                if (temp_score > current_score):
                                    current_imset = temp_imset
                                    current_score = temp_score
                                    no_steps += 1
                                    if show_steps:
                                        print("GES:", current_score, "\n", imset2dag(current_imset))
                            except ValueError:
                                pass
        return current_imset, no_steps, current_score

    
# A version of 'try_ges' that only adds in
# edges and does not remove them. Currently 
# not in use.

def try_ges_additions(input_imset, input_score, input_BIC, show_steps= False):
    no_steps = 0
    nodes = len(input_imset[-1][0])
    current_imset = copy.deepcopy(input_imset)
    current_score = input_score
    
    for i in range(nodes):
        for j in range(nodes):
            ne_i = set()
            if i != j and current_imset[set2imset_pos({i,j})][1] == 0:
                for k in range(nodes):
                    if k == i:
                        pass
                    elif (current_imset[set2imset_pos({i,k})][1] == 1):
                        ne_i.add(k)
                ne_i.discard(j);
            
            loop_set = powerset(ne_i)
            loop_set.pop(0)
            for S in loop_set:
                check_S = True
                for k in coordinatelist(S):
                    if (current_imset[set2imset_pos(k.union({i}))][1] != 1):
                        check_S = False
                        break
                if check_S:
                    check = True
                    temp_imset = copy.deepcopy(current_imset)
                    for k in powerset(S):
                        if (current_imset[set2imset_pos(k.union({i, j}))][1] == 1):
                            check = False
                            break
                        else:
                            temp_imset[set2imset_pos(k.union({i, j}))][1] = 1 - temp_imset[set2imset_pos(k.union({i, j}))][1]
                    if check:
                        if is_imset(temp_imset):
                            try:
                                temp_dag = imset2dag(temp_imset)
                                temp_score = score_BIC(input_BIC, temp_dag)
                                if (temp_score > current_score):
                                    current_imset = temp_imset
                                    current_score = temp_score
                                    no_steps += 1
                                    if show_steps:
                                        print("GES addition:", current_score, "\n", imset2dag(current_imset))
                            except ValueError:
                                pass
        return current_imset, no_steps, current_score


# Find all edge-pairs and, if we score better, 
# move to the next edge-pair. Notice that it 
# is a depth first version. show_steps option
# to analyze the algorithm.

def try_edges(input_imset, input_score, input_BIC, show_steps= False):
    no_steps = 0
    nodes = len(input_imset[-1][0])
    current_imset = copy.deepcopy(input_imset)
    current_score = input_score
    
    for i in range(nodes):
        for j in range(nodes):
            ne_i = set()
            if i != j:
                # Check whether we add or remove the edge (i,j)
                perm = current_imset[set2imset_pos({i,j})][1]
                # Find the neighbours of i, not counting j.
                for k in range(nodes):
                    if k == i:
                        pass
                    elif (current_imset[set2imset_pos({i,k})][1] == 1):
                        ne_i.add(k)
                ne_i.discard(j);
                
                # Check the possible S* sets.
                loop_set = powerset(ne_i)
                for S in loop_set:
                    check_S = True
                    # Check condition (2). Since S is a subset of ne_i
                    # we only need to check subsets of size 2 or more.
                    for k in coordinatelist(S):
                        if (current_imset[set2imset_pos(k.union({i}))][1] != 1):
                            check_S = False
                            break
                    # If S looks good, continue
                    if check_S:
                        check = True
                        temp_imset = copy.deepcopy(current_imset)
                        for k in powerset(S):
                            if (current_imset[set2imset_pos(k.union({i, j}))][1] != perm):
                                check = False
                                break
                            else:
                                temp_imset[set2imset_pos(k.union({i, j}))][1] = 1 - temp_imset[set2imset_pos(k.union({i, j}))][1]
                        if check:
                            # Looks promising, if temp_imset is a valid imset we have found an edge pair.
                            if is_imset(temp_imset):
                                try:
                                    temp_dag = imset2dag(temp_imset)
                                    temp_score = score_BIC(input_BIC, temp_dag)
                                    if (temp_score > current_score):
                                        current_imset = temp_imset
                                        current_score = temp_score
                                        no_steps += 1
                                        if show_steps:
                                            print("Edge:", current_score, "\n", imset2dag(current_imset))
                                        break
                                except ValueError:
                                    pass
    
    return current_imset, no_steps, current_score


# An alternative versions of "try_edges". 
# Only tries to add in edges. Used 
# for phased versions of greedy CIM. See 
# "try_edges" for more documentation.

def try_edges_additions(input_imset, input_score, input_BIC, show_steps= False):
    no_steps = 0
    nodes = len(input_imset[-1][0])
    current_imset = copy.deepcopy(input_imset)
    current_score = input_score
    
    for i in range(nodes):
        for j in range(nodes):
            ne_i = set()
            if i != j and current_imset[set2imset_pos({i,j})][1] == 0:
                for k in range(nodes):
                    if k == i:
                        pass
                    elif (current_imset[set2imset_pos({i,k})][1] == 1):
                        ne_i.add(k)
                ne_i.discard(j);

                loop_set = powerset(ne_i)
                for S in loop_set:
                    check_S = True
                    for k in coordinatelist(S):
                        if (current_imset[set2imset_pos(k.union({i}))][1] != 1):
                            check_S = False
                            break
                    if check_S:
                        check = True
                        temp_imset = copy.deepcopy(current_imset)
                        for k in powerset(S):
                            if (current_imset[set2imset_pos(k.union({i, j}))][1] != 0):
                                check = False
                                break
                            else:
                                temp_imset[set2imset_pos(k.union({i, j}))][1] = 1
                        if check:
                            if is_imset(temp_imset):
                                try:
                                    temp_dag = imset2dag(temp_imset)
                                    temp_score = score_BIC(input_BIC, temp_dag)
                                    if (temp_score > current_score):
                                        current_imset = temp_imset
                                        current_score = temp_score
                                        no_steps += 1
                                        if show_steps and no_steps == 1:
                                            print("Edge addition:", current_score, "\n", imset2dag(current_imset))
                                except ValueError:
                                    pass
    
    return current_imset, no_steps, current_score


# An alternative versions of "try_edges". 
# Only tries to remove edges. Used 
# for phased versions of greedy CIM. See 
# "try_edges" for more documentation.

def try_edges_removals(input_imset, input_score, input_BIC, show_steps= False):
    no_steps = 0
    nodes = len(input_imset[-1][0])
    current_imset = copy.deepcopy(input_imset)
    current_score = input_score
    
    for i in range(nodes):
        for j in range(nodes):
            ne_i = set()
            if i != j and current_imset[set2imset_pos({i,j})][1] == 1:
                for k in range(nodes):
                    if k == i:
                        pass
                    elif (current_imset[set2imset_pos({i,k})][1] == 1):
                        ne_i.add(k)
                ne_i.discard(j);

                loop_set = powerset(ne_i)
                for S in loop_set:
                    check_S = True
                    for k in coordinatelist(S):
                        if (current_imset[set2imset_pos(k.union({i}))][1] != 1):
                            check_S = False
                            break
                    
                    if check_S:
                        check = True
                        temp_imset = copy.deepcopy(current_imset)
                        for k in powerset(S):
                            if (current_imset[set2imset_pos(k.union({i, j}))][1] != 1):
                                check = False
                                break
                            else:
                                temp_imset[set2imset_pos(k.union({i, j}))][1] = 0
                        if check:
                            if is_imset(temp_imset):
                                try:
                                    temp_dag = imset2dag(temp_imset)
                                    temp_score = score_BIC(input_BIC, temp_dag)
                                    if (temp_score > current_score):
                                        current_imset = temp_imset
                                        current_score = temp_score
                                        no_steps += 1
                                        if show_steps and no_steps == 1:
                                            print("Edge Removal:", current_score, "\n", imset2dag(current_imset))
                                except ValueError:
                                    pass
    
    return current_imset, no_steps, current_score


# ### Breadth first
# 
# All functions in this section are the same
# as the ones above with the exception that they 
# are breadth-first instead of depth-first. Thus 
# they only take one step per pass (the one 
# creating the best score).


# A breadth first version of the Greedy CIM.
# See 'gcim' for better documentation.

def gcim_b(input_BIC, input_graph = None, input_imset = None, showsteps = False, edge_phase = True, turn_phase = True):
    if isinstance(input_graph, igraph.Graph):
        nodes = input_graph.vcount()
        start_imset = imset(input_graph)
        start_score = score_BIC(input_BIC, input_graph)
    elif input_graph != None:
        raise ValueError("Input graph needs to be an instance of igraph.Graph")
    elif input_imset != None:
        start_imset = input_imset
        start_score = score_BIC(input_BIC, imset2dag(input_imset))
    else:
        nodes = len(rbase.__dict__["$"](input_BIC, ".nodes"))
        start_dag = igraph.Graph(nodes, directed = True)
        start_imset = imset(start_dag)
        start_score = score_BIC(input_BIC, start_dag)
    total_steps = 0
    temp_int = 1
    temp_steps = -1
    temp_score = start_score
    temp_imset = start_imset
    while (total_steps > temp_steps):
        temp_steps = total_steps
        while (temp_int > 0):
            temp_imset, temp_int, temp_score = try_edges_b(temp_imset, temp_score, input_BIC, show_steps= showsteps)
            total_steps += temp_int
        while (temp_int > 0):
            temp_imset, temp_int, temp_score = try_turns_b(temp_imset, temp_score, input_BIC, show_steps= showsteps)
            total_steps += temp_int
        
    
    return temp_imset, temp_score, total_steps


# A phased breadth-first version of 
# greedy CIM. See 'gcim' for better 
# documentation.

def gcim_phased_b(input_BIC, input_graph = None, input_imset = None, showsteps = False, edge_phase = True, turn_phase = True):
    if isinstance(input_graph, igraph.Graph):
        nodes = input_graph.vcount()
        start_imset = imset(input_graph)
        start_score = score_BIC(input_BIC, input_graph)
    elif input_graph != None:
        raise ValueError("Input graph needs to be an instance of igraph.Graph")
    elif input_imset != None:
        start_imset = input_imset
        start_score = score_BIC(input_BIC, imset2dag(input_imset))
    else:
        nodes = len(rbase.__dict__["$"](input_BIC, ".nodes"))
        start_dag = igraph.Graph(nodes, directed = True)
        start_imset = imset(start_dag)
        start_score = score_BIC(input_BIC, start_dag)
    total_steps = 0
    temp_steps = -1
    temp_int = 1
    temp_imset = start_imset
    temp_score = start_score
    while (total_steps > temp_steps):
        temp_steps = total_steps
        while (temp_int > 0 and edge_phase):
            temp_imset, temp_int, temp_score = try_edges_additions_b(temp_imset, temp_score, input_BIC, show_steps= showsteps)
            total_steps += temp_int
        while (temp_int > 0 and turn_phase):
            temp_imset, temp_int, temp_score = try_turns_b(temp_imset, temp_score, input_BIC, show_steps= showsteps)
            total_steps += temp_int
    temp_steps = -1
    temp_int = 1
    while (total_steps > temp_steps):
        temp_steps = total_steps
        while (temp_int > 0 and edge_phase):
            temp_imset, temp_int, temp_score = try_edges_removals_b(temp_imset, temp_score, input_BIC, show_steps= showsteps)
            total_steps += temp_int
        while (temp_int > 0 and turn_phase):
            temp_imset, temp_int, temp_score = try_turns_b(temp_imset, temp_score, input_BIC, show_steps= showsteps)
            total_steps += temp_int
    
    return temp_imset, temp_score, total_steps


# A breadth-first version of 
# 'try_2additions'. 

def try_2additions_b(input_imset, input_score, input_BIC, show_steps = False):
    step_counter = 0
    nodes_set = set()
    for i in input_imset:
        nodes_set.update(i[0]);
    nodes = len(nodes_set);
    current_imset = copy.deepcopy(input_imset)
    current_score = input_score
    
    for i in range(nodes):
        for j in range(i+1,nodes):
            temp_imset = copy.deepcopy(input_imset)
            temp_imset[set2imset_pos({i,j})][1] = 1- temp_imset[set2imset_pos({i,j})][1]
            # if temp_imset is a valid imset
            # we only need to check the 2/3-sets.
            # As imset2pdag does this already
            # we just go ahead.
            try:
                temp_dag = imset2dag(temp_imset)
                temp_score = score_BIC(input_BIC, temp_dag)
                if (temp_score > current_score):
                    current_imset = temp_imset
                    current_score = temp_score
                    step_counter = 1
            except ValueError:
                pass
    
    
    if show_steps and step_counter == 1:
        print("2-addition:", current_score, "\n", imset2dag(current_imset))
    return current_imset, step_counter, current_score


# A breadth-first version of 
# 'try_2additions' only adding 
# edges.

def try_2additions_additions_b(input_imset, input_score, input_BIC, show_steps = False):
    step_counter = 0
    nodes_set = set()
    for i in input_imset:
        nodes_set.update(i[0]);
    nodes = len(nodes_set);
    current_imset = copy.deepcopy(input_imset)
    current_score = input_score
    
    for i in range(nodes):
        for j in range(i+1,nodes):
            temp_imset = copy.deepcopy(input_imset)
            if temp_imset[set2imset_pos({i,j})][1] == 0:
                temp_imset[set2imset_pos({i,j})][1] = 1
                try:
                    temp_dag = imset2dag(temp_imset)
                    temp_score = score_BIC(input_BIC, temp_dag)
                    if (temp_score > current_score):
                        current_imset = temp_imset
                        current_score = temp_score
                        step_counter = 1
                except ValueError:
                    pass
    
    
    if show_steps and step_counter == 1:
        print("2-addition:", current_score, "\n", imset2dag(current_imset))
    return current_imset, step_counter, current_score

# A breadth-first version of
# 'try_2additions' only removing 
# edges.

def try_2additions_removals_b(input_imset, input_score, input_BIC, show_steps = False):
    step_counter = 0
    nodes_set = set()
    for i in input_imset:
        nodes_set.update(i[0]);
    nodes = len(nodes_set);
    current_imset = copy.deepcopy(input_imset)
    current_score = input_score
    
    for i in range(nodes):
        for j in range(i+1,nodes):
            temp_imset = copy.deepcopy(input_imset)
            if temp_imset[set2imset_pos({i,j})][1] == 1:
                temp_imset[set2imset_pos({i,j})][1] = 0
                try:
                    temp_dag = imset2dag(temp_imset)
                    temp_score = score_BIC(input_BIC, temp_dag)
                    if (temp_score > current_score):
                        current_imset = temp_imset
                        current_score = temp_score
                        step_counter = 1
                except ValueError:
                    pass
    
    
    if show_steps and step_counter == 1:
        print("2-addition:", current_score, "\n", imset2dag(current_imset))
    return current_imset, step_counter, current_score


# A breadth-first version of 'try_edges'.
# As 'try_edges' is better comments, we 
# refer to that function.

def try_edges_b(input_imset, input_score, input_BIC, show_steps= False):
    step_counter = 0
    nodes = len(input_imset[-1][0])
    current_imset = copy.deepcopy(input_imset)
    current_score = input_score
    
    for i in range(nodes):
        for j in range(nodes):
            ne_i = set()
            if i== j:
                continue
                
            perm = input_imset[set2imset_pos({i,j})][1]
            for k in range(nodes):
                if k == i:
                    pass
                elif (input_imset[set2imset_pos({i,k})][1] == 1):
                    ne_i.add(k)
            ne_i.discard(j)

            loop_set = powerset(ne_i)
            for S in loop_set:
                check_S = True
                for k in coordinatelist(S):
                    if (input_imset[set2imset_pos(k.union({i}))][1] != 1):
                        check_S = False
                        break
                if check_S:
                    check = True
                    temp_imset = copy.deepcopy(input_imset)
                    for k in powerset(S):
                        if (input_imset[set2imset_pos(k.union({i, j}))][1] != perm):
                            check = False
                            break
                        else:
                            temp_imset[set2imset_pos(k.union({i, j}))][1] = 1 - temp_imset[set2imset_pos(k.union({i, j}))][1]
                    if check:
                        if is_imset(temp_imset):
                            try:
                                temp_dag = imset2dag(temp_imset)
                                temp_score = score_BIC(input_BIC, temp_dag)
                                if (temp_score > current_score):
                                    current_imset = temp_imset
                                    current_score = temp_score
                                    step_counter = 1
                            except ValueError:
                                pass
    if show_steps and step_counter == 1:
        print("Edge:", current_score, "\n", imset2dag(current_imset))
    return current_imset, step_counter, current_score


# A breadth-first version of 'try_edges',
# where we only try to add in edges. 
# As 'try_edges' is better comments, we 
# refer to that function.

def try_edges_additions_b(input_imset, input_score, input_BIC, show_steps= False):
    step_counter = 0
    nodes = len(input_imset[-1][0])
    current_imset = copy.deepcopy(input_imset)
    current_score = input_score
    
    for i in range(nodes):
        for j in range(nodes):
            ne_i = set()
            if i != j and input_imset[set2imset_pos({i,j})][1] == 0:
                
                for k in range(nodes):
                    if k == i or k == j:
                        pass
                    elif (input_imset[set2imset_pos({i,k})][1] == 1):
                        ne_i.add(k)

                loop_set = powerset(ne_i)
                for S in loop_set:
                    check_S = True
                    for k in coordinatelist(S):
                        if (input_imset[set2imset_pos(k.union({i}))][1] != 1):
                            check_S = False
                            break
                    if check_S:
                        check = True
                        temp_imset = copy.deepcopy(input_imset)
                        for k in powerset(S):
                            if (input_imset[set2imset_pos(k.union({i, j}))][1] != 0):
                                check = False
                                break
                            else:
                                temp_imset[set2imset_pos(k.union({i, j}))][1] = 1
                        if check:
                            if is_imset(temp_imset):
                                try:
                                    temp_dag = imset2dag(temp_imset)
                                    temp_score = score_BIC(input_BIC, temp_dag)
                                    if (temp_score > current_score):
                                        current_imset = temp_imset
                                        current_score = temp_score
                                        step_counter = 1
                                except ValueError:
                                    pass
    if show_steps and step_counter == 1:
        print("Adding edge with edge pair:", current_score, "\n", imset2dag(current_imset))
    return current_imset, step_counter, current_score
    

# A breadth-first version of 'try_edges',
# where we only try to add in edges. 
# As 'try_edges' is better comments, we 
# refer to that function.

def try_edges_removals_b(input_imset, input_score, input_BIC, show_steps= False):
    step_counter = 0
    nodes = len(input_imset[-1][0])
    current_imset = copy.deepcopy(input_imset)
    current_score = input_score
    
    for i in range(nodes):
        for j in range(nodes):
            ne_i = set()
            if i != j and input_imset[set2imset_pos({i,j})][1] == 1:
                
                for k in range(nodes):
                    if k == i or k == j:
                        pass
                    elif (input_imset[set2imset_pos({i,k})][1] == 1):
                        ne_i.add(k)

                loop_set = powerset(ne_i)
                for S in loop_set:
                    check_S = True
                    for k in coordinatelist(S):
                        if (input_imset[set2imset_pos(k.union({i}))][1] != 1):
                            check_S = False
                            break
                    if check_S:
                        check = True
                        temp_imset = copy.deepcopy(input_imset)
                        for k in powerset(S):
                            if (input_imset[set2imset_pos(k.union({i, j}))][1] != 1):
                                check = False
                                break
                            else:
                                temp_imset[set2imset_pos(k.union({i, j}))][1] = 0
                        if check:
                            if is_imset(temp_imset):
                                try:
                                    temp_dag = imset2dag(temp_imset)
                                    temp_score = score_BIC(input_BIC, temp_dag)
                                    if (temp_score > current_score):
                                        current_imset = temp_imset
                                        current_score = temp_score
                                        step_counter = 1
                                except ValueError:
                                    pass
    if show_steps and step_counter == 1:
        print("Removing edge with edge pair:", current_score, "\n", imset2dag(current_imset))
    return current_imset, step_counter, current_score


# A breadth-first version of 'try_edges', 
# where we only try to add in edges. 
# As 'try_turns' is better comments, we 
# refer to that function.

def try_turns_b(input_imset, input_score, input_BIC, show_steps= False):
    step_counter = 0
    nodes = len(input_imset[-1][0])
    current_imset = copy.deepcopy(input_imset)
    current_score = input_score
    
    for i in range(nodes):
        for j in range(nodes):
            # Make sure i - j in G
            if (i != j and input_imset[set2imset_pos({i,j})][1] == 1):
                # Find ne_G(i) and ne_G(j)
                ne_i = set()
                ne_j = set()
                for k in range(nodes):
                    if k == i or k == j:
                        pass
                    else:
                        if (input_imset[set2imset_pos({i,k})][1] == 1):
                            ne_i.add(k)
                        if (input_imset[set2imset_pos({j,k})][1] == 1):
                            ne_j.add(k)
                for S_i in powerset(ne_i):
                    check_i = True
                    # Check if S_i is reasonable
                    for temp_set in coordinatelist(S_i):
                        if (imset_value(input_imset, temp_set.union({i}), full_imset=True) != 1):
                            check_i = False
                            break
                        
                    if check_i:
                        for S_j in powerset(ne_j):
                            check_j = True
                            for temp_set in coordinatelist(S_j):
                                if (imset_value(input_imset, temp_set.union({j}), full_imset=True) != 1):
                                    check_j = False
                                    break
                            # To remove redundancy for Markov equivalence
                            if len(S_j) + len(S_i) == 0:
                                check_j = False
                            if check_j:
                                temp_imset = copy.deepcopy(input_imset)
                                perm = imset_value(temp_imset, S_i.union({i, j}), full_imset=True)
                                check_flip = True
                                
                                flip_set_i = powerset(S_i)
                                flip_set_j = powerset(S_j)
                                for temp_coord_set in flip_set_i:
                                    if not temp_coord_set.issubset(ne_j):
                                        if (perm != imset_value(temp_imset, temp_coord_set.union({i, j}), full_imset=True)):
                                            check_flip = False
                                            break
                                        else:
                                            temp_imset[set2imset_pos(temp_coord_set.union({i, j}))][1] = 1 - temp_imset[set2imset_pos(temp_coord_set.union({i, j}))][1] 
                                    
                                for temp_coord_set in flip_set_j:
                                    if not temp_coord_set.issubset(ne_i):
                                        if (1-perm != imset_value(temp_imset, temp_coord_set.union({i, j}), full_imset=True)):
                                            check_flip = False
                                            break
                                        else:
                                            temp_imset[set2imset_pos(temp_coord_set.union({i, j}))][1] = 1 - temp_imset[set2imset_pos(temp_coord_set.union({i, j}))][1] 
                                if check_flip and is_imset(temp_imset):
                                    try:
                                        temp_dag = imset2dag(temp_imset)
                                        temp_score = score_BIC(input_BIC, temp_dag)
                                        if (temp_score > current_score):
                                            current_imset = temp_imset
                                            current_score = temp_score
                                            step_counter = 1
                                    except ValueError:
                                        pass
                            
                
    if show_steps and step_counter == 1:
        print("Turn:", current_score, "\n", imset2dag(current_imset))
    return current_imset, step_counter, current_score


# Here follows an example of how to use the code.


# Here we initiate the imsetlist to use 
# in the naive_search. Uncomment if using
# the 'opt' algorithm.

# p = 8;
# imsetlist = try_all_dags(p)
# i = 0;
# while (i < len(imsetlist)):
#     j= i+1;
#     while (j < len(imsetlist)):
#         if (imsetlist[i] == imsetlist[j]):
#             imsetlist.pop(j);
#             j-=1;
#         j+=1;
#     i+=1;



# The data is collected from dataset_path"nodes-"no_nodes/dataset_type"-"dataset_type_run[model_name/graph_name]
# Choose your dataset and algorithms
models_name = ["model-"+str(i)+".csv" for i in range(1,3)]
graph_name = ["true-graph-"+str(i)+".csv" for i in range(1,3)]
no_nodes = [8]
samples_no = 10000 # For writing in result-file and dataset_path
dataset_type = "nbh"
dataset_type_run = [i/2 for i in range(1, 11)]
alg_vector_choose = ["ges", 'gies', "mmhc", "greedySP", "pc", "gcim", "ske", "true"]
alpha = 0.0001 # Cutof limit to be used in pc and ske
dataset_path = "sim_data_nbh_final/alpha-"+str(alpha)+"/samples-"+str(samples_no)+"/"


# Initiate some variables
# A sorted version of the alg_vector_choose to make coding easier
alg_vector = ["ges", "pc", "mmhc", "greedySP", "gcim", "gies", "ges_gcim", "ske", "gcim_phased", "gcim_backwards", "gcim_b",
              "gcim_phased_b", "opt", "true"]
alg_vector_copy = alg_vector.copy()
for i in alg_vector_copy:
    if alg_vector_choose.count(i) == 0:
        alg_vector.remove(i)
# Which algorithms to count the number of steps on
step_count_alg = alg_vector.copy()
for i in ["ges", "gies", "pc", "mmhc", "opt", "true"]:
    try:
        step_count_alg.remove(i)
    except ValueError:
        pass


# Print starting message
print("Local clock:", time.asctime(time.localtime(time.time())))
print("Testing with:")
print("Algorithms:", alg_vector)
print("From:", dataset_path + "nodes-" + str(no_nodes) + "/" + dataset_type + "-" + str(dataset_type_run))

for nodes in no_nodes:
    print("Starting testing with nodes: ", nodes)
    for run in dataset_type_run:
        print("Starting testing with " + dataset_type + ": " + str(run))

        # Score vectors to save the BIC score
        return_dags_vector = [alg_vector]
        for algo in alg_vector:
            exec("score_vector_" + algo + " = []")

        # Step vectors to get some extra data
        for algo in step_count_alg:
            exec("step_vector_" + algo + " = []")

        # Shd vectors to save the shds
        if alg_vector.count('opt') > 0:
            for algo in alg_vector:
                if algo != 'opt' and algo != 'true':
                    exec("shd_vector_" + algo + "_opt = []")
        if alg_vector.count('true') > 0:
            for algo in alg_vector:
                if algo != 'true':
                    exec("shd_vector_" + algo + "_true = []")

        # Run the algorithms
        print("[", end = "")
        progress_res = min(50, len(models_name))
        progress_vector = [numpy.floor(i * len(models_name) / progress_res) for i in range(1, progress_res)]
        for no in range(len(models_name)):
            if no in progress_vector:
                print("=", end = "")
            DATA = dataset_path + "nodes-" + str(nodes) + "/" + dataset_type + "-" + str(run) + "/" + models_name[no]
            BIC = initiate_BIC(DATA)
            step_count_vector_temp = []
            # Do the 'gies'
            if alg_vector.count('ges') > 0:
                return_gies_dag = gaussParDAG2igraph(pcalg.ges(BIC)[1])
                score_vector_gies.append(score_BIC(BIC, return_gies_dag))
            # Do the 'ges'
            if alg_vector.count('ges') > 0:
                return_ges_dag = gaussParDAG2igraph(
                    pcalg.ges(BIC, phase = rpy2.robjects.r['c']('forward', 'backward'), iterate = False)[1])
                score_vector_ges.append(score_BIC(BIC, return_ges_dag))
            # Do the 'pc'
            if alg_vector.count('pc') > 0:
                return_pc_dag = pc(BIC, input_cut_off= alpha)
                score_vector_pc.append(score_BIC(BIC, return_pc_dag))
            # Do the 'mmhc'
            if alg_vector.count('mmhc') > 0:
                return_mmhc_dag = mmhc(BIC)
                score_vector_mmhc.append(score_BIC(BIC, return_mmhc_dag))
            # Do greedySP
            if alg_vector.count('greedySP') > 0:
                return_greedySP_dag = greedySP(DATA, alpha)
                score_vector_greedySP.append(score_BIC(BIC,return_greedySP_dag))
            # Do the gcim
            if alg_vector.count('gcim') > 0:
                return_gcim = gcim(BIC)
                return_gcim_dag = imset2dag(return_gcim[0])
                score_vector_gcim.append(return_gcim[1])
                step_vector_gcim.append(return_gcim[2])
            # Do the skeletal 'gcim'
            if alg_vector.count('ske') > 0:
                skeleton_graph = skeleton(BIC, input_cut_off= alpha)
                try:
                    dir_graph = skeleton_graph.copy()
                    dir_graph.to_directed(mutual = True)
                    dir_graph = pdag2dag(dir_graph)
                except ValueError:
                    dir_graph = skeleton_graph.copy()
                    dir_graph.to_directed(mutual = False)
                return_ske = gcim(BIC, input_graph = dir_graph, edge_phase= False)
                return_ske_dag = imset2dag(return_ske[0])
                score_vector_ske.append(return_ske[1])
                step_vector_ske.append(return_ske[2])
            # Do the 'gcim_b')
            if alg_vector.count('gcim_b') > 0:
                return_gcim_b = gcim_b(BIC)
                return_gcim_b_dag = imset2dag(return_gcim_b[0])
                score_vector_gcim_b.append(return_gcim_b[1])
                step_vector_gcim_b.append(return_gcim_b[2])
            # Do the 'gcim_phased'
            if alg_vector.count('gcim_phased') > 0:
                return_gcim_phased = gcim_phased(BIC)
                return_gcim_phased_dag = imset2dag(return_gcim_phased[0])
                score_vector_gcim_phased.append(return_gcim_phased[1])
                step_vector_gcim_phased.append(return_gcim_phased[2])
            # Do the 'gcim_backwards'
            if alg_vector.count('gcim_backwards') > 0:
                initial_graph_gcim_backwards = igraph.Graph.Full(nodes)
                initial_graph_gcim_backwards.to_directed(mutual=False)
                return_gcim_backwards = gcim(BIC, initial_graph_gcim_backwards)
                return_gcim_backwards_dag = imset2dag(return_gcim_backwards[0])
                score_vector_gcim_backwards.append(return_gcim_backwards[1])
                step_vector_gcim_backwards.append(return_gcim_backwards[2])
            # Do the 'gcim_phased_b'
            if alg_vector.count('gcim_phased_b') > 0:
                return_gcim_phased_b = gcim_phased_b(BIC)
                return_gcim_phased_b_dag = imset2dag(return_gcim_phased_b[0])
                score_vector_gcim_phased_b.append(return_gcim_phased_b[1])
                step_vector_gcim_phased_b.append(return_gcim_phased_b[2])
            # Do the 'ges_gcim'
            if alg_vector.count('ges_gcim') > 0:
                if alg_vector.count('ges') > 0:
                    initial_imset_ges_gcim = imset(return_ges_dag)
                else:
                    initial_imset_ges_gcim = imset(gaussParDAG2igraph(pcalg.ges(BIC)[1]))
                return_ges_gcim = gcim(BIC, input_imset=initial_imset_ges_gcim)
                return_ges_gcim_dag = imset2dag(return_ges_gcim[0])
                score_vector_ges_gcim.append(return_ges_gcim[1])
                step_vector_ges_gcim.append(return_ges_gcim[2])
            # Do the 'naive_search' (opt)
            if alg_vector.count('opt') > 0:
                return_opt = naive_search(imsetlist, BIC)
                return_opt_dag = imset2dag(return_opt[0])
                score_vector_opt.append(return_opt[1])
            # Do the 'true'
            if alg_vector.count('true') > 0:
                return_true_dag = read_dag(
                    dataset_path + "nodes-" + str(nodes) + "/"+dataset_type + "-" + str(run) + "/" + graph_name[no])
                return_true_score = score_BIC(BIC, return_true_dag)
                score_vector_true.append(return_true_score)

            # Calculate the shd_vectors
            if alg_vector.count('opt') > 0:
                for algo in alg_vector:
                    if algo != 'opt' and algo != 'true':
                        eval("shd_vector_" + algo + "_opt.append(shd(return_" + algo + "_dag, return_opt_dag))")
            if alg_vector.count('true') > 0:
                for algo in alg_vector:
                    if algo != 'true':
                        eval("shd_vector_" + algo + "_true.append(shd(return_" + algo + "_dag, return_true_dag))")
            
            # Save the dags as edgelists for future reference
            return_dags_vector_temp = []
            for algo in alg_vector:
                return_dags_vector_temp.append(eval("return_" + algo + "_dag.get_edgelist()"))
            
            return_dags_vector.append(return_dags_vector_temp)
        print("]")
        
        # Write the results to a results.txt file.            
        result_file = open(dataset_path + "nodes-" + str(nodes) + "/" + dataset_type + "-" + str(run) + "/results.txt", "w")

        # Write what data we used
        result_file.write("Number of nodes=" + str(nodes) + "\n")
        result_file.write("Number of samples=" + str(samples_no) + "\n")
        result_file.write("Cut of limit used=" + str(alpha) + "\n")
        
        # Write the total BIC score
        for algo in alg_vector:
            result_file.write("total_score_" + algo + "=" + str(sum(eval("score_vector_" + algo))) + "\n")
        
        # Write the shd to file
        if alg_vector.count('opt') > 0:
            for algo in alg_vector:
                if algo != 'opt' and algo != 'true':
                    result_file.write("shd dist " + algo + " to opt="+ str(eval("shd_vector_" + algo + "_opt")+ "\n"))
        if alg_vector.count('true') > 0:
            for algo in alg_vector:
                if algo != 'true':
                    result_file.write("shd dist " + algo + " to true=" + str(eval("shd_vector_" + algo + "_true")) + "\n")

        # Write the score vectors
        for algo in alg_vector:
            result_file.write("\nscore_vector_" + algo + "=" + str(eval("score_vector_" + algo)) + "\n")
        
        # Write the step vectors.
        for algo in step_count_alg:
            result_file.write("step_vector_" + algo + "="+str(eval("step_vector_" + algo)) + "\n\n")
        # Close the result file
        result_file.close()
print("Run completed.")
print("Local clock:", time.asctime(time.localtime(time.time())))




for nodes in no_nodes:
    plot_vector = dataset_type_run
    
    # Shd vectors to save the shds
    if alg_vector.count('opt') > 0:
        for algo in alg_vector:
            if algo != 'opt' and algo != 'true':
                exec("shd_matrix_"+algo+"_opt = []")
    if alg_vector.count('true') > 0:
        for algo in alg_vector:
            if algo != 'true':
                exec("shd_matrix_"+algo+"_true = []")
    
    # Step vectors to get some extra data
    for algo in step_count_alg:
        exec("step_matrix_"+algo+" = []")
    
    # Get the data as a big matrix and fetch the plotting data
    for run in dataset_type_run:
        data_matrix = numpy.genfromtxt(dataset_path + "nodes-" + str(nodes) + "/" + dataset_type + "-" + str(run) + "/results.txt", dtype=None, delimiter = '=', encoding=None)
        
        for i in range(len(data_matrix)):
            for algo in alg_vector:
                if data_matrix[i][0] == "shd dist " + algo + " to true":
                    exec("shd_matrix_" + algo + "_true.append(eval(data_matrix[i][1]))")
                    break
                elif data_matrix[i][0] == "shd dist " + algo + " to opt":
                    exec("shd_matrix_" + algo + "_opt.append(eval(data_matrix[i][1]))")
                    break
                elif data_matrix[i][0] == "step_vector_" + algo:
                    exec("step_matrix_" + algo + ".append(eval(data_matrix[i][1]))")
            
    
    # Plot the average structural hamming distance between 
    # the true model and the return of the algorithms.
    matplotlib.pyplot.title("Average shd: "+str(nodes))
    if alg_vector.count('ges') > 0 and alg_vector.count('true') > 0:
        matplotlib.pyplot.plot(plot_vector, [sum(shd_matrix_ges_true[i])/len(shd_matrix_ges_true[i]) for i in range(len(shd_matrix_ges_true))], label = 'ges', color='blue', marker="d")
    if alg_vector.count('pc') > 0 and alg_vector.count('true') > 0:
        matplotlib.pyplot.plot(plot_vector, [sum(shd_matrix_pc_true[i])/len(shd_matrix_pc_true[i]) for i in range(len(shd_matrix_pc_true))], label = 'pc', color='orange', marker="o")
    if alg_vector.count('mmhc') > 0 and alg_vector.count('true') > 0:
        matplotlib.pyplot.plot(plot_vector, [sum(shd_matrix_mmhc_true[i])/len(shd_matrix_mmhc_true[i]) for i in range(len(shd_matrix_mmhc_true))], label = 'mmhc', color='cyan', marker="s")
    if alg_vector.count('greedySP') > 0 and alg_vector.count('true') > 0:
        matplotlib.pyplot.plot(plot_vector, [sum(shd_matrix_greedySP_true[i])/len(shd_matrix_greedySP_true[i]) for i in range(len(shd_matrix_greedySP_true))], label = 'greedySP', color='pink', marker="*")
    if alg_vector.count('gcim') > 0 and alg_vector.count('true') > 0:
        matplotlib.pyplot.plot(plot_vector, [sum(shd_matrix_gcim_true[i])/len(shd_matrix_gcim_true[i]) for i in range(len(shd_matrix_gcim_true))], label = 'gcim', color='magenta', marker="^")
    if alg_vector.count('gies') > 0 and alg_vector.count('true') > 0:
        matplotlib.pyplot.plot(plot_vector, [sum(shd_matrix_gies_true[i])/len(shd_matrix_gies_true[i]) for i in range(len(shd_matrix_gies_true))], label = 'gies', color='lime', marker="X")
    if alg_vector.count('ske') > 0 and alg_vector.count('true') > 0:
        matplotlib.pyplot.plot(plot_vector, [sum(shd_matrix_ske_true[i])/len(shd_matrix_ske_true[i]) for i in range(len(shd_matrix_ske_true))], label = 'ske', color='red', marker="v")
    matplotlib.pyplot.legend(loc = 'upper left')
    matplotlib.pyplot.xlim(0.5,7)
    
    matplotlib.pyplot.savefig(dataset_path + "avg_shd_nodes-" + str(nodes) + "_sample-" + str(samples_no) + "_alpha-" + str(alpha) + ".pdf")
    matplotlib.pyplot.show()
    # Plot proportion of time found true graph
    matplotlib.pyplot.title("Models recovered")
    if alg_vector.count('gcim') > 0 and alg_vector.count('true') > 0:
        matplotlib.pyplot.plot(plot_vector, [shd_matrix_gcim_true[i].count(0)/len(shd_matrix_gcim_true[i]) for i in range(len(shd_matrix_gcim_true))], label = 'gcim', color='magenta', marker="^")
    if alg_vector.count('gies') > 0 and alg_vector.count('true') > 0:
        matplotlib.pyplot.plot(plot_vector, [shd_matrix_gies_true[i].count(0)/len(shd_matrix_gies_true[i]) for i in range(len(shd_matrix_gies_true))], label = 'gies', color='lime', marker="X")
    if alg_vector.count('pc') > 0 and alg_vector.count('true') > 0:
        matplotlib.pyplot.plot(plot_vector, [shd_matrix_pc_true[i].count(0)/len(shd_matrix_pc_true[i]) for i in range(len(shd_matrix_pc_true))], label = 'pc', color='orange', marker="o",)
    if alg_vector.count('mmhc') > 0 and alg_vector.count('true') > 0:
        matplotlib.pyplot.plot(plot_vector, [shd_matrix_mmhc_true[i].count(0)/len(shd_matrix_mmhc_true[i]) for i in range(len(shd_matrix_mmhc_true))], label = 'mmhc', color='cyan', marker="s")
    if alg_vector.count('greedySP') > 0 and alg_vector.count('true') > 0:
        matplotlib.pyplot.plot(plot_vector, [shd_matrix_greedySP_true[i].count(0)/len(shd_matrix_greedySP_true[i]) for i in range(len(shd_matrix_greedySP_true))], label = 'greedySP', color='pink', marker="*")
    if alg_vector.count('ges') > 0 and alg_vector.count('true') > 0:
        matplotlib.pyplot.plot(plot_vector, [shd_matrix_ges_true[i].count(0)/len(shd_matrix_ges_true[i]) for i in range(len(shd_matrix_ges_true))], label = 'ges', color='blue', marker="d")
    if alg_vector.count('ske') > 0 and alg_vector.count('true') > 0:
        matplotlib.pyplot.plot(plot_vector, [shd_matrix_ske_true[i].count(0)/len(shd_matrix_ske_true[i]) for i in range(len(shd_matrix_ske_true))], label = 'ske', color='red', marker="v")
    matplotlib.pyplot.legend(loc = 'upper right')
    matplotlib.pyplot.ylim(0,1)
    matplotlib.pyplot.xlim(0.5,7)

    matplotlib.pyplot.savefig(dataset_path + "models_recovered_nodes-" + str(nodes) + "_sample-" + str(samples_no) + "_alpha-" + str(alpha) + ".pdf")
    matplotlib.pyplot.show()
    
    # Plot proportion average number of steps taken
    matplotlib.pyplot.title("Average number of steps taken: " + str(nodes)) 
    if alg_vector.count('gcim') > 0:
        matplotlib.pyplot.plot(plot_vector, [sum(step_matrix_gcim[i])/len(step_matrix_gcim[i]) for i in range(len(step_matrix_gcim))], label = 'gcim_steps', color='magenta', marker="^")
    if alg_vector.count('ske') > 0:
        matplotlib.pyplot.plot(plot_vector, [sum(step_matrix_ske[i])/len(step_matrix_ske[i]) for i in range(len(step_matrix_ske))], label = 'ske_steps', color='red',  marker="v")
    matplotlib.pyplot.legend(loc = 'right')
    matplotlib.pyplot.xlim(0.5,7)
    
    matplotlib.pyplot.savefig(dataset_path + "avg_no_steps_nodes-"+str(nodes)+".pdf")
    matplotlib.pyplot.show()

