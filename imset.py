
# coding: utf-8

# Code for calculating imsets and their difference. 


# Import modules.

# For graph handling.
import igraph;

# For math
import math;

# For testing the algorithms on random graphs.
import random;

# For Warnings 
import warnings;

# To do deep recursions
import sys
sys.setrecursionlimit(1500)
#import pypolymake;
#import numpy as np



# Produce all sets of size 2 or greater in a list.
# INPUT: a set
# RETURNS: all sets of cardiality greater or equal to 2 as a list.
def coordinatelist(someset):
    assert(isinstance(someset, set));
    size = len(someset)
    ret = [set({})];
    ret = powerset(someset);
    
    temp_list = ret.copy();
    for i in temp_list:
        if (len(i) < 2):
            ret.remove(i);
     
    return ret

# Produce all sets of size 2 and 3 in a list.
# INPUT: a set
# RETURNS: all sets of cardiality 2 or 3
def small_coordinatelist(someset):
    try:
        # duck typing
        someset.isdisjoint
    except AttributeError:
        raise TypeError(
            f"{powerset.__name__} accepts only a set-like object as parameter"
        ) from None
    
    size = len(someset)
    ret = [set({})];
    for element in someset:
        temp_list = ret.copy();
        #print("temp_list = ", temp_list, len(temp_list));
        for counter in range(len(temp_list)):
            #print(type(temp_list[counter]))
            if (len(temp_list[counter]) < 3):
                ret.append(temp_list[counter].union({element}));
    
    temp_list = ret.copy();
    for i in temp_list:
        if (len(i) < 2):
            ret.remove(i);
     
    return ret



# Produce the powerset as a list
# INPUT: a set.
# RETURNS: a list of all subsets.
def powerset(set_input):
    assert(isinstance(set_input, set));
    ret = [set()];
    for element in set_input:
        temp_list = ret.copy();
        for counter in range(len(temp_list)):
            ret.append(temp_list[counter].union({element}));
    return ret;



# Get the parentset and the children as sets instead of lists
# INPUT: A directed graph and a node.
# OUTPUTS: The paretset or the childset.

def parents(graph, node):
    return set(graph.predecessors(node));

def childs(graph, node):
    return set(graph.successors(node));



# Returns the value of the characteristic imset for the graph in the given coordinate.
# INPUT: a DAG and a set
# OUTPUT: a 0/1 value

def imset_coordinate(graph, coordinate_set):
    copy_set = coordinate_set.copy();
    node = next(iter(copy_set));
    temp_set = childs(graph, node).intersection(coordinate_set);
    while (len(temp_set) > 0):
        node = list(temp_set)[0];
        temp_set = childs(graph, node).intersection(coordinate_set);
    copy_set.discard(node);
    if (parents(graph, node).issuperset(copy_set)):
        return 1;
    return 0;



# Calls the coordinatelist function and calculates the imset value for each element.
# INPUT: a dag
# OUTPUT: a list of lists. The inner lists concists of [set, imsetcoordinate(graph, set)]

def imset(graph):
    assert (graph.is_dag());
    ret = [];
    vertex_set = set(range(graph.vcount()))
    coordinate_list = coordinatelist(vertex_set);
    for i in range(len(coordinate_list)):
        ret.append([coordinate_list[i], imset_coordinate(graph, coordinate_list[i])])
    return ret

# Calls the smallcoordinatelist function and calculates the imset value for all elements of size 2 and 3.
# Useful for checks where the full imset is not required ex. additions, buddings etc.
# INPUT: a dag
# OUTPUT: a list of lists. The inner lists concists of [set, imsetcoordinate(graph, set)]

def small_imset(graph):
    assert (graph.is_dag());
    ret=[];
    vertex_set = set(range(graph.vcount()))
    coordinate_list = small_coordinatelist(vertex_set);
    for i in range(len(coordinate_list)):
        ret.append([coordinate_list[i], imset_coordinate(graph, coordinate_list[i])])
    return ret
    


# # Conversion functions


# Functions for cropping imsets.
# INPUT: an imset and a size.
# RETURNS: The imset and a 

def imset2small_imset(imset_input):
    return imset_cutof(imset_input, size = 3);

def imset_cutof(imset_input, size = 3):
    ret = imset_input.copy();
    for i in imset_input:
        if (len(i[0]) > size):
            ret.remove(i);
    return ret;



# Separates the imset vector into two lists, preserving the order
# INPUT: a imset or smallimset
# OUTPUT: two lists

def imset2vec(imset_input):
    cordinatevalue = [];
    coordinatelist = [];
    for i in range(len(imset_input)):
        cordinatevalue.append(imset_input[i][1]);
        coordinatelist.append(imset_input[i][0]);
    return coordinatelist, cordinatevalue;



# Implementation of "A simple algorithm to construct a consistent
# extension of a partially oriented graph".
# https://ftp.cs.ucla.edu/pub/stat_ser/r185-dor-tarsi.pdf

def pdag2dag(graph_input):
    nodes = graph_input.vcount();
    graph_return = graph_input.copy();
    graph_temp = graph_input.copy();
    
    current_node = 0;
    while (current_node < nodes):
        ch_set = childs(graph_temp, current_node);
        pa_set = parents(graph_temp, current_node);
        ne_set = pa_set.intersection(ch_set);
        # If it is a sink
        if (len(ch_set) + len(pa_set) == 0):
            pass;
        elif (len(ch_set.difference(pa_set)) == 0):
            condition_b = True;
            for i in ne_set:
                if (not parents(graph_temp, i).union(childs(graph_temp, i), set({i})).issuperset(pa_set)):
                    condition_b = False;
            if (condition_b):
                for i in ne_set:
                    graph_return.delete_edges([(current_node, i)]);
                for i in ch_set:
                    graph_temp.delete_edges([(current_node, i)]);
                for i in pa_set:
                    graph_temp.delete_edges([(i, current_node)]);
                current_node = 0;
        current_node+=1;
            
    
    if (not graph_return.is_dag()):
        warnings.warn("In: pdag2dag. No extension exists, returning something. Dunno what.");
    return graph_return;



# Takes a full size imset and outputs a directed graph
# that is graphically equivalent to all graphs in the MEC.
# Undirected edges are portrated by mutual edges.

def imset2pdag(imset_input):
    nodes = math.ceil(math.log2(len(imset_input)));
    graph = igraph.Graph(nodes, directed = True);
    
    # Add edges according to the 2-sets
    for i in range(nodes):
        for j in range(i+1, nodes):
            if (imset_input[2**i+2**j-j-2][1] == 1):
                graph.add_edge(i,j);
                graph.add_edge(j,i);
    
    # Add immoralities according to the 3-sets
    for i in range(nodes):
        for j in range(i+1, nodes):
            for k in range(j+1,nodes):
                if (imset_input[2**i+2**j+2**k-k-2][1] == 1):
                    temp_graph = graph.induced_subgraph([i,j,k]);
                    temp_graph.to_undirected();
                    if (temp_graph.omega() < 3):
                        if (not graph.are_connected(i,j)):
                            try:
                                graph.delete_edges([(k,i)]);
                            except ValueError:
                                pass;
                            try:
                                graph.delete_edges([(k,j)]);
                            except ValueError:
                                pass;
                        elif (not graph.are_connected(i,k)):
                            try:
                                graph.delete_edges([(j,i)]);
                            except ValueError:
                                pass;
                            try:
                                graph.delete_edges([(j,k)]);
                            except ValueError:
                                pass;
                        elif (not graph.are_connected(j,k)):
                            try:
                                graph.delete_edges([(i,j)]);
                            except ValueError:
                                pass;
                            try:
                                graph.delete_edges([(i,k)]);
                            except ValueError:
                                pass;
                        else: 
                            warnings.warn("Something is wrong with the imset, returning rubbish (probably)");
    return graph;



# Takes a full size imset and returns a directed graph
# in the MEC. 

def imset2dag(imset_input):
    return pdag2dag(imset2pdag(imset_input));
    



# Calculates imset1 - imset2. Returns lists A and B of sets 
# such that imset1 - imset2 = \sum_{S\in A}e_S - \sum_{S\in B}e_S
# if passed two graphs, return the corresponding for their imsets.
# INPUT: Two graphs/imsets.
# RETURNS: Two lists.

def imsetdif(graph1, graph2):
    if (isinstance(graph1, igraph.Graph)):
        imset1 = imset(graph1);
    else:
        imset1 = graph1;
    if (isinstance(graph2, igraph.Graph)):
        imset2 = imset(graph2);
    else:
        imset2 = graph2;
    A=[];
    B=[];
    if (len(imset1)!=len(imset2)):
        print(imset1, imset2);
        warnings.warn("Imsets/graphs must be of equal size\n");
        return A, B;
    for i in range(len(imset1)):
        if (imset1[i][1] == 1):
            if (imset2[i][1] == 0):
                A.append(imset1[i][0]);
        else:
            if (imset2[i][1] == 1):
                B.append(imset1[i][0]);
    return A,B;



# Checks the value of the imset in the given coordinate.
# if given an invalid coordinate, returns -1.
def imset_value(imset_input, set_input):
    for i in imset_input:
        if (i[0] == set_input):
            return i[1];
    warnings.warn("Imset coordinate not found, returning -1");
    print("Currently checking set:", set_input);
    input("A quick pause for debugging.");
    return -1;
    


# A check to see wether two lists contains the 
# same elements regardless of order. Use only 
# when elements are unhashable and unsortable 
# (for example, sets).
# INPUT: 2 lists.
# RETURNS: A boolean stating wether they contain the same elements.

def set_lists_equal(list1, list2):
    copy1 = list1.copy();
    copy2 = list2.copy();
    for i in copy1:
        try:
            copy2.remove(i);
        except ValueError:
            return False;
        
    if (len(copy2)==0):
        return True;
    return False;
    



def is_markovequiv(graph1, graph2):
    if (isinstance(graph1, igraph.Graph)):
        imset1 = small_imset(graph1);
    else:
        imset1 = graph1.copy();
    if (isinstance(graph2, igraph.Graph)):
        imset2 = small_imset(graph2);
    else: 
        imset2 = graph2.copy();
        
    return (imset2small_imset(imset1) == imset2small_imset(imset2));



# Returns whether the pair (graph1, graph2) is an addition.
# Can take both graphs and imsets as inputs
# INPUT: Two gaphs/imsets (full size imsets only)
# OUTPUTS: A boolean

def is_addition(graph1, graph2):
    if (isinstance(graph1, igraph.Graph)):
        imset1 = imset(graph1);
    else:
        imset1 = graph1.copy();
    if (isinstance(graph2, igraph.Graph)):
        imset2 = imset(graph2);
    else: 
        imset2 = graph2.copy();
    A, B=imsetdif(imset1, imset2);
    if (len(A)+len(B) == 1):
        return True;
    return False;



# Returns whether the pair (graph1, graph2) is a budding.
# verbose option for debugging
# Can take both graphs and imsets as inputs
# INPUT: Two gaphs/imsets (full size imsets only)
# OUTPUTS: A boolean

def is_budding(graph1, graph2, verbose = False):
    if (isinstance(graph1, igraph.Graph)):
        imset1 = imset(graph1);
    else:
        imset1 = graph1.copy();
    if (isinstance(graph2, igraph.Graph)):
        imset2 = imset(graph2);
    else: 
        imset2 = graph2.copy();
    #imset1 = imset2small_imset(imset1);
    #imset2 = imset2small_imset(imset2);
    A, B = imsetdif(imset1, imset2);
    # Check that exactly one of the sets are empty, 
    # we don't have an addition and set that A is 
    # the non-empty one
    if (len(A)+len(B) < 2):
        if (verbose) :print("False beacuse 1"); 
        return False;
    if (len(A) > 0):
        if (len(B) >0):
            if (verbose) :print("False beacuse 2");
            return False;
    else:
        A = B;
        B = [];
    # Now make sure that A indeed is a budding set
    # Make sure they contain no sets of size 2.
    for i in A:
        if (len(i) < 3):
            if (verbose) :print("False beacuse 3");
            return False;
    # Find i, j and S*
    union_set = set();
    for i in A:
        union_set.update(i);
    intersection_set = union_set;
    for i in A:
        intersection_set = intersection_set.intersection(i);
    if (len(intersection_set) < 2 or len(intersection_set) > 3):
        if (verbose) :print("False beacuse 4");
        return False;
    nodei = -1;
    nodej = -1;
    if (len(intersection_set) == 3):
        temp_list = list(intersection_set);
        if (imset_value(imset1, set({temp_list[0], temp_list[1]})) == 0):
            nodei = temp_list[2];
            temp_list[2] = temp_list[0];
            temp_list[0]=nodei;
        elif (imset_value(imset1, set({temp_list[0], temp_list[2]})) == 0):
            nodei = temp_list[1];
            temp_list[1] = temp_list[0];
            temp_list[0]=nodei;
        elif (imset_value(imset1, set({temp_list[1], temp_list[2]})) == 0):
            nodei = temp_list[0];
        else: 
            if (verbose) :print("False beacuse 5");
            return False;
        for i in union_set.difference(intersection_set):
            if (imset_value(imset1, set({i, temp_list[1]})) == 0):
                nodej = temp_list[2];
        if (nodej == -1):
            nodej = temp_list[1]; 
    else:
        #print (intersection_set);
        node1 = list(intersection_set)[0];
        node2 = list(intersection_set)[1];
        
        for i in union_set.difference(intersection_set):
            if (imset_value(imset1, set({node1, i})) == 0):
                nodej = node1;
                nodei = node2;
                break;
        if (nodei == -1):
            for i in union_set.difference(intersection_set):
                if (imset_value(imset1, set({node2, i})) == 0):
                    nodej = node2;
                    nodei = node1;
                    break;
        if (nodei == -1):
            if (verbose) :print("False beacuse 6");
            return False;
    S = union_set.copy().difference({nodei, nodej});
    
    ne_set = set();
    for i in S:
        if (imset_value(imset1, {nodej, i}) == 1):
            ne_set.add(i);
    # If this is a budding we now have i, j, S and ne(i)\cap ne(j) 
    if (verbose):
        print ("i =", nodei);
        print ("j =", nodej);
        print ("S =", S);
        print ("ne =", ne_set);
    
    if (not imset_value(imset1, {nodei, nodej}) == 1):
        return False;
    
    pow_set = powerset(S);
    pow_set.remove(set());
    
    for i in pow_set:
        if (not imset_value(imset1, i.union({nodei})) == 1):
            if (verbose): print("False beacuse 7", i.union({nodei}));
            return False;
    
    
    bud_set = [];
    for i in pow_set:
        if (not i.issubset(ne_set)):
            bud_set.append(i.union({nodei, nodej}));
    if (verbose): 
        print(sorted(bud_set));
        print(sorted(A));
        
    if (set_lists_equal(A, bud_set)):
        return True;
    
    if (verbose): print("False beacuse default");
    return False;



# Returns whether the pair (graph1, graph2) is a budding.
# verbose option for debugging
# Can take both graphs and imsets as inputs
# INPUT: Two gaphs/imsets (full size imsets only)
# OUTPUTS: A boolean

def is_ges(graph1, graph2, verbose = False):
    if (isinstance(graph1, igraph.Graph)):
        imset1 = imset(graph1);
    else:
        imset1 = graph1.copy();
    if (isinstance(graph2, igraph.Graph)):
        imset2 = imset(graph2);
    else: 
        imset2 = graph2.copy();
    A, B = imsetdif(imset1, imset2);
    if (verbose): print(A, B);
    # Check that exactly one of the sets are empty, 
    # we don't have an addition and set that A is 
    # the non-empty one
    if (len(A)+len(B) < 2):
        if (verbose) :print("False beacuse 1"); 
        return False;
    if (len(A) > 0):
        if (len(B) >0):
            if (verbose) :print("False beacuse 2");
            return False;
    else:
        A = B;
        B = [];
    # Now make sure that A indeed is a budding set
    # Make sure they contain no sets of size 2.
    edge = [-1, -1];
    for i in A:
        if (len(i) == 2):
            if (edge[0] != -1):
                if (verbose) :print("False beacuse differs by more than one 2-set");
                return False;
            else:
                temp_list = list(i);
                edge[0] = temp_list[0];
                edge[1] = temp_list[1];
    if (edge[0] == -1):
            if (verbose) :print("False beacuse differs no 2-set");
            return False;
    
    union_set = set();
    for i in A:
        union_set.update(i);
    pow_set = powerset(union_set.difference(set(edge)));
    ges_set = [];
    for i in pow_set:
        ges_set.append(i.union(set(edge)));
    if (not set_lists_equal(ges_set, A)):
        if (verbose) :print("False beacuse 3");
        return False;
    
    check = True;
    pow_set.remove(set());
    for i in pow_set:
        if (imset_value(imset1, i.union({edge[0]})) == 0):
            check = False;
    
    if (check):
        return True;
    
    for i in pow_set:
        if (imset_value(imset1, i.union({edge[1]})) == 0):
            if (verbose) :print("False beacuse 3");
            return False;
    
    return True;



# Returns whether the pair (graph1, graph2) is a flip.
# verbose option for debugging
# Can take both graphs and imsets as options
# INPUT: Two gaphs/imsets (full size imsets only)
# OUTPUTS: A boolean

def is_flip(graph1, graph2, verbose = False):
    if (isinstance(graph1, igraph.Graph)):
        imset1 = imset(graph1);
    else:
        imset1 = graph1.copy();
    if (isinstance(graph2, igraph.Graph)):
        imset2 = imset(graph2);
    else: 
        imset2 = graph2.copy();
    #imset1 = imset2small_imset(imset1);
    #imset2 = imset2small_imset(imset2);
    A, B = imsetdif(imset2, imset1);
    if (verbose): print(A, B);
    if (len(A) == 0 or len(B) == 0):
        if (verbose): print("False beacuse 1");
        return False;
    for i in A:
        if (len(i) == 2):
            if (verbose): print("False because 2");
            return False;
    for i in B:
        if (len(i) == 2):
            if (verbose): print("False because 2");
            return False
    
    union_set1 = set();
    union_set2 = set();
    for i in A:
        union_set1.update(i);
    for i in B:
        union_set2.update(i);
    intersection_set = union_set1.copy();
    for i in B:
        intersection_set.intersection_update(i);
    
    if (len(intersection_set) != 2):
        if (verbose): print("False beacuse 3");
        return False;
    
    ne_set = set();
    intersection_list = list(intersection_set);
    for i in union_set1.union(union_set2).difference(intersection_set):
        if (imset_value(imset1, set({i, intersection_list[0]})) == 1 and imset_value(imset1, set({i, intersection_list[1]})) == 1):
            ne_set.add(i);
    if (verbose): print("ne_set =", ne_set);
    
    nodei = -1;
    nodej = -1;
    
    for i in union_set1.difference(ne_set).difference(intersection_set):
        if (imset_value(imset1, set({i, intersection_list[0]})) == 0):
            nodej = intersection_list[0];
            nodei = intersection_list[1];
            break;
    if (nodei == -1):
        nodej = intersection_list[1];
        nodei = intersection_list[0];
    
    Si=union_set1.difference(intersection_set);
    Sj=union_set2.difference(intersection_set);
    # If this is a budding we should have i, j, Si, and Sj now.
    if (verbose):
        print("i =", nodei);
        print("j =", nodej);
        print("Si =", Si);
        print("Sj =", Sj);
    
    
    pow_set_i = powerset(Si);
    pow_set_i.remove(set());
    for i in pow_set_i:
        if (imset_value(imset1, i.union({nodei})) == 0):
            if (verbose): print("False because 4");
            return False
    pow_set_j = powerset(Sj);
    pow_set_j.remove(set());
    for i in pow_set_j:
        if (imset_value(imset1, i.union({nodej})) == 0):
            if (verbose): print("False because 4");
            return False
    
    flip_set_pos = [];
    flip_set_neg = [];
    
    for i in pow_set_i:
        if (not i.issubset(ne_set)):
            flip_set_pos.append(i.union({nodei, nodej}));
    for i in pow_set_j:
        if (not i.issubset(ne_set)):
            flip_set_neg.append(i.union({nodei, nodej}));
    
    if (verbose): 
        print(flip_set_pos);
        print(A);
        print(flip_set_neg);
        print(B);
    
    if (set_lists_equal(A, flip_set_pos) and set_lists_equal(B, flip_set_neg)):
        return True;
    
    if (verbose): print("False because default");
    return False;



# Try all acyclic orientations of an undirected graph.
# Does this naively and does nothing to ignore markov equivalent graphs.
# INPUT: An undirected or partially directed graph.
# RETURNS: A list of imsets values for all acyclic orientations 

def try_acyclic_orientation(graph):
    imsetlist = [];
    if(graph.is_directed()):
        print("The input to 'try_acyclic_orientation' was directed.\nWe will only direct the mutual edges.\nWe suggest not doing this.");
        dir_graph = graph.copy()
    else:
        dir_graph = graph.copy().to_directed();
    try_acyclic_orientation_help(dir_graph, imsetlist);
    return imsetlist;


def try_acyclic_orientation_help(graph, imsetlist):
    if (graph.is_dag()):
        imset_temp = imset2vec(small_imset(graph))[1];
        imsetlist.append(imset_temp);
        #print(imset2vec(imset(graph))[1]);
    else:
        check = False;
        edgelist=graph.get_edgelist();
        edge = edgelist[0];
        for i in range(len(edgelist)):
            if (graph.is_mutual(edgelist[i])):
                edge = edgelist[i];
                i = len(edgelist);
                check = True;
        if (check):
            graph_copy1=graph.copy();
            graph_copy1.delete_edges([edge]);
            try_acyclic_orientation_help(graph_copy1, imsetlist);
            del graph_copy1;
            graph_copy2=graph.copy();
            graph_copy2.delete_edges([(edge[1], edge[0])]);
            try_acyclic_orientation_help(graph_copy2, imsetlist);
            del graph_copy2;
        



def try_all_dags(nodes):
    imsetlist = [];
    graph = igraph.Graph(nodes, directed = True);
    try_all_dags_help(graph, imsetlist, [0, 0]);
    return imsetlist;

def try_all_dags_help(graph, imsetlist, edge):
    if ((edge[0] == graph.vcount()-1) and (edge[1] ==  graph.vcount()-1)):
        imsetlist.append(imset(graph));
    else:
        edge[0] = edge[0] + 1;
        if (edge[0] == graph.vcount()):
            edge[0] = 0;
            edge[1] = edge[1] + 1;
        try_all_dags_help(graph.copy(), imsetlist, edge.copy());
        graph.add_edges([edge]);
        if (graph.is_dag()):
            try_all_dags_help(graph.copy(), imsetlist, edge.copy());
    


# Try all acyclic orientations of an undirected graph.
# Does this naively and does nothing to ignore markov equivalent graphs.
# INPUT: An undirected or partially directed graph.
# RETURNS: A list of small_imsets values for all acyclic orientations 

def small_try_acyclic_orientation(graph):
    imsetlist = [];
    if(graph.is_directed()):
        print("The input to 'small_try_acyclic_orientation' was directed.\nWe will only direct the mutual edges.\nWe suggest not doing this.");
        dir_graph = graph.copy()
    else:
        dir_graph = graph.copy();
        dir_graph.to_directed();
    small_try_acyclic_orientation_help(dir_graph, imsetlist);
    return imsetlist;

def small_try_acyclic_orientation_help(graph, imsetlist):
    if (graph.is_dag()):
        imset_temp = imset2vec(small_imset(graph))[1];
        imsetlist.append(imset_temp);
        #print(imset_temp);
    else:
        check = False;
        edgelist=graph.get_edgelist();
        edge = edgelist[0]
        for i in range(len(edgelist)):
            if (graph.is_mutual(edgelist[i])):
                edge = edgelist[i];
                i = len(edgelist);
                check = True;
        if (check):
            graph_copy1=graph.copy();
            graph_copy1.delete_edges([edge]);
            small_try_acyclic_orientation_help(graph_copy1, imsetlist);
            del graph_copy1;
            graph_copy2=graph.copy();
            graph_copy2.delete_edges([(edge[1], edge[0])]);
            small_try_acyclicorientation_help(graph_copy2, imsetlist);
            del graph_copy2;




# Code to test the is_addition and is_ges code.

for i in range(1): 
    rand_nodes = random.randint(5,5);
    rand_edges = random.randint(rand_nodes, 2*rand_nodes);
    g = g.Erdos_Renyi(rand_nodes, m=rand_edges);
    g.to_directed(mutual = False);
    h = g.copy();
    rand_int = random.randint(0, g.ecount()-1);
    edge = g.get_edgelist()[rand_int];
    h.delete_edges([edge]);
    
    if (h.is_dag()):
        imsetg = imset(g);
        imseth = imset(h);
        if (is_addition(imseth, imsetg)):
            pass;
        elif (is_ges(imseth, imsetg)):
            pass;
        else:
            print(g);
            print(h);
            print(edge);
            print("WhuuuuuuuuuuT?!?!?!?");
            break;

    


# Code to test the is_addition, is_budding, is_markovequiv, and is_flip code.

for i in range(1): 
    rand_nodes = random.randint(5,15);
    rand_edges = random.randint(rand_nodes, 2*rand_nodes);
    g = g.Erdos_Renyi(rand_nodes, m=rand_edges);
    g.to_directed(mutual = False);
    h = g.copy();
    rand_int = random.randint(0, g.ecount()-1);
    edge = g.get_edgelist()[rand_int];
    h.delete_edges([edge]);
    h.add_edges([(edge[1], edge[0])]);
    
    if (h.is_dag()):
        imsetg = imset(g);
        imseth = imset(h);
        if (is_markovequiv(imseth, imsetg)):
            pass;
        elif (is_addition(imsetg,imseth)):
            pass
        elif (is_budding(imsetg,imseth)):
            pass;
        elif (is_flip(imsetg, imseth)):
              pass;
        else:
            print(g);
            print(h);
            print(edge);
            print("WhuuuuuuuuuuT?!?!?!?");
            break;



# Code to test the imset2dag function.

for i in range(1000): 
    rand_nodes = random.randint(5,15);
    rand_edges = random.randint(rand_nodes, 2*rand_nodes);
    g = g.Erdos_Renyi(rand_nodes, m=rand_edges);
    g.to_directed(mutual = False);
    imsetg = imset(g);
    if (not is_markovequiv(g, imset2dag(imset(g)))):
        print(g);
        print("WhuuuuuuuuuuT?!?!?!?");
        break;



# Test all dags on 4 nodes


imsetlist = try_all_dags(4);


# Remove duplicates
i = 0;
while (i < len(imsetlist)):
    j= i+1;
    while (j < len(imsetlist)):
        if (imsetlist[i] == imsetlist[j]):
            imsetlist.pop(j);
            j-=1;
        j+=1;
    i+=1;

# Extract the coordinates
vertexlist = [];
for i in range(len(imsetlist)):
    vertexlist.append(imset2vec(imsetlist[i])[1]);


# Transform into homogenic coordinats
hom_coordinates = [];

for i in range(len(vertexlist)):
    temp_list = [1];
    for j in range(len(vertexlist[i])):
        temp_list.append(vertexlist[i][j]);
    hom_coordinates.append(temp_list);


print(len(vertexlist))





edgelist4 = [[0, 1], [0, 2], [0, 3], [1, 3], [2, 3], [0, 4], [0, 5], [1, 5], [3, 5], [4, 5], [0, 6], [2, 6], [3, 6], [4, 6], [5, 6], [0, 7], [1, 7], [2, 7], [3, 7], [4, 7], [5, 7], [6, 7], [1, 8], [2, 8], [3, 8], [7, 8], [1, 9], [4, 9], [5, 9], [7, 9], [1, 10], [3, 10], [5, 10], [6, 10], [7, 10], [8, 10], [9, 10], [0, 11], [1, 12], [8, 12], [10, 12], [11, 12], [2, 13], [8, 13], [11, 13], [12, 13], [3, 14], [8, 14], [12, 14], [13, 14], [4, 15], [11, 15], [5, 16], [10, 16], [11, 16], [12, 16], [13, 16], [15, 16], [6, 17], [10, 17], [11, 17], [12, 17], [13, 17], [15, 17], [16, 17], [7, 18], [8, 18], [10, 18], [11, 18], [12, 18], [13, 18], [14, 18], [15, 18], [16, 18], [17, 18], [0, 19], [1, 19], [3, 19], [5, 19], [7, 19], [11, 19], [12, 19], [14, 19], [16, 19], [18, 19], [4, 20], [5, 20], [7, 20], [9, 20], [15, 20], [16, 20], [18, 20], [19, 20], [3, 21], [5, 21], [6, 21], [7, 21], [10, 21], [12, 21], [14, 21], [16, 21], [17, 21], [18, 21], [19, 21], [20, 21], [0, 22], [1, 23], [9, 23], [10, 23], [22, 23], [2, 24], [22, 24], [3, 25], [10, 25], [21, 25], [22, 25], [23, 25], [24, 25], [4, 26], [9, 26], [20, 26], [22, 26], [23, 26], [25, 26], [5, 27], [9, 27], [23, 27], [26, 27], [6, 28], [10, 28], [21, 28], [22, 28], [23, 28], [24, 28], [25, 28], [26, 28], [7, 29], [9, 29], [10, 29], [20, 29], [21, 29], [22, 29], [23, 29], [24, 29], [25, 29], [26, 29], [27, 29], [28, 29], [0, 30], [1, 30], [3, 30], [5, 30], [7, 30], [19, 30], [22, 30], [23, 30], [25, 30], [27, 30], [29, 30], [2, 31], [3, 31], [7, 31], [8, 31], [13, 31], [18, 31], [24, 31], [25, 31], [29, 31], [30, 31], [3, 32], [5, 32], [6, 32], [7, 32], [10, 32], [16, 32], [17, 32], [18, 32], [21, 32], [23, 32], [25, 32], [27, 32], [28, 32], [29, 32], [30, 32], [31, 32], [0, 33], [11, 33], [19, 33], [22, 33], [30, 33], [1, 34], [8, 34], [9, 34], [10, 34], [12, 34], [19, 34], [20, 34], [21, 34], [23, 34], [30, 34], [31, 34], [32, 34], [33, 34], [2, 35], [8, 35], [13, 35], [24, 35], [31, 35], [33, 35], [34, 35], [3, 36], [8, 36], [10, 36], [14, 36], [19, 36], [20, 36], [21, 36], [25, 36], [30, 36], [31, 36], [32, 36], [33, 36], [34, 36], [35, 36], [4, 37], [9, 37], [15, 37], [20, 37], [26, 37], [33, 37], [34, 37], [36, 37], [5, 38], [9, 38], [10, 38], [16, 38], [19, 38], [20, 38], [21, 38], [27, 38], [30, 38], [31, 38], [32, 38], [33, 38], [34, 38], [35, 38], [36, 38], [37, 38], [6, 39], [10, 39], [17, 39], [21, 39], [28, 39], [32, 39], [33, 39], [34, 39], [35, 39], [36, 39], [37, 39], [38, 39], [7, 40], [10, 40], [18, 40], [21, 40], [29, 40], [32, 40], [34, 40], [36, 40], [38, 40], [39, 40], [0, 41], [1, 41], [3, 41], [5, 41], [7, 41], [11, 41], [12, 41], [16, 41], [18, 41], [19, 41], [22, 41], [23, 41], [25, 41], [29, 41], [30, 41], [33, 41], [34, 41], [36, 41], [38, 41], [40, 41], [2, 42], [3, 42], [7, 42], [8, 42], [12, 42], [13, 42], [14, 42], [16, 42], [18, 42], [24, 42], [25, 42], [29, 42], [31, 42], [34, 42], [35, 42], [36, 42], [38, 42], [40, 42], [41, 42], [4, 43], [5, 43], [7, 43], [9, 43], [15, 43], [16, 43], [18, 43], [20, 43], [23, 43], [25, 43], [26, 43], [27, 43], [29, 43], [34, 43], [36, 43], [37, 43], [38, 43], [40, 43], [41, 43], [2, 44], [4, 44], [6, 44], [7, 44], [2, 45], [3, 45], [5, 45], [6, 45], [7, 45], [8, 45], [10, 45], [13, 45], [16, 45], [17, 45], [18, 45], [31, 45], [32, 45], [35, 45], [36, 45], [38, 45], [39, 45], [40, 45], [42, 45], [44, 45], [7, 46], [8, 46], [9, 46], [10, 46], [18, 46], [29, 46], [34, 46], [36, 46], [38, 46], [44, 46], [45, 46], [13, 47], [15, 47], [16, 47], [17, 47], [18, 47], [44, 47], [45, 47], [46, 47], [14, 48], [18, 48], [20, 48], [21, 48], [36, 48], [38, 48], [46, 48], [47, 48], [8, 49], [10, 49], [23, 49], [24, 49], [25, 49], [28, 49], [29, 49], [31, 49], [32, 49], [34, 49], [35, 49], [36, 49], [39, 49], [40, 49], [42, 49], [46, 49], [24, 50], [25, 50], [26, 50], [28, 50], [29, 50], [44, 50], [46, 50], [47, 50], [49, 50], [27, 51], [29, 51], [31, 51], [32, 51], [36, 51], [38, 51], [45, 51], [46, 51], [49, 51], [50, 51], [34, 52], [35, 52], [36, 52], [37, 52], [38, 52], [39, 52], [44, 52], [45, 52], [46, 52], [47, 52], [48, 52], [49, 52], [50, 52], [51, 52], [7, 53], [16, 53], [18, 53], [25, 53], [29, 53], [34, 53], [36, 53], [38, 53], [40, 53], [42, 53], [43, 53], [44, 53], [45, 53], [46, 53], [47, 53], [48, 53], [49, 53], [50, 53], [51, 53], [52, 53], [9, 54], [10, 54], [12, 54], [15, 54], [16, 54], [17, 54], [18, 54], [20, 54], [21, 54], [34, 54], [37, 54], [38, 54], [39, 54], [40, 54], [43, 54], [46, 54], [47, 54], [48, 54], [52, 54], [53, 54], [11, 55], [22, 55], [33, 55], [41, 55], [10, 56], [12, 56], [21, 56], [23, 56], [32, 56], [34, 56], [41, 56], [42, 56], [43, 56], [49, 56], [52, 56], [53, 56], [54, 56], [55, 56], [13, 57], [24, 57], [31, 57], [35, 57], [42, 57], [47, 57], [49, 57], [50, 57], [52, 57], [53, 57], [55, 57], [56, 57], [14, 58], [21, 58], [25, 58], [32, 58], [36, 58], [42, 58], [48, 58], [49, 58], [52, 58], [53, 58], [56, 58], [57, 58], [15, 59], [20, 59], [26, 59], [37, 59], [43, 59], [47, 59], [50, 59], [52, 59], [53, 59], [54, 59], [55, 59], [56, 59], [57, 59], [16, 60], [21, 60], [27, 60], [32, 60], [38, 60], [43, 60], [51, 60], [52, 60], [53, 60], [54, 60], [56, 60], [59, 60], [10, 61], [17, 61], [21, 61], [28, 61], [32, 61], [39, 61], [46, 61], [47, 61], [48, 61], [49, 61], [50, 61], [51, 61], [52, 61], [54, 61], [55, 61], [56, 61], [57, 61], [58, 61], [59, 61], [60, 61], [10, 62], [18, 62], [20, 62], [21, 62], [29, 62], [31, 62], [32, 62], [40, 62], [41, 62], [42, 62], [43, 62], [46, 62], [47, 62], [48, 62], [49, 62], [50, 62], [51, 62], [53, 62], [54, 62], [55, 62], [56, 62], [57, 62], [58, 62], [59, 62], [60, 62], [61, 62], [11, 63], [12, 63], [13, 63], [16, 63], [17, 63], [18, 63], [19, 63], [21, 63], [30, 63], [31, 63], [32, 63], [33, 63], [34, 63], [35, 63], [38, 63], [39, 63], [40, 63], [41, 63], [42, 63], [55, 63], [56, 63], [57, 63], [60, 63], [61, 63], [62, 63], [0, 64], [2, 64], [3, 64], [6, 64], [7, 64], [11, 64], [13, 64], [14, 64], [17, 64], [18, 64], [19, 64], [21, 64], [33, 64], [35, 64], [36, 64], [39, 64], [40, 64], [41, 64], [42, 64], [4, 65], [6, 65], [7, 65], [15, 65], [17, 65], [18, 65], [20, 65], [21, 65], [36, 65], [37, 65], [39, 65], [40, 65], [43, 65], [44, 65], [47, 65], [48, 65], [52, 65], [53, 65], [64, 65], [3, 66], [5, 66], [6, 66], [7, 66], [13, 66], [14, 66], [16, 66], [17, 66], [18, 66], [19, 66], [20, 66], [21, 66], [32, 66], [35, 66], [36, 66], [38, 66], [39, 66], [40, 66], [41, 66], [42, 66], [43, 66], [45, 66], [47, 66], [48, 66], [52, 66], [53, 66], [64, 66], [65, 66], [21, 67], [22, 67], [24, 67], [25, 67], [28, 67], [29, 67], [33, 67], [35, 67], [36, 67], [39, 67], [40, 67], [41, 67], [42, 67], [55, 67], [57, 67], [58, 67], [61, 67], [62, 67], [64, 67], [20, 68], [21, 68], [25, 68], [26, 68], [28, 68], [29, 68], [36, 68], [37, 68], [39, 68], [40, 68], [43, 68], [47, 68], [48, 68], [50, 68], [52, 68], [53, 68], [57, 68], [58, 68], [59, 68], [61, 68], [62, 68], [65, 68], [67, 68], [21, 69], [32, 69], [36, 69], [38, 69], [39, 69], [40, 69], [48, 69], [51, 69], [52, 69], [53, 69], [58, 69], [60, 69], [61, 69], [62, 69], [66, 69], [68, 69], [3, 70], [7, 70], [13, 70], [14, 70], [17, 70], [18, 70], [19, 70], [21, 70], [25, 70], [29, 70], [30, 70], [31, 70], [32, 70], [33, 70], [35, 70], [36, 70], [38, 70], [39, 70], [40, 70], [41, 70], [42, 70], [57, 70], [58, 70], [61, 70], [62, 70], [63, 70], [64, 70], [66, 70], [67, 70], [69, 70], [0, 71], [1, 72], [71, 72], [2, 73], [44, 73], [45, 73], [71, 73], [3, 74], [45, 74], [66, 74], [70, 74], [71, 74], [72, 74], [73, 74], [4, 75], [44, 75], [65, 75], [71, 75], [73, 75], [74, 75], [5, 76], [45, 76], [66, 76], [71, 76], [72, 76], [73, 76], [74, 76], [75, 76], [6, 77], [44, 77], [73, 77], [75, 77], [7, 78], [44, 78], [45, 78], [46, 78], [53, 78], [65, 78], [66, 78], [70, 78], [71, 78], [72, 78], [73, 78], [74, 78], [75, 78], [76, 78], [77, 78], [8, 79], [45, 79], [46, 79], [49, 79], [72, 79], [73, 79], [74, 79], [76, 79], [78, 79], [9, 80], [46, 80], [54, 80], [72, 80], [74, 80], [75, 80], [76, 80], [78, 80], [79, 80], [10, 81], [46, 81], [61, 81], [62, 81], [77, 81], [78, 81], [79, 81], [80, 81], [11, 82], [55, 82], [63, 82], [71, 82], [12, 83], [54, 83], [56, 83], [63, 83], [72, 83], [79, 83], [80, 83], [82, 83], [13, 84], [45, 84], [47, 84], [57, 84], [63, 84], [66, 84], [68, 84], [70, 84], [73, 84], [79, 84], [82, 84], [83, 84], [14, 85], [48, 85], [58, 85], [66, 85], [68, 85], [69, 85], [70, 85], [74, 85], [79, 85], [83, 85], [84, 85], [15, 86], [47, 86], [54, 86], [59, 86], [65, 86], [68, 86], [75, 86], [80, 86], [82, 86], [83, 86], [84, 86], [16, 87], [45, 87], [46, 87], [47, 87], [48, 87], [51, 87], [53, 87], [54, 87], [60, 87], [63, 87], [66, 87], [69, 87], [70, 87], [76, 87], [79, 87], [80, 87], [81, 87], [82, 87], [83, 87], [84, 87], [85, 87], [86, 87], [17, 88], [47, 88], [61, 88], [66, 88], [68, 88], [77, 88], [81, 88], [84, 88], [86, 88], [87, 88], [18, 89], [45, 89], [46, 89], [47, 89], [48, 89], [51, 89], [53, 89], [54, 89], [62, 89], [63, 89], [65, 89], [66, 89], [68, 89], [69, 89], [70, 89], [78, 89], [79, 89], [80, 89], [81, 89], [82, 89], [83, 89], [84, 89], [85, 89], [86, 89], [87, 89], [88, 89], [19, 90], [63, 90], [66, 90], [70, 90], [71, 90], [72, 90], [74, 90], [76, 90], [78, 90], [82, 90], [83, 90], [85, 90], [87, 90], [89, 90], [20, 91], [48, 91], [54, 91], [56, 91], [58, 91], [59, 91], [60, 91], [62, 91], [65, 91], [66, 91], [68, 91], [69, 91], [74, 91], [75, 91], [76, 91], [78, 91], [80, 91], [83, 91], [85, 91], [86, 91], [87, 91], [89, 91], [90, 91], [21, 92], [48, 92], [58, 92], [61, 92], [62, 92], [66, 92], [68, 92], [69, 92], [81, 92], [85, 92], [87, 92], [88, 92], [89, 92], [91, 92], [22, 93], [55, 93], [67, 93], [71, 93], [82, 93], [90, 93], [23, 94], [49, 94], [56, 94], [72, 94], [79, 94], [80, 94], [83, 94], [90, 94], [91, 94], [93, 94], [24, 95], [49, 95], [50, 95], [57, 95], [67, 95], [68, 95], [73, 95], [79, 95], [84, 95], [93, 95], [94, 95], [25, 96], [46, 96], [48, 96], [49, 96], [50, 96], [51, 96], [53, 96], [58, 96], [67, 96], [68, 96], [69, 96], [70, 96], [74, 96], [79, 96], [80, 96], [81, 96], [85, 96], [90, 96], [91, 96], [92, 96], [93, 96], [94, 96], [95, 96], [26, 97], [50, 97], [59, 97], [68, 97], [75, 97], [80, 97], [86, 97], [91, 97], [93, 97], [94, 97], [95, 97], [96, 97], [27, 98], [51, 98], [60, 98], [69, 98], [76, 98], [80, 98], [87, 98], [91, 98], [94, 98], [96, 98], [97, 98], [28, 99], [50, 99], [61, 99], [68, 99], [77, 99], [81, 99], [88, 99], [92, 99], [95, 99], [96, 99], [97, 99], [29, 100], [46, 100], [48, 100], [49, 100], [50, 100], [51, 100], [53, 100], [62, 100], [67, 100], [68, 100], [69, 100], [70, 100], [78, 100], [79, 100], [80, 100], [81, 100], [89, 100], [90, 100], [91, 100], [92, 100], [93, 100], [94, 100], [95, 100], [96, 100], [97, 100], [98, 100], [99, 100], [30, 101], [63, 101], [70, 101], [71, 101], [72, 101], [74, 101], [76, 101], [78, 101], [82, 101], [83, 101], [87, 101], [89, 101], [90, 101], [93, 101], [94, 101], [96, 101], [98, 101], [100, 101], [31, 102], [45, 102], [49, 102], [51, 102], [56, 102], [57, 102], [58, 102], [60, 102], [62, 102], [63, 102], [66, 102], [69, 102], [70, 102], [73, 102], [74, 102], [76, 102], [78, 102], [79, 102], [83, 102], [84, 102], [85, 102], [87, 102], [89, 102], [94, 102], [95, 102], [96, 102], [98, 102], [100, 102], [101, 102], [32, 103], [51, 103], [60, 103], [61, 103], [62, 103], [69, 103], [81, 103], [87, 103], [88, 103], [89, 103], [92, 103], [96, 103], [98, 103], [99, 103], [100, 103], [102, 103], [33, 104], [55, 104], [82, 104], [93, 104], [34, 105], [52, 105], [53, 105], [56, 105], [83, 105], [91, 105], [94, 105], [102, 105], [104, 105], [35, 106], [52, 106], [57, 106], [68, 106], [70, 106], [84, 106], [95, 106], [102, 106], [104, 106], [105, 106], [36, 107], [48, 107], [52, 107], [53, 107], [58, 107], [68, 107], [69, 107], [70, 107], [85, 107], [91, 107], [92, 107], [96, 107], [102, 107], [103, 107], [105, 107], [106, 107], [37, 108], [52, 108], [59, 108], [68, 108], [86, 108], [91, 108], [97, 108], [104, 108], [105, 108], [106, 108], [107, 108], [38, 109], [51, 109], [52, 109], [53, 109], [60, 109], [69, 109], [87, 109], [91, 109], [92, 109], [98, 109], [102, 109], [103, 109], [105, 109], [106, 109], [107, 109], [108, 109], [39, 110], [52, 110], [61, 110], [68, 110], [69, 110], [81, 110], [88, 110], [92, 110], [99, 110], [103, 110], [105, 110], [106, 110], [107, 110], [108, 110], [109, 110], [40, 111], [53, 111], [62, 111], [69, 111], [89, 111], [92, 111], [100, 111], [103, 111], [107, 111], [109, 111], [110, 111], [41, 112], [55, 112], [56, 112], [62, 112], [63, 112], [66, 112], [67, 112], [70, 112], [71, 112], [72, 112], [74, 112], [76, 112], [78, 112], [82, 112], [83, 112], [87, 112], [89, 112], [90, 112], [93, 112], [94, 112], [96, 112], [100, 112], [101, 112], [104, 112], [105, 112], [42, 113], [45, 113], [49, 113], [53, 113], [56, 113], [57, 113], [58, 113], [60, 113], [62, 113], [63, 113], [66, 113], [67, 113], [68, 113], [69, 113], [70, 113], [73, 113], [74, 113], [76, 113], [78, 113], [79, 113], [83, 113], [84, 113], [85, 113], [87, 113], [89, 113], [94, 113], [95, 113], [96, 113], [100, 113], [102, 113], [105, 113], [106, 113], [107, 113], [109, 113], [111, 113], [112, 113], [43, 114], [53, 114], [54, 114], [56, 114], [58, 114], [59, 114], [60, 114], [62, 114], [65, 114], [66, 114], [68, 114], [69, 114], [74, 114], [75, 114], [76, 114], [78, 114], [80, 114], [83, 114], [86, 114], [87, 114], [89, 114], [91, 114], [94, 114], [96, 114], [97, 114], [98, 114], [100, 114], [105, 114], [107, 114], [108, 114], [109, 114], [111, 114], [112, 114], [113, 114], [0, 115], [2, 115], [3, 115], [6, 115], [7, 115], [64, 115], [71, 115], [73, 115], [74, 115], [77, 115], [78, 115], [1, 116], [3, 116], [7, 116], [8, 116], [10, 116], [12, 116], [18, 116], [34, 116], [36, 116], [72, 116], [74, 116], [78, 116], [79, 116], [81, 116], [83, 116], [89, 116], [115, 116], [3, 117], [5, 117], [6, 117], [7, 117], [10, 117], [16, 117], [17, 117], [18, 117], [21, 117], [32, 117], [45, 117], [66, 117], [73, 117], [74, 117], [76, 117], [77, 117], [78, 117], [79, 117], [81, 117], [84, 117], [85, 117], [87, 117], [88, 117], [89, 117], [92, 117], [102, 117], [103, 117], [115, 117], [116, 117], [11, 118], [12, 118], [13, 118], [16, 118], [17, 118], [18, 118], [55, 118], [56, 118], [57, 118], [61, 118], [62, 118], [63, 118], [64, 118], [66, 118], [67, 118], [70, 118], [82, 118], [83, 118], [84, 118], [87, 118], [88, 118], [89, 118], [115, 118], [116, 118], [117, 118], [3, 119], [7, 119], [12, 119], [14, 119], [16, 119], [18, 119], [19, 119], [21, 119], [25, 119], [29, 119], [34, 119], [36, 119], [56, 119], [58, 119], [61, 119], [62, 119], [63, 119], [64, 119], [66, 119], [67, 119], [70, 119], [74, 119], [78, 119], [83, 119], [85, 119], [87, 119], [89, 119], [90, 119], [92, 119], [96, 119], [100, 119], [105, 119], [107, 119], [115, 119], [116, 119], [117, 119], [118, 119], [22, 120], [24, 120], [25, 120], [28, 120], [29, 120], [55, 120], [57, 120], [61, 120], [62, 120], [67, 120], [93, 120], [95, 120], [96, 120], [99, 120], [100, 120], [115, 120], [118, 120], [119, 120], [10, 121], [21, 121], [23, 121], [25, 121], [28, 121], [29, 121], [32, 121], [34, 121], [36, 121], [39, 121], [49, 121], [56, 121], [57, 121], [58, 121], [61, 121], [62, 121], [79, 121], [81, 121], [83, 121], [84, 121], [85, 121], [88, 121], [89, 121], [92, 121], [94, 121], [95, 121], [96, 121], [99, 121], [100, 121], [102, 121], [103, 121], [105, 121], [106, 121], [107, 121], [110, 121], [116, 121], [118, 121], [119, 121], [120, 121], [3, 122], [7, 122], [18, 122], [25, 122], [29, 122], [30, 122], [31, 122], [32, 122], [34, 122], [36, 122], [57, 122], [61, 122], [62, 122], [63, 122], [70, 122], [74, 122], [78, 122], [83, 122], [87, 122], [89, 122], [96, 122], [100, 122], [101, 122], [102, 122], [103, 122], [107, 122], [115, 122], [116, 122], [117, 122], [118, 122], [119, 122], [120, 122], [121, 122], [33, 123], [34, 123], [35, 123], [36, 123], [39, 123], [55, 123], [56, 123], [57, 123], [58, 123], [61, 123], [63, 123], [64, 123], [67, 123], [70, 123], [104, 123], [105, 123], [106, 123], [107, 123], [110, 123], [115, 123], [116, 123], [118, 123], [119, 123], [120, 123], [121, 123], [122, 123], [3, 124], [7, 124], [18, 124], [25, 124], [29, 124], [34, 124], [36, 124], [40, 124], [41, 124], [42, 124], [57, 124], [62, 124], [63, 124], [70, 124], [74, 124], [78, 124], [83, 124], [89, 124], [96, 124], [100, 124], [107, 124], [111, 124], [112, 124], [113, 124], [118, 124], [119, 124], [122, 124], [123, 124], [0, 125], [11, 125], [19, 125], [33, 125], [41, 125], [64, 125], [71, 125], [82, 125], [90, 125], [104, 125], [112, 125], [115, 125], [118, 125], [119, 125], [123, 125], [1, 126], [8, 126], [10, 126], [12, 126], [19, 126], [21, 126], [34, 126], [41, 126], [42, 126], [72, 126], [79, 126], [83, 126], [90, 126], [105, 126], [112, 126], [113, 126], [116, 126], [118, 126], [119, 126], [123, 126], [125, 126], [2, 127], [8, 127], [13, 127], [31, 127], [35, 127], [42, 127], [44, 127], [45, 127], [46, 127], [47, 127], [52, 127], [53, 127], [64, 127], [65, 127], [66, 127], [70, 127], [73, 127], [79, 127], [84, 127], [102, 127], [106, 127], [113, 127], [115, 127], [116, 127], [117, 127], [118, 127], [122, 127], [123, 127], [124, 127], [125, 127], [126, 127], [3, 128], [8, 128], [10, 128], [14, 128], [19, 128], [20, 128], [21, 128], [31, 128], [36, 128], [41, 128], [42, 128], [43, 128], [45, 128], [46, 128], [48, 128], [52, 128], [53, 128], [64, 128], [65, 128], [66, 128], [70, 128], [74, 128], [79, 128], [81, 128], [85, 128], [90, 128], [91, 128], [92, 128], [102, 128], [107, 128], [112, 128], [113, 128], [114, 128], [115, 128], [116, 128], [117, 128], [119, 128], [122, 128], [123, 128], [124, 128], [125, 128], [126, 128], [127, 128], [4, 129], [15, 129], [20, 129], [37, 129], [43, 129], [44, 129], [47, 129], [52, 129], [53, 129], [65, 129], [75, 129], [86, 129], [91, 129], [108, 129], [114, 129], [125, 129], [127, 129], [128, 129], [5, 130], [10, 130], [16, 130], [19, 130], [20, 130], [21, 130], [38, 130], [41, 130], [42, 130], [43, 130], [45, 130], [47, 130], [52, 130], [53, 130], [66, 130], [76, 130], [79, 130], [87, 130], [90, 130], [91, 130], [92, 130], [102, 130], [109, 130], [112, 130], [113, 130], [114, 130], [117, 130], [118, 130], [119, 130], [125, 130], [126, 130], [127, 130], [128, 130], [129, 130], [6, 131], [10, 131], [17, 131], [21, 131], [39, 131], [44, 131], [45, 131], [46, 131], [47, 131], [48, 131], [52, 131], [64, 131], [65, 131], [66, 131], [77, 131], [81, 131], [88, 131], [92, 131], [110, 131], [115, 131], [116, 131], [117, 131], [118, 131], [119, 131], [123, 131], [125, 131], [126, 131], [127, 131], [128, 131], [129, 131], [130, 131], [7, 132], [10, 132], [18, 132], [19, 132], [20, 132], [21, 132], [40, 132], [41, 132], [42, 132], [43, 132], [45, 132], [47, 132], [53, 132], [66, 132], [78, 132], [79, 132], [89, 132], [90, 132], [91, 132], [92, 132], [111, 132], [112, 132], [113, 132], [114, 132], [117, 132], [118, 132], [119, 132], [124, 132], [127, 132], [128, 132], [130, 132], [131, 132], [9, 133], [10, 133], [20, 133], [21, 133], [34, 133], [36, 133], [37, 133], [38, 133], [39, 133], [43, 133], [46, 133], [47, 133], [48, 133], [52, 133], [53, 133], [54, 133], [79, 133], [80, 133], [81, 133], [83, 133], [84, 133], [85, 133], [86, 133], [87, 133], [88, 133], [91, 133], [92, 133], [105, 133], [106, 133], [107, 133], [108, 133], [109, 133], [110, 133], [113, 133], [114, 133], [126, 133], [127, 133], [128, 133], [129, 133], [130, 133], [131, 133], [19, 134], [30, 134], [31, 134], [33, 134], [34, 134], [35, 134], [36, 134], [38, 134], [41, 134], [42, 134], [63, 134], [70, 134], [82, 134], [83, 134], [84, 134], [85, 134], [87, 134], [90, 134], [101, 134], [102, 134], [104, 134], [105, 134], [106, 134], [107, 134], [109, 134], [112, 134], [113, 134], [118, 134], [119, 134], [122, 134], [123, 134], [124, 134], [125, 134], [126, 134], [127, 134], [128, 134], [130, 134], [0, 135], [2, 135], [3, 135], [6, 135], [7, 135], [11, 135], [13, 135], [17, 135], [18, 135], [19, 135], [21, 135], [33, 135], [35, 135], [36, 135], [39, 135], [40, 135], [41, 135], [42, 135], [64, 135], [71, 135], [73, 135], [74, 135], [78, 135], [82, 135], [84, 135], [89, 135], [90, 135], [112, 135], [113, 135], [115, 135], [118, 135], [119, 135], [123, 135], [124, 135], [125, 135], [127, 135], [128, 135], [131, 135], [132, 135], [1, 136], [3, 136], [7, 136], [8, 136], [10, 136], [12, 136], [13, 136], [14, 136], [17, 136], [18, 136], [19, 136], [21, 136], [34, 136], [35, 136], [36, 136], [39, 136], [40, 136], [41, 136], [42, 136], [72, 136], [74, 136], [78, 136], [79, 136], [83, 136], [84, 136], [85, 136], [89, 136], [90, 136], [112, 136], [113, 136], [116, 136], [118, 136], [119, 136], [123, 136], [124, 136], [126, 136], [127, 136], [128, 136], [131, 136], [132, 136], [135, 136], [4, 137], [6, 137], [7, 137], [15, 137], [17, 137], [18, 137], [20, 137], [21, 137], [36, 137], [37, 137], [39, 137], [40, 137], [43, 137], [44, 137], [47, 137], [52, 137], [53, 137], [65, 137], [73, 137], [74, 137], [75, 137], [77, 137], [78, 137], [84, 137], [86, 137], [88, 137], [89, 137], [91, 137], [113, 137], [114, 137], [127, 137], [128, 137], [129, 137], [131, 137], [132, 137], [135, 137], [7, 138], [9, 138], [10, 138], [17, 138], [18, 138], [20, 138], [21, 138], [29, 138], [34, 138], [36, 138], [37, 138], [39, 138], [40, 138], [43, 138], [46, 138], [47, 138], [48, 138], [52, 138], [53, 138], [54, 138], [61, 138], [62, 138], [74, 138], [78, 138], [79, 138], [80, 138], [81, 138], [83, 138], [84, 138], [85, 138], [86, 138], [88, 138], [89, 138], [91, 138], [92, 138], [96, 138], [100, 138], [106, 138], [107, 138], [110, 138], [111, 138], [113, 138], [114, 138], [127, 138], [128, 138], [131, 138], [132, 138], [133, 138], [136, 138], [137, 138], [21, 139], [22, 139], [24, 139], [25, 139], [28, 139], [29, 139], [33, 139], [35, 139], [36, 139], [39, 139], [40, 139], [41, 139], [42, 139], [55, 139], [57, 139], [61, 139], [62, 139], [67, 139], [82, 139], [84, 139], [89, 139], [90, 139], [93, 139], [95, 139], [96, 139], [100, 139], [104, 139], [106, 139], [112, 139], [113, 139], [118, 139], [119, 139], [120, 139], [123, 139], [124, 139], [135, 139], [10, 140], [21, 140], [23, 140], [25, 140], [28, 140], [29, 140], [34, 140], [35, 140], [36, 140], [39, 140], [40, 140], [41, 140], [42, 140], [43, 140], [49, 140], [52, 140], [53, 140], [56, 140], [57, 140], [58, 140], [61, 140], [62, 140], [79, 140], [83, 140], [84, 140], [85, 140], [88, 140], [89, 140], [90, 140], [91, 140], [92, 140], [94, 140], [95, 140], [96, 140], [100, 140], [105, 140], [106, 140], [107, 140], [110, 140], [111, 140], [112, 140], [113, 140], [114, 140], [118, 140], [119, 140], [121, 140], [123, 140], [124, 140], [136, 140], [138, 140], [139, 140], [20, 141], [21, 141], [25, 141], [26, 141], [28, 141], [29, 141], [36, 141], [37, 141], [39, 141], [40, 141], [43, 141], [47, 141], [50, 141], [52, 141], [53, 141], [57, 141], [59, 141], [61, 141], [62, 141], [68, 141], [84, 141], [85, 141], [86, 141], [88, 141], [89, 141], [91, 141], [92, 141], [95, 141], [96, 141], [97, 141], [99, 141], [100, 141], [106, 141], [107, 141], [108, 141], [110, 141], [111, 141], [113, 141], [114, 141], [137, 141], [138, 141], [139, 141], [140, 141], [3, 142], [4, 142], [5, 142], [6, 142], [7, 142], [9, 142], [10, 142], [20, 142], [21, 142], [25, 142], [26, 142], [28, 142], [29, 142], [36, 142], [37, 142], [38, 142], [39, 142], [40, 142], [43, 142], [44, 142], [45, 142], [46, 142], [48, 142], [50, 142], [52, 142], [53, 142], [65, 142], [66, 142], [68, 142], [74, 142], [75, 142], [76, 142], [78, 142], [80, 142], [91, 142], [96, 142], [97, 142], [100, 142], [114, 142], [128, 142], [129, 142], [130, 142], [131, 142], [132, 142], [133, 142], [137, 142], [138, 142], [141, 142], [19, 143], [20, 143], [21, 143], [22, 143], [23, 143], [25, 143], [26, 143], [28, 143], [29, 143], [30, 143], [32, 143], [33, 143], [34, 143], [36, 143], [37, 143], [39, 143], [40, 143], [41, 143], [43, 143], [55, 143], [56, 143], [58, 143], [59, 143], [61, 143], [62, 143], [63, 143], [67, 143], [68, 143], [70, 143], [90, 143], [91, 143], [93, 143], [94, 143], [96, 143], [97, 143], [100, 143], [101, 143], [112, 143], [114, 143], [119, 143], [120, 143], [121, 143], [122, 143], [123, 143], [124, 143], [139, 143], [140, 143], [141, 143], [0, 144], [4, 144], [5, 144], [6, 144], [7, 144], [22, 144], [26, 144], [27, 144], [28, 144], [29, 144], [30, 144], [32, 144], [33, 144], [37, 144], [38, 144], [39, 144], [40, 144], [41, 144], [43, 144], [2, 145], [6, 145], [7, 145], [24, 145], [28, 145], [29, 145], [31, 145], [32, 145], [35, 145], [38, 145], [39, 145], [40, 145], [42, 145], [44, 145], [45, 145], [50, 145], [51, 145], [52, 145], [53, 145], [73, 145], [78, 145], [95, 145], [100, 145], [102, 145], [113, 145], [127, 145], [131, 145], [144, 145], [3, 146], [5, 146], [6, 146], [7, 146], [21, 146], [25, 146], [26, 146], [27, 146], [28, 146], [29, 146], [30, 146], [31, 146], [32, 146], [36, 146], [37, 146], [38, 146], [39, 146], [40, 146], [41, 146], [42, 146], [43, 146], [45, 146], [50, 146], [51, 146], [52, 146], [53, 146], [66, 146], [68, 146], [69, 146], [70, 146], [74, 146], [76, 146], [78, 146], [91, 146], [96, 146], [97, 146], [98, 146], [99, 146], [100, 146], [101, 146], [102, 146], [103, 146], [112, 146], [113, 146], [114, 146], [117, 146], [122, 146], [142, 146], [144, 146], [145, 146], [11, 147], [15, 147], [16, 147], [17, 147], [18, 147], [32, 147], [33, 147], [37, 147], [38, 147], [39, 147], [40, 147], [41, 147], [43, 147], [55, 147], [59, 147], [60, 147], [61, 147], [62, 147], [63, 147], [82, 147], [86, 147], [87, 147], [89, 147], [112, 147], [114, 147], [118, 147], [123, 147], [144, 147], [13, 148], [16, 148], [17, 148], [18, 148], [31, 148], [32, 148], [35, 148], [38, 148], [39, 148], [40, 148], [42, 148], [45, 148], [47, 148], [50, 148], [51, 148], [52, 148], [53, 148], [57, 148], [59, 148], [60, 148], [61, 148], [62, 148], [63, 148], [66, 148], [68, 148], [69, 148], [70, 148], [84, 148], [86, 148], [87, 148], [88, 148], [89, 148], [95, 148], [97, 148], [98, 148], [99, 148], [100, 148], [102, 148], [103, 148], [106, 148], [108, 148], [109, 148], [110, 148], [113, 148], [114, 148], [117, 148], [118, 148], [122, 148], [123, 148], [127, 148], [130, 148], [131, 148], [134, 148], [145, 148], [147, 148], [5, 149], [7, 149], [16, 149], [18, 149], [19, 149], [20, 149], [21, 149], [26, 149], [27, 149], [28, 149], [29, 149], [30, 149], [32, 149], [33, 149], [36, 149], [37, 149], [38, 149], [39, 149], [40, 149], [41, 149], [43, 149], [59, 149], [60, 149], [61, 149], [62, 149], [63, 149], [66, 149], [68, 149], [69, 149], [70, 149], [76, 149], [78, 149], [87, 149], [89, 149], [90, 149], [91, 149], [96, 149], [97, 149], [98, 149], [100, 149], [101, 149], [108, 149], [109, 149], [112, 149], [114, 149], [130, 149], [134, 149], [143, 149], [144, 149], [146, 149], [147, 149], [6, 150], [7, 150], [17, 150], [18, 150], [21, 150], [28, 150], [29, 150], [32, 150], [33, 150], [35, 150], [36, 150], [37, 150], [38, 150], [39, 150], [40, 150], [41, 150], [42, 150], [43, 150], [52, 150], [53, 150], [61, 150], [62, 150], [64, 150], [65, 150], [66, 150], [67, 150], [68, 150], [69, 150], [70, 150], [78, 150], [86, 150], [87, 150], [89, 150], [91, 150], [95, 150], [96, 150], [100, 150], [102, 150], [110, 150], [112, 150], [113, 150], [114, 150], [118, 150], [123, 150], [127, 150], [131, 150], [144, 150], [145, 150], [146, 150], [147, 150], [148, 150], [149, 150], [71, 151], [73, 151], [74, 151], [75, 151], [76, 151], [78, 151], [82, 151], [84, 151], [86, 151], [87, 151], [89, 151], [90, 151], [91, 151], [93, 151], [95, 151], [96, 151], [97, 151], [98, 151], [100, 151], [101, 151], [102, 151], [112, 151], [113, 151], [114, 151], [144, 151], [145, 151], [146, 151], [147, 151], [148, 151], [149, 151], [150, 151], [6, 152], [7, 152], [17, 152], [18, 152], [28, 152], [29, 152], [32, 152], [39, 152], [61, 152], [62, 152], [73, 152], [74, 152], [77, 152], [78, 152], [84, 152], [87, 152], [88, 152], [89, 152], [95, 152], [96, 152], [99, 152], [100, 152], [102, 152], [103, 152], [106, 152], [110, 152], [115, 152], [117, 152], [118, 152], [120, 152], [122, 152], [123, 152], [127, 152], [131, 152], [144, 152], [145, 152], [146, 152], [147, 152], [148, 152], [150, 152], [151, 152], [33, 153], [35, 153], [37, 153], [38, 153], [39, 153], [41, 153], [42, 153], [43, 153], [52, 153], [53, 153], [82, 153], [84, 153], [86, 153], [87, 153], [88, 153], [90, 153], [91, 153], [102, 153], [104, 153], [106, 153], [108, 153], [109, 153], [110, 153], [112, 153], [113, 153], [114, 153], [118, 153], [123, 153], [125, 153], [127, 153], [129, 153], [130, 153], [131, 153], [134, 153], [144, 153], [145, 153], [147, 153], [148, 153], [149, 153], [150, 153], [151, 153], [152, 153], [6, 154], [7, 154], [17, 154], [18, 154], [28, 154], [29, 154], [33, 154], [35, 154], [37, 154], [39, 154], [40, 154], [41, 154], [42, 154], [43, 154], [52, 154], [53, 154], [61, 154], [62, 154], [78, 154], [86, 154], [89, 154], [95, 154], [100, 154], [110, 154], [111, 154], [112, 154], [113, 154], [114, 154], [118, 154], [123, 154], [124, 154], [127, 154], [131, 154], [132, 154], [135, 154], [137, 154], [139, 154], [141, 154], [150, 154], [151, 154], [152, 154], [153, 154], [64, 155], [65, 155], [66, 155], [67, 155], [68, 155], [70, 155], [71, 155], [73, 155], [74, 155], [75, 155], [76, 155], [78, 155], [82, 155], [84, 155], [85, 155], [86, 155], [87, 155], [89, 155], [90, 155], [91, 155], [93, 155], [95, 155], [96, 155], [97, 155], [100, 155], [101, 155], [102, 155], [112, 155], [113, 155], [114, 155], [115, 155], [117, 155], [118, 155], [119, 155], [120, 155], [122, 155], [124, 155], [125, 155], [127, 155], [128, 155], [129, 155], [130, 155], [132, 155], [134, 155], [135, 155], [137, 155], [139, 155], [141, 155], [150, 155], [151, 155], [152, 155], [153, 155], [154, 155], [0, 156], [4, 156], [5, 156], [6, 156], [7, 156], [71, 156], [75, 156], [76, 156], [77, 156], [78, 156], [115, 156], [117, 156], [125, 156], [129, 156], [130, 156], [131, 156], [132, 156], [135, 156], [137, 156], [144, 156], [151, 156], [152, 156], [153, 156], [1, 157], [5, 157], [7, 157], [9, 157], [10, 157], [23, 157], [29, 157], [34, 157], [38, 157], [72, 157], [76, 157], [78, 157], [80, 157], [81, 157], [94, 157], [100, 157], [116, 157], [117, 157], [121, 157], [126, 157], [130, 157], [131, 157], [132, 157], [133, 157], [136, 157], [138, 157], [140, 157], [156, 157], [3, 158], [5, 158], [6, 158], [7, 158], [10, 158], [21, 158], [25, 158], [28, 158], [29, 158], [32, 158], [66, 158], [68, 158], [74, 158], [75, 158], [76, 158], [77, 158], [78, 158], [80, 158], [81, 158], [91, 158], [92, 158], [96, 158], [97, 158], [98, 158], [99, 158], [100, 158], [103, 158], [115, 158], [116, 158], [117, 158], [119, 158], [120, 158], [121, 158], [122, 158], [128, 158], [129, 158], [130, 158], [131, 158], [132, 158], [133, 158], [135, 158], [136, 158], [137, 158], [138, 158], [139, 158], [140, 158], [141, 158], [142, 158], [146, 158], [151, 158], [152, 158], [156, 158], [157, 158], [11, 159], [15, 159], [16, 159], [17, 159], [18, 159], [55, 159], [59, 159], [61, 159], [62, 159], [63, 159], [82, 159], [86, 159], [87, 159], [88, 159], [89, 159], [117, 159], [118, 159], [125, 159], [129, 159], [130, 159], [131, 159], [132, 159], [134, 159], [135, 159], [137, 159], [139, 159], [141, 159], [147, 159], [151, 159], [152, 159], [153, 159], [156, 159], [10, 160], [12, 160], [16, 160], [17, 160], [18, 160], [21, 160], [32, 160], [34, 160], [38, 160], [39, 160], [54, 160], [56, 160], [59, 160], [60, 160], [61, 160], [62, 160], [63, 160], [80, 160], [81, 160], [83, 160], [86, 160], [87, 160], [88, 160], [89, 160], [91, 160], [92, 160], [94, 160], [97, 160], [98, 160], [99, 160], [100, 160], [103, 160], [105, 160], [108, 160], [109, 160], [110, 160], [116, 160], [117, 160], [118, 160], [119, 160], [121, 160], [122, 160], [123, 160], [126, 160], [130, 160], [131, 160], [132, 160], [133, 160], [134, 160], [136, 160], [138, 160], [140, 160], [141, 160], [157, 160], [159, 160], [5, 161], [7, 161], [16, 161], [18, 161], [19, 161], [20, 161], [21, 161], [29, 161], [34, 161], [38, 161], [59, 161], [61, 161], [62, 161], [63, 161], [66, 161], [68, 161], [76, 161], [78, 161], [87, 161], [89, 161], [90, 161], [91, 161], [92, 161], [94, 161], [96, 161], [100, 161], [109, 161], [117, 161], [119, 161], [121, 161], [125, 161], [126, 161], [128, 161], [129, 161], [130, 161], [131, 161], [132, 161], [133, 161], [134, 161], [135, 161], [136, 161], [137, 161], [138, 161], [139, 161], [140, 161], [141, 161], [143, 161], [149, 161], [151, 161], [153, 161], [156, 161], [157, 161], [158, 161], [159, 161], [160, 161], [22, 162], [23, 162], [25, 162], [26, 162], [28, 162], [29, 162], [55, 162], [56, 162], [59, 162], [61, 162], [62, 162], [67, 162], [68, 162], [93, 162], [94, 162], [96, 162], [97, 162], [99, 162], [100, 162], [120, 162], [121, 162], [139, 162], [140, 162], [141, 162], [143, 162], [144, 162], [146, 162], [147, 162], [149, 162], [150, 162], [151, 162], [152, 162], [154, 162], [156, 162], [157, 162], [158, 162], [159, 162], [160, 162], [161, 162], [5, 163], [7, 163], [16, 163], [18, 163], [23, 163], [25, 163], [27, 163], [29, 163], [30, 163], [32, 163], [34, 163], [38, 163], [56, 163], [60, 163], [61, 163], [62, 163], [63, 163], [76, 163], [78, 163], [87, 163], [89, 163], [94, 163], [96, 163], [98, 163], [100, 163], [101, 163], [103, 163], [105, 163], [109, 163], [117, 163], [121, 163], [122, 163], [130, 163], [134, 163], [143, 163], [144, 163], [146, 163], [147, 163], [149, 163], [151, 163], [152, 163], [153, 163], [156, 163], [157, 163], [158, 163], [159, 163], [160, 163], [161, 163], [162, 163], [33, 164], [34, 164], [37, 164], [38, 164], [39, 164], [55, 164], [56, 164], [59, 164], [60, 164], [61, 164], [63, 164], [67, 164], [68, 164], [104, 164], [105, 164], [108, 164], [109, 164], [110, 164], [121, 164], [123, 164], [125, 164], [126, 164], [129, 164], [130, 164], [131, 164], [133, 164], [134, 164], [135, 164], [136, 164], [137, 164], [138, 164], [139, 164], [140, 164], [141, 164], [143, 164], [144, 164], [147, 164], [149, 164], [150, 164], [152, 164], [153, 164], [154, 164], [156, 164], [157, 164], [159, 164], [160, 164], [161, 164], [162, 164], [163, 164], [5, 165], [7, 165], [16, 165], [18, 165], [29, 165], [34, 165], [38, 165], [40, 165], [41, 165], [43, 165], [59, 165], [62, 165], [63, 165], [76, 165], [78, 165], [87, 165], [89, 165], [94, 165], [100, 165], [109, 165], [111, 165], [112, 165], [114, 165], [124, 165], [125, 165], [126, 165], [129, 165], [130, 165], [132, 165], [133, 165], [134, 165], [135, 165], [136, 165], [137, 165], [138, 165], [139, 165], [140, 165], [141, 165], [143, 165], [149, 165], [151, 165], [153, 165], [154, 165], [161, 165], [162, 165], [163, 165], [164, 165], [6, 166], [7, 166], [17, 166], [18, 166], [21, 166], [28, 166], [29, 166], [39, 166], [61, 166], [62, 166], [64, 166], [65, 166], [66, 166], [67, 166], [68, 166], [75, 166], [76, 166], [77, 166], [78, 166], [86, 166], [87, 166], [88, 166], [89, 166], [91, 166], [92, 166], [96, 166], [97, 166], [99, 166], [100, 166], [108, 166], [110, 166], [115, 166], [117, 166], [118, 166], [119, 166], [120, 166], [123, 166], [125, 166], [128, 166], [129, 166], [130, 166], [131, 166], [132, 166], [135, 166], [137, 166], [139, 166], [141, 166], [150, 166], [151, 166], [152, 166], [153, 166], [154, 166], [155, 166], [156, 166], [158, 166], [159, 166], [161, 166], [162, 166], [164, 166], [0, 167], [22, 167], [30, 167], [33, 167], [41, 167], [71, 167], [93, 167], [101, 167], [104, 167], [112, 167], [115, 167], [120, 167], [122, 167], [123, 167], [124, 167], [125, 167], [134, 167], [135, 167], [139, 167], [144, 167], [151, 167], [152, 167], [153, 167], [156, 167], [162, 167], [163, 167], [164, 167], [1, 168], [9, 168], [10, 168], [23, 168], [30, 168], [32, 168], [34, 168], [41, 168], [43, 168], [72, 168], [80, 168], [94, 168], [101, 168], [105, 168], [112, 168], [114, 168], [116, 168], [121, 168], [122, 168], [123, 168], [124, 168], [126, 168], [133, 168], [134, 168], [136, 168], [138, 168], [140, 168], [157, 168], [162, 168], [163, 168], [164, 168], [167, 168], [2, 169], [24, 169], [31, 169], [35, 169], [42, 169], [44, 169], [45, 169], [50, 169], [52, 169], [53, 169], [73, 169], [95, 169], [102, 169], [106, 169], [113, 169], [115, 169], [117, 169], [120, 169], [122, 169], [123, 169], [124, 169], [127, 169], [134, 169], [135, 169], [137, 169], [139, 169], [141, 169], [145, 169], [151, 169], [152, 169], [153, 169], [167, 169], [3, 170], [10, 170], [25, 170], [30, 170], [31, 170], [32, 170], [36, 170], [41, 170], [42, 170], [43, 170], [45, 170], [50, 170], [52, 170], [53, 170], [68, 170], [70, 170], [74, 170], [80, 170], [91, 170], [96, 170], [101, 170], [102, 170], [103, 170], [107, 170], [112, 170], [113, 170], [114, 170], [115, 170], [116, 170], [117, 170], [119, 170], [120, 170], [121, 170], [122, 170], [123, 170], [124, 170], [128, 170], [133, 170], [134, 170], [135, 170], [136, 170], [137, 170], [138, 170], [139, 170], [140, 170], [141, 170], [142, 170], [146, 170], [151, 170], [152, 170], [158, 170], [162, 170], [163, 170], [167, 170], [168, 170], [169, 170], [4, 171], [9, 171], [20, 171], [26, 171], [37, 171], [43, 171], [44, 171], [46, 171], [50, 171], [52, 171], [53, 171], [65, 171], [68, 171], [75, 171], [80, 171], [91, 171], [97, 171], [108, 171], [114, 171], [129, 171], [133, 171], [137, 171], [138, 171], [141, 171], [142, 171], [144, 171], [145, 171], [146, 171], [149, 171], [150, 171], [151, 171], [153, 171], [154, 171], [156, 171], [157, 171], [158, 171], [161, 171], [162, 171], [164, 171], [165, 171], [166, 171], [167, 171], [168, 171], [169, 171], [170, 171], [5, 172], [9, 172], [10, 172], [20, 172], [27, 172], [30, 172], [31, 172], [32, 172], [38, 172], [41, 172], [42, 172], [43, 172], [45, 172], [46, 172], [51, 172], [52, 172], [53, 172], [76, 172], [80, 172], [81, 172], [91, 172], [98, 172], [101, 172], [102, 172], [103, 172], [109, 172], [112, 172], [113, 172], [114, 172], [117, 172], [122, 172], [130, 172], [133, 172], [134, 172], [142, 172], [144, 172], [145, 172], [146, 172], [149, 172], [151, 172], [152, 172], [153, 172], [156, 172], [157, 172], [158, 172], [161, 172], [163, 172], [164, 172], [165, 172], [167, 172], [168, 172], [169, 172], [170, 172], [171, 172], [6, 173], [10, 173], [28, 173], [32, 173], [39, 173], [44, 173], [45, 173], [46, 173], [50, 173], [51, 173], [52, 173], [65, 173], [68, 173], [77, 173], [81, 173], [99, 173], [103, 173], [110, 173], [115, 173], [116, 173], [117, 173], [120, 173], [121, 173], [122, 173], [123, 173], [131, 173], [133, 173], [135, 173], [136, 173], [137, 173], [138, 173], [139, 173], [140, 173], [141, 173], [142, 173], [144, 173], [145, 173], [146, 173], [150, 173], [152, 173], [153, 173], [154, 173], [156, 173], [157, 173], [158, 173], [162, 173], [163, 173], [164, 173], [166, 173], [167, 173], [168, 173], [169, 173], [170, 173], [171, 173], [172, 173], [7, 174], [10, 174], [29, 174], [30, 174], [31, 174], [32, 174], [40, 174], [41, 174], [42, 174], [43, 174], [45, 174], [50, 174], [53, 174], [78, 174], [80, 174], [100, 174], [101, 174], [102, 174], [103, 174], [111, 174], [112, 174], [113, 174], [114, 174], [115, 174], [116, 174], [117, 174], [120, 174], [121, 174], [122, 174], [124, 174], [132, 174], [135, 174], [136, 174], [137, 174], [138, 174], [139, 174], [140, 174], [141, 174], [142, 174], [146, 174], [151, 174], [152, 174], [154, 174], [158, 174], [162, 174], [163, 174], [165, 174], [170, 174], [171, 174], [172, 174], [173, 174], [8, 175], [10, 175], [31, 175], [32, 175], [34, 175], [35, 175], [36, 175], [38, 175], [39, 175], [42, 175], [45, 175], [46, 175], [49, 175], [50, 175], [51, 175], [52, 175], [53, 175], [79, 175], [80, 175], [81, 175], [94, 175], [95, 175], [96, 175], [97, 175], [98, 175], [99, 175], [102, 175], [103, 175], [105, 175], [106, 175], [107, 175], [108, 175], [109, 175], [110, 175], [113, 175], [114, 175], [116, 175], [117, 175], [121, 175], [122, 175], [123, 175], [124, 175], [126, 175], [127, 175], [128, 175], [130, 175], [131, 175], [133, 175], [134, 175], [136, 175], [138, 175], [140, 175], [141, 175], [168, 175], [169, 175], [170, 175], [171, 175], [172, 175], [173, 175], [19, 176], [20, 176], [30, 176], [33, 176], [34, 176], [36, 176], [37, 176], [38, 176], [41, 176], [43, 176], [68, 176], [70, 176], [90, 176], [91, 176], [93, 176], [94, 176], [96, 176], [97, 176], [98, 176], [101, 176], [104, 176], [105, 176], [107, 176], [108, 176], [109, 176], [112, 176], [114, 176], [119, 176], [120, 176], [121, 176], [122, 176], [123, 176], [124, 176], [125, 176], [126, 176], [128, 176], [129, 176], [130, 176], [133, 176], [134, 176], [135, 176], [136, 176], [137, 176], [138, 176], [139, 176], [140, 176], [141, 176], [143, 176], [149, 176], [151, 176], [153, 176], [161, 176], [162, 176], [163, 176], [164, 176], [165, 176], [167, 176], [168, 176], [170, 176], [171, 176], [172, 176], [33, 177], [35, 177], [36, 177], [37, 177], [39, 177], [41, 177], [42, 177], [43, 177], [52, 177], [53, 177], [64, 177], [65, 177], [67, 177], [68, 177], [70, 177], [91, 177], [93, 177], [95, 177], [96, 177], [97, 177], [99, 177], [101, 177], [102, 177], [104, 177], [106, 177], [107, 177], [108, 177], [110, 177], [112, 177], [113, 177], [114, 177], [115, 177], [119, 177], [120, 177], [122, 177], [123, 177], [124, 177], [125, 177], [127, 177], [128, 177], [129, 177], [131, 177], [134, 177], [135, 177], [137, 177], [139, 177], [141, 177], [150, 177], [151, 177], [152, 177], [153, 177], [154, 177], [155, 177], [162, 177], [164, 177], [166, 177], [167, 177], [169, 177], [170, 177], [171, 177], [173, 177], [176, 177], [0, 178], [4, 178], [5, 178], [6, 178], [7, 178], [22, 178], [26, 178], [28, 178], [29, 178], [30, 178], [32, 178], [33, 178], [37, 178], [38, 178], [39, 178], [40, 178], [41, 178], [43, 178], [71, 178], [75, 178], [76, 178], [78, 178], [93, 178], [97, 178], [100, 178], [101, 178], [112, 178], [114, 178], [115, 178], [117, 178], [120, 178], [122, 178], [123, 178], [124, 178], [125, 178], [129, 178], [130, 178], [131, 178], [132, 178], [134, 178], [135, 178], [137, 178], [139, 178], [141, 178], [144, 178], [151, 178], [152, 178], [153, 178], [154, 178], [156, 178], [162, 178], [163, 178], [164, 178], [165, 178], [167, 178], [171, 178], [172, 178], [173, 178], [174, 178], [1, 179], [5, 179], [7, 179], [9, 179], [10, 179], [23, 179], [26, 179], [27, 179], [28, 179], [29, 179], [30, 179], [32, 179], [34, 179], [37, 179], [38, 179], [39, 179], [40, 179], [41, 179], [43, 179], [72, 179], [76, 179], [78, 179], [80, 179], [94, 179], [97, 179], [98, 179], [100, 179], [101, 179], [112, 179], [114, 179], [116, 179], [117, 179], [121, 179], [122, 179], [123, 179], [124, 179], [126, 179], [130, 179], [131, 179], [132, 179], [133, 179], [134, 179], [136, 179], [138, 179], [140, 179], [141, 179], [157, 179], [162, 179], [163, 179], [164, 179], [165, 179], [168, 179], [171, 179], [172, 179], [173, 179], [174, 179], [178, 179], [2, 180], [6, 180], [7, 180], [24, 180], [28, 180], [29, 180], [31, 180], [32, 180], [35, 180], [38, 180], [39, 180], [40, 180], [42, 180], [44, 180], [45, 180], [50, 180], [52, 180], [53, 180], [73, 180], [75, 180], [76, 180], [77, 180], [78, 180], [95, 180], [97, 180], [99, 180], [100, 180], [102, 180], [113, 180], [114, 180], [115, 180], [117, 180], [120, 180], [122, 180], [123, 180], [124, 180], [127, 180], [129, 180], [130, 180], [131, 180], [132, 180], [134, 180], [135, 180], [137, 180], [139, 180], [141, 180], [145, 180], [151, 180], [152, 180], [153, 180], [154, 180], [169, 180], [171, 180], [172, 180], [173, 180], [174, 180], [178, 180], [7, 181], [8, 181], [10, 181], [18, 181], [28, 181], [29, 181], [31, 181], [32, 181], [34, 181], [35, 181], [38, 181], [39, 181], [40, 181], [42, 181], [45, 181], [46, 181], [49, 181], [50, 181], [51, 181], [52, 181], [53, 181], [61, 181], [62, 181], [76, 181], [78, 181], [79, 181], [80, 181], [81, 181], [87, 181], [89, 181], [94, 181], [95, 181], [97, 181], [98, 181], [99, 181], [100, 181], [102, 181], [103, 181], [108, 181], [109, 181], [110, 181], [111, 181], [113, 181], [114, 181], [116, 181], [117, 181], [121, 181], [122, 181], [123, 181], [124, 181], [126, 181], [127, 181], [130, 181], [131, 181], [132, 181], [133, 181], [134, 181], [136, 181], [138, 181], [140, 181], [141, 181], [171, 181], [172, 181], [173, 181], [174, 181], [175, 181], [179, 181], [180, 181], [11, 182], [15, 182], [16, 182], [17, 182], [18, 182], [32, 182], [33, 182], [37, 182], [38, 182], [39, 182], [40, 182], [41, 182], [43, 182], [55, 182], [59, 182], [61, 182], [62, 182], [63, 182], [82, 182], [86, 182], [87, 182], [89, 182], [93, 182], [97, 182], [100, 182], [101, 182], [104, 182], [108, 182], [112, 182], [114, 182], [117, 182], [118, 182], [120, 182], [122, 182], [123, 182], [124, 182], [125, 182], [129, 182], [130, 182], [131, 182], [132, 182], [134, 182], [135, 182], [137, 182], [139, 182], [141, 182], [147, 182], [151, 182], [152, 182], [153, 182], [154, 182], [159, 182], [162, 182], [163, 182], [164, 182], [165, 182], [178, 182], [10, 183], [12, 183], [16, 183], [17, 183], [18, 183], [32, 183], [34, 183], [37, 183], [38, 183], [39, 183], [40, 183], [41, 183], [42, 183], [43, 183], [52, 183], [53, 183], [54, 183], [56, 183], [59, 183], [60, 183], [61, 183], [62, 183], [63, 183], [80, 183], [83, 183], [86, 183], [87, 183], [89, 183], [94, 183], [97, 183], [98, 183], [99, 183], [100, 183], [101, 183], [102, 183], [103, 183], [105, 183], [108, 183], [109, 183], [110, 183], [111, 183], [112, 183], [113, 183], [114, 183], [116, 183], [117, 183], [118, 183], [121, 183], [122, 183], [123, 183], [124, 183], [126, 183], [130, 183], [131, 183], [132, 183], [133, 183], [134, 183], [136, 183], [138, 183], [140, 183], [141, 183], [160, 183], [162, 183], [163, 183], [164, 183], [165, 183], [179, 183], [181, 183], [182, 183], [13, 184], [16, 184], [17, 184], [18, 184], [31, 184], [32, 184], [35, 184], [38, 184], [39, 184], [40, 184], [42, 184], [45, 184], [47, 184], [50, 184], [52, 184], [53, 184], [57, 184], [59, 184], [61, 184], [62, 184], [63, 184], [84, 184], [86, 184], [87, 184], [88, 184], [89, 184], [95, 184], [97, 184], [98, 184], [99, 184], [100, 184], [102, 184], [103, 184], [106, 184], [108, 184], [109, 184], [110, 184], [111, 184], [113, 184], [114, 184], [117, 184], [118, 184], [120, 184], [121, 184], [122, 184], [123, 184], [124, 184], [127, 184], [129, 184], [130, 184], [131, 184], [132, 184], [133, 184], [134, 184], [135, 184], [136, 184], [137, 184], [138, 184], [139, 184], [140, 184], [141, 184], [148, 184], [151, 184], [152, 184], [153, 184], [154, 184], [180, 184], [181, 184], [182, 184], [183, 184]];





#edgelist5 = 





len(edgelist4)





edgelist_copy = edgelist4.copy();
interesting_edges = edgelist_copy.copy();





i = 5;
print(is_addition(imsetlist[edgelist_copy[i][0]], imsetlist[edgelist_copy[i][1]]));





edgelist_copy[5]





for i in range(len(edgelist_copy)):
    if (is_addition(imsetlist[edgelist_copy[i][0]], imsetlist[edgelist_copy[i][1]])):
        interesting_edges.remove(edgelist_copy[i]);
    elif (is_flip(imsetlist[edgelist_copy[i][0]], imsetlist[edgelist_copy[i][1]])):
        interesting_edges.remove(edgelist_copy[i]);
    elif (is_ges(imsetlist[edgelist_copy[i][0]], imsetlist[edgelist_copy[i][1]])):
        interesting_edges.remove(edgelist_copy[i]);
    elif (is_budding(imsetlist[edgelist_copy[i][0]], imsetlist[edgelist_copy[i][1]])):
        interesting_edges.remove(edgelist_copy[i]);
print(interesting_edges);





i=83;
print(imset2vec(imsetlist[interesting_edges[i][0]]), imset2vec(imsetlist[interesting_edges[i][1]])[1]);





print(len(interesting_edges), len(edgelist_copy));





print(imset2dag(imsetlist[100]))





interesting_edges_copy = interesting_edges.copy();
for i in range(len(interesting_edges)):
    if (imset_cutof(imsetlist[interesting_edges[i][0]], size = 2) != imset_cutof(imsetlist[interesting_edges[i][1]], size = 2)):
        interesting_edges_copy.remove(interesting_edges[i]);
print(len(interesting_edges_copy))





i=33;
print(imset2vec(imsetlist[interesting_edges_copy[i][0]]), imset2vec(imsetlist[interesting_edges_copy[i][1]])[1]);





count = 0;
for i in range(len(interesting_edges_copy)):
    A, B = imsetdif(imsetlist[interesting_edges_copy[i][0]], imsetlist[interesting_edges_copy[i][1]]);
    if (len(A) != 1 or len(B) != 1):
        print(imset2vec(imsetlist[interesting_edges_copy[i][0]])[1], imset2vec(imsetlist[interesting_edges_copy[i][1]])[1], len(A), len(B));
        count+=1;
print(count)
        





count = 0;
for i in range(len(interesting_edges_copy)):
    A, B = imsetdif(imsetlist[interesting_edges_copy[i][0]], imsetlist[interesting_edges_copy[i][1]]);
    if (len(A) == 1 and len(B) == 1):
        print(imset2vec(imsetlist[interesting_edges_copy[i][0]])[1], imset2vec(imsetlist[interesting_edges_copy[i][1]])[1], len(A), len(B));
        count+=1;
print(count)





count = 0;
for i in range(len(interesting_edges_copy)):
    A, B = imsetdif(imsetlist[interesting_edges_copy[i][0]], imsetlist[interesting_edges_copy[i][1]]);
    if ([len(A), len(B)] == [0,3] or [len(A), len(B)] == [3,0]):
        print(imset2dag(imsetlist[interesting_edges_copy[i][0]]),"\n", imset2dag(imsetlist[interesting_edges_copy[i][1]]), "\n", len(A), len(B), "\n\n");
        count+=1;
print(count)





i = 50
d = imset2dag(imsetlist[interesting_edges_copy[i][0]]);
h = imset2dag(imsetlist[interesting_edges_copy[i][1]]);
print(imsetlist[interesting_edges_copy[i][0]]);
print(d);
print(imsetlist[interesting_edges_copy[i][1]])
print(h);

