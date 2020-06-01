
# Import modules.

# For graph handling.
import igraph;

# For testing the algorithms on random graphs.
import random;

# For Warnings 
import warnings;



# Produce all sets of size 2 or greater in a list.
# INPUT: a set
# RETURNS: all sets of cardiality greater or equal to 2 as a list.
def coordinatelist(someset):
    try:
        # duck typing
        someset.isdisjoint
    except AttributeError:
        raise TypeError(
            f"{powerset.__name__} accepts only a set-like object as parameter"
        ) from None
    
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



# Returns whether the pair (graph1, graph2) is a flip.
# verbose option for debugging
# Can take both graphs and imsets as options

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
    
    ne_set = set();
    intersection_list = list(intersection_set);
    for i in union_set1.union(union_set2).difference(intersection_set):
        if (imset_value(imset1, set({i, intersection_list[0]})) == 1 and imset_value(imset1, set({i, intersection_list[1]})) == 1):
            ne_set.add(i);
            p
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



def try_acyclic_orientation(graph):
    imsetlist = [];
    if(graph.is_directed()):
        print("The input to 'try_acyclic_orientation' was directed.\nWe will only direct the mutual edges.\nWe suggest not doing this.");
        dir_graph = graph.copy()
    else:
        dir_graph = graph.copy();
        dir_graph.to_directed();
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






