# coding: utf-8


import igraph;




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
    for element in someset:
        temp_list = ret.copy();
        #print("temp_list = ", temp_list, len(temp_list));
        for counter in range(len(temp_list)):
            #print(type(temp_list[counter]))
            ret.append(temp_list[counter].union({element}));
    
    temp_list = ret.copy();
    for i in temp_list:
        if (len(i) < 2):
            ret.remove(i);
     
    return ret


# Get the parentset and the childred as sets instead of lists

def parents(graph, node):
    return set(graph.predecessors(node));

def childs(graph, node):
    return set(graph.successors(node));


# Returns the value of the characteristic imset for the graph in the given coordinate.
# INPUT: a DAG and a set
# OUTPUT: a 0/1 value
def imsetcoordinate(graph, coordinate_set):
    copy_set = coordinate_set.copy();
    node = 0;
    temp_set = childs(graph, node).intersection(coordinate_set);
    while (len(temp_set) > 0):
        node = list(temp_set)[0];
        temp_set = childs(graph, node).intersection(coordinate_set);
    copy_set.discard(node);
    if (parents(graph, node).issuperset(copy_set)):
        return 1;
    return 0;

def imset(graph):
    ret = [];
    vertex_set = set(range(graph.vcount()))
    coordinate_list = coordinatelist(vertex_set);
    print()
    for i in range(len(coordinate_list)):
        ret.append([coordinate_list[i], imsetcoordinate(graph, coordinate_list[i])])
    return ret


def imset2vec(imset_input):
    ret = [];
    for i in range(len(imset_input)):
        ret.append(imset_input[i][1]);
    return ret;


# Calculates imset1 - imset2. Returns lists A and B of sets 
# such that imset1 - imset2 = \sum_{S\in A}e_S - \sum_{S\in B}e_S

def imsetdif(imset1, imset2):
    A=[];
    B=[];
    if (len(imset1)!=len(imset2)):
        print("Imsets must be of equal size\n");
        return A, B;
    for i in range(len(imset1)):
        if (imset1[i][1] == 1):
            if (imset2[i][1] == 0):
                A.append(imset1[i][0]);
        else:
            if (imset2[i][1] == 1):
                B.append(imset1[i][0]);
    return A,B;

