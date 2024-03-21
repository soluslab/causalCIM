import numpy as np
import networkx as nx
from scipy.stats import chi2_contingency
import scipy.optimize as opt
from networkx.algorithms import tree
from QIGsimulations import (
    to_igraph
)
from QIGTree_hyperplanes import (
    getCoords,
    powerset,
    getInequalities
)
import time

#############################################


def getTimeToo(start):  # This function is not needed, it's just for time keeping
    tiempo = time.time() - start
    hours = str(int(tiempo // 3600))

    if (tiempo // 60) % 60 <= 9:
        minutes = "0" + str(int((tiempo // 60) % 60))
    else:
        minutes = str(int((tiempo // 60) % 60))

    if np.floor(tiempo) % 60 <= 9:
        seconds = "0" + str(int(np.floor(tiempo) % 60))
    else:
        seconds = str(int(np.floor(tiempo) % 60))

    time_vec = [int(hours), int(minutes), int(seconds)]

    return time_vec
    # return str(hours) + ":" + str(minutes) + ":" + str(seconds)

################################################################

########################################################
### Estimating a skeleton (that is a tree) from data ###
########################################################

# Calculating empirical mutual information:
def calc_MI(data, x, y, bins):
    c_xy = np.histogram2d(data[:, x], data[:, y], bins)[0] + np.full([bins, bins], 0.0000000001, float)
    g, p, dof, expected = chi2_contingency(c_xy, lambda_="log-likelihood")
    mi = 0.5 * g / c_xy.sum()
    return mi

# Producing minimum weight spanning tree. Valid methods are 'kruskal, 'prim', or 'boruvka'.
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

    adjMat = np.zeros([n, n], int)
    for e in min_weight_tree_edge_list:
        adjMat[e[0]][e[1]] = 1
        adjMat[e[1]][e[0]] = 1
    adjMat = adjMat + np.eye(n)

    return ig_min_weight_tree, edge_weights, min_weight_tree_edge_list, adjMat

########################################################
#### Gaussian BIC data vector for a given skeleton #####
########################################################

def fill(S, C, n):  # Fill the #C x #C sample covariance S to an nxn matrix by adding 0s
    filling = [[0 for j in range(n)] for i in range(n)]

    for i in range(len(C)):
        for j in range(len(C)):
            filling[C[i]][C[j]] = S[i][j]

    return filling


def getStandardImsetFunctional(D):
    n = len(D)
    m = len(D[0])
    bigCoords = [list(C) for C in powerset(range(m))]

    mu = [(1/n)*sum(D[i][j] for i in range(n)) for j in range(m)]
    S = np.cov(D, rowvar=False)
    BIC = []

    for C in bigCoords:
        c = len(C)
        if c == 0:
            BIC.append(0)
            continue

        S_C = np.array([[S[i][j] for j in C] for i in C])

        BIC.append(
            (1/2)*(n*np.log(np.linalg.det(S_C)) + np.log(n)*(c**2 - c)/2 + sum(
                np.matmul(np.matmul(x - mu, fill(np.linalg.inv(S_C), C, len(D[0]))), np.transpose(x- mu)) for x in D
            ))
        )

    return np.array(BIC)


def transformToCIM(BIC, coords):
    p = len(BIC)
    num_nodes = np.log2(p).astype(int)
    bigCoords = [list(C) for C in powerset(range(num_nodes))]

    ######### Projection not needed afterall? #################
    # perpProjBIC = np.zeros(len(BIC), float)
    # for i in range(p):
    #     S = bigCoords[i]
    #     s = len(S)
    #     if s > 1:
    #         perpProjBIC[i] = sum([BIC[j] for j in S]) - s
    #     else:
    #         perpProjBIC[i] = BIC[i]
    #
    # adjusted_bic = BIC - perpProjBIC
    ############################################################

    imsetBIC = [
        sum(
            [(-1)**(len(coords[j]) - len(bigCoords[i]))*(1 - BIC[i]) for i in range(len(bigCoords)) if
             all(k in coords[j] for k in bigCoords[i])]
        ) for j in range(len(coords))
    ]


    return imsetBIC

def getBIC(D, coords):
    # start2 = time.time()
    BIC = getStandardImsetFunctional(D)
    # print('fast?', getTimeToo(start2))
    return transformToCIM(BIC, coords)


########################################################
###################### Solver ##########################
########################################################

# Linear Solver
def CIMlinsolv(data, skeleton=None, skel_method='kruskal', skel_bins=5, opt_method='highs'):

    if skeleton == None:
        skeleton = MI_MWST(data, bins=skel_bins, method=skel_method)[3]

    skel_coords = getCoords(skeleton)


    [Amat, bmat] = getInequalities(skeleton, skel_coords)

    # start1 = time.time()
    c_vec = [-x for x in getBIC(data, skel_coords)]
    # print(getTimeToo(start1))

    solution = opt.linprog(c_vec, A_ub=Amat, b_ub=bmat, method=opt_method)

    return solution, skeleton, skel_coords



