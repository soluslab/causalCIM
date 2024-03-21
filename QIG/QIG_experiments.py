#import packages
import time
import numpy as np
import random
from numpy.random import default_rng


import ges

from QIGsimulations import (
    gen_samples,
    getPatternFromAdjMat,
    getPatternFromCIM,
    to_igraph,
    acc
)

from QIGTreelearn import (
    CIMlinsolv
)

from essential_flip_tree_search import(
    essential_flip_search
)

#############################################


def getTime(start):  # This function is not needed, it's just for time keeping
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




# Set number of desired experiments, nodes for tree and samples to be drawn.
num_exp = 10
num_nodes = 5
num_samples = 250

# Seeds for data-generation
seed = 54
random.seed(seed)
rng = default_rng(seed)
seeds = [random.randint(10*num_exp, 100*num_exp) for i in range(num_exp)]

# Generate data and store results
GES_graphs = np.empty((num_exp, num_nodes, num_nodes), int)
GEStime = np.empty((num_exp,3), float)

EFTtime = np.empty((num_exp,3), float)

QIGtime = np.empty((num_exp,3), float)
QIGlin_s_acc = np.empty(num_exp, float)
QIGlin_v_acc = np.empty(num_exp, float)
QIGlin_skels = np.empty((num_exp, num_nodes, num_nodes), int)
QIGlin_v = np.empty((num_exp, num_nodes, num_nodes, num_nodes), int)

true_adjmats = np.empty((num_exp, num_nodes, num_nodes), int)
true_skels = np.empty((num_exp, num_nodes, num_nodes), int)
true_v = np.empty((num_exp, num_nodes, num_nodes, num_nodes), int)


for e_idx in range(num_exp):
    idx = int(e_idx)

    #True model
    true_tree, order, true_adjmat, weights, samples = gen_samples(num_nodes, num_samples, seeds[idx])
    true_adjmats[idx] = true_adjmat
    true_pattern = getPatternFromAdjMat(true_adjmat)
    true_skels[idx], true_v[idx] = true_pattern

    #QIGlin
    start = time.time()
    QIGlin_result = CIMlinsolv(samples)
    QIGtime[idx] = getTime(start)
    est_pattern = getPatternFromCIM(QIGlin_result)
    QIGlin_skels[idx], QIGlin_v[idx] = est_pattern
    QIGlin_s_acc[idx], QIGlin_v_acc[idx] = acc(est_pattern, true_pattern)

    #EFT
    start = time.time()
    skel = to_igraph(true_tree, num_nodes)
    bic = ges.scores.GaussObsL0Pen(samples)
    eft_graph, eft_score = essential_flip_search(skel, samples)
    EFTtime[idx] = getTime(start)

    #GES
    start = time.time()
    ges_graph, score = ges.fit_bic(samples)
    GEStime[idx] = getTime(start)
    GES_graphs[idx] = ges_graph

    np.savez(
        "results.npz",
        GES_graphs=GES_graphs,
        GEStime=GEStime,
        EFTtime=EFTtime,
        QIGtime=QIGtime,
        QIGlin_s_acc=QIGlin_s_acc,
        QIGlin_v_acc=QIGlin_v_acc,
        QIGlin_skels=QIGlin_skels,
        QIGlin_v=QIGlin_v,
        true_adjmats=true_adjmats,
        true_skels=true_skels,
        true_v=true_v
    )


