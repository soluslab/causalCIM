# Importing Packages
from DAGTreeGenerator import (
    topological_sort,
    gen_samples,
    struct_hamming_sim,
    true_pos,
    false_pos
)
import numpy as np
import random
from numpy.random import default_rng, randint
import pandas as pd
import ges
import networkx as nx
import causaldag as cdag
from conditional_independence import (
    MemoizedCI_Tester,
    partial_correlation_suffstat,
    partial_correlation_test,
)

# Set number of desired experiments, nodes for tree and samples to be drawn.
num_exp = 10
num_nodes = 5
num_samples = 10000

# Set seed for generating a collection of seeds for gen_samples equal to the number of experiments.
# Seeds are drawn from a range dependent on the number of experiments.
seed = 10
random.seed(seed)
rng = default_rng(seed)
seeds = [random.randint(10*num_exp, 100*num_exp) for i in range(num_exp)]

# Generate data and get results:
num_sources = np.empty(num_exp, int)
gues_acc = np.empty(num_exp, float)
ges_acc = np.empty(num_exp, float)
gsp_acc = np.empty(num_exp, float)
ges_roc = np.empty([num_exp, 2], float)
gsp_roc = np.empty([num_exp, 2], float)
true_graphs = np.empty((num_exp, num_nodes, num_nodes), bool)
gues_graphs = np.empty((num_exp, num_nodes, num_nodes), bool)
ges_graphs = np.empty((num_exp, num_nodes, num_nodes), bool)
gsp_graphs = np.empty((num_exp, num_nodes, num_nodes), bool)

for e_idx in range(num_exp):
    idx = int(e_idx)
    # generate true dag and corresponding data
    true_tree, order, true_matrix, weights, samples = gen_samples(num_nodes, num_samples, seeds[e_idx])
    # while true_tree.sum() == 0:
    #     true_tree, order, matrix, weights, samples = gen_samples(num_nodes, num_samples)
    true_graphs[idx] = true_matrix[np.argsort(order)][:, np.argsort(order)]
    np.savetxt("synth_data_CIMtree/samples_" + str(idx), samples)
    np.savetxt("synth_data_CIMtree/dag_" + str(idx), true_matrix[np.argsort(order)][:, np.argsort(order)])

    def acc(g):
        true_DAG = cdag.DAG.from_amat(true_matrix[np.argsort(order)][:, np.argsort(order)].astype(int))
        true_CP = true_DAG.cpdag()
        true_cpdag = true_CP.to_amat()[0].astype(bool)
        return struct_hamming_sim(g, true_cpdag)

    def roc(g):
        true_DAG = cdag.DAG.from_amat(true_matrix[np.argsort(order)][:, np.argsort(order)].astype(int))
        true_CP = true_DAG.cpdag()
        true_cpdag = true_CP.to_amat()[0].astype(bool)
        return true_pos(g, true_cpdag), false_pos(g, true_cpdag)

    # GES result and accuracy
    ges_graph, score = ges.fit_bic(samples)
    ges_acc[idx] = acc(ges_graph.astype(bool))
    ges_roc[idx][0], ges_roc[idx][1] = roc(ges_graph.astype(bool))
    ges_graphs[idx] = ges_graph

    #GreedySP result and accuracy
    suffstat = cdag.partial_correlation_suffstat(samples)
    ci_tester = MemoizedCI_Tester(partial_correlation_test, suffstat, alpha=0.05)
    gsp_graph = cdag.gsp(set(range(num_nodes)), ci_tester)
    gsp_cpdag = gsp_graph.cpdag().to_amat()[0].astype(bool)
    gsp_acc[idx] = acc(gsp_cpdag)
    gsp_roc[idx][0], gsp_roc[idx][1] = roc(gsp_cpdag.astype(bool))
    gsp_graphs[idx] = nx.to_numpy_array(gsp_graph.to_nx())

    np.savez(
        "results.npz",
        ges_acc=ges_acc,
        gsp_acc=gsp_acc,
        ges_roc=ges_roc,
        gsp_roc=gsp_roc,
        gsp_graphs=gsp_graphs,
        ges_graphs=ges_graphs,
        true_graphs=true_graphs,
    )

# print(gsp_graphs)
# print(true_graphs)
# print(ges_acc)
# print(gsp_acc)






