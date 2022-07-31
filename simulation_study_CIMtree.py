# Importing Packages
from DAGTreeGenerator import (
    topological_sort,
    gen_samples,
    struct_hamming_sim,
    true_pos,
    false_pos,
    to_igraph
)
from RP_algorithm import rp
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
from essential_flip_tree_search import (
    essential_flip_search,
    eft
)

# Set number of desired experiments, nodes for tree and samples to be drawn.
num_exp = 100
num_nodes = 10
num_samples = 1000

# Set seed for generating a collection of seeds for gen_samples equal to the number of experiments.
# Seeds are drawn from a range dependent on the number of experiments.
seed = 10  # seed 10 used in reported simulations
random.seed(seed)
rng = default_rng(seed)
seeds = [random.randint(10*num_exp, 100*num_exp) for i in range(num_exp)]

# Generate data and get results:
num_sources = np.empty(num_exp, int)
ges_acc = np.empty(num_exp, float)
gsp_acc = np.empty(num_exp, float)
eft_acc = np.empty(num_exp, float)
ges_roc = np.empty([num_exp, 2], float)
gsp_roc = np.empty([num_exp, 2], float)
eft_roc = np.empty([num_exp, 2], float)
true_graphs = np.empty((num_exp, num_nodes, num_nodes), bool)
true_CPDAGs = np.empty((num_exp, num_nodes, num_nodes), bool)
ges_graphs = np.empty((num_exp, num_nodes, num_nodes), bool)
gsp_graphs = np.empty((num_exp, num_nodes, num_nodes), bool)
eft_graphs = np.empty((num_exp, num_nodes, num_nodes), bool)

ges_wo_skel_acc = np.empty(num_exp, float)
gsp_wo_skel_acc = np.empty(num_exp, float)
rp_wo_skel_acc = np.empty(num_exp, float)
eft_wo_skel_acc = np.empty(num_exp, float)
ges_wo_skel_roc = np.empty([num_exp, 2], float)
gsp_wo_skel_roc = np.empty([num_exp, 2], float)
rp_wo_skel_roc = np.empty([num_exp, 2], float)
eft_wo_skel_roc = np.empty([num_exp, 2], float)
ges_wo_skel_graphs = np.empty((num_exp, num_nodes, num_nodes), bool)
gsp_wo_skel_graphs = np.empty((num_exp, num_nodes, num_nodes), bool)
rp_wo_skel_graphs = np.empty((num_exp, num_nodes, num_nodes), bool)
eft_wo_skel_graphs = np.empty((num_exp, num_nodes, num_nodes), bool)

# Experiments
for e_idx in range(num_exp):
    idx = int(e_idx)
    # generate true dag and corresponding data
    true_tree, order, true_matrix, weights, samples = gen_samples(num_nodes, num_samples, seeds[e_idx])
    true_graphs[idx] = true_matrix[np.argsort(order)][:, np.argsort(order)]
    # To save the generated data and data-generating tree to text files uncomment the following two lines
    # and create a directory in the working directory called "synth_data_CIMtree"
    # np.savetxt("synth_data_CIMtree/samples_" + str(idx), samples)
    # np.savetxt("synth_data_CIMtree/dag_" + str(idx), true_matrix[np.argsort(order)][:, np.argsort(order)])

    # True CPDAG
    true_DAG = cdag.DAG.from_amat(true_matrix[np.argsort(order)][:, np.argsort(order)].astype(int))
    true_CP = true_DAG.cpdag()
    true_cpdag = true_CP.to_amat()[0].astype(bool)
    true_CPDAGs[idx] = true_cpdag

    # Get structural hamming distance between learned and true graph
    def acc(g):
        return struct_hamming_sim(g, true_cpdag)

    # Get true positives and false positives in CPDAG of learned matrix
    def roc(g):
        return true_pos(g, true_cpdag), false_pos(g, true_cpdag)

    # GES with true skeleton result, accuracy and true/false positives
    skel = to_igraph(true_tree, num_nodes)
    ges_graph, score = ges.fit_bic(samples, A0 = skel.get_adjacency_sparse().toarray(), phases = ['turning'])
    # to run GES without the skeleton as background knowledge
    # uncomment the following line and comment out the preceding line.
    # ges_graph, score = ges.fit_bic(samples)
    ges_acc[idx] = acc(ges_graph.astype(bool))
    ges_roc[idx][0], ges_roc[idx][1] = roc(ges_graph.astype(bool))
    ges_graphs[idx] = ges_graph

    #GreedySP with skeleton result, accuracy and true/false positives
    suffstat = cdag.partial_correlation_suffstat(samples)
    # Give greedySP the skeleton as a pair of fixed adjacencies and fixed gaps.
    tuple_edges = {tuple(e) for e in true_tree}
    edges_sorted = [sorted(e) for e in true_tree]
    non_edges = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if ((i < j) and (([i,j] in edges_sorted) == False)):
                non_edges = non_edges + [[i,j]]
    tuple_non_edges = {tuple(e) for e in non_edges}
    # Run greedySP given the skeleton
    ci_tester = MemoizedCI_Tester(partial_correlation_test, suffstat, alpha=0.05)
    gsp_graph = cdag.gsp(set(range(num_nodes)), ci_tester, fixed_adjacencies=tuple_edges, fixed_gaps=tuple_non_edges)
    gsp_cpdag = gsp_graph.cpdag().to_amat()[0].astype(bool)
    gsp_acc[idx] = acc(gsp_cpdag)
    gsp_roc[idx][0], gsp_roc[idx][1] = roc(gsp_cpdag.astype(bool))
    gsp_graphs[idx] = nx.to_numpy_array(gsp_graph.to_nx())

    #EFTSearch with true skeleton given as background knowledge
    bic = ges.scores.GaussObsL0Pen(samples)
    eft_graph, eft_score = essential_flip_search(skel, bic)
    eft_adj = eft_graph.get_adjacency()
    eft_cd_dag = cdag.DAG.from_amat(eft_adj)
    eft_cpdag = eft_cd_dag.cpdag().to_amat()[0].astype(bool)
    eft_acc[idx] = acc(eft_cpdag)
    eft_roc[idx][0], eft_roc[idx][1] = roc(eft_cpdag)
    eft_graphs[idx] = eft_cpdag

    # GES without true skeleton result, accuracy and true/false positives
    ges_wo_skel_graph, score_wo_skel = ges.fit_bic(samples)
    ges_wo_skel_acc[idx] = acc(ges_wo_skel_graph.astype(bool))
    ges_wo_skel_roc[idx][0], ges_wo_skel_roc[idx][1] = roc(ges_wo_skel_graph.astype(bool))
    ges_wo_skel_graphs[idx] = ges_wo_skel_graph

    # GreedySP without true skeleton result, accuracy and true/false positives
    suffstat = cdag.partial_correlation_suffstat(samples)
    ci_tester = MemoizedCI_Tester(partial_correlation_test, suffstat, alpha=0.05)
    gsp_wo_skel_graph = cdag.gsp(set(range(num_nodes)), ci_tester)
    gsp_wo_skel_cpdag = gsp_wo_skel_graph.cpdag().to_amat()[0].astype(bool)
    gsp_wo_skel_acc[idx] = acc(gsp_wo_skel_cpdag)
    gsp_wo_skel_roc[idx][0], gsp_wo_skel_roc[idx][1] = roc(gsp_wo_skel_cpdag.astype(bool))
    gsp_wo_skel_graphs[idx] = nx.to_numpy_array(gsp_wo_skel_graph.to_nx())

    # RP-algorithm without true skeleton given as background knowledge
    rp_wo_skel_adj = rp(samples)
    rp_wo_skel_cpdag = rp_wo_skel_adj.astype(bool)
    rp_wo_skel_acc[idx] = acc(rp_wo_skel_cpdag)
    rp_wo_skel_roc[idx][0], rp_wo_skel_roc[idx][1] = roc(rp_wo_skel_cpdag)
    rp_wo_skel_graphs[idx] = rp_wo_skel_cpdag

    # EFTSearch without true skeleton given as background knowledge
    # bic = ges.scores.GaussObsL0Pen(samples)
    eft_wo_skel_graph, eft_wo_skel_score = eft(samples)
    eft_wo_skel_adj = eft_wo_skel_graph.get_adjacency()
    eft_wo_skel_cd_dag = cdag.DAG.from_amat(eft_wo_skel_adj)
    eft_wo_skel_cpdag = eft_wo_skel_cd_dag.cpdag().to_amat()[0].astype(bool)
    eft_wo_skel_acc[idx] = acc(eft_wo_skel_cpdag)
    eft_wo_skel_roc[idx][0], eft_wo_skel_roc[idx][1] = roc(eft_wo_skel_cpdag)
    eft_wo_skel_graphs[idx] = eft_wo_skel_cpdag

    np.savez(
        "results.npz",
        ges_acc=ges_acc,
        gsp_acc=gsp_acc,
        eft_acc=eft_acc,
        ges_roc=ges_roc,
        gsp_roc=gsp_roc,
        eft_roc=eft_roc,
        gsp_graphs=gsp_graphs,
        ges_graphs=ges_graphs,
        eft_graphs=eft_graphs,
        ges_wo_skel_acc=ges_wo_skel_acc,
        gsp_wo_skel_acc=gsp_wo_skel_acc,
        rp_wo_skel_acc=rp_wo_skel_acc,
        eft_wo_skel_acc=eft_wo_skel_acc,
        ges_wo_skel_roc=ges_wo_skel_roc,
        gsp_wo_skel_roc=gsp_wo_skel_roc,
        rp_wo_skel_roc=rp_wo_skel_roc,
        eft_wo_skel_roc=eft_wo_skel_roc,
        gsp_wo_skel_graphs=gsp_wo_skel_graphs,
        ges_wo_skel_graphs=ges_wo_skel_graphs,
        rp_wo_skel_graphs=rp_wo_skel_graphs,
        eft_wo_skel_graphs=eft_wo_skel_graphs,
        true_graphs=true_graphs,
        true_CPDAGs=true_CPDAGs
    )

# # Example of how to view the results
# results = np.load('results.npz')
# print(results['ges_roc'])
# print(results['ges_acc'])
#
# print(results['gsp_roc'])
# print(results['gsp_acc'])
#
# print(results['eft_roc'])
# print(results['eft_acc'])






