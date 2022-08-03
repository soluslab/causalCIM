import pandas as pd
import numpy as np
import networkx as nx
from networkx.algorithms.tree.branchings import Edmonds
from networkx.algorithms import tree
import random
import ges
from DAGTreeGenerator import (
    struct_hamming_sim,
    true_pos,
    false_pos,
    to_igraph
)
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
from RP_algorithm import rp

# Import Sachs observational data
df = pd.read_csv('sachs_obs_data.csv', delimiter = ',')

# Convert to an numpy array
sachs_samples = df.values

# The true graph with variable ordering Raf, Mek, PLCg, PIP2, PIP3, Erk, Akt, PKA, PKC, p38, JNK.
true_graph = np.array([
    [0,1,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,1,0,0,0,0,0],
    [0,0,0,1,0,0,0,0,1,0,0],
    [0,0,0,0,0,0,0,0,1,0,0],
    [0,0,1,1,0,0,1,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0],
    [1,1,0,0,0,1,1,0,0,1,1],
    [0,0,0,0,0,0,0,0,0,1,0],
    [0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0]
])

# Get CPDAG of true graph
true_DAG = cdag.DAG.from_amat(true_graph)
true_CP = true_DAG.cpdag()
true_cpdag = true_CP.to_amat()[0].astype(bool)

# Get true skeleton
true_skel = true_cpdag + true_cpdag.T

# Collecting data
ges_w_skel_acc = np.empty(1, float)
ges_wo_skel_acc = np.empty(1, float)
gsp_wo_skel_acc = np.empty([1, 9], float)
eft_w_skel_acc = np.empty(1, float)
eft_wo_skel_acc = np.empty([1, 10], float)
deft_wo_skel_acc = np.empty([1, 10], float)
rp_wo_skel_acc = np.empty([1, 10], float)
ges_w_skel_roc = np.empty([1, 2], float)
ges_wo_skel_roc = np.empty([1, 2], float)
gsp_wo_skel_roc = np.empty([9, 2], float)
eft_w_skel_roc = np.empty([1, 2], float)
eft_wo_skel_roc = np.empty([10, 2], float)
deft_wo_skel_roc = np.empty([10, 2], float)
rp_wo_skel_roc = np.empty([10, 2], float)
gtruth_graph = np.empty((1, 11, 11), bool)
ges_w_skel_graph = np.empty((1, 11, 11), bool)
ges_wo_skel_graph = np.empty((1, 11, 11), bool)
gsp_wo_skel_graph = np.empty((9, 11, 11), bool)
eft_w_skel_graph = np.empty((1, 11, 11), bool)
eft_wo_skel_graph = np.empty((10, 11, 11), bool)
deft_wo_skel_graph = np.empty((10, 11, 11), bool)
rp_wo_skel_graph = np.empty((10, 11, 11), bool)

# Store true cpdag
gtruth_graph[0] = true_cpdag

# Run EFT with skeleton of "true graph" given as background knowledge
true_skel_edges = [
    [0, 1],
    [0, 7],
    [0, 8],
    [1, 5],
    [1, 7],
    [1, 8],
    [2, 3],
    [2, 4],
    [2, 8],
    [3, 4],
    [3, 8],
    [4, 6],
    [5, 7],
    [6, 7],
    [7, 9],
    [7, 10],
    [8, 9],
    [8, 10],
]
ig_true_skel = to_igraph(true_skel_edges, 11)

# Generate a random spanning tree of the skeleton:
layout = ig_true_skel.layout("grid")
random.seed(9)
permutation = list(range(ig_true_skel.vcount()))
random.shuffle(permutation)
ig_true_skel = ig_true_skel.permute_vertices(permutation)

new_layout = ig_true_skel.layout("grid")
for i in range(11):
    new_layout[permutation[i]] = layout[i]
layout = new_layout

spanning_tree = ig_true_skel.spanning_tree(weights=None, return_tree=True)
spanning_tree_adj = spanning_tree.get_adjacency()

# Need to extract a skeleton that is a tree.  Uses randomly generating spanning tree "spanning_tree."
bic = ges.scores.GaussObsL0Pen(sachs_samples)
eft_graph, eft_score = eft(sachs_samples, skeleton=spanning_tree)
eft_adj = eft_graph.get_adjacency()
eft_cd_dag = cdag.DAG.from_amat(eft_adj)
eft_cpdag = eft_cd_dag.cpdag().to_amat()[0].astype(bool)
eft_w_skel_acc[0] = struct_hamming_sim(eft_cpdag, true_cpdag)
eft_w_skel_roc[0][0] = true_pos(eft_cpdag, true_cpdag)
eft_w_skel_roc[0][1] = false_pos(eft_cpdag, true_cpdag)
eft_w_skel_graph[0] = eft_cpdag

# Run GES with skeleton of spanning tree of true graph given as background knowledge
ges_graph, score = ges.fit_bic(sachs_samples, A0 = spanning_tree.get_adjacency_sparse().toarray(), phases = ['turning'])
ges_w_skel_acc[0] = struct_hamming_sim(ges_graph.astype(bool), true_cpdag)
ges_w_skel_roc[0][0] = true_pos(ges_graph.astype(bool), true_cpdag)
ges_w_skel_roc[0][1] = false_pos(ges_graph.astype(bool), true_cpdag)
ges_w_skel_graph[0] = ges_graph

# Run EFT without skeleton as background knowledge
for e_idx in range(10):
    idx = int(e_idx)
    num_bins = 5*(idx + 1)
    eft_graph, eft_score = eft(sachs_samples, bins=num_bins)
    eft_adj = eft_graph.get_adjacency()
    eft_cd_dag = cdag.DAG.from_amat(eft_adj)
    eft_cpdag = eft_cd_dag.cpdag().to_amat()[0].astype(bool)
    eft_wo_skel_acc[0][idx] = struct_hamming_sim(eft_cpdag, true_cpdag)
    eft_wo_skel_roc[idx][0] = true_pos(eft_cpdag, true_cpdag)
    eft_wo_skel_roc[idx][1] = false_pos(eft_cpdag, true_cpdag)
    eft_wo_skel_graph[idx] = eft_cpdag

# Discretized data EFT
for e_idx in range(6):
    idx = int(e_idx)
    num_bins = 5*(idx + 1)
    deft_graph, deft_score = eft(sachs_samples, bins=num_bins, datatype="Discretize")
    deft_adj = deft_graph.get_adjacency()
    deft_cd_dag = cdag.DAG.from_amat(deft_adj)
    deft_cpdag = deft_cd_dag.cpdag().to_amat()[0].astype(bool)
    deft_wo_skel_acc[0][idx] = struct_hamming_sim(deft_cpdag, true_cpdag)
    deft_wo_skel_roc[idx][0] = true_pos(deft_cpdag, true_cpdag)
    deft_wo_skel_roc[idx][1] = false_pos(deft_cpdag, true_cpdag)
    deft_wo_skel_graph[idx] = deft_cpdag


# Run RP-algorithm without skeleton as background knowledge
for e_idx in range(10):
    idx = int(e_idx)
    num_bins = 5*(idx + 1)
    rp_wo_skel_adj = rp(sachs_samples, bins=num_bins)
    rp_wo_skel_cpdag = rp_wo_skel_adj.astype(bool)
    rp_wo_skel_acc[0][idx] = struct_hamming_sim(rp_wo_skel_cpdag, true_cpdag)
    rp_wo_skel_roc[idx][0] = true_pos(rp_wo_skel_cpdag, true_cpdag)
    rp_wo_skel_roc[idx][1] = false_pos(rp_wo_skel_cpdag, true_cpdag)
    rp_wo_skel_graph[idx] = rp_wo_skel_cpdag

# Run GES without skeleton of true graph given as background knowledge
ges_graph, score = ges.fit_bic(sachs_samples, phases = ['forward', 'turning', 'backward'])
ges_wo_skel_acc[0] = struct_hamming_sim(ges_graph.astype(bool), true_cpdag)
ges_wo_skel_roc[0][0] = true_pos(ges_graph.astype(bool), true_cpdag)
ges_wo_skel_roc[0][1] = false_pos(ges_graph.astype(bool), true_cpdag)
ges_wo_skel_graph[0] = ges_graph

# Run GreedySP without skeleton as background knowledge
alphas = [0.05, 0.025, 0.01, 0.005, 0.0025, 0.001, 0.0005, 0.00025, 0.0001]
for e_idx in range(9):
    idx = int(e_idx)
    suffstat = cdag.partial_correlation_suffstat(sachs_samples)
    ci_tester = MemoizedCI_Tester(partial_correlation_test, suffstat, alpha=alphas[idx])
    gsp_graph = cdag.gsp(set(range(11)), ci_tester)
    gsp_cpdag = gsp_graph.cpdag().to_amat()[0].astype(bool)
    gsp_wo_skel_acc[0][idx] = struct_hamming_sim(gsp_cpdag, true_cpdag)
    gsp_wo_skel_roc[idx][0] = true_pos(gsp_cpdag, true_cpdag)
    gsp_wo_skel_roc[idx][1] = false_pos(gsp_cpdag, true_cpdag)
    gsp_wo_skel_graph[idx] = gsp_cpdag


np.savez(
    "sachs_results.npz",
    ges_w_skel_acc=ges_w_skel_acc,
    ges_wo_skel_acc=ges_wo_skel_acc,
    gsp_wo_skel_acc=gsp_wo_skel_acc,
    eft_w_skel_acc=eft_w_skel_acc,
    eft_wo_skel_acc=eft_wo_skel_acc,
    deft_wo_skel_acc=deft_wo_skel_acc,
    rp_wo_skel_acc=rp_wo_skel_acc,
    ges_w_skel_roc=ges_w_skel_roc,
    ges_wo_skel_roc=ges_wo_skel_roc,
    gsp_wo_skel_roc=gsp_wo_skel_roc,
    eft_w_skel_roc=eft_w_skel_roc,
    eft_wo_skel_roc=eft_wo_skel_roc,
    deft_wo_skel_roc=deft_wo_skel_roc,
    rp_wo_skel_roc=rp_wo_skel_roc,
    ges_w_skel_graph=ges_w_skel_graph,
    ges_wo_skel_graph=ges_wo_skel_graph,
    gsp_wo_skel_graph=gsp_wo_skel_graph,
    eft_w_skel_graph=eft_w_skel_graph,
    eft_wo_skel_graph=eft_wo_skel_graph,
    deft_wo_skel_graph=deft_wo_skel_graph,
    rp_wo_skel_graph=rp_wo_skel_graph,
    gtruth_graph=gtruth_graph
    )