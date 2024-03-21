import numpy as np
from QIGTree_hyperplanes import (
    powerset
)
from QIGTreelearn import (
    fill
)


# data is a list of numpy arrays
# ints is a list of intervention targets
def pooledData(data, ints):
    m = len(ints)
    intSubsets = [list(C) for C in powerset(range(m))]
    pooledDataSets = {}

    for Z in intSubsets:
        pooledDataSets[intSubsets.index(Z)] = []

    for Z in intSubsets:
        df = data[0]
        for i in Z:
            df = np.concatenate((df, data[i]))
        pooledDataSets[intSubsets.index(Z)] = df

    return pooledDataSets

# pooleddict is a dictionary that is the output of pooledData(data, ints)
# targetssubsets is the powerset (with elements as list objects) of the indices of ints
# Z is a subset of the indices of the nonempty intervention targets (e.g. an element of targetssubsets)
def pool(pooleddict, targetsubsets, Z):

    return np.cov(pooleddict[targetsubsets.index(Z)], rowvar=False)

# produce the BIC data vector for the interventional standard imsets.
def idatavec(data, ints):
    m = data[0].shape[1]
    num_ints = len(ints)
    intcovs = [np.cov(data[i], rowvar=False) for i in range(num_ints)]
    num_int_samps = [data[i].shape[1] for i in range(num_ints)]
    nonobs = [x for x in range(num_ints) if x != 0]

    bigCoordsA = [list(C) for C in powerset(range(m))]
    bigCoordsA.remove([])
    bigCoordsZ = [list(C) for C in powerset(nonobs)]
    datavec = []

    intSubsets = [list(C) for C in powerset(range(num_ints))]
    pools = pooledData(data, ints)
    print('pools made')

    for A in bigCoordsA:
        print('set:', A)
        for Z in bigCoordsZ:
            ZZ = Z + [0]
            Zcomp = [i for i in range(num_ints) if ZZ.count(i) == 0]
            Sk_As = [np.array([[intcovs[k][i][j] for j in A] for i in A]) for k in range(num_ints)]
            SZ_pool = pool(pools, intSubsets, Zcomp)
            SA_poolZcomp = np.array([[SZ_pool[i][j] for j in A] for i in A])

            datavec += [
                sum([
                        (1/2)*sum([
                            # np.matmul(np.matmul(x, fill(np.linalg.inv(pool(pools, intSubsets, Z)), A, m)), np.transpose(x))
                            np.matmul(np.matmul(x, fill(np.linalg.inv(SA_poolZcomp), A, m)), np.transpose(x))
                            for x in data[i]
                        ])
                        - (num_int_samps[i] / 2)*np.log(np.linalg.det(SA_poolZcomp))
                    for i in Zcomp])
                +
                sum([
                    (1/2)*sum([
                        np.matmul(np.matmul(x, fill(np.linalg.inv(Sk_As[i]), A, m)), np.transpose(x))
                    for x in data[i]])
                    - (num_int_samps[i] / 2)*np.log(np.linalg.det(Sk_As[i]))
                for i in Z])
                # - sum([
                #     (num_int_samps[i] / 2)*np.log(np.linalg.det(Sk_As[i]))
                # for i in range(num_ints)])
                - (1 / 2)*(len(A) * (len(A) - 1)) / 2
                - (1 / 2)*sum([
                    len(A) * (len(A) - 1) / 2 for i in range(len(Z))
                ])
            ]
    return datavec

def getLeaves(adjmat):
    p = adjmat.shape[0]
    degs = [sum(adjmat[:, i]) for i in range(p)]
    leaves = []
    for i in range(p):
        if degs[i] == 1:
            leaves += [i]
    return leaves

def getLeafInts(ints, adjmat):

    p = adjmat.shape[0]
    adjustedmat = adjmat - np.eye(p)
    leaves = getLeaves(adjustedmat)
    leaf_ints = {}

    for i in range(len(ints) - 1):
        leaf_ints[i + 1] = []

    for l in leaves:
        for i in range(len(ints) - 1):
            if ints[i + 1].count(l) != 0:
                leaf_ints[i + 1].append([l, ints[i + 1]])


    keys = list(leaf_ints.keys())

    for k in keys:
        if leaf_ints[k] == []:
            leaf_ints.pop(k)

    return leaf_ints

def iExtendedAdjMat(ints, adjmat):
    p = adjmat.shape[0]
    leafints = getLeafInts(ints, adjmat)
    leafintkeys = list(leafints.keys())
    K = len(leafintkeys)
    ext_adjmat = np.eye(p + K)


    for i in range(p):
        for j in range(p):
            ext_adjmat[i][j] = adjmat[i][j]

    for k in leafintkeys:
        I = leafints[k]
        ext_adjmat[I[0][0]][leafintkeys.index(k) + p] = 1
        ext_adjmat[leafintkeys.index(k) + p][I[0][0]] = 1

    return ext_adjmat


