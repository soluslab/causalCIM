import numpy as np
from itertools import chain, combinations


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def getVertices(T):
    V = []

    for e in T:
        for i in e:
            if not i in V:
                V.append(i)

    sorted(V)
    return V


def getEdges(T):
    E = []

    for i in range(len(T) - 1):
        for j in range(len(T)):
            if j > i and T[i][j] == 1:
                E += [[i, j]]

    return E


def getDegree(T, i):
    return (sum(1 for j in range(len(T)) if T[i][j] == 1) - 1)


def neighbors(T, i):
    return [j for j in range(len(T)) if T[i][j] == 1 and not (i == j)]


def neighborhood(T, i):
    return [j for j in range(len(T)) if T[i][j] == 1]


def getAdjacencyMatrix(T):
    V = getVertices(T)

    M = np.array([[0 for j in V] for i in V], np.int32)

    for i in V:
        M[i, i] = 1

    for e in T:
        M[e[0], e[1]] = 1
        M[e[1], e[0]] = 1

    return M


def getCoords(T):  # returns the list of coordinates
    V = range(len(T))
    coords = []

    for i in range(len(T)):
        if sum(T[i][j] for j in range(len(T))) <= 2:
            continue

        S = [j for j in range(len(V)) if T[i][j] == 1]

        for R in powerset(S):
            if len(R) <= 2 or not (i in R):
                continue

            R = [j for j in R]

            if not R in coords:
                coords.append(R)

    return coords


def getMaxStars(T):  # returns the list of maximal stars in T
    V = range(len(T))
    maxStars = []

    for i in range(len(T)):
        S = [i] + [[j for j in V if T[i][j] == 1 and i != j]]

        if len(S[1]) > 1:
            maxStars.append(S)

    return maxStars


def getStarInequalities(T, coords, maxStars):
    b = [0 for i in coords]
    A = []

    for starPair in maxStars:
        for S in powerset(starPair[1]):
            if len(S) <= 1:
                continue

            S = sorted(list(S) + [starPair[0]])

            a = []

            for C in coords:
                if all(j in C for j in S) and all(j == starPair[0] or j in starPair[1] for j in C):
                    a.append((-1) ** (1 + len(C) - len(S)))
                else:
                    a.append(0)

            A.append(a)

    return [A, b]


def getBiDirectedEdgeInequalities(T, coords, maxStars):
    A = []
    b = []

    E = getEdges(T)

    for e in E:
        if not (getDegree(T, e[0]) > 1 and getDegree(T, e[1]) > 1):
            continue

        i = e[0]
        j = e[1]


        for starPair in maxStars:
            if starPair[0] == i:
                Si = starPair[1]
            elif starPair[0] == j:
                Sj = starPair[1]

        ind_i_from_j = []

        for C in coords:
            if i in C and j in C and all(k == i or k in Si for k in C) and len(C) > 2:
                ind_i_from_j.append((-1) ** (len(C) - 1))
            else:
                ind_i_from_j.append(0)

        ind_j_from_i = []

        for C in coords:
            if i in C and j in C and all(k == j or k in Sj for k in C) and len(C) > 2:
                ind_j_from_i.append((-1) ** (len(C) - 1))
            else:
                ind_j_from_i.append(0)

        A.append([ind_i_from_j[k] + ind_j_from_i[k] for k in range(len(coords))])
        b.append(1)

    return [A, b]


def getSeparatedTree(T):
    V = [i for i in range(len(T))]
    E = getEdges(T)
    leavesT = [e for e in E if getDegree(T, e[0]) == 1 or getDegree(T, e[1]) == 1]

    bigV = V
    bigE = []

    for e in E:
        if e in leavesT:
            bigE.append(e)
        else:
            v = len(bigV)
            bigV.append(v)

            bigE.append([e[0], v])
            bigE.append([e[1], v])

    return getAdjacencyMatrix(bigE)


def getForkedTrees(T, coords, maxStars):
    V = range(len(T))
    E = getEdges(T)

    bigT = getSeparatedTree(T)
    bigV = range(len(bigT))
    bigE = getEdges(bigT)

    oldTrees = [[i] for i in range(len(T)) if getDegree(T, i) > 1]

    oldTrees = []

    for i in V:
        if getDegree(T, i) > 1:
            F = []

            for j in neighborhood(bigT, i):
                if i == j or j >= len(V):
                    F.append(j)

            oldTrees.append(F)

    while len(bigV) > len(V):
        v = bigV[len(bigV) - 1]
        [e0, e1] = neighbors(bigT, v)

        midT = np.array([[np.array(bigT[k][l]) for l in range(len(bigT) - 1)] for k in range(len(bigT) - 1)])
        midT[e0][e1] = 1
        midT[e1][e0] = 1

        newTrees = []

        for i in range(len(oldTrees)):
            if not e0 in oldTrees[i]:
                continue

            Ti = oldTrees[i]

            for j in range(len(oldTrees)):
                if i == j or not (e1 in oldTrees[j]):
                    continue

                Tj = oldTrees[j]

                newT = sorted(Ti + Tj)
                newT = [newT[k] for k in range(len(newT) - 2)]

                isForked = True

                for k in [e0, e1]:
                    if not sum(midT[k][l] for l in newT) == 2:
                        continue
                    elif getDegree(midT, k) - sum(midT[k][l] for l in newT if not k == l) >= 2:
                        continue
                    elif any(not (j in range(len(T))) for j in neighbors(midT, k)):
                        continue
                    else:
                        isForked = False

                if isForked:
                    newTrees.append([newT[k] for k in range(len(newT))])

        for Ti in oldTrees:
            if not (e0 in Ti or e1 in Ti):
                newTrees.append(Ti)
                continue

            Ti = [j for j in Ti if not j == v]

            if e0 in Ti:
                w = e0
            elif e1 in Ti:
                w = e1

            if sum(midT[i][w] for i in Ti if not i == w) >= 2:
                newTrees.append(Ti)
            elif getDegree(midT, w) - sum(midT[i][w] for i in Ti if not i == w) >= 2:
                newTrees.append(Ti)
            elif any(not (j in range(len(T))) for j in neighbors(midT, w)):
                newTrees.append(Ti)

        bigV = range(len(bigV) - 1)
        bigT = midT
        oldTrees = newTrees

    return oldTrees


def getForkedTreeInequalities(T, coords, maxStars):
    A = []
    b = []

    for TPrime in getForkedTrees(T, coords, maxStars):
        a = [0 for C in coords]

        for c in TPrime:
            Nc = [i for i in range(len(T)) if T[c][i] == 1 and not i == c]

            if getDegree(T, c) - sum(T[c][i] for i in TPrime) + 1 >= 2:
                L = [i for i in Nc if i in TPrime]
                notL = [i for i in Nc if not i in L]

                for i in range(len(coords)):
                    C = coords[i]

                    if not sum(1 for l in notL if l in C) >= 2:
                        continue

                    if all((i == c or i in Nc) for i in C):
                        if TPrime == [2, 3, 6, 7] and C == [2, 3, 6]:
                            print(c)
                        a[i] = a[i] + (-1) ** (len(C) - 1) * (len([j for j in notL if j in C]) - 1)

            if sum(T[c][i] for i in TPrime) > 2:
                NcTPrime = sorted([c] + [j for j in Nc if j in TPrime])
                for i in range(len(coords)):
                    C = coords[i]

                    if all(i in NcTPrime for i in C):
                        a[i] = a[i] + (-1) ** (len(C))

        A.append(a)
        b.append(1)

    return [A, b]


def getInequalities(T, coords):
    maxStars = getMaxStars(T)

    [A, b] = getStarInequalities(T, coords, maxStars)

    ieqs = getBiDirectedEdgeInequalities(T, coords, maxStars)
    A.extend(ieqs[0])
    b.extend(ieqs[1])

    ieqs = getForkedTreeInequalities(T, coords, maxStars)
    A.extend(ieqs[0])
    b.extend(ieqs[1])

    return [A, b]




