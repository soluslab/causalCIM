import numpy as np
import pandas as pd
import scipy.optimize as opt
from QIGTree_hyperplanes import (
    getCoords,
    getInequalities
)
from QIGTreelearn import (
    MI_MWST,
    getStandardImsetFunctional,
    transformToCIM
)
from iQIGTreelearn import (
    getLeafInts,
    iExtendedAdjMat,
    idatavec
)



# object created by inserting a dataframe in the form of a numpy array or pandas dataframe if
# only observational data is present.
# If you have interventional data instantiate the object using a list of pandas dataframes and
# a list of intervention targets given as a list of lists.
# In the interventional case, the first dataframe in the list should be the observational data
# and the first list in the intervention targets should be the empty list.

class QIG:

    data = None
    datanp = None
    targets = [[]]

    bins = 5
    method = 'kruskal'
    skeleton = None
    coordinates = None
    inequalities = None

    BIC_sim = None
    BIC_cim = None
    leaf_targets = None
    leaf_ints = None

    def __init__(self, df, ints, Bins=None, Method=None):
        if Bins != None:
            self.bins = Bins

        if Method != None:
            self.method = Method

        if isinstance(df, np.ndarray) == True:
            self.data = pd.DataFrame(df)
            self.datanp = df
            self.skeleton = MI_MWST(self.datanp, bins=self.bins, method=self.method)
            self.coordinates = getCoords(self.skeleton[3])
            self.BIC_sim = getStandardImsetFunctional(self.datanp)
            self.BIC_cim = transformToCIM(self.BIC_sim, self.coordinates)
            self.inequalities = getInequalities(self.skeleton[3], self.coordinates)

        elif isinstance(df, pd.core.frame.DataFrame) == True:
            self.data = df
            self.datanp = df.to_numpy()
            self.skeleton = MI_MWST(self.datanp, bins=self.bins, method=self.method)
            self.coordinates = getCoords(self.skeleton[3])
            self.BIC_sim = getStandardImsetFunctional(self.datanp)
            self.BIC_cim = transformToCIM(self.BIC_sim, self.coordinates)
            self.inequalities = getInequalities(self.skeleton[3], self.coordinates)

        elif isinstance(df, list):
            self.data = df
            self.datanp = [x.to_numpy() for x in df]
            self.targets = ints
            self.skeleton = MI_MWST(self.datanp[0], bins=self.bins, method=self.method)[3]
            self.leaf_targets = getLeafInts(self.targets, self.skeleton)
            self.leaf_ints = [[]] + [self.targets[k] for k in list(self.leaf_targets.keys())]
            self.skeleton = iExtendedAdjMat(ints, self.skeleton)
            self.coordinates = getCoords(self.skeleton)
            self.BIC_sim = idatavec(self.datanp, self.leaf_ints)
            self.BIC_cim = transformToCIM(self.BIC_sim, self.coordinates)
            self.inequalities = getInequalities(self.skeleton, self.coordinates)

        else:
            print('You must insert data in the form of a pandas dataframe or numpy array to build this object.')


    def linsolv(self, opt_method='highs'):
        c_vec = [-x for x in self.BIC_cim]
        [Amat, bmat] = self.inequalities

        return opt.linprog(c_vec, A_ub=Amat, b_ub=bmat, method=opt_method)
