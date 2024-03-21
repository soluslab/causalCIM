# causalCIM

This code provides causal learning methods for estimating a DAG model via the geometry of characteristic imset (CIM) polytopes. 
A CIM polytope is the feasible region of an integer linear program in which each vertex corresponds to a possible DAG model. 

The methods here estimate DAG models where the skeleton (e.g., undirected graph of adjacencies) of the learned DAG is a tree. 
All methods estimate this skeleton in an initial phase using a minimum weight spanning forest algorithm where edge weights correspond to the negative mutual information of the associated random variables. 

The EFT algorithm, contained in essential_flip_tree_search.py, uses combinatorial interpretations of the edges of the CIM polytope to perform a greedy search for the BIC optimal DAG model. 
This method is a simplex-type algorithm. 
Input data for EFT should be observational data only. 

QIG, with main file QIG.py, provides more general methods.  
It supports either observational data or a combination of observational and interventional data. 
QIG identifies the defining inequalities of the CIM polytope, which can be used for general integer linear programming techniques. 
QIG.py may be used to create a QIG object which stores the estimated tree skeleton, the defining inequalities of the CIM polytope and the data vector for linear optimization methods for Gaussian data. 
A QIG object has a the associated method linsolv which returns the BIC optimal DAG model via a standard interior point method solver.
