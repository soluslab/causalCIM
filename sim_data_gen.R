library(pcalg)
library(graph)
library(MASS)

#parameter used for generating data
ps <- c(5)
ns <- 10000
neighs <- c(0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)

dagnum <- 100

#set random seed
set.seed(2313)

#generate random graphs
for(p in ps){
for(neigh in neighs){
	#function for generating random dags
	rnddag <- function(){
		wFUN <- function(m,a,b) { runif(m,a,b) }
		dag <- randDAG(p,d = neigh, wFUN=list(wFUN,0.25,1))
		dag <- as(dag, "matrix") * matrix(sample(c(-1, 1), p * p, replace=TRUE), nrow=p)
		dag <- lapply(1:p, function(i) list(edges=which(dag[i,] != 0), weights=dag[i, dag[i,] != 0]))
		names(dag) <- 1:p
		dag <- new("graphNEL", node=sapply(1:p, toString), edgeL=dag, edgemode="directed")
		return(dag)
	}

	#get list of dags
	dag.list <- lapply(1:dagnum, function(i) rnddag())
	i <- 1
	for(dag in dag.list){
		dat <- mvrnorm(ns, mu=rep(0,p), Sigma=trueCov(dag))
		write.csv(dat,paste0('nodes-', p, '/nbh-', neigh, '/model-',i,'.csv'))
		write.csv(as(dag,"matrix"),paste0('nodes-', p, '/nbh-', round(neigh, digits=2), '/true-graph-',i,'.csv'))
		i <- i+1
	}
}}

