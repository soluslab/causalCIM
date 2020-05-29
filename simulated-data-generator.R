#R scriptfor generating a simulated dataset for linear random Gaussian DAG models

#Load libraries
library("pcalg")
library("Rgraphviz")

#Generating a random DAG using the function randomDAG
p<-10 #number of nodes
myDAG<-randomDAG(n=p,prob=0.2,lB=0.1,uB=1)
plot(myDAG)

#Generating a linear Gaussian DAG model from a given random DAG
n<-10000 #number of samples
dat<-rmvDAG(n,myDAG,errDist="normal")

#Generating the sufficient statistic for the data sample from the Gaussian DAG
mySuffStat <- list(C = cor(dat), n = nrow(dat))
mySuffStat

#Running the PC algorithm on the sufficient statistic
pc.fit <- pc(mySuffStat, indepTest = gaussCItest, p=10, alpha = 0.01)
plot(pc.fit)

#Set the working directory to where we would like to store the sufficient statistics for each simulated model.
setwd("~/Dropbox/Research/alg-stat/graphical-models/simulations/simulated-dataset/100-RGDMs-n_1000-p_5-a_0.1")

#Generate a list of the true graphs.
trueGraphs <- function(N,p,prob,lB,uB) 
{
  trueGraphs <- list()
  for(i in 1:N) 
  {
    nam <- paste("trueGraph", i, sep = "")
    trueGraphs <- append(trueGraphs,assign(nam,randomDAG(p,prob,lB,uB)))
  }
  return(trueGraphs)
}


#Building a function for producing a directory of csv files containing samples from random Gaussian DAG models.
dataFileGenerator <- function(N,n,p,prob,lB,uB) #n is the sample size, N is the number of models, p is the number of nodes
  {
  TG <- trueGraphs(N,p,prob,lB,uB)
  for(i in 1:length(TG))
  {
    dat <- rmvDAG(n,TG[[i]],errDist="normal")
    write.csv(dat,paste0('model-',i,'-edge-prob-',prob,'.csv'))
    write.csv(as(TG[[i]],"matrix"),paste0('true-graph-',i,'-edge-prob-',prob,'.csv'))
  }
}



#Example for 1000 samples from 100  DAG models on 5 nodes with ER random graphs with edges with probability 0.3 and edgeweights between 0.1 and 1.
for(prob in 1:9){
  dataFileGenerator(10,1000,5,prob/10,0.1,1)
  }

#Reading in a data file and creating its sufficient statistic for the PC Algorithm.
produceSuffStat <- function(file) 
{
  dat <- (read.csv(file))[-c(1)]
  return(list(C = cor(dat), n = nrow(dat)))
}

#Using the sufficient statistic to run an instance of the PC Algorithm.
SuffStat <- produceSuffStat("model-1-edge-prob-0.4.csv")
pc.fit <- pc(SuffStat, indepTest = gaussCItest, p=5, alpha = 0.01)
plot(pc.fit)

#We compare the graph in file 'model-i.csv' to graph i in the list graphs of true graphs.
AdjMat <- read.csv("true-graph-1-edge-prob-0.4.csv")
AdjMat





