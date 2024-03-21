#set random seed
set.seed(1)

#get the data
samples <- read.csv('~liamsolus/Dropbox/Research/alg-stat/graphical-models/geometric_QIG/QIG_code/sachs_exp/sachs.csv', sep = '\t');

#no-ICAM
experiments <- c(1, 3, 4, 5, 6, 7, 8, 9)
t.list <- list(integer(0), c(7), c(9), c(4), c(2), c(5), c(9), c(8))
data.list <- lapply(experiments, function(t) log(samples[which(samples[['experiment']]==t), 1:11]))
#print(data.list)
i <- 0
for (D in data.list){
  i <- i+1 
  filename <- paste(i,"-sachs.csv",sep="")
  write.csv(D, file=filename)
}
#title <- "sachs_safe_choice.csv"
#save(data.list,file=title)
#save(data.list, t.list, title, file=title)