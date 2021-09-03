library(glmnet)
library(microbenchmark)

prefix <- '../data/'

ns <- c(100, 500, 1000, 2000)
ps <- 2**(1:14)

avg.times <- matrix(0., nrow=length(ps) * length(ns), ncol=3)

iters <- c(rep(10, 5), rep(5, 3), rep(1, 6))

curr_row <- 1
for (j in 1:length(ns))
{
    for (i in 1:length(ps)) 
    {
        n <- ns[j]
        p <- ps[i]

        X <- read.csv(paste(prefix, 'x_unif_', n, '_', p, '.csv', sep=''))
        y <- read.csv(paste(prefix, 'y_unif_', n, '_', p, '.csv', sep=''))

        X <- as.matrix(scale(X)) * sqrt(n/(n-1))
        y <- as.numeric(scale(y)) * sqrt(n/(n-1))

        out <- microbenchmark(
            glmnet(X, y, family='gaussian',
                   intercept=F, standardize.response=F, type.gaussian='covariance'),
            unit='ns', times=iters[i])
        avg.times[curr_row, 1] <- mean(out$time)
        avg.times[curr_row, 2] <- p
        avg.times[curr_row, 3] <- n
        curr_row <- curr_row + 1
    }
}
write.table(avg.times, sep=',', row.names=F, col.names=F)
