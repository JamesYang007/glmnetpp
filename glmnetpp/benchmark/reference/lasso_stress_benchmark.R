library(glmnet)

n <- 5000
ps <- c(10, 50, 100, 500, 1000, 2000)

for (p in ps)  
{
    X <- matrix(runif(n*p, -1, 1), nrow=n)
    y <- runif(n, -1, 1)

    X <- as.matrix(scale(X)) * sqrt(n/(n-1))
    y <- as.numeric(scale(y)) * sqrt(n/(n-1))

    ptm <- proc.time()
    glmnet.out <- glmnet(X, y, family='gaussian', alpha=1, standardize=F,
                         intercept=F, standardize.response=F, type.gaussian='covariance')
    diff <- proc.time() - ptm
    print(diff)
}
