library(glmnet)

n <- 100
ps <- 20000#c(10, 50, 100, 500, 1000, 2000)

for (p in ps)  
{
    X <- matrix(runif(n*p, -1, 1), nrow=n)
    y <- runif(n, -1, 1)

    X <- as.matrix(scale(X)) * sqrt(n/(n-1))
    y <- as.numeric(scale(y)) * sqrt(n/(n-1))

    ptm <- proc.time()
    glmnet.out <- glmnet(X, y, family='gaussian', alpha=1, standardize=F, nlambda=6,
                         intercept=F, standardize.response=F, type.gaussian='covariance')
    diff <- proc.time() - ptm
    print(diff)
    print(head(glmnet.out$lambda))
}
