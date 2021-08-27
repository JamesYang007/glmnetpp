from sklearn.linear_model import Lasso
import numpy as np
import timeit

n = 100
ps = np.array([20000]) # np.array([10, 50, 100, 500, 1000, 2000])

for p in ps:
    X = np.random.rand(n, p)
    y = np.random.rand(n)

    # center and scale
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    y = (y - np.mean(y)) / np.std(y)

    model = Lasso(fit_intercept=False, normalize=False, precompute=False,
                  max_iter=100000, tol=1e-7, warm_start=True)

    def critical_section():
        _, _, _, n_iters = model.path(X, y, l1_ratio=1, eps=1e-4, precompute=False,
                                      coef_init=np.zeros(p), return_n_iter=True)
        print(n_iters)

    print(timeit.timeit(critical_section, number=1))
