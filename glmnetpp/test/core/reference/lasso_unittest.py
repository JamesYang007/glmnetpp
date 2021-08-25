import numpy as np
from sklearn.linear_model import Lasso

# Link: https://towardsdatascience.com/regularized-linear-regression-models-dcf5aa662ab9
def lasso(X, y, l1, tol=1e-7, max_iter=100000, path_length=100, return_path=False):
    m, n = np.shape(X)
    B_star = np.zeros((n))
    l_max = max(list(abs(np.dot(np.transpose(X), y)))) / m
    l_path = np.geomspace(l_max, l1, path_length)
    coeffiecients = np.zeros((len(l_path), n))
    for i in range(len(l_path)):
        for j in range(max_iter):
            print("{i}, {j}\n{B_star}".format(i=i, j=j, B_star=B_star))
            B_s = B_star.copy()
            for j in range(n):
                k = np.where(B_s != 0)[0]
                print(k)
                update = (1/m)*((np.dot(X[:,j], y)- \
                                np.dot(np.dot(X[:,j], X[:,k]), B_s[k]))) + \
                                B_s[j]
                B_star[j] = (np.sign(update) * max(abs(update) - l_path[i], 0))
            print(max(abs(B_s - B_star)))
            if np.all(abs(B_s - B_star) < tol):
                break
        coeffiecients[i, :] = B_star
    if return_path:
        return [B_star, l_path, np.transpose(coeffiecients)]
    else:
        return B_star

def generate_data(is_fixed=True):
    if is_fixed:
        X = np.linspace(1, 15, 15).reshape((5,3))
        y = np.array([-0.2, 3.1, 0.03, 2.3, 10.3])

    else:
        X = np.genfromtxt('../../testutil/data/x_data_1.txt', delimiter=',')
        y = np.genfromtxt('../../testutil/data/y_data_1.txt', delimiter=',')

    # center and scale
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    y = (y - np.mean(y)) / np.std(y)

    n, p = X.shape
    lmda_max = max(abs(np.dot(np.transpose(X), y))) / n
    eps = 1e-2 if n < p else 1e-4
    lmda_min = eps * lmda_max

    return X, y, lmda_max, lmda_min, eps

def lasso_path_one_step_two_lmda():
    X, y, lmda_max, lmda_min, eps = generate_data()
    B_star, l_path, coeff = lasso(X, y, lmda_min,
                                  max_iter=1, path_length=2, return_path=True)
    return coeff

def lasso_path_two_step_two_lmda():
    X, y, lmda_max, lmda_min, eps = generate_data()
    B_star, l_path, coeff = lasso(X, y, lmda_min,
                                  max_iter=2, path_length=2, return_path=True)
    return coeff

def lasso_path_one_step_five_lmda():
    X, y, lmda_max, lmda_min, eps = generate_data()
    B_star, l_path, coeff = lasso(X, y, lmda_min,
                                  max_iter=1, path_length=5, return_path=True)
    return coeff

def lasso_path_ten_step_ten_lmda():
    X, y, lmda_max, lmda_min, eps = generate_data()
    B_star, l_path, coeff = lasso(X, y, lmda_min,
                                  max_iter=10, path_length=10, return_path=True)
    return coeff

def lasso_path_random_one_step_two_lmda():
    X, y, lmda_max, lmda_min, eps = generate_data(False)
    B_star, l_path, coeff = lasso(X, y, lmda_min,
                                  max_iter=1, path_length=2, return_path=True)
    return coeff

def lasso_path_random_ten_step_five_lmda():
    X, y, lmda_max, lmda_min, eps = generate_data(False)
    l_path = np.geomspace(lmda_max, lmda_min, 5)
    model = Lasso(fit_intercept=False, normalize=False,
                  max_iter=10, tol=1e-7,
                  warm_start=True)
    alphas, coefs, _ = model.path(X, y, l1_ratio=1, eps=eps, n_alphas=5,
                                  coef_init=np.zeros(X.shape[1]))
    print(alphas)
    return coefs

if __name__ == '__main__':
    np.set_printoptions(precision=16)
    coeff = lasso_path_random_ten_step_five_lmda()
    print('\n')
    print(coeff)
