import numpy as np

# Link: https://towardsdatascience.com/regularized-linear-regression-models-dcf5aa662ab9
def lasso(X, y, l1, tol=1e-6, max_iter=100000, path_length=100, return_path=False):
    #X = np.hstack((np.ones((len(X), 1)), X))
    m, n = np.shape(X)
    B_star = np.zeros((n))
    l_max = max(list(abs(np.dot(np.transpose(X[:, 1:]), y)))) / m
    # At or above l_max, all coefficients (except intercept) will be brought to 0
    if l1 >= l_max:
        return np.append(np.mean(y), np.zeros((n - 1)))
    l_path = np.geomspace(l_max, l1, path_length)
    print(l_path)
    coeffiecients = np.zeros((len(l_path), n))
    for i in range(len(l_path)):
        for _ in range(max_iter):
            B_s = B_star
            for j in range(n):
                k = np.where(B_s != 0)[0]
                update = (1/m)*((np.dot(X[:,j], y)- \
                                np.dot(np.dot(X[:,j], X[:,k]), B_s[k]))) + \
                                B_s[j]
                B_star[j] = (np.sign(update) * max(abs(update) - l_path[i], 0))
            if np.all(abs(B_s - B_star) < tol):
                break
        coeffiecients[i, :] = B_star
    if return_path:
        return [B_star, l_path, np.transpose(coeffiecients)]
    else:
        return B_star

X = np.linspace(1, 15, 15).reshape((5,3))
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
y = np.array([-0.2, 3.1, 0.03, 2.3, 10.3])
y = (y - np.mean(y)) / np.std(y)
n, p = X.shape
lmda_max = max(abs(np.dot(np.transpose(X), y))) / n
lmda_min = 1e-4 * lmda_max

def lasso_path_one_step():
    B_star, l_path, coeff = lasso(X, y, lmda_min,
                                  max_iter=1, path_length=2, return_path=True)
    print(coeff)

if __name__ == '__main__':
    np.set_printoptions(precision=16)
    print(lmda_min)
    lasso_path_one_step()
