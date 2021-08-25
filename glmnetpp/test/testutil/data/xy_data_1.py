import numpy as np

n = 100
p = 5
X = np.random.rand(n, p)
y = np.random.rand(n)

np.savetxt('x_data_1.txt', X, delimiter=',')
np.savetxt('y_data_1.txt', y, delimiter=',')
