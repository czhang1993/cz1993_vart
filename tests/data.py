import numpy as np

m = 3
n = 100

np.random.seed(seed=0)
w = np.random.normal(loc=0, scale=1, size=(2 * m + 2, 1))
w1 = w[:m, :]
w2 = w[m:2 * m, :]
w3 = w[2 * m:, :]

np.random.seed(seed=1)
u = np.random.normal(loc=0, scale=1, size=(2 * n, 1))
u1 = u[:n, :]
u2 = u[n:2 * n, :]

np.random.seed(seed=2)
x = np.concatenate(
  (np.ones(shape=(n, 1)),
   np.random.normal(loc=0, scale=1, size=(n, m - 1))),
  axis=1
)

z1 = np.concatenate(
  (np.ones(shape=(n, 1)),
   np.dot(x, w1) + u1),
  axis=1
)
z2 = np.dot(x, w2) + np.dot(z1, w3) + u2

p = 1 / (1 + np.exp(-z2))

y = p >= 0.5
y = y.astype(np.float64)

d = np.concatenate(
  (
    x[:, 1:],
    z1[:, 1:],
    y
  ), 
  axis=1
)

# print(np.unique(y, return_counts=True))
# print(d[:5, :])

dims = d.shape
dim1 = dims[0]
dim2 = dims[1]

n_train = int((4 / 5) * dim1)
n_test = int((1 / 5) * dim1)

x_train = d[:n_train, :dim2 - 1]
x_test = d[n_train:, :dim2 - 1]
y_train = d[:n_train, dim2 - 1:]
y_test = d[n_train:, dim2 - 1:]
