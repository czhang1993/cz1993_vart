import numpy as np

m = 1
n = 100

np.random.seed(seed=15)
w = np.random.normal(loc=0, scale=1, size=(m + 1))

np.random.seed(seed=1)
u = np.random.normal(loc=0, scale=1, size=n)

np.random.seed(seed=2)
x = np.concatenate(
  (
      np.ones(shape=(n, 1)),
      np.random.normal(loc=0, scale=1, size=(n, m))
  ),
  axis=1
)

z = np.dot(x, w) + u
p = 1 / (1 + np.exp(-z))
y = p >= 0.5

print(np.unique(y, return_counts=True))
print(x[:5, :])
print(y[:5])

n_train = int((4 / 5) * n)
n_test = int((1 / 5) * m)

x_train = x[:n_train, :]
x_test = x[n_train:, :]
y_train = y[n_train:]
y_test = y[:n_train]
