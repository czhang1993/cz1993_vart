import torch
from torch.optim import Adam
from algorithms.svi import VI, elbo
from tests.data import x_train, y_train

x_train = torch.Tensor(x_train)
y_train = torch.Tensor(y_train)

epochs = 500

model = VI()
optim = Adam(model.parameters())

for epoch in range(epochs):
    optim.zero_grad()
    y_pred, mu, log_var = model(x_train)  # need to change
    loss = -elbo(y_train, y_pred, mu, log_var)  # need to change
    loss.backward()
    optim.step()
