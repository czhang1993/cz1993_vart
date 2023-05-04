from cz1993_vart import VI, elbo
from cz1993_vart.tests import x_train, x_test, y_train, y_test

epochs = 500

model = VI()
optim = torch.optim.Adam(model.parameters())

for epoch in range(epochs):
    optim.zero_grad()
    y_pred, mu, log_var = model(x)  # need to change
    loss = -elbo(y, y_pred, mu, log_var)  # need to change
    loss.backward()
    optim.step()
    
