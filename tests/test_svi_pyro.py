import math
import os
import torch
import torch.distributions.constraints as constraints
import pyro
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO
import pyro.distributions as dist


adam_params = {
    "lr": 0.0005,
    "betas": (0.90, 0.999)
}
optimizer = Adam(adam_params)

svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

n_steps = 5000
for step in range(n_steps):
    svi.step(data)

# this is for running the notebook in our testing framework
smoke_test = ("CI" in os.environ)
n_steps = 2 if smoke_test else 2000

assert pyro.__version__.startswith("1.8.4")

# clear the param store in case we're in a REPL
pyro.clear_param_store()

# create some data with 6 observed heads and 4 observed tails
data = []
for _ in range(6):
    data.append(torch.tensor(1.0))
for _ in range(4):
    data.append(torch.tensor(0.0))

def model(data):
    alpha0 = torch.tensor(10.0)
    beta0 = torch.tensor(10.0)
    f = pyro.sample("latent_fairness", dist.Beta(alpha0, beta0))
    for i in range(len(data)):
        pyro.sample(
            "obs_{}".format(i),
            dist.Bernoulli(f),
            obs=data[i]
        )

def guide(data):
    alpha_q = pyro.param(
        "alpha_q",
        torch.tensor(15.0),
        constraint=constraints.positive
    )
    beta_q = pyro.param(
        "beta_q",
        torch.tensor(15.0),
        constraint=constraints.positive
    )
    pyro.sample(
        "latent_fairness",
        dist.Beta(alpha_q, beta_q)
    )

adam_params = {
    "lr": 0.0005,
    "betas": (0.90, 0.999)
}
optimizer = Adam(adam_params)

# setup the inference algorithm
svi = SVI(
    model,
    guide,
    optimizer,
    loss=Trace_ELBO()
)

# do gradient steps
for step in range(n_steps):
    svi.step(data)
    if step % 100 == 0:
        print(".", end="")

# grab the learned variational parameters
alpha_q = pyro.param("alpha_q").item()
beta_q = pyro.param("beta_q").item()

# here we use some facts about the Beta distribution
# compute the inferred mean of the coin's fairness
inferred_mean = alpha_q / (alpha_q + beta_q)
# compute inferred standard deviation
factor = beta_q / (alpha_q * (1.0 + alpha_q + beta_q))
inferred_std = inferred_mean * math.sqrt(factor)

print(
    "\nBased on the data and our prior belief, the fairness " +
    "of the coin is %.3f +- %.3f" % (inferred_mean, inferred_std)
)
