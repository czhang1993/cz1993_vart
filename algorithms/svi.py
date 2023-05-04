import numpy as np
import torch
from torch.nn import Module, Sequential, Linear, ReLU


# define a variational inference class, which is a super class of torch.nn.Module
class VI(Module):
    def __init__(self):
        super().__init__()
        self.q_mu = Sequential(
            Linear(1, 16),
            ReLU(),
            Linear(16, 1)
        )
        self.q_log_var = Sequential(
            Linear(1, 16),
            ReLU(),
            Linear(16, 1)
        )

    def reparameterise(self, mu, log_var):
        sigma = torch.exp(0.5 * log_var) + 1e-5
        eps = torch.rand_like(sigma)
        return mu + sigma + eps

    def forward(self, x):
        mu = self.q_mu(x)
        log_var = self.q_log_var(x)
        return self.reparameterise(mu, log_var), mu, log_var


# define the normal distribution log likelihood function
def ll_normal(y, mu, log_var):
    sigma = torch.exp(0.5 * log_var)
    return -0.5 * torch.log(2 * np.pi * sigma ** 2) - (1 / (2 * sigma ** 2)) * (y - mu) ** 2


# define the evidence lower bound function
def elbo(y, y_pred, mu, log_var):
    log_like = ll_normal(y, mu, log_var)
    # specify the prior as Normal(0, 1), and calculate the log prior based on that
    log_prior = ll_normal(y_pred, 0, torch.log(torch.Tensor(1.)))
    log_p_q = ll_normal(y_pred, mu, log_var)
    return (log_like + log_prior - log_p_q).mean()
  
