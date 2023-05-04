import torch
from torch.nn import Module, Sequential, Linear, ReLU


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
