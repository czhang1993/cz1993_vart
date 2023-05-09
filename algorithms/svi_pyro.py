from torch import Tensor
from pyro import sample
from pyro.distributions.dist import Beta, Bernoulli


def model(data):
    alpha0 = Tensor(10.0)
    beta0 = Tensor(10.0)
    f = sample(
        "latent_fairness",
        Beta(alpha0, beta0)
    )
    for i in range(len(data)):
        sample(
            "obs_{}".format(i),
            Bernoulli(f),
            obs=data[i]
        )
        
def guide(data):
    alpha_q = pyro.param(
        "alpha_q",
        Tensor(15.0),
        constraint=constraints.positive
    )
    beta_q = pyro.param(
        "beta_q",
        Tensor(15.0),
        constraint=constraints.positive
    )
    pyro.sample(
        "latent_fairness",
        Beta(alpha_q, beta_q)
    )
