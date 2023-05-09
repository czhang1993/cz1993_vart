from torch import Tensor


def model(data):
    pass
  
  
def guide(data):
    # parameter mu squared of the q distribution. 
    q_mu = pyro.param("q_mu", Tensor(?))
    # parameter sigma squared of the q distribution. 
    # the constraint is from torch.distributions.constraints.Constraint
    q_sigsq = pyro.param("q_sigsq", Tensor(?), constraint=constraints.positive)
    
