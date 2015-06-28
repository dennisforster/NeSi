import numpy as np
import scipy.special

def poisson_function(k, mu):
    pmf = np.exp(-mu) * mu ** k / scipy.special.gamma(k+1)
    return pmf

def log_poisson_function(k, mu):
    pmf = -mu + k * np.log(mu) - scipy.special.gammaln(k+1)
    return pmf