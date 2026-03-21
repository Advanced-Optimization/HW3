import numpy as np

def mutual_information(x, S, beta):
    """
    BALD / mutual information:
    0.5 * log(1 + (phi(x)^T S phi(x)) / sigma^2 )
    = 0.5 * log(1 + beta * phi(x)^T S phi(x))
    """
    K = S.shape[0]
    phix = phi(x, K).flatten()
    v = phix.T @ S @ phix
    return 0.5 * np.log(1.0 + beta * v)


def trace_reduction(x, S, beta):
    """
    A-optimal design (reduction in trace of S):
    (phi(x)^T S^2 phi(x)) / (sigma^2 + phi(x)^T S phi(x))
    = (phi(x)^T S^2 phi(x)) / (1/beta + phi(x)^T S phi(x))
    """
    K = S.shape[0]
    phix = phi(x, K).flatten()
    Sp = S @ phix
    numerator = phix.T @ (S @ Sp)
    denom = 1.0 / beta + phix.T @ S @ phix
    return numerator / denom


def logdet_reduction(x, S, beta):
    """
    D-optimal design (log-det reduction):
    log det(S) - log det(S')
    = log(1 + beta * phi(x)^T S phi(x))
    """

    K = S.shape[0]
    phix = phi(x, K).flatten()
    v = phix.T @ S @ phix
    return np.log(1.0 + beta * v)


def posterior_covariance_update(x, S, beta):
    """
    Rank-1 posterior covariance update:
    S' = S - S phi(x) phi(x)^T S / (sigma^2 + phi(x)^T S phi(x))
       = S - S phi(x) phi(x)^T S / (1/beta + phi(x)^T S phi(x))
    """
    K = S.shape[0]
    phix = phi(x, K).flatten()
    denom = 1.0 / beta + phix.T @ S @ phix
    Sphix = S @ phix
    return -np.trace(S - np.outer(Sphix, Sphix) / denom)