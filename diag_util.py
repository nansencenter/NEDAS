import numpy as np

def rmse():
    return

def sprd():
    return

def sample_correlation(x1, x2):
  x1_mean = np.mean(x1)
  x2_mean = np.mean(x2)
  x1p = x1 - x1_mean
  x2p = x2 - x2_mean
  cov = np.sum(x1p * x2p)
  x1_norm = np.sum(x1p ** 2)
  x2_norm = np.sum(x2p ** 2)
  norm = x1_norm * x2_norm
  corr = cov/np.sqrt(norm)
  return corr

