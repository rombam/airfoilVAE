import numpy as np
from scipy.linalg import sqrtm

def compute_fid(act1, act2):
    """ Calculates the FID between two distributions of data. 
    Inputs:
    - act1: numpy array containing the first dataset. One row per entry, one column per feature.
    - act1: numpy array containing the second dataset. One row per entry, one column per feature.
    Outputs:
    - fid: float containing the FID between the two distributions.
    """
	# Calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    # Calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)
    # Calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # Check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # Calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid