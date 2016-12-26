# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 11:28:02 2016

@author: Nathan Wycoff

Some functions for optimization by GP. I use these  to optimize my neural net
hyperparameters.

"""

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import norm

def gp_posterior(X, XX, f):
    """
    Compute the posterior mean of a gaussian process with SE covariance func.
    
    Uses Algorithm 2.1 from Rasmussen and Williams.
    
    Returns predictive means and variances as np.ndarrays
    
    :type X: np.ndarray
    :param X: matrix (2darray) of train locations.
    
    :type XX: np.ndarray
    :param XX: matrix (2darray) of test locations.
    
    :type f: np.ndarray
    :param f: vector (2d [k,1] array) of train responses.
    """
    
    #Small variance just for numerical stability reasons
    sigma_sq = 0.001
    
    #Heuristic, just put the length-scale at the mean of the distances.
    l = np.mean(pdist(np.vstack([X, XX])))
        
    #Get a list of the rows
    X_row = [X[i,:] for i in range(np.shape(X)[0])]
    XX_row = [XX[i,:] for i in range(np.shape(XX)[0])]
    
    #Calculate K, the covariance matrix
    k = lambda x,y: np.exp(-0.5 * np.linalg.norm(x - y, 1) / l)
    K = np.array([[k(x1,x2) for x1 in X_row] for x2 in X_row])
    KK = np.array([[k(xx,x) for xx in XX_row] for x in X_row])
    kkk = [k(xx, xx) for xx in XX_row]
    
    #Calculate mean
    L = np.linalg.cholesky(K + sigma_sq * np.identity(np.shape(K)[0]))
    alpha = np.linalg.solve(np.transpose(L), np.linalg.solve(L, f))
    mu = np.dot(np.transpose(KK), alpha)
    
    #Calculate variance the fast way; have a bug somewhere
    #vs = [np.linalg.solve(L, KK[i,:]) for i in range(np.shape(KK)[0])]
    #sig = np.reshape(np.array([kk - np.dot(np.transpose(v), v) for kk, v in zip(kkk, vs)]), [len(vs),1])
    
    #This method works, but is slower.
    sig = 1 - np.diag(np.dot(np.dot(np.transpose(KK), np.linalg.inv(K + sigma_sq * np.identity(np.shape(K)[0]))), KK))

    
    return(mu, sig)
    
def get_expected_improvement(mus, sigs, current_max):
    """
    Calculate expected improvement as shown in http://www.ressources-actuarielles.net/ext/isfa/1226.nsf/9c8e3fd4d8874d60c1257052003eced6/f84f7ac703bf5862c12576d8002f5259/$FILE/Jones98.pdf
    I optimize for a minimum as that's what they do in the paper (so we flip the sign of mu/current_max)
    
    The optimal choice is the one with the biggest EI.
    
    :type mus: np.ndarray
    :param mus: vector of means on testing grid
    
    :type sigs: np.ndarray
    :param sigs: vector of variances on testing grid
    
    :type current_max: float
    :param current_max: Value of currently discovered maximum.
    """
    
    current_min = -current_max#Flip the max sign
    mus = - mus#Flip the sign of the means
    sds = np.sqrt(sigs)#These have the same variance, get their sqrt.
    
    #Calculate the expected improvement (ei)
    phi = norm.cdf
    ei = (current_min - mus) * phi((current_min - mus) / sds) + sds * phi((current_min - mus) / sds)
    
    return(ei)