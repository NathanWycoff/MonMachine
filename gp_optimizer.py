# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 11:28:02 2016

@author: Nathan Wycoff

Some functions for optimization by GP. I use these  to optimize my neural net
hyperparameters.

Requries theano >= 0.9 (for automatic differentiation of left matrix divides.)
"""

import numpy as np

import theano
import theano.tensor as T
import theano.tensor.slinalg as LA
import scipy.stats as stats

class Gaussian_Process(object):
    """
    An implementation of a Gaussian Process for Regression problems.
    """
    def __init__(self, tau2_0 = 0.1, sigma2_0 = 0.1, l_0 = 0.1, eta = 0.1, debug = 1):
        """
        :type sigma_0: float
        :param sigma_0: starting value for variance.
        
        :type l_0: float
        :param l_0: starting value for length scale.
        
        :type eta: float
        :param eta: learning rate
        
        :type debug: int
        :param debug: verbosity
        """
        
        print "GP Initing..." if debug > 0 else 0
    
        ##################################################
        #### Prepare the -loglik gradient descent
    
        ##Init the shared vars
        X = T.dmatrix('X')
        f = T.dmatrix('f')
        self.tau2 = theano.shared(tau2_0)
        self.l = theano.shared(l_0)
        self.sigma2 = theano.shared(sigma2_0)
        
        #Make the covar matrix
        K = self.covFunc(X, X, self.l)
        
        #Get a numerically safe decomp
        L = LA.cholesky(K + self.tau2 * T.identity_like(K))
        
        #Calculate the weights for each of the training data; predictions are a weighted sum.
        alpha = LA.solve(T.transpose(L), LA.solve(L, f))
        
        ##Calculate - log marginal likelihood
        nloglik = - T.reshape(-0.5 * T.dot(T.transpose(f), alpha) - T.sum(T.log(T.diag(L))), [])
        
        #Get grad
        grads = [T.grad(nloglik, self.tau2), T.grad(nloglik, self.l), T.grad(nloglik, self.sigma2)]
        
        #Updates, make sure to keep the params positive
        updates = [(var, T.max([var - eta * grad, 0.1])) for var, grad in zip([self.tau2, self.l, self.sigma2], grads)]
        
        self._gd = theano.function(inputs = [X, f], updates = updates)
        
        print "Done" if debug > 0 else 0
    
    def fit(self, X, f, thresh = 0.1):
        """
        Estimate hyperparams using gradient descent on negative log martinal likelihood.
        
        :type X: np.ndarray
        :param X: matrix of train locations
        
        :type f: np.ndarray
        :param f: vector of training responses [k,1]
        
        :type thresh: float
        :param thresh: when to stop gradient descent?
        """
        

        while True:
            s_0 = self.tau2.get_value()
            l_0 = self.l.get_value()
            sigma2 = self.sigma2.get_value()
            
            self._gd(X, f)
            
            if max(abs(s_0 - self.tau2.get_value()), abs(l_0 - self.l.get_value()),\
                            abs(sigma2 - self.sigma2.get_value())) < thresh:
                break
            
            
    def covFunc(self, x1, x2, l, method = 'SE', mode = 'cross'):
        '''
        Factorization Implementation of distance function.
        
        This method copied directly from:
        
        https://github.com/shenxudeu/gp_theano/blob/master/gptheano_model.py
        
        Thanks Shen!
        '''
        if method == 'SE':
            ell = T.exp(l)
            sf2 = T.exp(2.*self.sigma2)
            if mode == 'cross':
                xx = T.sum((x1/ell)**2,axis=1).reshape((x1.shape[0],1))
                xc = T.dot((x1/ell), (x2/ell).T)
                cc = T.sum((x2/ell)**2,axis=1).reshape((1,x2.shape[0]))
                dist = xx - 2*xc + cc
            elif mode == 'self_test':
                tmp = T.sum(x1,axis=1).reshape((x1.shape[0],1))
                dist = T.zeros_like(tmp)
            else:
                raise NotImplementedError
            k = sf2 * T.exp(-dist/2)
        else:
            raise NotImplementedError
        return k
        
    
    def get_posterior(self, X, XX, f):
        """
        Compute the posterior mean of a gaussian process with SE covariance func.
        
        The reason I have this code twice is that i'm at present too lazy to replace
        this with theano code.
        
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
        sigma_sq = self.tau2.get_value()
        
        #Heuristic, just put the length-scale at the mean of the distances.
        l = self.l.get_value()
            
        #Get a list of the rows
        X_row = [X[i,:] for i in range(np.shape(X)[0])]
        XX_row = [XX[i,:] for i in range(np.shape(XX)[0])]
        
        #Calculate K, the covariance matrix
        k = lambda x,y: np.exp(-0.5 * np.linalg.norm(x - y, 2.0) / l)
        K = np.array([[k(x1,x2) for x1 in X_row] for x2 in X_row])
        KK = np.array([[k(xx,x) for xx in XX_row] for x in X_row])
        #kkk = [k(xx, xx) for xx in XX_row]
        
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
        
    def get_expected_improvement(self, mus, sigs, current_max):
        """
        Calculate expected improvement as shown in http://www.ressources-actuarielles.net/ext/isfa/1226.nsf/9c8e3fd4d8874d60c1257052003eced6/f84f7ac703bf5862c12576d8002f5259/$FILE/Jones98.pdf
        I optimize for a minimum as that's what they do in the paper (so we flip the sign of mu/current_max)
        
        The optimal choice is the one with the biggest EI.
        
        :type mus: np.ndarray
        :param mus: vector of means on testing grid from GP posterior
        
        :type sigs: np.ndarray
        :param sigs: vector of variances on testing grid from GP posterior
        
        :type current_max: float
        :param current_max: Value of currently discovered maximum.
        """
        
        current_min = -current_max#Flip the max sign
        mus = - mus#Flip the sign of the means
        sds = np.sqrt(sigs)#These have the same variance, get their sqrt.
        
        #Calculate the expected improvement (ei)
        PHI = stats.norm.cdf
        phi = stats.norm.pdf
        ei = (current_min - mus) * PHI((current_min - mus) / sds) + sds * phi((current_min - mus) / sds)
        
        return(ei)