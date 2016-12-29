# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 18:14:37 2016

@author: Nathan Wycoff

A battery of tests for the artificial neural net class.

They involve training several of neural nets to convergence. It takes time to run (1/2 hr on my laptop i7).
"""

import unittest
import numpy as np

import theano
import theano.tensor as T

#Import the custom built ANN class YOU SHOULD CHANGE THIS LINE
import sys
sys.path.append('/home/nathan/Documents/Documents/Self Study/MonMachine/Agents/')
from artificial_neural_net import artificial_neural_net

def gen_data(rng, n, p, q):
    """
    Generate data (X's, y's, w's) as desired, for use in training neural net.
    
    :type rng: numpy.random.RandomState
    :param rng: An rng state to generate random variates from
    
    :type n: int
    :param n: Number of samples.
    
    :type p: int
    :param p: Dimensionality of the regressor space, including intercept.
    
    :type q: int
    :param q: Dimensionality of the response space.
    """
    
    #Create regressors
    X = rng.normal(size=[n, p-1])
    X = np.concatenate([np.ones([n, 1]), X], axis = 1)
    
    #Generate betas
    w = rng.normal(size = [p,q])
    
    #Generate response
    sigma= 0.1
    y = np.dot(X, w) + rng.normal(size = [n,q], scale = sigma)
    
    return(X, w, y)

#Testing Parameters
iters = 2000000
thresh = 0.2
rng_seed = 1
class TestStringMethods(unittest.TestCase):
    
    def test_linear_regression(self):
        """
        The problem is univariate response, no mergers, no hidden layers.
        
        Test that the estimated coefs are close enough to the truth
        """
        
        #Generate some data
        p = 20
        q = 1
        n = 1000
        rng = np.random.RandomState(rng_seed)
        X, w, y = gen_data(rng = rng, n = n, p = p, q = q)
        
        #Get ann
        ann = artificial_neural_net(in_size = p, out_size = q, h = 0, h_size = 0, eta = 0.0001, lambda_l1 = 0, lambda_l2 = 0)
        
        
        #Do sgd
        for i in range(iters):
            #Get the current datum
            j = i % n
            x_i = np.reshape(X[j, :], [1,p])
            y_i = np.reshape(y[j,0], [1,1])
    
            #Get the cost gradient
            ann.grad_on(x_i, y_i, 0)
        
        #See if we're close enough
        self.assertGreater(thresh, np.linalg.norm(ann.layers[0].w.get_value() - w, np.float('inf')))
        
        
    def test_neural_net(self):
        """
        The problem is univariate response, no mergers, with 2 hidden layers.
        
        Test that our train MSE is small enough
        """
        
        #Generate some data
        p = 20
        q = 1
        n = 1000
        rng = np.random.RandomState(rng_seed)
        X, w, y = gen_data(rng = rng, n = n, p = p, q = q)
        
        #Get ann
        ann = artificial_neural_net(in_size = p, out_size = q, h = 2, h_size = 4, eta = 0.0001, lambda_l1 = 0, lambda_l2 = 0)
        
        
        #Do sgd
        for i in range(iters):
            #Get the current datum
            j = i % n
            x_i = np.reshape(X[j, :], [1,p])
            y_i = np.reshape(y[j,0], [1,1])
    
            #Get the cost gradient
            ann.grad_on(x_i, y_i, 0)
        
        #See if we're close enough
        mse = np.mean(np.square(ann.predict(X) - y))
        self.assertGreater(thresh, mse)
        
    def test_multiple_output_function(self):
        """
        The problem is multivariate response, no mergers, with no hidden layers.
        
        Test that only what should change does change.
        """
        
        #Generate some data
        p = 2
        q = 2
        n = 1000
        rng = np.random.RandomState(rng_seed)
        X, w, y = gen_data(rng = rng, n = n, p = p, q = q)
        
        #Get ann
        ann = artificial_neural_net(in_size = p, out_size = q, h = 0, h_size = 0, eta = 0.0001, lambda_l1 = 0, lambda_l2 = 0)
        
        
        #Do sgd to learn the first response coefs, make sure it doesn't touch the second
        g = 0
        
        first_coefs_init = ann.layers[0].w.get_value()[:,0]
        second_coefs_init = ann.layers[0].w.get_value()[:,1]
        
        for i in range(500):
            #Get the current datum
            j = i % n
            x_i = np.reshape(X[j, :], [1,p])
            y_i = np.reshape(y[j,g], [1,1])
    
            #Get the cost gradient
            ann.grad_on(x_i, y_i, g)
        
        #Make sure that the ones we sgd'd on changed and the others didn't.
        self.assertTrue(np.alltrue(first_coefs_init != ann.layers[0].w.get_value()[:,0]))
        self.assertTrue(np.alltrue(second_coefs_init == ann.layers[0].w.get_value()[:,1]))
        
        
        ## Same thing but for the other columns
        g = 1
        
        first_coefs_init = ann.layers[0].w.get_value()[:,0]
        second_coefs_init = ann.layers[0].w.get_value()[:,1]
        
        for i in range(500):
            #Get the current datum
            j = i % n
            x_i = np.reshape(X[j, :], [1,p])
            y_i = np.reshape(y[j,g], [1,1])
    
            #Get the cost gradient
            ann.grad_on(x_i, y_i, g)
        
        #Make sure that the ones we sgd'd on changed and the others didn't.
        self.assertTrue(np.alltrue(first_coefs_init == ann.layers[0].w.get_value()[:,0]))
        self.assertTrue(np.alltrue(second_coefs_init != ann.layers[0].w.get_value()[:,1]))
        
        
    def test_multiple_output_accuracy(self):
        """
        The problem is multivariate response, no mergers, with 2 hidden layers.
        
        Test that the predictions get good enough MSE.
        """
        
        #Generate some data
        p = 2
        q = 2
        n = 1000
        rng = np.random.RandomState(rng_seed)
        X, w, y = gen_data(rng = rng, n = n, p = p, q = q)
        
        #Get ann
        ann = artificial_neural_net(in_size = p, out_size = q, h = 2, h_size = 4, eta = 0.0001, lambda_l1 = 0, lambda_l2 = 0)
        
        for i in range(iters):
            for g in range(q):
                #Get the current datum
                j = i % n
                x_i = np.reshape(X[j, :], [1,p])
                y_i = np.reshape(y[j,g], [1,1])
        
                #Get the cost gradient
                ann.grad_on(x_i, y_i, g)
                
        #See if we're close enough
        mse = np.mean(np.square(ann.predict(X) - y))
        self.assertGreater(thresh, mse)
        
        
    def test_merging(self):
        """
        The problem is multivariate response, some mergers, with 2 hidden layers.
        
        Test that the predictions get good enough MSE.
        """
        
        #Generate some data
        p = 20
        q = 2
        n = 1000
        rng = np.random.RandomState(rng_seed)
        X, w, y = gen_data(rng = rng, n = n, p = p, q = q)
        
        #Get ann with mergers
        to_merge = [[0,1,2,3,4],[5,6,7,8,9],[10,11,12,13,14],[15,16,17,18,19]]
        ann = artificial_neural_net(in_size = p, out_size = q, h = 2, h_size = 4, eta = 0.0001, to_merge = to_merge, lambda_l1 = 0, lambda_l2 = 0)
        
        self.assertTrue(np.shape(ann.layers[0].w.get_value())[0] == 4)
        
        for i in range(500):
            for g in range(q):
                #Get the current datum
                j = i % n
                x_i = np.reshape(X[j, :], [1,p])
                y_i = np.reshape(y[j,g], [1,1])
                
                #Get the cost gradient
                ann.grad_on(x_i, y_i, g)
        
        #Split up some of merges
        ann.split(0)
        ann.split(1)
        ann.split(2)
        ann.split(3)
        
        self.assertTrue(np.shape(ann.layers[0].w.get_value())[0] == 20)
        
        for i in range(iters):
            for g in range(q):
                #Get the current datum
                j = i % n
                x_i = np.reshape(X[j, :], [1,p])
                y_i = np.reshape(y[j,g], [1,1])
                
                #Get the cost gradient
                ann.grad_on(x_i, y_i, g)
        
        
        #See if we're close enough
        mse = np.mean(np.square(ann.predict(X) - y))
        self.assertGreater(thresh, mse)
            
        
if __name__ == '__main__':
    unittest.main()