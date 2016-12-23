# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 22:13:10 2016

@author: Nathan Wycoff
A Multi-layer perceptron neural net. Inspired by http://deeplearning.net/tutorial/mlp.html
and some code portions are copied directly.

Uses Theano to make everything easy and beautiful.
"""

#Theano is a numerical computation library
import theano 
import theano.tensor as T

#Numpy is as well.
import numpy as np


class artifical_neural_net(object):
    """
    ANN/Multilayer Perceptron. A set of Layers and functions to train them.
    """
    def __init__(self, rng, in_size, out_size, h, h_size, eta = 0.01, max_err = 10.0):
        """
        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights
        
        :type in_size: int
        :param len_in: Dimensionality of input space
        
        :type out_size: int
        :param len_out: Dimensionality of output space
        
        :type h: uint
        :param h: Number of hidden layers
        
        :type h_size: uint
        :param h_size: Nodes per hidden layer
        
        :type eta: double
        :param eta: non-negative learning rate; premultiplies gradient in sgd
        
        :type max_err: double
        :param max_err: max_err is the absolute maximum error gradient; non-neg
        """
        X_var = T.dmatrix('X')#Represents input
        y_var = T.dmatrix('y')#Represents output
        
        ##Create and store the layers
        self.layers = []#Storage for layers
        len_in = in_size#The input of the first layer is the dim of the input space
        ID = 0#ID for each layer
        
        #Hidden layers
        in_vec = X_var#Start the pipeline
        for i in range(h):
            #Create a layer and append it.
            self.layers.append(Layer(rng, in_vec, len_in, h_size, ID, T.nnet.nnet.sigmoid))
            
            #From 1st iter on, input will be hidden size.
            len_in = h_size
            
            #Append the output of the last layer to the calculation
            in_vec = self.layers[-1].output
            
            #Increment ID
            ID += 1
            
        #Output layer
        self.layers.append(Layer(rng, in_vec, len_in, out_size, ID))
        #Use this function to predict vals.
        self.predict = theano.function([X_var], self.layers[-1].output)
        
        ##Prepare cost function for SGD
        cost = T.mean(T.square(self.layers[-1].output - y_var))
        grads = [T.grad(cost, layer.w) for layer in self.layers]
        
        #Prepare the error truncation function, limits the max absolute gradient.
        trunc_err = lambda x: x if T.le(max_err, abs(x)) else x / abs(x) * max_err
        
        #What's going to change when we do sgd, and how?
        updates = [
            (layer.w, layer.w - eta * trunc_err(grad))
                for grad, layer in zip(grads, self.layers)
            ]
            
        self.sgd = theano.function(inputs = [X_var,y_var], updates = updates)

        
    def grad_on(self, x, y):
        """
        Does sgd on target and updates params
        
        These params MUST be matrices (2D numpy arrays), even if the space is
        one dimensional, should be [n,1], not [n,].
        
        :type x: Input matrix (regressors)
        :type y: output matrix (response)
        """
        
        self.sgd(x, y)
        

class Layer(object):
    """
    Respresents one layer of a neural net, based off the deeplearning tutorial 
    at http://deeplearning.net/tutorial/mlp.html
    """
    def __init__(self, rng, in_vec, len_in, len_out, ID = '0', activation = lambda x: x):
        """
        :type in_vec: theano.tensor.TensorType
        :param in_vec: symbolic variable that describes the input of the
        architecture
        
        :type len_in: int
        :param len_in: Represents dimensionality of input space of this layer
        
        :type len_out: int
        :param len_out: Represents dimensionality of output space of this layer
        
        :type ID: int
        :param ID: ID of this layer, for debugging purposes.
        
        :type activation: function
        :param activation: Nonlinearity to be applied at this layer. Default is 
        identity.
        """
        
        #Initialize this layer's weights
        self.w = theano.shared(np.asarray(rng.normal(size = [len_in, len_out])), 'w' + str(ID))
        self.z = T.dot(in_vec, self.w)
        self.output = activation(self.z)
        
        #Store dims/ID
        self.ID = ID
        self.len_in = len_in
        self.len_out = len_out
        
        
    def __str__(self):
        return("Layer ID: " + str(self.ID) + " of dim " + str(self.len_in) + "x" + str(self.len_out))
        
    __repr__ = __str__