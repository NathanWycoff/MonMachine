# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 16:50:57 2016

##TODO:: copied weights never reset.

@author: nathan
"""

#Import the custom built ANN class
import os
os.chdir('/home/nathan/Documents/Documents/Self Study/MonMachine/Agents')
from artificial_neural_net import artificial_neural_net

#Theano is a numerical computation library
import theano
import theano.tensor as T

#Numpy is as well.
import numpy as np


#Q-learner with linear function approx, estimated by minimizing l2 cost using sgd.
class neural_q_learner(object):
    def __init__(self, state_size, action_size, eps_l = 0.1, eps_dyn = 0.9, \
        eps_decay = 100, h = 2, h_size = 4, eta = 0.0001, max_err = 10.0, to_merge = [], M = 100):
        """
        :type state_size: uint
        :param state_size: Dimensionality of the state space
        
        :type action_size: int
        :param action_size: How many actions to choose between?
        
        :type eps_l: float
        :param epsilon: asymptotic low value of epsilon for eps-greedy policy.
        
        :type eps_dyn: float
        :param eps_dyn: shrinking part of epsilon; goes to zero asymptotically
        
        :type eps_decay: float
        :param eps_decay: inverse decay rate for epsilon; smaller is faster decay.
        
        :type h: uint
        :param h: Number of hidden layers in neural net.
        
        :type h_size: uint
        :param h_size: Nodes per layer in hidden layers
        
        :type eta: float
        :param eta: Learning rate for sgd in neural net. If params are exploding, try reducing.
        
        :type max_err: float
        :param max_err: Maximum absolute gradient allowed in sgd. If params are exploding, try reducing.
        
        :type to_merge: list
        :param to_merge: List of lists, containing input dimensions to be confounded at first.
        
        :type M: int
        :param M: Every M updates, reclone the network.
        """
        #Store the action space
        self.action_size = action_size
        
        #Store the clone iterations
        self.M = M
        
        #Set up epsilon
        self.epsilon_low = eps_l
        self.epsilon_dyn = eps_dyn
        self.exploration_decay = eps_decay
        self.epsilon = eps_l + eps_dyn
        
        #Create ANN
        rng = np.random.RandomState()
        self.ann = artificial_neural_net(state_size, self.action_size, h = h, \
            h_size = h_size, eta = eta, max_err = max_err, to_merge = to_merge, rng = rng)
            
        #Clone (deep copy) ANN. Cloned network is used in building the target for sgd.
        self.ann_clone = self.ann.clone()
        
        #Initialize turn counter
        self.turn_counter = 0
        
    def new_turn(self):
        """
        Called on new turn, just increments turn counter and updates epsilon
        """
        self.turn_counter += 1
        self.epsilon = self.epsilon_low + pow(self.epsilon_dyn, np.floor(self.turn_counter / self.exploration_decay))
        
    def update(self, state_last, state_next, reward):
        """
        Do a Q-learning update. 
        
        :type state_last: np.ndarray
        :param state_last: 2d vector of state the decision was made at
        
        :type state_next: np.ndarray
        :param state_next: 2d vector (shape [p,1]) of the state we moved to after decision.
        
        :type reward: float
        :param reward: reward incurred on last state transition.
        """
        #If it's time to 
        if self.turn_counter % self.M == 0:
            self.ann_clone = self.ann.clone()
        
        #Create the sgd target
        max_q = np.max(self.ann_clone.predict(state_next))
        target = np.reshape(reward + max_q, [1,1])
        
        #Do SGD!
        self.ann.grad_on(state_last, target, self.last_action)
        
        
    #Take an action according to an epsilon-greedy policy
    def act(self, state):
        #Take a random action with probability epsilon
        if np.random.uniform() < self.epsilon:
            a = np.random.randint(0, self.action_size)
        #Take our current estimate of the best action otherwise.
        else:
            a = np.argmax(self.ann.predict(state))
                                  
        #Keep our last action to update Q later
        self.last_action = a
        
        return(a)