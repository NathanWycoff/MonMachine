#!/usr/bin/env python2
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
    def __init__(self, state_size, action_space, eps_l = 0.1, eps_dyn = 0.9, \
        eps_decay = 100, h = 2, h_size = 4, eta = 0.0001, max_err = 10.0, to_merge = []):
        """
        :type state_size: uint
        :param state_size: Dimensionality of the state space
        
        :type action_space: list
        :param action_space: Possible actions the learner is to decide from.
        
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
        """
        #Store the action space
        self.action_space = action_space
        self.action_size = len(action_space)
        
        #Set up epsilon
        self.epsilon_low = eps_l
        self.epsilon_dyn = eps_dyn
        self.exploration_decay = eps_decay
        self.epsilon = eps_l + eps_dyn
        
        #Create ANN
        rng = np.random.RandomState()
        self.ann = artificial_neural_net(rng, state_size, self.action_size, h = h, h_size = h_size, eta = eta, max_err = max_err, to_merge = to_merge )
        
        #Initialize turn counter
        self.turn_counter = 0
        
    #Called on new turn, for now just increments turn counter and update epsilon
    def new_turn(self):
        self.turn_counter += 1
        self.epsilon = self.epsilon_low + pow(self.epsilon_dyn, np.floor(self.turn_counter / self.exploration_decay))
        
    #Do the Q learning update when a new turn is reached.
    def update(self, state_last, state_next, reward):
        
    #Get the value of a stateaction given a state from env and an action from 
    #the actionspace.
    def Q(self, state, action, copy = False):
            
    
    #Take an action according to an epsilon-greedy policy
    def act(self, state):
        #Take a random action with probability epsilon
        if np.random.uniform() < self.epsilon:
            a = self.action_space[np.random.randint(0,len(self.action_space))]
        #Take our current estimate of the best action otherwise.
        else:
            current_state_Q = [self.Q(state, action) for action in self.action_space]
            a = self.action_space[current_state_Q.index(max(current_state_Q))]
                                  
        #Keep our last action to update Q later
        self.last_action = a
        
        return(a)


