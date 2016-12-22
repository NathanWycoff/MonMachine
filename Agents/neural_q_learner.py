#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 16:50:57 2016

@author: nathan
"""

##TODO:: copied weights never reset.
#Theano is a numerical computation library
import theano 
import theano.tensor as T

#Numpy is as well.
import numpy as np


#Q-learner with linear function approx, estimated by minimizing l2 cost using sgd.
class linear_q_learner(object):
    #state_size is an integer stating the dimensionality of the state-space, including bias unit.
    #state_type is a vector of length state_size, with entry 'C' in cell i if the ith dimension is continiuous and 'D' if it is categorical.
    def __init__(self, state_size, ep_l = 0.1, ep_dyn = 0.9, exploration_decay = 1, \
                 eta = 0.1, learning_decay = 10, max_err = 10):
        

    #Called on new turn, for now just increments turn counter and update epsilon
    def new_turn(self):
        self.turn_counter += 1
        self.epsilon = self.epsilon_low + pow(self.epsilon_dyn, np.floor(self.turn_counter / self.exploration_decay))
        self.eta = min(self.eta, pow(self.eta, np.floor(self.turn_counter / self.learning_decay)))
    
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


