#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 16:50:57 2016

@author: nathan
"""

##TODO:: copied weights never reset.

import numpy as np
#import theano

#Q-learner with linear function approx, estimated by minimizing l2 cost using sgd.
class linear_q_learner(object):
    #state_size is an integer stating the dimensionality of the state-space, including bias unit.
    #state_type is a vector of length state_size, with entry 'C' in cell i if the ith dimension is continiuous and 'D' if it is categorical.
    def __init__(self, state_size, ep_l = 0.1, ep_dyn = 0.9, exploration_decay = 1, \
                 eta = 0.1, learning_decay = 10, max_err = 10):
        #Possible Actions to take each turn
        self.action_space = ['1','2','3']
        
        #Store state space dimension.
        self.state_size = state_size

        #Initialize Weight vectors.
        self.w = np.random.normal(size=state_size * len(self.action_space), scale = 0.05)
        self.w_c = np.copy(self.w)
        
        self.alpha = 0.1#Learning Rate
        
        #Epsilon controls exploration
        self.epsilon_low = ep_l#Lower bound on exploration rate
        self.epsilon_dyn = ep_dyn#Dynamic Part of Exploration rate
        self.epsilon = self.epsilon_low + self.epsilon_dyn
        self.exploration_decay = exploration_decay#Epsilon_dyn's exponent increases every how many turns?
        
        #eta controls the learning rate
        self.eta = 0.1#Initial Learning Rate
        self.learning_decay = learning_decay
        
        #How many turns have elapsed?
        self.turn_counter = 0
        
        #Error clipping 
        self.clip_err = lambda x: x if abs(x) < max_err else x / abs(x) * max_err

    #Called on new turn, for now just increments turn counter and update epsilon
    def new_turn(self):
        self.turn_counter += 1
        self.epsilon = self.epsilon_low + pow(self.epsilon_dyn, np.floor(self.turn_counter / self.exploration_decay))
        self.eta = min(self.eta, pow(self.eta, np.floor(self.turn_counter / self.learning_decay)))
    
    #Do the Q learning update when a new turn is reached.
    def update(self, state_last, state_next, reward):
        #Get the design matrix row from the state string
        
        max_a = max([self.Q(state_next, action, copy = True) for action in self.action_space])
        prev_guess = self.Q(state_last, self.last_action)
        
        #Do Stochastic Gradient Descent
        #Only update weights corresponding to the chosen action.
        #Next line gets space of possible starting indices.
        ind_space = range(0, self.state_size * len(self.action_space), self.state_size)
        #Select the starting index for the action we've selected.
        act_ind = ind_space[self.action_space.index(self.last_action)]
        
        #Get the design matrix row wrt we are taking the gradient.
        x_vec = self.get_sa(state_last, self.last_action)
        
        delta = prev_guess - (reward + max_a)
        
        for i in range(self.state_size):
            j = act_ind + i
            self.w[j] = self.w[j] - self.clip_err(self.eta * 2 * x_vec[j] * delta)

        #self.w[act_ind + 0] = self.w[act_ind + 0] - self.clip_err(self.eta * (2 * (prev_guess - (reward + max_a))))
        #self.w[act_ind + 1] = self.w[act_ind + 1] - self.clip_err(self.eta * (2 * health * (prev_guess - (reward + max_a))))
        #self.w[act_ind + 2] = self.w[act_ind + 2] - self.clip_err(self.eta * (2 * (prev_guess - (reward + max_a))))
        #self.w[act_ind + 3] = self.w[act_ind + 3] - self.clip_err(self.eta * (2 * (prev_guess - (reward + max_a)))
        
        #self.Q[sa_last] = self.Q[sa_last] + self.alpha * (reward + max_a - self.Q[sa_last])

    #Given a state and action, get the state action vector in the design matrix.
    ## NEEDS GENERALIZAITION
    def get_sa(self, state, action):
        
        #The State-Action design matrix is just the state vector in different places
        #depending on the action.
        if action == '1':
            sa = np.append(state[:], np.zeros([len(state)*2,1]))
        if action == '2':
            sa = np.zeros([len(state),1])
            sa = np.append(sa, state[:])
            sa = np.append(sa, np.zeros([len(state),1]))
        if action == '3':
            sa = np.append(np.zeros([len(state)*2,1]), state[:])
            
        return(sa)
    
    #Get the value of a stateaction given a state from env and an action from 
    #the actionspace.
    def Q(self, state, action, copy = False):
        weights = self.w_c if copy else self.w
        x = self.get_sa(state, action)
        return(np.dot(x, weights))
            
    
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

#l = linear_q_learner(2)
#l.get_sa
