# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 16:50:57 2016

Implementation of a Q learner with feedforward neural net function approximator.

Inspired by Human Level Control through Deep Reinforcement Learning (Volodymyr et al 2015)

Does the same sgd update as in that paper, and does Experience Replay as described therein.

Starts experience replay only when it has a full memory.

@author: Nathan Wycoff
"""


#Import the custom built ANN class YOU SHOULD CHANGE THIS LINE
import sys
sys.path.append('/home/nathan/Documents/Documents/Self Study/MonMachine/')
from artificial_neural_net import artificial_neural_net

#Theano is a numerical computation library
import theano
import theano.tensor as T

#Just needed to sample a list
import random

#Numpy is as well.
import numpy as np


#Q-learner with linear function approx, estimated by minimizing l2 cost using sgd.
class neural_q_learner(object):
    def __init__(self, state_size, action_size, eps_l = 0.1, eps_dyn = 0.9, \
        eps_decay = 100, h = 2, h_size = 4, eta = 0.0001, max_err = 10.0, to_merge = [], \
        M = 100, lambda_l1 = 0.1, lambda_l2 = 0.1, mem_size = 5000, replay_size = 100):
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
        
        :type lambda_l1: float
        :param lambda_l1: nonneg coefficient for l1 regularization term
        
        :type lambda_l2: float
        :param lambda_l2: nonneg coefficient for l2 regularization term
        
        :type mem_size: int
        :param mem_size: How many memories to store for experience replay?
        
        :type replay_size: int
        :param replay_size: How many experience replays to do after each turn?
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
            h_size = h_size, eta = eta, max_err = max_err, to_merge = to_merge, \
            rng = rng, lambda_l1 = lambda_l1, lambda_l2 = lambda_l2)
            
        #Clone (deep copy) ANN. Cloned network is used in building the target for sgd.
        self.ann_clone = self.ann.clone()
        
        #Initialize turn counter
        self.turn_counter = 0
        
        #Set to false if you want to set epsilon to zero.
        self.exploring = True
        
        #For experience replay
        self.memory = []
        self.mem_size = mem_size
        self.replay_size = replay_size
        
    def new_turn(self):
        """
        Called on new turn, increments turn counter, update epsilon, and clone the network.
        
        Also do the experience replay
        """
        
        self.turn_counter += 1
        
        #Set epsilon
        if self.exploring:
            self.epsilon = self.epsilon_low + pow(self.epsilon_dyn, np.floor(self.turn_counter / self.exploration_decay))
        else:
            self.epsilon = 0
        
        #If it's time to reclone the network.
        if self.turn_counter % self.M == 0:
            self.ann_clone = self.ann.clone()
            
        ##Experience Replay
        #TODO: Make this asyncronous
        if len(self.memory) == self.mem_size:
            experience_sample = [self.memory[i] for i in np.random.choice(len(self.memory), self.replay_size)]
            for exp in experience_sample:
                self.update(exp[0], exp[1], exp[2], exp[3], replay = True)
            
        
    def update(self, state_last, state_next, reward, action = None, replay = False):
        """
        Do a Q-learning update. 
        
        :type state_last: np.ndarray
        :param state_last: 2d vector of state the decision was made at
        
        :type state_next: np.ndarray
        :param state_next: 2d vector (shape [p,1]) of the state we moved to after decision.
        
        :type reward: float
        :param reward: reward incurred on last state transition.
        
        :type action: int
        :param action: The action taken on state_last. By default, this agent
        stores the last action, and uses it if None is passed. If doing experience
        replay, pass the correct action instead.
        """
        #Get our last action if necessary
        action = self.last_action if action is None else action
        
        #Create the sgd target
        max_q = np.max(self.ann_clone.predict(state_next))
        target = np.reshape(reward + max_q, [1,1])
        
        #Do SGD!
        self.ann.grad_on(state_last, target, self.last_action)
        
        #Store the experience in our memory (s,s,r,a)
        if not replay:
            self.memory.append([state_last[:], state_next[:], reward, action])
            if len(self.memory) > self.mem_size:
                self.memory.pop(-1)
        
        
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