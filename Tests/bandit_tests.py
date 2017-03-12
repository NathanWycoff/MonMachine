# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 13:58:03 2016

@author: Nathan Wycoff

Application of the neural Q learner to an N armed bandit problem of increasing
complexity.

Will eventually become a test battery.
"""

import numpy as np
import matplotlib.pyplot as plt

##Import my other stuff YOU SHOULD CHANGE THIS LINE
import sys
sys.path.append('/home/nathan/Documents/Documents/Self Study/MonMachine/')

#Import my learning agents
from Agents.completely_random import random_learner
from Agents.neural_q_learner import neural_q_learner

class N_Armed_Bandit(object):
    """
    An implementation of a simple N-armed bandit problem.
    
    Arm means are drawn from a normal distribution.
    
    Rewards are drawn from a normal distribution around arm means with shared
    variance.
    """
    def __init__(self, n = 2, sigma = 1, base_measure = [0, 10]):
        """
        :type n: int
        :param n: number of arms
        
        :type sigma: float
        :param sigma: sd of draws from an arm (arms are normal dists).
        
        :type base_measure: list
        :param base_measure: mean and sd of the normal dist from which arm means
        are drawn.
        """
        
        self.arm_means = np.random.normal(base_measure[0], base_measure[1], size = n)
        self.sigma = sigma
        
        
    def pull(self, which):
        """Pull lever which"""
        r = np.random.normal(self.arm_means[which], self.sigma)
        return(r)


#HyperParams
seed = 1

######
######TEST 1
#See if we can learn which lever to pull using an intercept only model.

np.random.seed(seed)

#Params
n = 10
iters = 1000
bandit = N_Armed_Bandit(n = n)

#Initialize Learners
learner_1 = neural_q_learner(state_size = 1, action_size = n, h = 0, lambda_l1 = 0, lambda_l2 = 0)
learner_2 = random_learner(action_size = n)

#Init rewards
total_1 = 0
total_2 = 0

#Constant state vector
state = np.ones([1,1])

for i in range(iters):
    #Get the decisions
    decision_1 = learner_1.act(state)
    decision_2 = learner_2.act(state)
    
    #Execute the decisions
    reward_1 = bandit.pull(decision_1)
    reward_2 = bandit.pull(decision_2)
    
    #Update the learner
    learner_1.update(state, state, reward_1)
    learner_2.update(state, state, reward_2)
    
    #Call the new turn
    learner_1.new_turn()
    learner_2.new_turn()
    
    #Update totals
    total_1 += reward_1
    total_2 += reward_2

print "Exercise 1"
print total_1
print total_2

######
######TEST 2
#Can the neural learner learn from random moves, so that it makes start decisions
#from the beginning?

np.random.seed(seed)

#Params
n = 10
iters = 1000
bandit = N_Armed_Bandit(n = n)

#Initialize Learners
learner_1 = neural_q_learner(state_size = 1, action_size = n, h = 0, lambda_l1 = 0, lambda_l2 = 0)
learner_2 = random_learner(action_size = n)

#Init rewards
total_2 = 0

#Constant state vector
state = np.ones([1,1])

transitions = []

###Make some random moves.
for i in range(iters):
    #Get the decisions
    decision_2 = learner_2.act(state)
    
    #Execute the decisions
    reward_2 = bandit.pull(decision_2)
    
    #Update the learner
    learner_2.update(state, state, reward_2)
    
    #Call the new turn
    learner_2.new_turn()
    
    total_2 += reward_2
    
    #Store result [s_1, s_2, r, a]
    transitions.append([state, state, reward_2, decision_2])
    
##Learner_1 observes the transitions of learner_2.
for t in transitions:
    learner_1.update(t[0], t[1], t[2], t[3], False)
    learner_1.new_turn()

##Learner_2 makes some decisions.
for i in range(iters):
    #Get the decisions
    decision_1 = learner_1.act(state)
    
    #Execute the decisions
    reward_1 = bandit.pull(decision_1)
    
    #Update the learner
    learner_1.update(state, state, reward_1)
    
    #Call the new turn
    learner_1.new_turn()
    
    #Update totals
    total_1 += reward_1
    
print "Exercise 2"
print total_1
print total_2