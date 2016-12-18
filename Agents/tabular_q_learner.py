#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 17:00:46 2016

@author: nathan
"""

import numpy as np

#Tabular Q learning        
class tabular_q_learner(object):
    def __init__(self, ep_l = 0.1, ep_dyn = 0.9, exploration_decay = 1):
        #Get all state-action combinations.
        states = [str(y) + x + z for y in range(0,110,10) for x in ['A','B','C'] for z in ['A','B','C']]
        #Add terminal States.
        states = states + [x + y for y in ['A', 'B', 'C'] for x in ['W','L']]
        self.Q = dict(zip(states, [np.random.normal() for i in range(len(states))]))
        
        #Terminal States have known, fixed rewards
        [self.Q.update({'W' + x : 100}) for x in ['A','B','C']]
        [self.Q.update({'L' + x : 0}) for x in ['A','B','C']]
        
        self.alpha = 0.1#Learning Rate
        
        #Epsilon controls exploration
        self.epsilon_low = ep_l#Lower bound on exploration rate
        self.epsilon_dyn = ep_dyn#Dynamic Part of Exploration rate
        self.epsilon = self.epsilon_low + self.epsilon_dyn
        self.exploration_decay = exploration_decay#Epsilon_dyn's exponent increases every how many turns?
        
        self.turn_counter = 0#How many turns have elapsed?
        
        self.action_space = ['A','B','C']#Possible Actions to take each turn

    #Called on new turn, for now just increments turn counter and update epsilon
    def new_turn(self):
        self.turn_counter += 1
        self.epsilon = self.epsilon_low + pow(self.epsilon_dyn, np.floor(self.turn_counter / self.exploration_decay))
    
    #Do the Q learning update when a new turn is reached.
    def update(self, state_last, state_next, reward):
        state_last = self.parse_state(state_last)
        state_next = self.parse_state(state_next)
        max_a = max([self.Q[state_next + x] for x in self.action_space])
        sa_last = state_last + self.last_action
        self.Q[sa_last] = self.Q[sa_last] + self.alpha * (reward + max_a - self.Q[sa_last])
    
    #Turn the raw input from the environment into its internal state representation.
    #For this class, it is a string dictionary key
    def parse_state(self, state):
        #If the opponent is still alive, we are in an ingame state.
        if state[0] > 0:
            health_str = str(int(10.0 * np.floor(10.0 *  state[0] / 1000)))#Get percent health to lowest 10%
            return(health_str + state[1])
        
        #If the opponent is dead, we are in the win state.
        return('W')
        
    #Take an action according to an epsilon-greedy policy
    def act(self, state):
        state = self.parse_state(state)
        #Take a random action with probability epsilon
        if np.random.uniform() < self.epsilon:
            a = self.action_space[np.random.randint(0,len(self.action_space))]
        #Take our current estimate of the best action otherwise.
        else:
            last_state_Q = [self.Q[state + x] for x in self.action_space]
            a = self.action_space[last_state_Q.index(max(last_state_Q))]
                                  
        #Keep our last action to update Q later
        self.last_action = a
        
        return(a)