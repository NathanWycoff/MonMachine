# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 15:29:28 2016

@author: Nathan Wycoff

A game which is meant to be similar to pokemon, but which is totally understood
and under the programmer's control for prototyping of AI meant to play pokemon.
"""

import os
os.chdir('/home/nathan/Documents/Documents/Self Study/MonMachine')

#Numerics Libraries
import numpy as np
import matplotlib.pyplot as plt

#Convenience libraries
import tqdm

#Imports from other files
#TODO: This only works from iPython, make it work from standard cpython.
from Agents.linear_q_learner import linear_q_learner
from Agents.tabular_q_learner import tabular_q_learner
from Agents.completely_random import random_learner
from Agents.neural_q_learner import neural_q_learner
import entities as ent

#Feeds information about the game to the learners/agents
class Environment(object):
    def __init__(self, e1, e2, debug = 0):
        
        #Store the agent's entities.
        self.entity_1 = e1
        self.entity_2 = e2
        
        #Create state variables
        self.state_1 = [1, e2.health, e2.t, e2.p_miss, e2.d]
        self.state_2 = [1, e1.health, e1.t, e1.p_miss, e1.d]
        self.last_state_1 = self.state_1[:]
        self.last_state_2 = self.state_2[:]

        #Declare state types; for use when creating state vector
        self.state_types = ['C', 'C', ['A', 'B', 'C'], 'C', 'C']
        
        #When the game ends, the loop will end.
        self.game_over = False
        
        #Debug controls verbosity, higher is more.
        self.debug = debug
    
    #Get a local reference to the learners so they can fed the state/rewards.
    def add_learners(self, l1, l2):
        #Store the learning agents
        self.agent_1 = l1
        self.agent_2 = l2
    
    #Update the environments state and store the last state.
    #e1 and e2 should be the two entities.
    def update_state(self):
        #Update state for agent 1
        self.last_state_1 = self.state_1[:]
        self.state_1 = [1, np.log(abs(self.entity_2.health) + 0.1), self.entity_2.t, self.entity_2.p_miss, self.entity_2.d]
            
        #Update state for agent 2
        self.last_state_2 = self.state_2[:]
        self.state_2 = [1, np.log(abs(self.entity_1.health) + 0.1), self.entity_1.t, self.entity_1.p_miss, self.entity_1.d]

    #Turn a state list into a state vector for learner consumption.
    def get_state_vector(self, state_list):        
        #This function will be called to create one-hot vectors for categorical variates.
        #Assumes there is an intercept term, so the last element will return the zero vector.
        #Examples in 'A','B','C':
        #'A' -> [1,0]; 'B' -> [0,1]; 'C' -> [0,0]
        #Statisticians call these "dummy variables".
        format_categories = lambda x,i: np.array([1 if y == self.state_types[i].index(x) else 0 for y in range(len(self.state_types[i]))])[:-1]
        
        #Turn the list into a list of arrays containing either just the continuous
        #vars or a one hot vector for the categorical vars.
        enhanced_list = [np.array([x]) if self.state_types[i] == 'C' else \
             format_categories(x,i) for i,x in enumerate(state_list)]
        
        #Concat all the vectors into one vector to be returned.
        state_vec = np.concatenate(enhanced_list)
        state_vec = state_vec.reshape([1, len(state_vec)])
        
        return(state_vec)        
        
    #Inform the agents of the state transitions and of their rewards.
    def inform_agents(self):
        #Update Agent Turn Counters
        self.agent_1.new_turn()
        self.agent_2.new_turn()
        
        if not (self.entity_1.alive and self.entity_2.alive):
            if self.debug > 0:
                print "Game Ending..."
            self.game_over = True
            
        #Calculate reward, simply difference in enemy health
        reward_1 = 0 * (int(self.last_state_1[0]) - int(self.state_1[0])) if self.entity_2.alive else 100
        reward_2 = 0 * (int(self.last_state_2[0]) - int(self.state_2[0])) if self.entity_1.alive else 100
        
        if reward_1 < 0 or reward_2 < 0:
            if self.debug > -1:
                print 'WARN: Negative Reward'
        
        if self.debug > 0:
            print 'Agent 1 has reward ' + str(reward_1)
            print 'Agent 2 has reward ' + str(reward_2)
        
        #Update the agents.
        self.agent_1.update(self.get_state_vector(self.last_state_1), \
                            self.get_state_vector(self.state_1), reward_1)
        self.agent_2.update(self.get_state_vector(self.last_state_2), \
                            self.get_state_vector(self.state_2), reward_2)



#Sim Params

iters = 5000

#All this just to get the state vector size.
e1 = ent.random_entity()
e2 = ent.random_entity()
env = Environment(e1, e2)
state_size = np.shape(env.get_state_vector(env.state_1))[1]

#learner_1 = linear_q_learner(state_size, ep_l = 0.05, learning_decay = 10000, exploration_decay = 1000, eta = 0.5)
learner_1 = neural_q_learner(state_size, action_size = 3, eps_l = 0.1, eps_dyn = 0.9, h = 0)
learner_2 = random_learner(3)

wins_1 = 0

game_lengths = []

for it in tqdm.tqdm(range(iters)):
    
    e1 = ent.random_entity()
    e2 = ent.random_entity()
    
    env = Environment(e1, e2)
    env.add_learners(learner_1, learner_2)
    
    
    game_length = 0
    
    while (not env.game_over):
        #The agents get to make a decision, these are 1 indexed
        decision_1 = str(learner_1.act(env.get_state_vector(env.state_1))+1)
        decision_2 = str(learner_2.act(env.get_state_vector(env.state_2))+1)
        
        #Pick who goes first randomly
        whos_first = np.random.binomial(1,0.5)
        
        #Do the moves
        if whos_first:
            e1.move(decision_1, e2)
            e2.move(decision_2, e1) if e2.alive else 0
        else:
            e2.move(decision_2, e1)
            e1.move(decision_1, e2) if e1.alive else 0
        
        #Update state
        env.update_state()
        
        #Update Agents
        env.inform_agents()
        
        env.state_1
        
        if not e2.alive:
            wins_1 += 1
        
        game_length += 1
    
    game_lengths.append(game_length)

print 'Learner 1 win percentage: ' + str(100*(0.0 + wins_1) / iters) + '%'
plt.show()