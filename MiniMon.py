# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 15:29:28 2016

@author: Nathan Wycoff

A game which is meant to be similar to pokemon, but which is totally understood
and under the programmer's control for prototyping of AI meant to play pokemon.

Rules:

Entities take turns attacking each other.

There are 3 types of entities, type A, B and C.

There are 3 types of attacks: type A, B and C.

Each entity has access to all 3 attack types.

Attacks do double damage against entities of the same type, e.g., attacking an 
    entity of type A with an attack of type A deals double damage.
    
Attack type A deals 100 base damage, type B 110, and type C 120 damage.

"""

import os
os.chdir('/home/nathan/Documents/Documents/Self Study/MonMachine')

import numpy as np
import matplotlib.pyplot as plt
from Agents.linear_q_learner import linear_q_learner
from Agents.tabular_q_learner import tabular_q_learner



        


#Feeds information about the game to the learners/agents
class Environment(object):
    def __init__(self, e1, e2, l1, l2):
        self.agent_1 = l1
        self.agent_2 = l2
        self.entity_1 = e1
        self.entity_2 = e2
        #My ent, his ent
        self.state_1 = [1000, e2.t]
        self.state_2 = [1000, e1.t]
        self.last_state_1 = self.state_1[:]
        self.last_state_2 = self.state_2[:]
        
        self.starting_health = 1000.0
        
        self.game_over = False#When the game ends, the loop will end.
        #self.break_parent = False#Break the loop right now for debugging.
    
    #When an entity's health changes, record it.
    #Entity healths are recorded as percentages, floored to 10% intervals.
    def update_state(self, new_state_health, entity):
        if entity == '1':
            self.last_state_1[0] = self.state_1[0]
            self.state_1[0] = new_state_health
        elif entity == '2':
            self.last_state_2[0] = self.state_2[0]
            self.state_2[0] = new_state_health
    
    #Inform the agents of the state transitions and of their rewards.
    def inform_agents(self):
        #Update Agent Turn Counters
        self.agent_1.new_turn()
        self.agent_2.new_turn()
        
        #Check if someone died.
        #if not self.entity_1.alive:
        #    self.state_1 = 'L'
        #    self.state_2 = 'W'
        #    self.game_over = True
        #elif not self.entity_2.alive:
        #    self.state_1 = 'W'
        #    self.state_2 = 'L'
        #    self.game_over = True
        
        if not (self.entity_1.alive and self.entity_2.alive):
            self.game_over = True
        
        #Typecast list into str
        #state_1_s = ''.join(x for x in self.state_1)
        #state_2_s = ''.join(x for x in self.state_2)
        #last_state_1_s =  ''.join(x for x in self.last_state_1)
        #last_state_2_s =  ''.join(x for x in self.last_state_2)
            
        #Calculate reward, simply difference in enemy health
        reward_1 = 0 * (int(self.last_state_1[0]) - int(self.state_1[0])) if self.entity_2.alive else 100
        reward_2 = 0 * (int(self.last_state_2[0]) - int(self.state_2[0])) if self.entity_1.alive else 100
        
        if reward_1 < 0 or reward_2 < 0:
            self.game_over = True
            print 'WARN: Negative Reward'
        
        #print 'Agent 1 has reward ' + str(reward_1)
        #print 'Agent 2 has reward ' + str(reward_2)
        
        #Update the agents.
        self.agent_1.update(self.last_state_1, self.state_1, reward_1)
        self.agent_2.update(self.last_state_2, self.state_2, reward_2)



#Sim Params
iters = 30000
learner_1 = linear_q_learner(learning_decay = 10000000000, eta = 0.5)
learner_2 = tabular_q_learner()

wins_1 = 0

eps = []
etas = []

for it in range(iters):
    
    e1 = Entity()
    e2 = Entity()
    
    env = Environment(e1, e2, learner_1, learner_2)
    
    eps.append(learner_1.epsilon)
    etas.append(learner_1.eta)
    
    while (not env.game_over):
        #The agents get to make a decision
        decision_1 = learner_1.act(env.state_1)
        decision_2 = learner_2.act(env.state_2)
        
        #Pick who goes first randomly
        whos_first = np.random.binomial(1,0.5)
        
        #Do the moves
        if whos_first:
            e1.attack(e2, decision_1)
            e2.attack(e1, decision_2) if e2.alive else 0
        else:
            e2.attack(e1, decision_2)
            e1.attack(e2, decision_1) if e1.alive else 0
        
        #Update state
        env.update_state(e2.health, '1')
        env.update_state(e1.health, '2')
        
        #Update Agents
        env.inform_agents()
        
        if not e2.alive:
            wins_1 += 1

print 'Learner 1 win percentage: ' + str(100*(0.0 + wins_1) / iters) + '%'
plt.plot(range(iters), eps)
plt.plot(range(iters), etas)
plt.show()

