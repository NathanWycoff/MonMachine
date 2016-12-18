# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 19:08:13 2016

@author: nathan
"""
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

#A little game with healths
class Person(object):
    def __init__(self, ID, t = 1):
        self.health = 100
        self.ID = ID
    
    def attack(self, person, t):
        damage = np.random.normal(50,10) if t == person.t else np.random.normal(10,10)
        #print "Person " + str(self.ID) + " attacking Person " + str(person.ID) + " for " + str(damage)
        person.damage(damage)
    
    def damage(self, amount):
        self.health -= amount
    
    def __str__(self):
        return("Person " + str(self.ID) + " with " + str(self.health) + " health.")


#Agent which implements deep Q learning for RL/function approx with epsilon greedy action policy
class DQNAgent(object):
    def __init__(self, memory_capacity = 10, hidden_layers = 1, nodes_per_layer = 10, ep = 0.1):
        #Rename some stuff
        self.N = memory_capacity
        self.h = hidden_layers
        self.npl = nodes_per_layer
        
        
        #Init some things
        self.D = []
        self.theta = np.random.normal()
        
        self.action_space = [1,2,3,4]
    
    #Take an action according to an Epsilon-Greedy policy
    def act(self, state):
        if (ep > np.random.uniform()):
            return(np.random.randint(1,4))
        else:
            

iters = 500

agent = MCAgent()

for j in range(iters):
    person1 = Person(1, np.random.randint(1,4))
    person2 = Person(2, np.random.randint(1,4))
    while True:
        first = np.random.binomial(1,0.5)
        if first:
            #print 'Person 1 attacks first!'
            person1.attack(person2, np.random.randint(1,4))
            agent.observe_bg_health(person2.health)
            if person2.health < 0:
                agent.reward(1)
                break
            person2.attack(person1, np.random.randint(1,4))
            if person1.health < 0:
                agent.reward(0)
                break
        else:
            #print 'Person 2 attacks first!'
            person2.attack(person1, np.random.randint(1,4))
            if person1.health < 0:
                agent.reward(0)
                break
            person1.attack(person2, np.random.randint(1,4))
            agent.observe_bg_health(person2.health)
            if person2.health < 0:
                agent.reward(1)
                break
        

logistic = linear_model.LogisticRegression()
logistic.fit(np.array(agent.observed_bg_healths).reshape(-1,1), np.array(agent.result))
predicted = logistic.predict_proba(np.array(agent.observed_bg_healths).reshape(-1,1))[:,1]
plt.scatter(agent.observed_bg_healths, predicted)


#Monte Carlo Reinforcement Learning
#with Logistic Regression for linear function approximation
class MCAgent(object):
    def __init__(self):
        self.observed_bg_healths = []
        self.result = []
    
    def observe_bg_health(self, health):
        self.observed_bg_healths.append(health)
    
    def reward(self, r):
        todo = len(self.observed_bg_healths) - len(self.result)
        [self.result.append(r) for i in range(todo)]