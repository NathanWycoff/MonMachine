# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 15:29:28 2016

@author: Nathan Wycoff

A game which is meant to be similar to pokemon, but which is totally understood
and under the programmer's control for prototyping of AI meant to play pokemon.

Use Gaussian Processes to search for the optimal hyperparam combination.
As in Dernecourt and Lee (2016) link: https://arxiv.org/pdf/1609.08703v1.pdf
except we look at Expected Improvement (see GP literature, e.g.
http://www.ressources-actuarielles.net/ext/isfa/1226.nsf/9c8e3fd4d8874d60c1257052003eced6/f84f7ac703bf5862c12576d8002f5259/$FILE/Jones98.pdf
) instead of the mean.
"""

#Numerics Libraries
import numpy as np

#Convenience libraries
import tqdm

##Import my other stuff YOU SHOULD CHANGE THIS LINE
import sys
sys.path.append('/home/nathan/Documents/Self Study/MonMachine/')

#Import GP optimization funcs.
#from gp_optimizer import gp_posterior, get_expected_improvement

#Import my learning agents
from Agents.completely_random import random_learner
from Agents.neural_q_learner import neural_q_learner
from Agents.human_agent import Human_Agent

import cPickle as pickle

#Import the entities from my game.
import entities as ent

#Feeds information about the game to the learners/agents
class Environment(object):
    """
    In Reinforcement Learning, the environment is the pipline between the learning
    agents and their task.
    """
    def __init__(self, e1, e2, debug = 0):
        """
        :type e1: Entity
        :param e1: The first entity
        
        :type e2: Entity
        :param e2: The second entity
        
        :type debug: int
        :param debug: verbosity; -1 - None, 0 - Little, 1 - Lots
        """
        #Store the agent's entities.
        self.entity_1 = e1
        self.entity_2 = e2
        
        #Create state variables
        self.state_1 = [1, e1.health, e1.t, e1.p_miss, e1.d, e2.health, e2.t, e2.p_miss, e2.d]
        self.state_2 = [1, e2.health, e2.t, e2.p_miss, e2.d, e1.health, e1.t, e1.p_miss, e1.d]
        self.last_state_1 = self.state_1[:]
        self.last_state_2 = self.state_2[:]

        #Declare state types; for use when creating state vector
        #'C' means continuous. If its categorical, give a list containing possible vals.
        self.state_types = ['C', 'C', ['A', 'B', 'C'], 'C', 'C', 'C', ['A', 'B', 'C'], 'C', 'C']
        
        #When the game ends, the loop will end.
        self.game_over = False
        
        #Debug controls verbosity, higher is more.
        self.debug = debug
    
    #Get a local reference to the learners so they can fed the state/rewards.
    def add_learners(self, l1, l2):
        """
        Store the two learners into the environment.
        
        :type l1: learner
        :param l1: The learner who controls entity 1
        
        :type l2: learner
        :param l2: The learner who controls entity 2.
        """
        #Store the learning agents
        self.agent_1 = l1
        self.agent_2 = l2
    
    def update_state(self):
        """
        Update the environments state and store the last state.
        """
        #Update state for agent 1
        self.last_state_1 = self.state_1[:]
        self.state_1 = [1, self.entity_1.health, self.entity_1.t, self.entity_1.p_miss,\
            self.entity_1.d, self.entity_2.health, self.entity_2.t, self.entity_2.p_miss, self.entity_2.d]
            
        #Update state for agent 2
        self.last_state_2 = self.state_2[:]
        self.state_2 = [1, self.entity_2.health, self.entity_2.t, self.entity_2.p_miss,\
            self.entity_2.d, self.entity_1.health, self.entity_1.t, self.entity_1.p_miss, self.entity_1.d]

    #Turn a state list into a state vector for learner consumption.
    def get_state_vector(self, state_list):
        """
        This function will be called to create one-hot vectors for categorical variates.
        Assumes there is an intercept term, so the last element will return the zero vector.
        Examples in 'A','B','C':
        'A' -> [1,0]; 'B' -> [0,1]; 'C' -> [0,0]
        Statisticians call these "dummy variables".
        
        Uses the "state_types" var to tell which vars are categorical
        
        :type state_list: list
        :param state_list: The state in list form which we want to transfer into vector form.
        """
        
        format_categories = lambda x,i: np.array([1 if y == self.state_types[i].index(x) else 0 for y in range(len(self.state_types[i]))])[:-1]
        
        #Turn the list into a list of arrays containing either just the continuous
        #vars or a one hot vector for the categorical vars.
        enhanced_list = [np.array([x]) if self.state_types[i] == 'C' else \
             format_categories(x,i) for i,x in enumerate(state_list)]
        
        #Concat all the vectors into one vector to be returned.
        state_vec = np.concatenate(enhanced_list)
        state_vec = state_vec.reshape([1, len(state_vec)])
        
        return(state_vec)        
        
    def inform_agents(self):
        """
        Inform the agents of the state transitions and of their rewards.
        """
        #Update Agent Turn Counters
        self.agent_1.new_turn()
        self.agent_2.new_turn()
        
        if not (self.entity_1.alive and self.entity_2.alive):
            if self.debug > 0:
                print "Game Ending..."
            self.game_over = True
            
        #Calculate reward, simply whether they won or not.
        reward_1 = 0 if self.entity_2.alive else 100
        reward_2 = 0 if self.entity_1.alive else 100
        
        #I don't intend negative rewards in this game, so I warn when they happen.
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

    

def function_eval(h_params, train_iters = 10000, test_iters = 1000, max_game_length = 10):
    """
    Evaluate a hyperparam configuration by training and testing it against a 
    random agent.
    
    Returns their win rate (in [0,1])
    
    :type h_params: list
    :param h_params: list of hyperparams [h, h_size, lambda_1, lambda_2, mem_size, replay_size]
    
    :type train_iters: int
    :param train_iters: nonneg number of games played before evaluation
    
    :type test_iters: int
    :param test_iters: nonneg number of games played during evaluation
    
    :type max_game_length: int
    :param max_game_length: Number of turns before game is drawn in test mode.
    """
    
    #All this just to get the state vector size.
    e1 = ent.random_entity()
    e2 = ent.random_entity()
    env = Environment(e1, e2)
    state_size = np.shape(env.get_state_vector(env.state_1))[1]
    
    learner_1 = neural_q_learner(state_size, action_size = 3, eps_l = 0.1, eps_dyn = 0.9, \
        h = h_params[0], h_size = h_params[1], lambda_l1 = h_params[2], \
        lambda_l2 = h_params[3], mem_size = h_params[4], replay_size = h_params[5], \
        eta = 0.1, max_err = 0.1)
    #learner_1 = Human_Agent(1, 3)
    learner_2 = random_learner(3)
    
    print "Training Learner..."
    
    wins_1 = 0
    
    ####Train the learner
    for it in tqdm.tqdm(range(train_iters)):
        
        e1 = ent.random_entity()
        e2 = ent.random_entity()
        
        env = Environment(e1, e2)
        env.add_learners(learner_1, learner_2)
        
        turn_length = 0
        
        while (not env.game_over):
            #The agents get to make a decision, these are 1 indexed
            decision_1 = str(learner_1.act(env.get_state_vector(env.state_1))+1)
            #decision_1 = str(learner_1.act(env.state_1)+1)#Uncomment this line if human agent.
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
            
            if not e2.alive:
                wins_1 += 1
            
            turn_length += 1
            
            if turn_length > max_game_length:
                #print "Game Draw; max length exceeded"
                break
    
    ####Evaluate the learner
    print "Testing Learner..."
    learner_1.exploring = False
    wins = 0
    draws = 0
    for it in tqdm.tqdm(range(test_iters)):
        
        e1 = ent.random_entity()
        e2 = ent.random_entity()
        
        env = Environment(e1, e2)
        env.add_learners(learner_1, learner_2)
        
        turn_length = 0
        
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
                
            turn_length += 1
            if turn_length > max_game_length:
                #print "Game Draw; max length exceeded"
                draws += 1
                break
            
    print float(wins) / test_iters
    print float(draws) / test_iters
    
    
    return(float(wins_1) / test_iters)


###
####Evaluate agents
###
#Evaluate neural nets with different hyper params as compared to a random agent.
#hyper params to eval: [h, h_size, lambda_l1, lambda_l2, mem_size, replay_size]
#Specify different levels to eval
h_g = [0,1,2,3]
h_size_g = [2, 4, 8, 16, 32, 64]
lambda_l1_g = [0.001, 0.01, 0.1, 1, 10]
lambda_l2_g = [0.001, 0.01, 0.1, 1, 10]
mem_size_g = [10, 100, 1000]
replay_size_g = [0, 10, 100]

#Create all possible combinations
X = []
for h in h_g:
    for h_size in h_size_g:
        for l1 in lambda_l1_g:
            for l2 in lambda_l2_g:
                for mem in mem_size_g:
                    for rep in replay_size_g:
                        X.append([h, h_size, l1, l2, mem, rep])

###Do the actual search  
#params
search_iters = 10#How many hyperparam combinations to try?
r = 3#Initial random searches.

train_iters = 10000#How many games in testing stage?
test_iters = 1000#How many games in training stage?

h_params = [2, 8, 0.1, 0.1, 5000, 5000]

#Pick initial X's.
init_inds = np.random.choice(range(np.shape(X)[0]), r, replace = False)

#Evaluate Initial ones.
train = [X[i] for i in init_inds]
candidates = [X[i] for i in range(np.shape(X)[0]) if i not in init_inds]
f = np.array([function_eval(train[i], train_iters, test_iters) for i in range(np.shape(train)[0])])

#For real do the actual search
for i in range(search_iters):
    #Get the means and variances of the posterior process
    mu, sig = gp_posterior(np.array(train), np.array(candidates), f)
    
    #Get the top part of our interval
    ei = get_expected_improvement(mu, sig, max(f))
    
    #Next, look at the configuration with the highest potential.
    next_eval = np.argmax(ei)
    
    #Evaluate the new method
    new_f = function_eval(candidates[i], train_iters, test_iters)
    f = np.append(f, new_f)
    train.append(candidates[next_eval])
    candidates.pop(next_eval)
    
    print "Last f feval was " + str(f[-1])
    print "Best was " + str(max(f))
    
print "Optimal HyperParam combination is at index " + str(np.argmax(f)) + " of \"train\""

#############################################################################
#Generate random vs random play to optimize vs random without having to generate gameplay.
#Stores pairs of "state_last, state_next, reward, action".
#All this just to get the state vector size.
e1 = ent.random_entity()
e2 = ent.random_entity()
env = Environment(e1, e2)
state_size = np.shape(env.get_state_vector(env.state_1))[1]

train_iters = 100000
max_game_length = 1000

learner_1 = random_learner(3)
learner_2 = random_learner(3)

print "Training Learner..."

wins_1 = 0

sars_1 = []
sars_2 = []

####Train the learner
for it in tqdm.tqdm(range(train_iters)):
    
    e1 = ent.random_entity()
    e2 = ent.random_entity()
    
    env = Environment(e1, e2)
    env.add_learners(learner_1, learner_2)
    
    turn_length = 0
    
    while (not env.game_over):
        #The agents get to make a decision, these are 1 indexed
        decision_1 = str(learner_1.act(env.get_state_vector(env.state_1))+1)
        #decision_1 = str(learner_1.act(env.state_1)+1)#Uncomment this line if human agent.
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
        
        #Update storage
        reward_1 = int(not e2.alive) * 100
        reward_2 = int(not e1.alive) * 100
        
        sars_1.append([env.last_state_1, env.state_1, reward_1, decision_1])
        sars_2.append([env.last_state_2, env.state_2, reward_2, decision_2])
        
        if not e2.alive:
            wins_1 += 1
        
        turn_length += 1
        
        if turn_length > max_game_length:
            print "Game Draw; max length exceeded"
            break

sars = sars_1 + sars_2
f = open('/home/nathan/Documents/Self Study/MonMachine/sars.pkl', 'w')
pickle.dump(sars, f)
f.close()