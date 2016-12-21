#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 12:17:42 2016

Moves completely randomly; used for sanity testing new learners.

@author: nathan
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 16:50:57 2016

@author: nathan
"""

import numpy as np

#Completely Random Agent
class random_learner(object):
    def __init__(self):
        self.action_space = ['1', '2', '3']

    #Called on new turn, for now just increments turn counter and update epsilon
    def new_turn(self):
        pass
    
    #Do the Q learning update when a new turn is reached.
    def update(self, state_last, state_next, reward):
        pass
            
    
    #Take an action according to an epsilon-greedy policy
    def act(self, state):
        a = self.action_space[np.random.randint(0,len(self.action_space))]
        
        return(a)