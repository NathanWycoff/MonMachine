#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 12:17:42 2016

Moves completely randomly; used for sanity testing new learners.

@author: nathan
"""
import numpy as np

#Completely Random Agent
class random_learner(object):
    def __init__(self, action_size):
        self.action_size = action_size

    def new_turn(self):
        pass

    def update(self, state_last, state_next, reward):
        pass
            
    def act(self, state):
        a = np.random.randint(0,self.action_size)
        return(a)