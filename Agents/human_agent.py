# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 20:42:33 2016

@author: Nathan Wycoff

User Controlled Agent.
"""

class Human_Agent(object):
    """
    Human Controlled Agent for testing purposes.
    """
    def __init__(self, version, action_size):
        """
        
        :type version: int
        :param version: Which version of Minimon are we playing? (For now always 1)
        
        :type action_size: int
        :param action_size: Size of action space
        """
        self.turn_counter = 1
        self.version = version
        self.action_size = action_size
    
    def new_turn(self):
        """
        Called on new turn, increments turn counter.
        """
        
        self.turn_counter += 1
        
    def update(self, state_last, state_next, reward, action = None, replay = False):
        """
        The human is assumed to know that the only reward is offered when a game is won.
        """
        pass
        
        
    def act(self, state, version = 1):
        """
        Prompt the user for an action
        
        :type state: list
        :param state: The state list for this turn.
        
        
        """
        if self.version == 1:
            print "Your Entity of type " + state[2] + " has health " + str(state[1]) + ", p_miss = " + str(state[3]) + \
                " and d = " + str(state[4])
            print "Hostile Entity of type " + state[6] + " has health " + str(state[5]) + ", p_miss = " + str(state[7]) + \
                " and d = " + str(state[8])
        
        print "Select an Action 0-" + str(self.action_size-1)
        a = int(raw_input())
        
        return(a)