#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 17:21:15 2016

Classes implementing Entities for the MiniMon.

See entities_description.txt for more info.

@author: Nathan Wycoff
"""

import numpy as np

class Entity(object):
    """Entities are the equivalent of Pokemon in Minimon"""
    def take_damage(self, amount):
        """
        Called by other entites attacking this one.
        
        :type amount: float
        :param amount: Amount of damage.
        """
        self.health -= amount
        if self.health <= 0.0:
            if self.debug > 0:
                print 'Entity Dies!'
            self.alive = False
    
    def no_target_err(self):
        print "ERR: Specify a target for offensive moves"
        
    def __str__(self):
        return "Entity of type " + self.t + " with " + str(self.health) + " health."

    __repr__ = __str__

class Entity_A(Entity):
    def __init__(self, debug = 0):
        """
        :type debug: int
        :param debug: Verbosity of Mon, 0 is nothing, 1 is something.
        """
        #Set health stats
        self.health = 500.0
        self.alive = True
        
        #Set special stats
        self.p_miss = 0.0
        self.d = 1.0
        self.max_health = self.health
        
        #Track debug mode
        self.debug = debug
        
        #What type of entity?
        self.t = 'A'
        
    
    def move(self, which, target = None):
        """
        Make the entity execute a move
        
        :type which: int
        :param which: Which move to execute. Should be in {'1', '2', '3'}.
        
        :type target: Entity
        :param target: For offensive moves, which Entity to attack. For other moves, ignored.
        """
        #Move 1 defensive, permanently cause opponents to deal no damage with
        #probability 0.5. Doing this move twice has no effect.
        if which == '1':
            self.p_miss = 0.5
        
        #Move 2 offensive, high damage, high variance, type B.
        elif which == '2':
            #Make sure there's a target to attack
            if target is None:
                self.no_target_err()
                return()
            
            #Otherwise attack the target
            damage = 250 + np.random.normal(0,40)
            
            #Double the damage if the target is weak to our attack
            damage = damage * 2 if target.t == 'B' else damage
            
            #Check if the target is going to dodge.
            if target.t == 'A' and np.random.uniform() < target.p_miss:
                if self.debug > 0:
                    print 'Attack missed target'
                return
            
            #Hit the target.
            if self.debug > 0:
                print 'Attack hits for ' + str(damage) + ' damage.'
            target.take_damage(damage)
        
        #Move 3 offensive, high damage, high variance, type C.
        elif which == '3':
            #Make sure there's a target to attack
            if target is None:
                self.no_target_err()
                return()
            
            #Otherwise attack the target
            damage = 250 + np.random.normal(0,40)
            
            #Double the damage if the target is weak to our attack
            damage = damage * 2 if target.t == 'C' else damage
            
            #Check if the target is going to dodge.
            if target.t == 'A' and np.random.uniform() < target.p_miss:
                if self.debug > 0:
                    print 'Attack missed target'
                return
            
            #Hit the target.
            if self.debug > 0:
                print 'Attack hits for ' + str(damage) + ' damage.'
            target.take_damage(damage)
        else:
            raise ValueError("move should be in {'1','2',3'}")
            
class Entity_B(Entity):
    def __init__(self, debug = 0):
        """
        :type debug: int
        :param debug: Verbosity of Mon, 0 is nothing, 1 is something.
        """
        
        #Set health stats
        self.health = 1000.0
        self.alive = True
        
        #Set special stats
        self.p_miss = 0.0
        self.d = 1
        self.max_health = self.health
        
        #Track debug mode
        self.debug = debug
        
        #What type of entity?
        self.t = 'B'
        
    
    def move(self, which, target = None):
        """
        Make the entity execute a move
        
        :type which: int
        :param which: Which move to execute. Should be in {'1', '2', '3'}.
        
        :type target: Entity
        :param target: For offensive moves, which Entity to attack. For other moves, ignored.
        """
        #Move 1 tactical, increase damage multiplier by 1.5 each time
        #this move is performed.
        if which == '1':
            self.d = 1.5 * self.d
        
        #Move 2 offensive, medium damage, low variance, type B.
        elif which == '2':
            #Make sure there's a target to attack
            if target is None:
                self.no_target_err()
                return()
            
            #Otherwise attack the target
            damage = self.d * (150 + np.random.normal(0,10))
            
            #Double the damage if the target is weak to our attack
            damage = damage * 2 if target.t == 'B' else damage
            
            #Check if the target is going to dodge.
            if target.t == 'A' and np.random.uniform() < target.p_miss:
                if self.debug > 0:
                    print 'Attack missed target'
                return
            
            #Hit the target.
            if self.debug > 0:
                print 'Attack hits for ' + str(damage) + ' damage.'
            target.take_damage(damage)
        
        #Move 2 offensive, medium damage, low variance, type A
        elif which == '3':
            #Make sure there's a target to attack
            if target is None:
                self.no_target_err()
                return()
            
            #Otherwise attack the target
            damage = self.d * (150 + np.random.normal(0,10))
            
            #Double the damage if the target is weak to our attack
            damage = damage * 2 if target.t == 'A' else damage
            
            #Check if the target is going to dodge.
            if target.t == 'A' and np.random.uniform() < target.p_miss:
                if self.debug > 0:
                    print 'Attack missed target'
                return
            
            #Hit the target.
            if self.debug > 0:
                print 'Attack hits for ' + str(damage) + ' damage.'
            target.take_damage(damage)
        else:
            raise ValueError("move should be in {'1','2',3'}")

class Entity_C(Entity):
    def __init__(self, debug = 0):
        """
        :type debug: int
        :param debug: Verbosity of Mon, 0 is nothing, 1 is something.
        """
        #Set health stats
        self.health = 1500.0
        self.alive = True
        
        #Sepcial Stats
        self.max_health = self.health
        self.p_miss = 0.0
        self.d = 1.0
        
        #Track debug mode
        self.debug = debug
        
        #What type of entity?
        self.t = 'C'
        
    
    def move(self, which, target = None):
        """
        Make the entity execute a move
        
        :type which: int
        :param which: Which move to execute. Should be in {'1', '2', '3'}.
        
        :type target: Entity
        :param target: For offensive moves, which Entity to attack. For other moves, ignored.
        """
        #Move 1 defensive. Increase health by an eigth the amount lost.
        if which == '1':
            heal = (self.max_health - self.health) / 8.0
            if self.debug > 0:
                print 'Healed for ' + str(heal)
            self.health += heal
        
        #Move 2 offensive, low damage, medium variance, type B.
        elif which == '2':
            #Make sure there's a target to attack
            if target is None:
                self.no_target_err()
                return()
            
            #Otherwise attack the target
            damage = 100 + np.random.normal(0,20)
            
            #Double the damage if the target is weak to our attack
            damage = damage * 2 if target.t == 'C' else damage
            
            #Check if the target is going to dodge.
            if target.t == 'A' and np.random.uniform() < target.p_miss:
                if self.debug > 0:
                    print 'Attack missed target'
                return
            
            #Hit the target.
            if self.debug > 0:
                print 'Attack hits for ' + str(damage) + ' damage.'
            target.take_damage(damage)
        
        #Move 2 offensive, medium damage, low variance, type A
        elif which == '3':
            #Make sure there's a target to attack
            if target is None:
                self.no_target_err()
                return()
            
            #Otherwise attack the target
            damage = 100 + np.random.normal(0,20)
            
            #Double the damage if the target is weak to our attack
            damage = damage * 2 if target.t == 'A' else damage
            
            #Check if the target is going to dodge.
            if target.t == 'A' and np.random.uniform() < target.p_miss:
                if self.debug > 0:
                    print 'Attack missed target'
                return
            
            #Hit the target.
            if self.debug > 0:
                print 'Attack hits for ' + str(damage) + ' damage.'
            target.take_damage(damage)
        else:
            raise ValueError("move should be in {'1','2',3'}")
    

def random_entity(debug = 0):
    """Return a random entity"""
    a = np.random.randint(0,3)
    if a == 0:
        return(Entity_A(debug = debug))
    if a == 1:
        return(Entity_B(debug = debug))
    if a == 2:
        return(Entity_C(debug = debug))
