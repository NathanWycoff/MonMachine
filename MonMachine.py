# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 22:44:58 2016

@author: nathan
"""

# -*- coding: utf-8 -*-
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import NoAlertPresentException
import unittest, time, re
import numpy as np


#Represents a pokemon
class Mon(object):
    #Initialize some stuff
    def __init__(self, ID):
        self.ID = ID#What number in the roster?
        self.is_alive = True
        self.hp = 100.0
        self.status = 'Healthy'
    
    #When a mon takes damage, record it here.
    def damage(self, amount):
        self.hp = max(0, self.hp - amount)
        
        print str(self) + " took " + str(amount) + " damage."
        
        #If it's out of HP, kill it
        if abs(self.hp) < 0.0001:
            self.is_alive = False
            self.status = 'Dead'
    
    def __str__(self):
        return("Pokemon " + str(self.ID))
        
    __repr__ = __str__
            
#Handles Mons
class MonWrangler(object):
    def __init__(self, mons):
        self.mons = mons
    
    #Get a mon, specified by ID
    def get_mon(self, ID):
        return(self.mons[ID-1])
    
    #Pick the number of a random alive mon
    def get_random_alive_mon(self):
        alive_mons = [x for x in self.mons if x.is_alive]
        d = np.random.randint(0,len(alive_mons))
        return(alive_mons[d])
    
    #Track a dead mon
    def kill(self, ID):
        print "Pokemon " + str(ID) + " died. RIP in Peace"
        self.mons[ID-1].is_alive = False
    
    

##Prepare some stuff before startup

#Takes data from the webbrowser and feeds it to the agent in a 
#reinfocerment learning theoretic manner.
class Environment(object):
    def __init__(self, log):
        print "Environment is initing..."
        #Roster of mons
        self.kevin = MonWrangler([Mon(x) for x in [1,2,3,4,5,6]])
        
        #Game log
        self.log = log
        
        #Save text of log
        self.log_text = ''
        
        self.event_list = []
        
        #Initialize some things
        self.current_turn = 0
        self.current_mon = self.kevin.get_mon(1)
        
        #Are we currently in a game?
        self.in_game = False
    
    #Check the log for updates and take the appropriate actions
    def refresh(self):
        #print "Refreshing..."
        #See if there's anything new
        if self.log.text == self.log_text:
            #print "Nothing New..."
            pass
        else:
            print "Something new!"
            new_list = self.log.text.split('\n')
            new_entries = new_list[len(self.event_list):]
            self.event_list = new_list
            
            #Update the old one
            self.log_text = self.log.text
            
            print "New entries are\n " + str(new_entries)
            
            #Go through each possibility to update the state and take actions as necessary
            will_take_turn = False
            for entry in new_entries:
                
                #If a new turn has begun
                if re.match('Turn \d*', entry) is not None:
                    self.update_turn(entry.split(' ')[1])
                    will_take_turn = True
                         
                #Check to see if anyone on our side died
                if re.match('.+ fainted!', entry) is not None and re.match('^The opposing [a-zA-Z0-9_\s.\']+ fainted!', entry) is None:
                    self.funeral()
                
                #Check to see if one of their mons took damage.
                #Handles both decimal and nondecimal damage counts.
                match1_op = re.match('.*The opposing .+ lost \d*.\d% of its health!', entry) is not None
                match2_op = re.match('.*The opposing .+ lost \d*% of its health!', entry) is not None                
                if match1_op or match2_op:
                    damage = float(re.search('(\d*.\d%|\d*%)',entry).group(0)[:-1])
                    print "Their mon took " + str(damage) + " damage."
                else:
                    #Check to see if one of our mons took damage
                    #Matches both the case where decimal damage is dealt (1) or when it is a whole number (2)
                    match1 = re.match('.+ lost \d*.\d% of its health!', entry) is not None and re.match('The opposing [a-zA-Z0-9_\s.\']+ lost \d*.\d% of its health!', entry) is None
                    match2 = re.match('.+ lost \d*% of its health!', entry) is not None and re.match('The opposing [a-zA-Z0-9_\s.\']+ lost \d*% of its health!', entry) is None
                    if match1 or match2:
                        damage = float(re.search('(\d*.\d%|\d*%)',entry).group(0)[:-1])
                        self.current_mon.damage(damage)
                
                #Check if the battle's over
                if re.match('.* won the battle!', entry) is not None:
                    if re.match('monmachine',entry) is not None:
                        print "We win!"
                    else:
                        print "GG, we lost"
                    self.in_game = False
                
            
            #Otherwise, if at least one turn passed, make a move
            if will_take_turn:
                print "Taking Turn"
                print "Last entry was \"" + entry + "\""
                self.take_turn()
                        
    #Main action function, for now move randomly
    def take_turn(self):
        d = np.random.randint(1,4)
        try:
            driver.find_element_by_xpath("(//button[@name='chooseMove'])[" + str(d) + "]").click()
        except:#Caterpies only have moves in 1,2
            d = 1
            driver.find_element_by_xpath("(//button[@name='chooseMove'])[" + str(d) + "]").click()
        
    def switch_to_mon(self, mon):
        self.current_mon = mon
        to = mon.ID
        print "Switching to mon " + str(mon)
        driver.find_element_by_css_selector('.switchmenu > button:nth-child(' + str(to) + ')').click()
        #driver.find_element_by_xpath("(//button[@name='chooseSwitch'])[" + str(to) + "]").click()
    
    #Called when one of our mons die :(
    #For now, just summon a random mon
    def funeral(self):
        self.kevin.kill(self.current_mon.ID)
        self.switch_to_mon(self.kevin.get_random_alive_mon())
    
    #This function updates the turn counter and prints it out.
    def update_turn(self, turn):        
        self.current_turn = int(turn)
        print "It is now turn " + str(turn)

###################################
#######   INITIALIZATION  #########
###################################
driver = webdriver.Firefox()
driver.implicitly_wait(30)
base_url = "http://play.pokemonshowdown.com/"

driver.get(base_url + "/")

time.sleep(10)

#driver.find_element_by_css_selector("div.pad").click()
time.sleep(1)
driver.find_element_by_name("login").click()
time.sleep(1)
driver.find_element_by_name("username").clear()
time.sleep(1)
driver.find_element_by_name("username").send_keys("monmachine")
time.sleep(1)
driver.find_element_by_css_selector("button[type=\"submit\"]").click()
time.sleep(1)
driver.find_element_by_name("password").clear()
time.sleep(1)
driver.find_element_by_name("password").send_keys("Stats2016")
time.sleep(1)
driver.find_element_by_css_selector("button[type=\"submit\"]").click()
time.sleep(3)

driver.find_element_by_name("format").click()
time.sleep(1)
driver.find_element_by_xpath("(//button[@name='selectFormat'])[51]").click()
time.sleep(1)

original_url = driver.current_url#Until the search page loads, wait

driver.find_element_by_name("search").click()
time.sleep(1)

#Wait for the battle to load
while driver.current_url == original_url:
    time.sleep(1)


###################################
#####   MAIN GAMEPLY CYCLE  #######
###################################

log = driver.find_element_by_css_selector('.battle-log')
env = Environment(log)
i = 0
while env.in_game:
    i += 1
    env.refresh()
    time.sleep(2)