# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 22:44:58 2016

@author: Nathan Wycoff

This script interfaces with Pokemon Showdown.

It is not complete.

It relays rudimentary game information to rudimentary agents. More advanced
agents are being prototyped using MiniMon, see MiniMon.py and ./Agents
"""

from selenium import webdriver
import time, re
import numpy as np
import csv


class Mon(object):
    """Represents a Pokemon"""
    def __init__(self, ID):
        self.ID = ID#What number in the roster?
        self.is_alive = True
        self.hp = 100.0
        self.status = 'Healthy'
    
    def damage(self, amount):
        """When a Mon takes damage, record it here"""
        self.hp = max(0, self.hp - amount)
        
        print str(self) + " took " + str(amount) + " damage."
        
        #If it's out of HP, kill it
        if abs(self.hp) < 0.0001:
            self.is_alive = False
            self.status = 'Dead'
    
    def __str__(self):
        return("Pokemon " + str(self.ID))
        
    __repr__ = __str__
            
class MonWrangler(object):
    """Handles a collection of Mons"""
    def __init__(self, mons):
        self.mons = mons
    
    def get_mon(self, ID):
        """Get a mon, specified by ID"""
        return(self.mons[ID-1])
    
    def get_random_alive_mon(self):
        """Pick the number of a random alive mon"""
        alive_mons = [x for x in self.mons if x.is_alive]
        d = np.random.randint(0,len(alive_mons))
        if d > 0:
            return(alive_mons[d])
        else:
            return(None)
    
    def kill(self, ID):
        """Track a dead mon"""
        print "Pokemon " + str(ID) + " died. RIP in Peace"
        self.mons[ID-1].is_alive = False
    
class Entry(object):
    """An entry in the game log"""
    def __init__(self, ID, text):
        self.ID = ID
        self.text = unicode(text).encode('ascii','ignore')
        self.handled = False
    
    def __str__(self):
        if not self.handled:
            return("Unhandled entry with ID: " + str(self.ID) + " and text " + self.text)
        else:
            return("Handled entry with ID: " + str(self.ID) + " and text " + self.text)
    
    __repr__ = __str__

class EntryManager(object):
    """Handles entries to make sure that we process each"""
    def __init__(self):
        """Start with an empty list of entries"""
        self.entry_list = []
        self.entry_IDs = []
    
    def register_entries(self, entries):
        """Register new entries into our list if we don't have them already"""
        new_entries = [[i,x] for i,x in enumerate(entries) if i not in self.entry_IDs]
        for entry in new_entries:
            self.entry_list.append(Entry(entry[0],entry[1]))
        
        self.entry_IDs = [x.ID for x in self.entry_list]
    
    def get_unhandled_entries(self):
        """Return any entries not yet handled."""
        return([x for x in self.entry_list if not x.handled])
        
    
    
    

##Prepare some stuff before startup

class Environment(object):
    """
    Takes data from the webbrowser and feeds it to the agent in a reinfocerment 
    learning theoretic manner."""
    def __init__(self, log, agent):
        print "Environment is initing..."
        
        #Our AI
        self.agent = agent
        
        #Roster of mons for us and them
        self.our_mon_wrang = MonWrangler([Mon(x) for x in [1,2,3,4,5,6]])
        self.their_mon_wrang = MonWrangler([Mon(x) for x in [1,2,3,4,5,6]])
        
        #To handle log entries
        self.entry_manager = EntryManager()
        
        #Game log
        self.log = log
        
        #Save text of log
        #self.log_text = ''
        
        self.event_list = []
        
        #Initialize some things
        self.current_turn = 0
        self.current_mon = self.our_mon_wrang.get_mon(1)
        
        #Are we currently in a game?
        self.in_game = False
    
    def refresh(self):
        """Check the log for updates and take the appropriate actions"""
        #See if there's anything new
        self.entry_manager.register_entries(self.log.text.split('\n'))
        
        new_entries = self.entry_manager.get_unhandled_entries()
        
        print "New entries are\n " + str([x.text for x in new_entries])
        
        #Go through each possibility to update the state and take actions as necessary
        for entry_obj in new_entries:
            
            entry = entry_obj.text
            
            #Matches damage to ourselves and the opponent
            #Matches both the case where decimal damage is dealt (1) or when it is a whole number (2)
            match1_op = re.match('.*The opposing .+ lost \d*.\d% of its health!', entry) is not None
            match2_op = re.match('.*The opposing .+ lost \d*% of its health!', entry) is not None                
            match1 = re.match('.+ lost \d*.\d% of its health!', entry) is not None and re.match('The opposing [a-zA-Z0-9_\s.\']+ lost \d*.\d% of its health!', entry) is None
            match2 = re.match('.+ lost \d*% of its health!', entry) is not None and re.match('The opposing [a-zA-Z0-9_\s.\']+ lost \d*% of its health!', entry) is None
            
            #If a new turn has begun
            if re.match('Turn \d*', entry) is not None:
                self.update_turn(entry.split(' ')[1])
                print "Taking Turn"
                print "Last entry was \"" + entry + "\""
                self.take_turn()
                     
            #Check to see if anyone on our side died
            elif re.match('.+ fainted!', entry) is not None and re.match('^The opposing [a-zA-Z0-9_\s.\']+ fainted!', entry) is None:
                self.funeral()
            
            #Check to see if one of their mons took damage.
            elif match1_op or match2_op:
                damage = float(re.search('(\d*.\d%|\d*%)',entry).group(0)[:-1])
                print "Their mon took " + str(damage) + " damage."
                
            #Check to see if one of our mons took damage                
            elif match1 or match2:
                damage = float(re.search('(\d*.\d%|\d*%)',entry).group(0)[:-1])
                self.current_mon.damage(damage)
            
            #Check if the battle's over
            elif re.match('.* won the battle!', entry) is not None:
                
                if re.match('MonMachine',entry) is not None:
                    print "We win!"
                    self.agent.reward(100.0, 'Terminal')
                else:
                    print "GG, we lost"
                    self.agent.reward(0.0, 'Terminal')
                
                self.in_game = False
                
                agent.save()
                print 'Q and V updated'
                
            else:
                print "Unkown Entry in Log"
                print entry
            
            entry_obj.handled = True
                        
    def take_turn(self):
        """Main action function, prompt agent for action"""
        d = self.agent.act(self.current_mon.ID)
        self.agent.reward(0.0, self.current_mon.ID)
        try:
            driver.find_element_by_xpath("(//button[@name='chooseMove'])[" + str(d) + "]").click()
        except:#Maybe we did hyperbeam?
            d = 1
            driver.find_element_by_xpath("(//button[@name='chooseMove'])[" + str(d) + "]").click()
        
    
    def switch_to_mon(self, mon):
        """Called when switching active pokemon in game"""
        self.current_mon = mon
        to = mon.ID
        print "Switching to mon " + str(mon)
        driver.find_element_by_css_selector('.switchmenu > button:nth-child(' + str(to) + ')').click()
        #driver.find_element_by_xpath("(//button[@name='chooseSwitch'])[" + str(to) + "]").click()
    
    def funeral(self):
        """Called when one of our mons die :(
        For now, just summon a random mon
        """
        self.our_mon_wrang.kill(self.current_mon.ID)
        if False:#Next Mon is random
            next_mon = self.our_mon_wrang.get_random_alive_mon()
            if next_mon is not None:
                self.switch_to_mon(next_mon)
        else:
            try:
                next_mon = self.our_mon_wrang.get_mon(self.current_mon.ID + 1)
                self.switch_to_mon(next_mon)
            except:
                print "Out of Mons!"
    
    def update_turn(self, turn):        
        """This function updates the turn counter and prints it out."""
        self.current_turn = int(turn)
        print "It is now turn " + str(turn)

class Agent(object):
    """Tabular Q learning with epsilon greedy action selection"""
    def __init__(self, load = False):
        #Parameters
        self.epsilon = 0.5#Probability of exploratory action
        self.alpha = 0.5#Learning rate
        self.gamma = 1#Discount Rate
        
        #Initialize spaces
        self.state_space = [1,2,3,4,5,6]#State space is just what mon I have
        self.action_space = [1,2,3,4]
        
        #Initialize state and action value functions, or load them from dics
        if not load:
            self.V = [np.random.normal() for x in self.state_space]
            self.Q = [[np.random.normal() for x in self.action_space] for y in self.state_space]
        else:#This code copied from http://stackoverflow.com/questions/19838380/building-list-of-lists-from-csv-file
            #Read in old V
            with open('/home/nathan/Documents/Self Study/MonMachine/learned_V.csv', 'rU') as f:  #opens PW file
                self.V = list([float(x) for x in list(rec)] for rec in csv.reader(f, delimiter=','))
            
            #Read in old Q
            with open('/home/nathan/Documents/Self Study/MonMachine/learned_Q.csv', 'rU') as f:  #opens PW file
                self.Q = list([float(x) for x in list(rec)] for rec in csv.reader(f, delimiter=','))
        
        #Other Initialization
        self.last_state = 1
        self.last_action = 1
    
    def act(self, state):
        """Get an action from an epsilon greedy policy."""
        #If we do a random action
        if (np.random.uniform() < self.epsilon):
            print 'Taking an exploratory action.'
            d = np.random.randint(1,4)
            action = d
        else:
            print 'Taking a greedy action.'
            action = self.Q[state-1].index(max(self.Q[state-1])) + 1
        
        #Set up reward to be updated
        self.last_action = action
        self.last_state = state
        
        return(action)
    
    def reward(self, reward, new_state):
        """Call when a reward is given to update Q and V"""
        #Update Q
        optimal_value = max(self.Q[new_state-1]) if new_state != 'Terminal' else 0
        current_q = self.Q[self.last_state-1][self.last_action-1]
        self.Q[self.last_state-1][self.last_action-1] = current_q + self.alpha * \
            (reward + self.gamma * optimal_value - current_q)
        
        #Update V, not necessary for Q learning, but may be interesting to look at.
        for i in range(len(self.V)):
            optimal_action = self.Q[self.last_state-1].index(max(self.Q[self.last_state-1])) + 1
            self.V[i] = sum([self.epsilon / 3.0 * self.Q[i][x-1] if x != optimal_action else self.epsilon * self.Q[i][x-1] for x in self.action_space])
    
    def save(self):
        """Save learned table"""
        target = '/home/nathan/Documents/Self Study/MonMachine/learned_Q.csv'
        with open(target, 'w') as f:
            writer = csv.writer(f)
            writer.writerows(self.Q)
        
        target = '/home/nathan/Documents/Self Study/MonMachine/learned_V.csv'
        with open(target, 'w') as f:
            writer = csv.writer(f)
            writer.writerows(self.Q)
    
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
driver.find_element_by_name("username").send_keys("MonMachine")
time.sleep(1)
driver.find_element_by_css_selector("button[type=\"submit\"]").click()
time.sleep(1)
driver.find_element_by_name("password").clear()
time.sleep(1)
driver.find_element_by_name("password").send_keys("Stats2016")
time.sleep(1)
driver.find_element_by_css_selector("button[type=\"submit\"]").click()


games = 0
while True:
    
    print "Beginning Game " + str(games)
    
    driver.find_element_by_name("finduser").click()
    driver.find_element_by_name("data").clear()
    driver.find_element_by_name("data").send_keys("MonTonSoup")
    driver.find_element_by_css_selector("button[type=\"submit\"]").click()
    driver.find_element_by_name("challenge").click()
    driver.find_element_by_name("format").click()
    driver.find_element_by_xpath("(//button[@name='selectFormat'])[70]").click()
    original_url = driver.current_url#Until the search page loads, wait
    driver.find_element_by_name("makeChallenge").click()
    #Wait for the battle to load
    while driver.current_url == original_url:
        time.sleep(1)
    ###################################
    #####   MAIN GAMEPLY CYCLE  #######
    ###################################
    
    log = driver.find_element_by_css_selector('.battle-log')
    agent = Agent(load=True)
    env = Environment(log, agent)
    i = 0
    env.in_game = True
    while env.in_game:
        i += 1
        env.refresh()
        time.sleep(2)
    
    #Click back to the main menu
    driver.find_element_by_name('closeAndMainMenu').click()