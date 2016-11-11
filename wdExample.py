# -*- coding: utf-8 -*-
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import NoAlertPresentException
import unittest, time, re
import numpy as np


##Prepare some stuff before startup
    
#Stores info about the game
class Status(object):
    def __init__(self, log):
        #Roster of mons
        self.our_mons = [1,2,3,4,5,6]
        
        #Game log
        self.log = log
        
        #Save text of log
        self.log_text = log.text
        
        #Initialize some things
        self.current_turn = 0
    
    #Check the log for updates and take the appropriate actions
    def refresh(self):
        #See if there's anything new
        if self.log.text == self.log_text:
            return
        else:
            new_entries = self.log.text[len(self.log_text):]
            new_entries = new_entries.split('\n')
            
            #Go through each possibility to update the state and take actions as necessary
            for entry in new_entries:
                
                #If a new turn has begun
                if re.match('Turn \d*', x) is not None:
                    self.update_turn(x.split(' ')[1])
                    self.take_turn()
                    
    #Main action function, for now move randomly
    def take_turn(self):
        d = np.random.randint(1,4)
        driver.find_element_by_xpath("(//button[@name='chooseMove'])[" + str(d) + "]").click()
    
    #This function updates the turn counter and prints it out.
    def update_turn(self, turn):        
        self.current_turn = int(turn)
        print "It is now turn " + str(self.current_turn)
    
    #If one of our mons die, keep track of it
    def track_death(self, number):
        self.our_mons.remove(number)
        
    #Pick the number of a random alive mon
    def get_random_friendly_mon(self):
        d = np.random.randint(0,len(self.our_mons))
        return(self.our_mons[d])
        
    #Refresh, checking what turn number it is
    def get_current_turn(self, log):
        text = log.text.split('\n')
        turns = [x for x in text if re.match('Turn \d*', x) is not None]
        
        return(int(turns[-1].split(' ')[1]))
        
    #Refresh, checking if the game is awaiting our input
    def is_myturn(self, log):
        text = log.text.split('\n')[-1]
        waiting = re.match('Turn \d*', text) is not None
        
        timer = re.match('^[a-zA-Z0-9_]+ has \d* seconds left.',text) is not None
        
        return(waiting or timer)

current_turn = 0


#Start the thing

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
#driver.find_element_by_css_selector("div.battle-controls > p").click()
#driver.find_element_by_xpath("(//button[@name='chooseSwitch'])[4]").click()
#driver.find_element_by_css_selector("div.ps-room.ps-room-opaque").click()

#Main gameplay cycle
log = driver.find_element_by_css_selector('.battle-log')
status = Status(log)
while True:
    
    
    


#Main gameplay cycle
while True:            
    print 'Getting battle log....'
    log = driver.find_element_by_css_selector('.battle-log')
    
    while not is_myturn(log):
        print "Not my turn"
        time.sleep(0.5)
    
    current_turn = get_current_turn(log)
    
    d = np.random.randint(1,4)
    driver.find_element_by_xpath("(//button[@name='chooseMove'])[" + str(d) + "]").click()
    
    while current_turn == get_current_turn(log):
        time.sleep(0.5)
        print "Waiting For Opponent to Make a Move...."
    
        
    


class WdExample(unittest.TestCase):
    def setUp(self):
        self.driver = webdriver.Firefox()
        self.driver.implicitly_wait(30)
        self.base_url = "http://play.pokemonshowdown.com/"
        self.verificationErrors = []
        self.accept_next_alert = True
    
    def test_wd_example(self):
        driver = self.driver
        
        driver.get(self.base_url + "/")
        
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
        #driver.find_element_by_css_selector("div.battle-controls > p").click()
        #driver.find_element_by_xpath("(//button[@name='chooseSwitch'])[4]").click()
        #driver.find_element_by_css_selector("div.ps-room.ps-room-opaque").click()
        
        #Main gameplay cycle
        while True:            
            print 'Getting battle log....'
            driver.find_element_by_css_selector('.battle-log')
            
            d = np.random.randint(1,4)
            try:
                driver.find_element_by_xpath("(//button[@name='chooseMove'])[" + d + "]").click()
            except:
                try:
                    driver.find_element_by_xpath("(//button[@name='chooseMove'])[1]").click()
                except:
                    driver.find_element_by_name("closeAndRematch").click()
                    driver.find_element_by_name("chooseMove").click()
                
            


        #driver.find_element_by_name("openTimer").click()
        #driver.find_element_by_css_selector("div.ps-room.ps-room-opaque").click()
        #driver.find_element_by_name("setTimer").click()
        
    
    def is_element_present(self, how, what):
        try: self.driver.find_element(by=how, value=what)
        except NoSuchElementException as e: return False
        return True
    
    def is_alert_present(self):
        try: self.driver.switch_to_alert()
        except NoAlertPresentException as e: return False
        return True
    
    def close_alert_and_get_its_text(self):
        try:
            alert = self.driver.switch_to_alert()
            alert_text = alert.text
            if self.accept_next_alert:
                alert.accept()
            else:
                alert.dismiss()
            return alert_text
        finally: self.accept_next_alert = True
    
    def tearDown(self):
        self.driver.quit()
        self.assertEqual([], self.verificationErrors)

if __name__ == "__main__":
    unittest.main()
