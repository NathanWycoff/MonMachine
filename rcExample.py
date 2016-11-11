# -*- coding: utf-8 -*-
from selenium import selenium
import unittest, time, re
import numpy as np

class rcExample(unittest.TestCase):
    def setUp(self):
        self.verificationErrors = []
        self.selenium = selenium("localhost", 4444, "*chrome", "http://play.pokemonshowdown.com/")
        self.selenium.start()
    
    def test_rc_example(self):
        sel = self.selenium
        #Start up the first game
        sel.open("/")
        sel.click("css=div.pad")
        sel.click("name=login")
        sel.click("name=login")
        sel.type("name=username", "monmachine")
        sel.click("css=button[type=\"submit\"]")
        sel.type("name=password", "Stats2016")
        sel.click("css=button[type=\"submit\"]")
        sel.click("name=format")
        sel.click("xpath=(//button[@name='selectFormat'])[51]")
        sel.click("name=search")
        
        #sel.click("css=div.battle-controls > p")
        #sel.click("xpath=(//button[@name='chooseSwitch'])[4]")
        
        while True:
            d = np.random.randint(1,4)#Try something random
            try:
                sel.click("xpath=(//button[@name='chooseMove'])[" + str(d) + "]")
            except:#If it doesn't work, might be hyperbeam?
                try:
                    sel.click("xpath=(//button[@name='chooseMove'])[1]")
                except:
                    sel.click("name=closeAndRematch")
                    sel.click("name=chooseMove")
                

    def tearDown(self):
        self.selenium.stop()
        self.assertEqual([], self.verificationErrors)

if __name__ == "__main__":
    unittest.main()
