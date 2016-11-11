# -*- coding: utf-8 -*-
from selenium import selenium
import unittest, time, re

class BattleExample2(unittest.TestCase):
    def setUp(self):
        self.verificationErrors = []
        self.selenium = selenium("localhost", 4444, "*chrome", "http://pokemonshowdown.com/")
        self.selenium.start()
    
    def test_battle_example2(self):
        sel = self.selenium
        sel.open("/search?client=ubuntu&channel=fs&q=pokemon+showdown&ie=utf-8&oe=utf-8")
        sel.click(u"link=Pokémon Showdown! battle simulator")
        sel.wait_for_page_to_load("30000")
        sel.click("xpath=(//button[@name='chooseMove'])[4]")
        sel.click("xpath=(//button[@name='chooseSwitch'])[4]")
        sel.click("xpath=(//button[@name='chooseMove'])[4]")
        sel.click("xpath=(//button[@name='chooseMove'])[3]")
        sel.click("name=chooseMove")
        sel.click("xpath=(//button[@name='chooseMove'])[3]")
        sel.click("xpath=(//button[@name='chooseMove'])[3]")
        sel.click("xpath=(//button[@name='chooseMove'])[3]")
        sel.click("xpath=(//div[@onclick='BattleTooltips._handleClickFor(event)'])[6]")
        sel.click("xpath=(//button[@name='chooseMove'])[3]")
        sel.click("xpath=(//button[@name='chooseMove'])[2]")
        sel.click("xpath=(//button[@name='chooseSwitch'])[4]")
        sel.click("xpath=(//button[@name='chooseMove'])[4]")
        sel.click("xpath=(//button[@name='chooseSwitch'])[3]")
        sel.click("name=chooseMove")
        sel.click("name=chooseMove")
        sel.click("name=chooseSwitch")
        sel.click("xpath=(//button[@name='chooseMove'])[3]")
        sel.click("xpath=(//button[@name='chooseMove'])[2]")
        sel.click("xpath=(//button[@name='chooseMove'])[2]")
        sel.click("xpath=(//button[@name='chooseMove'])[3]")
        sel.click("name=chooseSwitch")
        sel.click("name=chooseMove")
        sel.click("name=chooseMove")
        sel.click("xpath=(//button[@name='chooseSwitch'])[3]")
        sel.click("name=chooseMove")
        sel.click("xpath=(//button[@name='chooseMove'])[4]")
        sel.click("name=chooseMove")
        sel.click("name=chooseMove")
        sel.click("xpath=(//button[@name='chooseSwitch'])[2]")
        sel.click("xpath=(//button[@name='chooseMove'])[2]")
    
    def tearDown(self):
        self.selenium.stop()
        self.assertEqual([], self.verificationErrors)

if __name__ == "__main__":
    unittest.main()
