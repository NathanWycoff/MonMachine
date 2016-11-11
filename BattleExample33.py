# -*- coding: utf-8 -*-
from selenium import selenium
import unittest, time, re

class BattleExample33(unittest.TestCase):
    def setUp(self):
        self.verificationErrors = []
        self.selenium = selenium("localhost", 4444, "*chrome", "http://play.pokemonshowdown.com/")
        self.selenium.start()
    
    def test_battle_example33(self):
        sel = self.selenium
        sel.open("/")
        sel.click("name=format")
        sel.click("xpath=(//button[@name='selectFormat'])[51]")
        sel.click("name=search")
        sel.click("css=div.ps-room.ps-room-opaque")
        sel.click("css=div.battle-log")
        sel.click("xpath=(//textarea[@type='text'])[2]")
        sel.click("name=chooseMove")
        sel.click("xpath=(//button[@name='chooseMove'])[3]")
        sel.click("name=chooseSwitch")
        sel.click("xpath=(//button[@name='chooseMove'])[2]")
        sel.click("name=chooseSwitch")
        sel.click("xpath=(//button[@name='chooseMove'])[3]")
        sel.click("xpath=(//button[@name='chooseMove'])[4]")
        sel.click("xpath=(//button[@name='chooseMove'])[2]")
        sel.click("css=div.ps-room.ps-room-opaque")
        sel.click("name=chooseMove")
        sel.click("css=div.switchselect")
        sel.click("name=chooseMove")
        sel.click("xpath=(//button[@name='chooseMove'])[4]")
        sel.click("css=div.switchselect")
        sel.click("xpath=(//button[@name='chooseMove'])[4]")
        sel.click("xpath=(//button[@name='chooseMove'])[3]")
        sel.click("xpath=(//button[@name='chooseMove'])[3]")
        sel.click("xpath=(//button[@name='chooseMove'])[2]")
        sel.click("name=chooseMove")
        sel.click("xpath=(//button[@name='chooseMove'])[3]")
        sel.click("xpath=(//button[@name='chooseMove'])[3]")
        sel.click("xpath=(//button[@name='chooseMove'])[3]")
        sel.click("xpath=(//button[@name='chooseMove'])[3]")
        sel.click("xpath=(//button[@name='chooseMove'])[3]")
        sel.click("xpath=(//button[@name='chooseMove'])[3]")
        sel.click("xpath=(//button[@name='chooseMove'])[4]")
        sel.click("xpath=(//button[@name='chooseMove'])[4]")
        sel.click("xpath=(//button[@name='chooseSwitch'])[2]")
        sel.click("xpath=(//button[@name='chooseMove'])[4]")
        sel.click("name=chooseSwitch")
        sel.click("xpath=(//button[@name='chooseMove'])[3]")
        sel.click("xpath=(//textarea[@type='text'])[2]")
        sel.click("name=closeAndRematch")
        sel.click("name=search")
        sel.click("name=cancelSearch")
        sel.click("name=chooseMove")
        sel.click("name=chooseMove")
    
    def tearDown(self):
        self.selenium.stop()
        self.assertEqual([], self.verificationErrors)

if __name__ == "__main__":
    unittest.main()
