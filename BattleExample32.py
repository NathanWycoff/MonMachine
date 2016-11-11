# -*- coding: utf-8 -*-
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import NoAlertPresentException
import unittest, time, re

class BattleExample32(unittest.TestCase):
    def setUp(self):
        self.driver = webdriver.Firefox()
        self.driver.implicitly_wait(30)
        self.base_url = "http://play.pokemonshowdown.com/"
        self.verificationErrors = []
        self.accept_next_alert = True
    
    def test_battle_example32(self):
        driver = self.driver
        driver.get(self.base_url + "/")
        time.sleep(10)
        driver.find_element_by_name("format").click()
        driver.find_element_by_xpath("(//button[@name='selectFormat'])[51]").click()
        driver.find_element_by_name("search").click()
        driver.find_element_by_css_selector("div.ps-room.ps-room-opaque").click()
        driver.find_element_by_css_selector("div.battle-log").click()
        driver.find_element_by_xpath("(//textarea[@type='text'])[2]").click()
        driver.find_element_by_name("chooseMove").click()
        driver.find_element_by_xpath("(//button[@name='chooseMove'])[3]").click()
        driver.find_element_by_name("closeAndRematch").click()
        driver.find_element_by_name("search").click()
        driver.find_element_by_name("cancelSearch").click()
    
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
