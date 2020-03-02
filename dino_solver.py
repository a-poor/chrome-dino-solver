
from selenium import webdriver
from selenium.webdriver.common.keys import Keys

URL = 'chrome://dino/'
MAIN_CONTENT = '#main-content'


driver = webdriver.Chrome('./chromedriver')



driver.get(URL)






