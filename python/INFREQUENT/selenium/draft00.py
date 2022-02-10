from selenium import webdriver
from selenium.webdriver.common.keys import Keys


# example0
browser = webdriver.Chrome()
browser.get('http://seleniumhq.org/')
browser.close()


# example1
browser = webdriver.Chrome()
browser.get('http://www.yahoo.com')
browser.title
elem = browser.find_element_by_name('p')
elem.send_keys('seleniumhq' + Keys.RETURN)
browser.quit()
