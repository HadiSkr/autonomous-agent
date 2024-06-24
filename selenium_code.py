from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

from selenium.webdriver.common.action_chains import ActionChains

driver = webdriver.Chrome()

driver.implicitly_wait(10)

driver.get('https://stackoverflow.com')
driver.find_element(By.XPATH, "//textarea[@name='q'] | //input[@name='q']").send_keys("\ue009a")
driver.find_element(By.XPATH, "//textarea[@name='q'] | //input[@name='q']").send_keys(Keys.END)
driver.find_element(By.XPATH, "//textarea[@name='q'] | //input[@name='q']").send_keys("core JavaScript problems")
driver.find_element(By.XPATH, "//textarea[@name='q'] | //input[@name='q']").send_keys(Keys.ENTER)
driver.find_element(By.XPATH, "//textarea[@name='g-recaptcha-response'] | //input[@name='g-recaptcha-response']").send_keys("\ue009a")

driver.quit()
