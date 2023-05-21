from selenium import webdriver

PATH = "/Applications/chromedriver"
driver = webdriver.Chrome(PATH)

driver.get("https://www.bbc.com/news/business")
while (True):
    pass

