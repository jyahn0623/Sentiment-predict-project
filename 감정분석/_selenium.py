from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import NoSuchElementException
import csv
import os, sys
import time

driver = webdriver.Chrome('C:/Users/AhnJuyoung/Downloads/chromedriver_win32 (1)/chromedriver')
driver.implicitly_wait(3)
stop_label = '더보기'
writer = csv.writer(open(sys.argv[1], 'a', newline=''))
f = open(sys.argv[2], 'r')
for url in f.readlines():
    t_url = url.replace('\n', '')
    driver.get(t_url)
    try:
        driver.find_element_by_class_name("pi_btn_count").click()

        while True:
            if driver.find_element_by_class_name('u_cbox_page_more').text == stop_label:
                print("More datas!")
                driver.execute_script("arguments[0].click();", driver.find_element_by_class_name('u_cbox_page_more'))
            else:
                break        
        comments = driver.find_elements_by_class_name('u_cbox_contents')
        for c in comments:
            writer.writerow([c.text])
            print(c.text)
    except:
        pass
    time.sleep(1)
    
