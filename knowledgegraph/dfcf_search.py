# # 首先获取整个页面，这里我使用的是urllib,
# import urllib.request as request
# from bs4 import BeautifulSoup	#导入
#
# code=600008
# url = f'https://data.eastmoney.com/gljy/detail/{code}.html'
# headers = {
#         'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36'
#     }
#
# req = request.urlopen(url)  # url为你想获取的页面的url
# index = req.read()  # index为你获取的整个页面
#
# # 接下来使用bs4解析这个页面
# soup = BeautifulSoup(index, 'lxml')
# print(soup.find_all("div", class_="dataview-body"))




# import requests
# from lxml import etree
# import time
# from bs4 import BeautifulSoup	#导入
#
#
# code=600008
# sess = requests.session()
# url = f'https://data.eastmoney.com/gljy/detail/{code}.html'
# headers = {
#         'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36'
#     }
#
#
# # 获取查询到的网页内容（全部）
# details = sess.get(url, headers=headers, timeout=10)
# details.raise_for_status()
# details.encoding = 'utf-8'  # linux utf-8
# details_soup = BeautifulSoup(details.text, features="html.parser")
# message = details_soup.text
# time.sleep(2)
# print(message)




# response = requests.get(url, headers=headers)
# html = response.text
# HTML = etree.HTML(html)
#
# soup = BeautifulSoup(response.text,"lxml")
# div_list1 = soup.find_all('div',class_='dataview-body')
# print(div_list1)

# #法人代表
# FRDB = str(HTML.xpath('/html[1]/body[1]/div[2]/div[8]/div[2]/div[10]/div[1]/div[2]/div[2]/table[1]/tbody[1]/tr[1]/td[2]/div[1]'))
# print(FRDB)




import importlib #提供import语句
import sys
import time #提供延时功能
import xlrd #excel文件读取
import os
import xlwt #excel文件写入
from pandas import read_csv
from csv_2_xls import csv2xls
from xlutils.copy import copy #excel文件复制
from selenium import webdriver #浏览器操作库
from selenium.webdriver.common.by import By

import time
from selenium import webdriver

#伪装成浏览器，防止被识破
option = webdriver.ChromeOptions()
# option.add_argument('--user-agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36"')
option.add_argument("--enable-javascript")
driver = webdriver.Chrome(chrome_options=option)

CODE='600008'

driver.get("https://data.eastmoney.com/gljy/")
time.sleep(1)
# 向搜索框注入文字
driver.find_element(by=By.XPATH,value='/html[1]/body[1]/div[2]/div[8]/div[2]/div[2]/div[2]/div[1]/form[1]/input[1]').send_keys(CODE)
time.sleep(1)
#单击搜索按钮
srh_btn = driver.find_element(by=By.XPATH,value='/html[1]/body[1]/div[2]/div[8]/div[2]/div[2]/div[2]/div[1]/form[1]/input[2]')
srh_btn.click()
time.sleep(1)
print (driver.current_url)  # current_url 方法可以得到当前页面的URL
html_source = driver.page_source
print(html_source)
driver.quit()





