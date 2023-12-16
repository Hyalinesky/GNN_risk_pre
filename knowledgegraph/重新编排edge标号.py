import requests
from lxml import etree
import importlib #提供import语句
import sys
import time #提供延时功能
import xlrd #excel文件读取
import os
import xlwt #excel文件写入
from pandas import read_csv
from csv_2_xls import csv2xls
import csv
from changecsv import changecsv
from xlutils.copy import copy #excel文件复制
from selenium import webdriver #浏览器操作库
from selenium.webdriver.common.by import By
import importlib #提供import语句
from pandas import read_csv
import pandas as pd

from xlutils.copy import copy #excel文件复制
from selenium import webdriver #浏览器操作库
from selenium.webdriver.common.by import By

edge = pd.read_csv("edge.csv", encoding="utf_8", header=0)
node = pd.read_csv("node.csv", encoding="utf_8", header=0)
data = []
key = 0

for i in range(len(edge)):
    for j in range(len(node)):
        if str(edge['to_id'][i]) == str(node['code'][j]):
            edge['to_id'][i] = str(j)
            key = 1
    if key == 0:
        edge = edge.drop(i, inplace=False)
    key = 0
edge.to_csv('edge.csv', header=['to_id', 'from_id'], index=0, encoding='utf_8')

