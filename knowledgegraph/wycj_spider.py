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

importlib.reload(sys)

def get_code(infile,outfile):
    info_list = []
    #伪装成浏览器，防止被识破
    option = webdriver.ChromeOptions()
    option.add_argument('--user-agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.146 Safari/537.36"')
    driver = webdriver.Chrome(options=option)

    #打开登录页面
    driver.get('http://quotes.money.163.com/f10/zycwzb_300898.html#01c01')
    time.sleep(5)#等待20s，完成手动登录操作
    # 手动登录操作


    #从excel获取查询单位

    # worksheet = xlrd.open_workbook(u'get_comname_cache.xls')
    worksheet = xlrd.open_workbook(infile)
    sheet1 = worksheet.sheet_by_name("sheet1")#excel有多个sheet，检索该名字的sheet表格
    rows = sheet1.nrows # 获取行数
    inc_list = []
    for i in range(0,rows) :
        col1 = sheet1.cell_value(i, 0) # 取第1列数据
        data = sheet1.cell_value(i, 1) # 取第2列数据
        inc_list.append(data)
    inc_list = getUniqueItems(inc_list)
    print(inc_list)
    inc_len = len(inc_list)

    # #写回数据
    # df2 = xlwt.Workbook()
    # table2 = df2.add_sheet('sheet1', cell_overwrite_ok=True)

    #开启爬虫
    for i in range(inc_len):
        txt = inc_list[i]
        # for k in range(inc_len):
        #     table2.write(k, 1, '')
        # for j in range(inc_len-i):
        #     table2.write(j, 1, inc_list[i+j])
        # df2.save('get_comname_cache.xls')
        time.sleep(1)


        #清楚搜索框内容
        driver.find_element(by=By.XPATH,value='/html[1]/body[1]/div[2]/div[1]/div[2]/div[3]/input[2]').clear()
        # 向搜索框注入下一个公司地址
        driver.find_element(by=By.XPATH,value='/html[1]/body[1]/div[2]/div[1]/div[2]/div[3]/input[2]').send_keys(txt)
        # #搜索按钮
        # srh_btn = driver.find_element(by=By.XPATH,value='/html[1]/body[1]/div[1]/div[1]/div[1]/div[1]/div[1]/div[1]/div[1]/div[1]/span[1]/button[1]')
        # srh_btn.click()
        try:
            code = driver.find_element(by=By.XPATH,value='/html[1]/body[1]/div[5]/table[1]/tbody[1]/tr[1]/td[1]').text
            time.sleep(1)

        except:
            code = ''
        print(code)
        info=[code]
        info_list.append(info)
        save_csv(info_list,outfile)
        info_list=[]

    driver.close()



def save_csv(info_list,outfile):
    f = open(outfile, mode='a', newline='', encoding='utf-8')
    csv_writer = csv.writer(f)
    csv_writer.writerows(info_list)

def getUniqueItems(iterable):
    seen = set()
    result = []
    for item in iterable:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

if __name__=="__main__":

    info_list=get_code('东方财富网.xls','cache.csv')
    csv2xls('cache.csv','cache.xls')
