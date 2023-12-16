import importlib #提供import语句
import sys
import time #提供延时功能
import xlrd #excel文件读取
import os
import xlwt #excel文件写入
from pandas import read_csv
from csv_2_xls import csv2xls
import csv

from xlutils.copy import copy #excel文件复制
from selenium import webdriver #浏览器操作库
from selenium.webdriver.common.by import By

importlib.reload(sys)


def spider():
    info_list = []

    #伪装成浏览器，防止被识破
    option = webdriver.ChromeOptions()
    option.add_argument('--user-agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.146 Safari/537.36"')
    driver = webdriver.Chrome(options=option)

    #打开登录页面
    driver.get('https://www.qichacha.com/user_login')
    time.sleep(100)#等待20s，完成手动登录操作
    # 手动登录操作

    # csv2xls("东方财富网.csv","东方财富网.xls")
    #从excel获取查询单位
    worksheet = xlrd.open_workbook(u'dfcf_getdet_cache.xls')
    sheet1 = worksheet.sheet_by_name("sheet1")#excel有多个sheet，检索该名字的sheet表格
    rows = sheet1.nrows # 获取行数
    inc_list = []
    for i in range(0,rows) :
        data = sheet1.cell_value(i, 1) # 取第2列数据
        inc_list.append(data)
    inc_list = getUniqueItems(inc_list)
    inc_len = len(inc_list)
    print(inc_list)
    print("共" + str(inc_len)+ "个公司")

    #写回数据
    df2 = xlwt.Workbook()
    table2 = df2.add_sheet('sheet1',cell_overwrite_ok=True)

    #开启爬虫
    for i in range(inc_len):
        txt = inc_list[i]
        NAME = txt
        for k in range(inc_len):
            table2.write(k, 1, '')
        for j in range(inc_len-i):
            table2.write(j, 1, inc_list[i+j])
        df2.save('dfcf_getdet_cache.xls')
        # txt='江苏孜航精密五金有限公司'
        time.sleep(1)

        if (i == 0):
            # 向搜索框注入文字
            driver.find_element(by=By.XPATH,
                                value='/html[1]/body[1]/div[1]/div[2]/section[1]/div[1]/div[1]/div[1]/div[1]/div[1]/div[1]/input[1]').send_keys(
                txt)
            # 单击搜索按钮
            srh_btn = driver.find_element(by=By.XPATH,
                                          value='/html[1]/body[1]/div[1]/div[2]/section[1]/div[1]/div[1]/div[1]/div[1]/div[1]/div[1]/span[1]/button[1]')
            srh_btn.click()
        else:
            # 清楚搜索框内容
            driver.find_element(by=By.XPATH,
                                value='/html[1]/body[1]/div[1]/div[1]/div[1]/div[1]/div[1]/div[1]/div[1]/div[1]/input[1]').clear()
            time.sleep(1)
            # 向搜索框注入下一个公司地址
            driver.find_element(by=By.XPATH,
                                value='/html[1]/body[1]/div[1]/div[1]/div[1]/div[1]/div[1]/div[1]/div[1]/div[1]/input[1]').send_keys(txt)
            time.sleep(2)
            # 搜索按钮
            srh_btn = driver.find_element(by=By.XPATH,
                                          value='/html[1]/body[1]/div[1]/div[1]/div[1]/div[1]/div[1]/div[1]/div[1]/div[1]/span[1]/button[1]')
            srh_btn.click()

        time.sleep(3)
        try:
            # 获取网页地址，进入
            inner = driver.find_element(by=By.XPATH,value='/html[1]/body[1]/div[1]/div[2]/div[2]/div[3]/div[1]/div[2]/div[1]/table[1]/tr[1]/td[3]/div[1]/span[1]/span[1]/a[1]').get_attribute('href')
            driver.get(inner)
            time.sleep(2)
            # 弹出框按钮
            try:
                try:
                    # 成立年数
                    CLNS = driver.find_element(by=By.XPATH,value='/html[1]/body[1]/div[1]/div[2]/div[5]/div[1]/div[1]/div[2]/section[2]/div[2]/table[1]/tr[2]/td[6]/span[1]/span[1]').text
                    CLNS = str(2022-int(CLNS[0:4]))
                    # 注册资本
                    ZCZB = driver.find_element(by=By.XPATH,
                                               value='/html[1]/body[1]/div[1]/div[2]/div[5]/div[1]/div[1]/div[2]/section[2]/div[2]/table[1]/tr[3]/td[2]').text
                    ZCZB = ZCZB.rstrip('万元人民币')
                    # 实缴资本
                    SJZB = driver.find_element(by=By.XPATH,
                                               value='/html[1]/body[1]/div[1]/div[2]/div[5]/div[1]/div[1]/div[2]/section[2]/div[2]/table[1]/tr[3]/td[4]').text
                    SJZB = SJZB.rstrip('万元人民币')
                    #股东人数
                    try:
                        GDRS = driver.find_element(by=By.XPATH,
                                               value='/html[1]/body[1]/div[1]/div[2]/div[5]/div[1]/div[1]/div[2]/section[3]/div[2]/div[1]/span[1]/span[1]/a[1]/span[2]').text
                    except:
                        GDRS = '1'
                    #企业标签
                    try:
                        Label = driver.find_element(by=By.XPATH,
                                               value='/html[1]/body[1]/div[1]/div[2]/div[2]/div[1]/div[1]/div[1]/div[2]/div[1]/div[2]/div[1]/div[5]/div[2]/span[1]/span[1]/span[1]').text
                    except:
                        Label = ''
                    #小型企业S标签
                    if Label == '小型':
                        Slabel = '1'
                    else:
                        Slabel = '0'
                    #微型企业XS标签
                    if Label == '微型':
                        XSlabel = '1'
                    else:
                        XSlabel = '0'
                    #分支机构数
                    try:
                        FZJGS = driver.find_element(by=By.XPATH,
                                                    value='/html[1]/body[1]/div[1]/div[2]/div[5]/div[1]/div[1]/div[1]/a[7]').text
                        FZJGS = FZJGS.lstrip('分支机构')
                        FZJGS = FZJGS.strip()
                    except:
                        FZJGS = '0'
                    #控制企业数
                    try:
                        KZQYS = driver.find_element(by=By.XPATH,
                                                    value='/html[1]/body[1]/div[1]/div[2]/div[5]/div[1]/div[1]/div[1]/a[12]').text
                        KZQYS = KZQYS.ltrip('控制企业')
                        KZQYS = KZQYS.strip(' ')
                    except:
                        KZQYS = '0'
                    #间接持股企业数
                    try:
                        JJCGQYS = driver.find_element(by=By.XPATH,
                                                    value='/html[1]/body[1]/div[1]/div[2]/div[5]/div[1]/div[1]/div[1]/a[13]').text
                        JJCGQYS = JJCGQYS.lstrip('间接持股企业')
                        JJCGQYS = JJCGQYS.strip(' ')
                    except:
                        JJCGQYS = '0'
                    #全球关联企业数
                    try:
                        QQGLQYS = driver.find_element(by=By.XPATH,
                                                    value='/html[1]/body[1]/div[1]/div[2]/div[5]/div[1]/div[1]/div[1]/a[17]').text
                        QQGLQYS = QQGLQYS.lstrip('全球关联企业')
                        QQGLQYS = QQGLQYS.strip(' ')

                    except:
                        QQGLQYS = '0'
                    #协同股东数
                    try:
                        XTGDS = driver.find_element(by=By.XPATH,
                                                    value='/html[1]/body[1]/div[1]/div[2]/div[5]/div[1]/div[1]/div[1]/a[18]').text
                        XTGDS = XTGDS.lstrip('协同股东')
                        XTGDS = XTGDS.strip(' ')
                    except:
                        XTGDS = '0'

                    time.sleep(2)

                    #经营异常
                    try:
                        srh_btn = driver.find_element(by=By.XPATH,
                                                      value='/html[1]/body[1]/div[1]/div[2]/div[4]/div[1]/div[1]/div[1]/a[3]/h2[1]/div[1]')
                        srh_btn.click()
                        time.sleep(1)
                        try:
                            JYYC = driver.find_element(by=By.XPATH,
                                                        value='/html[1]/body[1]/div[1]/div[2]/div[5]/div[3]/div[1]/div[1]/a[5]').text
                            if JYYC != '经营异常 0':
                                JYYC = '1'
                            else:
                                JYYC = '0'
                        except:
                            JYYC = '0'
                        time.sleep(1)
                    except:
                        JYYC = '0'
                        pass

                    print((inc_len-i),NAME, CLNS, ZCZB, GDRS, XTGDS, Slabel, XSlabel, KZQYS, JJCGQYS, QQGLQYS, JYYC)
                    info = [NAME, CLNS, ZCZB, GDRS, XTGDS, Slabel, XSlabel, KZQYS, JJCGQYS, QQGLQYS, JYYC]
                    info_list.append(info)
                    save_csv(info_list)
                    info_list = []



                    time.sleep(5)
                except:
                    pass
            except:
                pass
        except:
            pass

    driver.close()
    return info_list

def save_csv(info_list):
    #4.存储数据
    #存储路径
    f = open('投资公司扩展data.csv',mode='a',newline='',encoding='utf-8')
    csv_writer=csv.writer(f)
    csv_writer.writerows(info_list)

def getUniqueItems(iterable):
    seen = set()
    result = []
    for item in iterable:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

def run():
    info_list = spider()
    # save_csv(info_list)

if __name__=="__main__":
    run()