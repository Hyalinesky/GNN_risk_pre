# from bs4 import BeautifulSoup
# import requests
# import time
# from csv_2_xls import csv2xls
#
# # 保持会话
# # 新建一个session对象
# sess = requests.session()
#
# # 添加headers（header为自己登录的企查查网址，输入账号密码登录之后所显示的header，此代码的上方介绍了获取方法）
# afterLogin_headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36'}
#
# # post请求(代表着登录行为，登录一次即可保存，方便后面执行查询指令)
# login = {'user': '18946592236', 'password': 'weiwuweiwu123'}
# sess.post('https://www.qcc.com', data=login, headers=afterLogin_headers)
#
#
# def get_company_message(company):
#     # 获取查询到的网页内容（全部）
#     search = sess.get('https://www.qcc.com/search?key={}'.format(company), headers=afterLogin_headers, timeout=10)
#     search.raise_for_status()
#     search.encoding = 'utf-8'  # linux utf-8
#     soup = BeautifulSoup(search.text, features="html.parser")
#     href = soup.find_all('a', {'class': 'title'})[0].get('href')
#     time.sleep(4)
#     # 获取查询到的网页内容（全部）
#     details = sess.get(href, headers=afterLogin_headers, timeout=10)
#     details.raise_for_status()
#     details.encoding = 'utf-8'  # linux utf-8
#     details_soup = BeautifulSoup(details.text, features="html.parser")
#     message = details_soup.text
#     time.sleep(2)
#     return message
#
#
# import pandas as pd
#
#
# def message_to_df(message, company):
#     # print('message:')
#     # print (message)
#     list_companys = []
#     Registration_status = []
#     Date_of_Establishment = []
#     registered_capital = []
#     contributed_capital = []
#     # Approved_date = []
#     Unified_social_credit_code = []
#     Organization_Code = []
#     companyNo = []
#     Taxpayer_Identification_Number = []
#     sub_Industry = []
#     enterprise_type = []
#     Business_Term = []
#     Registration_Authority = []
#     staff_size = []
#     Number_of_participants = []
#     sub_area = []
#     company_adress = []
#     # Business_Scope = []
#
#     list_companys.append(company)
#     Registration_status.append(message.split('登记状态')[1].split('成立日期')[0].replace(' ', ''))
#     Date_of_Establishment.append(message.split('成立日期：')[1].split('复制')[0].replace(' ', ''))
#     registered_capital.append(message.split('注册资本：')[1].split('人民币')[0].replace(' ', ''))
#     contributed_capital.append(message.split('实缴资本')[1].split('\n')[1].split('人民币')[0].replace(' ', ''))
#     # Approved_date.append(message.split('核准日期')[1].split('\n')[1].replace(' ', ''))
#     try:
#         credit = message.split('统一社会信用代码：')[1].split('复制')[0].split('法定代表人')[0].replace(' ', '')
#         Unified_social_credit_code.append(credit)
#     except:
#         credit = message.split('统一社会信用代码')[2].split('复制')[0].replace(' ', '')
#         Unified_social_credit_code.append(credit)
#     Organization_Code.append(message.split('组织机构代码')[1].split('复制')[0].replace(' ', ''))
#     companyNo.append(message.split('工商注册号')[1].split('复制')[0].replace(' ', ''))
#     Taxpayer_Identification_Number.append(message.split('纳税人识别号')[1].split('复制')[0].replace(' ', ''))
#     try:
#         sub = message.split('所属行业')[1].split('英文名')[0].replace(' ', '')
#         sub_Industry.append(sub)
#     except:
#         sub = ''
#         sub_Industry.append(sub)
#     enterprise_type.append(message.split('企业类型')[1].split('营业期限')[0].replace(' ', ''))
#     Business_Term.append(message.split('营业期限')[1].split('纳税人资质')[0].replace(' ', ''))
#     Registration_Authority.append(message.split('登记机关')[1].split('进出口企业代码')[0].replace(' ', ''))
#     staff_size.append(message.split('人员规模')[1].split('人')[0].replace(' ', ''))
#     try:
#         Number_of_participants.append(message.split('参保人数')[1].split('（')[0].replace(' ', ''))
#     except:
#         Number_of_participants.append('')
#     sub_area.append(message.split('所属地区')[1].split('登记机关')[0].replace(' ', ''))
#     # try:
#     #     adress = message.split('经营范围')[0].split('企业地址')[1].split('查看地图')[0].split('\n')[2].replace(' ', '')
#     #     company_adress.append(adress)
#     # except:
#     #     adress = message.split('经营范围')[1].split('企业地址')[1].split()[0]
#     #     company_adress.append(adress)
#     # Business_Scope.append(message.split('经营范围')[1].split('\n')[1].replace(' ', ''))
#     df = pd.DataFrame({'公司': company, \
#                        '登记状态': Registration_status, \
#                        '成立日期': Date_of_Establishment, \
#                        '注册资本': registered_capital, \
#                        '实缴资本': contributed_capital, \
#                        # '核准日期': Approved_date, \
#                        '统一社会信用代码': Unified_social_credit_code, \
#                        '组织机构代码': Organization_Code, \
#                        '工商注册号': companyNo, \
#                        '纳税人识别号': Taxpayer_Identification_Number, \
#                        '所属行业': sub_Industry, \
#                        '企业类型': enterprise_type, \
#                        '营业期限': Business_Term, \
#                        '登记机关': Registration_Authority, \
#                        '人员规模': staff_size, \
#                        '参保人数': Number_of_participants, \
#                        '所属地区': sub_area, \
#                        # '企业地址': company_adress, \
#                        # '经营范围': Business_Scope
#                        })
#
#     return df
#
#
# # 测试所用
# companys = ['深圳市腾讯计算机系统有限公司','阿里巴巴（中国）有限公司']
#
# # 实际所用
# # df_companys = pd.read_csv('自己目录的绝对路径/某某.csv')
# # companys = df_companys['公司名称'].tolist()
#
#
# for company in companys:
#     try:
#         messages = get_company_message(company)
#         print("try")
#     except:
#         print("pass")
#         pass
#     else:
#         print("else")
#         df = message_to_df(messages, company)
#         if (company == companys[0]):
#             df.to_csv('企查查.csv', index=False, header=True)
#         else:
#             df.to_csv('企查查.csv', mode='a+', index=False, header=False)
#     time.sleep(1)
#
#     # messages = get_company_message(company)
#     # df = message_to_df(messages, company)
#     # if (company == companys[0]):
#     #     df.to_csv('企查查.csv', index=False, header=True)
#     # else:
#     #     df.to_csv('企查查.csv', mode='a+', index=False, header=False)
#     # time.sleep(1)
#
# csv2xls('企查查.csv','企查查.xls')



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


#伪装成浏览器，防止被识破
option = webdriver.ChromeOptions()
option.add_argument('--user-agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.146 Safari/537.36"')
driver = webdriver.Chrome(options=option)

#打开登录页面
driver.get('https://www.qichacha.com/user_login')
time.sleep(20)#等待20s，完成手动登录操作
# 手动登录操作


#从excel获取查询单位

#
# worksheet = xlrd.open_workbook(u'企查查.xls')
# sheet1 = worksheet.sheet_by_name("sheet1")#excel有多个sheet，检索该名字的sheet表格
# rows = sheet1.nrows # 获取行数
# inc_list = []
# for i in range(1,rows) :
#     data = sheet1.cell_value(i, 1) # 取第2列数据
#     inc_list.append(data)
# print(inc_list)
# inc_len = len(inc_list)
#
# #写回数据
# writesheet1 = copy(worksheet)# 这里复制了一个excel，没有直接写回最初的文件。
# writesheet2 = writesheet1.get_sheet(0)#同样获得第一个sheet
# style = xlwt.easyxf('font:height 240, color-index red, bold on;align: wrap on, vert centre, horiz center');

#开启爬虫
inc_len=1
for i in range(inc_len):
    # txt = inc_list[i]
    txt='江苏孜航精密五金有限公司'
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
        # 向搜索框注入下一个公司地址
        driver.find_element(by=By.XPATH,
                            value='/html[1]/body[1]/div[1]/div[1]/div[1]/div[1]/div[1]/div[1]/div[1]/div[1]/input[1]').send_keys(txt)
        # 搜索按钮
        srh_btn = driver.find_element(by=By.XPATH,
                                      value='/html[1]/body[1]/div[1]/div[1]/div[1]/div[1]/div[1]/div[1]/div[1]/div[1]/span[1]/button[1]')
        srh_btn.click()

    time.sleep(1)
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

                print(CLNS, ZCZB, SJZB, GDRS, XTGDS, Slabel, XSlabel, FZJGS, KZQYS, JJCGQYS, QQGLQYS, JYYC)
            except:
                # srh_btn = driver.find_element(by=By.XPATH,value='//*[@id="firstcaseModal"]/div/div/div[2]/button')
                # srh_btn.click()
                pass
        except:
            pass



        # srh_btn = driver.find_element(by=By.XPATH,
        #                               value='/html[1]/body[1]/div[1]/div[2]/div[2]/div[3]/div[1]/div[2]/div[1]/table[1]/tr[1]/td[3]/div[1]/span[1]/span[1]/a[1]')
        # srh_btn.click()
        # print(1)
        # try:
        #     print(2)
        #     # 成立日期
        #     CLRQ = driver.find_element(by=By.XPATH,value='/html[1]/body[1]/div[1]/div[2]/div[5]/div[1]/div[1]/div[2]/section[2]/div[2]/table[1]/tr[2]/td[6]/span[1]/span[1]')
        #     # 注册资本
        #     ZCZB = driver.find_element(by=By.XPATH,
        #                                value='/html[1]/body[1]/div[1]/div[2]/div[5]/div[1]/div[1]/div[2]/section[2]/div[2]/table[1]/tr[3]/td[2]')
        #     # 实缴资本
        #     SJZB = driver.find_element(by=By.XPATH,
        #                                value='/html[1]/body[1]/div[1]/div[2]/div[5]/div[1]/div[1]/div[2]/section[2]/div[2]/table[1]/tr[3]/td[4]')
        #
        #     print(CLRQ, ZCZB, SJZB)
        #     time.sleep(2)
        # except:
        #     print(3)
        #     pass
    except:
        print(2)
        pass
    # print(credit_code)
    # writesheet2.write(i+1, 15, credit_code)  # 第16列数据sheet1.write(i, j, data[j])
    # writesheet1.save(u'test2.xls')

driver.close()
