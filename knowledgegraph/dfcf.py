import requests
import csv
import numpy as np
import json
from csv_2_xls import csv2xls


#用户代理
headers={
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36'
}


def get_resp(url):
    response = requests.get(url)
    txt_data = response.text
    json_data = txt_data
    for i in txt_data:
        if i !='{':
            json_data = json_data.lstrip(i)
        else:
            break
    json_data=json_data.rstrip(');')
    json_data=json.loads(json_data)
    print (json_data)
    return json_data

def parse_json(json_data):
    data_list=json_data['result']['data']
    info_list=[]
    for i in range(0, len(data_list)):
        NOTICE_DATE = data_list[i]['NOTICE_DATE']  #日期
        NOTICE_DATE = NOTICE_DATE[0:4]
        if NOTICE_DATE == '2022':
            SECURITY_CODE = data_list[i]['SECURITY_CODE']
            IS_CONTROL = data_list[i]['IS_CONTROL']

            RELATED_PARTY = data_list[i]['RELATED_PARTY']
            RELATED_PARTY = RELATED_PARTY.split('及其')[0]
            RELATED_PARTY = RELATED_PARTY.split(',')[0]


            RELATED_RELATION = data_list[i]['RELATED_RELATION']
            if RELATED_RELATION == '实际控制人':
                RELATED_PARTY = ''

            TRADE_AMT = data_list[i]['TRADE_AMT']
            print(SECURITY_CODE, RELATED_PARTY)
            info = [SECURITY_CODE, RELATED_PARTY]
            info_list.append(info)
    return info_list

def save_csv(info_list):
    #4.存储数据
    #存储路径
    f = open('东方财富网.csv',mode='a',newline='',encoding='utf-8')
    csv_writer=csv.writer(f)
    csv_writer.writerows(info_list)

def run(url):
    json_data=get_resp(url)
    info_list=parse_json(json_data)
    save_csv(info_list)


if __name__ == "__main__":
    with open("网易财经.csv", "rt",encoding='UTF-8') as csvfile:
        reader = csv.reader(csvfile)
        column = [row[0] for row in reader]
        for CODE in column:
            url = f'https://datacenter-web.eastmoney.com/api/data/v1/get?sortColumns=NOTICE_DATE&sortTypes=-1&pageSize=50&pageNumber=1&reportName=RPT_RELATED_TRADE&columns=SECURITY_CODE%2CSECURITY_NAME_ABBR%2CRELATED_PARTY%2CTRADESECURITY_CODE%2CTRADESECURITY_NAME_ABBR%2CRELATED_RELATION%2CTRADE_AMT%2CCURRENCY%2CCURRENCY_NAME%2CTRADE_PROFILE%2CTRADE_WAY%2CNETPROFIT%2CDATE_TYPE_CODE%2CNOTICE_DATE%2CPAY_WAY%2COPERATE_INCOME%2CEID%2CIS_CONTROL&filter=(SECURITY_CODE%3D%22{CODE}%22)&source=WEB&client=WEB'
            # f''构造成变量，其中page={page}更新数字
            json_data=get_resp(url)
            try:
                pages = json_data['result']['pages']
            except:
                pages = 0
            for page in range(1,pages+1):
                url = f'https://datacenter-web.eastmoney.com/api/data/v1/get?sortColumns=NOTICE_DATE&sortTypes=-1&pageSize=50&pageNumber={page}&reportName=RPT_RELATED_TRADE&columns=SECURITY_CODE%2CSECURITY_NAME_ABBR%2CRELATED_PARTY%2CTRADESECURITY_CODE%2CTRADESECURITY_NAME_ABBR%2CRELATED_RELATION%2CTRADE_AMT%2CCURRENCY%2CCURRENCY_NAME%2CTRADE_PROFILE%2CTRADE_WAY%2CNETPROFIT%2CDATE_TYPE_CODE%2CNOTICE_DATE%2CPAY_WAY%2COPERATE_INCOME%2CEID%2CIS_CONTROL&filter=(SECURITY_CODE%3D%22{CODE}%22)&source=WEB&client=WEB'
                run(url)

    csv2xls('东方财富网.csv', '东方财富网.xls')
    # csv2xls("东方财富网.csv","dfcf_getdet_cache.xls")
    # csv2xls("东方财富网.csv", "get_comname_cache.xls")


    # CODE=600007
    # url = f'https://datacenter-web.eastmoney.com/api/data/v1/get?sortColumns=NOTICE_DATE&sortTypes=-1&pageSize=50&pageNumber=1&reportName=RPT_RELATED_TRADE&columns=SECURITY_CODE%2CSECURITY_NAME_ABBR%2CRELATED_PARTY%2CTRADESECURITY_CODE%2CTRADESECURITY_NAME_ABBR%2CRELATED_RELATION%2CTRADE_AMT%2CCURRENCY%2CCURRENCY_NAME%2CTRADE_PROFILE%2CTRADE_WAY%2CNETPROFIT%2CDATE_TYPE_CODE%2CNOTICE_DATE%2CPAY_WAY%2COPERATE_INCOME%2CEID%2CIS_CONTROL&filter=(SECURITY_CODE%3D%22{CODE}%22)&source=WEB&client=WEB'
    # # f''构造成变量，其中page={page}更新数字
    # json_data = get_resp(url)
    # pages = json_data['result']['pages']
    # for page in range(1, pages+1):
    #     url = f'https://datacenter-web.eastmoney.com/api/data/v1/get?sortColumns=NOTICE_DATE&sortTypes=-1&pageSize=50&pageNumber={page}&reportName=RPT_RELATED_TRADE&columns=SECURITY_CODE%2CSECURITY_NAME_ABBR%2CRELATED_PARTY%2CTRADESECURITY_CODE%2CTRADESECURITY_NAME_ABBR%2CRELATED_RELATION%2CTRADE_AMT%2CCURRENCY%2CCURRENCY_NAME%2CTRADE_PROFILE%2CTRADE_WAY%2CNETPROFIT%2CDATE_TYPE_CODE%2CNOTICE_DATE%2CPAY_WAY%2COPERATE_INCOME%2CEID%2CIS_CONTROL&filter=(SECURITY_CODE%3D%22{CODE}%22)&source=WEB&client=WEB'
    #     run(url)
















