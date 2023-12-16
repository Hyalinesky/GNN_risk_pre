import requests
import csv
import numpy as np
import get_data1
from csv_2_xls import csv2xls

#用户代理
headers={
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36'
}
#查看网易财经股票总页数
url = f'http://quotes.money.163.com/hs/service/diyrank.php?host=http%3A%2F%2Fquotes.money.163.com%2Fhs%2Fservice%2Fdiyrank.php&page=0&query=STYPE%3AEQA&fields=NO%2CSYMBOL%2CNAME%2CPRICE%2CPERCENT%2CUPDOWN%2CFIVE_MINUTE%2COPEN%2CYESTCLOSE%2CHIGH%2CLOW%2CVOLUME%2CTURNOVER%2CHS%2CLB%2CWB%2CZF%2CPE%2CMCAP%2CTCAP%2CMFSUM%2CMFRATIO.MFRATIO2%2CMFRATIO.MFRATIO10%2CSNAME%2CCODE%2CANNOUNMT%2CUVSNEWS&sort=PERCENT&order=desc&count=24&type=query'
response = requests.get(url, headers=headers)
pagecount=response.json()['pagecount']

def get_resp(url):
    #1.发出请求
    response=requests.get(url,headers=headers)
    #2.获取数据
    #.json()
    #.text
    #.content()   音频/视频/图片
    json_data=response.json()
    return json_data

def parse_json(json_data):
    #3.解析数据
    #json格式中{}内用标签引用(eg.['list'])；[]内用顺序引用(eg.[0])
    data_list=json_data['list']
    info_list=[]
    for i in range(0, len(data_list)):
        CODE = data_list[i]['CODE']
        NAME = data_list[i]['NAME']
        SNAME = data_list[i]['SNAME']
        SYMBOL = data_list[i]['SYMBOL']
        try:
            ANNOUNMT = data_list[i]['ANNOUNMT']
        except:
            ANNOUNMT = ''
        # print(ANNOUNMT)
        if ANNOUNMT != '':
            ANNOUNMT = 1
        else:
            ANNOUNMT = 0
        print(SYMBOL, ANNOUNMT)
        info = [SYMBOL, ANNOUNMT]
        # info.extend(get_data1.get_data(SYMBOL))
        # info.extend(ANNOUNMT)
        info_list.append(info)

        # info_plus=get_data1.get_data(SYMBOL)
        # info_list.append(info_plus)
    return info_list

def save_csv(info_list):
    #4.存储数据
    #存储路径
    f = open('网易财经.csv',mode='a',newline='',encoding='utf-8')
    csv_writer=csv.writer(f)
    csv_writer.writerows(info_list)


def run(url):
    json_data=get_resp(url)
    info_list=parse_json(json_data)
    save_csv(info_list)

if __name__ == "__main__":
    for page in range(0, pagecount):
        url = f'http://quotes.money.163.com/hs/service/diyrank.php?host=http%3A%2F%2Fquotes.money.163.com%2Fhs%2Fservice%2Fdiyrank.php&page={page}&query=STYPE%3AEQA&fields=NO%2CSYMBOL%2CNAME%2CPRICE%2CPERCENT%2CUPDOWN%2CFIVE_MINUTE%2COPEN%2CYESTCLOSE%2CHIGH%2CLOW%2CVOLUME%2CTURNOVER%2CHS%2CLB%2CWB%2CZF%2CPE%2CMCAP%2CTCAP%2CMFSUM%2CMFRATIO.MFRATIO2%2CMFRATIO.MFRATIO10%2CSNAME%2CCODE%2CANNOUNMT%2CUVSNEWS&sort=PERCENT&order=desc&count=24&type=query'
        #f''构造成变量，其中page={page}更新数字
        run(url)
    csv2xls("网易财经.csv", "网易财经.xls")









