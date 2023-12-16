import pandas as pd
def changecsv():
    df_list = [] #创建新列表用来存储提取出来的列表

    df = pd.read_csv("公司名称.csv")##读取CSV文件数据
    data1 = df.iloc[:,0]#选取文件中1列数据
    df_list.append(data1)#将选取的数据添加到列表

    df = pd.read_csv("东方财富网.csv")##读取CSV文件数据
    data2 = df.iloc[:,1:5]#选取文件中某行某列数据
    df_list.append(data2)#将选取的数据添加到列表

    df2 = pd.concat(df_list,axis=1)#将列表数据按列合并，axis=1表示按列整合
    df2.to_csv("关联交易.csv",index=False)#将整合好的数据输入到新建的csv文件中

    print('csv changed!')
