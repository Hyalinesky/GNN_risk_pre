import csv
from pandas import read_csv
import pandas as pd
import numpy as np

node = pd.read_csv("edge.csv", encoding="utf_8", header=0)
nodewname = pd.read_csv("node&name.csv", encoding="utf_8", header=0)

def is_nan(nan):
    return nan != nan

def change_y():
    nodey = pd.read_csv("node.csv", encoding="utf_8", header=0)
    for i in range(len(nodey)):
        nodewname.loc[i,'y'] = nodey.loc[i,'y']


#改一下y的值
change_y()

#交换一下from和to的顺序
cols = list(node)
cols.insert(0,cols.pop(cols.index('from_id')))
node = node.loc[:,cols]

nodenew=node.copy()
# print(nodewname.loc[433,'Corp_name'])



for i in range(len(node)):
    nodenew.loc[i,'to_id'] = nodewname.loc[node.loc[i,'to_id'],'Corp_name']
    nodenew.loc[i, 'from_id'] = nodewname.loc[node.loc[i, 'from_id'], 'Corp_name']


for i in range(len(node)):
    nodenew.loc[i, 'type'] = '投资'
    nodenew.loc[i, 3] = '企业'
    nodenew.loc[i, 4] = '企业'
    # Rgst_years
    if is_nan(nodewname.loc[node.loc[i, 'from_id'], 'Rgst_years']):
        nodenew.loc[i, 5] = '10'
    else:
        nodenew.loc[i, 5] = str(nodewname.loc[node.loc[i, 'from_id'], 'Rgst_years']).split('.')[0]

    #Corp_Rgst_Cap
    if str(nodewname.loc[node.loc[i, 'from_id'], 'Corp_Rgst_Cap']) == None:
        nodenew.loc[i, 6] = '100'
    else:
        nodenew.loc[i, 6] = ((str(nodewname.loc[node.loc[i, 'from_id'], 'Corp_Rgst_Cap'])).split('.')[0]).replace(',','')

    #Shareholders_Num
    if is_nan(nodewname.loc[node.loc[i, 'from_id'], 'Shareholders_Num']):
        nodenew.loc[i, 7] = '10'
    else:
        nodenew.loc[i, 7] = str(nodewname.loc[node.loc[i, 'from_id'], 'Shareholders_Num']).split('.')[0]

    #Employees_Num
    if is_nan(nodewname.loc[node.loc[i, 'from_id'], 'Employees_Num']):
        nodenew.loc[i, 8] = '100'
    else:
        nodenew.loc[i, 8] = str(nodewname.loc[node.loc[i, 'from_id'], 'Employees_Num']).split('.')[0]

    #y
    if is_nan(nodewname.loc[node.loc[i, 'from_id'], 'Employees_Num']):
        nodenew.loc[i, 9] = '暂无风险'
    else:
        if str(nodewname.loc[node.loc[i, 'from_id'], 'y']).split('.')[0] == '0':
            nodenew.loc[i, 9] = '暂无风险'
        else:
            nodenew.loc[i, 9] = '有风险'



    if is_nan(nodenew.loc[i,'from_id']):
        nodenew = nodenew.drop(i, inplace=False)
    elif is_nan(nodenew.loc[i,'to_id']):
        nodenew = nodenew.drop(i, inplace=False)
print(nodenew)


nodenew.to_csv('relation.txt',index=0,header=0,mode='w')


