from py2neo import Graph
import codecs
import os
import json
import base64
import pandas as pd

graph = Graph(
    "http://localhost:7474",
    auth=("neo4j",
    "weiwuweiwu123")
)
CA_LIST = {"暂无风险":0,"有风险":1,}
similar_words = {
    "企业": "企业",
}

json_data = {'data': [], "links": []}

# records = pd.read_json(r'records.json')
f = open('records_link.json','r',encoding="utf_8_sig")
link = json.load(f)
# df = pd.DataFrame(data)
f.close()
begin_id = link[0]['p']['start']['identity']
for i in range(len(link)):
    link_item = {}
    link_item['source'] = link[i]['p']['start']['identity']-begin_id
    link_item['target'] = link[i]['p']['end']['identity']-begin_id
    link_item['value'] = '投资'
    json_data['links'].append(link_item)

f = open('records_node.json','r',encoding="utf_8_sig")
node = json.load(f)
f.close()
for i in range(len(node)):
    data_item = {}
    data_item['name'] = node[i]['n']['properties']['Name']
    data_item['category'] = CA_LIST[node[i]['n']['properties']['Warning']]
    data_item['Rgst_years'] = node[i]['n']['properties']['Rgst_years']
    data_item['Corp_Rgst_Cap'] = node[i]['n']['properties']['Corp_Rgst_Cap']
    data_item['Shareholders_Num'] = node[i]['n']['properties']['Shareholders_Num']
    data_item['Employees_Num'] = node[i]['n']['properties']['Employees_Num']
    data_item['Warning'] = node[i]['n']['properties']['Warning']
    json_data['data'].append(data_item)

print(json_data)

f = open('data.json', 'w',encoding="utf_8_sig")
f.write(str(json_data).replace('\'','\"'))
f.close()
