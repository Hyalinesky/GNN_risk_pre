from neo_db.config import graph, CA_LIST, similar_words
# from spider.show_profile import get_profile
import codecs
import os
import json
import base64

def query(name):
    data = graph.run(
    "match(p )-[r]->(n:Company{Name:'%s'}) return  p.Name,r.relation,n.Name,p.cate,n.cate,p.Rgst_years,n.Rgst_years,p.Corp_Rgst_Cap,n.Corp_Rgst_Cap,p.Shareholders_Num,n.Shareholders_Num,p.Employees_Num,n.Employees_Num,p.Warning,n.Warning\
        Union all\
    match(p:Company {Name:'%s'}) -[r]->(n) return p.Name, r.relation, n.Name, p.cate, n.cate,p.Rgst_years,n.Rgst_years,p.Corp_Rgst_Cap,n.Corp_Rgst_Cap,p.Shareholders_Num,n.Shareholders_Num,p.Employees_Num,n.Employees_Num,p.Warning,n.Warning" % (name,name)
    )
    data = list(data)
    # print(data)
    return get_json_data(data)

def queryf2(name):
    data = graph.run(
        "match(p1 )-[r1]->(p2)-[r2]->(n:Company{Name:'%s'}) return  p1.Name,r1.relation,p2.Name,r2.relation,n.Name,p1.cate,p2.cate,n.cate,p1.Rgst_years,p2.Rgst_years,n.Rgst_years,p1.Corp_Rgst_Cap,p2.Corp_Rgst_Cap,n.Corp_Rgst_Cap,p1.Shareholders_Num,p2.Shareholders_Num,n.Shareholders_Num,p1.Employees_Num,p2.Employees_Num,n.Employees_Num,p1.Warning,p2.Warning,n.Warning\
            Union all\
        match(p1:Company {Name:'%s'}) -[r1]->(p2)-[r2]->(n) return p1.Name,r1.relation,p2.Name,r2.relation,n.Name,p1.cate,p2.cate,n.cate,p1.Rgst_years,p2.Rgst_years,n.Rgst_years,p1.Corp_Rgst_Cap,p2.Corp_Rgst_Cap,n.Corp_Rgst_Cap,p1.Shareholders_Num,p2.Shareholders_Num,n.Shareholders_Num,p1.Employees_Num,p2.Employees_Num,n.Employees_Num,p1.Warning,p2.Warning,n.Warning" % (
        name, name)
    )
    data = list(data)
    if not data:
        data = graph.run(
            "match(p )-[r]->(n:Company{Name:'%s'}) return  p.Name,r.relation,n.Name,p.cate,n.cate,p.Rgst_years,n.Rgst_years,p.Corp_Rgst_Cap,n.Corp_Rgst_Cap,p.Shareholders_Num,n.Shareholders_Num,p.Employees_Num,n.Employees_Num,p.Warning,n.Warning\
                Union all\
            match(p:Company {Name:'%s'}) -[r]->(n) return p.Name, r.relation, n.Name, p.cate, n.cate,p.Rgst_years,n.Rgst_years,p.Corp_Rgst_Cap,n.Corp_Rgst_Cap,p.Shareholders_Num,n.Shareholders_Num,p.Employees_Num,n.Employees_Num,p.Warning,n.Warning" % (
            name, name)
        )
        data = list(data)
        return get_json_data(data)
    # print(data)
    return get_json_data2(data)

def get_json_data(data):
    json_data={'data':[],"links":[]}
    d=[]
    
    for i in data:
        print(i["p.Name"], i["r.relation"], i["n.Name"], i["p.cate"], i["n.cate"])
        d.append(i['p.Name']+"_"+i['p.cate']+"_"+i['p.Rgst_years']+"_"+i['p.Corp_Rgst_Cap']+"_"+i['p.Shareholders_Num']+"_"+i['p.Employees_Num']+"_"+i['p.Warning'])
        d.append(i['n.Name']+"_"+i['n.cate']+"_"+i['n.Rgst_years']+"_"+i['n.Corp_Rgst_Cap']+"_"+i['n.Shareholders_Num']+"_"+i['n.Employees_Num']+"_"+i['n.Warning'])
        d=list(set(d))
    name_dict={}
    count=0
    for j in d:
        j_array=j.split("_")
    
        data_item={}
        name_dict[j_array[0]]=count
        count+=1
        data_item['name']=j_array[0]
        data_item['category']=CA_LIST[j_array[6]]
        data_item['Rgst_years'] = j_array[2]
        data_item['Corp_Rgst_Cap'] = j_array[3]
        data_item['Shareholders_Num'] = j_array[4]
        data_item['Employees_Num'] = j_array[5]
        data_item['Warning'] = j_array[6]
        json_data['data'].append(data_item)
    for i in data:
   
        link_item = {}
        
        link_item['source'] = name_dict[i['p.Name']]
        
        link_item['target'] = name_dict[i['n.Name']]
        # link_item['value'] = i['r.relation']
        link_item['value'] = '投资'
        json_data['links'].append(link_item)
    # print(json_data)
    return json_data
# f = codecs.open('./static/test_data.json','w','utf-8')
# f.write(json.dumps(json_data,  ensure_ascii=False))

def get_json_data2(data):
    json_data = {'data': [], "links": []}
    d = []

    for i in data:
        # print(i["p.Name"], i["r.relation"], i["n.Name"], i["p.cate"], i["n.cate"])
        d.append(i['p1.Name'] + "_" + i['p1.cate'] + "_" + i['p1.Rgst_years'] + "_" + i['p1.Corp_Rgst_Cap'] + "_" + i[
            'p1.Shareholders_Num'] + "_" + i['p1.Employees_Num'] + "_" + i['p1.Warning'])
        d.append(i['p2.Name'] + "_" + i['p2.cate'] + "_" + i['p2.Rgst_years'] + "_" + i['p2.Corp_Rgst_Cap'] + "_" + i[
            'p2.Shareholders_Num'] + "_" + i['p2.Employees_Num'] + "_" + i['p2.Warning'])
        d.append(i['n.Name'] + "_" + i['n.cate'] + "_" + i['n.Rgst_years'] + "_" + i['n.Corp_Rgst_Cap'] + "_" + i[
            'n.Shareholders_Num'] + "_" + i['n.Employees_Num'] + "_" + i['n.Warning'])
        d = list(set(d))
    name_dict = {}
    count = 0
    for j in d:
        j_array = j.split("_")

        data_item = {}
        name_dict[j_array[0]] = count
        count += 1
        data_item['name'] = j_array[0]
        data_item['category'] = CA_LIST[j_array[6]]
        data_item['Rgst_years'] = j_array[2]
        data_item['Corp_Rgst_Cap'] = j_array[3]
        data_item['Shareholders_Num'] = j_array[4]
        data_item['Employees_Num'] = j_array[5]
        data_item['Warning'] = j_array[6]
        json_data['data'].append(data_item)
    for i in data:
        link_item = {}
        link_item['source'] = name_dict[i['p1.Name']]
        link_item['target'] = name_dict[i['p2.Name']]
        # link_item['value'] = i['r1.relation']
        link_item['value'] = '投资'
        json_data['links'].append(link_item)

        link_item = {}
        link_item['source'] = name_dict[i['p2.Name']]
        link_item['target'] = name_dict[i['n.Name']]
        # link_item['value'] = i['r2.relation']
        link_item['value'] = '投资'
        json_data['links'].append(link_item)
    # print(json_data)
    return json_data

# def get_KGQA_answer(array):
#     data_array=[]
#     for i in range(len(array)-2):
#         if i==0:
#             name=array[0]
#         else:
#             name=data_array[-1]['p.Name']
           
#         data = graph.run(
#             "match(p)-[r:%s{relation: '%s'}]->(n:Company{Name:'%s'}) return  p.Name,n.Name,r.relation,p.cate,n.cate" % (
#                 similar_words[array[i+1]], similar_words[array[i+1]], name)
#         )
       
#         data = list(data)
#         print(data)
#         data_array.extend(data)
        
#         print("==="*36)
#     with open("./spider/images/"+"%s.jpg" % (str(data_array[-1]['p.Name'])), "rb") as image:
#             base64_data = base64.b64encode(image.read())
#             b=str(base64_data)
          
#     return [get_json_data(data_array), get_profile(str(data_array[-1]['p.Name'])), b.split("'")[1]]
# def get_answer_profile(name):
#     with open("./spider/images/"+"%s.jpg" % (str(name)), "rb") as image:
#         base64_data = base64.b64encode(image.read())
#         b = str(base64_data)
#     return [get_profile(str(name)), b.split("'")[1]]



        



