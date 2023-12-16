# -*- coding: utf-8 -*-
import json
import time
import csv
import pandas as pd

from py2neo import Graph, Node, Relationship,Subgraph
from py2neo import NodeMatcher, RelationshipMatcher

# 连接Neo4j
url = "http://localhost:7474"
username = "neo4j"
password = "weiwuweiwu123"
graph = Graph(url, auth=(username, password),name="neo4j")
graph.delete_all()
graph.begin()
print("neo4j info: {}".format(str(graph)))

#获取公司1
with open('公司Node1.csv','r', encoding="utf-8") as f:
    reader = csv.reader(f)
    Com1 = [row[0] for row in reader]
    print(Com1)

#获取公司2
with open('公司Node2.csv','r', encoding="utf-8") as f:
    reader = csv.reader(f)
    Com2 = [row[0] for row in reader]
    print(Com2)
#获取公司3
with open('公司to公司.csv','r', encoding="utf-8") as f:
    reader = csv.reader(f)
    Com3 = [row[1] for row in reader]
    print(Com3)
#获取公司4
with open('公司to公司.csv','r', encoding="utf-8") as f:
    reader = csv.reader(f)
    Com4 = [row[0] for row in reader]
    print(Com4)


# 创建公司1节点
node_matcer = NodeMatcher(graph)
create_node_cnt = 0
for com in Com1:
    label = u"公司"
    name = com
    find_node = node_matcer.match(label, name=name).first()#先寻找有无该节点
    if find_node is None:
        node = Node(label, name=name)
        graph.create(node)
        create_node_cnt += 1
        print(f"create {create_node_cnt} nodes.")
# 创建公司2节点
node_matcer = NodeMatcher(graph)
create_node_cnt = 0
for com in Com2:
    label = u"公司"
    name = com
    find_node = node_matcer.match(label, name=name).first()#先寻找有无该节点
    if find_node is None:
        node = Node(label, name=name)
        graph.create(node)
        create_node_cnt += 1
        print(f"create {create_node_cnt} nodes.")
# 创建公司3节点
node_matcer = NodeMatcher(graph)
create_node_cnt = 0
for com in Com3:
    label = u"公司"
    name = com
    find_node = node_matcer.match(label, name=name).first()#先寻找有无该节点
    if find_node is None:
        node = Node(label, name=name)
        graph.create(node)
        create_node_cnt += 1
        print(f"create {create_node_cnt} nodes.")
# 创建公司4节点
node_matcer = NodeMatcher(graph)
create_node_cnt = 0
for com in Com4:
    label = u"公司"
    name = com
    find_node = node_matcer.match(label, name=name).first()#先寻找有无该节点
    if find_node is None:
        node = Node(label, name=name)
        graph.create(node)
        create_node_cnt += 1
        print(f"create {create_node_cnt} nodes.")


# 创建关系
create_rel_cnt = 0
relation_matcher = RelationshipMatcher(graph)
#股票名称→地域
for i in range(0,len(Com3)):
    if len(Com3[i]) != 0:
        s_node, s_label = Com3[i], u"公司"
        e_node, e_label = Com4[i], u"公司"
        rel = "投资"
        start_node = node_matcer.match(s_label, name=s_node).first()
        end_node = node_matcer.match(e_label, name=e_node).first()

        if start_node is not None and end_node is not None:
            r_type = relation_matcher.match([start_node, end_node], r_type=rel).first()
            if r_type is None:
                graph.create(Relationship(start_node, rel, end_node))
                create_rel_cnt += 1
                print(f"create {create_rel_cnt} relations.")

