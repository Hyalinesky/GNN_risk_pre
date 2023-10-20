from py2neo import Graph, Node, Relationship,NodeMatcher
from config import graph

with open("../raw_data/relation.txt",encoding='utf8') as f:
    for line in f.readlines():
        rela_array=line.strip("\n").split(",")
        print(rela_array)
        node_matcer = NodeMatcher(graph)
        find_node = node_matcer.match(Name=rela_array[0]).first()  # 先寻找有无该节点
        if find_node is None:
            graph.run("MERGE(p: Company{cate:'%s',Name: '%s',Rgst_years: '%s',Corp_Rgst_Cap: '%s',Shareholders_Num: '%s',Employees_Num: '%s',Warning: '%s'})"%(rela_array[3],rela_array[0],rela_array[5],rela_array[6],rela_array[7],rela_array[8],rela_array[9]))

        find_node = node_matcer.match(Name=rela_array[1]).first()  # 先寻找有无该节点
        if find_node is None:
            graph.run("MERGE(p: Company{cate:'%s',Name: '%s',Rgst_years: '%s',Corp_Rgst_Cap: '%s',Shareholders_Num: '%s',Employees_Num: '%s',Warning: '%s'})"%(rela_array[4],rela_array[1],rela_array[5],rela_array[6],rela_array[7],rela_array[8],rela_array[9]))

        graph.run(
            "MATCH(e: Company), (cc: Company) \
            WHERE e.Name='%s' AND cc.Name='%s'\
            CREATE(e)-[r:%s{relation: '%s'}]->(cc)\
            RETURN r" % (rela_array[0], rela_array[1], rela_array[2],rela_array[2])

        )
        
