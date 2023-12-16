from py2neo import Graph, Node, Relationship
import json

# 连接到Neo4j数据库
graph = Graph("http://localhost:7474", auth=("neo4j", "weiwuweiwu123"),name="neo4j")
graph.delete_all()
graph.begin()
print("neo4j info: {}".format(str(graph)))

# 读取节点数据
with open('records_node.json', 'r', encoding="utf_8_sig") as node_file:
    nodes_data = json.load(node_file)

# 遍历节点数据，创建节点并添加到数据库
for node_data in nodes_data:
    properties = node_data["n"]["properties"]
    
    try:
        start_identity = node_data["n"]["identity"]
        node = Node(*node_data["n"]["labels"], **properties)
        graph.create(node)
    except KeyError as e:
        print(f"Skipping node with identity {start_identity} due to KeyError: {e}")
        continue

# 读取连边数据
with open('records_link.json', 'r', encoding="utf_8_sig") as link_file:
    links_data = json.load(link_file)

# 遍历连边数据，创建关系并添加到数据库
for link_data in links_data:
    start_identity = link_data["p"]["start"]["identity"]
    end_identity = link_data["p"]["end"]["identity"]
    relationship_type = link_data["p"]["segments"][0]["relationship"]["type"]
    
    try:
        start_node = graph.nodes[start_identity]
        end_node = graph.nodes[end_identity]
        relationship = Relationship(start_node, relationship_type, end_node)
        graph.create(relationship)
    except KeyError as e:
        print(f"Skipping relationship with start identity {start_identity} and end identity {end_identity} due to KeyError: {e}")
        continue

print("数据导入完成")