# 企业风险预测
文件树:<br>
1)  app.py是整个系统的主入口<br>
2)  templates文件夹是HTML的页面<br>
     |-index.html 欢迎界面<br> 
     |-search.html 搜索企业关系页面<br>
     |-all_relation.html 所有企业关系页面<br>
3)  static文件夹存放css和js，是页面的样式和效果的文件<br>
4)  raw_data文件夹是存在数据处理后的三元组文件<br>
5)  neo_db文件夹是知识图谱构建模块<br>
     |-config.py 配置参数<br>
     |-create_graph.py 创建知识图谱，图数据库的建立<br>
     |-query_graph.py 知识图谱的查询<br>
<hr>

部署步骤：<br>
* 0.安装所需的库 执行pip install -r requirement.txt<br>
* 1.先下载好neo4j图数据库，并配好环境（注意neo4j需要jdk8）。修改neo_db目录下的配置文件config.py,设置图数据库的账号和密码。<br>
* 2.切换到neo_db目录下，执行python  create_graph.py 建立知识图谱<br>
* 5.运行python app.py,浏览器打开localhost:5000即可查看<br>
