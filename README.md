# 基于大数据和知识图谱的企业风险预测平台

1. 文件树

```
1.│  app.py  系统的主入口
│           
2.├─analysis  存放前期数据分析的代码及结果
│          
3.├─data  存放处理后的图数据
│  └─graph
│      ├─processed  封装后的数据集
│      └─raw  原始数据
4.├─GNN  存放图神经网络模型及训练代码 包括GCN、GCNmf、GAT等
│          
5.├─knowledgegraph  存放爬虫处理数据的过程及中间结果，以及将图数据存入neo4j数据库的代码
│          
6.├─neo_db  网页端知识图谱构建模块
│          
7.├─raw_data  存放网页端图数据处理后的三元组文件
│      relation.txt
│      
8.├─static  存放css和js，是页面的样式和效果的文件
│          
9.├─templates
│      .DS_Store
│      all_relation.html  所有企业关系
│      index.html  开始界面
│      search.html  搜索界面
│      
10.└─校徽  校徽图标库
```

2. 操作流程

   - 使用平台

     1. 下载安装所需要的库 pip install -r requirement.txt
     2. 下载好neo4j图数据库，并配好环境（注意neo4j需要jdk8）。修改neo_db目录下的配置文件config.py,设置图数据库的账号和密码。
     3. 切换到neo_db目录下，执行python create_graph.py 建立知识图谱
     4. 运行python app.py,浏览器打开localhost:5000即可查看

   - 训练模型

     1. 安装pytorch和torch_geometric (以torch1.9 cpu版本为例)

        1. 安装torch1.9（推荐离线）：

           安装教程：https://blog.csdn.net/qq_37541097/article/details/117993519

           离线安装包：https://download.pytorch.org/whl/torch_stable.html

        2. 安装torchgeometric

        ```bash
        conda install pyg -c pyg -c conda-forge
        ```

        ​	3. 下载torchgeometric其他工具离线安装包

        ​		https://pytorch-geometric.com/whl/
        ​		安装四个包，注意对应关系

        4. pip install XXX.whl

     2. （可选）运行GNN/dataset.py 创建数据集

     3. 运行GNN/GCN.py 训练

        



