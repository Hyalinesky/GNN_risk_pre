# 图神经网络预测企业风险

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
11.images 存放镜像
```

2. 操作流程

   - 使用平台

     1. 下载安装所需要的库 pip install -r requirement.txt
     2. 下载好neo4j图数据库，并配好环境（注意neo4j需要jdk8）。修改neo_db目录下的配置文件config.py,设置图数据库的账号和密码。
     3. 数据存入neo4j。进入knowledgegraph目录运行json2neo4j.py，将企业数据存入neo4j。也可以使用自己的数据。
     4. 切换到neo_db目录下，执行python create_graph.py 建立知识图谱
     5. 运行python app.py,浏览器打开localhost:5000即可查看

   - 训练模型

     1. 安装pytorch和torch_geometric (以torch1.9 cpu版本为例)

        1. 安装torch1.9（推荐离线）：

           安装教程：https://blog.csdn.net/qq_37541097/article/details/117993519

           离线安装包：https://download.pytorch.org/whl/torch_stable.html

        2. 安装torchgeometric

            conda install pyg -c pyg -c conda-forge

        3. 下载torchgeometric其他工具离线安装包

            https://pytorch-geometric.com/whl/
            ​安装四个包，注意对应关系

        4. pip install XXX.whl

     2. （可选）运行GNN/dataset.py 创建自己的数据集

     3. 运行GCN.py 训练

        

3. 镜像

   由于镜像文件较大，因此上传到百度网盘：

   链接：https://pan.baidu.com/s/1TK3GJ3Gy0atthIUy23GIIA?pwd=hc5a 
   提取码：hc5a 
   --来自百度网盘超级会员V2的分享

   - 平台+数据库

     - 镜像保存在/images中，在当前目录中使用命令，将镜像导入到docker中

       ```shell
       docker load -i images-neo4j.tar
       docker load -i images-web.tar
       ```
     
     - 执行docker-compose.yml
     
       ```shell
       docker-compose up
       ```
     
     - 该过程耗时较长（配置代理之后352.7s左右），并且由于国外镜像的缘故，可能下载速度较慢。
     
       可以为本机配置docker镜像代理
     
     - 访问项目
     
       [http://localhost:5000](http://localhost:5000)
       
       
     


   - GNN模型训练

     - 镜像保存在/images中，在当前目录中使用命令，将镜像导入到docker中

       ```shell
       docker load -i /images/gnn_model.tar
       ```

     - 然后使用docker run来运行镜像

       ```shell
       docker run --name gnns -it gnn_model /bin/bash
       ```

     - 运行完成之后，使用命令训练：

       ```shell
       python3 GCN.py
       ```


​          

​        

​        
