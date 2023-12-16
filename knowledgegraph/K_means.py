import numpy as np
import pandas as pd

node = pd.read_csv(r'node.csv', encoding="utf_8", header=0)
data = np.array(node.iloc[:,1:97],type(float))
from sklearn.cluster import KMeans

X = data
# 实例化K-Means算法模型，先使用12个簇尝试聚类
cluster = KMeans(n_clusters=12, random_state=0)
# 使用数据集X进行训练
cluster = cluster.fit(X)
# 调用属性labels_，查看聚类结果
print(cluster.labels_)
print(pd.value_counts(cluster.labels_))
print(cluster.cluster_centers_)

# y_list=[]
# for i in range(data.shape[0]):
#     if cluster.labels_[i] != 0:
#         node['y'][i] = 1
#     else:
#         node['y'][i] = 0
# node.to_csv('nodenew.csv', header=list(node), index=0, encoding='utf_8')

