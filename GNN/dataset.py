import torch
from torch_geometric.data import Data, InMemoryDataset, NeighborSampler
import torch_geometric.transforms as T
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from typing import Optional, Callable, List
import warnings
warnings.filterwarnings('ignore')

class Knowledge_graph(InMemoryDataset):
    def __init__(self, root:str,
                transform: Optional[Callable] = None,
                pre_transform: Optional[Callable] = None):
        super(Knowledge_graph, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, 'raw')
    
    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, 'processed')
    
    @property
    def num_relations(self) -> str:
        # 关系类别数量
        return self.data.edge_type.max().item() + 1
    
    @property
    def num_classes(self) -> str:
        # 节点类别数量
        return self.data.y.max().item() + 1
    
    @property
    def raw_file_names(self) -> str:
        return ['edge', 'node']
    
    @property
    def processed_file_names(self) -> str:
        return 'data.pt'
    
    def download(self):
        pass
    
    def process(self):
        graph_dir = os.listdir(self.root)
        first = True
        num_nodes = 0
        # 显式指定类别编号
        relation_dict = {'tx':0, 'tmt':1, 'invest':2, 'guar':3}

        edge = pd.read_csv(os.path.join(self.root, 'edge.csv'))
        edge['type'] = edge['type'].apply(lambda x:relation_dict[x])#批量化处理将type转成索引表中的int数字

        src = torch.from_numpy(edge['from_id'].values).type(torch.long) + num_nodes
        dst = torch.from_numpy(edge['to_id'].values).type(torch.long) + num_nodes
        edge_1, edge_2 = torch.stack([src, dst], dim=0), torch.stack([dst, src], dim=0)#edge_1表示src指向dst的一个大小为(2,num_edges)的张量，第一列中的每个元素对应指向第二列的每个元素
        edge_index = torch.cat([edge_1,edge_2], dim=1)#得到边的COO稀疏矩阵,这个矩阵将from->to和to->from都加入了index表中，作用是将单向边变为双向边

        rel = torch.from_numpy(edge['type'].values).type(torch.long)
        edge_type = torch.cat([2 * rel, 2 * rel + 1], dim=0)#？映射

        node = pd.read_csv(os.path.join(self.root, 'node.csv'))
        # node = node.iloc[:,1:].astype(float)  # numpy强制类型转换
        x = torch.from_numpy(np.array(node.iloc[:,1:])).type(torch.float32)
        y = torch.from_numpy(node['y'].values).type(torch.long)

        if first:
            first = False
            data = Data(x=x, edge_index=edge_index, y=y)
            data.edge_type = edge_type
        else:
            data.x = torch.cat([data.x, x], dim=0)
            data.edge_index = torch.cat([data.edge_index, edge_index], dim=1)
            data.y = torch.cat([data.y, y], dim=0)
            data.edge_type = torch.cat([data.edge_type, edge_type], dim=0)
        num_nodes = edge_index.max().item() + 1
        
        indices = []
        for i in range(data.y.max().item()+1):
            index = (data.y == i).nonzero().view(-1)#滤掉了不等于i的元素。nonzero输出结果为等于i的元素的位置，view整理成张量。index为data.y中等于i的所有元素的位置张量。
            index = index[torch.randperm(index.size(0))]#打乱index的顺序。randperm函数将index中0维张量中的数据打乱顺序
            print(index)
            indices.append(index)#切片中加入index


        train_idx = torch.cat([i[:int(len(i)*0.7)] for i in indices], dim=0)#前70%训练集，后30%测试集
        test_idx = torch.cat([i[int(len(i)*0.7):] for i in indices], dim=0)


        data.train_idx = train_idx
        data.test_idx = test_idx

        data = data if self.pre_transform is None else self.pre_transform(data)
            
        data, slices = self.collate([data])
        #collate() 函数将多个数据样本合并成一个 mini-batch，并返回该 mini-batch 数据及其对应的索引张量 slices
        torch.save((data,slices), self.processed_paths[0])