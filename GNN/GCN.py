import os
import os.path as osp
import argparse
import json
import pandas as pd

import numpy as np
import torch
from torch.nn import BatchNorm1d
import torch.nn.functional as F
from tqdm import tqdm

import torch_geometric.transforms as T
from torch_geometric.data import NeighborSampler
from torch_geometric.nn import GCNConv
from sklearn.metrics import average_precision_score, roc_curve, roc_auc_score, precision_recall_curve
from sklearn.metrics import precision_score, recall_score, f1_score
from dataset import Knowledge_graph
from FocalLoss import FocalLoss

# form torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='data\graph')
parser.add_argument("--batch-size", type=int, default=1024,
                    help="Mini-batch size. If -1, use full graph training.")
parser.add_argument("--fanout", type=int, default=-1,
                    help="Fan-out of neighbor sampling.")
# fanout是指每个节点的邻居数，也就是每个节点向外扩展的边数，通常用于指定每个节点在采样邻居时应该选择多少个。-1指full-graph模式训练，使用所有邻居
parser.add_argument("--n-layers", type=int, default=2,
                    help="number of propagation rounds")
# n-layers指的是GNN中堆叠的层数，也就是每次对图进行更新时，信息可以在整个图中通过多次传播来不断地更新
parser.add_argument("--dropout", type=float, default=0.2,
                    help="number of dropout rate")
parser.add_argument("--h-dim", type=int, default=32,
                    help="number of hidden units")
parser.add_argument("--validation", type=bool, default=True)
parser.add_argument("--early_stopping", type=int, default=10)
parser.add_argument("--n-epoch", type=int, default=200)
parser.add_argument("--test-file", type=str, default="/home/icdm/icdm2022_large/test_session1_ids.csv")
parser.add_argument("--json-file", type=str, default="pyg_pred.json")
parser.add_argument("--inference", type=bool, default=False)
# parser.add_argument("--record-file", type=str, default="record.txt")
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--gamma", type=float, default=1.0)
parser.add_argument("--alpha", type=float, default=0.35)
parser.add_argument("--model-id", type=str, default="1-1")
parser.add_argument("--device-id", type=str, default="0")

args = parser.parse_args(args=[])

os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
# 为推理设置可用设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    # 设置可用 GPU 的数量
    num_gpus = torch.cuda.device_count()
else:
    num_gpus = 0
print(f'num of gpu: {num_gpus}')
dataset = Knowledge_graph(args.dataset, pre_transform=T.NormalizeFeatures())
# from torch_geometric.datasets import Planetoid
# dataset = Planetoid(root='.',name='Cora')
data = dataset[0]

if args.inference == False:
    train_idx = data.train_idx
    val_idx = data.test_idx
else:
    test_idx = data.test_idx

# Mini-Batch
# if args.inference == False:
#     train_loader = NeighborSampler(edge_index=data.edge_index,
#                                    sizes = [args.fanout]*args.n_layers,
#                                    node_idx = data.train_idx,
#                                    shuffle=True, batch_size=args.batch_size, num_workers=16)
#     val_loader = NeighborSampler(edge_index=data.edge_index,
#                                    sizes = [args.fanout]*args.n_layers,
#                                    node_idx = data.test_idx,
#                                    shuffle=False, batch_size=args.batch_size, num_workers=16)
# else:
#     test_loader = NeighborSampler(edge_index=data.edge_index,
#                                   sizes = [args.fanout]*args.n_layers,
#                                   node_idx = data.test_idx,
#                                   shuffle=True, batch_size=args.batch_size, num_workers=16)
if args.inference == False:
    train_loader = NeighborSampler(edge_index=data.edge_index,
                                   sizes = [args.fanout]*args.n_layers,
                                   node_idx = data.train_idx,
                                   shuffle=True, batch_size=args.batch_size)
    val_loader = NeighborSampler(edge_index=data.edge_index,
                                   sizes = [args.fanout]*args.n_layers,
                                   node_idx = data.test_idx,
                                   shuffle=False, batch_size=args.batch_size)

else:
    test_loader = NeighborSampler(edge_index=data.edge_index, 
                                  sizes = [args.fanout]*args.n_layers, 
                                  node_idx = data.test_idx, 
                                  shuffle=True, batch_size=args.batch_size)


num_relations = dataset.num_relations


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, n_layers=2, dropout=0.4):
        super().__init__()
        self.convs = torch.nn.ModuleList()#self.convs 是一个 ModuleList，其中包含所有的 GCNConv 层
        self.norms = torch.nn.ModuleList()#self.norms 是一个 ModuleList，其中包含所有的 BatchNorm1d 层
        self.dropout= dropout#dropout：丢弃层丢弃率
        self.convs.append(GCNConv(in_channels, hidden_channels))# GCNConv 是 PyTorch Geometric 库中实现的图卷积层。第一个参数是输入通道，第二个参数是输出通道
                                                                # in_channels：输入通道（个数），比如节点分类中表示每个节点的特征数
                                                                # out_channels：输出通道（个数），最后一层GCNConv的输出通道为节点类别数（节点分类）
                                                                # hidden_channels 是作为一个输入变量存在，用于指定隐藏层的特征通道数。
                                                                # 具体来说，它控制了第一个图卷积层的输出通道数和第二个图卷积层的输入通道数，也决定了批量归一化层的输入通道数。
                                                                # 这个变量在初始化时被传递给 GCN 类的构造函数，然后被用来创建模型中的一些层。
                                                                # 因此，我们可以认为 hidden_channels 是一个模型超参数，需要根据具体问题进行调整。
        self.norms.append(BatchNorm1d(hidden_channels))#BatchNorm1d：一维批量归一化
        self.convs.append(GCNConv(hidden_channels, out_channels))

    def forward(self, x, edges):#forward 方法接收两个参数：输入节点特征张量 x 和边的索引列表 edges。
        # x 是形状为 (N, in_channels) 的节点特征矩阵(N个结点，in_channels个向量)
        # edges 是一个由元组 (edge_index, edge_type) 组成的列表，其中 edge_index 是边索引矩阵，是一个形状为 (2, num_edges) 的张量，其中 num_edges 是边的数量，edge_type 是边类型。
        # print(edges)
        for i, (edge_index, edge_type) in enumerate(edges):# for 循环遍历 edges 列表中的所有元组，每次都会将输入特征张量 x 通过一个 GCNConv 层传递，
                                                           # 然后应用批量归一化层、ReLU 激活函数和 dropout 层（仅在前面的隐藏层中应用）。
        # 对于每个 (edge_index, edge_type)，我们先将节点特征矩阵 x 作为第一层图卷积的输入，经过 GCNConv 层得到第一层的输出，
        # 然后根据当前层数的编号和总层数的关系进行不同的处理。如果当前层不是最后一层，我们将输出结果进行批量归一化、ReLU 激活和 dropout 操作，
        # 并将结果作为下一层图卷积的输入；如果当前层是最后一层，我们直接返回输出结果。
        # 在每个层的计算过程中，还记录了经过当前层前的最后一层输出，即 hidden 变量，方便在后续的损失计算中使用。
        #     print(edge_index)
            x = self.convs[i](x, edge_index)
            if i < len(self.convs) - 1:
                hidden = x.clone()
                x = self.norms[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x, hidden                                    #hidden 是在前面的隐藏层中传递的特征向量。该值在返回时被返回，以用于损失计算或其他目的。


in_dim = data.x.size(1)# data.x 表示该数据对象中的节点特征矩阵，其形状为 (N, in_dim)，其中 N 是节点个数，in_dim 是每个节点的特征维度。
                       # data.x.size(1) 就是获取节点特征矩阵的第二个维度的大小，也就是每个节点的特征维度 in_dim。
out_dim = dataset.num_classes

model = GCN(in_channels=in_dim, hidden_channels=args.h_dim, out_channels=out_dim, \
             n_layers=args.n_layers, dropout=args.dropout).to(device)

if args.inference:
    model.load_state_dict(torch.load(osp.join("best_model", 'GCN-' + args.model_id + ".pth")))
else:
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = torch.optim.RAdam(model.parameters(), lr=args.lr)
focalloss = FocalLoss(gamma=args.gamma, alpha=args.alpha)


def train(epoch):
    model.train()

    pbar = tqdm(total=int(len(train_loader.dataset)), ascii=True)
    pbar.set_description(f'Epoch {epoch:02d}')

    total_loss = total_correct = total_examples = 0
    true_positive = pre_positive = exp_positive = precision = recall = F1_score = 0
    y_pred = []
    y_true = []

    for batch_size, n_id, adjs in train_loader:
        # adjs holds a list of (edge_index, e_id, size) tuples
        edges = []
        adjs = [adj.to(device) for adj in adjs]
        for e_index, e_id, size in adjs: 
            edges.append((e_index, data.edge_type[e_id]))
        y = data.y[n_id[:batch_size]].to(device)
                    
        optimizer.zero_grad()

        # print(adjs)


        out, hidden = model(data.x[n_id].to(device), edges)[:batch_size]
        out = out[:batch_size]
        hidden = hidden[:batch_size]
        # loss = F.cross_entropy(out, y)
        # print(out.shape)
        # print(y.shape)
        loss = focalloss(out, y)
        loss.backward()
        optimizer.step()

        y_pred.append(F.softmax(out, dim=1)[:, 1].detach().cpu())
        y_true.append(y.cpu())
        total_loss += float(loss) * batch_size
        total_correct += int((out.argmax(dim=-1) == y).sum())
        total_examples += batch_size

        pbar.update(batch_size)
    pbar.close()
    fpr, tpr, thresholds = roc_curve(torch.cat(y_true).numpy(), torch.cat(y_pred).numpy())
    ks_score = max(tpr- fpr)
    auc_score = roc_auc_score(torch.cat(y_true).numpy(), torch.cat(y_pred).numpy())

    if (recall + precision) != 0:
        F1_score = (2 * recall * precision) / (recall + precision)
    else:
        F1_score = 0

    return total_loss / total_examples, total_correct / total_examples, ks_score, auc_score



@torch.no_grad()
def val():
    model.eval()
    pbar = tqdm(total=int(len(val_loader.dataset)), ascii=True)
    pbar.set_description(f'Epoch {epoch:02d}')
                    
    total_loss = total_correct = total_examples = 0
    true_positive = pre_positive = exp_positive = precision = recall = F1_score = 0
    y_pred = []
    y_true = []
    for batch_size, n_id, adjs in val_loader:
        # adjs holds a list of (edge_index, e_id, size) tuples
        edges = []
        adjs = [adj.to(device) for adj in adjs]
        for e_index, e_id, size in adjs: 
            edges.append((e_index, data.edge_type[e_id]))
        y = data.y[n_id[:batch_size]].to(device)
                    
        out, hidden = model(data.x[n_id].to(device), edges)[:batch_size]
        out = out[:batch_size]
        hidden = hidden[:batch_size]
        # loss = F.cross_entropy(out, y)
        loss = focalloss(out, y)

        y_pred.append(F.softmax(out, dim=1)[:, 1].detach().cpu())
        y_true.append(y.cpu())
        total_loss += float(loss) * batch_size
        total_correct += int((out.argmax(dim=-1) == y).sum())
        total_examples += batch_size

    pbar.close()
    fpr, tpr, thresholds = roc_curve(torch.cat(y_true).numpy(), torch.cat(y_pred).numpy())
    ks_score = max(tpr- fpr)
    auc_score = roc_auc_score(torch.cat(y_true).numpy(), torch.cat(y_pred).numpy())

    return total_loss / total_examples, total_correct / total_examples, ks_score, auc_score

@torch.no_grad()#不需要计算梯度，也无需进行反向传播
def test():
    model.eval()
    pbar = tqdm(total=int(len(test_loader.dataset)), ascii=True)
    pbar.set_description(f'Generate Final Result:')
    y_pred = []
    y_true = []
    y_label = []
    for batch_size, n_id, adjs in test_loader:
        # adjs holds a list of (edge_index, e_id, size) tuples
        edges = []
        adjs = [adj.to(device) for adj in adjs]
        for e_index, e_id, size in adjs: 
            edges.append((e_index, data.edge_type[e_id]))
        y = data.y[n_id[:batch_size]].to(device)
                    
        out, hidden = model(data.x[n_id].to(device), edges)[:batch_size]
        out = out[:batch_size]
        hidden = hidden[:batch_size]
        
        # loss = F.cross_entropy(out, y)
        loss = focalloss(out, y)

        y_pred.append(F.softmax(out, dim=1)[:, 1].detach().cpu())
        y_true.append(y.cpu())
        y_label.append(out.argmax(dim=-1).detach().cpu())
        pbar.update(batch_size)
    pbar.close()

    return torch.cat(y_true).numpy(), torch.cat(y_pred).numpy(), torch.cat(y_label).numpy()

# writer = SummaryWriter("logs")

def draw_fig(list,name,epoch,savefig):
    # 我这里迭代了200次，所以x的取值范围为(0，200)，然后再将每次相对应的准确率以及损失率附在x上
    x1 = range(1, epoch+1)
    y1 = list
    if name=="loss":
        plt.cla()
        # plt.title('Train loss vs. epoch', fontsize=20)
        plt.plot(x1, y1, '.-')
        plt.xlabel('epoch', fontsize=20)
        plt.ylabel('loss', fontsize=20)
        plt.grid()
        if os.path.exists(savefig+"/Train_loss.png"):
            os.remove(savefig+"/Train_loss.png")
        plt.savefig(savefig+"/Train_loss.png")
        # plt.show()
    elif name =="acc":
        plt.cla()
        # plt.title('Train accuracy vs. epoch', fontsize=20)
        plt.plot(x1, y1, '.-')
        plt.xlabel('epoch', fontsize=20)
        plt.ylabel('accuracy', fontsize=20)
        plt.grid()
        if os.path.exists(savefig+"/Train_accuracy.png"):
            os.remove(savefig+"/Train_accuracy.png")
        plt.savefig(savefig+"/Train_accuracy.png")
        # plt.show()

def out_txt(list,name,epoch,savefig):
    x = range(1, epoch + 1)
    y = list
    out_list = []
    if name == "loss":
        for l in x:
            line = [l,y[l - 1]]
            out_list.append(line)
        np.savetxt(savefig + "/GCN_loss_records.txt", out_list,fmt="%d %.20f")
    elif name == "acc":
        for l in x:
            line = [l, y[l - 1]]
            out_list.append(line)
        np.savetxt(savefig + "/GCN_acc_records.txt", out_list,fmt="%d %.20f")

if args.inference == False:
    print("Start training")
    train_acc_list = []
    train_loss_list = []

    val_ks_list = [0.]
    val_auc_list = [0.]
    val_F1_list = [0.]
    ave_val_auc = 0
    best_epoch = 0
    for epoch in range(1, args.n_epoch + 1):
        train_loss, train_acc, train_ks, train_auc= train(epoch)
        print(f'Train: Epoch {epoch:02d}, Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, KS_Score: {train_ks:.4f}, AUC_score: {train_auc:.4f}')
        train_acc_list.append(float(train_acc))
        train_loss_list.append(float(train_loss))
        if args.validation and epoch >= args.early_stopping:
            val_loss, val_acc, val_ks, val_auc = val()
            print(f'Val: Epoch: {epoch:02d}, Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, KS_Score: {val_ks:.4f}, AUC_score: {val_auc:.4f}')
            # if val_auc > np.max(val_auc_list):
            #     torch.save(model.state_dict(), osp.join("best_model", 'GCN-'+args.model_id + ".pth"))
            #     best_epoch = epoch
            if val_auc > np.max(val_F1_list):
                torch.save(model.state_dict(), osp.join("best_model", 'GCN-'+args.model_id + ".pth"))
                best_epoch = epoch
            val_ks_list.append(float(val_ks))
            val_auc_list.append(float(val_auc))
            val_F1_list.append(float(val_auc))
            ave_val_auc = np.average(val_auc_list)
            # writer.add_scalar()
    print(f"Complete Trianing (Model id: {args.model_id}, Best epoch: {best_epoch})")
    save_path='./logs/GCN'
    # draw_fig(train_acc_list,'acc',args.n_epoch,save_path)
    # draw_fig(train_loss_list, 'loss', args.n_epoch,save_path)
    # out_txt(train_acc_list,'acc',args.n_epoch,save_path)
    # out_txt(train_loss_list, 'loss', args.n_epoch, save_path)

#    with open(args.record_file, 'a+') as f:
#        f.write(f"{args.model_id:2d} {args.h_dim:3d} {args.n_layers:2d} {args.lr:.4f} {end:02d} {float(val_ap_list[-1]):.4f} {np.argmax(val_ap_list)+5:02d} {float(np.max(val_ap_list)):.4f}\n")


if args.inference == True:
    y_true, y_pred, y_label = test()
    print(y_label)
    print(precision_score(y_true, y_label), recall_score(y_true, y_label), f1_score(y_true, y_label))