import dgl
import torch
import numpy as np
from torch_geometric.data import Data
from dgl import subgraph

g = dgl.load_graphs('/home/yuhanli/wangpeisong/Topology-Pattern-Enhanced-Unsupervised-Group-level-Graph-Anomaly-Detection/peer_graphs/wu')[0][0]
x = g.ndata['feature']  # 假设节点特征存储在'feat'中
# y = g.ndata['label']  # 假设节点标签存储在'label'中
edges = g.edges()

edge_index = torch.tensor([edges[0].numpy(), edges[1].numpy()], dtype=torch.long)

# batch = torch.zeros_like(y, dtype=torch.long)  # 假设所有节点最初都属于子图0
# batch[y == 1] = 1  # 将标签为1的节点分配到子图1

# # 创建新的y向量，用于表示子图的标签（这里我们假设子图1为异常，标记为1）
# new_y = torch.zeros(2, dtype=torch.long)  # 假设有两个子图
# new_y[1] = 1  # 标记子图1为异常

# 创建Data对象，保存两个子图的信息
data = Data(x=x, edge_index=edge_index)
torch.save(data, 'wu.pt')
