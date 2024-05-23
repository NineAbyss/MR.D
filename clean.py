import torch
from torch_geometric.utils import degree
from torch_geometric.data import Data

def remove_isolated_nodes(data):
    # 计算所有节点的度数
    deg = degree(data.edge_index[0], num_nodes=data.num_nodes) +degree(data.edge_index[1], num_nodes=data.num_nodes)
    
    # 找出度数大于0的节点的掩码
    mask = deg > 0
    
    # 过滤节点特征矩阵
    if data.x is not None:
        data.x = data.x[mask]
        data.batch = data.batch[mask]
    edge_index = data.edge_index[:, mask[data.edge_index[0]] & mask[data.edge_index[1]]]

    idx_mapping = torch.full((deg.size(0),), -1, dtype=torch.long)
    idx_mapping[mask] = torch.arange(mask.sum())
    # 应用映射
    edge_index = idx_mapping[edge_index]
    
    # 更新data对象
    data.edge_index = edge_index
    data.num_nodes = mask.sum()
    
    return data


data = torch.load('/home/yuhanli/wangpeisong/Topology-Pattern-Enhanced-Unsupervised-Group-level-Graph-Anomaly-Detection/aamas_inject_03.pt')
data = remove_isolated_nodes(data)
torch.save(data, 'aamas_inject_03.pt')