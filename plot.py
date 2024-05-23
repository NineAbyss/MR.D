import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
from collections import defaultdict
# 假设G已经被加载，是一个pytorch_geometric的Data对象
G = torch.load('./data/aamas_34_1.0.pt')



# 假设G是你的pyg Data对象
# G.edge_index: [2, num_edges]，存储边的信息
# G.batch: [num_nodes]，存储节点属于哪个子图的信息

def compute_subgraph_edges(G):
    edge_index = G.edge_index
    batch = G.batch

    # 初始化一个字典来存储每个子图的边
    subgraph_edges = defaultdict(int)

    # 遍历所有的边
    for i in range(edge_index.shape[1]):
        node_a, node_b = edge_index[:, i]
        subgraph_a = batch[node_a].item()
        subgraph_b = batch[node_b].item()

        # 如果这条边的两个节点属于同一个子图，计数加一
        if subgraph_a == subgraph_b:
            subgraph_edges[subgraph_a] += 1

    # 因为无向图中每条边被计算了两次，所以需要除以2
    for subgraph in subgraph_edges:
        subgraph_edges[subgraph] //= 2

    return subgraph_edges

# 计算每个子图的边数
import matplotlib.pyplot as plt

# 假设subgraph_edges_count是上一步计算得到的每个子图的边数
subgraph_edges_count = compute_subgraph_edges(G)

# 获取所有子图的边数列表
edges_counts = list(subgraph_edges_count.values())



# 绘制分布图
plt.hist(edges_counts, bins=20, alpha=0.75, color='blue')
plt.title('子图边数量分布')
plt.xlabel('num_edges')
plt.ylabel('num_subgraphs')
plt.grid(True)

# 保存图表
plt.savefig('subgraph_edges_distribution.png')

# 显示图表
plt.show()

def print_subgraphs_with_edges_more_than_20(G):
    # 计算每个子图的边数
    subgraph_edges_count = compute_subgraph_edges(G)
    
    # 遍历每个子图的边数
    for subgraph, edges_count in subgraph_edges_count.items():
        # 如果边数大于20
        if edges_count > 15:
            # 打印该子图的标签
            print(f"子图 {subgraph} 的标签是: {G.y[subgraph].item()}")

# 调用函数
print_subgraphs_with_edges_more_than_20(G)