import dgl
import torch
import numpy as np
from torch_geometric.data import Data
from dgl import subgraph

g = dgl.load_graphs('peer_graphs/wu')[0][0]
x = g.ndata['feature']  
edges = g.edges()

edge_index = torch.tensor([edges[0].numpy(), edges[1].numpy()], dtype=torch.long)


data = Data(x=x, edge_index=edge_index)
torch.save(data, 'wu.pt')
