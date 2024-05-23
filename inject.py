import argparse
import os
import torch
from utils import *
from training import train, test
from initializer import *
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./data')
parser.add_argument('--real_world_name', type=str, default='aamas')#simML AMLpublic Ethereum_TSGN Citeseer Cora
parser.add_argument('--dataset', type=str, default='complete')
parser.add_argument('--anomaly_type', type=str, default='complete')
parser.add_argument('--size', type=int, default=200)
parser.add_argument('--anomaly_ratio', type=float, default=0.02)
parser.add_argument('--dim', type=int, default=50)
parser.add_argument('--anomaly_scale', type=float, default=0.3)
parser.add_argument('--anomaly_attr_ratio', type=float, default=1.0)
parser.add_argument('--diff_ratio', type=int, default=2)
parser.add_argument('--half_num', type=int, default=10)
parser.add_argument('--random_seed', type=int, default=12345)
parser.add_argument('--num_anchors', type=int, default=0)

parser.add_argument('--gae_type', type=str, default='dominant')#dominant#subGAE
parser.add_argument('--gae_embedding_channels', type=int, default=64)
parser.add_argument('--gae_hidden_channels', type=int, default=16)
parser.add_argument('--gae_num_layer', type=int, default=2)
parser.add_argument('--gae_out_channels', type=int, default=16)
parser.add_argument('--gae_num_features', type=int, default=2)
parser.add_argument('--gae_epochs', type=int, default=1000)
parser.add_argument('--gae_lr', type=float, default=1e-3)

parser.add_argument('--gcl_input_dim', type=int, default=1)
parser.add_argument('--gcl_hidden_dim', type=int, default=32)
parser.add_argument('--gcl_num_layer', type=int, default=2)
parser.add_argument('--gcl_output_dim', type=int, default=16)
parser.add_argument('--gcl_epochs', type=int, default=100)
parser.add_argument('--gcl_lr', type=float, default=1e-2)
parser.add_argument('--q', type=int, default=90)
parser.add_argument('--inner_epochs', type=int, default=20)
parser.add_argument('--inner_lr', type=float, default=1e-4)
parser.add_argument('--contamination', type=float, default=0.15)

parser.add_argument('--warmup', type=int, default=90)
parser.add_argument('--alpha', type=float, default=1.0)
parser.add_argument('--beta_1', type=float, default=1.)
parser.add_argument('--beta_2', type=float, default=1.)
parser.add_argument('--per_ratio', type=float, default=.0)
parser.add_argument('--convergence', type=float, default=1e-4)
parser.add_argument('--ending_rounds', type=int, default=1)
args = parser.parse_args()
def load_data(args):

    FILE_NAME = "{}.pt".format(args.real_world_name)
    FILE_PATH = os.path.join(args.data_dir, FILE_NAME)
    data = torch.load(FILE_PATH)
    # anomaly_flag = data.y.numpy()
    data.x = data.x.float()
    return data
data = load_data(args)
# batch = set(data.batch.cpu().numpy())


ano_sub = {
    'path': 1,  
    'tree': 1,  
    'circuit': 5,  
}

New_G = GraphProcessor().anomaly_group_injector(data, ano_sub)
print(0)
torch.save(data, 'aamas_inject_03.pt')

