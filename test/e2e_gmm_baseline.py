import dgl
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GMMConv
from torch_geometric.data import Data

from load_dataset import *

from utils import *
import sys
import time
from perf_time import * 

import rabbit

import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
torch.manual_seed(970702)

if (len(sys.argv) < 2 or len(sys.argv) > 4):
    print("Usage: python test_dgl_gcn.py [dataset] [outcsv_name | optional] [fullcsv_name | optional]")
    exit(1)

dataset_dir = sys.argv[1]
if len(sys.argv) >= 3:
    outcsv_name = sys.argv[2]
else:
    outcsv_name = None

if len(sys.argv) == 4:
    fullcsv_name = sys.argv[3]
else:
    fullcsv_name = None

# Load and initialize input data
if ('.pt' in dataset_dir):
    graph_data = dataset_load(dataset_dir)
    in_dim, out_dim = dataset_prop(graph_data)
    num_nodes = graph_data.num_nodes
elif ('.mtx' in dataset_dir):
    graph_coo = read_mtx(dataset_dir)
    num_nodes = graph_coo.get_shape()[0]
    in_dim = 32
elif ('.csr' in dataset_dir):
    graph_csr = read_csr(dataset_dir)
    num_nodes = graph_csr.get_shape()[0]
    in_dim = 32
else:
    exit

reorder = True

# Transform to CSR format
if ('.pt' in dataset_dir):
    #graph_data = graph_data.to(device)
    graph_csr = graph_to_csr(graph_data, reorder = reorder, show_degree = True)
elif ('.mtx' in dataset_dir):
    graph_csr = coo_to_csr(num_nodes, graph_coo, reorder = reorder, show_degree = True)
elif ('.csr' in dataset_dir):
    graph_csr = csr_transform(graph_csr, add_self_loop = True, reorder = reorder, show_degree = True)
    pass
else:
    exit

# Set device and move data
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
perf_time_init(dataset_dir)

graph_coo = graph_csr.tocoo()
coorow = graph_coo.row.astype(int)
coocol = graph_coo.col.astype(int)
edge_index = np.array([coocol, coorow])
edge_index = torch.tensor(edge_index, dtype=torch.long).to(device)

# test setup
gcn_hidden = 64
out_dim = 32
print("layer feature: ", in_dim, gcn_hidden, out_dim)

# prepare data
m = len(graph_csr.indptr) - 1
nnz = len(graph_csr.indices)
avg_rnz = 1.0 * nnz / m
max_rnz = max(graph_csr.indptr[1: m+1] - graph_csr.indptr[0: m])

print("graph info: m {:d} nnz {:d} rnz {:f} max_rnz {:d}".format(m, nnz, avg_rnz, max_rnz))

input_feature = torch.randn(m, in_dim).to(device)

class PYG_GMM(torch.nn.Module):
    def __init__(self, in_feats, n_hidden, out_feats, paras = None):
        super(PYG_GMM, self).__init__()
        self.conv1 = GMMConv(in_feats, n_hidden, 8, 1)
        self.conv2 = GMMConv(n_hidden, out_feats, 8, 1)

    def forward(self, edge_index, edge_weight, x):
        x = self.conv1(x, edge_index, edge_weight)
        x = self.conv2(x, edge_index, edge_weight)
        return x

g = dgl.graph((graph_data.edge_index[0], graph_data.edge_index[1]))

pyg_model = PYG_GMM(in_dim, gcn_hidden, out_dim).to(device)
pyg_model.eval()

# edge_index = edge_index.to(device)
edge_weight = torch.randn(nnz, 8).to(device)

pyg_model(edge_index, edge_weight, input_feature)

perf_time_start("PYG kernel")
pyg_model(edge_index, edge_weight, input_feature)
model_time = perf_time_end()

# print()

if (outcsv_name != None):
    with open(outcsv_name, "a+") as f:
        f.write("{},{:f}\n".format(dataset_dir, model_time))
