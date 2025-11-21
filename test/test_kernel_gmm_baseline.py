import dgl
import torch
import torch.nn.functional as F
import torch.nn as nn
from load_dataset import *

from utils import *
import sys
sys.path.append('../../test/lib')
from ugcg_test import *
from kg_test import *

import time
from perf_time import * 

import rabbit

import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
torch.manual_seed(970702)

if (len(sys.argv) < 3 or len(sys.argv) > 4):
    print("Usage: python test_dgl_gcn.py baseline_type [dataset] [outcsv_name | optional] [fullcsv_name | optional]")
    exit(1)

baseline_type = sys.argv[1]
dataset_dir = sys.argv[2]
if len(sys.argv) >= 4:
    outcsv_name = sys.argv[3]
else:
    outcsv_name = None

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

# test setup
in_dim = 32
gcn_hidden = 32
out_dim = 32
# out_dim = int((out_dim + 31) / 32) * 32

print("layer feature: ", in_dim, gcn_hidden, out_dim)

weight_init = True

if (weight_init):
    random_seed = 123
    torch.manual_seed(random_seed)
    weight0 = torch.randn(in_dim, gcn_hidden)
    weight1 = torch.randn(gcn_hidden, out_dim)
    att_src0 = torch.randn(1, 1, gcn_hidden)
    att_dst0 = torch.randn(1, 1, gcn_hidden)
    att_src1 = torch.randn(1, 1, out_dim)
    att_dst1 = torch.randn(1, 1, out_dim)
    bias0 = torch.randn(gcn_hidden)
    bias1 = torch.randn(out_dim)
    paras = {'w0': weight0, 'w1': weight1, 'b0': bias0, 'b1': bias1, \
    'as0': att_src0, 'ad0': att_dst0, 'as1': att_src1, 'ad1':att_dst1}
else:
    paras = None

# input_feature = torch.randn(num_nodes, in_dim)

# Set device and move data
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# input_feature = input_feature.to(device)
perf_time_init(dataset_dir)

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

# prepare data
m = len(graph_csr.indptr) - 1
nnz = len(graph_csr.indices)
avg_rnz = 1.0 * nnz / m
max_rnz = max(graph_csr.indptr[1: m+1] - graph_csr.indptr[0: m])

fd = 32
fd2 = 2
rowptr = torch.tensor(graph_csr.indptr).to(device)
indices = torch.tensor(graph_csr.indices).to(device)
f = torch.randn(m, fd).to(device)
fo = torch.zeros(m, fd).to(device)
# fo2 = torch.zeros(m, fd).to(device)
p = torch.randn(m, fd2).to(device)
mu = torch.randn(1, fd2).to(device)
diag = torch.randn(1, fd2).to(device)
e = torch.zeros(nnz, 1).to(device)
# e2 = torch.zeros(nnz, 1).to(device)

perf_time_start("preprocessing")
if (baseline_type == 'UGCG'):
    ugcg_model = ugcg_model_init("GMM", in_dim, gcn_hidden, out_dim, kernel_dim=fd2, kernel_num=1).to(device)
elif (baseline_type == 'PCKGNN'):
    pass

perf_time_end()

perf_time_start("warmup")
if (baseline_type == 'UGCG'):
    out_ugcg, _ = ugcg_model_run(ugcg_model, graph_csr, f, device, p = p, mu = mu, sigma = diag)
# elif (baseline_type == 'PCKGNN'):
#     out_kg2, _, _ = kg_run_tmp(graph_csr, f, in_dim, gcn_hidden, out_dim, \
#     device, balance = 2, paras = paras, detail_test = 0, reorder = False, time_label = 'KG balance 2', name = 'GAT')
perf_time_end()

perf_time_start("execution")
if (baseline_type == 'UGCG'):
    out_ugcg, layer_time = ugcg_model_run(ugcg_model, graph_csr, f, device, p = p, mu = mu, sigma = diag)
# elif (baseline_type == 'PCKGNN'):
#     out_kg2, _, layer_time = kg_run_tmp(graph_csr, f, in_dim, gcn_hidden, out_dim, \
#     device, balance = 2, paras = paras, detail_test = 2, reorder = False, time_label = 'KG balance 2', name = 'GAT')
perf_time_end()

print("layer time", layer_time)

if (outcsv_name):
    f1 = open(outcsv_name, "a")
    f1.write("{},{},{:f}\n".format(dataset_dir, baseline_type, layer_time))
