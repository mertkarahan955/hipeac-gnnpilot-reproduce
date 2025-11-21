from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch.nn.functional as F

import torch
import KGGNN

import utils
from perf_time import *
import time

def cusp_run(name, csr, input_feature, feat_dim, device, warmup=10, repetitions=100):
    rowptr = torch.tensor(csr.indptr)
    indices = torch.tensor(csr.indices)
    values = torch.ones_like(indices, dtype = torch.float32)

    num_nodes = len(rowptr) - 1
    num_edges = len(indices)
    rowptr = rowptr.to(device)
    indices = indices.to(device)
    values = values.to(device)

    if (name == 'SpMM'):
        output_feature = torch.zeros(num_nodes, feat_dim).to(device)
        test_time = KGGNN.kg_gcn_run_cusparse(rowptr, indices, feat_dim, values, input_feature, output_feature, warmup, repetitions)
    elif (name == 'SDDMM'):
        feat_len = input_feature[0, :, :].size(1)

        #print(num_nodes, feat_len, num_edges)
        input1 = input_feature[0, :, :].t().reshape(num_nodes, -1)
        # print(input_feature[0, :, :].size())
        # print(input_feature[0, :, :])
        #input1 = input_feature[0, :, :]
        input2 = input_feature[1, :, :]

        # print(input1)
        # print(input2)

        test_time = KGGNN.kg_sddmm_run_cusparse(rowptr, indices, \
        input1, input2, feat_len, values, \
        warmup, repetitions)
        output_feature = values

    return output_feature, test_time