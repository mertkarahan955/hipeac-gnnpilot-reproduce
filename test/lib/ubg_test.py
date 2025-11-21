import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.cpp_extension import load

from dgl import DGLGraph

from scipy import sparse
import numpy as np

import utils
from perf_time import *

import time

gnc = load(
    name="gnncompile",
    sources=[
        "../UBG/src/kernel.cpp",
        "../UBG/src/kernel_generated.cu",
        "../UBG/src/util.cu",
        "../UBG/src/data.cu"
    ],
    build_directory='lib/gnncompile')

class UBG_GCN(nn.Module):
    def __init__(self, in_feats, n_hidden, out_feats):
        at = gnc.gcn_init(ptrs, idxs, vals)
        gnc.gcn_schedule(at, 32)

def ubg_model_run(name, csr, in_dim, gcn_hidden, out_dim, input_feature, 
device, paras = None, detail_test = 0):

    rowptr = csr.indptr

    rowptr = torch.tensor(rowptr).to(device)
    indices = torch.tensor(csr.indices).to(device)
    vals = torch.ones(len(csr.indices)).to(device)
    # degree = 1.0 / torch.tensor(rowptr[1:] - rowptr[:-1], dtype=torch.float32)
    # degree = degree.to(device)
    degrees = (rowptr[1:] - rowptr[:-1]).to(device)

    num_nodes = input_feature.size(0)
    num_edges = indices.size(0)

    if (name == "GCN"):
        if (paras == None):
            weight0 = torch.randn([in_dim, gcn_hidden]).to(device)
            weight1 = torch.randn([gcn_hidden, out_dim]).to(device)
            bias0 = torch.randn([gcn_hidden]).to(device)
            bias1 = torch.randn([out_dim]).to(device)
        else:
            weight0 = paras['w0'].to(device)
            weight1 = paras['w1'].to(device)
            bias0 = paras['b0'].to(device)
            bias1 = paras['b1'].to(device)

        at = gnc.gcn_init(rowptr, indices, vals)
        gnc.gcn_schedule(at, 32)

        output_feat0 = torch.zeros([num_nodes, gcn_hidden], dtype=torch.float).to(device)
        output_feat1 = torch.zeros([num_nodes, out_dim], dtype=torch.float).to(device)

        # GCN Layer run
        perf_time_start("UBG") if detail_test > 0 else 0

        # GCN Layer 0 (in_dim, gcn_hidden)
        perf_time_start("UBG MM0") if detail_test == 1 else 0
        feat0 = torch.mm(input_feature, weight0)
        perf_time_end() if detail_test == 1 else 0

        #print(feat0)

        perf_time_start("UBG AGGR0") if detail_test > 0 else 0
        gnc.gcn_run(at, feat0, output_feat0, degrees, 128, 1)
        perf_time_end() if detail_test > 0 else 0

        #print(output_feat0)

        perf_time_start("UBG OTHER0") if detail_test == 1 else 0
        output_feat0 = output_feat0 + bias0
        output_feat0 = F.relu(output_feat0)
        perf_time_end() if detail_test == 1 else 0

        # GCN Layer 1 (gcn_hidden, out_dim)
        perf_time_start("UBG MM1") if detail_test == 1 else 0
        feat1 = torch.mm(output_feat0, weight1)
        perf_time_end() if detail_test == 1 else 0

        perf_time_start("UBG AGGR1") if detail_test > 0 else 0
        gnc.gcn_run(at, feat1, output_feat1, degrees, 128, 1)
        perf_time_end() if detail_test > 0 else 0

        perf_time_start("UBG OTHER1") if detail_test == 1 else 0
        output_feat1 = output_feat1 + bias1
        output_feat1 = F.log_softmax(output_feat1, dim=1)
        perf_time_end() if detail_test == 1 else 0

        ubg_time = perf_time_end() if detail_test > 0 else 0

        # GCN Layer run end

        return output_feat1, ubg_time
    
    elif (name == "SpMM"):
        if (paras == None):
            weight0 = torch.randn([in_dim, out_dim]).to(device)
            bias0 = torch.randn([out_dim]).to(device)
        else:
            weight0 = paras['w0'].to(device)
            bias0 = paras['b0'].to(device)

        at = gnc.gcn_init(rowptr, indices, vals)
        gnc.gcn_schedule(at, 32)

        num_nodes = input_feature.size(0)
        output_feat0 = torch.zeros([num_nodes, out_dim], dtype=torch.float).to(device)

        # GCN Layer run
        perf_time_start("UBG") if detail_test > 0 else 0

        # GCN Layer 0 (in_dim, gcn_hidden)
        perf_time_start("UBG MM0") if detail_test == 1 else 0
        feat0 = torch.mm(input_feature, weight0)
        perf_time_end() if detail_test == 1 else 0

        perf_time_start("UBG AGGR0") if detail_test > 0 else 0
        gnc.gcn_run(at, feat0, output_feat0, degrees, 128, 1)
        perf_time_end() if detail_test > 0 else 0

        ubg_time = perf_time_end() if detail_test > 0 else 0

        return output_feat0, ubg_time
        
    elif (name == "SDDMM"):

        at = gnc.sddmm_init(rowptr, indices)

        output_feat0 = torch.zeros([num_edges], dtype=torch.float).to(device)

        perf_time_start("UBG") if detail_test > 0 else 0
        gnc.sddmm_run(at, input_feature[0, :, :], input_feature[1, :, :], output_feat0, 128, 0)
        ubg_time = perf_time_end() if detail_test > 0 else 0

        return output_feat0, ubg_time

    elif (name == "GAT"):
        if (paras == None):
            weight0 = torch.randn([in_dim, gcn_hidden]).to(device)
            weight1 = torch.randn([gcn_hidden, out_dim]).to(device)
            bias0 = torch.randn([gcn_hidden]).to(device)
            bias1 = torch.randn([out_dim]).to(device)
            att_src0 = torch.randn([1, 1, gcn_hidden]).to(device)
            att_dst0 = torch.randn([1, 1, gcn_hidden]).to(device)
            att_src1 = torch.randn([1, 1, out_dim]).to(device)
            att_dst1 = torch.randn([1, 1, out_dim]).to(device)
        else:
            weight0 = paras['w0'].to(device)
            weight1 = paras['w1'].to(device)
            bias0 = paras['b0'].to(device)
            bias1 = paras['b1'].to(device)
            att_src0 = paras['as0'].to(device)
            att_dst0 = paras['ad0'].to(device)
            att_src1 = paras['as1'].to(device)
            att_dst1 = paras['ad1'].to(device)

        at = gnc.gat_init(rowptr, indices)
        gnc.gat_schedule(at, 32)

        output_feat0 = torch.zeros([num_nodes, gcn_hidden], dtype=torch.float).to(device)
        output_feat1 = torch.zeros([num_nodes, out_dim], dtype=torch.float).to(device)
        weight_lr0 = torch.cat((att_dst0.reshape(-1, 1), att_src0.reshape(-1, 1)), 1)
        weight_lr1 = torch.cat((att_dst1.reshape(-1, 1), att_src1.reshape(-1, 1)), 1)
        # print(weight_lr0.size())

        # GCN Layer run
        perf_time_start("UBG") if detail_test > 0 else 0

        # GCN Layer 0 (in_dim, gcn_hidden)
        perf_time_start("UBG MM0") if detail_test == 1 else 0
        feat0 = torch.mm(input_feature, weight0)
        perf_time_end() if detail_test == 1 else 0

        #print(feat0, att_lr0)

        perf_time_start("UBG AGGR0") if detail_test > 0 else 0
        att_lr0 = torch.mm(feat0, weight_lr0)
        gnc.gat_run(at, feat0, att_lr0, output_feat0, 128, 1)
        perf_time_end() if detail_test > 0 else 0

        #print(output_feat0)

        perf_time_start("UBG OTHER0") if detail_test == 1 else 0
        output_feat0 = output_feat0 + bias0
        perf_time_end() if detail_test == 1 else 0

        # print(output_feat0)

        # GCN Layer 1 (gcn_hidden, out_dim)
        perf_time_start("UBG MM1") if detail_test == 1 else 0
        feat1 = torch.mm(output_feat0, weight1)
        perf_time_end() if detail_test == 1 else 0

        perf_time_start("UBG AGGR1") if detail_test > 0 else 0
        att_lr1 = torch.mm(feat1, weight_lr1)
        gnc.gat_run(at, feat1, att_lr1, output_feat1, 128, 1)
        perf_time_end() if detail_test > 0 else 0

        perf_time_start("UBG OTHER1") if detail_test == 1 else 0
        output_feat1 = output_feat1 + bias1
        perf_time_end() if detail_test == 1 else 0

        ubg_time = perf_time_end() if detail_test > 0 else 0

        # GCN Layer run end

        return output_feat1, ubg_time

    else:
        print("Wrong argument of ubg_model_run!")

        return None

