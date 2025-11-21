import torch
import torch.nn as nn
import torch.nn.functional as F

import GESpMM

from scipy import sparse
import numpy as np

from perf_time import *

#ge = load(
#    name="gespmmcompile",
#    sources=[
#        "../dgSPARSE/example/ge-spmm/mmio.hpp",
#        "../dgSPARSE/example/ge-spmm/spmm.cu.cc",
#        "../dgSPARSE/example/ge-spmm/spmm_util.hpp",
#    ],
#    build_directory='lib/gespmmcompile')

def ge_spmm_run(csr, feat_dim, feat0):
    rowptr = torch.tensor(csr.indptr).cpu()
    indices = torch.tensor(csr.indices).cpu()
    degrees = (rowptr[1:] - rowptr[:-1]).cpu()

    num_nodes = rowptr.size(0) - 1
    non_zeros = indices.size(0)

    output_feat0 = torch.zeros([num_nodes, feat_dim], dtype=torch.float).cpu()

    #print(rowptr, indices, num_nodes, non_zeros)

    best_time0 = GESpMM.ge_spmm(num_nodes, feat_dim, non_zeros, rowptr, indices,
    degrees, feat0.cpu(), output_feat0)

    print("GE best time: {:.3f} ms".format(best_time0))

    return output_feat0, best_time0

def ge_model_run(name, csr, in_dim, gcn_hidden, out_dim, 
input_feature, device, paras = None, detail_test = 0):

    device = 'cpu'
    input_feature = input_feature.to(device)

    # num_nodes = graph_data.x.size(0)
    # non_zeros = graph_data.edge_index.size(1)
    # edge_data = graph_data.edge_index.cpu()
    # vals = torch.ones(edge_data.size(1))
    # iidxs = edge_data[0].numpy()
    # jidxs = edge_data[1].numpy()

    # coo = sparse.coo_matrix((vals.numpy(), (jidxs, iidxs)), shape = [num_nodes, num_nodes])
    # csr = coo.tocsr()

    # # Add self loop
    # iidxs = np.array([i for i in range(0, num_nodes)])
    # jidxs = np.array([i for i in range(0, num_nodes)])
    # vals = torch.ones(num_nodes).numpy()
    # csr2 = sparse.coo_matrix((vals, (jidxs, iidxs)), shape = [num_nodes, num_nodes]).tocsr()

    # csr = csr + csr2

    rowptr = csr.indptr

    rowptr = torch.tensor(rowptr).to(device)
    indices = torch.tensor(csr.indices).to(device)
    vals = torch.ones(len(csr.indices)).to(device)
    degrees = (rowptr[1:] - rowptr[:-1]).to(device)

    num_nodes = rowptr.size(0) - 1
    non_zeros = indices.size(0)

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

        num_nodes = input_feature.size(0)
        output_feat0 = torch.zeros([num_nodes, gcn_hidden], dtype=torch.float).to(device)
        output_feat1 = torch.zeros([num_nodes, out_dim], dtype=torch.float).to(device)

        # GCN Layer run
        perf_time_start("GE") if detail_test > 0 else 0

        perf_time_start("GE MM0") if detail_test == 1 else 0
        feat0 = torch.mm(input_feature, weight0)
        perf_time_end() if detail_test == 1 else 0

        print (num_nodes, gcn_hidden, non_zeros)
        print(rowptr[:10])
        print(indices[:10])

        perf_time_start("GE AGGR0") if detail_test > 0 else 0
        best_time0 = GESpMM.ge_spmm(num_nodes, gcn_hidden, non_zeros, rowptr, indices,
        degrees, feat0, output_feat0)
        #return None
        perf_time_end() if detail_test > 0 else 0

        return None

        perf_time_start("GE OTHER0") if detail_test == 1 else 0
        output_feat0 = output_feat0 + bias0
        output_feat0 = F.relu(output_feat0)
        perf_time_end() if detail_test == 1 else 0

        # GCN Layer 1 (gcn_hidden, out_dim)
        perf_time_start("GE MM1") if detail_test == 1 else 0
        feat1 = torch.mm(output_feat0, weight1)
        perf_time_end() if detail_test == 1 else 0

        perf_time_start("GE AGGR1") if detail_test > 0 else 0
        best_time1 = GESpMM.ge_spmm(num_nodes, out_dim, non_zeros, rowptr, indices,
        degrees, feat1, output_feat1)
        perf_time_end() if detail_test > 0 else 0

        perf_time_start("GE OTHER1") if detail_test == 1 else 0
        output_feat1 = output_feat1 + bias1
        output_feat1 = F.log_softmax(output_feat1, dim=1)
        perf_time_end() if detail_test == 1 else 0
    

        ge_time = perf_time_end() if detail_test > 0 else 0

        # GCN Layer run end

        return output_feat1, best_time0, best_time1
    
    else:
        print("Wrong argument of ubg_model_run!")

        return None

