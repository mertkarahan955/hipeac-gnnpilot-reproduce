from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent) + '/GNNAdvisor')

import torch
import torch.nn as nn
import torch.nn.functional as F
import GNNAdvisor as GNNA
from gnn_conv import *
from dataset import *

from scipy import sparse
import numpy as np

from perf_time import *

class GNNA_SpMM(nn.Module):
    def __init__(self, in_feats, out_feats, paras = None):
        super(GNNA_SpMM, self).__init__()
        self.name = "SpMM"
        if (paras != None):
            self.conv1 = GCNConv(in_feats, out_feats, weights = paras['w0'], bias = paras['b0'])
        else:
            self.conv1 = GCNConv(in_feats, out_feats)

    def forward(self, inputInfo, input_feature):
        x, spmm_time1 = self.conv1(input_feature, inputInfo.set_input())
        return x, spmm_time1

class GNNA_GCN(nn.Module):
    def __init__(self, in_feats, n_hidden, out_feats, paras = None):
        super(GNNA_GCN, self).__init__()
        self.name = "GCN"
        if (paras != None):
            self.conv1 = GCNConv(in_feats, n_hidden, weights = paras['w0'], bias = paras['b0'])
            self.conv2 = GCNConv(n_hidden, out_feats, weights = paras['w1'], bias = paras['b1'])
        else:
            self.conv1 = GCNConv(in_feats, n_hidden)
            self.conv2 = GCNConv(n_hidden, out_feats)

    def forward(self, inputInfo, input_feature):
        x, spmm_time1 = self.conv1(input_feature, inputInfo.set_input())
        x = F.relu(x)
        x, spmm_time2 = self.conv2(x, inputInfo.set_hidden())
        return F.log_softmax(x, dim=1), spmm_time1, spmm_time2

class GNNA_GIN(nn.Module):
    def __init__(self, in_feats, n_hidden, out_feats):
        super(GNNA_GIN, self).__init__()
        self.name = "GIN"
        self.conv1 = GCNConv(in_feats, n_hidden)
        self.conv2 = GCNConv(n_hidden, out_feats)

    def forward(self, inputInfo, input_feature):
        #print(x)
        x = self.conv1(input_feature, inputInfo.set_input())
        #print(x[0].size())
        x = F.relu(x[0])
        x = self.conv2(x, inputInfo.set_hidden())
        return F.log_softmax(x[0], dim=1)


def gnna_model_init(name, in_dim, n_hidden, out_dim, paras = None):

    if (name == "GCN"):
        model = GNNA_GCN(in_dim, n_hidden, out_dim, paras)
    elif (name == "SpMM"):
        model = GNNA_SpMM(in_dim, out_dim, paras)
    elif (name == "GIN"):
        model = GNNA_GIN(in_dim, n_hidden, out_dim)
    else:
        print("Wrong argument of gnna_model_init!")
        return None

    model.eval()

    return model

def gnna_model_run(model, inputInfo, input_feature, detail_test = False):

    if (model.name == "GCN"):
        output, spmm_time1, spmm_time2 = model(inputInfo, input_feature)

        if (detail_test):
            perf_time_set("GNNA AGGR0", spmm_time1 / 1e3)
            perf_time_set("GNNA AGGR1", spmm_time2 / 1e3)

        return output, spmm_time1, spmm_time2
    
    if (model.name == "SpMM"):
        output, spmm_time1 = model(inputInfo, input_feature)

        if (detail_test):
            perf_time_set("GNNA AGGR0", spmm_time1 / 1e3)
        
        return output, spmm_time1
    
    if (model.name == "GIN"):
        output = model(inputInfo, input_feature)
        return output


def degree_func(x):
    '''
    node degrees function
    '''
    if x > 0:
        return x
    else:
        return 1

def gnna_data_init(name, csr, in_dim, out_dim, device):
    # edge_data = graph_data.edge_index.cpu()
    # vals = torch.ones(edge_data.size(1)).numpy()
    # iidxs = edge_data[0].numpy()
    # jidxs = edge_data[1].numpy()

    # num_nodes = graph_data.num_nodes
    # # Use in-degrees
    # coo = sparse.coo_matrix((vals, (jidxs, iidxs)), shape = [num_nodes, num_nodes])
    # csr = coo.tocsr()

    # # Add self loop
    # iidxs = np.array([i for i in range(0, num_nodes)])
    # jidxs = np.array([i for i in range(0, num_nodes)])
    # vals = torch.ones(num_nodes).numpy()
    # csr2 = sparse.coo_matrix((vals, (jidxs, iidxs)), shape = [num_nodes, num_nodes]).tocsr()

    # csr = csr + csr2

    rowptr = torch.tensor(csr.indptr)
    #rowptr = torch.cat((rowptr, torch.tensor([graph_data.num_nodes])), 0)
    indices = torch.tensor(csr.indices)
    degrees = (rowptr[1:] - rowptr[:-1]).tolist()
    degrees = torch.FloatTensor(list(map(degree_func, degrees))).to(device)
    enable_rabbit = False
    manual_mode = True
    verbose_mode = False

    # When manual_mode == True, the following setups count
    if (name == "GCN"):
        partSize = 32
        dimWorker = 32
        warpPerBlock = 8
        sharedMem = 100
        hidden = 64
    elif (name == "SpMM"):
        partSize = 32
        dimWorker = 32
        warpPerBlock = 8
        sharedMem = 100
        hidden = 64
    elif (name == "GIN"):
        partSize = 32
        dimWorker = 32
        warpPerBlock = 8
        sharedMem = 100
        hidden = 64
    else:
        print("Wrong argument of gnna_model_init!")
        return None

    edge_data = csr.nonzero()
    dataset = custom_dataset2(np.array(edge_data), in_dim, out_dim, device)

    inputInfo = inputProperty(rowptr, indices, degrees,
                            partSize, dimWorker, warpPerBlock, sharedMem,
                            hiddenDim=hidden, dataset_obj=dataset, enable_rabbit=enable_rabbit,
                            manual_mode=manual_mode, verbose=verbose_mode)
    
    inputInfo.decider()

    inputInfo = inputInfo.set_input()
    if verbose_mode:
        print('----------------------------')
        inputInfo.print_param()
        print()

    inputInfo = inputInfo.set_hidden()
    if verbose_mode:
        inputInfo.print_param()
        print()
        print('----------------------------')

    #print(inputInfo.row_pointers)
    
    partPtr, part2Node = GNNA.build_part(inputInfo.partSize, inputInfo.row_pointers)

    # if verbose_mode:
    #     print("# Build nb_part (s): {:.3f}".format(build_neighbor_parts))

    inputInfo.row_pointers  = inputInfo.row_pointers.to(device)
    inputInfo.column_index  = inputInfo.column_index.to(device)
    inputInfo.partPtr = partPtr.int().to(device)
    inputInfo.part2Node  = part2Node.int().to(device)

    return inputInfo

