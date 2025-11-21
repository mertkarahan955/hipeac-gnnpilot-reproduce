import argparse
import time
import torch
import torch.nn.functional as F
import dgl
# import dgl.data

import torch.nn as nn
import sys
import GPUtil
sys.path.append('../UGCG')
sys.path.append('../../UGCG')
from layers.gatconv_layer import GATConv
from layers.gmmconv_layer import GMMConv
from util.indicator import *
import scipy.sparse as sp

class UGCG_GAT(torch.nn.Module):
    def __init__(self, in_feats, n_hidden, out_feats, relu_l, paras = None):
        super(UGCG_GAT, self).__init__()
        self.conv1 = GATConv(in_feats, n_hidden, 1, negative_slope = relu_l)
        self.conv2 = GATConv(n_hidden, out_feats, 1, negative_slope = relu_l)
        self.model_name = 'GAT'

        state_dict = self.state_dict()
        # print(state_dict)

        if (paras):
            state_dict = self.state_dict()
            state_dict['conv1.W'] = paras['w0']
            state_dict['conv1.attn_r'] = paras['as0'].view(-1, 1, n_hidden)
            state_dict['conv1.attn_l'] = paras['ad0'].view(-1, 1, n_hidden)
            state_dict['conv1.bias'] = paras['b0'].view(-1, 1, n_hidden)
            state_dict['conv2.W'] = paras['w1']
            state_dict['conv2.attn_r'] = paras['as1'].view(-1, 1, out_feats)
            state_dict['conv2.attn_l'] = paras['ad1'].view(-1, 1, out_feats)
            state_dict['conv2.bias'] = paras['b1'].view(-1, 1, out_feats)

            self.load_state_dict(state_dict)
    
    def forward(self, row_ptr, col_idx, input_feature):
        x = input_feature
        x, _ = self.conv1(row_ptr, col_idx, None, None, input_feature)
        x, layer_time = self.conv2(row_ptr, col_idx, None, None, x)
        return x, layer_time

class UGCG_GMM(torch.nn.Module):
    def __init__(self, in_feats, n_hidden, out_feats, dim, n_kernels, paras = None):
        super(UGCG_GMM, self).__init__()
        self.conv1 = GMMConv(in_feats, n_hidden, dim, n_kernels)
        self.conv2 = GMMConv(n_hidden, out_feats, dim, n_kernels)
        self.model_name = 'GMM'
    
    def forward(self, row_ptr, col_idx, input_feature, p, mu, sigma):
        x = input_feature
        x, _ = self.conv1(row_ptr, col_idx, None, None, None, x, p, mu, sigma)
        x, layer_time = self.conv2(row_ptr, col_idx, None, None, None, x, p, mu, sigma)
        return x, layer_time


def ugcg_model_init(name, in_dim, gcn_hidden, out_dim, relu_l = 0.2, kernel_dim = 0, kernel_num = 0, paras = None):

    if (name == "GAT"):
        model = UGCG_GAT(in_dim, gcn_hidden, out_dim, relu_l, paras)
    elif (name == "GMM"):
        model = UGCG_GMM(in_dim, gcn_hidden, out_dim, kernel_dim, kernel_num)
    else:
        print("Wrong argument of dgl_model_init!")
        return None
    
    model.eval()

    return model

def ugcg_model_run(model, graph_csr, feature, device, p = None, mu = None, sigma = None):

    rowptr = torch.tensor(graph_csr.indptr)
    indices = torch.tensor(graph_csr.indices)
    rowptr = rowptr.to(device)
    indices = indices.to(device)

    if (model.model_name == "GAT"):
        output, layer_time = model(rowptr, indices, feature)
    elif(model.model_name == "GMM"):
        output, layer_time = model(rowptr, indices, feature, p, mu, sigma)

    return output, layer_time

