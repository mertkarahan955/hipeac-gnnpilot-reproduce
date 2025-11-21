import dgl
import torch
import torch.nn.functional as F
import torch.nn as nn
from dgl.nn import GraphConv
from dgl.nn import GATConv
from dgl.nn import GINConv
from load_dataset import *
import sys

class DGL_GCN(nn.Module):
    def __init__(self, in_feats, n_hidden, out_feats, paras = None):
        super(DGL_GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, n_hidden, activation = F.relu, norm='right')
        self.conv2 = GraphConv(n_hidden, out_feats, norm='right')
        
        if (paras):
            state_dict = self.state_dict()
            state_dict['conv1.weight'] = paras['w0']
            state_dict['conv1.bias'] = paras['b0']
            state_dict['conv2.weight'] = paras['w1']
            state_dict['conv2.bias'] = paras['b1']
            self.load_state_dict(state_dict)
            #print("weight init", self.state_dict()['conv1.weight'])
    
    def forward(self, g, features):
        x = self.conv1(g, features)
        x = self.conv2(g, x)
        return F.log_softmax(x, dim=1)

class DGL_GAT(nn.Module):
    def __init__(self, in_feats, n_hidden, out_feats, relu_l = 0.2, paras = None):
        super(DGL_GAT, self).__init__()
        self.conv1 = GATConv(in_feats, n_hidden, num_heads = 1, negative_slope = relu_l)
        self.conv2 = GATConv(n_hidden, out_feats, num_heads = 1, negative_slope = relu_l)
        
        state_dict = self.state_dict()
        # print(state_dict)
        # print(state_dict['conv1.fc.weight'].size())
        # print(state_dict['conv1.attn_l'].size())
        # print(state_dict['conv1.attn_r'].size())

        if (paras):
            state_dict = self.state_dict()
            state_dict['conv1.fc.weight'] = paras['w0'].transpose(0, 1)
            state_dict['conv1.attn_l'] = paras['as0']
            state_dict['conv1.attn_r'] = paras['ad0']
            state_dict['conv1.bias'] = paras['b0']

            state_dict['conv2.fc.weight'] = paras['w1'].transpose(0, 1)
            state_dict['conv2.attn_l'] = paras['as1']
            state_dict['conv2.attn_r'] = paras['ad1']
            state_dict['conv2.bias'] = paras['b1']

            self.load_state_dict(state_dict)
            #print("weight init", self.state_dict()['conv1.weight'])
    
    def forward(self, g, features):
        x = self.conv1(g, features)
        #print(x)
        x = self.conv2(g, x)
        return x

class DGL_GIN(torch.nn.Module):
    def __init__(self, in_dim, gcn_hidden, out_dim, paras = None):
        super(DGL_GIN, self).__init__()
        nn1 = nn.Linear(in_dim, gcn_hidden)
        nn2 = nn.Linear(gcn_hidden, out_dim)

        self.conv1 = GINConv(nn1)
        self.conv2 = GINConv(nn2)

        if (paras):
            state_dict = self.state_dict()
            state_dict['conv1.apply_func.weight'] = paras['w0'].transpose(0, 1)
            state_dict['conv1.apply_func.bias'] = paras['b0']
            state_dict['conv2.apply_func.weight'] = paras['w1'].transpose(0, 1)
            state_dict['conv2.apply_func.bias'] = paras['b1']

            self.load_state_dict(state_dict)

    def forward(self, g, features):
        x = self.conv1(g, features)
        x = F.relu(x)
        x = self.conv2(g, x)
        return F.log_softmax(x, dim = 1)

def dgl_model_init(name, in_dim, gcn_hidden, out_dim, relu_l = 0.2, paras = None):

    if (name == "GCN"):
        model = DGL_GCN(in_dim, gcn_hidden, out_dim, paras)
    elif (name == "GAT"):
        model = DGL_GAT(in_dim, gcn_hidden, out_dim, relu_l = relu_l, paras = paras)
    elif (name == "GIN"):
        model = DGL_GIN(in_dim, gcn_hidden, out_dim, paras = paras)
    else:
        print("Wrong argument of dgl_model_init!")
        return None

    model.eval()

    return model

def dgl_model_run(model, g, feature):

    output = model(g, feature)

    return output

def dgl_data_init(graph_data):

    g = dgl.graph((graph_data.edge_index[0], graph_data.edge_index[1]))
    #g = dgl.add_self_loop(g)

    return g