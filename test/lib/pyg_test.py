import torch_geometric
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, GINConv
from torch_geometric.data import Data

import time

class PYG_GCN(torch.nn.Module):
    def __init__(self, in_feats, n_hidden, out_feats, paras = None):
        super(PYG_GCN, self).__init__()
        self.conv1 = GCNConv(in_feats, n_hidden)
        self.conv2 = GCNConv(n_hidden, out_feats)

        if (paras):
            state_dict = self.state_dict()
            state_dict['conv1.lin.weight'] = paras['w0'].transpose(0, 1)
            state_dict['conv1.bias'] = paras['b0']
            state_dict['conv2.lin.weight'] = paras['w1'].transpose(0, 1)
            state_dict['conv2.bias'] = paras['b1']
            self.load_state_dict(state_dict)
            #print(self.state_dict()['conv1.lin.weight'])

    def forward(self, data, input_feature):
        x, edge_index = input_feature, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        t2 = time.perf_counter()
        return F.log_softmax(x, dim=1)

class PYG_GAT(torch.nn.Module):
    def __init__(self, in_feats, n_hidden, out_feats, relu_l, paras = None):
        super(PYG_GAT, self).__init__()
        self.conv1 = GATConv(in_feats, n_hidden, negative_slope = relu_l)
        self.conv2 = GATConv(n_hidden, out_feats, negative_slope = relu_l)
        
        state_dict = self.state_dict()
        # print(state_dict)
        if (paras):
            state_dict = self.state_dict()
            state_dict['conv1.lin_src.weight'] = paras['w0'].transpose(0, 1)
            state_dict['conv1.lin_dst.weight'] = paras['w0'].transpose(0, 1)
            state_dict['conv1.att_src'] = paras['as0']
            state_dict['conv1.att_dst'] = paras['ad0']
            state_dict['conv1.bias'] = paras['b0']

            state_dict['conv2.lin_src.weight'] = paras['w1'].transpose(0, 1)
            state_dict['conv2.lin_dst.weight'] = paras['w1'].transpose(0, 1)
            state_dict['conv2.att_src'] = paras['as1']
            state_dict['conv2.att_dst'] = paras['ad1']
            state_dict['conv2.bias'] = paras['b1']

            self.load_state_dict(state_dict)

    def forward(self, data, input_feature):
        x, edge_index = input_feature, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        return x

class PYG_GIN(torch.nn.Module):
    def __init__(self, in_dim, gcn_hidden, out_dim, paras = None):
        super(PYG_GIN, self).__init__()
        nn1 = nn.Linear(in_dim, gcn_hidden)
        nn2 = nn.Linear(gcn_hidden, out_dim)

        self.conv1 = GINConv(nn1)
        self.conv2 = GINConv(nn2)

        if (paras):
            state_dict = self.state_dict()
            state_dict['conv1.nn.weight'] = paras['w0'].transpose(0, 1)
            state_dict['conv1.nn.bias'] = paras['b0']
            state_dict['conv2.nn.weight'] = paras['w1'].transpose(0, 1)
            state_dict['conv2.nn.bias'] = paras['b1']

            self.load_state_dict(state_dict)

    def forward(self, data, input_feature):
        x, edge_index = input_feature, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        
        return F.log_softmax(x, dim = 1)


def pyg_model_init(name, in_dim, gcn_hidden, out_dim, relu_l = 0.2, nn = nn, paras = None):

    if (name == "GCN"):
        model = PYG_GCN(in_dim, gcn_hidden, out_dim, paras)
    elif (name == "GAT"):
        model = PYG_GAT(in_dim, gcn_hidden, out_dim, relu_l, paras = paras)
    elif (name == "GIN"):
        model = PYG_GIN(in_dim, gcn_hidden, out_dim, paras)
    else:
        print("Wrong argument of pyg_model_init!")
        return None

    model.eval()

    return model

def pyg_model_run(model, graph_data, input_feature):

    output = model(graph_data, input_feature)

    return output

