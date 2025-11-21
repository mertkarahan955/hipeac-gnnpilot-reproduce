import torch
import scanpy as sc

def read_mtx(filename):
    adata = sc.read(filename)
    data = adata.X
    return data

def dataset_load(name):
    name = name
    dataset = torch.load(name)
    return dataset

def dataset_prop(dataset):
    in_feats = dataset.x.size(1)
    if (dataset.y.dim() == 2):
        if (dataset.y.size(1) > 1):
            out_feats = dataset.y.size(1)
        else:
            out_feats = len(torch.unique(dataset.y))
    elif (dataset.y.dim() == 1):
        out_feats = len(torch.unique(dataset.y))

    print("Input dimension: {}, Output dimension: {}".format(in_feats, out_feats))
    print("Nodes: {}, Edges: {}".format(dataset.x.size(0), dataset.edge_index.size(1)))

    return in_feats, out_feats
