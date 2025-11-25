# load_dataset.py  — sadece ilgili kısım / fonksiyonu değiştir
import torch
import scanpy as sc
import numpy as np

def read_mtx(filename):
    adata = sc.read(filename)
    data = adata.X
    return data

class DataObj:
    """
    Minimal adapter object so that dict-like datasets can be used
    with code that expects attributes like .x, .edge_index, .num_nodes, .y, .edge_feat
    """
    def __init__(self, d: dict):
        # Map possible names to attributes expected by test scripts
        # Node features
        if 'x' in d:
            self.x = to_tensor_if_needed(d['x'])
        elif 'node_feat' in d:
            self.x = to_tensor_if_needed(d['node_feat'])
        elif 'node_features' in d:
            self.x = to_tensor_if_needed(d['node_features'])
        else:
            self.x = None

        # Edge index (expect shape [2, E])
        if 'edge_index' in d:
            self.edge_index = to_tensor_if_needed(d['edge_index'])
        elif 'edge_indices' in d:
            self.edge_index = to_tensor_if_needed(d['edge_indices'])
        else:
            self.edge_index = None

        # Edge features (optional)
        if 'edge_feat' in d:
            self.edge_feat = to_tensor_if_needed(d['edge_feat'])
        else:
            self.edge_feat = None

        # Labels (optional)
        if 'y' in d:
            self.y = to_tensor_if_needed(d['y'])
        elif 'label' in d:
            self.y = to_tensor_if_needed(d['label'])
        else:
            self.y = None

        # num_nodes (optional) — fallback to x.shape[0]
        if 'num_nodes' in d:
            self.num_nodes = int(d['num_nodes'])
        elif self.x is not None:
            # torch tensor or numpy array
            try:
                self.num_nodes = int(self.x.shape[0])
            except Exception:
                self.num_nodes = None
        else:
            self.num_nodes = None

    def to_dict(self):
        """If needed, convert back to dict"""
        return {'x': self.x, 'edge_index': self.edge_index, 'edge_feat': self.edge_feat, 'y': self.y, 'num_nodes': self.num_nodes}

def to_tensor_if_needed(x):
    """
    Convert numpy arrays or lists to torch.Tensor; pass through torch.Tensor.
    Keep dtype where reasonable.
    """
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x
    # numpy array or scipy sparse matrix
    try:
        import numpy as _np
        import scipy.sparse as _sp
        if isinstance(x, _sp.spmatrix):
            # convert sparse matrix to dense torch tensor if small, otherwise leave as scipy
            # But tests expect node_feat as dense; convert to dense numpy then tensor
            arr = x.toarray()
            return torch.from_numpy(arr).float()
        if isinstance(x, _np.ndarray):
            return torch.from_numpy(x).float()
    except Exception:
        pass
    # fallback: try to create tensor from list
    try:
        return torch.tensor(x).float()
    except Exception:
        # unknown type — return as-is
        return x

def dataset_load(name):
    """
    Load a .pt saved dataset. If the loaded object is a dict (OGB style), wrap it into DataObj so that
    the rest of the test code can access attributes like .x, .edge_index, .num_nodes.
    """
    dataset = torch.load(name)
    # If it's a dict, wrap with DataObj for backward-compatible attribute access
    if isinstance(dataset, dict):
        return DataObj(dataset)
    else:
        # assume it's already a PyG/DGL object with attributes
        return dataset

def dataset_prop(dataset):
    """
    Backwards compatible dataset_prop: accept either DataObj (our wrapper) or original dict or PyG Data
    """
    # If DataObj
    if isinstance(dataset, DataObj):
        x = dataset.x
        y = getattr(dataset, "y", None)
        in_feats = x.shape[1] if (hasattr(x, "shape") and x is not None) else None
        if y is None:
            out_feats = 0
        else:
            if hasattr(y, "dim") and y.dim() > 0:
                if y.dim() == 2:
                    out_feats = y.size(1) if y.size(1) > 1 else int(torch.unique(y).numel())
                else:
                    out_feats = int(torch.unique(y).numel())
            else:
                out_feats = int(len(y))
        print("Input dimension: {}, Output dimension: {}".format(in_feats, out_feats))
        try:
            print("Nodes: {}, Edges: {}".format(dataset.num_nodes, dataset.edge_index.shape[1] if dataset.edge_index is not None else 'Unknown'))
        except:
            pass
        return in_feats, out_feats

    # If dict (old fallback)
    if isinstance(dataset, dict):
        # reuse prior logic (short)
        if 'node_feat' in dataset:
            x = dataset['node_feat']
        elif 'x' in dataset:
            x = dataset['x']
        else:
            x = None

        y = dataset.get('y', None)
        # shape handling
        in_feats = x.shape[1] if hasattr(x, "shape") else None
        if y is None:
            out_feats = 0
        else:
            try:
                out_feats = y.shape[1] if hasattr(y, "shape") and len(y.shape) > 1 else int(len(set(y)))
            except:
                out_feats = 0
        print("Input dimension: {}, Output dimension: {}".format(in_feats, out_feats))
        return in_feats, out_feats

    # Otherwise assume PyG / object with attributes
    x = getattr(dataset, "x", None)
    y = getattr(dataset, "y", None)
    in_feats = x.size(1) if hasattr(x, "size") else None
    if y is None:
        out_feats = 0
    else:
        out_feats = y.size(1) if (hasattr(y, "size") and y.dim() == 2) else len(torch.unique(y))
    print("Input dimension: {}, Output dimension: {}".format(in_feats, out_feats))
    try:
        print("Nodes: {}, Edges: {}".format(dataset.x.size(0), dataset.edge_index.size(1)))
    except:
        pass
    return in_feats, out_feats
