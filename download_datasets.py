from ogb.nodeproppred import NodePropPredDataset
d = NodePropPredDataset(name='ogbn-products', root='./datasets')
print('products downloaded OK')