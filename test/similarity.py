from scipy import sparse
import numpy as np
import torch
from functools import cmp_to_key
from perf_time import *

def get_top(i, top):
    if (top[i] != i):
        top[i] = get_top(top[i], top)
        return (top[i])
    else:
        return i

def similarity_reorder_merge(nodes, edge_index, similarity, l1_size):
    edges = edge_index.size(1)
    edge_index = edge_index.tolist()
    similarity = similarity.tolist()

    print(len(edge_index[0]), len(similarity))

    # print(edge_index)
    # print(similarity)

    edge_perm = list(range(0, edges))

    # print(edge_perm)

    def similarity_compare(x, y):
        return similarity[x] > similarity[y]

    edge_perm = sorted(edge_perm, key = lambda x: similarity[x], reverse = True)
    #edge_perm = sorted(edge_perm, reverse = True)

    #print(edge_perm)

    union_size = [1] * nodes
    top = list(range(0, nodes))

    merges = 0

    for edge in range(edges):
        i = edge_index[0][edge_perm[edge]]
        j = edge_index[1][edge_perm[edge]]

        topi = get_top(i, top)
        topj = get_top(j, top)

        #print(i, j, similarity[edge_perm[edge]])

        # if (similarity[edge_perm[edge]] < 0.02):
        #     break

        #if (topi != topj):
        if (union_size[i] + union_size[j] <= l1_size and topi != topj):
            top[topj] = topi
            union_size[topi] += union_size[topj]
            merges += 1

        if (merges == nodes - 1):
            break

    for node in range(nodes):
        get_top(node, top)
    node_perm = list(range(0, nodes))
    
    def top_compare(x, y):
        return top[x] < top[y]
    
    node_perm = sorted(node_perm, key = lambda x: top[x])

    for edge in range(edges):
        i = edge_index[0][edge]
        j = edge_index[1][edge]

        edge_index[0][edge] = node_perm[i]
        edge_index[1][edge] = node_perm[j]

    unique_top = set(top)
    print(len(unique_top))

    edge_index = torch.Tensor(edge_index)

    return edge_index

    #print(top)
    #print(edge_index)