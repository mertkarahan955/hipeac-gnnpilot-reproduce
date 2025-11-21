from scipy import sparse
import numpy as np
import torch
import similarity as sim
from perf_time import *
from math import sqrt

def csr_similarity2(RowPtr, ColIdx, Vals):
    
    # print(RowPtr)
    # print(ColIdx)
    similarity = 0.0
    act_count = 0

    for i in range(len(RowPtr) - 1):
    #for i in Cython.Shadow.CythonDotParallel.prange(len(RowPtr) - 1):
        n1 = set(ColIdx[RowPtr[i]: RowPtr[i+1]])
        size_n1 = RowPtr[i + 1] - RowPtr[i]
        for j in range(RowPtr[i], RowPtr[i + 1]):
            nid = ColIdx[j]
            if nid == i:
                Vals[j] = 0.0
            else:
                n2 = set(ColIdx[RowPtr[nid]: RowPtr[nid+1]])
                n_common = list(n1 & n2)
                # if len(n_common) > common_size:
                #     common_size = len(n_common)
                Vals[j] = 1.0 * len(n_common) / size_n1
            similarity += Vals[j]

    similarity = similarity / (len(ColIdx) - len(RowPtr) + 1)
    print(similarity)

    # block_size = 1024

    # for i in range(100):
    #     # if (RowPtr[i+1] - RowPtr[i] == 1):
    #     #     continue
    #     # act_count += 1

    #     # ibin_size = i / block_size
    #     # ibin_count = 0

    #     # for j in range(RowPtr[i], RowPtr[i+1]):
    #     #     if (ColIdx[j] >= ibin_size and ColIdx[j] < ibin_size + block_size):
    #     #         ibin_count += 1
    #     n1 = set(ColIdx[RowPtr[i]: RowPtr[i+1]])
    #     print(ColIdx[RowPtr[i]: RowPtr[i+1]])
    #     common_size = 0
    #     for j in range(RowPtr[i], RowPtr[i + 1]):
    #         nid = ColIdx[j]
    #         if nid == i:
    #             continue
    #         n2 = set(ColIdx[RowPtr[nid]: RowPtr[nid+1]])
    #         n_common = list(n1 & n2)
    #         if len(n_common) > common_size:
    #             common_size = len(n_common)
    #     item = 1.0 * common_size / len(n1)
    #     #print("i", i, len(n1), item)
    #     similarity += ibin_count / (RowPtr[i+1] - RowPtr[i])

    # similarity /= act_count
    # print("similarity", similarity)

def graph_to_csr(graph_data, add_self_loop = True, use_similarity = False, \
show_degree = False, reorder = False):
    num_nodes = graph_data.x.size(0)
    edge_data = graph_data.edge_index.cpu()

    if (reorder):
        edge_data = rabbit.reorder(edge_data.int())

    vals = torch.ones(edge_data.size(1))
    iidxs = edge_data[0].numpy()
    jidxs = edge_data[1].numpy()

    coo = sparse.coo_matrix((vals.numpy(), (jidxs, iidxs)), shape = [num_nodes, num_nodes])
    csr = coo.tocsr()

    #csr_similarity2(csr.indptr, csr.indices, csr.data)
    if (use_similarity):
        perf_time_start("csr_similarity")
        similarity = rabbit.csr_similarity(torch.tensor(csr.indptr, dtype=torch.int32), 
        torch.tensor(csr.indices, dtype=torch.int32))

        edge_data = sim.similarity_reorder_merge(num_nodes, edge_data, similarity, 1024)
        perf_time_end()

        vals = torch.ones(edge_data.size(1))
        iidxs = edge_data[0].numpy()
        jidxs = edge_data[1].numpy()

        coo = sparse.coo_matrix((vals.numpy(), (jidxs, iidxs)), shape = [num_nodes, num_nodes])
        csr = coo.tocsr()

    # Add self loop
    if (add_self_loop):
        iidxs = np.array([i for i in range(0, num_nodes)])
        jidxs = np.array([i for i in range(0, num_nodes)])
        vals = torch.ones(num_nodes).numpy()
        csr2 = sparse.coo_matrix((vals, (jidxs, iidxs)), shape = [num_nodes, num_nodes]).tocsr()
        csr = csr + csr2

    if (show_degree):
        rowptr = csr.indptr
        degrees = torch.tensor(rowptr[1:] - rowptr[:-1], dtype=torch.float)
        print("Num nodes: {}".format(len(rowptr) - 1))
        print("Num edges: {}".format(len(csr.indices)))
        print("Average degree: {:.2f}".format(torch.mean(degrees, dim=0).item()))
        print("Variation degree: {:.2f}".format(torch.var(degrees, dim=0).item()))
        print("Max degree: {:.2f}".format(torch.max(degrees).item()))

    return csr

def coo_to_csr(num_nodes, graph_coo, add_self_loop = True, use_similarity = False, \
show_degree = False, reorder = False):

    csr = graph_coo.tocsr()

    #csr_similarity2(csr.indptr, csr.indices, csr.data)
    if (use_similarity):
        perf_time_start("csr_similarity")
        similarity = rabbit.csr_similarity(torch.tensor(csr.indptr, dtype=torch.int32), 
        torch.tensor(csr.indices, dtype=torch.int32))

        edge_data = sim.similarity_reorder_merge(num_nodes, edge_data, similarity, 1024)
        perf_time_end()

        vals = torch.ones(edge_data.size(1))
        iidxs = edge_data[0].numpy()
        jidxs = edge_data[1].numpy()

        coo = sparse.coo_matrix((vals.numpy(), (jidxs, iidxs)), shape = [num_nodes, num_nodes])
        csr = coo.tocsr()

    # Add self loop
    if (add_self_loop):
        iidxs = np.array([i for i in range(0, num_nodes)])
        jidxs = np.array([i for i in range(0, num_nodes)])
        vals = torch.ones(num_nodes).numpy()
        csr2 = sparse.coo_matrix((vals, (jidxs, iidxs)), shape = [num_nodes, num_nodes]).tocsr()
        csr = csr + csr2

    if (show_degree):
        rowptr = csr.indptr
        degrees = torch.tensor(rowptr[1:] - rowptr[:-1], dtype=torch.float)
        print("Num nodes: {}".format(len(rowptr) - 1))
        print("Num edges: {}".format(len(csr.indices)))
        print("Average degree: {:.2f}".format(torch.mean(degrees, dim=0).item()))
        print("Variation degree: {:.2f}".format(torch.var(degrees, dim=0).item()))
        print("Max degree: {:.2f}".format(torch.max(degrees).item()))

    return csr

def write_csr(filename, RowPtr, ColIdx):
    f = open(filename, "w")
    n = len(RowPtr) - 1
    nnz = len(ColIdx)
    f.write("{} {}\n".format(n, nnz))
    for ptr in RowPtr:
        f.write("{} ".format(ptr))
    f.write("\n")
    for col in ColIdx:
        f.write("{} ".format(col))
    f.write("\n")

def read_csr(filename):
    f = open(filename, "r")
    line = f.readline()
    tmp = line.strip().split()
    m = int(tmp[0])
    nnz = int(tmp[1])

    line = f.readline()
    rowptr = np.array([int(ptr) for ptr in line.strip().split()], dtype=np.int32)

    line = f.readline()
    indices = np.array([int(idx) for idx in line.strip().split()], dtype=np.int32)

    values = np.ones(nnz)

    csr = sparse.csr_matrix((m, m))
    csr.indptr = rowptr
    csr.indices = indices
    csr.data = values

    return csr

def coo_get_metrics(num_nodes, graph_coo, print_out = False, print_file = None):

    csr = graph_coo.tocsr()
    rowptr = csr.indptr
    degrees = torch.tensor(rowptr[1:] - rowptr[:-1], dtype=torch.float)

    if (print_file):
        write_csr(print_file, rowptr, csr.indices)
    nnz = len(csr.indices)
    avg_degree = torch.mean(degrees, dim=0).item()
    var_degree = torch.var(degrees, dim=0).item()
    dev_degree = sqrt(var_degree)
    max_degree = torch.max(degrees).item()
    if (print_out):
        print("Num nodes: {:d}".format(len(rowptr) - 1))
        print("Num edges: {:d}".format(nnz))
        print("Average degree: {:.2f}".format(avg_degree))
        print("Variation degree: {:.2f}".format(var_degree))
        print("Max degree: {:.2f}".format(max_degree))
    
    return nnz, avg_degree, dev_degree, max_degree

def csr_get_metrics(num_nodes, csr, print_out = False, print_file = None):

    rowptr = csr.indptr
    degrees = torch.tensor(rowptr[1:] - rowptr[:-1], dtype=torch.float)

    if (print_file):
        write_csr(print_file, rowptr, csr.indices)
    nnz = len(csr.indices)
    avg_degree = torch.mean(degrees, dim=0).item()
    var_degree = torch.var(degrees, dim=0).item()
    dev_degree = sqrt(var_degree)
    max_degree = torch.max(degrees).item()
    if (print_out):
        print("Num nodes: {:d}".format(len(rowptr) - 1))
        print("Num edges: {:d}".format(nnz))
        print("Average degree: {:.2f}".format(avg_degree))
        print("Variation degree: {:.2f}".format(var_degree))
        print("Max degree: {:.2f}".format(max_degree))
    
    return nnz, avg_degree, dev_degree, max_degree

# no reorder
def csr_transform(csr, add_self_loop = False, use_similarity = False, \
show_degree = False, reorder = False):

    num_nodes = len(csr.indptr) - 1
    num_edges = len(csr.indices)

    # if (reorder):
    #     # edge_data = for () in graph_data.edge_index.cpu()
    #     edge_data = rabbit.reorder(edge_data.int())

    # Add self loop
    if (add_self_loop):
        iidxs = np.array([i for i in range(0, num_nodes)])
        jidxs = np.array([i for i in range(0, num_nodes)])
        vals = torch.ones(num_nodes).numpy()
        csr2 = sparse.coo_matrix((vals, (jidxs, iidxs)), shape = [num_nodes, num_nodes]).tocsr()
        csr = csr + csr2

    if (show_degree):
        rowptr = csr.indptr
        degrees = torch.tensor(rowptr[1:] - rowptr[:-1], dtype=torch.float)
        print("Num nodes: {}".format(len(rowptr) - 1))
        print("Num edges: {}".format(len(csr.indices)))
        print("Average degree: {:.2f}".format(torch.mean(degrees, dim=0).item()))
        print("Variation degree: {:.2f}".format(torch.var(degrees, dim=0).item()))
        print("Max degree: {:.2f}".format(torch.max(degrees).item()))
    
    return csr
