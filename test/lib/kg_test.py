from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch.nn.functional as F

import torch
# import KGGNN
torch.ops.load_library("../KG_GNN/build/libKGGNN.so")
# torch.ops.load_library("../../KG_GNN/build/libKGGNN.so")
print(torch.ops.KGGNN)

import utils
from perf_time import *
import time

from ctypes import *

import math

detail_test = True

@torch.jit.script
def jit_mm(x, y):
    return torch.mm(x, y)

@torch.jit.script
def jit_other(x, y):
    x = x + y
    x = F.relu(x)
    return x

class kg_paras():
    def __init__(self):
        self.strategy_id = [2]
        self.strategy_id = [2, 6]
        #self.strategy_id = [6]

def kg_autotuning(csr, input_feature, in_dim, out_dim, device, name = 'GCN'):

    m = len(csr.indptr) - 1
    nnz = len(csr.indices)
    avg_rnz = 1.0 * nnz / m
    max_rnz = max(csr.indptr[1: m+1] - csr.indptr[0: m])

    rowptr = torch.tensor(csr.indptr)
    indices = torch.tensor(csr.indices)

    rowptr_d = rowptr.to(device)
    indices_d = indices.to(device)

    paras = kg_paras()

    best_time = -1
    best_strategy = -1
    best_paras = []

    for strategy in paras.strategy_id:
        if (strategy == 2):

            for alpha in [1, 5, 10, 15]:
                i = max(math.pow(2, math.ceil(math.log2(avg_rnz))),
                math.pow(2, math.ceil(math.log2(alpha))))
                #i = 16
                ed = math.pow(2, math.floor(math.log2(max_rnz)))
                ed = max(ed, 1024)

                output_tmp = torch.zeros(m, out_dim).to(device)
                weight_lr_tmp = torch.randn(out_dim, 2).to(device)
                edge_weight_tmp = torch.zeros(nnz).to(device)
                sum_vec_tmp = torch.zeros(m).to(device)

                while (i <= ed):

                    if (name == 'GCN'):
                        ana_info = torch.ops.KGGNN.kg_gcn_balance2(rowptr, indices, int(i), alpha)

                        perf_time_start("tuning")
                        torch.ops.KGGNN.kg_gcn_run_balance(rowptr_d, indices_d, input_feature, output_tmp, ana_info)
                        tuning_time = perf_time_end(verbose = False)
                    elif (name == 'GAT'):

                        ana_info = torch.ops.KGGNN.kg_gcn_balance2(rowptr, indices, int(i), alpha)

                        perf_time_start("tuning")
                        torch.ops.KGGNN.kg_gat_run_balance(rowptr_d, indices_d, input_feature, output_tmp, 
                        weight_lr_tmp, sum_vec_tmp, edge_weight_tmp, 0.2, ana_info)
                        tuning_time = perf_time_end(verbose = False)

                        #print(tuning_time)

                    if (tuning_time < best_time or best_time == -1):
                        best_time = tuning_time
                        best_strategy = 2
                        best_paras = [i, alpha]

                    torch.ops.KGGNN.kg_gcn_finalize(ana_info)
                    i = i * 2
            
            print("Best balance 2", best_paras)
            print("Best time 2", best_time)
        
        best_time6 = -1

        if (strategy == 6):
            bin_size_list = [256, 512, 1024]#, 2048, 4096, 8192]
            #bin_size_list = [960, 1920]
            pack_size_list = [1]
            #pack_size_list = [1, 8, 64, 128, 256]
            #alpha_list = [5, 10, 15, 20]
            alpha_list = [5]
            bin_block_list = [1, 2, 4, 8, 16, 32, 64]

            bin_thresh_list = [item for item in alpha_list]
            #bin_thresh_list = [0, 5, 10, 15, 20]

            output_tmp = torch.zeros(m, out_dim).to(device)

            for bin_size in bin_size_list:
                for pack_size in pack_size_list:
                    for alpha in alpha_list:
                        for bin_block in bin_block_list:
                            st = int(max(math.ceil(math.log2(avg_rnz)), math.ceil(math.log2(alpha))))
                            ed = int(math.floor(math.log2(max_rnz)))
                            st = 7
                            ed = 8
                            wsize_list = [int(math.pow(2, item)) for item in range(st, ed)]
                            for wsize in wsize_list:
                                for bin_thresh in bin_thresh_list:
                                    if (bin_thresh < alpha):
                                        continue
                                    
                                    #print(bin_size, pack_size, alpha, bin_block)
                                    
                                    ana_info = torch.ops.KGGNN.kg_gcn_bin_pack(rowptr, indices, bin_size, pack_size,
                                    bin_thresh, bin_block, wsize, alpha)

                                    perf_time_start("tuning")
                                    torch.ops.KGGNN.kg_gcn_run_balance(rowptr_d, indices_d, input_feature, output_tmp, ana_info)
                                    tuning_time = perf_time_end(verbose = False)

                                    if (tuning_time < best_time or best_time == -1):
                                        best_time = tuning_time
                                        best_strategy = 6
                                        best_paras = [bin_size, pack_size, bin_thresh, bin_block, wsize, alpha]

                                    if (tuning_time < best_time6 or best_time6 == -1):
                                        best_time6 = tuning_time
                                    
                                    torch.ops.KGGNN.kg_gcn_finalize(ana_info)

                                    # print("current", [bin_size, pack_size, bin_thresh, bin_block, wsize, alpha])
                                    # print("best", best_strategy, best_paras)
                                    # print(tuning_time, best_time)
            print("best balance 6 time", best_time6)

    print(best_strategy, best_paras, best_time)
    return best_strategy, best_paras

def get_ana(rowptr, indices, feat_len, balance, op_name = 'GCN'):
    m = len(rowptr) - 1
    nnz = len(indices)
    avg_rnz = float(nnz) / m

    if (balance == -1):
        if (avg_rnz > 100):
            balance = 6
        else:
            balance = 2

    if (balance):
        if (balance == 1):
            ana_info = KGGNN.kg_gcn_balance(rowptr, indices, 32)
        if (balance == 2):
            alpha = 0
            if (op_name == 'GCN' or op_name == 'SpMM'):
                alpha = 10
                delta = 4
                if (avg_rnz < 16):
                    np = 128
                elif (avg_rnz < 128):
                    np = 256
                else:
                    np = 512
                np = np / math.ceil(1.0 * feat_len / 32 / 2)

            elif (op_name == 'GAT'):
                alpha = 10
                delta = 4
                np = 128
                #np = math.pow(2, int(math.log2(avg_rnz + 1)) + delta)
            elif (op_name == 'SDDMM'):
                alpha = 0
                delta = 2
                np = math.pow(2, int(math.log2(avg_rnz + 1)) + delta)
            else:
                print("Op not implemented!")
                return None
            
            if (np < 32):
                np = 32

            print("avg_nnz {:2f} parameter {:.0f} {:.0f}".format(avg_rnz, np, alpha))
            ana_info = torch.ops.KGGNN.kg_gcn_balance2(rowptr, indices, int(np), int(alpha))

            # GCN
            # arxiv collab mag
            # ana_info = KGGNN.kg_gcn_balance2(rowptr, indices, 128, 10)
            # ppa products
            # ana_info = KGGNN.kg_gcn_balance2(rowptr, indices, 256, 15)
            # reddit
            # ana_info = KGGNN.kg_gcn_balance2(rowptr, indices, 512, 15)
            # arxiv collab mag 512
            # ana_info = KGGNN.kg_gcn_balance2(rowptr, indices, 64, 10)
            # ppa products 512
            # ana_info = KGGNN.kg_gcn_balance2(rowptr, indices, 512, 15)
            # ddi 
            # ana_info = KGGNN.kg_gcn_balance2(rowptr, indices, 1024, 15)

            # GAT
            # arxiv collab mag
            # ana_info = KGGNN.kg_gcn_balance2(rowptr, indices, 64, 10)
            # ppa products
            # ana_info = KGGNN.kg_gcn_balance2(rowptr, indices, 128, 10)
            # ddi
            # ana_info = KGGNN.kg_gcn_balance2(rowptr, indices, 1024, 10)

            # SDDMM
            # arxiv collab mag
            #ana_info = KGGNN.kg_gcn_balance2(rowptr, indices, 32, 0)
            # ppa products
            # ana_info = KGGNN.kg_gcn_balance2(rowptr, indices, 64, 0)
            # ddi products reddit
            #ana_info = KGGNN.kg_gcn_balance2(rowptr, indices, 1024, 0)

        if (balance == 3):
            ana_info = torch.ops.KGGNN.kg_gcn_balance3(rowptr, indices, 128, 15)
        if (balance == 4):
            ana_info = torch.ops.KGGNN.kg_gcn_balance4(rowptr, indices, 1024)
        if (balance == 5):
            ana_info = torch.ops.KGGNN.kg_gcn_schedule_locality(rowptr, indices, 1024)
        if (balance == 6):
            #ana_info = KGGNN.kg_gcn_bin_pack(rowptr, indices, 1024, 8, 15, 8, 128, 15)
            #proteins
            #ana_info = KGGNN.kg_gcn_bin_pack(rowptr, indices, 1024, 1, 10, 24, 128, 10)
            #ana_info = KGGNN.kg_gcn_bin_pack(rowptr, indices, 1024, 1, 10, 16, 256, 5)
            #ana_info = KGGNN.kg_gcn_bin_pack(rowptr, indices, 1920, 1, 20, 8, 256, 15)
            #ana_info = KGGNN.kg_gcn_bin_pack(rowptr, indices, 960, 1, 15, 8, 256, 10)
            #ana_info = KGGNN.kg_gcn_bin_pack(rowptr, indices, 4096, 64, 20, 8, 1024, 10)
            #ddi
            #ana_info = KGGNN.kg_gcn_bin_pack(rowptr, indices, 960, 256, 20, 32, 64, 10)
            #ana_info = KGGNN.kg_gcn_bin_pack(rowptr, indices, 960, 64, 20, 8, 64, 5)
            #ana_info = KGGNN.kg_gcn_bin_pack(rowptr, indices, 1024, 8, 20, 64, 128, 15)
            #ana_info = KGGNN.kg_gcn_bin_pack(rowptr, indices, 1070, 4, 15, 8, 32, 15)
            #helloworld
            #ana_info = KGGNN.kg_gcn_bin_pack(rowptr, indices, 960, 256, 0, 32, 64, 10)
            #test ppa
            #ana_info = KGGNN.kg_gcn_bin_pack(rowptr, indices, 1024, 1, 20, 24, 64, 10)

            #GAT proteins
            #ana_info = KGGNN.kg_gcn_bin_pack(rowptr, indices, 256, 1, 10, 24, 128, 10)
            #ddi
            ana_info = torch.ops.KGGNN.kg_gcn_bin_pack(rowptr, indices, 1024, 256, 20, 32, 64, 10)
    else:
        ana_info = None
    return ana_info

def kg_run_kernel(name, csr, input_feature, feat_dim, device,
balance = 0, detail_test = 0, time_label = 'Pck'):
    rowptr = torch.tensor(csr.indptr)
    indices = torch.tensor(csr.indices)

    ana_info = get_ana(rowptr, indices, feat_dim, balance, op_name = name)

    num_nodes = len(rowptr) - 1
    num_edges = len(indices)
    rowptr = rowptr.to(device)
    indices = indices.to(device)

    if (name == 'SpMM'):

        output0 = torch.zeros(num_nodes, feat_dim).to(device)

        perf_time_start(time_label) if detail_test > 0 else 0
        if (ana_info):
            torch.ops.KGGNN.kg_gcn_run_balance(rowptr, indices, input_feature, output0, ana_info)
        else:
            torch.ops.KGGNN.kg_gcn_run(rowptr, indices, input_feature, output0)
        aggr0_time = perf_time_end() if detail_test > 0 else 0

    elif (name == 'SDDMM'):

        output0 = torch.zeros(num_edges).to(device)

        perf_time_start(time_label) if detail_test > 0 else 0
        #print(input_feature[0, :, :].size(), input_feature[1, :, :].size())
        if (ana_info):
            torch.ops.KGGNN.kg_sddmm_run_balance(rowptr, indices, input_feature[0, :, :], \
            input_feature[1, :, :], output0, ana_info)
        else:
            print("Not implemented!")
        aggr0_time = perf_time_end() if detail_test > 0 else 0

    del rowptr
    del indices
    if (ana_info):
        torch.ops.KGGNN.kg_gcn_finalize(ana_info)
    #torch.cuda.synchronize()

    return output0, aggr0_time


def kg_run_tmp(csr, input_feature, in_dim, gcn_hidden, out_dim, device, 
balance = 0, paras = None, detail_test = 0, reorder = False, time_label = 'KG', name = 'GCN'):

    if (name == 'GCN'):
        #csr = utils.graph_to_csr(graph_data, reorder = reorder)
        rowptr = torch.tensor(csr.indptr)
        indices = torch.tensor(csr.indices)
        
        ana_info = get_ana(rowptr, indices, in_dim, balance, op_name = 'GCN')

        num_nodes = len(rowptr) - 1
        rowptr = rowptr.to(device)
        indices = indices.to(device)
        degree = 1.0 / torch.tensor(csr.indptr[1:] - csr.indptr[:-1], dtype=torch.float32)
        degree = degree.to(device)

        if (paras != None):
            weight0 = paras['w0'].to(device)
            bias0 = paras['b0'].to(device)
            weight1 = paras['w1'].to(device)
            bias1 = paras['b1'].to(device)
        else:
            weight0 = torch.randn(in_dim, gcn_hidden).to(device)
            bias0 = torch.randn(gcn_hidden).to(device)
            weight1 = torch.randn(gcn_hidden, out_dim).to(device)
            bias1 = torch.randn(out_dim).to(device)

        output0 = torch.zeros(num_nodes, gcn_hidden).to(device)
        output1 = torch.zeros(num_nodes, out_dim).to(device)
        
        perf_time_start(time_label) if detail_test > 0 else 0

        scripted_mm0 = torch.jit.script(jit_mm, example_inputs=[(input_feature, weight0)])
        scripted_mm1 = torch.jit.script(jit_mm, example_inputs=[(output0, weight1)])
        scripted_ot = torch.jit.script(jit_other, example_inputs=[(input_feature, weight0)])
        # print(type(scripted_mm0))  # torch.jit.ScriptFunction
        # # See the compiled graph as Python code
        # print(scripted_mm0.code)

        # print(input_feature)

        # print(input_feature, weight0)

        # perf_time_start("KG MM0")
        # KGGNN.kg_nn_gcn_fused_run(rowptr, indices, input_feature, weight0, output0, ana_info)
        # perf_time_end()

        perf_time_start(time_label + " MM0") if detail_test == 1 else 0
        feat0 = torch.mm(input_feature, weight0)
        #feat0 = scripted_mm0(input_feature, weight0)
        perf_time_end() if detail_test == 1 else 0

        # perf_time_start("KG MM02") if detail_test == 1 else 0
        # feat0 = torch.mm(input_feature, weight0)
        # #feat0 = scripted_mm0(input_feature, weight0)
        # perf_time_end() if detail_test == 1 else 0

        # print(output0)
        #print(feat0)
        # print(output0 - feat0)
        
        perf_time_start(time_label + " AGGR0") if detail_test > 0 else 0
        if (ana_info):
            torch.ops.KGGNN.kg_gcn_run_balance_with_deg(rowptr, indices, feat0, output0, degree, ana_info)
        else:
            torch.ops.KGGNN.kg_gcn_run(rowptr, indices, feat0, output0)
        aggr0_time = perf_time_end() if detail_test > 0 else 0

        #print("Memory bandwidth: {:.2f} GB/s".format(len(csr.indices) * (gcn_hidden + 1) * 4 / 1024 / 1024 / aggr0_time))

        #print(output0)

        #return output0, aggr0_time
        
        perf_time_start(time_label + " OTHER0") if detail_test == 1 else 0
        #output0 = scripted_ot(output0, bias0)
        output0 = F.relu(output0 + bias0)
        perf_time_end() if detail_test == 1 else 0

        perf_time_start(time_label + " MM1") if detail_test == 1 else 0
        feat1 = torch.mm(output0, weight1)
        #feat1 = scripted_mm1(output0, weight1)
        perf_time_end() if detail_test == 1 else 0

        perf_time_start(time_label + " AGGR1") if detail_test > 0 else 0
        if (ana_info):
            #KGGNN.kg_gcn_run_balance_with_deg(rowptr, indices, feat0, output0, degree, ana_info)
            torch.ops.KGGNN.kg_gcn_run_balance_with_deg(rowptr, indices, feat1, output1, degree, ana_info)
        else:
            torch.ops.KGGNN.kg_gcn_run(rowptr, indices, feat1, output1)
        perf_time_end() if detail_test > 0 else 0

        perf_time_start(time_label + " OTHER1") if detail_test == 1 else 0
        output1 = F.log_softmax(output1 + bias1, dim=1)
        perf_time_end() if detail_test == 1 else 0

        kg_time = perf_time_end() if detail_test > 0 else 0

        return output1, kg_time

    elif (name == 'GAT'):
    
        rowptr = torch.tensor(csr.indptr)
        indices = torch.tensor(csr.indices)
        
        ana_info = get_ana(rowptr, indices, in_dim, balance, op_name = 'GAT')

        num_nodes = len(rowptr) - 1
        num_edges = len(indices)
        rowptr = rowptr.to(device)
        indices = indices.to(device)

        if (paras == None):
            weight0 = torch.randn([in_dim, gcn_hidden]).to(device)
            weight1 = torch.randn([gcn_hidden, out_dim]).to(device)
            bias0 = torch.randn([gcn_hidden]).to(device)
            bias1 = torch.randn([out_dim]).to(device)
            att_src0 = torch.randn([1, 1, gcn_hidden]).to(device)
            att_dst0 = torch.randn([1, 1, gcn_hidden]).to(device)
            att_src1 = torch.randn([1, 1, out_dim]).to(device)
            att_dst1 = torch.randn([1, 1, out_dim]).to(device)
        else:
            weight0 = paras['w0'].to(device)
            weight1 = paras['w1'].to(device)
            bias0 = paras['b0'].to(device)
            bias1 = paras['b1'].to(device)
            att_src0 = paras['as0'].to(device)
            att_dst0 = paras['ad0'].to(device)
            att_src1 = paras['as1'].to(device)
            att_dst1 = paras['ad1'].to(device)

        output_feat0 = torch.zeros([num_nodes, gcn_hidden], dtype=torch.float).to(device)
        output_feat1 = torch.zeros([num_nodes, out_dim], dtype=torch.float).to(device)
        weight_lr0 = torch.cat((att_dst0.reshape(-1, 1), att_src0.reshape(-1, 1)), 1)
        weight_lr1 = torch.cat((att_dst1.reshape(-1, 1), att_src1.reshape(-1, 1)), 1)

        # print(att_src0, att_dst0, weight_lr0)

        sum_vec0 = torch.zeros([num_nodes], dtype=torch.float).to(device)
        sum_vec1 = torch.zeros([num_nodes], dtype=torch.float).to(device)
        edge_weight0 = torch.zeros([num_edges], dtype=torch.float).to(device)
        edge_weight1 = torch.zeros([num_edges], dtype=torch.float).to(device)

        # GCN Layer run
        perf_time_start(time_label) if detail_test > 0 else 0

        # GCN Layer 0 (in_dim, gcn_hidden)
        perf_time_start(time_label + " MM0") if detail_test == 1 else 0
        feat0 = torch.mm(input_feature, weight0)
        #att_lr0 = torch.mm(feat0, weight_lr0)
        perf_time_end() if detail_test == 1 else 0

        perf_time_start(time_label + " AGGR0") if detail_test > 0 else 0
        if (ana_info):
            torch.ops.KGGNN.kg_gat_run_balance(rowptr, indices, feat0, output_feat0, weight_lr0, sum_vec0, edge_weight0, 0.2, ana_info)
        aggr0_time = perf_time_end() if detail_test > 0 else 0

        # print("edge_weight0", edge_weight0)
        # print("sum_vec0", sum_vec0)
        # print("res", output_feat0)

        perf_time_start(time_label + " OTHER0") if detail_test == 1 else 0
        output_feat0 = output_feat0 + bias0
        #print("output_feat0", output_feat0)
        perf_time_end() if detail_test == 1 else 0

        # print("feat0", feat0)
        # print("att_lr0", att_lr0)
        # print("max_vec", max_vec0)

        # print(feat0.size(), output_feat0.size())

        perf_time_start(time_label + " MM1") if detail_test == 1 else 0
        feat1 = torch.mm(output_feat0, weight1)
        #att_lr1 = torch.mm(feat1, weight_lr1)
        perf_time_end() if detail_test == 1 else 0
        
        perf_time_start(time_label + " AGGR1") if detail_test > 0 else 0
        if (ana_info):
            torch.ops.KGGNN.kg_gat_run_balance(rowptr, indices, feat1, output_feat1, weight_lr1, sum_vec1, edge_weight1, 0.2, ana_info)
        aggr0_time = perf_time_end() if detail_test > 0 else 0

        perf_time_start(time_label + " OTHER1") if detail_test == 1 else 0
        output_feat1 = output_feat1 + bias1
        perf_time_end() if detail_test == 1 else 0

        kg_time = perf_time_end() if detail_test > 0 else 0

        return output_feat1, kg_time, aggr0_time

    elif (name == 'GIN'):
        
        rowptr = torch.tensor(csr.indptr)
        indices = torch.tensor(csr.indices)
        
        ana_info = get_ana(rowptr, indices, in_dim, balance, op_name = 'GCN')

        num_nodes = len(rowptr) - 1
        rowptr = rowptr.to(device)
        indices = indices.to(device)
        degree = 1.0 / torch.tensor(csr.indptr[1:] - csr.indptr[:-1], dtype=torch.float32)
        degree = degree.to(device)

        if (paras != None):
            weight0 = paras['w0'].to(device)
            bias0 = paras['b0'].to(device)
            weight1 = paras['w1'].to(device)
            bias1 = paras['b1'].to(device)
        else:
            weight0 = torch.randn(in_dim, gcn_hidden).to(device)
            bias0 = torch.randn(gcn_hidden).to(device)
            weight1 = torch.randn(gcn_hidden, out_dim).to(device)
            bias1 = torch.randn(out_dim).to(device)
        
        aggr0 = torch.zeros(num_nodes, in_dim).to(device)
        aggr1 = torch.zeros(num_nodes, gcn_hidden).to(device)

        eps = 0.0

        perf_time_start(time_label) if detail_test > 0 else 0

        perf_time_start(time_label + " AGGR0") if detail_test > 0 else 0
        if (ana_info):
            torch.ops.KGGNN.kg_gin_run_balance(rowptr, indices, input_feature, aggr0, eps, ana_info)
        aggr0_time = perf_time_end() if detail_test > 0 else 0

        perf_time_start(time_label + " MM0") if detail_test == 1 else 0
        output0 = torch.mm(aggr0, weight0)
        perf_time_end() if detail_test == 1 else 0

        perf_time_start(time_label + " OTHER0") if detail_test == 1 else 0
        output0 = output0 + bias0
        output0 = F.relu(output0)
        perf_time_end() if detail_test == 1 else 0

        perf_time_start(time_label + " AGGR1") if detail_test > 0 else 0
        if (ana_info):
            torch.ops.KGGNN.kg_gin_run_balance(rowptr, indices, output0, aggr1, eps, ana_info)
        aggr1_time = perf_time_end() if detail_test > 0 else 0

        perf_time_start(time_label + " MM1") if detail_test == 1 else 0
        output1 = torch.mm(aggr1, weight1)
        perf_time_end() if detail_test == 1 else 0

        perf_time_start(time_label + " OTHER1") if detail_test == 1 else 0
        output1 = output1 + bias1
        output1 = F.log_softmax(output1, dim = 1)
        perf_time_end() if detail_test == 1 else 0

        kg_time = perf_time_end() if detail_test > 0 else 0

        return output1, kg_time

    else:
        print("Not implemented!")
        return None, None