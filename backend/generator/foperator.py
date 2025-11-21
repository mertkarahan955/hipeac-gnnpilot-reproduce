from utils import *
from graph_construct import Go

def is_dimwise(gnn_op):
    return gnn_op.var_output.info[1][0:2] == 'fd' and gnn_op.var_output.ref[1] == ':'

class foperator:
    ops = []
    var = []
    parallel = []
    launch_setup = ''

    def __init__(self):
        self.ops = []
        self.var = []
        self.parallel = []
        self.launch_setup = ''

    def add_op(self, gnn_op, gs, ss, ds):
        self.ops.append(gnn_op)
        
        # if (gnn_op.name == 'linear'):
        #     self.var.append(gnn_op.linear_in[0])
        #     if (not gnn_op.linear_in[1] in self.var):
        #         self.var.append(gnn_op.linear_in[1])
        #     if (not gnn_op.output in self.var):
        #         self.var.append(gnn_op.output)

        for var_input in gnn_op.expr.var_input:
            if (not var_input.name in self.var):
                self.var.append(var_input.name)
        if (not gnn_op.var_output.name in self.var):
            self.var.append(gnn_op.var_output.name)

        op_parallel = ''
        if (gnn_op.name == 'gather'):
            if (gnn_op.var_output.info[1][0:2] == 'fd' and gnn_op.var_output.ref[1] == ':'):
                op_parallel = ds
            else:
                op_parallel = gs
        elif (gnn_op.name == 'scatter'):
            op_parallel = ss
        elif (gnn_op.name == 'linear'):
            pass
        else:
            gen_warning("Not implemented!")
        
        self.parallel.append(op_parallel)
        
        if (self.launch_setup == ''):
            self.launch_setup = op_parallel
        elif (not self.launch_setup == op_parallel):
            self.launch_setup = 'hetero'
        
        # fused 'e' operators are special
        if (self.launch_setup == 'e' and len(self.ops) > 1):
            self.launch_setup = 'hetero'

def show_fop_list(fp_list):
    for fid, fop in enumerate(fp_list):
        print("fop", fid, end=' ')
        for ops in fop.ops:
            print(ops.name, end=' ')
        print(fop.var, fop.parallel, fop.launch_setup)

def graph_traverse(Go_tmp, node, traverse_list, gnn_op_list, fs):
    for edge in Go_tmp.out_edges(node):
        neighbor = edge[1]
        if (not neighbor in traverse_list and not(is_dimwise(gnn_op_list[neighbor]) and fs == 'ne')):
            flag = True
            # very important: dependent reference must be row-wise
            for var in gnn_op_list[neighbor].expr.var_input:
                if (var.ref[0] == 'j'):
                    for old_op in traverse_list:
                        if (gnn_op_list[old_op].var_output.name == var.name and var.ref[0] == 'j'):
                            flag = False

            # flag = False
            for neighbor_neighbor in Go_tmp.in_edges(neighbor):
                if (not neighbor_neighbor[0] in traverse_list):
                    flag = False
                    break
            if (flag):
                traverse_list.append(neighbor)


def get_fop_list(gnn_op_list, gs, ss, ds, fs, kid):
    fop_list = []
    Go_tmp = Go.copy()
    while (len(Go_tmp.nodes) > 0):
        if (fs == 'no'):
            wavefront_node_list = []
            for node in Go_tmp.nodes:
                if (Go_tmp.in_degree[node] == 0):
                    wavefront_node_list.append(node)

            for node in wavefront_node_list:
                fop = foperator()
                fop.add_op(gnn_op_list[node], gs, ss, ds)
                fop_list.append(fop)
                Go_tmp.remove_node(node)
        elif (fs == 'ne' or fs == 'all' or fs == 'all_d'):
            traverse_list = []

            old_len = 0
            while (len(Go_tmp.nodes) != old_len):
                old_len = len(Go_tmp.nodes)

                wavefront_node_list = []
                for node in Go_tmp.nodes:
                    if (Go_tmp.in_degree[node] == 0):
                        wavefront_node_list.append(node)

                for node in wavefront_node_list:
                    gnn_op = gnn_op_list[node]
                    if (gnn_op.name == "linear" or (is_dimwise(gnn_op) and fs == 'ne')):
                        fop = foperator()
                        fop.add_op(gnn_op_list[node], gs, ss, ds)
                        fop_list.append(fop)
                        Go_tmp.remove_node(node)

            traverse_list = []
            for node in Go_tmp.nodes:
                if (Go_tmp.in_degree[node] == 0):
                    if (not(is_dimwise(gnn_op) and fs == 'ne')):
                        traverse_list.append(node)

            old_len = 0
            while (old_len != len(traverse_list)):
                old_len = len(traverse_list)
                for i in range(len(traverse_list)):
                    graph_traverse(Go_tmp, traverse_list[i], traverse_list, gnn_op_list, fs)

            fop = foperator()
            for node in traverse_list:
                fop.add_op(gnn_op_list[node], gs, ss, ds)
                Go_tmp.remove_node(node)
            if (fop.launch_setup == 'hetero' and fs =='all_d'):
                fop.launch_setup = 'hetero_d'
            if (len(fop.ops) > 0):
                fop_list.append(fop)

        else:
            inter_error("Not implemented")

    print("fop list {:d}".format(kid))
    show_fop_list(fop_list)
    return fop_list

# def get_fop_list2(gnn_op_list, op_level_list, gs, ss, ds, fs):
#     fop_list = []
#     if (fs == 'no'):
#         for op_level in op_level_list:
#             for op_id in op_level:
#                 fop = foperator()
#                 fop.add_op(gnn_op_list[op_id], gs, ss, ds)
#                 fop_list.append(fop)
#     elif (fs == 'ne'):
#         # intra-level fusion
#         for op_level in op_level_list:
#             # linear is seperate
#             for op_id in op_level:
#                 if (gnn_op_list[op_id].name == "linear"):
#                     fop = foperator()
#                     fop.add_op(gnn_op_list[op_id], gs, ss, ds)
#     elif (fs == 'all'):
#         gen_warning("Not implemented!")
#     else:
#         gen_warning("Not implemented!")
#     # print(fop_list)
#     return fop_list
