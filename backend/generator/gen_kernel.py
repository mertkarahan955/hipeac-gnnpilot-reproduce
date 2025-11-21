from data_def import data_list, dim_list
from gen_func import generate_head, generate_body
from gen_func_fused import generate_fused_head, generate_fused_body, generate_hetero_plus_dynamic_kernels
from utils import *

def generate_kernel(fop_id, fop, f, kid):
    # check for
    if (fop.launch_setup == 'hetero'):        
        # We now let all fused kernels use hetero+
        fop.launch_setup = 'hetero+'

        # for i, (gnn_op, parallel) in enumerate(zip(fop.ops[:-1], fop.parallel[:-1])):
        # for op in fop.ops[:-1]:
        #     if (op.name == "gather"):
        #         fop.launch_setup = 'hetero+'

    # single op kernel
    if (len(fop.ops) == 1):
        if (fop.ops[0].name == "linear"):
            pass
        else:
            f.write("__global__ void k_{:d}_fop_{:d}(".format(kid, fop_id))
            f.write("int numnodes, int numedges, int *RowPtr, int *ColIdx, ")
            for dim_term in dim_list:
                f.write("int {}, ".format(dim_term))
            f.write("row_panel* info_list, int *info_n, row_panel* ep_list, int *ep_n, neighbor_group *ng_list, int *ng_n")

            for var_name in fop.var:
                var_info = data_list[var_name]
                f.write(", {}".format(var_info[2]))
                if (not (var_info[0] == 'c' and var_info[1] == 1)):
                    f.write("*")
                f.write(" {}".format(var_name))

            f.write(") {\n")

            generate_head(fop.parallel[0], f)
            generate_body(fop.ops[0], fop.parallel[0], f)

            f.write("}\n")
    # multiple op fused kernel
    elif (len(fop.ops) > 1):
        if (fop.ops[0].name == "linear"):
            gen_error("linear op fused")
        else:
            if ("hetero" in fop.launch_setup):
                generate_hetero_plus_dynamic_kernels('callee', fop, f, fop_id, kid)

            f.write("__global__ void k_{:d}_fop_{:d}(".format(kid, fop_id))
            f.write("int numnodes, int numedges, int *RowPtr, int *ColIdx, ")
            for dim_term in dim_list:
                f.write("int {}, ".format(dim_term))
            f.write("row_panel* info_list, int *info_n, row_panel* ep_list, int *ep_n, neighbor_group *ng_list, int *ng_n")
            #f.write("int numnodes, int numedges, int *RowPtr, int *ColIdx, int featlen, row_panel* info_list, int *info_n")

            for var_name in fop.var:
                var_info = data_list[var_name]
                f.write(", {}".format(var_info[2]))
                if (not (var_info[0] == 'c' and var_info[1] == 1)):
                    f.write("*")
                f.write(" {}".format(var_name))

            f.write(") {\n")

            if ("hetero" in fop.launch_setup):
                generate_fused_head(fop, f)
                generate_fused_body(fop, f, fop_id, kid)
            else:
                generate_head(fop.launch_setup, f)
                for gnn_op in fop.ops:
                    f.write("{ ")
                    generate_body(gnn_op, fop.launch_setup, f)
                    f.write("}\n")
                    if (fop.launch_setup == 'e'):
                        f.write("__syncthreads();\n")
            # else:
            #     gen_error("Not implemented")

            f.write("}\n")
    else:
        gen_error("Not implemented")
