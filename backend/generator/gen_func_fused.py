from utils import *
#from gen_func import generate_head, generate_body, generate_gather_one_row, generate_scatter_one_row
from gen_func import *
from gen_expr import generate_expr, generate_variable
from data_def import data_list, dim_list

def generate_hetero_head(fop, f):
    f.write("int lane_id = threadIdx.x % WARP_SIZE;\n")
    f.write("if (blockIdx.x >= *info_n) return;\n")
    f.write("row_panel info_tmp = info_list[blockIdx.x];\n")

    f.write("int local_wid = threadIdx.x / WARP_SIZE;\n")
    f.write("int global_tid = blockIdx.x * blockDim.x + threadIdx.x;\n")
    f.write("int global_wid = global_tid / WARP_SIZE;\n")

    if (len(list(filter(lambda x: x == "n", fop.parallel)))):
        f.write("int row_tmp_n_st = info_tmp.row_st + threadIdx.x;\n")
        f.write("int row_tmp_n_ed = info_tmp.row_ed;\n")
    
    if (len(list(filter(lambda x: x == "ne" or x == "nd", fop.parallel)))):
        f.write("int row_tmp_ne_st = info_tmp.row_st + local_wid;\n")
        f.write("int row_tmp_ne_ed = info_tmp.row_ed;\n")
        
    if (len(list(filter(lambda x: x == "e" or x == "ed", fop.parallel)))):
        f.write("int rows = info_tmp.row_ed - info_tmp.row_st;\n" \
        "int col_st = RowPtr[info_tmp.row_st];\n" \
        "int col_ed = RowPtr[info_tmp.row_ed];\n" \
        "int nnzs = col_ed - col_st;\n")

        if (len(list(filter(lambda x: x == "ed", fop.parallel)))):
            f.write("int nnz_per_warp = (nnzs + HE_BLOCK_WARP_SIZE - 1) / HE_BLOCK_WARP_SIZE;\n" \
            "int warp_col_st = col_st + nnz_per_warp * local_wid;\n" \
            "int warp_col_ed = (warp_col_st + nnz_per_warp < col_ed)? warp_col_st + nnz_per_warp: col_ed;\n" \
            "int warp_row_st = info_tmp.row_st, warp_row_ed;\n" \
            "while (warp_row_st < info_tmp.row_ed - 1 && warp_col_st > RowPtr[warp_row_st + 1]) {\n" \
            "warp_row_st++;\n" \
            "}\n" \
            "warp_row_ed = warp_row_st;\n" \
            "while (warp_row_ed < info_tmp.row_ed && warp_col_ed > RowPtr[warp_row_ed]) {\n" \
            "warp_row_ed++;\n" \
            "}\n")

def generate_fused_head(fop, f):
    parallel = fop.launch_setup

    if ("hetero" in parallel):
        generate_hetero_head(fop, f)
    else:
        generate_head(parallel, f)

def generate_hetero_gather_n_ne_op(gnn_op, parallel, f):
    if (parallel == "n"):
        f.write("for (int row_tmp = row_tmp_n_st; row_tmp < row_tmp_n_ed; row_tmp += HE_BLOCK_SIZE) {\n")
    elif (parallel == "ne"):
        f.write("for (int row_tmp = row_tmp_ne_st; row_tmp < row_tmp_ne_ed; row_tmp += HE_BLOCK_WARP_SIZE) {\n")

    if (gnn_op.op == 'SUM' or gnn_op.op == 'AVG'):
        f.write("{} res_tmp = 0;\n".format(gnn_op.var_output.info[2]))
    elif (gnn_op.op == 'MUL'):
        f.write("{} res_tmp = 1;\n".format(gnn_op.var_output.info[2]))
    elif (gnn_op.op == 'MAX'):
        # Maybe need a better initilization
        f.write("{} res_tmp = -99999;\n".format(gnn_op.var_output.info[2]))
    else:
        gen_error("GNN op not implemented")

    if (parallel == "n"):
        f.write("for (int nnz_tmp = RowPtr[row_tmp]; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp++) {\n")
    elif (parallel == "ne"):
        f.write("for (int nnz_tmp = RowPtr[row_tmp] + lane_id; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp+=WARP_SIZE) {\n")
    
    f.write("int col_tmp = ColIdx[nnz_tmp];\n")

    if (gnn_op.op == 'SUM' or gnn_op.op == 'AVG'):
        f.write("res_tmp += {};".format(generate_expr(gnn_op.expr, f)))
    elif (gnn_op.op == 'MUL'):
        f.write("res_tmp *= {};".format(generate_expr(gnn_op.expr, f)))
    elif (gnn_op.op == 'MAX'):
        tmp_expr = generate_expr(gnn_op.expr, f)
        f.write("res_tmp = ({} > res_tmp)? {}: res_tmp;".format(tmp_expr, tmp_expr))
    else:
        gen_error("GNN op not implemented")

    f.write("\n}\n")

    if (len(gnn_op.expr.out_expr) > 0):
        f.write("res_tmp = res_tmp {};\n".format(generate_expr(gnn_op.expr.out_expr, f)))

    if (gnn_op.op == 'AVG'):
        f.write("res_tmp = res_tmp / (RowPtr[row_tmp + 1] - RowPtr[row_tmp]);\n")

    if (parallel == "n"):
        f.write("{} = res_tmp;\n".format(generate_variable(gnn_op.var_output)))
    elif (parallel == "ne"):
        f.write("for (int offset_tmp = WARP_SIZE / 2; offset_tmp > 0; offset_tmp /= 2) {\n")
        f.write("{} comm_tmp = __shfl_down_sync(0xffffffff, res_tmp, offset_tmp);\n".format(gnn_op.var_output.info[2]))
        if (gnn_op.op == 'SUM' or gnn_op.op == 'AVG'):
            f.write("res_tmp += comm_tmp;\n")
        elif (gnn_op.op == 'MAX'):
            f.write("res_tmp = (comm_tmp > res_tmp)? comm_tmp: res_tmp;\n")
        f.write("}\n")
        f.write("if (lane_id == 0) {} = res_tmp;\n".format(generate_variable(gnn_op.var_output)))
    else:
        gen_error("GNN op not implemented")
    
    f.write("}\n")

def generate_hetero_gather_e_op(gnn_op, f):
    f.write("using WarpReduce = cub::WarpReduce<{}>;\n".format(gnn_op.var_output.info[2]))
    f.write("__shared__ typename WarpReduce::TempStorage temp_storage[HE_BLOCK_WARP_SIZE];\n" \
    "int row_tmp = info_tmp.row_st;\n" \
    "for (int col_iter = col_st; col_iter < col_ed; col_iter += blockDim.x) {\n" \
    "int nnz_tmp = col_iter + threadIdx.x;\n" \
    "int col_tmp = ColIdx[nnz_tmp];\n" \
    "int new_line_flag = 0; \n")

    f.write("{} res_tmp = 0;\n".format(gnn_op.var_output.info[2]))
    f.write("if (nnz_tmp < col_ed) {\n" \
    "while (RowPtr[row_tmp + 1] <= nnz_tmp) row_tmp++;\n" \
    "new_line_flag = (RowPtr[row_tmp] == nnz_tmp || lane_id == 0);\n")
    # needs correction
    if (gnn_op.op == "AVG"):
        f.write("res_tmp = {} / (RowPtr[row_tmp + 1] - RowPtr[row_tmp]);\n".format(generate_expr(gnn_op.expr, f)))
    else:
        f.write("res_tmp = {};\n".format(generate_expr(gnn_op.expr, f)))

    if (len(gnn_op.expr.out_expr) > 0):
        f.write("res_tmp = res_tmp {};\n".format(generate_expr(gnn_op.expr.out_expr, f)))

    f.write("}\n")

    if (gnn_op.op == 'MAX'):
        f.write("else\nres_tmp = -99999;\n")

    if (len(gnn_op.expr.out_expr) > 0):
        f.write("res_tmp = res_tmp {};\n".format(generate_expr(gnn_op.expr.out_expr, f)))

    if (gnn_op.op == 'SUM' or gnn_op.op == 'AVG'):
        cub_op = "cub::Sum()"
    elif (gnn_op.op == 'MAX'):
        cub_op = "cub::Max()"
    else:
        gen_error("GNN op not implemented")

    f.write("{} reduce_tmp = WarpReduce(temp_storage[local_wid]).HeadSegmentedReduce(" \
    "res_tmp, new_line_flag, {});\n".format(gnn_op.var_output.info[2], cub_op))
    f.write("if (new_line_flag) ")
    if (gnn_op.op == 'SUM' or gnn_op.op == 'AVG'):
        f.write("atomicAdd(&{}, reduce_tmp);\n".format(generate_variable(gnn_op.var_output)))
    # using atomicMax may have problems when max value < 0
    elif (gnn_op.op == 'MAX'):
        f.write("atomicMax")
        if (gnn_op.var_output.info[2] == 'float'):
            f.write("_float")
        f.write("(&{}, reduce_tmp);\n".format(generate_variable(gnn_op.var_output)))
    f.write("__syncwarp();\n}\n")

def generate_hetero_scatter_op(gnn_op, parallel, f):
    if (parallel == "n"):
        f.write("for (int row_tmp = row_tmp_n_st; row_tmp < row_tmp_n_ed; row_tmp += HE_BLOCK_SIZE) {\n")
        f.write("for (int nnz_tmp = RowPtr[row_tmp]; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp++) {\n")
    elif (parallel == "ne"):
        f.write("for (int row_tmp = row_tmp_ne_st; row_tmp < row_tmp_ne_ed; row_tmp += HE_BLOCK_WARP_SIZE) {\n")
        f.write("for (int nnz_tmp = RowPtr[row_tmp] + lane_id; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp+=WARP_SIZE) {\n")
    elif (parallel == "e"):
        f.write("{ int row_tmp = info_tmp.row_st;\n" \
        "for (int nnz_tmp = col_st + threadIdx.x; nnz_tmp < col_ed; nnz_tmp += HE_BLOCK_SIZE) {\n" \
        "while (RowPtr[row_tmp + 1] <= nnz_tmp) row_tmp++;\n")
    else:
        gen_error("Wrong parallel setup")
    f.write("int col_tmp = ColIdx[nnz_tmp];\n")
    f.write("{} = {};".format(generate_variable(gnn_op.var_output), generate_expr(gnn_op.expr, f)))
    f.write("\n}\n}\n")

def generate_hetero_gather_nd_op(gnn_op, f):
    f.write("for (int row_tmp = row_tmp_ne_st; row_tmp < row_tmp_ne_ed; row_tmp += HE_BLOCK_WARP_SIZE) {\n")
    f.write("for (int k_tmp = lane_id; k_tmp < featlen; k_tmp += WARP_SIZE) {\n")
    if (gnn_op.op == 'SUM' or gnn_op.op == 'AVG'):
        f.write("{} res_tmp = 0;\n".format(gnn_op.var_output.info[2]))
    elif (gnn_op.op == 'MUL'):
        f.write("{} res_tmp = 1;\n".format(gnn_op.var_output.info[2]))
    elif (gnn_op.op == 'MAX'):
        # Maybe need a better initilization
        f.write("{} res_tmp = -99999;\n".format(gnn_op.var_output.info[2]))
    else:
        gen_error("GNN op not implemented")
    f.write("for (int nnz_tmp = RowPtr[row_tmp]; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp++) {\n")
    f.write("int col_tmp = ColIdx[nnz_tmp];\n")

    if (gnn_op.op == 'SUM' or gnn_op.op == 'AVG'):
        f.write("res_tmp += {};".format(generate_expr(gnn_op.expr, f)))
    elif (gnn_op.op == 'MUL'):
        f.write("res_tmp *= {};".format(generate_expr(gnn_op.expr, f)))
    elif (gnn_op.op == 'MAX'):
        tmp_expr = generate_expr(gnn_op.expr, f)
        f.write("res_tmp = ({} > res_tmp)? {}: res_tmp;".format(tmp_expr, tmp_expr))
    else:
        gen_error("GNN op not implemented")

    f.write("\n}\n")
    if (len(gnn_op.expr.out_expr) > 0):
        f.write("res_tmp = res_tmp {};\n".format(generate_expr(gnn_op.expr.out_expr, f)))

    if (gnn_op.op == 'AVG'):
        f.write("res_tmp = res_tmp / (RowPtr[row_tmp + 1] - RowPtr[row_tmp]);\n")

    f.write("{} = res_tmp;\n".format(generate_variable(gnn_op.var_output)))
    f.write("}\n}\n")

def generate_hetero_gather_ed_op(gnn_op, f):
    
    f.write("if (warp_col_ed <= col_ed) {\n" \
    "for (int row_tmp = warp_row_st; row_tmp < warp_row_ed; row_tmp++) {\n")
    
    f.write("int current_col_st = RowPtr[row_tmp];\n" \
    "int current_col_ed = RowPtr[row_tmp + 1];\n" \
    "if (row_tmp == warp_row_st) current_col_st = warp_col_st;\n" \
    "if (row_tmp == warp_row_ed - 1) current_col_ed = warp_col_ed;\n")

    f.write("for (int k_tmp = lane_id; k_tmp < featlen; k_tmp += WARP_SIZE) {\n")

    if (gnn_op.op == 'SUM' or gnn_op.op == 'AVG'):
        f.write("{} res_tmp = 0;\n".format(gnn_op.var_output.info[2]))
    elif (gnn_op.op == 'MUL'):
        f.write("{} res_tmp = 1;\n".format(gnn_op.var_output.info[2]))
    elif (gnn_op.op == 'MAX'):
        # Maybe need a better initilization
        f.write("{} res_tmp = -99999;\n".format(gnn_op.var_output.info[2]))
    else:
        gen_error("GNN op not implemented")

    f.write("for (int nnz_tmp = current_col_st; nnz_tmp < current_col_ed; nnz_tmp++) {\n")
    f.write("int col_tmp = ColIdx[nnz_tmp];\n")

    if (gnn_op.op == 'SUM' or gnn_op.op == 'AVG'):
        f.write("res_tmp += {};".format(generate_expr(gnn_op.expr, f)))
    elif (gnn_op.op == 'MUL'):
        f.write("res_tmp *= {};".format(generate_expr(gnn_op.expr, f)))
    elif (gnn_op.op == 'MAX'):
        tmp_expr = generate_expr(gnn_op.expr, f)
        f.write("res_tmp = ({} > res_tmp)? {}: res_tmp;".format(tmp_expr, tmp_expr))
    else:
        gen_error("GNN op not implemented")
    f.write("\n}\n")

    if (len(gnn_op.expr.out_expr) > 0):
        f.write("res_tmp = res_tmp {};\n".format(generate_expr(gnn_op.expr.out_expr, f)))

    if (gnn_op.op == 'AVG'):
        f.write("res_tmp = res_tmp / (RowPtr[row_tmp + 1] - RowPtr[row_tmp]);\n")

    if (gnn_op.op == 'SUM' or gnn_op.op == 'AVG'):
        f.write("atomicAdd(&{}, res_tmp);\n".format(generate_variable(gnn_op.var_output)))
    # using atomicMax may have problems when max value < 0
    elif (gnn_op.op == 'MAX'):
        f.write("atomicMax")
        if (gnn_op.var_output.info[2] == 'float'):
            f.write("_float")
        f.write("(&{}, res_tmp);\n".format(generate_variable(gnn_op.var_output)))

    f.write("}}}\n")

def generate_gather_one_row_dynamic(gnn_op, parallel, f):

    if (parallel == 'nd' or parallel == 'ed' or parallel == 'ngd'):
        f.write("for (int k_tmp = lane_id; k_tmp < featlen; k_tmp += WARP_SIZE) {\n")

        res_tmp_init(gnn_op, f)
        
        f.write("for (int nnz_tmp = idx_st; nnz_tmp < idx_ed; nnz_tmp++) {\n")
        f.write("int col_tmp = ColIdx[nnz_tmp];\n")

        res_tmp_gather(gnn_op, f)
        
        f.write("\n}\n")

        res_tmp_writeback_dimwise(gnn_op, f)
        
        f.write("}\n")

    else:
        res_tmp_init(gnn_op, f)

        f.write("for (int nnz_tmp = idx_st + lane_id; nnz_tmp < idx_ed; nnz_tmp+=WARP_SIZE) {\n")
        f.write("int col_tmp = ColIdx[nnz_tmp];\n")

        res_tmp_gather(gnn_op, f)

        f.write("}\n")

        res_tmp_writeback(gnn_op, "e", f)

def generate_scatter_one_row_dynamic(gnn_op, f):
    f.write("for (int nnz_tmp = idx_st + lane_id; nnz_tmp < idx_ed; nnz_tmp += WARP_SIZE) {\n")
    f.write("int col_tmp = ColIdx[nnz_tmp];\n")
    f.write("{} = {};".format(generate_variable(gnn_op.var_output), generate_expr(gnn_op.expr, f)))
    f.write("}\n")

def generate_hetero_plus_dynamic_kernels(task, fop, f, fop_id, kid):
    op_buffer = []
    dop_buffer = []
    var_buffer = []
    var_list = []
    for i, (gnn_op, parallel) in enumerate(zip(fop.ops, fop.parallel)):
        # gather operator is not the last one
        if (fop.launch_setup == "hetero_d"):
            if (not (gnn_op.name == "gather" and "d" in parallel)):
                dop_buffer.append([(gnn_op, "one row")])
                var_list.append([])
            else:
                dop_buffer.append([(gnn_op, parallel)])
                for var in gnn_op.expr.var_input:
                    if (not var.name in var_buffer):
                        var_buffer.append(var.name)
                if (not gnn_op.var_output.name in var_buffer):
                    var_buffer.append(gnn_op.var_output.name)
                var_list.append([var for var in var_buffer])
                var_buffer.clear()
        else:
            if (gnn_op.name == "gather" and i != len(fop.ops) - 1):
                if (len(op_buffer) > 0):
                    #print("new dop", [op[0].expr.expr for op in op_buffer])
                    dop_buffer.append([op for op in op_buffer])
                    var_list.append([var for var in var_buffer])
                dop_buffer.append([(gnn_op, "one row")])
                var_list.append([])
                op_buffer.clear()
                var_buffer.clear()
            else:
                op_buffer.append((gnn_op, parallel))
                for var in gnn_op.expr.var_input:
                    if (not var.name in var_buffer):
                        var_buffer.append(var.name)
                if (not gnn_op.var_output.name in var_buffer):
                    var_buffer.append(gnn_op.var_output.name)

    if (len(op_buffer) > 0):
        dop_buffer.append(op_buffer)
        var_list.append(var_buffer)

    # for i, dop in enumerate(dop_buffer):
    #     print("dop:", i, [op[0].expr.expr for op in dop])
    
    if (task == 'caller'):
        for i, (dop, tmp_varlist) in enumerate(zip(dop_buffer, var_list)):
            if (fop.launch_setup == "hetero_d"):
                for j, dop_tmp in enumerate(dop):
                    if (dop_tmp[1] == "one row"):
                        gnn_op = dop_tmp[0]
                        if (gnn_op.name == "gather"):
                            f.write("{\n")
                            res_tmp_init(gnn_op, f)
                            f.write("for (int nnz_tmp = RowPtr[row_tmp] + threadIdx.x; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp+=HE_BLOCK_SIZE) {\n")
                            f.write("int col_tmp = ColIdx[nnz_tmp];\n")
                            res_tmp_gather(gnn_op, f)
                            f.write("}\n")
                            res_tmp_writeback(gnn_op, "e", f)
                            f.write("}__syncthreads();\n")
                        elif (gnn_op.name == "scatter"):
                            f.write("for (int nnz_tmp = RowPtr[row_tmp] + threadIdx.x; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp += HE_BLOCK_SIZE) {\n")
                            f.write("int col_tmp = ColIdx[nnz_tmp];\n")
                            f.write("{} = {};".format(generate_variable(gnn_op.var_output), generate_expr(gnn_op.expr, f)))
                            f.write("}__syncthreads();\n")
                    else:
                        f.write("if (threadIdx.x == 0) ")
                        # print(kid, fop_id, i)
                        f.write("k{:d}_fop{:d}_dp{:d}<<<dynamic_blocks, DP_BLOCK_SIZE>>>".format(kid, fop_id, i))
                        f.write("(RowPtr, ColIdx, row_tmp")
                        for dim_term in dim_list:
                            f.write(", {}".format(dim_term))
                        for var in tmp_varlist:
                            f.write(", {}".format(var))
                        f.write(");\n")
            else:
                gnn_op = dop[0][0]
                if (gnn_op.name == "gather" and dop[0][1] == "one row"):
                    f.write("{\n")
                    res_tmp_init(gnn_op, f)
                    f.write("for (int nnz_tmp = RowPtr[row_tmp] + threadIdx.x; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp+=HE_BLOCK_SIZE) {\n")
                    f.write("int col_tmp = ColIdx[nnz_tmp];\n")
                    res_tmp_gather(gnn_op, f)
                    f.write("}\n")
                    res_tmp_writeback(gnn_op, "e", f)
                    f.write("}\n")
                elif (gnn_op.name == "scatter" and dop[0][1] == "one row"):
                    f.write("for (int nnz_tmp = RowPtr[row_tmp] + threadIdx.x; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp += HE_BLOCK_SIZE) {\n")
                    f.write("int col_tmp = ColIdx[nnz_tmp];\n")
                    f.write("{} = {};".format(generate_variable(gnn_op.var_output), generate_expr(gnn_op.expr, f)))
                    f.write("}\n")
                else:
                    f.write("if (threadIdx.x == 0) ")
                    f.write("k{:d}_fop{:d}_dp{:d}<<<dynamic_blocks, DP_BLOCK_SIZE>>>".format(kid, fop_id, i))
                    f.write("(RowPtr, ColIdx, row_tmp")
                    for dim_term in dim_list:
                        f.write(", {}".format(dim_term))
                    for var in tmp_varlist:
                        f.write(", {}".format(var))
                    f.write(");\n")
    elif (task == 'callee'):
        # print(dop_buffer, fop_id, kid)
        for i, (dop, tmp_varlist) in enumerate(zip(dop_buffer, var_list)):
            # print(dop[0][0].expr.expr, tmp_varlist)
            if (not dop[0][1] == "one row"):
                f.write("__global__ void k{:d}_fop{:d}_dp{:d}(int *RowPtr, int *ColIdx, int row_tmp".format(kid, fop_id, i))
                for dim_term in dim_list:
                    f.write(", int {}".format(dim_term))
                for var_name in tmp_varlist:
                    var_info = data_list[var_name]
                    f.write(", {}".format(var_info[2]))
                    if (not (var_info[0] == 'c' and var_info[1] == 1)):
                        f.write("*")
                    f.write(" {}".format(var_name))
                f.write(") {\n")

                f.write("int lane_id = threadIdx.x % WARP_SIZE;\n")
                f.write("int idx_st = RowPtr[row_tmp] + blockIdx.x * DP_NNZ_PER_BLOCK + threadIdx.x / WARP_SIZE * DP_NNZ_PER_WARP;\n")
                f.write("int idx_ed = idx_st + DP_NNZ_PER_WARP;\n")
                f.write("if (idx_ed > RowPtr[row_tmp + 1]) idx_ed = RowPtr[row_tmp + 1];\n")

                # generate one row body
                for dop_tmp in dop:
                    gnn_op = dop_tmp[0]
                    parallel = dop_tmp[1]
                    if (gnn_op.name == "gather"):
                        generate_gather_one_row_dynamic(gnn_op, parallel, f)
                    if (gnn_op.name == "scatter"):
                        generate_scatter_one_row_dynamic(gnn_op, f)

                f.write("}\n")
    else:
        gen_error("Wrong dynamic kernel task!")

def generate_fused_body(fop, f, fop_id, kid, long_dynamic=True):
    #if (fop.launch_setup == "hetero" or fop.launch_setup == "hetero+"):
    # All fused fop use "hetero+"
    # fop.launch_setup = "hetero+"
    if ("hetero" in fop.launch_setup):
        if (long_dynamic):
            f.write("if (info_tmp.col_st == -1) {\n")
        for i, (gnn_op, parallel) in enumerate(zip(fop.ops, fop.parallel)):
            f.write("{ ")
            # if (fop.launch_setup == "hetero"):
            if (parallel == "n"):
                if (gnn_op.name == "gather"):
                    generate_hetero_gather_n_ne_op(gnn_op, parallel, f)
                elif (gnn_op.name == "scatter"):
                    generate_hetero_scatter_op(gnn_op, parallel, f)
            if (parallel == "ne"):
                if (gnn_op.name == "gather"):
                    generate_hetero_gather_n_ne_op(gnn_op, parallel, f)
                elif (gnn_op.name == "scatter"):
                    generate_hetero_scatter_op(gnn_op, parallel, f)
            if (parallel == "e"):
                if (gnn_op.name == "scatter"):
                    generate_hetero_scatter_op(gnn_op, parallel, f)
                elif (gnn_op.name == "gather"):
                    generate_hetero_gather_e_op(gnn_op, f)
            if (parallel == "nd" or parallel == "ngd"):
                if (gnn_op.name == "gather"):
                    generate_hetero_gather_nd_op(gnn_op, f)
            if (parallel == "ed"):
                if (gnn_op.name == "gather"):
                    generate_hetero_gather_ed_op(gnn_op, f)
            # else:
            #     generate_body(gnn_op, parallel, f)
            f.write("}\n__syncthreads();\n")
        
        if (long_dynamic):
            f.write("}\nelse {\n")
            if (fop.launch_setup == 'hetero'):
                for i, (gnn_op, parallel) in enumerate(zip(fop.ops, fop.parallel)):
                    f.write("{ ")
                    if (gnn_op.name == "gather"):
                        generate_gather_one_row(gnn_op, parallel, f)
                    if (gnn_op.name == "scatter"):
                        generate_scatter_one_row(gnn_op, f)
                    f.write("}\n__syncthreads();\n")
            # hetero+
            else:
                # f.write("const int nnz_per_warp = 128;\n")
                # f.write("const int dynamic_bsize = 128;\n")
                # f.write("const int nnz_per_block = nnz_per_warp * dynamic_bsize / WARP_SIZE;\n")
                f.write("int row_tmp = info_tmp.row_st;\n")
                f.write("int nnzs = RowPtr[row_tmp + 1] - RowPtr[row_tmp];\n")
                f.write("int dynamic_blocks = (nnzs + DP_NNZ_PER_BLOCK - 1) / DP_NNZ_PER_BLOCK;\n")

                generate_hetero_plus_dynamic_kernels('caller', fop, f, fop_id, kid)

            f.write("}\n")
    else:
        for i, (gnn_op, parallel) in enumerate(zip(fop.ops, fop.parallel)):
            f.write("{ ")
            generate_body(gnn_op, parallel, f)
            f.write("}\n")

    return