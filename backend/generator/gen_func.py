from utils import *
from gen_expr import generate_expr, generate_variable

def generate_head(parallel, f):
    f.write("int lane_id = threadIdx.x % WARP_SIZE;\n")
    if (parallel == "n"):
        f.write("int row_tmp = blockIdx.x * blockDim.x + threadIdx.x;\n")
        f.write("if (row_tmp >= numnodes) return;\n")
    elif (parallel == "ne" or parallel == "nd"):
        f.write("int row_tmp = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;\n")
        f.write("if (row_tmp >= numnodes) return;\n")
    elif (parallel == "e" or parallel == "ed"):
        f.write("if (blockIdx.x >= *ep_n) return;\n")
        f.write("row_panel info_tmp = ep_list[blockIdx.x];\n")
        f.write("int local_wid = threadIdx.x / WARP_SIZE;\n")
        f.write("int global_tid = blockIdx.x * blockDim.x + threadIdx.x;\n")
        f.write("int global_wid = global_tid / WARP_SIZE;\n")
    elif (parallel == "ngd"):
        f.write("int global_tid = blockIdx.x * blockDim.x + threadIdx.x;\n")
        f.write("int global_wid = global_tid / WARP_SIZE;\n")
        f.write("if (global_wid >= *ng_n) return;\n")
        f.write("neighbor_group ng_tmp = ng_list[global_wid];\n")
        f.write("int row_tmp = ng_tmp.row_st;\n")
        f.write("int col_st = ng_tmp.col_st;\n")
        f.write("int col_ed = ng_tmp.col_st + NG_SIZE;\n")
        f.write("if (col_ed >= RowPtr[row_tmp + 1]) col_ed = RowPtr[row_tmp + 1];\n")
    else:
        gen_error("Not implemented")

def generate_short_head(parallel, f):
    f.write("int rows = info_tmp.row_ed - info_tmp.row_st;\n" \
    "int col_st = RowPtr[info_tmp.row_st];\n" \
    "int col_ed = RowPtr[info_tmp.row_ed];\n" \
    "int nnzs = col_ed - col_st;\n")

    if (parallel == "ed"):
        f.write("int nnz_per_warp = (nnzs + E_BLOCK_WARP_SIZE - 1) / E_BLOCK_WARP_SIZE;\n" \
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
    
def res_tmp_init(gnn_op, f):
    if (gnn_op.op == 'SUM' or gnn_op.op == 'AVG'):
        f.write("{} res_tmp = 0;\n".format(gnn_op.var_output.info[2]))
    elif (gnn_op.op == 'MUL'):
        f.write("{} res_tmp = 1;\n".format(gnn_op.var_output.info[2]))
    elif (gnn_op.op == 'MAX'):
        # Maybe need a better initilization
        f.write("{} res_tmp = -99999;\n".format(gnn_op.var_output.info[2]))
    else:
        gen_error("GNN op not implemented")

def res_tmp_gather(gnn_op, f):
    if (gnn_op.op == 'SUM' or gnn_op.op == 'AVG'):
        f.write("res_tmp += {};".format(generate_expr(gnn_op.expr, f)))
    elif (gnn_op.op == 'MUL'):
        f.write("res_tmp *= {};".format(generate_expr(gnn_op.expr, f)))
    elif (gnn_op.op == 'MAX'):
        tmp_expr = generate_expr(gnn_op.expr, f)
        f.write("res_tmp = ({} > res_tmp)? {}: res_tmp;".format(tmp_expr, tmp_expr))
    else:
        gen_error("GNN op not implemented")

def res_tmp_writeback(gnn_op, parallel, f):
    if (gnn_op.op == 'AVG'):
        f.write("res_tmp = res_tmp / (RowPtr[row_tmp + 1] - RowPtr[row_tmp]);\n")
    
    if (len(gnn_op.expr.out_expr) > 0):
        f.write("res_tmp = res_tmp {};\n".format(generate_expr(gnn_op.expr.out_expr, f)))

    if (parallel == "n"):
        f.write("{} = res_tmp;\n".format(generate_variable(gnn_op.var_output)))
    elif (parallel == "ne" or parallel == "e"):
        f.write("for (int offset_tmp = WARP_SIZE / 2; offset_tmp > 0; offset_tmp /= 2) {\n")
        f.write("{} comm_tmp = __shfl_down_sync(0xffffffff, res_tmp, offset_tmp);\n".format(gnn_op.var_output.info[2]))
        if (gnn_op.op == 'SUM' or gnn_op.op == 'AVG'):
            f.write("res_tmp += comm_tmp;\n")
        elif (gnn_op.op == 'MAX'):
            f.write("res_tmp = (comm_tmp > res_tmp)? comm_tmp: res_tmp;\n")
        f.write("}\n")

        if (parallel == "ne"):
            f.write("if (lane_id == 0) {} = res_tmp;\n".format(generate_variable(gnn_op.var_output)))
        elif (parallel == "e"):
            f.write("if (lane_id == 0) {\n")
            if (gnn_op.op == 'SUM' or gnn_op.op == 'AVG'):
                f.write("atomicAdd(&{}, res_tmp);\n".format(generate_variable(gnn_op.var_output)))
            # using atomicMax may have problems when max value < 0
            elif (gnn_op.op == 'MAX'):
                f.write("atomicMax")
                if (gnn_op.var_output.info[2] == 'float'):
                    f.write("_float")
                f.write("(&{}, res_tmp);\n".format(generate_variable(gnn_op.var_output)))
            f.write("}\n")
    else:
        gen_error("GNN op not implemented")

# def res_tmp_writeback_atomic(gnn_op, parallel, f):

def res_tmp_writeback_dimwise(gnn_op, f):
    if (gnn_op.op == 'AVG'):
        f.write("res_tmp = res_tmp / (RowPtr[row_tmp + 1] - RowPtr[row_tmp]);\n")

    if (len(gnn_op.expr.out_expr) > 0):
        f.write("res_tmp = res_tmp {};\n".format(generate_expr(gnn_op.expr.out_expr, f)))

    if (gnn_op.op == 'SUM' or gnn_op.op == 'AVG'):
        f.write("atomicAdd(&{}, res_tmp);\n".format(generate_variable(gnn_op.var_output)))
    # using atomicMax may have problems when max value < 0
    elif (gnn_op.op == 'MAX'):
        f.write("atomicMax")
        if (gnn_op.var_output.info[2] == 'float'):
            f.write("_float")
        f.write("(&{}, res_tmp);\n".format(generate_variable(gnn_op.var_output)))


def generate_gather_n_ne_op(gnn_op, parallel, f):
    res_tmp_init(gnn_op, f)

    if (parallel == "n"):
        f.write("for (int nnz_tmp = RowPtr[row_tmp]; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp++) {\n")
    elif (parallel == "ne"):
        f.write("for (int nnz_tmp = RowPtr[row_tmp]+lane_id; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp+=WARP_SIZE) {\n")

    f.write("int col_tmp = ColIdx[nnz_tmp];\n")

    res_tmp_gather(gnn_op, f)

    f.write("}\n")

    res_tmp_writeback(gnn_op, parallel, f)

def generate_gather_e_op(gnn_op, f):
    f.write("using WarpReduce = cub::WarpReduce<{}>;\n".format(gnn_op.var_output.info[2]))
    f.write("__shared__ typename WarpReduce::TempStorage temp_storage[E_BLOCK_WARP_SIZE];\n" \
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

def generate_scatter_op(gnn_op, parallel, f):
    if (parallel == "n"):
        f.write("for (int nnz_tmp = RowPtr[row_tmp]; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp++) {\n")
    elif (parallel == "ne"):
        f.write("for (int nnz_tmp = RowPtr[row_tmp] + lane_id; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp+=WARP_SIZE) {\n")
    elif (parallel == "e"):
        f.write("int row_tmp = info_tmp.row_st;\n" \
        "for (int nnz_tmp = col_st + threadIdx.x; nnz_tmp < col_ed; nnz_tmp += blockDim.x) {\n" \
        "while (RowPtr[row_tmp + 1] <= nnz_tmp) row_tmp++;\n")
    else:
        gen_error("Wrong parallel setup")
    f.write("int col_tmp = ColIdx[nnz_tmp];\n")
    f.write("{} = {};".format(generate_variable(gnn_op.var_output), generate_expr(gnn_op.expr, f)))
    f.write("\n}\n")

def generate_gather_nd_op(gnn_op, f):
    f.write("for (int k_tmp = lane_id; k_tmp < featlen; k_tmp += WARP_SIZE) {\n")
    res_tmp_init(gnn_op, f)

    f.write("for (int nnz_tmp = RowPtr[row_tmp]; nnz_tmp < RowPtr[row_tmp + 1]; nnz_tmp++) {\n")
    f.write("int col_tmp = ColIdx[nnz_tmp];\n")

    res_tmp_gather(gnn_op, f)

    f.write("}\n")

    if (len(gnn_op.expr.out_expr) > 0):
        f.write("res_tmp = res_tmp {};\n".format(generate_expr(gnn_op.expr.out_expr, f)))

    if (gnn_op.op == 'AVG'):
        f.write("res_tmp = res_tmp / (RowPtr[row_tmp + 1] - RowPtr[row_tmp]);\n")

    f.write("{} = res_tmp;\n".format(generate_variable(gnn_op.var_output)))
    f.write("}\n")

def res_tmp_write_atomic(gnn_op, f):

    if (len(gnn_op.expr.out_expr) > 0):
        f.write("res_tmp = res_tmp {};\n".format(generate_expr(gnn_op.expr.out_expr, f)))

    if (gnn_op.op == 'SUM' or gnn_op.op == 'AVG'):
        f.write("atomicAdd(&{}, res_tmp);\n".format(generate_variable(gnn_op.var_output)))
    # using atomicMax may have problems when max value < 0
    elif (gnn_op.op == 'MAX'):
        f.write("atomicMax")
        if (gnn_op.var_output.info[2] == 'float'):
            f.write("_float")
        f.write("(&{}, res_tmp);\n".format(generate_variable(gnn_op.var_output)))

def generate_gather_ed_op(gnn_op, f):
    
    f.write("if (warp_col_ed <= col_ed) {\n" \
    "for (int row_tmp = warp_row_st; row_tmp < warp_row_ed; row_tmp++) {\n")
    
    f.write("int current_col_st = RowPtr[row_tmp];\n" \
    "int current_col_ed = RowPtr[row_tmp + 1];\n" \
    "if (row_tmp == warp_row_st) current_col_st = warp_col_st;\n" \
    "if (row_tmp == warp_row_ed - 1) current_col_ed = warp_col_ed;\n")

    f.write("for (int k_tmp = lane_id; k_tmp < featlen; k_tmp += WARP_SIZE) {\n")
    res_tmp_init(gnn_op, f)

    f.write("for (int nnz_tmp = current_col_st; nnz_tmp < current_col_ed; nnz_tmp++) {\n")
    f.write("int col_tmp = ColIdx[nnz_tmp];\n")

    res_tmp_gather(gnn_op, f)
    f.write("}\n")

    if (gnn_op.op == 'AVG'):
        f.write("res_tmp = res_tmp / (RowPtr[row_tmp + 1] - RowPtr[row_tmp]);\n")

    res_tmp_write_atomic(gnn_op, f)

    f.write("}}}\n")

def generate_gather_ngd_op(gnn_op, f):
    f.write("for (int k_tmp = lane_id; k_tmp < featlen; k_tmp += WARP_SIZE) {\n")
    res_tmp_init(gnn_op, f)
    f.write("for (int nnz_tmp = col_st; nnz_tmp < col_ed; nnz_tmp++) {\n")
    f.write("int col_tmp = ColIdx[nnz_tmp];\n")
    res_tmp_gather(gnn_op, f)
    f.write("}\n")

    if (gnn_op.op == 'AVG'):
        f.write("res_tmp = res_tmp / (RowPtr[row_tmp + 1] - RowPtr[row_tmp]);\n")

    res_tmp_write_atomic(gnn_op, f)
    
    f.write("}\n")

def generate_gather_one_row(gnn_op, parallel, f):

    if (parallel == 'nd' or parallel == 'ed'):
        f.write("int row_tmp = info_tmp.row_st;\n")
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
        
        f.write("for (int nnz_tmp = info_tmp.col_st + local_wid; nnz_tmp < info_tmp.col_ed; nnz_tmp+=HE_BLOCK_WARP_SIZE) {\n")
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
        
        f.write("}\n")

    else:
        f.write("int row_tmp = info_tmp.row_st;\n")

        if (gnn_op.op == 'SUM' or gnn_op.op == 'AVG'):
            f.write("{} res_tmp = 0;\n".format(gnn_op.var_output.info[2]))
        elif (gnn_op.op == 'MUL'):
            f.write("{} res_tmp = 1;\n".format(gnn_op.var_output.info[2]))
        elif (gnn_op.op == 'MAX'):
            # Maybe need a better initilization
            f.write("{} res_tmp = -99999;\n".format(gnn_op.var_output.info[2]))
        else:
            gen_error("GNN op not implemented")

        f.write("for (int nnz_tmp = info_tmp.col_st + threadIdx.x; nnz_tmp < info_tmp.col_ed; nnz_tmp += HE_BLOCK_SIZE) {\n")
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

        f.write("}\n")

        if (gnn_op.op == 'AVG'):
            f.write("res_tmp = res_tmp / (RowPtr[row_tmp + 1] - RowPtr[row_tmp]);\n")
        if (len(gnn_op.expr.out_expr) > 0):
            f.write("res_tmp = res_tmp {};\n".format(generate_expr(gnn_op.expr.out_expr, f)))

        f.write("for (int offset_tmp = WARP_SIZE / 2; offset_tmp > 0; offset_tmp /= 2) {\n")
        f.write("{} comm_tmp = __shfl_down_sync(0xffffffff, res_tmp, offset_tmp);\n".format(gnn_op.var_output.info[2]))
        if (gnn_op.op == 'SUM' or gnn_op.op == 'AVG'):
            f.write("res_tmp += comm_tmp;\n")
        elif (gnn_op.op == 'MAX'):
            f.write("res_tmp = (comm_tmp > res_tmp)? comm_tmp: res_tmp;\n")
        f.write("}\n")

        f.write("if (lane_id == 0) {\n")
        if (gnn_op.op == 'SUM' or gnn_op.op == 'AVG'):
            f.write("atomicAdd(&{}, res_tmp);\n".format(generate_variable(gnn_op.var_output)))
        # using atomicMax may have problems when max value < 0
        elif (gnn_op.op == 'MAX'):
            f.write("atomicMax")
            if (gnn_op.var_output.info[2] == 'float'):
                f.write("_float")
            f.write("(&{}, res_tmp);\n".format(generate_variable(gnn_op.var_output)))

        f.write("}\n")

def generate_scatter_one_row(gnn_op, f):
    f.write("int row_tmp = info_tmp.row_st;\n")
    f.write("for (int nnz_tmp = info_tmp.col_st + threadIdx.x; nnz_tmp < info_tmp.col_ed; nnz_tmp += HE_BLOCK_SIZE) {\n")
    f.write("int col_tmp = ColIdx[nnz_tmp];\n")
    f.write("{} = {};".format(generate_variable(gnn_op.var_output), generate_expr(gnn_op.expr, f)))
    f.write("}\n")

def generate_body(gnn_op, parallel, f, long_dynamic=True):
    if (parallel == "n"):
        if (gnn_op.name == "gather"):
            generate_gather_n_ne_op(gnn_op, "n", f)
        elif (gnn_op.name == "scatter"):
            generate_scatter_op(gnn_op, "n", f)
    elif (parallel == "ne"):
        if (gnn_op.name == "gather"):
            generate_gather_n_ne_op(gnn_op, "ne", f)
        elif (gnn_op.name == "scatter"):
            generate_scatter_op(gnn_op, "ne", f)
    elif (parallel == "e"):
        if (long_dynamic):
            f.write("if (info_tmp.col_st == -1){ \n")
        generate_short_head(parallel, f)
        if (gnn_op.name == "scatter"):
            generate_scatter_op(gnn_op, "e", f)
        elif (gnn_op.name == "gather"):
            generate_gather_e_op(gnn_op, f)
        if (long_dynamic):
            f.write("}\nelse {\n")
            if (gnn_op.name == "scatter"):
                generate_scatter_one_row(gnn_op, f)
            elif (gnn_op.name == "gather"):
                generate_gather_one_row(gnn_op, parallel, f)
            f.write("}\n")
    elif (parallel == "nd"):
        if (gnn_op.name == "gather"):
            generate_gather_nd_op(gnn_op, f)
    elif (parallel == "ed"):
        if (long_dynamic):
            f.write("if (info_tmp.col_st == -1){ \n")
        if (gnn_op.name == "gather"):
            generate_short_head(parallel, f)
            generate_gather_ed_op(gnn_op, f)
        if (long_dynamic):
            f.write("}\nelse {\n")
            generate_gather_one_row(gnn_op, parallel, f)
            f.write("}\n")
    elif (parallel == "ngd"):
        if (gnn_op.name == "gather"):
            generate_gather_ngd_op(gnn_op, f)
    else:
        gen_error("Not implemented")
