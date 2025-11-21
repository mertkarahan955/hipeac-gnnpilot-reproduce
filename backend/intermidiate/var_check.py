from utils import *
from data_def import data_list

def check_ref_output(data_list, var, type_gnnop):
    # print(type_gnnop, var.ref)
    if (type_gnnop == "gather"):
        if (not var.ref[0] == "i"):
            return False
        if (var.ref[1] == "i" or var.ref[1] == "j"):
            return False
        return True

    if (type_gnnop == "scatter"):
        if (not var.ref[0] == "nnz"):
            return False
        if (var.ref[1] == "i" or var.ref[1] == "j"):
            return False
        return True

    if (type_gnnop == "linear"):
        return True

    return False

def check_ref_input(data_list, var, type_gnnop):
    # print(type_gnnop, var.ref)
    if (type_gnnop == "gather"):
        if (var.ref[0] == ":"):
            return False
        if (var.ref[1] == "i" or var.ref[1] == "j"):
            return False
        return True

    if (type_gnnop == "scatter"):
        if (var.ref[0] == ":"):
            return False
        if (var.ref[1] == "i" or var.ref[1] == "j"):
            return False
        return True

    if (type_gnnop == "linear"):
        return True

def var_check(data_list, gnn_op_list):
    for gnn_op in gnn_op_list:
        #if (gnn_op.name == "gather" or gnn_op.name == "scatter"):
        var_output = gnn_op.var_output
        var_name = var_output.name

        if (not check_ref_output(data_list, var_output, gnn_op.name)):
            inter_error("Wrong output variable indices of {}".format(var_name))
        
        output_dim = var_output.ref[1]
        
        for var_input in gnn_op.expr.var_input:
            if (not check_ref_input(data_list, var_input, gnn_op.name)):
                inter_error("Wrong input variable indices of {}".format(var_name))

        # to do: check the dimension of input variables
        # input_dim = var_input.ref[1]
        # if (input_dim != output_dim):
        #     inter_error("Wrong input dim")

        # else:
        #     inter_error("Wrong operator type")

