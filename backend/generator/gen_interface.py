from data_def import data_list, dim_list
from utils import *

def generate_interface(name, fop_list, f, kid):

    f.write("void {}_kernel_{:d}(".format(name, kid))

    # graph data
    f.write("int64_t info_addr, torch::Tensor RowPtr, torch::Tensor ColIdx")

    for dim_term in dim_list:
        f.write(", int64_t {}".format(dim_term))

    featlen_var = None
    for var in data_list.keys():
        f.write(", torch::Tensor {}".format(var))
        if (data_list[var][1] == 'fd'):
            featlen_var = var

    f.write(") {\n")

    f.write("kg_info* info_ = (kg_info*)info_addr;\n")
    f.write("if (info_ == nullptr) {\n")
    f.write("  throw std::runtime_error(\"info_ is null in {}_kernel_{:d}\");\n".format(name, kid))
    f.write("}\n")
    f.write("if (info_->rp_info == nullptr || info_->rp_n == nullptr || ")
    f.write("info_->ep_info == nullptr || info_->ep_n == nullptr || ")
    f.write("info_->ng_info == nullptr || info_->ng_n == nullptr) {\n")
    f.write("  throw std::runtime_error(\"Some preprocessing pointers are null in {}_kernel_{:d}\");\n".format(name, kid))
    f.write("}\n")
    f.write("int numnodes = RowPtr.size(0) - 1;\n")
    f.write("int numedges = ColIdx.size(0);\n")
    # f.write("int featlen = ")
    # if (featlen_var == None):
    #     f.write("1;\n")
    # else:
    #     f.write("{}.size(1);\n".format(featlen_var))
    f.write("int num_n_block = (numnodes + N_BLOCK_SIZE - 1) / N_BLOCK_SIZE;\n")
    f.write("int num_ne_block = (numnodes + NE_BLOCK_WARP_SIZE - 1) / NE_BLOCK_WARP_SIZE;\n")
    f.write("int num_e_block = info_->ep_n_host;\n")
    #f.write("int num_he_block = (numnodes + HE_BLOCK_WARP_SIZE - 1) / HE_BLOCK_WARP_SIZE;\n")
    f.write("int num_he_block = info_->rp_n_host;\n")
    f.write("int num_ngd_block = (info_->ng_n_host + NGD_BLOCK_WARP_SIZE - 1) / NGD_BLOCK_WARP_SIZE;\n")

    for (fid, fop) in enumerate(fop_list):
        # CUDA kernel
        if (not fop.ops[0].name == "linear"):
            # print(fop.ops[0].name, fop.ops[0].expr.expr, fop.launch_setup)
            f.write("k_{:d}_fop_{:d}".format(kid, fid))
            if (fop.launch_setup == "n"):
                num_block = "num_n_block"
                block_size = "N_BLOCK_SIZE"
            elif (fop.launch_setup == "ne" or fop.launch_setup == "nd"):
                num_block = "num_ne_block"
                block_size = "NE_BLOCK_SIZE"
            elif (fop.launch_setup == "e" or fop.launch_setup == "ed"):
                num_block = "num_e_block"
                block_size = "E_BLOCK_SIZE"
            elif (fop.launch_setup == "ngd"):
                num_block = "num_ngd_block"
                block_size = "NGD_BLOCK_SIZE"
            elif ("hetero" in fop.launch_setup):
                num_block = "num_he_block"
                block_size = "HE_BLOCK_SIZE"
            else:
                gen_error("CUDA launch setup not implemented")
            f.write("<<<{}, {}>>>".format(num_block, block_size))

            f.write("(numnodes, numedges, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(), ")
            for dim_term in dim_list:
                f.write("{}, ".format(dim_term))
            # optimize this
            f.write("info_->rp_info, info_->rp_n, info_->ep_info, info_->ep_n, info_->ng_info, info_->ng_n")
            for var in fop.var:
                f.write(", {}.data_ptr<{}>()".format(var, data_list[var][2]))
            f.write(");\n")
        # torch mm kernel
        else:
            f.write("torch::matmul_outf({}, {}, {});\n".format( \
            fop.ops[0].expr.var_input[0].name, fop.ops[0].expr.var_input[1].name, \
            fop.ops[0].var_output.name))

    f.write("}\n")

def generate_python_interface(name, fop_list, f, kid):
    func_name = "{}_kernel_{:d}".format(name, kid)
    f.write("m.def(\"{}\", &{});\n".format(func_name, func_name))
