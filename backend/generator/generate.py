from foperator import get_fop_list
from gen_interface import generate_interface, generate_python_interface
from gen_kernel import generate_kernel
from gen_cmake import generate_cmake

def kg_eliminate(gs, ss, ds, fs):
    if (ds == 'ngd' and 'all' in fs):
        return True
    return False

def kg_generate(code_name, gnn_op_list, gs, ss, ds, fs, f):
    generate_code_head(f)
    
    fop_list = get_fop_list(gnn_op_list, gs, ss, ds, fs, kid)
    generate_code(code_name, fop_list, f, 0)

    f.write("TORCH_LIBRARY({}, m) {{\n".format(code_name + "lib"))
    generate_python_interface(code_name, fop_list, f, 0)
    f.write("m.def(\"preprocessing\", &preprocessing);\n")
    f.write("}")

def kg_generate_all(code_name, gnn_op_list, f, \
    gs_list = ['n', 'ne', 'e'], ss_list = ['n', 'ne', 'e'], ds_list = ['nd', 'ed'], fs_list = ['no', 'ne', 'full']):
    generate_code_head(f)
    
    gather_setups = gs_list
    scatter_setups = ss_list
    dim_setups = ds_list
    fuse_setups = fs_list

    kid = 0
    for gs in gather_setups:
        for ss in scatter_setups:
            for ds in dim_setups:
                for fs in fuse_setups:
                    if (kg_eliminate(gs, ss, ds, fs)):
                        continue
                    fop_list = get_fop_list(gnn_op_list, gs, ss, ds, fs, kid)
                    generate_code(code_name, fop_list, f, kid)
                    kid += 1

    f.write("TORCH_LIBRARY({}, m) {{\n".format(code_name + "lib"))
    for i in range(kid):
        generate_python_interface(code_name, fop_list, f, i)
    f.write("m.def(\"preprocessing\", &preprocessing);\n")
    f.write("}")

def generate_code(code_name, fop_list, f, kid):
    for fop_id, fop in enumerate(fop_list):
        generate_kernel(fop_id, fop, f, kid)
    generate_interface(code_name, fop_list, f, kid)

def generate_code_head(f):
    f.write("#include <torch/extension.h>\n" \
    "#include <cub/cub.cuh>\n" \
    "#include \"preprocessing.h\"\n" \
    "#define WARP_SIZE 32\n" \
    "#define N_BLOCK_SIZE 128\n" \
    "#define NE_BLOCK_SIZE 128\n" \
    "#define NE_BLOCK_WARP_SIZE (NE_BLOCK_SIZE / WARP_SIZE)\n" \
    "#define E_BLOCK_SIZE 128\n" \
    "#define E_BLOCK_WARP_SIZE (E_BLOCK_SIZE / WARP_SIZE)\n" \
    "#define HE_BLOCK_SIZE 128\n"
    "#define HE_BLOCK_WARP_SIZE (HE_BLOCK_SIZE / WARP_SIZE)\n" \
    "#define NGD_BLOCK_SIZE 128\n" \
    "#define NGD_BLOCK_WARP_SIZE (NGD_BLOCK_SIZE / WARP_SIZE)\n" \
    "#define kg_max(a, b) ((a)>(b)? (a): (b))\n"
    "#define kg_min(a, b) ((a)<(b)? (a): (b))\n\n"
    "extern int64_t preprocessing_cuda(int m, int nnz, int *RowPtr, int *ColIdx, bool long_dynamic);\n"
    "int64_t preprocessing(torch::Tensor RowPtr, torch::Tensor ColIdx, int64_t long_dynamic) {\n"
    "int m = RowPtr.size(0) - 1;\n"
    "int nnz = ColIdx.size(0);\n"
    "return preprocessing_cuda(m, nnz, RowPtr.data_ptr<int>(), ColIdx.data_ptr<int>(), long_dynamic);\n"
    "}\n\n"
    "__device__ static float atomicMax_float(float* addr, float val) {\n" \
    "int* addr_as_int = (int*)addr;\n" \
    "int old = *addr_as_int;\n" \
    "int expected;\n" \
    "do {\n" \
    "expected = old;\n" \
    "old = ::atomicCAS(addr_as_int, expected,\n" \
    "__float_as_int(::fmaxf(val, __int_as_float(expected))));\n" \
    "} while (expected != old);\n" \
    "return __int_as_float(old);\n" \
    "}\n\n"
    )
