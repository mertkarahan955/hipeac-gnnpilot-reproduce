import sys
sys.path.append("backend/utils")
sys.path.append("backend/parser")
sys.path.append("backend/intermidiate")
sys.path.append("backend/generator")

from KG_parser import *
from var_check import *
from graph_construct import construct_graph, topo_sort
from generate import kg_generate, kg_generate_all, generate_cmake

if (not len(sys.argv) == 3):
    print("Usage: python dsl_run.py [input dsl code] [output module name]")
    exit

in_file = sys.argv[1]
code_name = sys.argv[2]
out_file = code_name + ".cu"

result = kg_expr.parse_file(in_file)

# print(gnn_op_list[0].var_output)

var_check(data_list, gnn_op_list)

construct_graph(gnn_op_list)
# op_level_list = topo_sort(gnn_op_list)

f = open(out_file, "w")
# kg_generate(code_name, gnn_op_list, op_level_list, "e", "e", "nd", "no", f)

# all
# gs_list = ["n", "ne", "e"]
# ss_list = ["n", "ne", "e"]
# ds_list = ["nd", "ed", "ngd"]
# fs_list = ["no", "ne", "all", "all_d"]

# gather
# gs_list = ["ne"]
# ss_list = ["ne"]
# ds_list = ["nd"]
# fs_list = ["no"]

# model
gs_list = ["ne", "e"]
ss_list = ["ne", "e"]
ds_list = ["nd", "ed", "ngd"]
fs_list = ["no", "ne", "all_d"]

# gs_list = ["ne"]
# ss_list = ["ne"]
# ds_list = ["nd"]
# fs_list = ["no"]

kg_generate_all(code_name, gnn_op_list, f, gs_list = gs_list, ss_list = ss_list, ds_list = ds_list, fs_list = fs_list)

f2 = open("CMakeLists.txt", "w")
generate_cmake(code_name, f2)