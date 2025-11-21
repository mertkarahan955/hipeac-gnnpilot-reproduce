import pyparsing as pp
from utils import *

data_list = {}
reserve_data_list = {}
dim_list = []

# Reserved var name: RowPtr, ColIdx
# Do not use them
reserve_data_list['RowPtr'] = ('n', 1, 'int')
reserve_data_list['ColIdx'] = ('e', 1, 'int')
reserve_data_list['numnodes'] = ('c', 1, 'int')
reserve_data_list['numedges'] = ('c', 1, 'int')

def data_func(loc, tok):
    variable_name = tok[0]
    dim2 = tok[4]
    if (variable_name in reserve_data_list):
        parse_error(loc, "usage of reserved variable {}".format(variable_name))
    if (variable_name in data_list):
        parse_warning(loc, "redefinition of {}".format(variable_name))
    if (tok[2] == "v_data"):
        data_list[variable_name] = ('n', dim2, 'float')
    elif (tok[2] == "e_data"):
        data_list[variable_name] = ('e', dim2, 'float')
    elif (tok[2] == "data"):
        data_list[variable_name] = ('c', (dim2, tok[6]), 'float')

variable = pp.Word(pp.alphas)

def dim_func(tok):
    if (tok[0][0: 2] == "fd"):
        dim_id = tok[0][2:]
        feat_name = 'featlen' + dim_id
        if (not feat_name in reserve_data_list):
            reserve_data_list[feat_name] = ('c', 1, 'int')
            dim_list.append(feat_name)

dim_expr = (~pp.Suppress('fd') + pp.Word(pp.nums)) | pp.Combine(pp.Literal('fd') + pp.Optional(pp.Word(pp.nums)))
dim_expr.setParseAction(dim_func)

set_data = pp.Word(pp.alphas) + ":" + \
        (((pp.Literal("v_data") | pp.Literal("e_data")) + "(" + dim_expr + ")") \
        | (pp.Literal("data") + "(" + dim_expr + "," + dim_expr + ")"))
set_data.setParseAction(data_func)
