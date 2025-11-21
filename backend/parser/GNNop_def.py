import pyparsing as pp
#from compute_def import compute_expr_c, variable_expr, variable_op, var_class, compute_expr_class
from compute_def import *
from utils import *

gnn_op_list = []

class gnn_op:
    name = ''
    op = ''
    def __init__(self, n, e, out, op = ''):
        self.name = n
        self.op = op
        self.expr = e
        self.var_output = out

max_op = pp.Literal('MAX')
sum_op = pp.Literal('SUM')
avg_op = pp.Literal('AVG')
gather_op = max_op | sum_op | avg_op

def gather_op_func(loc, tok):
    gnn_op_list.append(gnn_op(tok[2], tok[6], tok[0], tok[4]))

    try:
        res = tmp_factor_expr.parse_string(tok[6].expr)
        new_expr = ""
        out_expr = ""
        for tok in res:
            if (isinstance(tok, out_expr_class)):
                out_expr += tok.expr
            else:
                new_expr += tok
        if (len(out_expr) > 0):
            gnn_op_list[-1].expr.expr = new_expr
            gnn_op_list[-1].expr.out_expr = out_expr

    except pp.ParseException as pe:
        #print("not outable")
        pass

gather_op = variable_expr + "=" + pp.Literal("gather") + "(" + gather_op + "," + compute_expr_c + ")"
gather_op.setParseAction(gather_op_func)

def scatter_op_func(loc, tok):
    # print(tok)
    gnn_op_list.append(gnn_op(tok[2], tok[4], tok[0]))
    # print("check", gnn_op_list[0].expr.var_input[0].name, tok[0].name)

scatter_op = variable_expr + "=" + pp.Literal("scatter") + "(" + compute_expr_c + ")"
scatter_op.setParseAction(scatter_op_func)

def linear_op_func(loc, tok):
    current_input_list = [var_class(tok[4], (':', ':')), var_class(tok[6], (':', ':'))]
    current_output = var_class(tok[0], (':', ':'))
    current_expr = compute_expr_class('', current_input_list)
    gnn_op_list.append(gnn_op(tok[2], current_expr, current_output))

linear_op = variable_op + "=" + pp.Literal("linear") + "(" + variable_op + "," + variable_op + ")"
linear_op.setParseAction(linear_op_func)

GNNop = gather_op | scatter_op | linear_op