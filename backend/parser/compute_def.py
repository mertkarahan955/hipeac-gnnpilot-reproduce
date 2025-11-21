import pyparsing as pp
from utils import *
from data_def import data_list, dim_expr

class var_class:
    name = ''
    ref = ('1', '1')
    def __init__(self, n, r):
        self.name = n
        self.ref = r
        self.info = data_list[n]

def data_func(loc, tok):
    if (not tok[0] in data_list):
        parse_error(loc, "{} not defined".format(tok[0]))

# currently only support 2 dimensional data
# i means the current row, j means the current column
variable_op = ~pp.Suppress("max") + ~pp.Suppress("exp") + ~pp.Suppress("GaussK") + pp.Word(pp.alphas)
variable_op.setParseAction(data_func)
index_expr = pp.Word(pp.nums) | "i" | "j" | "nnz" | ":"
index_expr_noi = pp.Word(pp.nums) | "j" | "nnz" | ":"

index = pp.Literal("(") + index_expr + "," + index_expr + pp.Literal(")")
index.setParseAction(lambda tok: (tok[1], tok[3]))

variable_expr = variable_op + index
variable_expr.setParseAction(lambda tok: var_class(tok[0], tok[1]))

arith_op = pp.one_of("+ - * /")
mult_div_op = pp.one_of("* /")

lpar = pp.Literal("(")
rpar = pp.Literal(")")

tmp_expr = pp.Forward()
tmp_term = pp.Forward()
GaussK_term = pp.Forward()

class GaussK_class:
    def __init__(self, p, mu, diag, d):
        self.tok = []
        self.tok.append(p)
        self.tok.append(mu)
        self.tok.append(diag)
        self.tok.append(d)

def GaussK_func(tok):
    return GaussK_class(list(tok[2]), tok[4], tok[6], tok[8])

GaussK_term <<= pp.Literal("GaussK") + lpar + pp.Group(tmp_expr) + "," + tmp_expr + "," + tmp_expr + "," + dim_expr + rpar
GaussK_term.setParseAction(GaussK_func)

tmp_term <<= lpar + tmp_expr + rpar | pp.Literal("max") + lpar + tmp_expr + "," + tmp_expr + rpar \
            | pp.Literal("exp") + lpar + tmp_expr + rpar | GaussK_term | variable_expr
tmp_expr <<= tmp_term + arith_op + tmp_expr | tmp_term

compute_expr = pp.Optional(mult_div_op) + tmp_expr

# compute_expr = pp.Forward()
# # compute_expr <<= variable_expr
# compute_expr << variable_expr + pp.ZeroOrMore(arith_op - variable_expr)
# compute_expr << pp.Literal("MAX") + "(" + compute_expr + "," + compute_expr + ")"
# compute_expr = pp.Literal("haha")

# def compute_expr_func(loc, tok):
#     print(tok)
#     # return tok
#     # var_res = []
#     # print(tok)
#     # for token in tok:
#     #     if (isinstance(token, var_class) and len(list(filter(lambda x: x.name == token.name, var_res))) == 0):
#     #         var_res.append(token)
#     # return var_res, list(tok)

# compute_expr.setParseAction(compute_expr_func)

class compute_expr_class:
    expr = ''
    out_expr = ''
    var_input = []
    def __init__(self, e, v):
        self.expr = e
        self.var_input = v

def compute_expr_func(loc, tok):
    try:
        # print("parsing", tok)
        res = compute_expr.parse_string(tok[1])
        # print("result", res)
        # print("compute", res)

        var_res = []
        for token in res:
            if (isinstance(token, var_class) and len(list(filter(lambda x: x.name == token.name, var_res))) == 0):
                var_res.append(token)
            if (isinstance(token, GaussK_class)):
                for var in token.tok[0]:
                    if isinstance(var, var_class) and len(list(filter(lambda x: x.name == var.name, var_res))) == 0:
                        var_res.append(var)
                for var in token.tok[1: 3]:
                    if isinstance(var, var_class) and len(list(filter(lambda x: x.name == var.name, var_res))) == 0:
                        var_res.append(var)

        # print("var_res", var_res)
        return compute_expr_class(tok[1], var_res)

    except pp.ParseException as pe:
        parse_error(loc, "failed compute expression parse {}".format(tok[1]))

    # if (var_res == None):
    #     print("haha")
    # print(var_res)
    # return compute_expr_class(tok[1], var_res)

compute_expr_c = pp.Literal("\"") + ... + pp.Literal("\"")
compute_expr_c.setParseAction(compute_expr_func)

# multiple / division factor 
# Only remove right operations
variable_expri = variable_op + pp.Literal("(") + "i" + "," + index_expr + pp.Literal(")")
variable_exprnoi = variable_op + pp.Literal("(") + index_expr_noi + "," + index_expr + pp.Literal(")")

tmp_factor_term = lpar + tmp_expr + rpar | variable_exprnoi

class out_expr_class:
    expr = ''
    def __init__(self, e):
        self.expr = e

def tmp_func(loc, tok):
    print(tok)

outr_factor = mult_div_op + variable_expri
outr_factor.setParseAction(lambda tok: out_expr_class("".join(tok)))
# outl_factor = variable_expri
# outl_factor.setParseAction(lambda tok: out_expr_class("*".join(tok)))

tmp_factor_expr = tmp_factor_term + pp.ZeroOrMore(mult_div_op + tmp_factor_term) + pp.ZeroOrMore(outr_factor)

# def tmp_func(loc, tok):
#     print(tok)
# tmp_factor_expr.setParseAction(tmp_func)
