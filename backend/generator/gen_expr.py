from compute_def import *

def get_ref(ref, GaussK = False):
    if (ref == 'i'):
        return 'row_tmp'
    elif (ref == 'j'):
        return 'col_tmp'
    elif (ref == 'nnz'):
        return 'nnz_tmp'
    elif (ref == ':'):
        if (GaussK):
            return "GaussK_d"
        return 'k_tmp'
    else:
        return ref

def get_dim(dim):
    if ('fd' in dim):
        dim_id = ""
        if (len(dim) > 2):
            dim_id = dim[2:]
        return 'featlen' + dim_id
    else:
        return dim

def generate_variable(var):
    if (var.info[0] == 'c'):
        return "{}[{} * {} + {}]".format(var.name, get_ref(var.ref[0]), get_dim(var.info[1][1]), get_ref(var.ref[1]))
    else:
        return "{}[{} * {} + {}]".format(var.name, get_ref(var.ref[0]), get_dim(var.info[1]), get_ref(var.ref[1]))
        # f.write("{}[".format(var.name))
        # f.write("{} * {} + {}".format(get_ref(var.ref[0]), get_dim(var.info[1]), get_ref(var.ref[1])))
        # f.write("]")

def generate_variable_GaussK(var):
    if (var.info[0] == 'c'):
        return "{}[{} * {} + {}]".format(var.name, get_ref(var.ref[0], GaussK = True), \
        get_dim(var.info[1][1]), get_ref(var.ref[1], GaussK = True))
    else:
        return "{}[{} * {} + {}]".format(var.name, get_ref(var.ref[0], GaussK = True), \
        get_dim(var.info[1]), get_ref(var.ref[1], GaussK = True))

def generate_GaussK(GaussK, f):
    toks = GaussK.tok
    f.write("float GaussK_tmp = 0;\n")
    f.write("for (int GaussK_d = 0; GaussK_d < {}; GaussK_d++) {{\n".format(get_dim(toks[3])))
    res_tmp = ""
    for tok in toks[0]:
        if (isinstance(tok, var_class)):
            res_tmp += generate_variable_GaussK(tok)
        else:
            res_tmp += tok
    f.write("float GaussK_res_tmp = ({}) - {};\n".format(res_tmp, generate_variable_GaussK(toks[1])))
    f.write("GaussK_tmp = GaussK_res_tmp * GaussK_res_tmp * {} * {};".format( \
    generate_variable_GaussK(toks[2]), generate_variable_GaussK(toks[2])))
    f.write(" }\n")

def generate_expr(expr_instance, f):
    #print(expr_instance.expr)
    if (isinstance(expr_instance, str)):
        toks = compute_expr.parse_string(expr_instance)
    else:
        toks = compute_expr.parse_string(expr_instance.expr)
    retstr = ''
    for token in toks:
        if (isinstance(token, var_class)):
            retstr += generate_variable(token)
        elif (isinstance(token, GaussK_class)):
            generate_GaussK(token, f)
            retstr += "GaussK_tmp"
        elif (token == "max"):
            retstr += "kg_max"
        elif (token == "exp"):
            retstr += "expf"
        else:
            retstr += token
    return retstr
