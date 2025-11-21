import pyparsing as pp
from data_def import *
from GNNop_def import *

comment = pp.Literal("#") + ... + "#"

kg_expr = pp.OneOrMore(set_data | GNNop | comment)
