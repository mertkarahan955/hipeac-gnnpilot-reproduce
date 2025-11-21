
def parse_warning(loc, notice):
    print("Warning at {:d}: {}".format(loc, notice))

def parse_error(loc, notice):
    raise Exception("Error at {:d}: {}".format(loc, notice))

def inter_warning(notice):
    print("compiling warning: {}".format(notice))

def inter_error(notice):
    raise Exception("compiling error: {}".format(notice))

def gen_warning(notice):
    print("generating warning: {}".format(notice))

def gen_error(notice):
    raise Exception("generating error: {}".format(notice))