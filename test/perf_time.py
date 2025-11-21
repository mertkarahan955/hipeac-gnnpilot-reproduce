import torch
import time

time_s = []
name_s = []
record = []

def perf_time_init(name):
    global case_name
    case_name = name

def perf_time_start(name):
    torch.cuda.synchronize()
    utils_time1 = time.perf_counter()
    time_s.append(utils_time1)
    name_s.append(name)

def perf_time_end(verbose = True):
    torch.cuda.synchronize()
    utils_time2 = time.perf_counter()
    utils_time1 = time_s.pop()
    name = name_s.pop()
    duration = (utils_time2 - utils_time1) * 1e3
    record.append((name, duration))
    if (verbose):
        print("--" * len(time_s) + "{}: {:.3f} ms".format(name, duration))
    return duration

def perf_time_set(name, duration):
    record.append((name, duration))
    print("--" * len(time_s) + "{}: {:.3f} ms".format(name, duration))

# case name, perf name, time
def perf_time_tofile(file_name):
    with open(file_name, 'a') as f:
        for perf_item in record:
            f.write("{},{},{:.5f}\n".format(case_name, perf_item[0], perf_item[1]))