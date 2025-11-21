M = 32768
N = 32
K = 512

mt = 16
nt = 32
kt = 32

ms = M / mt
ns = N / nt

mem = 0.912
cal = 34.1

T = 1024 * 1024 * 1024 * 1024
ns = 1024 * 1024

memory_access = 4 * (mt * K + K * nt + mt * nt)
calculation = mt * K * nt

mem_time = memory_access / mem / T
cal_time = calculation / cal / T

print(memory_access)
print(calculation)
print(mem_time)
print(cal_time)

max_time = mem_time if mem_time > cal_time else cal_time

print(max_time * ms * ns)
