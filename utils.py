###
# utils.py
#
# Misc. utility and helper functions for MoE experiments.
# Dylan Everingham
# 09.12.2025
###

# Time profiling
from functools import wraps
import time

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        print("func:%r args:[%r, %r] took: %2.4f sec" % (f.__name__, args, kw, te-ts))
        return result
    return wrap

# Parameter counting
from prettytable import PrettyTable
def count_params(model, show_table=False):
    if show_table: table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        if show_table: table.add_row([name, params])
        total_params += params
    if show_table: print(table)
    if show_table: print(f"Total Trainable Params: {total_params}")
    return total_params