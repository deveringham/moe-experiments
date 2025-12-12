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