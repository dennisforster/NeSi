import resource
import sys
import os

def mem_usage():
    # return the memory usage in MB
    try:
        import psutil
        process = psutil.Process(os.getpid())
        mem = process.get_memory_info()[0] / float(2 ** 20)
    except:
        mem = mem_usage_resource()
    return mem

def mem_usage_resource():
    # faster than psutils but doesn't capture recently liberated RAM
    rusage_denom = 1024.
    if sys.platform == 'darwin':
        # ... it seems that in OSX the output is different units ...
        rusage_denom = rusage_denom * rusage_denom
    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / rusage_denom
    return mem