
from __future__ import division

import sys
import os
import numpy as np
from mpi4py import MPI

#=============================================================================
# Parallel & pretty printer

typemap = {
    np.dtype('float64'): MPI.DOUBLE,
    np.dtype('float32'): MPI.FLOAT,
    np.dtype('int16'): MPI.SHORT,
    np.dtype('int32'): MPI.INT,
    np.dtype('int64'): MPI.LONG,
    np.dtype('uint16'): MPI.UNSIGNED_SHORT,
    np.dtype('uint32'): MPI.UNSIGNED_INT,
    np.dtype('uint64'): MPI.UNSIGNED_LONG,
}


def pprint(obj="", comm=MPI.COMM_WORLD, end='\n'):
    """
    Parallel print: Make sure only one of the MPI processes
    calling this function actually prints something. All others
    (comm.rank != 0) return without doing enything.
    """
    if comm.rank != 0:
        return

    if isinstance(obj, str):
        sys.stdout.write(obj+end)
        sys.stdout.flush()
    else:
        sys.stdout.write(repr(obj))
        sys.stdout.write(end)
        sys.stdout.flush()