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

def allsort(my_array, axis=-1, kind='quicksort', order=None, comm=MPI.COMM_WORLD):
    """
    Parallel (collective) version of numpy.sort
    """
    shape = my_array.shape
    all_shape = list(shape)
    all_shape[axis] = comm.allreduce(shape[axis])

    if not my_array.dtype in typemap:
        raise TypeError("Dont know how to handle arrays of type %s" % my_array.dtype)
    mpi_type = typemap[my_array.dtype]

    my_sorted = np.sort(my_array, axis, kind, order)

    all_array = np.empty(shape=all_shape, dtype=my_sorted.dtype)
    comm.Allgather((my_sorted, mpi_type), (all_array, mpi_type))

    all_sorted = np.sort(all_array, axis, 'mergesort', order)
    return all_sorted

def allargsort(my_array, axis=-1, kind='quicksort', order=None, comm=MPI.COMM_WORLD):
    """
    Parallel (collective) version of numpy.argsort
    """
    shape = my_array.shape
    all_shape = list(shape)
    all_shape[axis] = comm.allreduce(shape[axis])

    if not my_array.dtype in typemap:
        raise TypeError("Dont know how to handle arrays of type %s" % my_array.dtype)
    mpi_type = typemap[my_array.dtype]

    my_sorted = np.argsort(my_array, axis, kind, order)

    all_array = np.empty(shape=all_shape, dtype=my_sorted.dtype)
    comm.Allgather((my_sorted, mpi_type), (all_array, mpi_type))

    all_sorted = np.argsort(all_array, axis, kind, order)
    return all_sorted

def allmean(my_a, axis=None, dtype=None, out=None, comm=MPI.COMM_WORLD):
    """
    Parallel (collective) version of numpy.mean
    """
    shape = my_a.shape
    if axis is None:
        N = comm.allreduce(my_a.size)
    else:
        N = comm.allreduce(shape[axis])

    my_sum = np.sum(my_a, axis, dtype)

    if my_sum is np.ndarray:
        sum = np.empty_like(my_sum)
        comm.Allreduce( (my_sum, typemap[my_sum.dtype]), (sum, typemap[sum.dtype]))
        sum /= N
        return sum
    else:
        return comm.allreduce(my_sum) / N

def allsum(my_a, axis=None, dtype=None, out=None, comm=MPI.COMM_WORLD):
    """
    Parallel (collective) version of numpy.sum
    """
    my_sum = np.sum(my_a, axis, dtype)

    if my_sum is np.ndarray:
        sum = np.empty_like(my_sum)
        comm.Allreduce( (my_sum, typemap[my_sum.dtype]), (sum, typemap[sum.dtype]))
        return sum
    else:
        return comm.allreduce(my_sum)


#It works, but it need from creating an *output* array in all the processes,
#highly unefficient
def asimetric_gather(my_array, N, output, comm=MPI.COMM_WORLD):
    """
    In the *output* array all *my_array* will be stack in the first dimension,
    at process 0. All the rest of the dimensions are assumed to be equal among
    the different processes. It's the user's responsability to ensure that all

    my_array have the same dtype/
    """
    py_type = my_array.dtype
    mpi_type = typemap[py_type]
    if comm.rank == 0:
        my_N = N//comm.size
        shape = list(my_array.shape)
        my_temp_array = np.empty(shape, py_type)
        #We create the new array to allgather
        shape[0] = N
        full_gather = np.empty(shape, py_type)
        full_gather[0:my_N, ...] = my_array.copy()

    for i in xrange(1, comm.size-1):
        if comm.rank == i:
            comm.Send([my_array, mpi_type], tag=2*i)
        if comm.rank == 0:
            comm.Recv([my_temp_array, mpi_type], source=i, tag=2*i)
            full_gather[i*my_N:(i+1)*my_N, ...] = my_temp_array.copy()
            my_temp_array.fill(0.)

    if comm.rank == 0:
        #We need to create an array of different size for the last one
        shape[0] = N - (comm.size-1)*my_N
        my_temp_array = np.empty(shape, py_type)

    if comm.rank == (comm.size-1):
        comm.Send([my_array, mpi_type], tag=(comm.size-1) )

    if comm.rank == 0:
        comm.Recv([my_temp_array, mpi_type], source=(comm.size-1),tag=(comm.size-1) )
        full_gather[(comm.size-1)*my_N:N, ...] = my_temp_array.copy()

        output[...] = full_gather[...]


def getHostName():
    '''
    Return the host name.
    '''
    return os.uname()[1]

def getRankSizeWithinHost(comm=MPI.COMM_WORLD):
    '''
    Get the rank and size of current process within current host.
    '''
    hostname = getHostName()
    hostlist = comm.allgather(hostname)
    rank=0
    size=0
    for i in xrange(comm.size):
        if hostlist[i] == hostname:
            size = size+1
            if i<comm.rank:
                rank = rank+1
    return (rank,size)

