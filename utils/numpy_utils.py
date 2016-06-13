import numpy as np
from scipy import sparse


def save_sparse(filename, array):
    np.savez(filename, data=array.data, indices=array.indices, indptr=array.indptr, shape=array.shape)
    #np.savez(filename, data=array.data, row=array.row, col=array.col, shape=array.shape)


def save_sparse_list(filename, arr_list):
    kwds = {}
    for i in range(len(arr_list)):
        array = arr_list[i]
        kwds['data%d' % i] = array.data
        kwds['indices%d' % i] = array.indices
        kwds['indptr%d' % i] = array.indptr
        #kwds['row%d' % i] = array.row
        #kwds['col%d' % i] = array.col
        kwds['shape%d' % i] = array.shape
    np.savez(filename, **kwds)


def load_sparse(filename):
    loader = np.load(filename)
    return sparse.csc_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'],
                             dtype=np.float32)
    #return sparse.coo_matrix((loader['data'], (loader['row'], loader['col'])), shape=loader['shape'], dtype=np.float32)


def load_sparse_list(filename):
    loader = np.load(filename)
    arr_list = []
    for i in range(len(loader.files) / 4):
        arr_list.append(sparse.csc_matrix((loader['data%d' % i], loader['indices%d' % i], loader['indptr%d' % i]),
                                          shape=loader['shape%d' % i]))
        #arr_list.append(sparse.coo_matrix((loader['data%d' % i], (loader['row%d' % i], loader['col%d' % i])),
        #                                  shape=loader['shape%d' % i], dtype=np.float32))
    return arr_list
