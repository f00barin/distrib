import h5py
import numpy as np
import scipy.sparse as ss


def exttruthtest(mfname, vfname):
    f = h5py.File(mfname, 'r')
    dataset = f['data']
    data = np.empty(dataset.shape, dataset.dtype)
    dataset.read_direct(data)
    dataset = f['indices']
    indices = np.empty(dataset.shape, dataset.dtype)
    dataset.read_direct(indices)
    dataset = f['indptr']
    indptr = np.empty(dataset.shape, dataset.dtype)
    dataset.read_direct(indptr)
    dataset = f['shape']
    sob = np.empty(dataset.shape, dataset.dtype)
    dataset.read_direct(sob)
    shape = (sob[0], sob[1])
    f.close()
    truth = ss.csr_matrix((data,indices,indptr), shape=shape)


    f = h5py.File(vfname, 'r')
    dataset = f['values']
    total = np.empty(dataset.shape, dataset.dtype)
    dataset.read_direct(total)
    f.close()

    trainarr = total[:10000]
    candarr = total[:25446]
    testarr = total[25446:]

    ttruth = truth[testarr]
    return ttruth[:, total], testarr


def extractvals(vfname):
    f = h5py.File(vfname, 'r')
    dataset = f['values']
    total = np.empty(dataset.shape, dataset.dtype)
    dataset.read_direct(total)
    testarr = total[(len(total)-1000):]
    f.close()
    
    return total, testarr


def extractmat(mfname):

    f = h5py.File(mfname, 'r')
    dataset = f['data']
    data = np.empty(dataset.shape, dataset.dtype)
    dataset.read_direct(data)
    dataset = f['indices']
    indices = np.empty(dataset.shape, dataset.dtype)
    dataset.read_direct(indices)
    dataset = f['indptr']
    indptr = np.empty(dataset.shape, dataset.dtype)
    dataset.read_direct(indptr)
    dataset = f['shape']
    sob = np.empty(dataset.shape, dataset.dtype)
    dataset.read_direct(sob)
    shape = (sob[0], sob[1])
    f.close()
    return ss.csr_matrix((data,indices,indptr), shape=shape)





    







