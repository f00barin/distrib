#!/usr/bin/python

from time import time
import h5py
import dstns
import numpy as np
import argparse
import scipy.sparse as ss 

parser = argparse.ArgumentParser()
parser.add_argument("-ntr", "--numtrain", type=int, help="number of training candidates")
parser.add_argument("-nsub", "--notsub", help="training is not a subset of all the similarity candidates", action="store_true")
parser.add_argument("-ncan", "--numcandidates", type=int, help="number of candidates for similarity score")
parser.add_argument("-nte", "--numtest", type=int, help="number of testing candidates")
parser.add_argument("-nps", "--numprefsuff", type=int, help="number of prefix-suffix pairs")
parser.add_argument("--trinner", help="compute average rank for training set inner product and display", action="store_true")
parser.add_argument("--teinner", help="compute average rank for testing set inner product and display", action="store_true")
parser.add_argument("--svd", help="compute svd matrix and save it", action="store_true")
parser.add_argument("--loadvals", help="if the list of values used as train, candidates, test are in a file then this loads it", action="store_true")
parser.add_argument("--travg", help="save the list of average ranks for a normal training set with dimensionality reduction", action="store_true")
parser.add_argument("--teavg", help="save the list of avergage ranks for a normal testing set with dimensionality reduction", action="store_true")
parser.add_argument("--trpca", help="save the list of average ranks for a normal training set with PCA and dimensionality reduction", action="store_true")
parser.add_argument("--tepca", help="save the list of average ranks for a normal testing set with PCA and dimensionality reduction", action="store_true")
parser.add_argument("--loadsvd", help="load svd matrices from disk", action="store_true")
parser.add_argument("--step", type=int, help="the iteration step for k")
args = parser.parse_args()

##########################################################################
#######Loading Representation#############################################
f = h5py.File('represent.hdf5', 'r')
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
Representation = ss.csr_matrix((data,indices,indptr), shape=shape)
##########################################################################
##########################################################################

#########################################################################
#########Loading true similarity scores##################################
f = h5py.File('truth-sparse.hdf5', 'r')
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
Truth = ss.csr_matrix((data,indices,indptr), shape=shape)
######################################################################

if args.loadvals:

    f = h5py.File('values-used.hdf5', 'r')
    
    dataset = f['values']
    total = np.empty(dataset.shape, dataset.dtype)
    dataset.read_direct(total)
    
    f.close()

else:

    total = range(Representation.shape[0])
    
    for i in range(10):
        np.random.shuffle(total)

    f = h5py.File('values-used.hdf5', 'w')
    
    f.create_dataset('values', data=total)
    
    f.close()


if args.notsub:

    trainarr = total[:args.numtrain]
    candarr = total[args.numtrain:(args.numtrain+args.numcandidates)]
    testarr = total[(args.numtrain+args.numcandidates):(args.numtrain+args.numcandidates+args.numtest)]
    
else:

    trainarr = total[:args.numtrain]
    candarr = total[:args.numcandidates]
    testarr = total[args.numcandidates:(args.numcandidates+args.numtest)]
    

trainmat = Representation[trainarr]
candimat = Representation[candarr]
testmat = Representation[testarr]

trainmat, candimat, testmat = dstns.splicematrix(Representation, trainmat,
        candimat, testmat, args.numprefsuff)

truthtemp = Truth[trainarr]
truthtrain = truthtemp[:, candarr]
del truthtemp

truthtemp = Truth[testarr]
truthtest = truthtemp[:, candarr]
del truthtemp

if args.trinner:
    Ctrinner = dstns.Compute(main_matrix=trainmat, transpose_matrix=candimat,
            truth_matrix=truthtrain)
    trinner_result = Ctrinner.matcal('basic')
    Rtrinner = dstns.Compute(main_matrix=trainmat, transpose_matrix=candimat,
            truth_matrix=truthtrain, result_matrix=trinner_result)
    if args.notsub:
        trinner_avgrank = Rtrinner.test_ranking()
    else:
        trinner_avgrank = Rtrinner.train_ranking()

    print 'training inner product, average rank = ', trinner_avgrank

if args.teinner:
    Cteinner = dstns.Compute(main_matrix=testmat, transpose_matrix=candimat,
            truth_matrix=truthtest)
    teinner_result = Cteinner.matcal('basic')

    Rteinner = dstns.Compute(main_matrix=testmat, transpose_matrix=candimat,
            truth_matrix=truthtest, result_matrix=teinner_result)
    teinner_avgrank = Rteinner.test_ranking()

    print 'testing inner product, average rank = ', teinner_avgrank

if args.svd:
    Cproj = dstns.Compute(main_matrix=trainmat, transpose_matrix=candimat,
            truth_matrix=truthtrain)
    prmatrix, presult, prfresult = Cproj.matcal('regular')
    print 'projection function, frobenious norm', prfresult

    Csvd = dstns.Compute(main_matrix=trainmat, transpose_matrix=candimat,
            truth_matrix=truthtrain, projection_matrix=prmatrix,
            result_matrix=presult)
    UT, S, VT = Csvd.matrixsvd()

    f = h5py.File('svd-dataset.hdf5', 'w')
    f.create_dataset('singular_UT', data=UT)
    f.create_dataset('singular_S', data=S)
    f.create_dataset('singular_VT', data=VT)
    f.close()

if args.travg:
    travg_list = []
    if args.loadsvd:
        f = h5py.File('svd-dataset.hdf5', 'r')
        dataset = f['singular_UT']
        UT = np.empty(dataset.shape, dataset.dtype)
        dataset.read_direct(UT)
        dataset = f['singular_S']
        S = np.empty(dataset.shape, dataset.dtype)
        dataset.read_direct(S)
        dataset = f['singular_VT']
        VT = np.empty(dataset.shape, dataset.dtype)
        dataset.read_direct(VT)
        f.close()

    Ctrdimred = dstns.Compute(main_matrix=trainmat, transpose_matrix=candimat,
            truth_matrix=truthtrain, step=args.step)
    trdimred_result_list = Ctrdimred.dimred(UT, S, VT)

    for i in trdimred_result_list:
        Ctravg = dstns.Compute(main_matrix=trainmat, transpose_matrix=candimat,
                truth_matrix=truthtrain, result_matrix=i)
        if args.notsub:
            travg = Ctravg.test_ranking()
            travg_list.append(travg)
        else:
            travg = Ctravg.train_ranking()
            travg_list.append(travg)

    f = h5py.File('training-dataset.hdf5', 'w')
    f.create_dataset('svd_list_avg_ranks', data=travg_list)
    f.close()

if args.teavg:
    teavg_list = []
    if args.loadsvd:
        f = h5py.File('svd-dataset.hdf5', 'r')
        dataset = f['singular_UT']
        UT = np.empty(dataset.shape, dataset.dtype)
        dataset.read_direct(UT)
        dataset = f['singular_S']
        S = np.empty(dataset.shape, dataset.dtype)
        dataset.read_direct(S)
        dataset = f['singular_VT']
        VT = np.empty(dataset.shape, dataset.dtype)
        dataset.read_direct(VT)
        f.close()

    Ctedimred = dstns.Compute(main_matrix=testmat, transpose_matrix=candimat,
            truth_matrix=truthtest, step=args.step)
    tedimred_result_list = Ctedimred.dimred(UT, S, VT)

    for i in tedimred_result_list:
        Cteavg = dstns.Compute(main_matrix=testmat, transpose_matrix=candimat,
                truth_matrix=truthtest, result_matrix=i)
        teavg = Cteavg.test_ranking()
        teavg_list.append(teavg)

    f = h5py.File('testing-dataset.hdf5', 'w')
    f.create_dataset('svd_list_avg_ranks', data=teavg_list)
    f.close()

if args.trpca:
    trpca_list = []
    Ctrpca = dstns.Compute(main_matrix=trainmat, transpose_matrix=candimat,
            truth_matrix=truthtrain, step=args.step, p='all')
    trpca_result_list = Ctrpca.matrixpca()

    for i in trpca_result_list:
        Ctrpcavg = dstns.Compute(main_matrix=trainmat,
                transpose_matrix=candimat, truth_matrix=truthtrain,
                result_matrix=i)

        if args.notsub:
            trpcavg = Ctrpcavg.test_ranking()
            trpca_list.append(trpcavg)
        else:
            trpcavg = Ctrpcavg.train_ranking()
            trpca_list.append(trpcavg)

    f = h5py.File('training-pca.hdf5', 'w')
    f.create_dataset('svd_list_avg_ranks', data=trpca_list)
    f.close()

if args.tepca:
    tepca_list = []
    Ctepca = dstns.Compute(main_matrix=testmat, transpose_matrix=candimat,
            truth_matrix=truthtest, step=args.step, p='all')
    tepca_result_list = Ctepca.matrixpca()

    for i in tepca_result_list:
        Ctepcavg = dstns.Compute(main_matrix=testmat,
                transpose_matrix=candimat, truth_matrix=truthtest,
                result_matrix=i)

        tepcavg = Ctepcavg.test_ranking()
        tepca_list.append(tepcavg)

    f = h5py.File('testing-pca.hdf5', 'w')
    f.create_dataset('svd_list_avg_ranks', data=tepca_list)
    f.close()

