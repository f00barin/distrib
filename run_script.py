#!/usr/bin/python

from time import time
import h5py
import dstns
import numpy as np
import argparse
import scipy.sparse as ss 

parser = argparse.ArgumentParser()
parser.add_argument("-ntr", "--numtrain", type=int, help='''number of training candidates''', required=True)
parser.add_argument("-nsub", "--notsub", help='''training is not a subset of all the similarity candidates''', action="store_true")
parser.add_argument("-ncan", "--numcandidates", type=int, help="number of candidates for similarity score", required=True)
parser.add_argument("-nte", "--numtest", type=int, help="number of testing candidates", required=True)
parser.add_argument("-nps", "--numprefsuff", type=int, help="number of prefix-suffix pairs", required=True)
parser.add_argument("--smoothing", type=int, help="smoothing of matrix type, 1 = 'l1', 2 = 'ppmi' and 3 = 'l2' supported", required=True)
parser.add_argument("--loadvals", help="if the list of values used as train, candidates, test are in a file then this loads it", action="store_true")
parser.add_argument("--trinner", help="compute average rank for training set inner product and display", action="store_true")
parser.add_argument("--teinner", help="compute average rank for testing set inner product and display", action="store_true")
parser.add_argument("--svd", help="compute svd matrix and save it", action="store_true")
parser.add_argument("--travg", help="save the list of average ranks for a normal training set with dimensionality reduction", action="store_true")
parser.add_argument("--teavg", help="save the list of avergage ranks for a normal testing set with dimensionality reduction", action="store_true")
parser.add_argument("--trpca", help="save the list of average ranks for a normal training set with PCA and dimensionality reduction", action="store_true")
parser.add_argument("--tepca", help="save the list of average ranks for a normal testing set with PCA and dimensionality reduction", action="store_true")
parser.add_argument("--loadsvd", help="load svd matrices from disk", action="store_true")
parser.add_argument("--trhatavg", help="baseline - matrix_hat avg ranks training", action="store_true")
parser.add_argument("--tehatavg", help="baseline - matrix_hat avg ranks testing", action="store_true")
parser.add_argument("--step", type=int, help="the iteration step for k")
parser.add_argument("--sparsemul", help="multiply using pysparse matrix - good for big matrices", action="store_true")
parser.add_argument("--truthfl", help="the truth file 1 = ukb-dot, 2=ukb-cos, 3=path", type=int)
parser.add_argument("--dumptruth", help="dump used truth matrices", action="store_true")
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
if args.truthfl == 1:
    f = h5py.File('truth-sparse.hdf5', 'r')
elif args.truthfl == 2:
    f = h5py.File('truth-ukb-cos.hdf5', 'r')
elif args.truthfl == 3: 
    f = h5py.File('truth-path.hdf5', 'r')

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
hatmat = Representation[total]

if args.smoothing == 1:
    trainmat, candimat, testmat, hatmat = dstns.l1_splicematrix(Representation, trainmat,
        candimat, testmat, hatmat, args.numprefsuff)
elif args.smoothing == 2:
    trainmat, candimat, testmat, hatmat = dstns.ppmi_splicematrix(Representation,
            trainmat, candimat, testmat, hatmat, args.numprefsuff)
elif args.smoothing == 3:
    trainmat, candimat, testmat, hatmat = dstns.l2_splicematrix(Representation, trainmat,
        candimat, testmat, hatmat, args.numprefsuff)


truthtemp = Truth[trainarr]
truthtrain = truthtemp[:, candarr]
truthtrhat = truthtemp[:, total]
del truthtemp

truthtemp = Truth[testarr]
truthtest = truthtemp[:, candarr]
truthtehat = truthtemp[:, total]
del truthtemp

if args.dumptruth:

    f = h5py.File('truth-training.hdf5', 'w')
    f.create_dataset('data', data=truthtrain.data)
    f.create_dataset('indices', data=truthtrain.indices)
    f.create_dataset('indptr', data=truthtrain.indptr)
    f.create_dataset('shape', data=truthtrain.shape)
    f.close()

    f = h5py.File('truth-testing.hdf5', 'w')
    f.create_dataset('data', data=truthtest.data)
    f.create_dataset('indices', data=truthtest.indices)
    f.create_dataset('indptr', data=truthtest.indptr)
    f.create_dataset('shape', data=truthtest.shape)
    f.close()




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
    if args.sparsemul:
        Ctrdimred = dstns.Compute(main_matrix=trainmat, transpose_matrix=candimat,
                truth_matrix=truthtrain, step=args.step)
        trdimred_result_list = Ctrdimred.spdimred(UT, S, VT)
    else:
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

    sv = np.where(travg_list == np.array(travg_list).min())[0][0]
    f = h5py.File('training-best.hdf5', 'w')
    f.create_dataset('data', data=trdimred_result_list[sv].data)
    f.create_dataset('indices', data=trdimred_result_list[sv].indices)
    f.create_dataset('indptr', data=trdimred_result_list[sv].indptr)
    f.create_dataset('shape', data=trdimred_result_list[sv].shape)
    f.close()

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

    if args.sparsemul:
        Ctedimred = dstns.Compute(main_matrix=testmat, transpose_matrix=candimat,
                truth_matrix=truthtest, step=args.step)
        tedimred_result_list = Ctedimred.spdimred(UT, S, VT)
    else:
        Ctedimred = dstns.Compute(main_matrix=testmat, transpose_matrix=candimat,
                truth_matrix=truthtest, step=args.step)
        tedimred_result_list = Ctedimred.dimred(UT, S, VT)


    for i in tedimred_result_list:
        Cteavg = dstns.Compute(main_matrix=testmat, transpose_matrix=candimat,
                truth_matrix=truthtest, result_matrix=i)
        teavg = Cteavg.test_ranking()
        teavg_list.append(teavg)

    sv = np.where(teavg_list == np.array(teavg_list).min())[0][0]
    f = h5py.File('testing-best.hdf5', 'w')
    f.create_dataset('data', data=tedimred_result_list[sv].data)
    f.create_dataset('indices', data=tedimred_result_list[sv].indices)
    f.create_dataset('indptr', data=tedimred_result_list[sv].indptr)
    f.create_dataset('shape', data=tedimred_result_list[sv].shape)
    f.close()



    f = h5py.File('testing-dataset.hdf5', 'w')
    f.create_dataset('svd_list_avg_ranks', data=teavg_list)
    f.close()

if args.trpca:
    trpca_list = []
    Ctrpca = dstns.Compute(main_matrix=trainmat, transpose_matrix=hatmat,
            truth_matrix=truthtrhat, step=args.step, p='all')
    trpca_result_list = Ctrpca.matrixpca()

    for i in trpca_result_list:
        Ctrpcavg = dstns.Compute(main_matrix=trainmat,
                transpose_matrix=hatmat, truth_matrix=truthtrhat,
                result_matrix=i)

        trpcavg = Ctrpcavg.train_ranking()
        trpca_list.append(trpcavg)

    f = h5py.File('training-pca.hdf5', 'w')
    f.create_dataset('svd_list_avg_ranks', data=trpca_list)
    f.close()

if args.tepca:
    tepca_list = []
    Ctepca = dstns.Compute(main_matrix=testmat, transpose_matrix=hatmat,
            truth_matrix=truthtehat, step=args.step, p='all')
    tepca_result_list = Ctepca.matrixpca()

    for i in tepca_result_list:
        Ctepcavg = dstns.Compute(main_matrix=testmat,
                transpose_matrix=hatmat, truth_matrix=truthtehat,
                result_matrix=i)

        tepcavg = Ctepcavg.train_ranking()
        tepca_list.append(tepcavg)

    f = h5py.File('testing-pca.hdf5', 'w')
    f.create_dataset('svd_list_avg_ranks', data=tepca_list)
    f.close()

if args.trhatavg:
    trhat_list = []

    if args.sparsemul:
            
        Ctrhat = dstns.Compute(main_matrix=trainmat, transpose_matrix=candimat,
                truth_matrix=truthtrain, step=args.step)
        trhat_result_list = Ctrhat.spmatrixhat()
    else:
        Ctrhat = dstns.Compute(main_matrix=trainmat, transpose_matrix=candimat,
                truth_matrix=truthtrain, step=args.step)
        trhat_result_list = Ctrhat.matrixhat()



    for i in trhat_result_list:
        Ctrhatvg = dstns.Compute(main_matrix=trainmat,
                transpose_matrix=candimat, truth_matrix=truthtrain,
                result_matrix=i)

        if args.notsub:
            trhatvg = Ctrhatvg.test_ranking()
            trhat_list.append(trhatvg)
        else:
            trhatvg = Ctrhatvg.train_ranking()
            trhat_list.append(trhatvg)

    sv = np.where(trhat_list == np.array(trhat_list).min())[0][0]
    f = h5py.File('training-hat-best.hdf5', 'w')
    f.create_dataset('data', data=trhat_result_list[sv].data)
    f.create_dataset('indices', data=trhat_result_list[sv].indices)
    f.create_dataset('indptr', data=trhat_result_list[sv].indptr)
    f.create_dataset('shape', data=trhat_result_list[sv].shape)
    f.close()

    f = h5py.File('training-hat.hdf5', 'w')
    f.create_dataset('svd_list_avg_ranks', data=trhat_list)
    f.close()

if args.tehatavg:
    tehat_list = []
    
    if args.sparsemul:

        Ctehat = dstns.Compute(main_matrix=testmat, transpose_matrix=candimat,
                truth_matrix=truthtest, step=args.step, p='all')
        tehat_result_list = Ctehat.spmatrixhat()
    else:
        Ctehat = dstns.Compute(main_matrix=testmat, transpose_matrix=candimat,
                truth_matrix=truthtest, step=args.step, p='all')
        tehat_result_list = Ctehat.matrixhat()



    for i in tehat_result_list:
        Ctehatvg = dstns.Compute(main_matrix=testmat,
                transpose_matrix=candimat, truth_matrix=truthtest,
                result_matrix=i)

        tehatvg = Ctehatvg.test_ranking()
        tehat_list.append(tehatvg)

    sv = np.where(tehat_list == np.array(tehat_list).min())[0][0]
    f = h5py.File('testing-hat-best.hdf5', 'w')
    f.create_dataset('data', data=tehat_result_list[sv].data)
    f.create_dataset('indices', data=tehat_result_list[sv].indices)
    f.create_dataset('indptr', data=tehat_result_list[sv].indptr)
    f.create_dataset('shape', data=tehat_result_list[sv].shape)
    f.close()

    f = h5py.File('testing-hat.hdf5', 'w')
    f.create_dataset('svd_list_avg_ranks', data=tehat_list)
    f.close()

