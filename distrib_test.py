import sys
import fileinput
import re
from nltk import trigrams
from nltk.corpus import wordnet, wordnet_ic
from recipy import Counter
import numpy as np
import sklearn.preprocessing as sk
import scipy.sparse as ss
from collections import defaultdict, OrderedDict
import h5py
from sparsesvd import sparsesvd


Input1 = sys.argv[1]
Input2 = sys.argv[2]
Input3 = sys.argv[3]

tri_freq = Counter()

hashpref = defaultdict(list)
scorepref = defaultdict(list)
reversehash = defaultdict(list)


def prefsuff():
    arr = []

    for line in fileinput.input([Input1]):
        punctuation = re.compile(r'[-.?!,":;()|0-9]')
        line = punctuation.sub("", line.lower())
        tokens = re.findall(r'\w+', line, flags=re.UNICODE | re.LOCALE)
        tri_tokens = trigrams(tokens)

        for tri_token in tri_tokens:
            pref_suff = tri_token[0] + "," + tri_token[2]
            tri_tok = pref_suff + "-" + tri_token[1]
            tri_freq[tri_tok] += 1

    fileinput.close()

    combo = list(tri_freq.elements())

    for i in combo:
        word = i.split(r'-')[1]
        prefsuff = i.split(r'-')[0]

        if prefsuff not in hashpref[word]:
            hashpref[word].append(prefsuff)
            scorepref[word].append(tri_freq[i])

        if word not in reversehash[prefsuff]:
            reversehash[prefsuff].append(word)

    content = [word.strip() for word in open(Input2)]

    for i in content:
        rows = []

        for j in reversehash.keys():

            if i in reversehash[j]:
                value = hashpref[i].index(j)

            else:
                value = 0
            rows.append(value)

        arr.append(rows)

    W = sk.normalize(ss.csr_matrix(np.array(arr, dtype=np.float64)), norm='l1',
            axis=1)
    return W


def test_prefsuff():
    arr = []

    for line in fileinput.input([Input1]):
        punctuation = re.compile(r'[-.?!,":;()|0-9]')
        line = punctuation.sub("", line.lower())
        tokens = re.findall(r'\w+', line, flags=re.UNICODE | re.LOCALE)
        tri_tokens = trigrams(tokens)

        for tri_token in tri_tokens:
            pref_suff = tri_token[0] + "," + tri_token[2]
            tri_tok = pref_suff + "-" + tri_token[1]
            tri_freq[tri_tok] += 1

    fileinput.close()

    combo = list(tri_freq.elements())

    for i in combo:
        word = i.split(r'-')[1]
        prefsuff = i.split(r'-')[0]

        if prefsuff not in hashpref[word]:
            hashpref[word].append(prefsuff)
            scorepref[word].append(tri_freq[i])

        if word not in reversehash[prefsuff]:
            reversehash[prefsuff].append(word)

    content = [word.strip() for word in open(Input3)]

    for i in content:
        rows = []

        for j in reversehash.keys():

            if i in reversehash[j]:
                value = hashpref[i].index(j)

            else:
                value = 0
            rows.append(value)

        arr.append(rows)

    W = sk.normalize(ss.csr_matrix(np.array(arr, dtype=np.float64)), norm='l1',
            axis=1)
    return W



def truth():
    content = [word.strip() for word in open(Input2)]
    truth_arr = []

    for i in content:
        similarity = []
        synA = wordnet.synset(i + ".n.01")

        for j in content:
            synB = wordnet.synset(j + ".n.01")
            sim = synA.path_similarity(synB)
            similarity.append(sim)

        truth_arr.append(similarity)

    D = ss.csr_matrix(np.array(truth_arr, dtype=np.float64))
    return D



def test_truth():
    content = [word.strip() for word in open(Input2)]
    test_content = [word.strip() for word in open(Input3)]

    truth_arr = []

    for i in content:
        similarity = []
        synA = wordnet.synset(i + ".n.01")

        for j in test_content:
            synB = wordnet.synset(j + ".n.01")
            sim = synA.path_similarity(synB)
            similarity.append(sim)

        truth_arr.append(similarity)

    D = ss.csr_matrix(np.array(truth_arr, dtype=np.float64))
    return D

def lch_truth():
    content = [word.strip() for word in open(Input2)]
    truth_arr = []

    for i in content:
        similarity = []
        synA = wordnet.synset(i + ".n.01")

        for j in content:
            synB = wordnet.synset(j + ".n.01")
            sim = synA.lch_similarity(synB)
            similarity.append(sim)

        truth_arr.append(similarity)

    D = ss.csr_matrix(np.array(truth_arr, dtype=np.float64))
    return D


def wup_truth():
    content = [word.strip() for word in open(Input2)]
    truth_arr = []

    for i in content:
        similarity = []
        synA = wordnet.synset(i + ".n.01")

        for j in content:
            synB = wordnet.synset(j + ".n.01")
            sim = synA.wup_similarity(synB)
            similarity.append(sim)

        truth_arr.append(similarity)

    D = ss.csr_matrix(np.array(truth_arr, dtype=np.float64))
    return D


def jcn_truth():
    semcor_ic = wordnet_ic.ic('ic-semcor.dat')
    content = [word.strip() for word in open(Input2)]
    truth_arr = []

    for i in content:
        similarity = []
        synA = wordnet.synset(i + ".n.01")

        for j in content:
            synB = wordnet.synset(j + ".n.01")
            sim = synA.jcn_similarity(synB, semcor_ic)
            similarity.append(sim)

        truth_arr.append(similarity)

    D = ss.csr_matrix(np.array(truth_arr, dtype=np.float64))
    return D


def lin_truth():
    semcor_ic = wordnet_ic.ic('ic-semcor.dat')
    content = [word.strip() for word in open(Input2)]
    truth_arr = []

    for i in content:
        similarity = []
        synA = wordnet.synset(i + ".n.01")

        for j in content:
            synB = wordnet.synset(j + ".n.01")
            sim = synA.lin_similarity(synB, semcor_ic)
            similarity.append(sim)

        truth_arr.append(similarity)

    D = ss.csr_matrix(np.array(truth_arr, dtype=np.float64))
    return D


def matrix(W, D):
    Winv = ss.csr_matrix(np.linalg.pinv(W.todense()))
    WTinv = ss.csr_matrix(np.linalg.pinv(W.transpose().todense()))
    A = ((Winv * D) * WTinv)
    Result = ((W * A) * (W.transpose().tocsr()))
    return Result, A, A.shape




def semi_matrix(W, WT, A):
    Result = ((W * A) * (WT.transpose().tocsr()))
    return Result


def fnorm(W, D):
    S = 0
    Diff = (W - D).todense()

    for i in range(0, Diff.shape[0]):

        for j in range(0, Diff.shape[1]):
            S += (Diff[i, j]) * (Diff[i, j])

    Result = np.sqrt(S)
    return Result


def svd_matrix(W, D):
    Winv = ss.csr_matrix(np.linalg.pinv(W.todense()))
    WTinv = ss.csr_matrix(np.linalg.pinv(W.transpose().todense()))
#    A = np.dot(np.dot(Winv, D), WTinv)
    A = ((Winv * D) * WTinv)
    A = A.tocsc()
    res_dict = {}
    old_z = 0

    for k in range(270, 280):
        (ut, s, vt) = sparsesvd(A, k)
        U = ss.csr_matrix(ut.T)
        S = ss.csr_matrix(np.diag(s))
        V = ss.csr_matrix(vt)
        L = (W * U) * (S * V * W.transpose())
        z = U.shape[1]

        if z == old_z:
            break

        else:
            Res = fnorm(L, D)
            res_dict[z] = Res
            Result = OrderedDict(sorted(res_dict.items(),
                key=lambda t: np.float64(t[1])))
            old_z = z

    return Result

def test_svd_matrix(W, WT, D, DT):
    Winv = ss.csr_matrix(np.linalg.pinv(W.todense()))
    WTinv = ss.csr_matrix(np.linalg.pinv(W.transpose().todense()))
#    A = np.dot(np.dot(Winv, D), WTinv)
    A = ((Winv * D) * WTinv)
    A = A.tocsc()
    res_dict = {}
    old_z = 0

    for k in range(270, 280):
        (ut, s, vt) = sparsesvd(A, k)
        U = ss.csr_matrix(ut.T)
        S = ss.csr_matrix(np.diag(s))
        V = ss.csr_matrix(vt)
        L = (W * U) * (S * V * WT.transpose())
        z = U.shape[1]

        if z == old_z:
            break

        else:
            Res = fnorm(L, DT)
            res_dict[z] = Res
            Result = OrderedDict(sorted(res_dict.items(),
                key=lambda t: np.float64(t[1])))
            old_z = z

    return Result


#def l1em(W,D):
            # We are trying to learn the quantity x

#    (u, s, v) = ssl.svds(A,k)

#    Result = np.dot( np.dot(W,A), W.transpose().tocsr())
#    Result = ((W*A)*(W.transpose().tocsr()))
#    return Result


def simple(W):
    Result = (W * W.transpose())
    return Result


def rank(Input1, Input2, D, R):

    f = h5py.File('all_data', 'w')
    content = [word.strip() for word in open(Input)]
    test_content = [word.strip() for word in open(Input2)]
 
    k = 0
    rank = []
    tr_rank = []
    ResultMatrix = R.todense()
    TruthMatrix = D.todense()
    ranking = []
    for i in content:
        l = 0
        d = {}
        e = {}
        r = 0
        rc = 0
        tr = 0 
        iter = 0
        trr = 0
        iterr = 0
#        print i

#        print "\t Truth \t\t Calculation"
#        print "\t________________________________"
        for j in test_content:
            d[str(j)] = ResultMatrix[k, l]
            e[str(j)] = TruthMatrix[k, l]
            l += 1

        C = OrderedDict(reversed(sorted(d.items(), key=lambda
            t: np.float(t[1])))).keys()

        T = OrderedDict(reversed(sorted(e.items(), key=lambda
            t: np.float(t[1])))).keys()
        tar_words = []
        list_rank = 0

        for m in range(0,5):
            print '\t', T[m], '\t\t' ,C[m]
            tar_words.append(C[m])
            rc += (C.index(T[m])+1)

            iter += 1
            tr += iter
            list_rank = float(rc/5.0)
            ranking.append(list_rank)
        print "\t_________________________________"
        print "\t", tr, "\t\t", rc, '\t', list_rank
        k += 1

#        list_ind = []
#        words = []
#
#        for n in T:
#            x = (C.index(n) - T.index(n))
#            r += x ** 2
#            list_ind.append((C.index(n) + 1))
#            words.append(n)
#            iterr += 1
#            trr += iterr

        label_rank = i + '_rank'
        label_words = i + '_words'


        f.create_dataset(label_rank, data=list_rank)
        f.create_dataset(label_words, data=tar_words)

    avg_rank = (float(sum(ranking)/337.0))
    label_avg_rank = '_avg_rank'
    f.create_dataset(label_avg_rank, data=avg_rank)
    print "Average Rank", avg_rank
    f.close()


if __name__ == '__main__':
    W = prefsuff()
    D = truth()
    WT = test_prefsuff()
    DT = test_truth()
#    D = lch_truth()
#    D = wup_truth()
#    D = jcn_truth()
#    D = lin_truth()
#    X, A, S = matrix(W, D)
#    R = svd_matrix(W,D)
#    Z = fnorm(D,D)
##   print Z
#    print R
#    Winv = ss.csr_matrix(np.linalg.pinv(W.todense()))
#    WTinv = ss.csr_matrix(np.linalg.pinv(W.transpose().todense()))
#    X = (((W*Winv)*D))*(WTinv*W.transpose().tocsr())
#    print X.shape
#    Y = fnorm(X,D)
#    Z = fnorm (R,D)
#    print Y,Z

#    f = h5py.File('projection.hdf5', 'r')
#    dataset = f['D']
#    data = np.empty(dataset.shape, dataset.dtype)
#    dataset.read_direct(data)
#    dataset = f['Ind']
#    indices = np.empty(dataset.shape, dataset.dtype)
#    dataset.read_direct(indices)
#    dataset = f['IP']
#    indptr = np.empty(dataset.shape, dataset.dtype)
#    dataset.read_direct(indptr)
#    dataset = f['S']
#    sob = np.empty(dataset.shape, dataset.dtype)
#    dataset.read_direct(sob)
#    S = (sob[0], sob[1])
#    f.close()
#    A = ss.csr_matrix((data,indices,indptr), shape=S)
#    R = semi_matrix(W, WT, A)
#    R = simple(W)

#    f = h5py.File('projection.hdf5', 'w')
#    f.create_dataset('D', data=A.data)
#    f.create_dataset('Ind', data=A.indices)
#    f.create_dataset('IP', data=A.indptr)
#    f.create_dataset('S', data=S)
#    f.close()

    Input = sys.argv[2]
    Input2 = sys.argv[3]
    rank(Input, Input2, DT, R)
