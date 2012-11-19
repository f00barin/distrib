import sys, fileinput, re
from nltk import trigrams
from nltk.corpus import stopwords, wordnet, wordnet_ic
from recipy import Counter
import numpy as np
import sklearn.preprocessing as sk
import scipy as sp
import scipy.sparse as ss
import scipy.sparse.linalg as ssl
from collections import defaultdict, OrderedDict 
import h5py
from sparsesvd import sparsesvd


Input1 = sys.argv[1]
Input2 = sys.argv[2]
tri_freq = Counter()

hashpref = defaultdict(list)
scorepref = defaultdict(list)
reversehash = defaultdict(list)

def prefsuff():
    arr = []
    for line in fileinput.input([Input1]):
        punctuation = re.compile(r'[-.?!,":;()|0-9]')
        line = punctuation.sub("", line.lower())
        tokens = re.findall(r'\w+', line, flags = re.UNICODE | re.     LOCALE)
        words = filter(lambda x: x not in stopwords.words('english'),  tokens)

        tri_tokens =  trigrams(tokens)

        for tri_token in tri_tokens:
            pref_suff = tri_token[0]+","+tri_token[2]
            tri_tok = pref_suff+"-"+tri_token[1]
            tri_freq[tri_tok] +=1
            
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
            else: value =0
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
        synA = wordnet.synset(i+".n.01")
        for j in content:
            synB = wordnet.synset(j+".n.01")
            sim  = synA.path_similarity(synB)
            similarity.append(sim)
        truth_arr.append(similarity)
    D = ss.csr_matrix(np.array(truth_arr, dtype=np.float64))
    return D

def lch_truth():
    content = [word.strip() for word in open(Input2)]
    truth_arr = []
    for i in content:
        similarity = []
        synA = wordnet.synset(i+".n.01")

        for j in content:
            synB = wordnet.synset(j+".n.01")
            sim  = synA.lch_similarity(synB)
            similarity.append(sim)
        truth_arr.append(similarity)
    D = ss.csr_matrix(np.array(truth_arr, dtype=np.float64))
    return D

def wup_truth():
    content = [word.strip() for word in open(Input2)]
    truth_arr = []
    for i in content:
        similarity = []
        synA = wordnet.synset(i+".n.01")
        for j in content:
            synB = wordnet.synset(j+".n.01")
            sim  = synA.wup_similarity(synB)
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
        synA = wordnet.synset(i+".n.01")
        for j in content:
            synB = wordnet.synset(j+".n.01")
            sim  = synA.jcn_similarity(synB, semcor_ic)
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
        synA = wordnet.synset(i+".n.01")
        for j in content:
            synB = wordnet.synset(j+".n.01")
            sim  = synA.lin_similarity(synB, semcor_ic)
            similarity.append(sim)
        truth_arr.append(similarity)
    D = ss.csr_matrix(np.array(truth_arr, dtype=np.float64))
    return D

def matrix(W,D):
    Winv = ss.csr_matrix(np.linalg.pinv(W.todense()))
    WTinv = ss.csr_matrix(np.linalg.pinv(W.transpose().todense()))
    A = ((Winv*D)*WTinv)
    Result = ((W*A)*(W.transpose().tocsr()))
    return Result, A, A.shape

def semi_matrix(W,A):
#    Winv = ss.csr_matrix(np.linalg.pinv(W.todense()))
#    WTinv = ss.csr_matrix(np.linalg.pinv(W.transpose().todense()))

#    A = ((Winv*D)*WTinv)
    Result = ((W*A)*(W.transpose().tocsr()))
    return Result


def svd_matrix(W,D):
    Winv = ss.csr_matrix(np.linalg.pinv(W.todense()))
    WTinv = ss.csr_matrix(np.linalg.pinv(W.transpose().todense()))
#    A = np.dot(np.dot(Winv, D), WTinv)
    A = ((Winv*D)*WTinv)
    k = np.rank(A)-1
    A = A.tocsc()
    (ut, s, vt) = sparsesvd(A, 337)
#    (u, s, v) = ssl.svds(A,k) 
    print ut.shape, s.shape, vt.shape
    print np.allclose(A.todense(), np.dot(ut.T, np.dot(np.diag(s), vt)))
    print np.allclose(vt, np.dot(ut.T, np.diag(s)))
#    Result = np.dot( np.dot(W,A), W.transpose().tocsr())
#    Result = ((W*A)*(W.transpose().tocsr()))
#    return Result

def simple(W):
    Result = (W * W.transpose())
    return Result

def rank(Input, D, R):
    content = [word.strip() for word in open(Input)]
    k = 0
    rank = tr_rank = []
    ResultMatrix = R.todense()
    TruthMatrix = D.todense()
    for i in content:
        l = 0
        d = {}
        e = {}
        r = 0
        rc = 0
        tr = 0
        trr = 0
        iter = 0
        iterr = 0 
        print i

        print "\t Truth \t\t Calculation" 
        print "\t________________________________"
        for j in content:
            d[str(j)] = ResultMatrix[k,l]
            e[str(j)] = TruthMatrix[k,l]
            l += 1
        C = OrderedDict(reversed(sorted(d.items(), key=lambda t: np.float(t[1])))).keys()

        T = OrderedDict(reversed(sorted(e.items(), key=lambda t: np.float(t[1])))).keys()

        for m in range(0,15):
            print '\t', T[m], '\t\t' ,C[m]

            rc += (C.index(T[m])+1)
            iter += 1
            tr += iter
        print "\t_________________________________"
        print "\t", tr, "\t\t", rc 
        k += 1
        for n in T:
            r += C.index(n)+1
            iterr += 1
            trr += iterr
        rank.append(r/float(len(content)))
        tr_rank.append(trr/float(len(content)))
    Av_rank = sum(rank)/float(len(content))
    TAv_rank = sum(tr_rank)/float(len(content))
    print "Average Rank", Av_rank
    print "Truth Average Rank", TAv_rank

if __name__ == '__main__':
    W = prefsuff()
#    D = truth()
#    D = lch_truth()
#    D = wup_truth()
#    D = jcn_truth()
    D = lin_truth()
#    R, A, S= matrix(W,D)
#    svd_matrix(W,D)



    f = h5py.File('projection.hdf5', 'r')
    dataset = f['D']
    data = np.empty(dataset.shape, dataset.dtype)
    dataset.read_direct(data)
    dataset = f['Ind']
    indices = np.empty(dataset.shape, dataset.dtype)
    dataset.read_direct(indices)
    dataset = f['IP']
    indptr = np.empty(dataset.shape, dataset.dtype)
    dataset.read_direct(indptr)
    dataset = f['S']
    sob = np.empty(dataset.shape, dataset.dtype)
    dataset.read_direct(sob)
    S = (sob[0], sob[1])
    f.close()
    A = ss.csr_matrix((data,indices,indptr), shape=S)
#    R = semi_matrix(W,A)
    R = simple(W)
    
#    f = h5py.File('projection.hdf5', 'w')
#    dset = f.create_dataset('D', data=A.data)
#    dset = f.create_dataset('Ind', data=A.indices)
#    dset = f.create_dataset('IP', data=A.indptr)
#    dset = f.create_dataset('S', data=S)
#    f.close()
    Input = sys.argv[2]
    rank(Input, D, R)
