#!/usr/bin/python
import fileinput
import re
from os import remove as rm
from nltk import trigrams, bigrams
from nltk.corpus import wordnet, wordnet_ic
import numpy as np
import sklearn.preprocessing as sk
import scipy.sparse as ss
from collections import defaultdict, OrderedDict, Counter
from sparsesvd import sparsesvd
from bisect import bisect_left
from scipy.io import mmwrite
from pysparse import spmatrix
import scipy.sparse.linalg as ssl
import h5py
import scikits.learn.utils.extmath as slue
from sklearn.utils.extmath import randomized_svd as fast_svd

def spmatrixmul(matrix_a, matrix_b):
    """
    Sparse Matrix Multiplication using pysparse matrix

    Objective:
    ----------
    To multiply two sparse matrices - relatively dense

    Reason:
    -------
    Scipy.sparse unfortunately has matrix indices with datatype
    int32. While pysparse is more robust and more efficient.

    Process:
    --------
    It saves the scipy matrices to disk in the standard matrix market
    format to the disk. Then reads it to a pysparse format and uses
    Pysparse's inbuilt matrixmultipy operation. The result is
    converted back to a scipy csr matrix.

    This function takes two scipy matrices as input.

    """
    sp_matrix_a = spmatrix.ll_mat(matrix_a.shape[0], matrix_a.shape[1])
    sp_matrix_b = spmatrix.ll_mat(matrix_b.shape[0], matrix_b.shape[1])
    # read it to form a pysparse spmatrix.
    sp_matrix_a.update_add_at(matrix_a.tocoo().data, matrix_a.tocoo().row,
            matrix_a.tocoo().col)
    sp_matrix_b.update_add_at(matrix_b.tocoo().data, matrix_b.tocoo().row,
            matrix_b.tocoo().col)
    # multiply the matrices.
    sp_result = spmatrix.matrixmultiply(sp_matrix_a, sp_matrix_b)
    #conversion to scipy sparse matrix
    data, row, col = sp_result.find()
    result = ss.csr_matrix((data, (row, col)), shape=sp_result.shape)
    #deleting files and refreshing memory
    del sp_result, sp_matrix_a, sp_matrix_b, matrix_a, matrix_b

    return result

   


class Get_truth(object):

    def __init__(self, **kwargs):
        self.hdf5file = kwargs['file']
        self.rows = kwargs['rows']
        self.sparsity = kwargs['sparsity']
        self.trows = kwargs['trows']
        self.name = kwargs['name']
    
    def sparsout(self, matrix):
        for i in range(matrix.shape[0]):
            remval = ((matrix.sum(axis=1)[i] / matrix.shape[1])[0,0] * self.sparsity) / 100
            remlist = (np.where(matrix[i] <= remval))[1].tolist()[0]
            for x in remlist:
                matrix[i,x] = 0

        return matrix
    
    def convert(self, matrix):
        
        psp_mat = spmatrix.ll_mat(matrix.shape[0], matrix.shape[1])
        
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if matrix[i,j]:
                    psp_mat[i,j] = matrix[i,j]

        
        data, row, col = psp_mat.find()
        csrmat = ss.csr_matrix((data, (row, col)), shape=psp_mat.shape)

        return csrmat


    def get_truth(self):
        
        f = h5py.File(self.hdf5file, 'r')
        dataset = f[self.name]
        temp = np.empty(dataset.shape, dataset.dtype)
        dataset.read_direct(temp)
        
        truthmat = np.matrix(temp)
        
        total_col = (truthmat.shape[1] - self.trows) 
        trowstop = (truthmat.shape[0] - self.trows)

        temp_truth = truthmat[0:self.rows, 0:total_col]
        temp_test = truthmat[trowstop:truthmat.shape[1], 0:total_col]

        np_truth = self.sparsout(temp_truth)
        np_test = self.sparsout(temp_test)

        return self.convert(np_truth), self.convert(np_test)

def np_pseudoinverse(Mat):

    result = np.linalg.pinv(Mat.todense()) 
    
    return ss.csr_matrix(np.nan_to_num(result))

def fast_pseudoinverse(matrix, precision):

    if matrix.shape[0] <= matrix.shape[1]:
        val = int((precision * matrix.shape[0]) / 100)
        u, s, vt = slue.fast_svd(matrix, val)
        UT = ss.csr_matrix(np.nan_to_num(u.transpose()))
        SI = ss.csr_matrix(np.nan_to_num(np.diag(1 / s)))
        VT = ss.csr_matrix(np.nan_to_num(vt))

        temp_matrix = spmatrixmul(VT.transpose(), SI)
        pinv_matrix = spmatrixmul(temp_matrix, UT)
        del u, s, vt, UT, SI, VT, temp_matrix

    else:
        val = int((precision * matrix.transpose().shape[0]) / 100)
        u, s, vt = slue.fast_svd(matrix.transpose(), val)
        UT = ss.csr_matrix(np.nan_to_num(u.transpose()))
        SI = ss.csr_matrix(np.nan_to_num(np.diag(1 / s)))
        VT = ss.csr_matrix(np.nan_to_num(vt))

        temp_matrix = spmatrixmul(UT.transpose(), SI)
        pinv_matrix = spmatrixmul(temp_matrix, VT)
        del u, s, vt, UT, SI, VT, temp_matrix



    return pinv_matrix.tocsr()




def pseudoinverse(Mat, precision):
    """
    Pseudoinverse computation.

    Objective:
    ----------
    To compute pseudoinverse using Singular Value Depcomposition

    Reason:
    -------
    SVD using Scipy is slow and consumes a lot of memory, similarly
    pysparse matrix consumes a lot of memory. This is a better
    alternative to a direct computation of inverse.

    Process:
    --------
    The function uses sparsesvd to compute the SVD of a sparse matrix,
    there is a precision attached in the function, this controls the
    cutting (or the k) of the SVD. Precision is actually a percentage
    and uses this to get the k.

        k = (Precision/100) * rows of the matrix.


    The function takes a sparse matrix and a precision score as the input.

    """
    matrix = Mat.tocsc()
    if matrix.shape[0] <= matrix.shape[1]:

        k = int((precision * matrix.shape[0]) / 100)
        ut, s, vt = sparsesvd(matrix.tocsc(), k)
        UT = ss.csr_matrix(ut)
        SI = ss.csr_matrix(np.diag(1 / s))
        VT = ss.csr_matrix(vt)

        temp_matrix = spmatrixmul(VT.transpose(), SI)
        pinv_matrix = spmatrixmul(temp_matrix, UT)
        del ut, s, vt, UT, SI, VT, temp_matrix

    else:

        k = int((precision * matrix.transpose().shape[0]) / 100)
        ut, s, vt = sparsesvd(matrix.transpose().tocsc(), k)
        UT = ss.csr_matrix(ut)
        SI = ss.csr_matrix(np.diag(1 / s))
        VT = ss.csr_matrix(vt)

        temp_matrix = spmatrixmul(UT.transpose(), SI)
        pinv_matrix = spmatrixmul(temp_matrix, VT)
        del ut, s, vt, UT, SI, VT, temp_matrix

    return pinv_matrix.tocsr()

def psp_pseudoinverse(Mat, precision):

    list_nz = (Mat.sum(axis=1) == 1) 
    list_mat = []
    
    for i in range(list_nz):
        if list_nz[i]:
            list_mat.append(i)
    
    temp_Mat = Mat[list_mat, :]
    matrix = spmatrix.ll_mat(temp_Mat.shape[0], temp_Mat.shape[1])
    matrix.update_add_at(temp_Mat.tocoo().data, temp_Mat.tocoo().row,
            temp_Mat.tocoo().col)

    if matrix.shape[0] <= matrix.shape[1]:

        k = int((precision * matrix.shape[0]) / 100)
        ut, s, vt = sparsesvd(matrix.tocsc(), k)
        UT = ss.csr_matrix(ut)
        SI = ss.csr_matrix(np.diag(1 / s))
        VT = ss.csr_matrix(vt)

        temp_matrix = spmatrixmul(VT.transpose(), SI)
        pinv_matrix = spmatrixmul(temp_matrix, UT)
        del ut, s, vt, UT, SI, VT, temp_matrix

    else:

        k = int((precision * matrix.transpose().shape[0]) / 100)
        ut, s, vt = sparsesvd(matrix.transpose().tocsc(), k)
        UT = ss.csr_matrix(ut)
        SI = ss.csr_matrix(np.diag(1 / s))
        VT = ss.csr_matrix(vt)

        temp_matrix = spmatrixmul(UT.transpose(), SI)
        pinv_matrix = spmatrixmul(temp_matrix, VT)
        del ut, s, vt, UT, SI, VT, temp_matrix

    return pinv_matrix.tocsr()




def sci_pseudoinverse(Mat, precision):
    """
    Pseudoinverse computation.
    pseudoinverse using scipy.
    The function takes a sparse matrix and a precision score as the input.

    """
    matrix = Mat.tocsc()
    if matrix.shape[0] <= matrix.shape[1]:
        val = int((precision * matrix.shape[0]) / 100)
        u, s, vt = ssl.svds(matrix.tocsc(), k=val)
        UT = ss.csr_matrix(np.nan_to_num(u.transpose()))
        SI = ss.csr_matrix(np.nan_to_num(np.diag(1 / s)))
        VT = ss.csr_matrix(np.nan_to_num(vt))

        temp_matrix = spmatrixmul(VT.transpose(), SI)
        pinv_matrix = spmatrixmul(temp_matrix, UT)
        del u, s, vt, UT, SI, VT, temp_matrix

    else:
        val = int((precision * matrix.transpose().shape[0]) / 100)
        u, s, vt = ssl.svds(matrix.transpose().tocsc(), k=val)
        UT = ss.csr_matrix(np.nan_to_num(u.transpose()))
        SI = ss.csr_matrix(np.nan_to_num(np.diag(1 / s)))
        VT = ss.csr_matrix(np.nan_to_num(vt))

        temp_matrix = spmatrixmul(UT.transpose(), SI)
        pinv_matrix = spmatrixmul(temp_matrix, VT)
        del u, s, vt, UT, SI, VT, temp_matrix



    return pinv_matrix.tocsr()




def cfor(first, test, update):
    """
    Function that imitates for loop in gnu-c and c++

    Function requires: value to be initilaized, condition and update type.
    """
    while test(first):
        yield first
        first = update(first)


class RemoveCol(object):
    """
    Removing columns from the matrix

    Objective:
    ----------
    To remove columns from the matrix.
    """
    def __init__(self, lilmatrix):
        self.lilmatrix = lilmatrix

    def removecol(self, j):
        if j < 0:
            j += self.lilmatrix.shape[1]

        if j < 0 or j >= self.lilmatrix.shape[1]:
            raise IndexError('column index out of bounds')

        rows = self.lilmatrix.rows
        data = self.lilmatrix.data
        for i in xrange(self.lilmatrix.shape[0]):
            pos = bisect_left(rows[i], j)
            if pos is len(rows[i]):
                continue
            elif rows[i][pos] is j:
                rows[i].pop(pos)
                data[i].pop(pos)
                if pos is len(rows[i]):
                    continue
            for pos2 in xrange(pos, len(rows[i])):
                rows[i][pos2] -= 1

        self.lilmatrix._shape = (self.lilmatrix._shape[0],
                self.lilmatrix._shape[1] - 1)
        del rows, data, i, j
        return self.lilmatrix

def splicematrix(matrix_a, matrix_b, matrix_c, value): 

    ''' the matrix_a should be the WT or it should contain all the rows with
    maximum row elements for maximum profit :P
    '''
    retain_array = np.array(matrix_a.tocsc().sum(axis=0).tolist()[0]).argsort()[::-1][:value]

    return sk.normalize(matrix_a.tocsc()[:,retain_array].tocsr(), norm='l1', axis=1), sk.normalize(matrix_b.tocsc()[:,retain_array].tocsr(), norm='l1', axis=1), sk.normalize(matrix_c.tocsc()[:,retain_array].tocsr(), norm='l1', axis=1)


def sparsify(m, value=100):

    matrix = m.tolil()
    rows, columns = matrix.shape
    sparseindex = matrix.mean(axis=1) * (value/float(100))
    for r in range(rows):
        z = np.where(matrix.tocsr()[r].todense() < sparseindex[r])[1]
        for c in range(z.shape[1]):
            var = int(z[0,c])
            matrix[r, var] =  0

    return matrix.tocsr()


 


def old_splicematrix(matrix_a, matrix_b, matrix_c, value):

    A = matrix_a.tolil()
    B = matrix_b.tolil()
    C = matrix_c.tolil()
    listx = A.sum(axis=0).argsort().tolist()[0]
    
    remcol_a = RemoveCol(A.tolil())
    remcol_b = RemoveCol(B.tolil())
    remcol_c = RemoveCol(C.tolil())
    j = 0

    while j < len(list_sum_a) and j < len(list_sum_b) and j < len(list_sum_c):
        col_sum_a = list_sum_a[j]
        col_sum_b = list_sum_b[j]
        col_sum_c = list_sum_c[j]

        if col_sum_a <= splice_value and col_sum_b <= splice_value and col_sum_c <= splice_value:
            remcol_a.removecol(j)
            remcol_b.removecol(j)
            remcol_c.removecol(j)
            list_sum_a.remove(col_sum_a)
            list_sum_b.remove(col_sum_b)
            list_sum_c.remove(col_sum_c)

        else:
            j += 1

    return A, B, C


class Represent(object):

    default = None

    def __init__(self, source, target, **kwargs):

        self.source = source
        self.target = target
        
        if 'total_prefsuffs' in kwargs:
            self.total_prefsuffs = kwargs['total_prefsuffs']
        else:
            self.total_prefsuffs = 0

        if 'threshold' in kwargs:
            self.threshold = kwargs['threshold']
        else:
            self.threshold = 0

    def suffix(self):

        bi_freq = Counter()
        hashpref = defaultdict(list)
        scorepref = defaultdict(list)
        reversehash = defaultdict(list)

        for line in fileinput.input(self.source):
            punctuation = re.compile(r'[-.?!,":;()|0-9]')
            line = punctuation.sub("", line.lower())
            tokens = re.findall(r'\w+', line, flags=re.UNICODE | re.LOCALE)
            bi_tokens = bigrams(tokens)

            for bi_token in bi_tokens:
                bi_tok = bi_token[1] + ':1:' + bi_token[0]
                bi_freq[bi_tok] += 1

        fileinput.close()

        combo = list(bi_freq.elements())

        for i in combo:
            word = i.split(r':1:')[1]
            suffix = i.split(r':1:')[0]

            if suffix not in hashpref[word]:
                hashpref[word].append(suffix)
                scorepref[word].append(bi_freq[i])

            if word not in reversehash[suffix]:
                reversehash[suffix].append(word)

        content = [word.strip() for word in open(self.target)]
        M = ss.lil_matrix((len(content), len(reversehash.keys())), dtype=np.float64)
        x = 0

        for i in content:
            y = 0

            for j in reversehash.keys():

                if i in reversehash[j]:
                    value = hashpref[i].index(j)
                else:
                    value = 0

                M[x, y] = value
                y += 1

            x += 1

        W = sk.normalize(M.tocsr(), norm='l1', axis=1)


        del hashpref, scorepref, reversehash, bi_freq, bi_tokens, M, content

        return W.tocsr()

    def prefix(self):

        bi_freq = Counter()
        hashpref = defaultdict(list)
        scorepref = defaultdict(list)
        reversehash = defaultdict(list)

        for line in fileinput.input(self.source):
            punctuation = re.compile(r'[-.?!,":;()|0-9]')
            line = punctuation.sub("", line.lower())
            tokens = re.findall(r'\w+', line, flags=re.UNICODE | re.LOCALE)
            bi_tokens = bigrams(tokens)

            for bi_token in bi_tokens:
                bi_tok = bi_token[0] + ':1:' + bi_token[1]
                bi_freq[bi_tok] += 1

        fileinput.close()

        combo = list(bi_freq.elements())

        for i in combo:
            word = i.split(r':1:')[1]
            prefix = i.split(r':1:')[0]

            if prefix not in hashpref[word]:
                hashpref[word].append(prefix)
                scorepref[word].append(bi_freq[i])

            if word not in reversehash[prefix]:
                reversehash[prefix].append(word)

        content = [word.strip() for word in open(self.target)]
        M = ss.lil_matrix((len(content), len(reversehash.keys())), dtype=np.float64)
        x = 0

        for i in content:
            y = 0

            for j in reversehash.keys():

                if i in reversehash[j]:
                    value = hashpref[i].index(j)
                else:
                    value = 0

                M[x, y] = value
                y += 1

            x += 1

        W = sk.normalize(M.tocsr(), norm='l1', axis=1)


        del hashpref, scorepref, reversehash, bi_freq, bi_tokens, M, content

        return W.tocsr()

    def prefsuff(self):

        tri_freq = Counter()
        content = set(word.strip() for word in open(self.target))

        for line in fileinput.input(self.source):
            punctuation = re.compile(r'[-.?!,":;()|0-9]')
            line = punctuation.sub("", line.lower())
            tokens = re.findall(r'\w+', line, flags=re.UNICODE | re.LOCALE)
            tokens_set = set(tokens)
            intersection = content.intersection(tokens_set)

            if intersection:
                tri_tokens = trigrams(tokens)
                for tri_token in tri_tokens:
                    if tri_token[1] in content:
                        pref_suff = tri_token[0] + "," + tri_token[2]
                        tri_tok = pref_suff + ':1:' + tri_token[1]
                        tri_freq[tri_tok] += 1
                        

        fileinput.close()
        return tri_freq

    def oldremovex(self,tri_freq):

        revhash = defaultdict(list)

        for i in list(tri_freq.elements()):
            word = i.split(r':1:')[1]
            prefsuff = i.split(r':1:')[0]

            if word not in revhash[prefsuff]:
                revhash[prefsuff].append(word)

        removable = (len(revhash.keys()) - self.total_prefsuffs)
        sorted_reversehash = sorted(revhash.iteritems(), key=lambda x: len(x[1]), reverse=True)

        temp_val = 0

        if removable != len(revhash.keys()):
            
            while temp_val < removable:
                popped = sorted_reversehash.pop()
                rem = popped[0]+':1:*'
                poplist = filter(lambda name: re.match(rem, name),
                        tri_freq.iterkeys())
                for pop_element in poplist:
                    tri_freq.pop(pop_element)

                temp_val += 1

        return tri_freq

    def represent_ps(self, trifreq):

        hashpref = defaultdict(list)
        scorepref = defaultdict(list)
        reversehash = defaultdict(list)

        content = [word.strip() for word in open(self.target)]

        for i in list(trifreq.elements()):
            (prefsuff, word) = i.split(r':1:')
            if prefsuff not in hashpref[word]:
                hashpref[word].append(prefsuff)
                scorepref[word].append(trifreq[i])

            if word not in reversehash[prefsuff]:
                reversehash[prefsuff].append(word)
        print 'done with getting the hashpref and scorepref'
        M = ss.lil_matrix((len(content), len(reversehash.keys())), dtype=np.float64)
        x = 0

        revkeylist = reversehash.keys()
        
        for fword in content:
            flist = hashpref[fword]
            for i in flist:
                y = revkeylist.index(i)
                pos = hashpref[fword].index(i)
                value = scorepref[fword][pos]
                M[x, y] = value

            x += 1



        return M.tocsr()

    def __del__(self):
        self.free()


class Similarity(object):

    default = None

    def __init__(self, wordset_a, wordset_b=default, threshold=default):

        self.wordset_a = wordset_a

        if threshold is None:
            self.threshold = 0
        else:
            self.threshold = threshold

        if wordset_b is None:
            self.wordset_b = wordset_a
        else:
            self.wordset_b = wordset_b

    def path(self):
        content_a = [word.strip() for word in open(self.wordset_a)]
        content_b = [word.strip() for word in open(self.wordset_b)]

        truth_mat = np.zeros(shape=(len(content_a), len(content_b)))
        x = 0

        for i in content_a:
            y = 0
            synA = wordnet.synset(i + ".n.01")

            for j in content_b:

                synB = wordnet.synset(j + ".n.01")
                sim = synA.path_similarity(synB)
                truth_mat[x, y] = sim
                y += 1

            x += 1
        
        return truth_mat
                
#
#        del truth_mat, content_a, content_b
#        return D

    def lch(self):
        content_a = [word.strip() for word in open(self.wordset_a)]
        content_b = [word.strip() for word in open(self.wordset_b)]

        truth_mat = np.zeros(shape=(len(content_a), len(content_b)))
        x = 0

        for i in content_a:
            y = 0
            synA = wordnet.synset(i + ".n.01")

            for j in content_b:

                synB = wordnet.synset(j + ".n.01")
                sim = synA.lch_similarity(synB)
                truth_mat[x, y] = sim
                y += 1

            x += 1


        return truth_mat

    def wup(self):
        content_a = [word.strip() for word in open(self.wordset_a)]
        content_b = [word.strip() for word in open(self.wordset_b)]

        truth_mat = np.zeros(shape=(len(content_a), len(content_b)))

        x = 0
        for i in content_a:
            y = 0
            synA = wordnet.synset(i + ".n.01")

            for j in content_b:

                synB = wordnet.synset(j + ".n.01")
                sim = synA.wup_similarity(synB)
                truth_mat[x, y] = sim
                y += 1
            x += 1
        return truth_mat


    def jcn(self):
        semcor_ic = wordnet_ic.ic('ic-semcor.dat')
        content_a = [word.strip() for word in open(self.wordset_a)]
        content_b = [word.strip() for word in open(self.wordset_b)]

        truth_mat = np.zeros(shape=(len(content_a), len(content_b)))

        x = 0

        for i in content_a:
            y = 0
            synA = wordnet.synset(i + ".n.01")

            for j in content_b:

                synB = wordnet.synset(j + ".n.01")
                sim = synA.jcn_similarity(synB, semcor_ic)
                truth_mat[x, y] = sim
                y += 1
            x += 1

        return truth_mat

    def lin(self):
        semcor_ic = wordnet_ic.ic('ic-semcor.dat')
        content_a = [word.strip() for word in open(self.wordset_a)]
        content_b = [word.strip() for word in open(self.wordset_b)]

        truth_mat = np.zeros(shape=(len(content_a), len(content_b)))
       
        x = 0
        for i in content_a:
            y = 0
            synA = wordnet.synset(i + ".n.01")

            for j in content_b:

                synB = wordnet.synset(j + ".n.01")
                sim = synA.lin_similarity(synB, semcor_ic)
                truth_mat[x, y] = sim
                y += 1

            x += 1


        return truth_mat

    def random_sim(self):
        content_a = [word.strip() for word in open(self.wordset_a)]
        content_b = [word.strip() for word in open(self.wordset_b)]

        D = ss.rand(len(content_a), len(content_b), density=0.1, format='csr', dtype=np.float64)
        return D

    def __del__(self):
        self.free()


class Compute(object):

    default = None

    def __init__(self, **kwargs):

        if 'main_matrix' in kwargs:
            self.main_matrix = kwargs['main_matrix']
        if 'test_matrix' in kwargs:
            self.test_matrix = kwargs['test_matrix']


        
        if 'precision' in kwargs:
            self.precision = kwargs['precision']
        else:
            self.precision = 100

        if 'transpose_matrix' in kwargs:
            self.transpose_matrix = kwargs['transpose_matrix'].transpose()
            self.are_equal = 'unset'
        else:
            self.transpose_matrix = self.main_matrix.transpose()
            self.are_equal = 'set'

        if 'truth_matrix' in kwargs:
            self.truth_matrix = kwargs['truth_matrix']

        if 'projection_matrix' in kwargs:
            self.projection_matrix = kwargs['projection_matrix']

        if 'wordset_a' in kwargs:
            self.wordset_a = kwargs['wordset_a']

        if 'wordset_b' in kwargs:
            self.wordset_b = kwargs['wordset_b']
        else:
            self.wordset_b = self.wordset_a

        if 'result_matrix' in kwargs:
            self.result_matrix = kwargs['result_matrix']

        if 'svd' in kwargs:
            self.svd = kwargs['svd']
        else:
            self.svd = None

    def fnorm(self, value):

        sumsquare = 0
        mat = value.todense()

        for i in range(0, mat.shape[0]):

            for j in range(0, mat.shape[1]):

                sumsquare += ((mat[i, j]) * (mat[i, j]))

        result = np.sqrt(sumsquare)
        return result

    def matcal(self, type):

        global main_mat_inv 
        global transpose_matrix_inv

        if type is 'regular':

            if self.svd is 'scipy':

                if self.are_equal is 'set':

                    main_mat_inv = sci_pseudoinverse(self.main_matrix, self.precision)

                else:

                    main_mat_inv = sci_pseudoinverse(self.main_matrix, self.precision)
                    transpose_matrix_inv = sci_pseudoinverse(self.transpose_matrix, self.precision)
            elif self.svd is 'sparsesvd':
                print 'here'

                if self.are_equal is 'set':
                    main_mat_inv = pseudoinverse(self.main_matrix, self.precision)
                else:
                    main_mat_inv = pseudoinverse(self.main_matrix, self.precision)
                    transpose_matrix_inv = pseudoinverse(self.transpose_matrix, self.precision)

            elif self.svd is 'fast':

                if self.are_equal is 'set':

                    main_mat_inv = fast_pseudoinverse(self.main_matrix, self.precision)

                else:

                    main_mat_inv = fast_pseudoinverse(self.main_matrix, self.precision)
                    transpose_matrix_inv = fast_pseudoinverse(self.transpose_matrix, self.precision)
        

            else:

                main_mat_inv = np_pseudoinverse(self.main_matrix)
                transpose_matrix_inv = np_pseudoinverse(self.transpose_matrix)

           # step-by-step multiplication
            temp_matrix = spmatrixmul(self.truth_matrix, transpose_matrix_inv)
            print 'got the transpose_matrix_inv' 

            projection_matrix = spmatrixmul(main_mat_inv, temp_matrix)
            print 'got the main_mat_inv' 

            del temp_matrix

            temp_matrix = spmatrixmul(self.main_matrix, projection_matrix)
            result = spmatrixmul(temp_matrix, self.transpose_matrix.tocsr())
            del temp_matrix

            difference = (result - self.truth_matrix)
            fresult = self.fnorm(difference)

            return projection_matrix, result, fresult

        elif type is 'basic':

            result = spmatrixmul(self.main_matrix, self.transpose_matrix.tocsr())

            difference = (result - self.truth_matrix)
            fresult = self.fnorm(difference)
            return result, fresult

        elif type is 'testing':

            temp_matrix = spmatrixmul(self.main_matrix, self.projection_matrix)
            result = spmatrixmul(temp_matrix, self.transpose_matrix.tocsr())
            del temp_matrix

            difference = (result - self.truth_matrix)
            fresult = self.fnorm(difference)
            return result, fresult
        elif type is 'identity':
            o = np.ones(self.truth_matrix.shape[0])
            identity_matrix = ss.lil_matrix(self.truth_matrix.shape)
            identity_matrix.setdiag(o)

           
            temp_matrix = spmatrixmul(identity_matrix.tocsr(), transpose_matrix_inv)
            print 'got the transpose_matrix_inv' 

            projection_matrix = spmatrixmul(main_mat_inv, temp_matrix)
            print 'got the main_mat_inv' 

            del temp_matrix

            temp_matrix = spmatrixmul(self.main_matrix, projection_matrix)
            result = spmatrixmul(temp_matrix, self.transpose_matrix.tocsr())
            del temp_matrix

            difference = (result - self.truth_matrix)
            fresult = self.fnorm(difference)

            return projection_matrix, result, fresult

    def usvmatrix(self, U, S, VT):

        svd_dict = {}
        result_list = []
        rank = U.shape[0]

        for k in cfor(1, lambda j: j <= rank, lambda j: j + 25):

            ut = U[:k]
            s = S[:k]
            vt = VT[:k]
            matrix_u = ss.csr_matrix(ut.T)
            matrix_s = ss.csr_matrix(np.diag(s))
            matrix_vt = ss.csr_matrix(vt)

            temp_matrix = spmatrixmul(self.main_matrix, matrix_u)
            temp_matrix_a = spmatrixmul(matrix_s, matrix_vt)
            temp_matrix_b = spmatrixmul(temp_matrix_a, self.transpose_matrix.tocsr())
            matrix_result = spmatrixmul(temp_matrix, temp_matrix_b)
            del temp_matrix, temp_matrix_a, temp_matrix_b

            result_list.append(matrix_result)
            difference = (matrix_result - self.truth_matrix)
            fresult = self.fnorm(difference)
            svd_dict[k] = fresult
            print 'k = ', k, 'fresult = ', fresult
            del matrix_result, fresult, difference

        return svd_dict, result_list

    def matrixsvd(self):
        svd_matrix = self.projection_matrix.tocsc()
        svd_dict = {}
        result_list = []

        if self.svd is 'scipy':
            Utemp, Stemp, VTtemp = ssl.svds(svd_matrix.tocsc(),
                    k=(int (self.projection_matrix.tocsr().shape[0] *
                        self.precision)/100))

            U = np.nan_to_num(Utemp.transpose())
            S = np.nan_to_num(Stemp)
            VT = np.nan_to_num(VTtemp)

        elif self.svd is 'sparsesvd':
            (U, S, VT) = sparsesvd(svd_matrix, (int (svd_matrix.shape[0] * self.precision)/100))

        elif self.svd is 'fast':

            Utemp, Stemp, VTtemp = fast_svd(svd_matrix,
                    (int (self.projection_matrix.tocsr().shape[0] *
                        self.precision)/100))

            U = np.nan_to_num(Utemp.transpose())
            S = np.nan_to_num(Stemp)
            VT = np.nan_to_num(VTtemp)

        else: 

            Utemp, Stemp, VTtemp = np.linalg.svd(svd_matrix.todense())

            U = np.nan_to_num(Utemp.transpose())
            S = np.nan_to_num(Stemp)
            VT = np.nan_to_num(VTtemp)



        


        rank = U.shape[0]

        for k in cfor(1, lambda i: i <= rank, lambda i: i + 25):

            ut = U[:k]
            s = S[:k]
            vt = VT[:k]
            matrix_u = ss.csr_matrix(ut.T)
            matrix_s = ss.csr_matrix(np.diag(s))
            matrix_vt = ss.csr_matrix(vt)

            temp_matrix = spmatrixmul(self.main_matrix, matrix_u)
            temp_matrix_a = spmatrixmul(matrix_s, matrix_vt)
            temp_matrix_b = spmatrixmul(temp_matrix_a, self.transpose_matrix.tocsr())
            matrix_result = spmatrixmul(temp_matrix, temp_matrix_b)
            del temp_matrix, temp_matrix_a, temp_matrix_b

            result_list.append(matrix_result)
            difference = (matrix_result - self.truth_matrix)
            fresult = self.fnorm(difference)
            svd_dict[k] = fresult
            print 'k = ', k, 'fresult = ', fresult
            del matrix_result, fresult, difference

        return svd_dict, result_list, U, S, VT

    def pca(self):
        svd_dict = {}
        result_list = []


#        U, S, VT = sparsesvd(spmatrixmul(self.transpose_matrix.transpose(),
#            self.transpose_matrix).tocsc(),
#            (int(self.precision*self.transpose_matrix.shape[0])/100))

        U, S, VT = sparsesvd(spmatrixmul(self.transpose_matrix.transpose()).tocsc(), (int(self.precision*self.transpose_matrix.shape[0])/100))

        
        for k in cfor(1, lambda i: i < U.shape[1], lambda i: i+25):
            ut = U[:k]
            matrix_u = ss.csr_matrix(ut.T)
            matrix_ut = ss.csr_matrix(ut)

            temp_matrix   = spmatrixmul(self.test_matrix, matrix_u)
            temp_matrix_a = spmatrixmul(matrix_ut,
                    self.transpose_matrix.transpose())
            matrix_result = spmatrixmul(temp_matrix, temp_matrix_a)
            del temp_matrix, temp_matrix_a

            result_list.append(matrix_result)
            difference = (matrix_result - self.truth_matrix)
            fresult = self.fnorm(difference)
            svd_dict[k] = fresult
            print 'k = ', k, 'fresult = ', fresult
            del matrix_result, fresult, difference

        return svd_dict, result_list, U, S, VT

    def wsvd(self):

        svd_dict = {}
        result_list = []
        if self.are_equal is 'unset':

            mat_ut, mat_s, mat_vt = sparsesvd(self.main_matrix.tocsc(),
                    self.main_matrix.shape[0])
            rank = mat_ut.shape[0]
            mat_utt, mat_st, mat_vtt = sparsesvd(self.transpose_matrix.tocsc(),
                    self.transpose_matrix.shape[0])

            for k in cfor(1, lambda i: i <= rank, lambda i: i + 25):
                ut = mat_ut[:k]
                s = mat_s[:k]
                vt = mat_vt[:k]
                utt = mat_utt[:k]
                st = mat_st[:k]
                vtt = mat_vtt[:k]
                UT = ss.csr_matrix(ut)
                SI = ss.csr_matrix(np.diag(1 / s))
                VT = ss.csr_matrix(vt)
                UTT = ss.csr_matrix(utt)
                SIT = ss.csr_matrix(np.diag(1 / st))
                VTT = ss.csr_matrix(vtt)

                temp_matrix_a = spmatrixmul(VT.transpose(), SI)
                temp_matrix_b = spmatrixmul(temp_matrix_a, UT)
                temp_matrix_c = spmatrixmul(temp_matrix_b, self.truth_matrix)
                temp_matrix_d = spmatrixmul(VTT.transpose(), SIT)
                temp_matrix_e = spmatrixmul(temp_matrix_d, UTT)

                projection_func = spmatrixmul(temp_matrix_c, temp_matrix_e)
                del temp_matrix_a, temp_matrix_b, temp_matrix_c, temp_matrix_d, temp_matrix_e

                temp_matrix = spmatrixmul(self.main_matrix, projection_func)
                matrix_result = spmatrixmul(temp_matrix, self.transpose_matrix.tocsr())
                del temp_matrix

                result_list.append(matrix_result)

                difference = (matrix_result - self.truth_matrix)
                fresult = self.fnorm(difference)
                svd_dict[k] = fresult
                del matrix_result, difference, fresult, ut, s, vt, utt, st, vtt, UT, SI, VT, UTT, SIT, VTT, projection_func

        else:
            mat_ut, mat_s, mat_vt = sparsesvd(self.main_matrix.tocsc(),
                    self.main_matrix.shape[0])
            rank = mat_ut.shape[0]

            for k in cfor(1, lambda i: i <= rank, lambda i: i + 25):
                ut = mat_ut[:k]
                s = mat_s[:k]
                vt = mat_vt[:k]
                UT = ss.csr_matrix(ut)
                SI = ss.csr_matrix(np.diag(1 / s))
                VT = ss.csr_matrix(vt)

                temp_matrix_a = spmatrixmul(VT.transpose(), SI)
                temp_matrix_b = spmatrixmul(temp_matrix_a, UT)
                temp_matrix_c = spmatrixmul(temp_matrix_b, self.truth_matrix)
                temp_matrix_d = spmatrixmul(UT.transpose(), SI)
                temp_matrix_e = spmatrixmul(temp_matrix_d, VT)

                projection_func = spmatrixmul(temp_matrix_c, temp_matrix_e)
                del temp_matrix_a, temp_matrix_b, temp_matrix_c, temp_matrix_d, temp_matrix_e

                temp_matrix = spmatrixmul(self.main_matrix, projection_func)
                matrix_result = spmatrixmul(temp_matrix, self.transpose_matrix.tocsr())
                del temp_matrix

                result_list.append(matrix_result)
                difference = (matrix_result - self.truth_matrix)
                fresult = self.fnorm(difference)
                svd_dict[k] = fresult
                del matrix_result, difference, fresult, ut, s, UT, SI, VT, projection_func

        result = OrderedDict(sorted(svd_dict.items(),
                    key=lambda t: np.float64(t[1])))

        return result, result_list

    def ranking(self):
        content_a = [word.strip() for word in open(self.wordset_a)]
        content_b = [word.strip() for word in open(self.wordset_b)]

        result_matrix = self.result_matrix.todense()
        truth_matrix = self.truth_matrix.todense()
        row = 0
        targets = []
        rankings = []
        result_word_list = []
        truth_word_list = []

        for i in content_a:
            targets.append(i)
            column = 0
            result_dict = {}
            truth_dict = {}

            for j in content_b:

                result_dict[str(j)] = result_matrix[row, column]
                truth_dict[str(j)] = truth_matrix[row, column]
                column += 1

            result_sort = OrderedDict(reversed(sorted(result_dict.items(),
                key=lambda t: np.float(t[1])))).keys()

            truth_sort = OrderedDict(reversed(sorted(truth_dict.items(),
                key=lambda t: np.float(t[1])))).keys()

            result_words = []
            truth_words = []
            iteration = 0
            rank = 0
            rank_count = 0
            tr_rank = 0

            for l in range(0, 10):

                result_words.append(result_sort[l])
                truth_words.append(truth_sort[l])
                rank_count += (result_sort.index(truth_sort[l]) + 1)
                iteration += 1
                tr_rank += iteration

            rank = float(rank_count / 10.0)
            reference = float(tr_rank / 10.0)
            result_word_list.append(result_words)
            truth_word_list.append(truth_words)
            rankings.append(rank)

            row += 1

        avg_rank = (float(sum(rankings) / len(rankings)))

        return reference, avg_rank, rankings, result_word_list, truth_word_list, targets

    def __del__(self):
        self.free()

