#!/usr/bin/python

import fileinput
import re
from os import remove as rm
from nltk import trigrams, bigrams
from nltk.corpus import wordnet, wordnet_ic
from recipy import Counter
import numpy as np
import sklearn.preprocessing as sk
import scipy.sparse as ss
from collections import defaultdict, OrderedDict
from sparsesvd import sparsesvd
from bisect import bisect_left
from scipy.io import mmwrite
from pysparse import spmatrix


def spmatrixmul(matrix_a, matrix_b):
    # write to disk.
    mmwrite('matrix_a.mtx', matrix_a)
    mmwrite('matrix_b.mtx', matrix_b)
    # read it to form a pysparse spmatrix.
    sp_matrix_a = spmatrix.ll_mat_from_mtx('matrix_a.mtx')
    sp_matrix_b = spmatrix.ll_mat_from_mtx('matrix_b.mtx')
    # multiply the matrices.
    sp_result = spmatrix.matrixmultiply(sp_matrix_a, sp_matrix_b)
    #conversion to scipy sparse matrix
    data, row, col = sp_result.find()
    result = ss.csr_matrix((data, (row, col)), shape=sp_result.shape)
    #deleting files and refreshing memory
    rm('matrix_a.mtx')
    rm('matrix_b.mtx')
    del sp_result, sp_matrix_a, sp_matrix_b, matrix_a, matrix_b

    return result


def pseduoinverse(Mat, precision):
    matrix = Mat.tocsc()
    k = int((precision * matrix.shape[0]) / 100)
    ut, s, vt = sparsesvd(matrix.tocsc(), k)
    UT = ss.csr_matrix(ut)
    SI = ss.csr_matrix(np.diag(1 / s))
    VT = ss.csr_matrix(vt)

    temp_matrix = spmatrixmul(VT.transpose(), SI)
    pinv_matrix = spmatrixmul(temp_matrix, UT)
    del temp_matrix

    temp_matrix = spmatrixmul(UT.transpose(), SI)
    pinv_matrix_t = spmatrixmul(temp_matrix, VT)
    del ut, s, vt, UT, SI, VT, temp_matrix

    return pinv_matrix.tocsr(), pinv_matrix_t.tocsr()


def sparsify(matrix, value):

    WL = matrix.tolil()
    WL_rows, WL_columns = WL.nonzero()
    avg = (float(sum(matrix.data)) / float(len(matrix.data)))
    t_value = (avg * value) / 100

    for i in range(0, len(WL_rows)):

        if WL[(WL_rows[i]), (WL_columns[i])] < t_value:
            WL[(WL_rows[i]), (WL_columns[i])] = 0

    return WL


def cfor(first, test, update):
    while test(first):
        yield first
        first = update(first)


class RemoveCol(object):
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
            if pos == len(rows[i]):
                continue
            elif rows[i][pos] == j:
                rows[i].pop(pos)
                data[i].pop(pos)
                if pos == len(rows[i]):
                    continue
            for pos2 in xrange(pos, len(rows[i])):
                rows[i][pos2] -= 1

        self.lilmatrix._shape = (self.lilmatrix._shape[0],
                self.lilmatrix._shape[1] - 1)
        del rows, data, i, j
        return self.lilmatrix


class Represent(object):
    default = None

    def __init__(self, source, target, **kwargs):

        self.source = source
        self.target = target

        if 'splice' in kwargs:
            self.splice = kwargs['splice']
        else:
            self.splice = 0

        if 'threshold' in kwargs:
            self.threshold = kwargs['threshold']
        else:
            self.threshold = 0

    def splicemat(self, matrix, value):

        WL = matrix.tolil()
        list_sum = WL.sum(axis=0).tolist()[0]
        mean = WL.sum(axis=0).mean()
        splice_value = (mean * value) / 100
        remcol = RemoveCol(WL.tolil())
        j = 0

        while j < len(list_sum):
            col_sum = list_sum[j]

            if col_sum < splice_value:
                remcol.removecol(j)
                list_sum.remove(col_sum)

            else:
                j += 1

        W = sk.normalize(WL.tocsr(), norm='l1', axis=1)
        del WL, matrix

        return W


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
                bi_tok = bi_token[1] + "-" + bi_token[0]
                bi_freq[bi_tok] += 1

        fileinput.close()

        combo = list(bi_freq.elements())

        for i in combo:
            word = i.split(r'-')[1]
            suffix = i.split(r'-')[0]

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

        if self.splice != 0:
            W = self.splicemat(M, self.splice)

        else:
            W = sk.normalize(M.tocsr(), norm='l1', axis=1)

        if self.threshold != 0:
            W = sparsify(W, self.threshold)

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
                bi_tok = bi_token[0] + "-" + bi_token[1]
                bi_freq[bi_tok] += 1

        fileinput.close()

        combo = list(bi_freq.elements())

        for i in combo:
            word = i.split(r'-')[1]
            prefix = i.split(r'-')[0]

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

        if self.splice != 0:
            W = self.splicemat(M, self.splice)

        else:
            W = sk.normalize(M.tocsr(), norm='l1', axis=1)

        if self.threshold != 0:
            W = sparsify(W, self.threshold)

        del hashpref, scorepref, reversehash, bi_freq, bi_tokens, M, content

        return W.tocsr()

    def prefsuff(self):

        tri_freq = Counter()
        hashpref = defaultdict(list)
        scorepref = defaultdict(list)
        reversehash = defaultdict(list)

        for line in fileinput.input(self.source):
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

        if self.splice != 0:
            W = self.splicemat(M, self.splice)

        else:
            W = sk.normalize(M.tocsr(), norm='l1', axis=1)

        if self.threshold != 0:
            W = sparsify(W, self.threshold)

        del hashpref, scorepref, reversehash, tri_freq, tri_tokens, M, content

        return W.tocsr()

    def __del__(self):
        self.free()


class Similarity(object):

    default = None

    def __init__(self, wordset_a, wordset_b=default, threshold=default):

        self.wordset_a = wordset_a
        
        if threshold == None:
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

        truth_mat = ss.lil_matrix((len(content_a), len(content_b)), dtype=np.float64)
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
        
        if self.threshold != 0:
            D = sparsify(truth_mat.tocsr(), self.threshold)
        else:
            D = truth_mat.tocsr()

        del truth_mat, content_a, content_b
        return D

    def lch(self):
        content_a = [word.strip() for word in open(self.wordset_a)]
        content_b = [word.strip() for word in open(self.wordset_b)]

        truth_mat = ss.lil_matrix((len(content_a), len(content_b)), dtype=np.float64)
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

        if self.threshold != 0:
            D = sparsify(truth_mat.tocsr(), self.threshold)
        else:
            D = truth_mat.tocsr()

        del truth_mat, content_a, content_b
        return D

    def wup(self):
        content_a = [word.strip() for word in open(self.wordset_a)]
        content_b = [word.strip() for word in open(self.wordset_b)]

        truth_mat = ss.lil_matrix((len(content_a), len(content_b)), dtype=np.float64)
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

        if self.threshold != 0:
            D = sparsify(truth_mat.tocsr(), self.threshold)
        else:
            D = truth_mat.tocsr()

        del truth_mat, content_a, content_b
        return D

    def jcn(self):
        semcor_ic = wordnet_ic.ic('ic-semcor.dat')
        content_a = [word.strip() for word in open(self.wordset_a)]
        content_b = [word.strip() for word in open(self.wordset_b)]

        truth_mat = ss.lil_matrix((len(content_a), len(content_b)), dtype=np.float64)
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

        if self.threshold != 0:
            D = sparsify(truth_mat.tocsr(), self.threshold)
        else:
            D = truth_mat.tocsr()

        del truth_mat, content_a, content_b
        return D

    def lin(self):
        semcor_ic = wordnet_ic.ic('ic-semcor.dat')
        content_a = [word.strip() for word in open(self.wordset_a)]
        content_b = [word.strip() for word in open(self.wordset_b)]

        truth_mat = ss.lil_matrix((len(content_a), len(content_b)), dtype=np.float64)
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

        if self.threshold != 0:
            D = sparsify(truth_mat.tocsr(), self.threshold)
        else:
            D = truth_mat.tocsr()
        
        del truth_mat, content_a, content_b
        return D

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

    def fnorm(self, value):

        sumsquare = 0
        mat = value.todense()

        for i in range(0, mat.shape[0]):

            for j in range(0, mat.shape[1]):

                sumsquare += ((mat[i, j]) * (mat[i, j]))

        result = np.sqrt(sumsquare)
        return result

    def matcal(self, type):

        if self.are_equal is 'set':

            main_mat_inv, transpose_matrix_inv = pseduoinverse(self.main_matrix, self.precision)
        else:
            main_mat_inv, transpose_val1 = pseduoinverse(self.main_matrix, self.precision)
            transpose_matrix_inv, transpose_val2 = pseduoinverse(self.transpose_matrix, self.precision)

        if type is 'regular':
           # step-by-step multiplication
            temp_matrix = spmatrixmul(self.truth_matrix, transpose_matrix_inv)
            projection_matrix = spmatrixmul(main_mat_inv, temp_matrix)
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

    def matrixsvd(self):
        svd_matrix = self.projection_matrix.tocsc()
        svd_dict = {}
        result_list = []
        (U, S, VT) = sparsesvd(svd_matrix.tocsc(),
                (self.projection_matrix.tocsr().shape[0]))
        rank = U.shape[0]

        for k in cfor(1, lambda i: i <= rank, lambda i: i + 100):

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
            del matrix_result, fresult, difference

        result = OrderedDict(sorted(svd_dict.items(),
                    key=lambda t: np.float64(t[1])))

        return result, result_list, U, S, VT

    def wsvd(self):

        svd_dict = {}
        result_list = []
        if self.are_equal is 'unset':

            mat_ut, mat_s, mat_vt = sparsesvd(self.main_matrix.tocsc(),
                    self.main_matrix.shape[0])
            rank = mat_ut.shape[0]
            mat_utt, mat_st, mat_vtt = sparsesvd(self.transpose_matrix.tocsc(),
                    self.transpose_matrix.shape[0])

            for k in cfor(1, lambda i: i <= rank, lambda i: i + 100):
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

            for k in cfor(1, lambda i: i <= rank, lambda i: i + 100):
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

