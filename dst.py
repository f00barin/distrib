#!/usr/bin/python

import fileinput
import re
from nltk import trigrams
from nltk.corpus import wordnet, wordnet_ic
from recipy import Counter
import numpy as np
import sklearn.preprocessing as sk
import scipy.sparse as ss
from collections import defaultdict, OrderedDict
from sparsesvd import sparsesvd


class Represent(object):

    def __init__(self, source, target):
        self.source = source
        self.target = target

    def prefsuff(self):

        arr = []
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

        for i in content:
            rows = []

            for j in reversehash.keys():

                if i in reversehash[j]:
                    value = hashpref[i].index(j)
                else:
                    value = 0

                rows.append(value)

            arr.append(rows)

        W = sk.normalize(ss.csr_matrix(np.array(arr, dtype=np.float64)),
        norm='l1', axis=1)
        return W


class Similarity(object):

    default = None

    def __init__(self, wordset_a, wordset_b=default):

        self.wordset_a = wordset_a

        if wordset_b is None:
            self.wordset_b = wordset_a
        else:
            self.wordset_b = wordset_b

    def path(self):
        content_a = [word.strip() for word in open(self.wordset_a)]
        content_b = [word.strip() for word in open(self.wordset_b)]

        truth_arr = []

        for i in content_a:
            similarity = []
            synA = wordnet.synset(i + ".n.01")

            for j in content_b:

                synB = wordnet.synset(j + ".n.01")
                sim = synA.path_similarity(synB)
                similarity.append(sim)

            truth_arr.append(similarity)

        D = ss.csr_matrix(np.array(truth_arr, dtype=np.float64))
        return D

    def lch(self):
        content_a = [word.strip() for word in open(self.wordset_a)]
        content_b = [word.strip() for word in open(self.wordset_b)]

        truth_arr = []

        for i in content_a:
            similarity = []
            synA = wordnet.synset(i + ".n.01")

            for j in content_b:

                synB = wordnet.synset(j + ".n.01")
                sim = synA.lch_similarity(synB)
                similarity.append(sim)

            truth_arr.append(similarity)

        D = ss.csr_matrix(np.array(truth_arr, dtype=np.float64))
        return D

    def wup(self):
        content_a = [word.strip() for word in open(self.wordset_a)]
        content_b = [word.strip() for word in open(self.wordset_b)]

        truth_arr = []

        for i in content_a:
            similarity = []
            synA = wordnet.synset(i + ".n.01")

            for j in content_b:

                synB = wordnet.synset(j + ".n.01")
                sim = synA.wup_similarity(synB)
                similarity.append(sim)

            truth_arr.append(similarity)

        D = ss.csr_matrix(np.array(truth_arr, dtype=np.float64))
        return D

    def jcn(self):
        semcor_ic = wordnet_ic.ic('ic-semcor.dat')
        content_a = [word.strip() for word in open(self.wordset_a)]
        content_b = [word.strip() for word in open(self.wordset_b)]

        truth_arr = []

        for i in content_a:
            similarity = []
            synA = wordnet.synset(i + ".n.01")

            for j in content_b:

                synB = wordnet.synset(j + ".n.01")
                sim = synA.jcn_similarity(synB, semcor_ic)
                similarity.append(sim)

            truth_arr.append(similarity)

        D = ss.csr_matrix(np.array(truth_arr, dtype=np.float64))
        return D

    def lin(self):
        semcor_ic = wordnet_ic.ic('ic-semcor.dat')
        content_a = [word.strip() for word in open(self.wordset_a)]
        content_b = [word.strip() for word in open(self.wordset_b)]

        truth_arr = []

        for i in content_a:
            similarity = []
            synA = wordnet.synset(i + ".n.01")

            for j in content_b:
                
                synB = wordnet.synset(j + ".n.01")
                sim = synA.lin_similarity(synB, semcor_ic)
                similarity.append(sim)

            truth_arr.append(similarity)

        D = ss.csr_matrix(np.array(truth_arr, dtype=np.float64))
        return D


class Compute(object):

    default = None

    def __init__(self, **kwargs):

        if 'main_matrix' in kwargs:
            self.main_matrix = kwargs['main_matrix']

        if 'transpose_matrix' in kwargs:
            self.transpose_matrix = kwargs['transpose_matrix'].transpose()
        else:
            self.transpose_matrix = self.main_matrix.transpose()

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

        main_mat_inv = ss.csr_matrix(np.linalg.pinv
        (self.main_matrix.todense()))

        transpose_matrix_inv = ss.csr_matrix(np.linalg.pinv
        (self.transpose_matrix.todense()))

        if type is 'regular':

            projection_matrix = ((main_mat_inv * self.truth_matrix) *
            transpose_matrix_inv)
            result = ((self.main_matrix * projection_matrix) *
                    self.transpose_matrix.tocsr())
            difference = (result - self.truth_matrix)
            fresult = self.fnorm(difference)
            return projection_matrix, projection_matrix.shape, result, fresult

        elif type is 'basic':

            result = (self.main_matrix * self.transpose_matrix.tocsr())
            difference = (result - self.truth_matrix)
            fresult = self.fnorm(difference)
            return result, fresult

        elif type is 'testing':

            result = ((self.main_matrix * self.projection_matrix) *
                    self.transpose_matrix.tocsr())
            difference = (result - self.truth_matrix)
            fresult = self.fnorm(difference)
            return result, fresult

    def matrixsvd(self):

        svd_matrix = self.projection_matrix.tocsc()
        svd_dict = {}
        result_list = []
        old_z = 0
#            (temp1, temp2, temp3) = sparsesvd(svd_matrix,
#            (svd_matrix.shape[0] - 1))
#            rank = (temp1.shape[0] - 1)

#            for k in range((rank - 20), rank):
        for k in range(279, 280):

            (ut, s, vt) = sparsesvd(svd_matrix, k)
            matrix_u = ss.csr_matrix(ut.T)
            matrix_s = ss.csr_matrix(np.diag(s))
            matrix_vt = ss.csr_matrix(vt)
            matrix_result = ((self.main_matrix * matrix_u) * (matrix_s *
                matrix_vt * self.transpose_matrix))
            result_list.append(matrix_result)
            z = matrix_u.shape[0]

            if z == old_z:
                break

            else:
                difference = (matrix_result - self.truth_matrix)
                fresult = self.fnorm(difference)
                svd_dict[z] = fresult
                old_z = z

        result = OrderedDict(sorted(svd_dict.items(),
                    key=lambda t: np.float64(t[1])))
        return result, result_list

    def ranking(self):
        content_a = [word.strip() for word in open(self.wordset_a)]
        content_b = [word.strip() for word in open(self.wordset_b)]

        result_matrix = self.result_matrix.todense()
        truth_matrix = self.truth_matrix.todense()
        row = 0
        rankings = []
        result_word_list = []
        truth_word_list = []

        for i in content_a:
            column = 0
            result_dict = {}
            truth_dict = {}
            print "\n\n", i,"\n\n"

            for j in content_b:

                result_dict[str(j)] = result_matrix[row, column]
                truth_dict[str(j)] = truth_matrix[row, column]
                column += 1

            result_sort = OrderedDict(reversed(sorted(result_dict.items(),
                key=lambda t: np.float(t[1])))).keys()

            truth_sort = OrderedDict(reversed(sorted(truth_dict.items(),
                key=lambda t: np.float(t[1])))).keys()
#            print result_sort
#            print truth_sort
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

        avg_rank = (float(sum(rankings)/len(rankings)))
#        return reference, result_sequence,
#        truth_sequence, rankings, avg_rank
        return reference

