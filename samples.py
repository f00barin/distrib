#!/usr/bin/python

import numpy as np
import scipy.sparse as ss
from tabulate import tabulate 


wordlist = 'all_wordnet.txt'
content = [word.strip() for word in open(wordlist)]


def sampledef(matrix, format, words):
    word_dict = {}
    farray = np.array(matrix.todense().argsort(axis=1)[::, ::-1][:,
        :int(words)])
    (rows, cols) = farray.shape
    
    for row in range(rows):
        simlist = []
        for col in range(cols):
            simlist.append(content[farray[row, col]])

        word_dict[content[format[row]]] = ', '.join(map(str, simlist))

    return word_dict
    

def samplespl(matrix, format, words):
    word_dict = {}
    farray = np.array(matrix.todense().argsort(axis=1)[::, ::-1][:,
        :(int(words)+1)])
    (rows, cols) = farray.shape

    for row in range(rows):
        simlist = []
        cands = farray[row]
        try:
            rem = np.where(cands == format[row])[0][0]
            temp_arr = np.delete(cands, rem, axis=0)
            cands = temp_arr
        except: 
            temp_arr = cands[:words]
            cands = temp_arr
            
        for col in range(cols-1):
            simlist.append(content[cands[col]])

        word_dict[content[format[row]]] = ', '.join(map(str, simlist))

    return word_dict
 


default = None

def pprint(source1, source2=default, source3=default):
    simlist = []
    for key in source1:
        ll = []
        ll.append(key)
        ll.append(source1[key])
        if source2 != None:
            ll.append(source2[key])
        if source3 != None:
            ll.append(source3[key])

        simlist.append(ll)

    if source3 != None and  source2 != None:
        print tabulate(simlist, ["word", "source1", "source2", "source3"], tablefmt="grid")
    elif source2 != None:
        print tabulate(simlist, ["word", "source1", "source2"], tablefmt="grid")
    else:
        print tabulate(simlist, ["word", "source1"], tablefmt="grid")








    


