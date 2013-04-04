#!/usr/bin/python

import numpy as np
import scipy.sparse as ss
from tabulate import tabulate 


wordlist = 'all_wordnet.txt'
content = [word.strip() for word in open(wordlist)]

def printinformat(num, den):
    scoretionary = dict(zip(num, den))
    s = '; '.join(['%s = %s' % (key, value) for (key, value) in scoretionary.items()])
    return s


def sampledef(matrix, format, words, totvalarray):
    word_dict = {}
    farray = np.array(matrix.todense().argsort(axis=1)[::, ::-1][:,
        :int(words)])
    (rows, cols) = farray.shape
    
    for row in range(rows):
        simlist = []
        scorelist = []
        for col in range(cols):

            simlist.append(content[totvalarray[farray[row, col]]])
            scorelist.append(matrix[row, farray[row,col]])

#        word_dict[content[format[row]]] = printinformat(*([', '.join(map(str, simlist)), ', '.join(map(str, scorelist))]))
        word_dict[content[format[row]]] = printinformat(simlist, scorelist)


    return word_dict
    
def samplespl(matrix, format, words, totvalarray):
    word_dict = {}
    farray = np.array(matrix.todense().argsort(axis=1)[::, ::-1][:,
        :(int(words)+1)])
    (rows, cols) = farray.shape

    for row in range(rows):
        simlist = []
        scorelist = []

        cands = farray[row]
        try:
            rem = np.where(cands == format[row])[0][0]
            temp_arr = np.delete(cands, rem, axis=0)
            cands = temp_arr
        except: 
            temp_arr = cands[:words]
            cands = temp_arr
            
        for col in range(cols-1):
            simlist.append(content[totvalarray[cands[col]]])
            scorelist.append(matrix[row, cands[col]])

#        word_dict[content[format[row]]] = printinformat(*([', '.join(map(str, simlist)), ', '.join(map(str, scorelist))]))
        word_dict[content[format[row]]] = printinformat(simlist, scorelist)

    return word_dict
 


default = None

def pprint(source1, source2=default, source3=default, source4=default):
    simlist = []
    for key in source1:
        ll = []
        ll.append(key)
        ll.append(source1[key])
        if source2 != None:
            ll.append(source2[key])
        if source3 != None:
            ll.append(source3[key])
        if source4 != None:
            ll.append(source4[key])

        simlist.append(ll)

    if source4 != None and source3 != None and  source2 != None:
        print tabulate(simlist, ["word", "source1", "source2", "source3", "source4"], tablefmt="orgtbl")
    elif source3 != None and  source2 != None:
        print tabulate(simlist, ["word", "source1", "source2", "source3"], tablefmt="orgtbl")
    elif source2 != None:
        print tabulate(simlist, ["word", "source1", "source2"], tablefmt="orgtbl")
    else:
        print tabulate(simlist, ["word", "source1"], tablefmt="orgtbl")








    


