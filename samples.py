#!/usr/bin/python

import numpy as np
import scipy.sparse as ss
from tabulate import tabulate 
import itertools

wordlist = 'all_wordnet.txt'
content = [word.strip() for word in open(wordlist)]

def printinformat(sim, score):
#    scoretionary = dict(zip(num, den))
#    s = '; '.join(['%s = %s' % (key, value) for (key, value) in scoretionary.items()])
#    return s
    vallist = []
    for val in range(len(sim)):
        vallist.append(str(sim[val])+' = '+str(score[val]))
    return vallist

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
def find_max(d):
    max_key = len(max(d.keys(), key=len))
    max_val = len(max(list(itertools.chain(*d.values())), key=len)) 
    return max_key, max_val
#
#def pprint(source1, source2=default, source3=default, source4=default):
#    simlist = []
#    for key in source1:
#        ll = []
#        ll.append(key)
#        ll.append(source1[key])
#        if source2 != None:
#            ll.append(source2[key])
#        if source3 != None:
#            ll.append(source3[key])
#        if source4 != None:
#            ll.append(source4[key])
#
#        simlist.append(ll)
#
#    if source4 != None and source3 != None and  source2 != None:
#        print tabulate(simlist, ["word", "source1", "source2", "source3", "source4"], tablefmt="orgtbl")
#    elif source3 != None and  source2 != None:
#        print tabulate(simlist, ["word", "source1", "source2", "source3"], tablefmt="orgtbl")
#    elif source2 != None:
#        print tabulate(simlist, ["word", "source1", "source2"], tablefmt="orgtbl")
#    else:
#        print tabulate(simlist, ["word", "source1"], tablefmt="orgtbl")
#

def pprint(source1, source2=default, source3=default, source4=default):
    if source4 != None:
        print 'nothing found;'
    elif source3 != None:
        max_key, max_val1 = find_max(source1)
        max_key, max_val2 = find_max(source2)
        max_key, max_val3 = find_max(source3)

        listall = [max_key, max_val1, max_val2, max_val3]
        wf = str('WORD').ljust(max_key)
        s1f = str('SOURCE 1').ljust(max_val1)
        s2f = str('SOURCE 2').ljust(max_val2)
        s3f = str('SOURCE 3').ljust(max_val3)
        print "| %s | %s | %s | %s |" % (wf, s1f, s2f, s3f)
        underline = "-+-".join(['-' * x for x in listall])
        print '+-%s-+' % underline

        for key in source1:
            pkey = str(key).ljust(max_key)
            print "| %s |" % pkey
            vals1 = source1[key]
            vals2 = source2[key]
            vals3 = source3[key]

            for val in range(len(vals1)):
                pval1 = str(vals1[val]).ljust(max_val1)
                space = " " * int(max_key)
                pval2 = str(vals2[val]).ljust(max_val2)
                pval3 = str(vals3[val]).ljust(max_val3)
                print "| %s | %s | %s | %s |"  % (space, pval1, pval2, pval3)

            underline = "-+-".join(['-' * col for col in listall])
            print '+-%s-+' % underline

    elif source2 != None:
        max_key, max_val1 = find_max(source1)
        max_key, max_val2 = find_max(source2)
        listall = [max_key, max_val1, max_val2]
        wf = str('WORD').ljust(max_key)
        s1f = str('SOURCE 1').ljust(max_val1)
        s2f = str('SOURCE 2').ljust(max_val2)
        print "| %s | %s | %s |" % (wf, s1f, s2f)
        underline = "-+-".join(['-' * x for x in listall])
        print '+-%s-+' % underline

        for key in source1:
            pkey = str(key).ljust(max_key)
            print "| %s |" % pkey
            vals1 = source1[key]
            vals2 = source2[key]

            for val in range(len(vals1)):
                pval1 = str(vals1[val]).ljust(max_val1)
                space = " " * int(max_key)
                pval2 = str(vals2[val]).ljust(max_val2)
                print "| %s | %s | %s |"  % (space, pval1, pval2)

            underline = "-+-".join(['-' * col for col in listall])
            print '+-%s-+' % underline

    else:
        max_key, max_val = find_max(source1)
        wf = str('WORD').ljust(max_key)
        sf = str('SOURCE 1').ljust(max_val)

        print "| %s | %s |" % (wf, sf)
        underline = "-+-".join(['-' * x for x in find_max(source1)])
        print '+-%s-+' % underline
        for key in source1:
            pkey = str(key).ljust(max_key)
            print "| %s |" % pkey
            for val in source1[key]:
                pval = str(val).ljust(max_val)
                space = " " * int(max_key + 2)
                print "%s | %s |"  % (space, pval)
            underline = "-+-".join(['-' * col for col in find_max(source1)])
            print '+-%s-+' % underline
        

