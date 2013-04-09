#!/usr/bin/python

import re
import numpy as np
import fileinput
from nltk.probability import *
import nltk

default=0

def sortedlist(corpus, wordlist, cutoff=default):
    words = nltk.FreqDist()
    for line in fileinput.input([corpus]):
        punctuation = re.compile(r'`\'\\\/[-.?!,":;()|0-9]')
        line = punctuation.sub("", line.lower())
        tokens = re.findall(r'\w+', line, flags=re.UNICODE | re.LOCALE)
        for token in tokens:
            words.inc(token)
    fileinput.close()
    hash = {}
    for key in wordlist:
        hash[key] = words[key]
    
    newhash = { k: v for k, v in hash.iteritems() if v > cutoff }
 

    sortedlist = sorted(newhash, key=newhash.get, reverse=True)

    sortedvals = []

    for val in sortedlist:
        sortedvals.append(sortedlist.index(val))

    temp = sortedvals[:11000]

    for i in range(10):
        np.random.shuffle(temp) 

    total = temp[:10000]+sortedvals[11000:]+temp[10000:]

    return total

        
 
