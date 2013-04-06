#!/usr/bin/python

import re
import fileinput
from nltk.probability import *
import nltk


def sortedlist(corpus, wordlist):
    words = nltk.FreqDist()
    for line in fileinput.input([corpus]):
        punctuation = re.compile(r'`\'\\\/[-.?!,":;()|0-9]')
        line = punctuation.sub("", line.lower())
        tokens = re.findall(r'\w+', line, flags=re.UNICODE | re.LOCALE)
        for token in tokens:
            words.inc(token)
    fileinput.close()
    newhash = {}
    for key in wordlist:
        newhash[key] = words[key]

 

    sortedlist = sorted(newhash, key=newhash.get, reverse=True)

    sortedvals = []

    for val in sortedlist:
        sortedvals.append(sortedlist.index(val))

    temp = sortedvals[:11000]

    for i in range(10):
        np.random.shuffle(temp) 

    total = temp[:10000]+sortedvals[11000:]+temp[10000:]

    return total

        
 
