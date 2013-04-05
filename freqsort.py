#!/usr/bin/python

import re
import fileinput
from nltk.probability import *
import nltk


def sortfn(corpus, dictionary):
    words = nltk.FreqDist()
    for line in fileinput.input([corpus]):
        punctuation = re.compile(r'`\'\\\/[-.?!,":;()|0-9]')
        line = punctuation.sub("", line.lower())
        tokens = re.findall(r'\w+', line, flags=re.UNICODE | re.LOCALE)
        for token in tokens:
            words.inc(token)
    fileinput.close()
    newhash = {}
    for key in dictionary:
        newhash[key] = words[key]

    sortedlist = sorted(newhash, key=newhash.get, reverse=True)

    return newhash, sortedlist

        

