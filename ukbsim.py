import numpy as np
import os 
import sys
import bz2

simil = {}
content_set = set([word.strip() for word in open('all_wordnet.txt')])

def storetodisk():
    rootdir = "/home/pranava/local/ukb/UKBsim/wn30g.all.trunc1000.ppv/en/"

    for root, subFolders, files in os.walk(rootdir):
        for file in files:
            try:
                file.split('.n.ppv.bz2')[1]
                word = file.split('.n.ppv.bz2')[0]
                if word in content_set:
                    fl = os.path.join(root, file)
                    upl = bz2.BZ2File(fl)
                    scstack = upl.readlines()
                    scdict = {}
                    for iter in scstack:
                        temp = iter.strip().split('\t')
                        scdict[temp[0]] = temp[1]
                    simil[word] = scdict

            except:
                pass


def dotsimilarity(word1, word2):
    try:
        
        dict1 = simil[word1]
    except:
        return 0
    try:
        dict2 = simil[word2]
    except:
        return 0
    result = 0
    for synset in dict1:
        if synset in dict2:
            result += ((np.float64(dict1[synset])) *
                    np.float64(dict2[synset]))
    
    return round(result, 10)

def cossimilarity(word1, word2):
    try:
        dict1 = simil[word1]
    except:
        return 0
    try:
        dict2 = simil[word2]
    except:
        return 0 

    result = 0

    mod = ((np.array(dict2.values(), dtype=np.float64) ** 
        2).sum() * (np.array(dict2.values(), dtype=np.float64) ** 2).sum())

    for synset in dict1:
        if synset in dict2:
            result += ((np.float64(dict1[synset])) *
            np.float64(dict2[synset]))
    
    return round((result / mod), 10)


