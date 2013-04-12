import numpy as np
import scipy.sparse as ss 

def extractdeps(file):
    depdict = {}
    content = [word.strip() for word in open(file, 'r')]
    for word in content:
        freq = word.split()[0]
        seq = word.split()[1]
        key = seq.split('__OBJ')[0]
        depdict[key] = freq

    return depdict

def extracthm(d):

    m = [x.split('__')[0] for x in d.iterkeys()]
    h = [y.split('__')[1] for y in d.iterkeys()]

    modifiers = list(set(m))
    heads = list(set(h))
    
    return heads, modifiers

def representhm(d, h, m):
    hm = ss.lil_matrix((len(h), len(m)), dtype=np.float64)
    x = 0
    for head in h:
        y = 0
        for mod in m:
            try:
                hm[x,y] = d[mod+'__'+head]

            except: 
                hm[x,y] = 0

            y += 1
        
        x += 1

    return hm.tocsr()
    
    


