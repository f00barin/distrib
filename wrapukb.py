import ukbsim
import sys
import numpy as np
import h5py 

wordset_a = sys.argv[1]
wordset_b = 'all_wordnet.txt'

content_a = [word.strip() for word in open(wordset_a)]
content_b = [word.strip() for word in open(wordset_b)]

truth_mat = np.zeros(shape=(len(content_a), len(content_b)), dtype=np.float64)
ukbsim.storetodisk() 

x = 0

for i in content_a:
    y = 0
    for j in content_b:
        truth_mat[x, y] = ukbsim.dotsimilarity(i,j)
        y += 1
    x += 1

f = h5py.File(wordset_a+'.hdf5', 'w')
f.create_dataset('elements', data=truth_mat)
f.close()






