import numpy as np
import scipy.sparse as ss

wordlist = 'all_wordnet.txt'
content = [word.strip() for word in open(wordlist)]


default = None
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
 



#def pprintdicts(dict1, dict2=default, dict3=default, dict4=default):




    


