# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from IPython.core.display import Image 

# <headingcell level=1>

# Experimental Details: 

# <markdowncell>

# *About selection of all Nouns: Nouns initially are ordered as they appear in the corpus. This is not done alphabetically but on the order of their occurrence.*
# 
# Training candidates, the scoring candidates and the testing candidates are randomly selected. The prefix and suffix columns are independent of any of the training/scoring/testing candidates and are sorted in a decreasing order for the whole distribution. The splicing function is here: https://github.com/f00barin/distrib/blob/master/dstns.py#L328-L337 - this is approximately matlab style. Here, matrix_a is the entire representation with each row corresponding to nouns that are in the original order and the columns: all the prefix-suffixes with their frequency of occurrence for the noun. 
# 
# The fucntion takes the whole representaion as matrix_a, gets the column sum and then does an argsort on the sum array (index-sorted), reverses it (decreasing order) and cuts the required numbers (the value argument). The resultant array is extracted for training (matrix_b), scoring (matrix_c) and testing (matrxi_d). All these matrices are then normalized with l1 normalization for each row. 
# 
# 
# **Theory:**
# 
# **Computation of bilinear function is:**
# 
# $bilin = \phi(train)^+ . sim_{test}. \phi(candidates)^+$
# 
# where:
# 
# * $sim_{test}$ = Similarity function comprising of similarity scores for 5000 training candidates with 25,446 candidates.
#         
# * $\phi(train)$ = A matrix with frequency scores of training candidates versus best x prefix-suffix word pairs.
# 
# * $\phi(candidates)$ = A matrix with frequency scores of all candidates versus best x prefix-suffix word pairs - transposed.
# 
# 
# **Next we calculate SVD of the bilinear function as:**
# 
# $(U, \Sigma, V^\intercal) = SVD(bilin)$
# 
# $Result_{training} = (\phi(train) . U[:, 1:k]) . (\Sigma_k . V^\intercal[1:k] . \phi(candidates))$
# 
# $Result_{testing} = (\phi(test) . U[:, 1:k]) . (\Sigma_k . V^\intercal[1:k] . \phi(candidates))$
# 
# 
# **Control experiments:**
# 
# *Partial Identity matrix:*
# 
# $sim_{control} = [diag(matrix) = 1]$
# 
# $bilin_{control} = \phi(train)^+ . sim_{control}. \phi(candidates)^+$
# 
# 
# *and $PCA$:*
# 
# $Mat = \frac{\phi(candidates) - mean(\phi(candidates))}{\sqrt{(n-1)}}$
# 
# $Cov = Mat * Mat^\intercal$
# 
# $U, \Sigma , V^\intercal = SVD(Cov)$
# 
# $Result_{(training / testing)} = \phi(train) . V[:, 1:k] . V^\intercal[1:k] . \phi(candidates)$

# <headingcell level=2>

# Results:

# <headingcell level=3>

# Set One

# <headingcell level=4>

# Constants:

# <markdowncell>

# Number of Training candidates = 5,000 ; training candidates $\subset$ scoring candidates.
# 
# Number of scoring candidates = 25,446 
# 
# Number of Testing candidates = 1000 

# <headingcell level=4>

# Variant 1: Prefix-suffix = 1000 

# <markdowncell>

# *Graph 1*: Plot of the training, scoring and testing candidate random distribution. 
# 
# Here, each point represents the candidate in the [training/scoring/testing] set vs the candidate number in the original set. 

# <codecell>

Image(filename='/home/pranava/Downloads/ipython/1000-rand-sub.png') 

# <markdowncell>

# *Graph 2*: Plot of Average rank vs k - truncation constraint for SVD of the bilinear function matrix.
# 

# <codecell>

Image(filename='/home/pranava/Downloads/ipython/1000-avg-sub.png') 

# <headingcell level=4>

# Variant 2: Prefix-suffix = 2500

# <markdowncell>

# *Graph 1*: Plot of the training, scoring and testing candidate random distribution. 
# 
# Here, each point represents the candidate in the [training/scoring/testing] set vs the candidate number in the original set. 

# <codecell>

Image(filename='/home/pranava/Downloads/ipython/2500-rand-sub.png') 

# <markdowncell>

# *Graph 2*: Plot of Average rank vs k - truncation constraint for SVD of the bilinear function matrix.

# <codecell>

Image(filename='/home/pranava/Downloads/ipython/2500-avg-sub.png')

# <headingcell level=4>

# Variant 3: Prefix-suffix = 3500

# <markdowncell>

# *Graph 1*: Plot of the training, scoring and testing candidate random distribution.
# 
# Here, each point represents the candidate in the [training/scoring/testing] set vs the candidate number in the original set.

# <codecell>

Image(filename='/home/pranava/Downloads/ipython/3500-rand-sub.png') 

# <markdowncell>

# *Graph 2*: Plot of Average rank vs k - truncation constraint for SVD of the bilinear function matrix.

# <codecell>

Image(filename='/home/pranava/Downloads/ipython/3500-avg-sub.png')

# <headingcell level=3>

# Set Two

# <headingcell level=4>

# Constants:

# <markdowncell>

# Number of Training candidates = 10,000 ; training candidates $\not \subset$ scoring candidates.
# 
# Number of scoring candidates = 15,446 
# 
# Number of Testing candidates = 1000 

# <headingcell level=4>

# Variant 1: Prefix-suffix = 1000 

# <markdowncell>

# *Graph 1*: Plot of the training, scoring and testing candidate random distribution. 
# 
# Here, each point represents the candidate in the [training/scoring/testing] set vs the candidate number in the original set. 

# <codecell>

Image(filename='/home/pranava/Downloads/ipython/1000-rand-nsub.png') 

# <markdowncell>

# *Graph 2*: Plot of Average rank vs k - truncation constraint for SVD of the bilinear function matrix.

# <codecell>

Image(filename='/home/pranava/Downloads/ipython/1000-avg-nsub.png') 

# <headingcell level=4>

# Variant 2: Prefix-suffix = 2500 

# <markdowncell>

# *Graph 1*: Plot of the training, scoring and testing candidate random distribution. 
# 
# Here, each point represents the candidate in the [training/scoring/testing] set vs the candidate number in the original set. 

# <codecell>

Image(filename='/home/pranava/Downloads/ipython/2500-rand-nsub.png') 

# <markdowncell>

# *Graph 2*: Plot of Average rank vs k - truncation constraint for SVD of the bilinear function matrix.

# <codecell>

Image(filename='/home/pranava/Downloads/ipython/2500-avg-nsub.png') 

# <headingcell level=4>

# Variant 3: Prefix-suffix = 3500 

# <markdowncell>

# *Graph 1*: Plot of the training, scoring and testing candidate random distribution. 
# 
# Here, each point represents the candidate in the [training/scoring/testing] set vs the candidate number in the original set. 

# <codecell>

Image(filename='/home/pranava/Downloads/ipython/3500-rand-nsub.png') 

# <markdowncell>

# *Graph 2*: Plot of Average rank vs k - truncation constraint for SVD of the bilinear function matrix.

# <codecell>

Image(filename='/home/pranava/Downloads/ipython/3500-avg-nsub.png') 

