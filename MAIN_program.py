from __future__ import division, print_function
from nltk import Text
from nltk.corpus import reuters
import nltk 
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize, sent_tokenize 
from nltk.stem.porter import PorterStemmer
from nltk.corpus import wordnet
import collections, re
import numpy, scipy.io
import os
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
stop_words = set(stopwords.words('english')) 
 
#tokenized = reuters.words()
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()

document_vector = []

def stemming(word):
    
    # stemming of words
    porter = PorterStemmer()
    stemmed = porter.stem(word)
         
    return stemmed   

for fname in reuters.fileids():
    
    tokenized = reuters.words(fname)
    final_string = ""
    for i in tokenized: 
        
        # Word tokenizers is used to find the words  
        # and punctuation in a string 
        wordsList = nltk.word_tokenize(i) 
        
        # removing stop words from wordList 
        wordsList = [w for w in wordsList if not w in stop_words]  
        
        #  Using a Tagger. Which is part-of-speech  
        # tagger or POS-tagger.  
        tagged = nltk.pos_tag(wordsList)
        if tagged:
            word = stemming(tagged[0][0])
            syns = wordnet.synsets(word)
            
            #for x in syns:
             #   final_list.append(x.lexname())
            final_string += tagged[0][0] + " "
        
    document_vector.append(final_string)
        
    '''to produce an output for testing''' 
    if len(document_vector) > 100:
        break
           #print(tagged[0][0])
        #print(tagged) 
bagsofwords = [ collections.Counter(re.findall(r'\w+', txt))
           for txt in document_vector]

m= 0 
for i in bagsofwords:
    temp_dict = {}
    for key in i:
        syns = wordnet.synsets(key)
        temp = i[key]
        for x in syns:
            if x.lexname() in temp_dict:
                temp_dict[x.lexname()] += temp
            else:
                temp_dict[x.lexname()] = temp
    bagsofwords[m] = temp_dict
    m = m + 1
        
#for i in bagsofwords:
 #   print (i)
 
# Creating file for matlab 
#scipy.io.savemat(os.path.expanduser("~/Desktop/arrdata.mat"), mdict={'arr': bagsofwords})

    
'''Fuzzy C Means'''

colors = ['b', 'orange', 'g', 'r', 'c', 'm', 'y', 'k', 'Brown', 'ForestGreen']

# =============================================================================
# # Define three cluster centers
# centers = [[4, 2],
#            [1, 7],
#            [5, 6]]
# 
# =============================================================================
# Define three cluster sigmas in x and y, respectively
# =============================================================================
# sigmas = [[0.8, 0.3],
#           [0.3, 0.5],
#           [1.1, 0.7]]
# 
# =============================================================================
# Generate test data
np.random.seed(42)  # Set seed for reproducibility
xpts = np.zeros(1)
ypts = np.zeros(1)
labels = np.zeros(1)
x = 0

for i in bagsofwords:
    x = x + 1
    for key in i:
        xpts = np.hstack((xpts, x))
        ypts = np.hstack((ypts, i[key]))
        #labels = np.hstack((labels, np.ones(200) * (x-1)))
        
    '''to produce an output for testing''' 
    if x > 100:
        break

# Visualize the test data
# =============================================================================
# fig0, ax0 = plt.subplots()
# for label in xpts:
#     ax0.plot(xpts[label], ypts[label], '.',
#              color=colors[label%8])
# ax0.set_title('Test data: 200 points x3 clusters.')
# 
# =============================================================================
# Set up the loop and plot
fig1, axes1 = plt.subplots(3, 3, figsize=(8, 8))
alldata = np.vstack((xpts, ypts))
fpcs = []

for ncenters, ax in enumerate(axes1.reshape(-1), 2):
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        alldata, ncenters, 2, error=0.005, maxiter=100, init=None)

    # Store fpc values for later
    fpcs.append(fpc)

    # Plot assigned clusters, for each data point in training set
    cluster_membership = np.argmax(u, axis=0)
    for j in range(ncenters):
        ax.plot(xpts[cluster_membership == j],
                ypts[cluster_membership == j], '.', color=colors[j])

    # Mark the center of each fuzzy cluster
    for pt in cntr:
        ax.plot(pt[0], pt[1], 'rs')

    ax.set_title('Centers = {0}; FPC = {1:.2f}'.format(ncenters, fpc))
    ax.axis('off')

fig1.tight_layout()