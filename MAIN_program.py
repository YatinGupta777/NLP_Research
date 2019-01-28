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
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
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
    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    n_clusters = ncenters
    silhouette_avg = silhouette_score(alldata, cluster_membership)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(("Silhouette analysis for Fuzzy C Means clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')

plt.show()

fig1.tight_layout()
fig2, ax2 = plt.subplots()
ax2.plot(np.r_[2:11], fpcs)
ax2.set_xlabel("Number of centers")
ax2.set_ylabel("Fuzzy partition coefficient")
