import string
from nltk.corpus import wordnet
from nltk.tokenize import  word_tokenize
from nltk.corpus import stopwords
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
import networkx as nx
from textblob import Word
import re
str = "123456790abcdefABCDEF!@#$%^&*()_+<>?,./"
G=nx.Graph()
f = open("questions.txt","r")
fout = open("QUERIES_for_training_Expanded","w", encoding="utf-8")
stop_words=set(stopwords.words("english"))

test_string = "Homemade glass cleaner?"

def stemming(word):
    
    # stemming of words
    porter = PorterStemmer()
    stemmed = porter.stem(word)
         
    return stemmed   
'''For Test String'''

test_string = test_string.replace('\n','')
''.join([i for i in test_string if i.isalpha()])
wordsList = nltk.word_tokenize(test_string) 
filtered_sentence = [w for w in wordsList if not w in stop_words]
for i in filtered_sentence:
    i = stemming(i)

for x in filtered_sentence:
    word = Word(x)
    if(len(word.synsets) > 1):
        w = word.synsets[1]
    
        G.add_node(w.name())
        for h in w.hypernyms():
            print (h)
            G.add_node(h.name())
            G.add_edge(w.name(),h.name())
            
        for h in w.hyponyms():
            print (h)
            G.add_node(h.name())
            G.add_edge(w.name(),h.name())

'''For Whole file'''
# =============================================================================
# with open('questions.txt',encoding="ISO-8859-1",newline='') as f:
#    
#     for line in f:
#     
#         if line and not line.startswith("<"):
#             #print(line)
#             line=line.replace('\n','')
#             wordsList = nltk.word_tokenize(line) 
#             filtered_sentence = [w for w in wordsList if not w in stop_words]
#             for i in filtered_sentence:
#                 stemming(i)
# 
#             for x in filtered_sentence:
#                 word = Word(x)
#                 if(len(word.synsets) > 1):
#                     w = word.synsets[1]
#                 
#                 G.add_node(w.name())
#                 for h in w.hypernyms():
#                     #print (h)
#                     G.add_node(h.name())
#                     G.add_edge(w.name(),h.name())
#                     
#                 for h in w.hyponyms():
#                     #print (h)
#                     G.add_node(h.name())
#                     G.add_edge(w.name(),h.name())
# =============================================================================
            
# =============================================================================
#         synonyms_string=' '.join(synonyms)
#         synonyms=[]
#         fout.write(synonyms_string)
#         fout.write('\n')
# =============================================================================

print (G.nodes(data=True))
plt.show()
nx.draw(G, width=2, with_labels=True)
#plt.savefig("path.png")


f.close()
fout.close()


#Source
#http://intelligentonlinetools.com/blog/2016/09/05/getting-wordnet-information-and-building-and-building-graph-with-python-and-networkx/
#https://github.com/ellisa1419/Wordnet-Query-Expansion