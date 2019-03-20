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
f = open("questions.txt","r")
fout = open("QUERIES_for_training_Expanded","w", encoding="utf-8")
stop_words=set(stopwords.words("english"))


def stemming(word):
    
    # stemming of words
    porter = PorterStemmer()
    stemmed = porter.stem(word)
         
    return stemmed   

with open('questions.txt',encoding="ISO-8859-1",newline='') as f:
   
    for line in f:
    
        if line and not line.startswith("<"):
            #print(line)
            line=line.replace('\n','')
            wordsList = nltk.word_tokenize(line) 
            filtered_sentence = [w for w in wordsList if not w in stop_words]
            for i in filtered_sentence:
                stemming(i)
            synonyms=[]
            count=0
            for x in filtered_sentence:
                
                for syn in wordnet.synsets(x):
                    for l in syn.lemmas() :
                        if(count<3):
                            if l.name() not in synonyms:
                                synonyms.append(l.name())
                                count+=1
                                
                count=0
            
        synonyms_string=' '.join(synonyms)
        synonyms=[]
        fout.write(synonyms_string)
        fout.write('\n')

		
f.close()
fout.close()


#Source
#https://github.com/ellisa1419/Wordnet-Query-Expansion