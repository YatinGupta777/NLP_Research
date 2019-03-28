from nltk.corpus import wordnet
from nltk.tokenize import  word_tokenize
from nltk.corpus import stopwords
import nltk 
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize, sent_tokenize 
from nltk.stem.porter import PorterStemmer
from nltk.corpus import wordnet
import numpy, scipy.io
import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from textblob import Word
import re
import gensim 
from gensim.models import KeyedVectors

# Load vectors directly from the file
model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
G=nx.Graph()
stop_words=set(stopwords.words("english"))


# =============================================================================
# str = "123456790abcdefABCDEF!@#$%^&*()_+<>?,./"
# f = open("questions.txt","r")
# fout = open("QUERIES_for_training_Expanded","w", encoding="utf-8")
# 
# =============================================================================
test_string = "Inventions in science and technology ?"

word_data = []

def stemming(word):
    
    # stemming of words
    porter = PorterStemmer()
    stemmed = porter.stem(word)
         
    return stemmed   
def intersection(lst1, lst2): 
    return list(set(lst1) & set(lst2))
'''Creating bag of words'''
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
#                 word_data.append(x)
# =============================================================================

'''For Test String'''

'''POS TAGGING'''
test_string = test_string.replace('\n','')
''.join([i for i in test_string if i.isalpha()])
wordsList = nltk.word_tokenize(test_string) 
filtered_sentence = [w for w in wordsList if not w in stop_words]
tagged = nltk.pos_tag(wordsList) 
#Adjective,Adverb,verb,noun
tag_list = ["JJ","JJS","JJR","NN","NNS","NNP","NNPS","RB","RBR","RBS","VB","VBD","VBG","VBN","VBP","VBZ","WRB"]
x = []
for i in tagged:
    if(i[1] in tag_list):
        x.append(i[0])
    
filtered_sentence = x
'''STEMMING'''
x = []
#for i in filtered_sentence:
#    x.append(stemming(i))
#filtered_sentence = x

'''Wordnet graph'''

for x in filtered_sentence:
        try:
            word = Word(x)
            if(len(word.synsets) > 1):
                w = word.synsets[1]
            word_data.append(w.name().partition('.')[0])
        
            G.add_node(w.name().partition('.')[0])
            for h in w.hypernyms():
                #print (h)
                word_data.append(h.name().partition('.')[0])
                G.add_node(h.name().partition('.')[0])
                G.add_edge(w.name().partition('.')[0],h.name().partition('.')[0])
                
                for k in h.hypernyms():
                    #print (h)
                    word_data.append(k.name().partition('.')[0])
                    G.add_node(k.name().partition('.')[0])
                    G.add_edge(h.name().partition('.')[0],k.name().partition('.')[0])
                    
                    for j in k.hypernyms():
                        #print (h)
                        word_data.append(j.name().partition('.')[0])
                        G.add_node(j.name().partition('.')[0])
                        G.add_edge(k.name().partition('.')[0],j.name().partition('.')[0])
                        
                    for j in k.hyponyms():
                        #print (h)
                        word_data.append(j.name().partition('.')[0])
                        G.add_node(j.name().partition('.')[0])
                        G.add_edge(k.name().partition('.')[0],j.name().partition('.')[0])   
                    
                for k in h.hyponyms():
                    #print (h)
                    word_data.append(k.name().partition('.')[0])
                    G.add_node(k.name().partition('.')[0])
                    G.add_edge(h.name().partition('.')[0],k.name().partition('.')[0])    
                    
                    for j in k.hypernyms():
                        #print (h)
                        word_data.append(j.name().partition('.')[0])
                        G.add_node(j.name().partition('.')[0])
                        G.add_edge(k.name().partition('.')[0],j.name().partition('.')[0])
                        
                    for j in k.hyponyms():
                        #print (h)
                        word_data.append(j.name().partition('.')[0])
                        G.add_node(j.name().partition('.')[0])
                        G.add_edge(k.name().partition('.')[0],j.name().partition('.')[0])   
                
            for h in w.hyponyms():
                #print (h)
                word_data.append(h.name().partition('.')[0])
                G.add_node(h.name().partition('.')[0])
                G.add_edge(w.name().partition('.')[0],h.name().partition('.')[0])
                        
                for k in h.hypernyms():
                    #print (h)
                    word_data.append(k.name().partition('.')[0])
                    G.add_node(k.name().partition('.')[0])
                    G.add_edge(h.name().partition('.')[0],k.name().partition('.')[0])
                    
                    for j in k.hypernyms():
                        #print (h)
                        word_data.append(j.name().partition('.')[0])
                        G.add_node(j.name().partition('.')[0])
                        G.add_edge(k.name().partition('.')[0],j.name().partition('.')[0])
                        
                    for j in k.hyponyms():
                        #print (h)
                        word_data.append(j.name().partition('.')[0])
                        G.add_node(j.name().partition('.')[0])
                        G.add_edge(k.name().partition('.')[0],j.name().partition('.')[0])   
                    
                for k in h.hyponyms():
                    #print (h)
                    word_data.append(k.name().partition('.')[0])
                    G.add_node(k.name().partition('.')[0])
                    G.add_edge(h.name().partition('.')[0],k.name().partition('.')[0])    
                    
                    for j in k.hypernyms():
                        #print (h)
                        word_data.append(j.name().partition('.')[0])
                        G.add_node(j.name().partition('.')[0])
                        G.add_edge(k.name().partition('.')[0],j.name().partition('.')[0])
                        
                    for j in k.hyponyms():
                        #print (h)
                        word_data.append(j.name().partition('.')[0])
                        G.add_node(j.name().partition('.')[0])
                        G.add_edge(k.name().partition('.')[0],j.name().partition('.')[0])
        except AttributeError as e: 
            continue
for u,v,a in G.edges(data=True):
    try:
        x = model.similarity(u,v)
        G[u][v]['weight'] = abs(x)
    except KeyError:
         continue

for u,v,a in G.edges(data=True):
    print(u,v,a)

bw_centrality = nx.betweenness_centrality(G, normalized=True,weight='weight')
d_centrality = nx.degree_centrality(G)
c_centrality = nx.closeness_centrality(G,distance='weight')
pr = nx.pagerank_numpy(G, alpha=0.9,weight='weight')
hub,authority=nx.hits_numpy(G)


avg_bw = 0
avg_d = 0
avg_c = 0
avg_pr = 0
avg_hub = 0
avg_authority = 0

for i in bw_centrality:
    avg_bw += bw_centrality[i]

avg_bw = avg_bw/len(bw_centrality)

for i in d_centrality:
    avg_d += d_centrality[i]

avg_d = avg_d/len(d_centrality)

for i in c_centrality:
    avg_c += c_centrality[i]

avg_c = avg_c/len(c_centrality)

for i in pr:
    avg_pr += pr[i]

avg_pr= avg_pr/len(pr)

for i in hub:
    avg_hub += hub[i]

avg_hub = avg_hub/len(hub)
for i in authority:
    avg_authority += authority[i]

avg_authority = avg_authority/len(authority)

bw_words = []
d_words = []
c_words = []
pr_words = []
hub_words = []
authority_words = []

for i in bw_centrality:
    if bw_centrality[i] > avg_bw:
        bw_words.append(i)
for i in d_centrality:
    if d_centrality[i] > avg_d:
        d_words.append(i)
for i in c_centrality:
    if c_centrality[i] > avg_c:
        c_words.append(i)
for i in pr:
    if pr[i] > avg_pr:
        pr_words.append(i)
for i in hub:
    if hub[i] > avg_hub:
        hub_words.append(i)
for i in authority:
    if authority[i] > avg_authority:
        authority_words.append(i)        
        
#Itrating thorugh all words      
final_words = []        
for i in bw_centrality:
    count = 0
    if i in bw_words:
        count = count + 1
    if i in d_words:
        count = count + 1
    if i in c_words:
        count = count + 1
    if i in pr_words:
        count = count + 1
    if i in hub_words:
        count = count + 1
    if i in authority_words:
        count = count + 1    
    
    if count >= 3:
        final_words.append(i)
        

final_query = ""

for i in final_words:
    if i not in filtered_sentence:
        final_query += i + " "
print (final_query)
# =============================================================================
# print (bw_centrality)
# print (d_centrality)
# print (c_centrality)
# =============================================================================
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
#                     G.add_edge(w .name(),h.name())
# =============================================================================
            
# =============================================================================
#         synonyms_string=' '.join(synonyms)
#         synonyms=[]
#         fout.write(synonyms_string)
#         fout.write('\n')
# =============================================================================

#print (G.nodes(data=True))
plt.show()
nx.draw(G, width=2, with_labels=True)
#plt.savefig("path.png")


# =============================================================================
# f.close()
# fout.close()
# =============================================================================


#Source
#http://intelligentonlinetools.com/blog/2016/09/05/getting-wordnet-information-and-building-and-building-graph-with-python-and-networkx/
#https://github.com/ellisa1419/Wordnet-Query-Expansion
# =============================================================================
# nyms = ['hypernyms', 'hyponyms', 'meronyms', 'holonyms', 'part_meronyms', 'sisterm_terms', 'troponyms', 'inherited_hypernyms']
# for x in filtered_sentence:
#     for synset in (wordnet.synsets(x)):
#         for i in nyms:
#             try:
#                 print (getattr(synset, i))
#             except AttributeError as e: 
#                 print (e)
#                 continue
# =============================================================================

