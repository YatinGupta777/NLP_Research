from nltk import Text
from nltk.corpus import reuters
import nltk 
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize, sent_tokenize 
from nltk.stem.porter import PorterStemmer
from nltk.corpus import wordnet
import collections, re
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
        
for i in bagsofwords:
    print (i) 
