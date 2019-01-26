from nltk import Text
from nltk.corpus import reuters
import nltk 
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize, sent_tokenize 
stop_words = set(stopwords.words('english')) 
 
tokenized = reuters.words()
 
for i in tokenized: 
      
    # Word tokenizers is used to find the words  
    # and punctuation in a string 
    wordsList = nltk.word_tokenize(i) 
  
    # removing stop words from wordList 
    wordsList = [w for w in wordsList if not w in stop_words]  
  
    #  Using a Tagger. Which is part-of-speech  
    # tagger or POS-tagger.  
    tagged = nltk.pos_tag(wordsList) 
  
    print(tagged) 

# Get the collocations that don't contain stop-words
#text.collocations() 
# United States; New York; per cent; Rhode Island; years ago; Los Angeles; White House; ...
 
# Get words that appear in similar contexts
#text.similar('Monday', 5) 
# april march friday february january
 
# Get common contexts for a list of words
#text.common_contexts(['August', 'June']) 
# since_a in_because last_when between_and last_that and_at ...
 
# Get contexts for a word
#text.concordance('Monday')
# said . Trade Minister Saleh said on Monday that Indonesia , as the world ' s s
# Reuters to clarify his statement on Monday in which he said the pact should be
#  the 11 - member CPA which began on Monday . They said producers agreed that c
# ief Burkhard Junger was arrested on Monday on suspicion of embezzlement and of
# ween one and 1 . 25 billion dlrs on Monday and Tuesday . The spokesman said Mo
# ay and Tuesday . The spokesman said Monday ' s float included 500 mln dlrs in 
 