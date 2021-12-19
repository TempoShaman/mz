import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
import operator
from numpy import array
from numpy import mean
from numpy import cov
from numpy.linalg import eig
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
import string
from collections import defaultdict
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression



isis = pd.read_csv('tweets.csv')


print(isis.head())
print()
print(isis.describe())
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
# Step 1. Basic Analysis

# 1.1 most frequent words used in tweets
## I will preprocess all the tweets to lowercase, remove stopwords such as the, in etc and also stem the words. Also, I wil try to separate hashtags to individual words wherever possible eg. 
## #AmazingDay ---> amazing day
    
def tokenize(text):
    stem = nltk.stem.SnowballStemmer('english')
    text = text.lower()

    for token in nltk.word_tokenize(text):
        if token in string.punctuation: continue
        yield stem.stem(token)

def vectorize(doc):
    features = defaultdict(int)
    for token in tokenize(doc):
        features[token] += 1
    return features



def preprocess(tweet):
    # A number of the tweets start with ENGLISH TRANSLATIONS: so i will remove it 
    tweet = re.sub(r'ENGLISH TRANSLATION:','',tweet)
    #I will also strip the tweets of non-alphabetic characters except #
    tweet = re.sub(r'[^A-Za-z# ]','',tweet)
    
    words = tweet.strip().split()
  
    hashtags = [word for word in words if re.match(r'#',word)!=None]
    words = [word.lower() for word in words if word not in hashtags]
    
    # remove stopwords and stem words using porter stemmer
    p_stem = PorterStemmer()
    words = [p_stem.stem(word.lower()) for word in words if word not in stopwords.words('english')]
    
    for hashtag in hashtags:
        hashtag = re.sub(r'#',hashtag,'')
        words_tag = []
        current_word = ''
        for a in hashtag:
            if a.isupper() and current_word!='':
                words_tag.append(current_word)
                current_word = ''+ a.lower()
            else:
                current_word = current_word + a.lower()
        words_tag.append(current_word)
        words.extend(words_tag)
    words = list(set(words))
    
  
    
    return words



# using the above function I will add another column "wordlist" to the dataframe

isis['wordlist'] = [preprocess(tweet) for tweet in isis['tweets']]



#Plot of frequency of various words used in the tweets

all_words = [word for wordlist in isis['wordlist'] for word in wordlist]


def find_principal_components(n, data):
    pca = PCA(n_components = n)
    principalComponents = pca.fit_transform(data)
    return pd.DataFrame(pca.components_, columns=data.columns)


length_all = len(all_words)
wordcount = dict([(word,all_words.count(word)) for word in set(all_words)])
print(length_all)


wordcount = sorted(wordcount.items(), key = operator.itemgetter(1))
wordcount.reverse()






wordcount = wordcount[2:] 
del wordcount[3]
del wordcount[1]
top20 = wordcount[:20]
top20_words = [word for (word,count) in top20]
top20_freq = [count for (word,count) in top20]
indexes = np.arange(len(top20_words))
width = 0.7
plt.figure(figsize=(15,15))
plt.bar(indexes, top20_freq, width)
plt.xticks(indexes + width/2 , top20_words)
plt.show()

# location analysis 
unique_locations = isis['location'].unique()
unique_counts = dict([(loc,list(isis['location']).count(loc)) for loc in unique_locations])
unique_counts = sorted(unique_counts.items(),key = operator.itemgetter(1))
unique_counts.reverse()
for (loc,counts) in unique_counts:
    print("Location of tweeters: ", loc,counts)

def tweet_subject(tweet):
    tweet = re.sub('ENGLISH TRANSLATION:','',tweet)
    tweet = re.sub('ENGLISH TRANSLATIONS:','',tweet)
    tokenized = nltk.word_tokenize(tweet.lower())
    tagged = nltk.pos_tag(tokenized)
    nouns = [(word) for (word,tag) in tagged if re.match(r'NN',tag)!=None]
    return nouns

isis['tweet_subjects'] = [tweet_subject(tweet) for tweet in isis['tweets']]
#most frequent sujects
all_subjects = [word for wordlist in isis['tweet_subjects'] for word in wordlist]
all_subjects_counts =dict([(word,all_subjects.count(word)) for word in set(all_subjects) ])
all_subjects_counts = sorted(all_subjects_counts.items(), key = operator.itemgetter(1))
all_subjects_counts.reverse()
print('TOTAL UNIQUE SUBJECTS : ', len(all_subjects_counts))
for (a,b) in all_subjects_counts[:30]:
    print(a,b)
    
    #plotting the top 20 most frequent words
all_subjects_counts = all_subjects_counts[3:]
del all_subjects_counts[1]
top20_sub = all_subjects_counts[:20]
top20_words = [word for (word,count) in top20_sub]
top20_freq = [count for (word,count) in top20_sub]
indexes = np.arange(len(top20_words))
width = 0.7
plt.figure(figsize=(20,20))
plt.bar(indexes, top20_freq, width)
plt.xticks(indexes + width/2 , top20_words)
plt.show()