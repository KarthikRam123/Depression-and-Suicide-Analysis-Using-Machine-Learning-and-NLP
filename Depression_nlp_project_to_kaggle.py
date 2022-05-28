#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
from nltk import re
import numpy as np
import keras.layers
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D, Conv1D, Activation
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
import html
import string
import time
import nltk
from keras.callbacks import EarlyStopping,TensorBoard,ModelCheckpoint
import seaborn as sns


# In[3]:


import csv
df = pd.read_csv('Cleaned_Depression_Vs_Suicide.csv', lineterminator = '\n')


# In[3]:


df


# In[4]:


df.dropna(axis=0,inplace=True)


# In[5]:


df.isnull().sum()


# In[6]:


df.describe() 


# In[7]:


def convert_lower(text):
    lower_text = text.lower()
    return lower_text

df["text"] = df['text'].apply(lambda x: convert_lower(x))

# removing punctuation

def remove_punctuations(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text

df["text"] = df['text'].apply(remove_punctuations)

# removing numbers

df['text'] = df['text'].str.replace('\d+', '')


# In[8]:


nltk.download('punkt')


# In[9]:


#tokenization

df['tokenized_text'] = df.apply(lambda row: nltk.word_tokenize(row['text']), axis=1)


# In[10]:


nltk.download('stopwords')


# In[11]:


# removing stopwrods

stopwords = nltk.corpus.stopwords.words("english")

def stopwords_remove(text):
    text_cleaned = [word for word in text if word not in stopwords]
    return text_cleaned

df["tokenized_text"] = df["tokenized_text"].apply(lambda x: stopwords_remove(x))


# In[12]:


from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("english")

df['stemmed'] = df['tokenized_text'].apply(lambda x: [stemmer.stem(y) for y in x])


# In[13]:


df


# In[14]:


input2_corrected = [" ".join(x) for x in df['stemmed']]

from sklearn.feature_extraction.text import TfidfVectorizer 

tfidf_vectorizer=TfidfVectorizer(use_idf=True) 
 

tfidf_vectorizer_vectors=tfidf_vectorizer.fit_transform(input2_corrected)


# In[15]:


tfidf_vectorizer_vectors


# In[16]:


def dummies(x):
    if x == 'SuicideWatch':
        return 1
    if x == 'depression':
        return 2

df['class'] = df['class'].apply(lambda x: dummies(x))


# In[17]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(tfidf_vectorizer_vectors, df['class'], test_size=0.3, random_state=101)


# In[18]:


df


# In[19]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver='sag')

lr.fit(X_train,y_train)
logistic_predictions = lr.predict(X_test)


# In[20]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(y_test,logistic_predictions))
print(classification_report(y_test,logistic_predictions))
print(accuracy_score(y_test, logistic_predictions))


# In[21]:


from sklearn.naive_bayes import MultinomialNB
MNB = MultinomialNB()

MNB.fit(X_train, y_train)
predicted = MNB.predict(X_test)


# In[22]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(y_test,predicted))
print(classification_report(y_test,predicted))
print(accuracy_score(y_test, predicted))

