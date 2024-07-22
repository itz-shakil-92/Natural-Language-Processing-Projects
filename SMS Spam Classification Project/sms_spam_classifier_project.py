# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#importing dataset

import pandas as pd

messages=pd.read_csv('smsspamcollection/SMSSPamCollection',sep='\t',names=['label','message'])


#Data cleaning and preprocessing
import re   #regular expression
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
corpus=[]

for i in range(len(messages)):
    review=re.sub('[^a-zA-Z]',' ',messages['message'][i])
    review=review.lower()
    review=review.split()
    review=[ps.stem(word) for word in review if not word  in stopwords.words('english')]
    review=' '.join(review)
    corpus.append(review)


#Creating the bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv =CountVectorizer(max_features=2500)
X=cv.fit_transform(corpus).toarray()

y=pd.get_dummies(messages['label'])
y=y.iloc[:,1].values
   