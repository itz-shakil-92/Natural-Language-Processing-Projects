'''Project: SMS Spam Classifier using NLP and Machine learning'''

'''Using Bag of words text preprocessing'''

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


#train test split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=100)


#train model using naive bayes classifier
from sklearn.naive_bayes import MultinomialNB
spam_detect_model=MultinomialNB().fit(X_train, y_train)
y_pred=spam_detect_model.predict(X_test) 


#confusion metrics of the model
from sklearn.metrics import confusion_matrix
confusion_m=confusion_matrix(y_test, y_pred)


#accuracy score of the model
from sklearn.metrics import accuracy_score
accuracy_of_model=accuracy_score(y_test, y_pred)
print(f"The Accuracy score of the model is : {accuracy_of_model}")
   
   
   