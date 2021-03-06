import re
import numpy as np
import sys
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer

new_file2 = []
new_file3 = []
new_file5 = []
new_file6 = []
temp_string = ' '
sentiment_list1 = []
sentiment_list2 = []
twitter_list1 =[]
twitter_list2 =[]

with open(sys.argv[1]) as file:
    new_file1 = file.readlines()

# new_file2 is made by separate list [[],[],[]]
for i in new_file1:
    i = i.lower()
    i = re.sub('(?:https:\/\/)?(?:www\.)?t\.co\/(?:(?:\w)*#!\/)?(?:pages\/)?(?:[\w\-]*\/)*([\w\-]*)',' ', i)
    i = re.sub('[’!"&\'()*+,-./:;<=>?[\\]^`{|}~]+',' ',i)
    i = i.split()
    new_file2.append(i)

# add to the sentiment_list1
for i in new_file2:
    if i[-2] == 'neutral':
        sentiment_list1.append('neutral')
    elif i[-2] == 'negative':
        sentiment_list1.append('negative')
    elif i[-2] == 'positive':
        sentiment_list1.append('positive')

for i in new_file2:
    i.pop()
    i.pop()
    i.pop()
    i.remove(i[0])
    twitter_list1.append(i)

for i in twitter_list1:
    for j in i:
        if len(j) < 2:
            i.remove(j)

# change to the string
for i in twitter_list1:
    new_file3.append(' '.join(i))

# Create text
text_data = np.array(new_file3)

# Create bag of words
count = CountVectorizer(lowercase=False)
bag_of_words = count.fit_transform(text_data)
feature = count.get_feature_names()
#print(feature)

# Create feature matrix
X = bag_of_words.toarray()

# Create target vector
y = np.array(sentiment_list1)
X_train = X
y_train = y

# model
clf = MultinomialNB()
model = clf.fit(X_train, y_train)

with open(sys.argv[2]) as file:
    new_file4 = file.readlines()

# new_file5 is made by separate list [[],[],[]]
for i in new_file4:
    i = i.lower()
    i = re.sub('(?:https:\/\/)?(?:www\.)?t\.co\/(?:(?:\w)*#!\/)?(?:pages\/)?(?:[\w\-]*\/)*([\w\-]*)',' ', i)
    i = re.sub('[’!"&\'()*+,-./:;<=>?[\\]^`{|}~]+','',i)
    i = i.split()
    new_file5.append(i)

for i in new_file5:
    i.pop()
    i.pop()
    i.pop()
    sentiment_list2.append(i[0])
    i.remove(i[0])
    twitter_list2.append(i)

for i in twitter_list2:
    for j in i:
        if len(j) < 2:
            i.remove(j)

# change to the string
for i in twitter_list2:
    new_file6.append(' '.join(i))

Y = count.transform(new_file6)
y_test = Y.toarray()
predicted_y = model.predict(y_test)

for i in range(len(predicted_y)):
    if i == len(predicted_y) - 1:
        string = sentiment_list2[i] + ' ' + predicted_y[i]
    else:
        string = sentiment_list2[i] + ' ' + predicted_y[i] + '\n'
    print(string,end ='')




'''
new_file = new_file3 + new_file6
sentiment_list = sentiment_list1 + sentiment_list2

# Create text
text_data = np.array(new_file)

# Create bag of words
count = CountVectorizer()
bag_of_words = count.fit_transform(text_data)
feature = count.get_feature_names()
#print(feature)

# Create feature matrix
X = bag_of_words.toarray()

# Create target vector
y = np.array(sentiment_list)
X_train = X[:1500]
X_test = X[1500:]
y_train = y[:1500]
y_test = y[1500:]'''
'''
# model
clf = MultinomialNB()
model = clf.fit(X_train, y_train)
predicted_y = model.predict(X_test)

positive = []
negative = []
neutral = []
for i in sentiment_list:
    if i == 1:
        positive.append(i)
    if i == -1:
        negative.append(i)
    if i == 0:
        neutral.append(i)'''

#print(len(positive))
#print(len(negative))
#print(len(neutral))
#print(sentiment_list)
#print(len(sentiment_list))'''
'''
# output
print('TwitterID'+ ' ' + 'sentiment')
for i in range(len(predicted_y)):
    print(i+1,end='')
    if predicted_y[i] == 1:
        print(' ' + 'positive')
    if predicted_y[i] == 0:
        print(' ' + 'neutral')
    if predicted_y[i] == -1:
        print(' ' + 'negative')'''

#print(y_test, predicted_y)
#print(model.predict_proba(X_test))
#print(accuracy_score(y_test, predicted_y))