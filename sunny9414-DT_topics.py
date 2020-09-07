import numpy as np
import re
import sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn import tree
from sklearn.model_selection import train_test_split
from collections import defaultdict
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer

new_file2 = []
new_file4 = []
temp_string = ' '
twitter_list1 = []
twitter_list2 = []
topic_list1 = []
topic_list2 = []

with open(sys.argv[1]) as file:
    new_file1 = file.readlines()

for i in new_file1:
    i = i.lower()
    i = re.sub('(?:https:\/\/)?(?:www\.)?t\.co\/(?:(?:\w)*#!\/)?(?:pages\/)?(?:[\w\-]*\/)*([\w\-]*)', ' ', i)
    i = re.sub('[’!"&\'()*+,-./:;<=>?[\\]^`{|}~]+', ' ', i)
    i = i.split()
    new_file2.append(i)

# new_film2 is made by separate list [[],[],[]]
for i in new_file2:
    topic_list1.append(i[-3])

# twitter_list1 是 twitter content
for i in new_file2:
    i.pop()
    i.pop()
    i.pop()
    i = i[1:]
    for j in i:
        if len(j) < 2:
            i.remove(j)
    twitter_list1.append(' '.join(i))

# Create text
text_data = np.array(twitter_list1)

# Create bag of words
count = CountVectorizer(max_features = 200,lowercase=False)
bag_of_words = count.fit_transform(text_data)
feature = count.get_feature_names()
# print(feature)

# Create feature matrix
X = bag_of_words.toarray()

# Create target vector
y = np.array(topic_list1)
X_train = X
y_train = y

# model
# model
clf = tree.DecisionTreeClassifier(criterion='entropy',random_state = 0, min_samples_leaf = 20)
model = clf.fit(X_train, y_train)

with open(sys.argv[2]) as file:
    new_file3 = file.readlines()

for i in new_file3:
    i = i.lower()
    i = re.sub('(?:https:\/\/)?(?:www\.)?t\.co\/(?:(?:\w)*#!\/)?(?:pages\/)?(?:[\w\-]*\/)*([\w\-]*)', ' ', i)
    i = re.sub('[’!"&\'()*+,-./:;<=>?[\\]^`{|}~]+', ' ', i)
    i = i.split()
    new_file4.append(i)

# new_film4 is made by separate list [[],[],[]]
# twitter_list2 是 twitter content
for i in new_file4:
    i.pop()
    i.pop()
    i.pop()
    topic_list2.append(i[0])
    i = i[1:]
    for j in i:
        if len(j) < 2:
            i.remove(j)
    twitter_list2.append(' '.join(i))

Y = count.transform(twitter_list2)
y_test = Y.toarray()
predicted_y = model.predict(y_test)

for i in range(len(predicted_y)):
    if i == len(predicted_y) - 1:
        string = topic_list2[i] + ' ' + predicted_y[i]
    else:
        string = topic_list2[i] + ' ' + predicted_y[i] + '\n'
    print(string, end='')




'''import numpy as np
import re
import sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn import tree
from sklearn.model_selection import train_test_split
from collections import defaultdict
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer

topics_dict1 = defaultdict(list)
new_file2 = []
new_file4 = []
temp_string = ' '
twitter_list1 =[]
twitter_list2 = []
topic_list1 = []
topic_list2 = []

with open(sys.argv[1]) as file:
    new_file1 = file.readlines()

for i in new_file1:
    i = i.lower()
    i = re.sub('(?:https:\/\/)?(?:www\.)?t\.co\/(?:(?:\w)*#!\/)?(?:pages\/)?(?:[\w\-]*\/)*([\w\-]*)',' ', i)
    i = re.sub('[’!"&\'()*+,-./:;<=>?[\\]^`{|}~]+',' ',i)
    i = i.split()
    new_file2.append(i)

# new_film2 is made by separate list [[],[],[]]
for i in new_file2:
    topic_list1.append(i[-3])

# twitter_list1 是 twitter content
for i in new_file2:
    i.pop()
    i.pop()
    i.pop()
    i = i[1:]
    for j in i:
        if len(j) < 2:
            i.remove(j)
    twitter_list1.append(' '.join(i))

with open(sys.argv[2]) as file:
    new_file3 = file.readlines()

for i in new_file3:
    i = i.lower()
    i = re.sub('(?:https:\/\/)?(?:www\.)?t\.co\/(?:(?:\w)*#!\/)?(?:pages\/)?(?:[\w\-]*\/)*([\w\-]*)',' ', i)
    i = re.sub('[’!"&\'()*+,-./:;<=>?[\\]^`{|}~]+',' ',i)
    i = i.split()
    new_file4.append(i)

# new_film4 is made by separate list [[],[],[]]
for i in new_file4:
    topic_list2.append(i[-3])

# twitter_list2 是 twitter content
for i in new_file4:
    i.pop()
    i.pop()
    i.pop()
    i = i[1:]
    for j in i:
        if len(j) < 2:
            i.remove(j)
    twitter_list2.append(' '.join(i))

# x
twitter_list = twitter_list1 + twitter_list2
# y
topic_list = topic_list1 + topic_list2


# Create text
text_data = np.array(twitter_list)

# Create bag of words
count = CountVectorizer(max_features = 200)
bag_of_words = count.fit_transform(text_data)
feature = count.get_feature_names()

# Create feature matrix
X = bag_of_words.toarray()

# Create target vector
y = np.array(topic_list)
X_train = X[:1500]
X_test = X[1500:]
y_train = y[:1500]
y_test = y[1500:]

# model
clf = tree.DecisionTreeClassifier(criterion='entropy',random_state = 0, min_samples_leaf = 20)
model = clf.fit(X_train, y_train)
predicted_y = model.predict(X_test)

# for counting the topics number
for i in range(len(topic_list)):
    topics_dict1[topic_list[i]].append(i)
for key in topics_dict1:
    print(key)
    print(len(topics_dict1[key]))

#print(topics_dict1)

# output
print('TwitterID' + ' ' + 'TopicID')
for i in range(len(predicted_y)):
    print(i + 1, end='')
    print(' ' + predicted_y[i])'''

#print(y_test, predicted_y)
#print(model.predict_proba(X_test))
#print(accuracy_score(y_test, predicted_y))