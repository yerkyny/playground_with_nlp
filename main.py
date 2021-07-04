import numpy as np
import pandas as pd
import sklearn
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import  KFold
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv('IMDB_Dataset.csv')
new_df = df[:1000]
Reviews = new_df['review']
y = new_df['sentiment'].map(lambda sent: int(sent=='positive'))

import nltk
nltk.download()

from nltk.corpus import stopwords
sw_list = list(stopwords.words('english'))

import string

def clear(text):
    text = text.replace('<', ' ')
    text = text.replace('.', ' ')
    clean_text = ''
    for word in text.lower().split():
        if word != 'br':
            clean_text+= ' ' + word
    clean_text = clean_text.translate(str.maketrans('', '', string.punctuation))
    return clean_text

clean_Reviews = Reviews.map(clear)
print(clear(Reviews[0]))

count_vec = CountVectorizer()
count_Reviews = count_vec.fit_transform(clean_Reviews)

tfidf_vec = TfidfVectorizer()
tfidf_Reviews = tfidf_vec.fit_transform(clean_Reviews)

grid = {'C': np.power(10.0, np.arange(-5, 5))}
cv = KFold(n_splits=5, shuffle=True, random_state=241)
clf = SVC(kernel='linear', random_state=241)
gs = sklearn.model_selection.GridSearchCV(clf, grid, scoring='accuracy', cv=cv)
gs.fit(count_Reviews, y)

print(gs.best_params_)

svm = SVC(C = 1, kernel='linear', random_state=42)
svm.fit(tfidf_Reviews,y)
coefs = abs(svm.coef_.toarray())
coefs[0]

def max_n(obj, n):
    res = np.array([])
    count = 0
    while count<n:
        max = 0
        indmax = 0
        for k in range(len(obj)):
            if obj[k] > max:
                max = obj[k]
                indmax = k
        res = np.append(res,indmax)
        obj = np.delete(obj,indmax)
        count+=1
    return res
max_20 = max_n(coefs[0],20)
max_20

some_list = tfidf_vec.get_feature_names()
range_ = list(map(lambda x: int(x),max_20))
a = []
for i in range_:
    a.append(some_list[i])
sorted(a)
