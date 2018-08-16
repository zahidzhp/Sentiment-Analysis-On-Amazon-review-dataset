from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import SVC
import pandas as pd
from sklearn.model_selection import train_test_split

import nltk
from nltk import PorterStemmer
from nltk.corpus import stopwords


def filter_stop_words(train_sentences, stop_words):
    for i, sentence in enumerate(train_sentences):
        new_sent = [word for word in sentence.split() if word not in stop_words]
        train_sentences[i] = ' '.join(new_sent)
    return train_sentences


data = pd.read_csv('amazon_review_updated.csv')

stop_words = set(stopwords.words("english"))


Y = data['Sentiment'].values
X = data['Text'].values
X = filter_stop_words(X, stop_words)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

classificationText = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf',SVC(kernel='linear')), ])

classificationText = classificationText.fit(X_train, Y_train)



import numpy as np

prediction_target = classificationText.predict(X_test)


print("Accuracy in Categorization in percentage : ", (np.mean(prediction_target == Y_test)) * 100)
tunes_parameters=[{'kernel':['rbf'],'gama':[1e-2,1e-4,1e-4,1e-5],
                   'C':[0.001,0.01,0.1,10,25,50,100,1000]},
                  {'kernel':['sigmoid'],'gamma':[1e-2,1e-3,1e-4,1e-5], 'C':[0.001,0.01,0.1,10,25,50,100,1000]},
                  {'kernel':['linear'],'C':[0.001,0.01,0.1,10,25,50,100,1000]}]

from sklearn.model_selection import GridSearchCV
parameters_model = {'vect__ngram_range': [(1, 1), (1, 2),(1,3)],
              'tfidf__use_idf': (True, False),
              }
graphSearch_classification = GridSearchCV(classificationText, parameters_model,tunes_parameters, n_jobs=1)
graphSearch_classification = graphSearch_classification.fit(X_train,Y_train)


res=graphSearch_classification.best_score_
print(graphSearch_classification.best_params_)


print("Accuracy of SVM with Graph Search :", 100*res)