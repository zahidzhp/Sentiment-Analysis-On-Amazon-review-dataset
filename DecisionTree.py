import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd
from sklearn.model_selection import train_test_split

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



from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import tree


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

classificationText = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', tree.DecisionTreeClassifier())])


classificationText = classificationText.fit(X_train, Y_train)
import numpy as np

prediction_target = classificationText.predict(X_test)

print("Accuracy in sentiment detection in percentage : ", (np.mean(prediction_target == Y_test)) * 100)

from sklearn.model_selection import GridSearchCV
parameters_model = {'vect__ngram_range': [(1, 1), (1, 2)],
              'tfidf__use_idf': (True, False),
              'clf__alpha': (1e-2, 1e-3),
                    }
graphSearch_classification = GridSearchCV(classificationText, parameters_model, n_jobs=1)
graphSearch_classification = graphSearch_classification.fit(X_train,Y_train )


res=graphSearch_classification.best_score_
print("Best parameters set found on development set:")
print()
print(graphSearch_classification.best_params_)

print("Best Score found :", 100*res)

print("Grid scores on development set:")
print()
means = graphSearch_classification.cv_results_['mean_test_score']
stds = graphSearch_classification.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, graphSearch_classification.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
        % (mean, std * 2, params))
print()