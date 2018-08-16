

import numpy as np
import pandas as pd
import nltk
from nltk import PorterStemmer
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import string

nltk.download('punkt')
nltk.download('stopwords')

from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Bidirectional
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re


data = pd.read_csv('amazon_review_updated.csv')
# Keeping only the necessary columns


print(data.head())


print(data[data['Sentiment'] == 'Positive'].size)
print(data[data['Sentiment'] == 'Negative'].size)


def filter_stop_words(train_sentences, stop_words):
    for i, sentence in enumerate(train_sentences):
        new_sent = [word for word in sentence.split() if word not in stop_words]
        train_sentences[i] = ' '.join(new_sent)
    return train_sentences


def stemmer(train_sentences):
    for i, sentence in enumerate(train_sentences):
        new_sent = [port.stem(word) for word in sentence.split()]
        train_sentences[i] = ' '.join(new_sent)
    return train_sentences


stop_words = set(stopwords.words("english"))

port = PorterStemmer()


max_fatures = 200
train_sentences = data['Text'].values
train_sentences = filter_stop_words(train_sentences, stop_words)
train_sentences = stemmer(train_sentences)

tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(train_sentences)
X = tokenizer.texts_to_sequences(train_sentences)
X = pad_sequences(X)

embed_dim = 128
lstm_out = 196

model = Sequential()
model.add(Embedding(max_fatures, embed_dim, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(Bidirectional(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2)))
model.add(Dense(2, activation='sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

Y = pd.get_dummies(data['Sentiment']).values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

batch_size = 32
model.fit(X_train, Y_train, epochs=10, batch_size=batch_size, verbose=2)

validation_size = 1500

X_validate = X_test[-validation_size:]
Y_validate = Y_test[-validation_size:]
X_test = X_test[:-validation_size]
Y_test = Y_test[:-validation_size]
score, acc = model.evaluate(X_test, Y_test, verbose=2, batch_size=batch_size)
print("score: %.2f" % (score))
print("acc: %.2f" % (acc))

pos_cnt, neg_cnt, pos_correct, neg_correct = 0, 0, 0, 0
for x in range(len(X_validate)):

    result = model.predict(X_validate[x].reshape(1, X_test.shape[1]), batch_size=1, verbose=2)[0]

    if np.argmax(result) == np.argmax(Y_validate[x]):
        if np.argmax(Y_validate[x]) == 0:
            neg_correct += 1
        else:
            pos_correct += 1

    if np.argmax(Y_validate[x]) == 0:
        neg_cnt += 1
    else:
        pos_cnt += 1

print("pos_acc", pos_correct / pos_cnt * 100, "%")
print("neg_acc", neg_correct / neg_cnt * 100, "%")

review = ['The European Union makes it impossible for our farmers and workers and companies to do business in Europe (U.S. has a $151 Billion trade deficit), and then they want us to happily defend them through NATO, and nicely pay for it. Just doesn’t work!.']
# vectorizing the tweet by the pre-fitted tokenizer instance
review = tokenizer.texts_to_sequences(review)
# padding the tweet to have exactly the same shape as `embedding_2` input
review = pad_sequences(review, maxlen=16, dtype='int32', value=0)
print(review)
sentiment = model.predict(review, batch_size=1, verbose=2)[0]
if (np.argmax(sentiment) == 0):
    print("Negative")
elif (np.argmax(sentiment) == 1):
    print("Positive")


