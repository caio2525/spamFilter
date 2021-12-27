import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import string
import nltk
import re
from sklearn.base import BaseEstimator, TransformerMixin
import dill
from urlextract import URLExtract
import pickle
from flask import Flask, request

#nltk.download('stopwords')
#nltk.download('punkt')

with open('punct.pkl', 'rb') as f:
    punk = dill.load(f)
    f.close()

with open('stopwords.pkl', 'rb') as f:
    stwords = dill.load(f)
    f.close()

class customTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, re, extractor):
        self.re = re
        self.extractor = extractor

    def pre_processing(self, text):
        urls = list(set(self.extractor.find_urls(text)))
        for url in urls:
            text = text.replace(url, " URL ")
        text = self.re.sub(r'\d+(?:\.\d*)?(?:[eE][+-]?\d+)?', 'NUMBER', text)
        lowercase = text.lower()
        remove_ponc = ''.join([character for character in lowercase if character not in punk])
        tokenize = nltk.tokenize.word_tokenize(remove_ponc)
        remove_stopwords = [word for word in tokenize if word not in stwords]
        return(remove_stopwords)

    def fit(self, X, y=None):
        return(self)

    def transform(self, X, y=None):
        processed_text = np.array([self.pre_processing(text) for text in X], dtype='object')
        return np.c_[X, processed_text]

class customPredictor:
    def __init__(self, threshold = 85):
        self.spam_messages = []
        self.ham_messages = []
        self.spam_words = []
        self.ham_words = []
        self.spam_problt = 0.5
        self.ham_problt = 0.5
        self.unique_spam_words = []
        self.unique_ham_words = []
        self.prob_word_given_spam = {}
        self.prob_word_given_ham = {}
        self.threshold = threshold

    def splitSpamHamMessages(self, X, y):
        for i in range(len(X)):
            if(y[i] == 'spam'):
                self.spam_messages.append(X[i])
            elif(y[i] == 'ham'):
                self.ham_messages.append(X[i])

    def calcSpamHamProb(self):
        self.spam_problt = len(self.spam_messages) / (len(self.spam_messages) + len(self.ham_messages))
        self.ham_problt = len(self.ham_messages) / (len(self.spam_messages) + len(self.ham_messages))

    def bagOfWord(self):
        for text in self.spam_messages:
            for word in text:
                self.spam_words.append(word)

        for text in self.ham_messages:
            for word in text:
                self.ham_words.append(word)

    def uniqueSpamHamWords(self):
        for word in self.spam_words:
            if(word not in self.unique_spam_words):
                self.unique_spam_words.append(word)

        for word in self.ham_words:
            if( word not in self.unique_ham_words):
                self.unique_ham_words.append(word)

    def probWordGivenSpamHam(self):
        for word in self.unique_spam_words:
            self.prob_word_given_spam[word] = self.spam_words.count(word) / len(self.spam_words) * 100

        for word in self.unique_ham_words:
            self.prob_word_given_ham[word] = self.ham_words.count(word) / len(self.ham_words) * 100

    def fit(self, X, y):
        self.splitSpamHamMessages(X, y)
        self.calcSpamHamProb()
        self.bagOfWord()
        self.uniqueSpamHamWords()
        self.probWordGivenSpamHam()

    def calcProb(self, text):
        prob_spam_given_words = 100 * self.spam_problt;
        prob_ham_given_words = 100 * self.ham_problt;
        for word in text:
            if word in self.prob_word_given_spam:
                if word in self.prob_word_given_ham:
                    prob_spam_given_words *= self.prob_word_given_spam[word]
                    prob_ham_given_words *= self.prob_word_given_ham[word]

            elif word not in self.prob_word_given_spam:
                if word in self.prob_word_given_ham:
                    prob_spam_given_words *= 0.6 * 100
                    prob_ham_given_words *= 0.4 * 100

            elif word in self.prob_word_given_spam:
                if word not in self.prob_word_given_ham:
                    prob_spam_given_words *= 0.6 * 100
                    prob_ham_given_words *= 0.4 * 100

            elif word not in self.prob_word_given_spam:
                if word not in self.prob_word_given_ham:
                    prob_spam_given_words *= 0.4 * 100
                    prob_ham_given_words *= 0.6 * 100
        return(prob_spam_given_words / (prob_spam_given_words + prob_ham_given_words) * 100)

    def classify(self, spam_probability):
        if (spam_probability > self.threshold):
            return('spam')
        else:
            return('ham')

    def predicao(self, text):
        spam_probability = self.calcProb(text)
        return(self.classify(spam_probability))

    def predict(self, X):
        predictions = [self.predicao(text) for text in X]
        return np.c_[predictions]



#entrada = input('digite um email ')

#processed_input = transformer.transform(X = np.array([entrada]))
#print(processed_input)



#print(processed_input[:, 1:])
#pred = model.predict(X = processed_input[:, 1:])
#print(pred)

app = Flask('app')

@app.route('/test', methods=['GET'])
def test():
    return "Pinging Model Application!!"

@app.route("/predict", methods=["POST"])
def predict():
    json_data = request.json
    entrada = json_data['entrada']
    print(entrada)

    with open('custom_transformer2.pkl', 'rb') as f:
        transformer = dill.load(f)
        f.close()

    processed_input = transformer.transform(X = np.array([entrada]))
    print(processed_input)

    with open("model.bin",mode='rb') as model_f:
        model = pickle.load(model_f)
        model_f.close()

    print(processed_input[:, 1:])
    pred = model.predict(X = processed_input[:, 1:])
    print(pred[0][0])

    return pred[0][0]
