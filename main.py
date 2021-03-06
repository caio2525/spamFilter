import numpy as np
import dill
import pickle
import re
from sklearn.base import BaseEstimator, TransformerMixin
from urlextract import URLExtract
import nltk
from flask import Flask, request, render_template, make_response, redirect
import string
from flask_cors import CORS, cross_origin

class customTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, stopwords, punctuation, re, extractor):
        self.stopwords = stopwords
        self.punctuation  = punctuation
        self.re = re
        self.extractor = extractor

    def pre_processing(self, text):
        urls = list(set(self.extractor.find_urls(text)))
        for url in urls:
            text = text.replace(url, " URL ")
        text = self.re.sub(r'\d+(?:\.\d*)?(?:[eE][+-]?\d+)?', 'NUMBER', text)
        lowercase = text.lower()
        remove_ponc = ''.join([character for character in lowercase if character not in self.punctuation ])
        tokenize = nltk.tokenize.word_tokenize(remove_ponc)
        remove_stopwords = [word for word in tokenize if word not in self.stopwords]
        return(remove_stopwords)

    def fit(self, X, y=None):
        return(self)

    def transform(self, X, y=None):
        processed_text = np.array([self.pre_processing(text) for text in X], dtype='object')
        return np.c_[X, processed_text]

class customPredictor:
    def __init__(self, threshold = 85, spam_problt = 0.5, ham_problt = 0.5):
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
        #self.calcSpamHamProb()
        self.bagOfWord()
        self.uniqueSpamHamWords()
        self.probWordGivenSpamHam()

    def calcProb(self, text):
        print('calcProb text ', text)
        print('type calcProb text ', type(text))
        #print('self.prob_word_given_spam', self.prob_word_given_spam)
        prob_spam_given_words = 100 * self.spam_problt;
        prob_ham_given_words = 100 * self.ham_problt;
        for word in text:
            pWGS = self.prob_word_given_spam.get(word)
            pWGH = self.prob_word_given_ham.get(word)

            if pWGS:
                if pWGH:
                    prob_spam_given_words *= pWGS
                    prob_ham_given_words *= pWGH

            if not pWGS:
                if pWGH:
                    prob_spam_given_words *= 0.4 * 100
                    prob_ham_given_words *= 0.6 * 100

            if pWGS:
                if not pWGH:
                    prob_spam_given_words *= 0.6 * 100
                    prob_ham_given_words *= 0.4 * 100

            if not pWGS:
                if not pWGH:
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
        print('X', X)
        print('type(X)', type(X))
        predictions = [self.predicao(text) for text in X]
        return np.c_[predictions]

def predizer(entrada):
    nltk.download('stopwords')
    nltk.download('punkt')

    stopwords = nltk.corpus.stopwords.words('english')
    punctuation = string.punctuation
    extractor = URLExtract()

    transformer = customTransformer(stopwords, punctuation, re, extractor)

    processed_input = transformer.transform(X = np.array([entrada]))
    print('processed_input:', processed_input)


    with open('./model_files/model.bin', 'rb') as f:
        model = dill.load(f)
        f.close()

    print('processed_input[0, 1:]', processed_input[0, 1:])
    print('type(processed_input[0, 1:])', type(processed_input[0, 1:]))
    pred = model.predict(X = processed_input[:, 1:])
    print('pred', pred[0][0])

    return pred[0][0];


app = Flask('app')
CORS(app)

@app.route('/', methods=['GET', 'POST'])
def test():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        form_data = request.form
        print('form_data', form_data)
        entrada = form_data['email']
        pred = predizer(entrada)

        if(pred == "spam"):
            return render_template('result.html', value='Spam', back="red")
        else:
            return render_template('result.html', value='Ham', back="green")

@app.route("/predict", methods=["POST"])
@cross_origin()
def predict():
    json_data = request.json
    entrada = json_data['entrada']
    print('entrada:', entrada)

    nltk.download('stopwords')
    nltk.download('punkt')

    stopwords = nltk.corpus.stopwords.words('english')
    punctuation = string.punctuation
    extractor = URLExtract()

    transformer = customTransformer(stopwords, punctuation, re, extractor)

    processed_input = transformer.transform(X = np.array([entrada]))
    print('processed_input:', processed_input)


    with open('./model_files/model.bin', 'rb') as f:
        model = dill.load(f)
        f.close()

    print('processed_input[0, 1:]', processed_input[0, 1:])
    print('type(processed_input[0, 1:])', type(processed_input[0, 1:]))
    pred = model.predict(X = processed_input[:, 1:])
    print('pred', pred[0][0])

    response = make_response(pred[0][0])
    #response.headers.set('Content-Type', 'multipart/form-data')
    #response.headers.add('Access-Control-Allow-Origin', '*')

    return response
