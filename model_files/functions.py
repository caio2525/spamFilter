import numpy as np
import pandas as pd
import string
import re
from sklearn.base import BaseEstimator, TransformerMixin
from urlextract import URLExtract
import dill
import pickle





def makePredictions(entrada, punk, stwords, transformer):
    processed_input = transformar(transformer, np.array([entrada]), punk, stwords)
    print('processed_input: ', processed_input)

def transformar(transformer, X, punk, stwords):
    processed_text = np.array([transformer.pre_processing(text, punk, stwords) for text in X], dtype='object')
    return np.c_[X, processed_text]
