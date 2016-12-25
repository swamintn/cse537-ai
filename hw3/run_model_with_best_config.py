#!/usr/bin/env python
"""
Script to run the best classifier configuration using a .pkl file

The current best configuration is,
    'Naive Bayes alpha=0.01 fit_prior=False; unigram; tfidfvectorizer; stemmer; stopper; feature selection - SelectFromModel threshold=30'

The script needs the following file in the current working directory,
    full_model.pkl

Run using,
    run_model_with_best_config.py <Location of Test data>

The test data location must be in a fashion similar to the
Selected_20NewsGroup/Test directory structure

Example:
    run_model_with_best_config.py /user/Selected_20NewsGroup/Test

@author: Swaminathan Sivaraman
"""

import sklearn
from   sklearn import datasets
from   sklearn import svm
from   sklearn import metrics

from   sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from   sklearn.datasets                import load_files
from   sklearn.externals               import joblib

from   sklearn.naive_bayes             import MultinomialNB

from   sklearn.feature_selection       import SelectFromModel, SelectPercentile
from   sklearn.feature_selection       import chi2

from   nltk.stem import SnowballStemmer, PorterStemmer
from   nltk.corpus import stopwords

from   tabulate   import tabulate

import numpy as np
import os
import sys
import cPickle

def remove_headers_and_stem_and_stop(t_data, stemmer=None, stop=True):
    stop_ws = set(stopwords.words('english'))
    for i in range(len(t_data.data)):
        lines = t_data.data[i].splitlines()
        new_lines = []
        is_header = True
        for j, line in enumerate(lines):
            if is_header and ':' in line:
                continue
            is_header = False
            if (stemmer or stop) and line:
                words, new_words = line.split(' '), []
                for word in words:
                    word = word.decode('utf-8', 'ignore')
                    if stop and word.lower() in stop_ws:
                        continue
                    if stemmer:
                        word = str(stemmer.stem(word))
                    new_words.append(str(word))             
                new_line = ' '.join(new_words)
            else:
                new_line = line
            new_lines.append(new_line)
        t_data.data[i] = '\n'.join(new_lines)

def load_files_correctly(loc, stemmer=None, stop=True):
    t_data = load_files(loc)
    remove_headers_and_stem_and_stop(t_data, stemmer, stop)
    return t_data


def main():
    test_dir = sys.argv[1]

    print 'Loading model from full_model.pkl in current working directory...'

    stemmer = PorterStemmer()
    test_stem_stop = load_files_correctly(test_dir, stemmer=stemmer, stop=True)

    with open('full_model.pkl', 'rb') as f:
        values = cPickle.load(f)

    # The expected order is name, vectorizer, feature_selector, classifier
    name, vectroizer, feature_selector, classifier = values

    vectorized_data = vectroizer.transform(test_stem_stop.data)
    selected_data = feature_selector.transform(vectorized_data)
    pred = classifier.predict(selected_data)

    precision = metrics.precision_score(test_stem_stop.target, pred, average='macro')                                                                          
    recall    = metrics.recall_score(test_stem_stop.target, pred, average='macro')                                                                             
    f1_score  = metrics.f1_score(test_stem_stop.target, pred, average='macro')

    print "Model name:", name
    print "Precision:", precision
    print "Recall:", recall
    print "Precision/Recall:", precision / recall
    print "F1 Score:", f1_score


if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == '--help':
        print __doc__
        sys.exit()
    main()
