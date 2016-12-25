#!/usr/bin/env python 
"""
Script used to generate a model and dump it to a file for the best configuration

First argument must be location of train data.

>> generate_model_with_best_config.py <Dir of training data> 

The script will generate an output model file 'full_model.pkl' in the current 
working directory

The test data location must be in a fashion similar to the
Selected_20NewsGroup/Training directory structure

Example:
    generate_model_with_best_config.py /user/Selected_20NewsGroup/Training

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
import cPickle
import sys


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
    data_loc = sys.argv[1]
    
    print 'Training the model...'

    stemmer = PorterStemmer()
    # Get the training data
    train_stem_stop  = load_files_correctly(data_loc, stemmer=stemmer, stop=True)
    tfid_u_stem_stop = TfidfVectorizer(ngram_range=(1,1), decode_error='ignore')

    tfid_u_train_stem_stop = tfid_u_stem_stop.fit_transform(train_stem_stop.data)

    name = 'Naive Bayes alpha=0.01 fit_prior=False; unigram; tfidfvectorizer; stemmer; stopper; feature selection - SelectFromModel threshold=30'
    clf = MultinomialNB(alpha=0.01, fit_prior=False)
    clf.fit(tfid_u_train_stem_stop, train_stem_stop.target)
    model = SelectFromModel(clf, threshold=30, prefit=True)
    X_new_train = model.transform(tfid_u_train_stem_stop)
    clf = MultinomialNB(alpha=0.01, fit_prior=False)
    clf.fit(X_new_train, train_stem_stop.target)
    # This is the best model. Dump all transformers to pwd
    values = [name, tfid_u_stem_stop, model, clf]
    with open('full_model.pkl', 'wb') as f:
        cPickle.dump(values, f)

    print "Model name:", name
    print "\nSaved model to 'full_model.pkl' file in current working directory"


if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == '--help':
        print __doc__
        sys.exit()
    main()
