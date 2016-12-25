#!/usr/bin/env python 
"""
Script used to find the best configuration for the Naive Bayes classifier

First argument must be location of train & test data.

>> get_best_naive_bayes.py <Dir of training+test data> 

The script will generate an ouuput model file 'full_model.pkl' in the current 
working directory

Example:
    get_best_naive_bayes.py /user/Selected_20NewsGroup

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


from   nltk.stem   import PorterStemmer
from   nltk.corpus import stopwords

from   tabulate    import tabulate

import numpy as np
import os
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

    stemmer = PorterStemmer()

    # Get data for all combos
    train_stem_stop       = load_files_correctly(os.path.join(data_loc, 'Training'), stemmer=stemmer, stop=True)
    test_stem_stop        = load_files_correctly(os.path.join(data_loc, 'Test'), stemmer=stemmer, stop=True)
    train_no_stem_stop    = load_files_correctly(os.path.join(data_loc, 'Training'), stemmer=None, stop=True)
    test_no_stem_stop     = load_files_correctly(os.path.join(data_loc, 'Test'), stemmer=None, stop=True)
    train_stem_no_stop    = load_files_correctly(os.path.join(data_loc, 'Training'), stemmer=stemmer, stop=False)
    test_stem_no_stop     = load_files_correctly(os.path.join(data_loc, 'Test'), stemmer=stemmer, stop=False)
    train_no_stem_no_stop = load_files_correctly(os.path.join(data_loc, 'Training'), stemmer=None, stop=False)
    test_no_stem_no_stop  = load_files_correctly(os.path.join(data_loc, 'Test'), stemmer=None, stop=False)

    tfid_u_stem_stop       = TfidfVectorizer(ngram_range=(1,1), decode_error='ignore')
    tfid_u_stem_no_stop    = TfidfVectorizer(ngram_range=(1,1), decode_error='ignore')
    tfid_u_no_stem_stop    = TfidfVectorizer(ngram_range=(1,1), decode_error='ignore')
    tfid_u_no_stem_no_stop = TfidfVectorizer(ngram_range=(1,1), decode_error='ignore')

    count_u_stem_stop       = CountVectorizer(ngram_range=(1,1), decode_error='ignore')
    count_u_stem_no_stop    = CountVectorizer(ngram_range=(1,1), decode_error='ignore')
    count_u_no_stem_stop    = CountVectorizer(ngram_range=(1,1), decode_error='ignore')
    count_u_no_stem_no_stop = CountVectorizer(ngram_range=(1,1), decode_error='ignore')

    tfid_u_train_stem_stop = tfid_u_stem_stop.fit_transform(train_stem_stop.data)
    tfid_u_test_stem_stop = tfid_u_stem_stop.transform(test_stem_stop.data)

    tfid_u_train_stem_no_stop = tfid_u_stem_no_stop.fit_transform(train_stem_no_stop.data)
    tfid_u_test_stem_no_stop = tfid_u_stem_no_stop.transform(test_stem_no_stop.data)

    tfid_u_train_no_stem_stop = tfid_u_no_stem_stop.fit_transform(train_no_stem_stop.data)
    tfid_u_test_no_stem_stop = tfid_u_no_stem_stop.transform(test_no_stem_stop.data)

    tfid_u_train_no_stem_no_stop = tfid_u_no_stem_no_stop.fit_transform(train_no_stem_no_stop.data)
    tfid_u_test_no_stem_no_stop = tfid_u_no_stem_no_stop.transform(test_no_stem_no_stop.data)

    count_u_train_stem_stop = count_u_stem_stop.fit_transform(train_stem_stop.data)
    count_u_test_stem_stop = count_u_stem_stop.transform(test_stem_stop.data)

    count_u_train_stem_no_stop = count_u_stem_no_stop.fit_transform(train_stem_no_stop.data)
    count_u_test_stem_no_stop = count_u_stem_no_stop.transform(test_stem_no_stop.data)

    count_u_train_no_stem_stop = count_u_no_stem_stop.fit_transform(train_no_stem_stop.data)
    count_u_test_no_stem_stop = count_u_no_stem_stop.transform(test_no_stem_stop.data)

    count_u_train_no_stem_no_stop = count_u_no_stem_no_stop.fit_transform(train_no_stem_no_stop.data)
    count_u_test_no_stem_no_stop = count_u_no_stem_no_stop.transform(test_no_stem_no_stop.data)

    # Vectorize data
    res = []

    name = 'Naive Bayes alpha=0.01; unigram; countvectorizer; no stemmer; no stopper; no feature selection'
    clf = MultinomialNB(alpha=0.01)
    clf.fit(count_u_train_no_stem_no_stop, train_no_stem_no_stop.target)
    pred = clf.predict(count_u_test_no_stem_no_stop)
    res.append({'name': name, 'clf': clf, 'pred': pred, 'target': test_no_stem_no_stop.target})

    name = 'Naive Bayes alpha=0.01; unigram; tfidfvectorizer; no stemmer; no stopper; no feature selection'
    clf = MultinomialNB(alpha=0.01)                                                     
    clf.fit(tfid_u_train_no_stem_no_stop, train_no_stem_no_stop.target)             
    pred = clf.predict(tfid_u_test_no_stem_no_stop)                             
    res.append({'name': name, 'clf': clf, 'pred': pred, 'target': test_no_stem_no_stop.target})
        
    name = 'Naive Bayes alpha=0.01 fit_prior=False; unigram; tfidfvectorizer; no stemmer; no stopper; no feature selection'
    clf = MultinomialNB(alpha=0.01, fit_prior=False)
    clf.fit(tfid_u_train_no_stem_no_stop, train_no_stem_no_stop.target)
    pred = clf.predict(tfid_u_test_no_stem_no_stop)
    res.append({'name': name, 'clf': clf, 'pred': pred, 'target': test_no_stem_no_stop.target})
    
    name = 'Naive Bayes alpha=0.01; unigram; countvectorizer; stemmer; no stopper; no feature selection'
    clf = MultinomialNB(alpha=0.01)                                                     
    clf.fit(count_u_train_stem_no_stop, train_stem_no_stop.target)             
    pred = clf.predict(count_u_test_stem_no_stop)                             
    res.append({'name': name, 'clf': clf, 'pred': pred, 'target': test_stem_no_stop.target})
    
    name = 'Naive Bayes alpha=0.01; unigram; countvectorizer; no stemmer; stopper; no feature selection'
    clf = MultinomialNB(alpha=0.01)                                                     
    clf.fit(count_u_train_no_stem_stop, train_no_stem_stop.target)             
    pred = clf.predict(count_u_test_no_stem_stop)                             
    res.append({'name': name, 'clf': clf, 'pred': pred, 'target': test_no_stem_stop.target})
    
    name = 'Naive Bayes alpha=0.01; unigram; countvectorizer; stemmer; stopper; no feature selection'
    clf = MultinomialNB(alpha=0.01)                                                     
    clf.fit(count_u_train_stem_stop, train_stem_stop.target)             
    pred = clf.predict(count_u_test_stem_stop)                             
    res.append({'name': name, 'clf': clf, 'pred': pred, 'target': test_stem_stop.target})
    
    name = 'Naive Bayes alpha=0.01 fit_prior=False; unigram; tfidfvectorizer; no stemmer; stopper; no feature selection'
    clf = MultinomialNB(alpha=0.01, fit_prior=False)
    clf.fit(tfid_u_train_no_stem_stop, train_no_stem_stop.target)
    pred = clf.predict(tfid_u_test_no_stem_stop)
    res.append({'name': name, 'clf': clf, 'pred': pred, 'target': test_no_stem_stop.target})
    
    name = 'Naive Bayes alpha=0.01 fit_prior=False; unigram; tfidfvectorizer; stemmer; stopper; no feature selection'
    clf = MultinomialNB(alpha=0.01, fit_prior=False)
    clf.fit(tfid_u_train_stem_stop, train_stem_stop.target)
    pred = clf.predict(tfid_u_test_stem_stop)
    res.append({'name': name, 'clf': clf, 'pred': pred, 'target': test_stem_stop.target})
    
    name = 'Naive Bayes alpha=0.01; unigram; countvectorizer; no stemmer; stopper; feature selection - SelectPercentile=80'
    ch2 = SelectPercentile(chi2, percentile=80)
    X_train = ch2.fit_transform(count_u_train_no_stem_stop, train_no_stem_stop.target)
    X_test = ch2.transform(count_u_test_no_stem_stop)
    clf = MultinomialNB(alpha=0.01)
    clf.fit(X_train, train_no_stem_stop.target)
    pred = clf.predict(X_test)
    res.append({'name': name, 'clf': clf, 'pred': pred, 'target': test_no_stem_stop.target})
    
    name = 'Naive Bayes alpha=0.01 fit_prior=False; unigram; tfifdfvectorizer; stemmer; stopper; feature selection - SelectPercentile=80'
    ch2 = SelectPercentile(chi2, percentile=80)
    X_train = ch2.fit_transform(tfid_u_train_stem_stop, train_stem_stop.target)
    X_test = ch2.transform(tfid_u_test_stem_stop)
    clf = MultinomialNB(alpha=0.01, fit_prior=False)
    clf.fit(X_train, train_stem_stop.target)
    pred = clf.predict(X_test)
    res.append({'name': name, 'clf': clf, 'pred': pred, 'target': test_stem_stop.target})

    name = 'Naive Bayes alpha=0.01; unigram; countvectorizer; no stemmer; stopper; feature selection - SelectFromModel threshold=39'
    clf = MultinomialNB(alpha=0.01)
    clf.fit(count_u_train_no_stem_stop, train_no_stem_stop.target)
    model = SelectFromModel(clf, threshold=30, prefit=True)
    X_new = model.transform(count_u_train_no_stem_stop)
    X_new_test = model.transform(count_u_test_no_stem_stop)
    clf = MultinomialNB(alpha=0.01)
    clf.fit(X_new, train_no_stem_stop.target)
    pred = clf.predict(X_new_test)
    res.append({'name': name, 'clf': clf, 'pred': pred, 'target': test_no_stem_stop.target})

    name = 'Naive Bayes alpha=0.01 fit_prior=False; unigram; tfidfvectorizer; stemmer; stopper; feature selection - SelectFromModel threshold=30'
    clf = MultinomialNB(alpha=0.01, fit_prior=False)
    clf.fit(tfid_u_train_stem_stop, train_stem_stop.target)
    model = SelectFromModel(clf, threshold=30, prefit=True)
    X_new_train = model.transform(tfid_u_train_stem_stop)
    X_new_test = model.transform(tfid_u_test_stem_stop)
    clf = MultinomialNB(alpha=0.01, fit_prior=False)
    clf.fit(X_new_train, train_stem_stop.target)
    pred = clf.predict(X_new_test)
    res.append({'name': name, 'clf': clf, 'pred': pred, 'target': test_stem_stop.target})
    # This is the best model
    best_model = name

    rows = []
    headers = ['No.', 'Name', 'Precision', 'Recall', 'Precision/Recall', 'F1 Score']
    for i, val in enumerate(res):                                                                                                                                               
        precision = metrics.precision_score(val['target'], val['pred'], average='macro')                                                                          
        recall    = metrics.recall_score(val['target'], val['pred'], average='macro')                                                                             
        f1_score  = metrics.f1_score(val['target'], val['pred'], average='macro')
        row = [i+1, val['name'], precision, recall, precision/recall, f1_score]
        rows.append(row)
    print ('\nResults - ')
    print tabulate(rows, headers, tablefmt='orgtbl')
    print('\n')

    print 'Best model is ->', best_model

if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == '--help':
        print __doc__
        sys.exit()
    main()
