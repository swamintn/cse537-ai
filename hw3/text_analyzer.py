#!/usr/bin/env python
"""
Script used to analyze different Machine Learning-based text classifiers

First argument must be location of train & test data, second argument must be an
image filename for which the output plots must be created. The data location
must have 'Training' and 'Test' subdirectories and should be in a format
readable by sklearn.datasets.load_files.

>> text_analyzer.py <Dir of training+test data> <Output plot filename>

Example:
    text_analyzer.py /user/Selected_20NewsGroup /user/output_plots.png

The script currently analyzes these models,
    1) Naive Bayes
    2) Random Forest
    3) Logistic Regression
    4) SVM (LinearSVC)

(Analysis is done for both unigram and bigram splits)

@author: Swaminathan Sivaraman
"""

import sklearn
from   sklearn import datasets
from   sklearn import svm
from   sklearn import metrics

from   sklearn.feature_extraction.text import CountVectorizer
from   sklearn.datasets                import load_files
from   sklearn.externals               import joblib

from   sklearn.ensemble                import RandomForestClassifier
from   sklearn.naive_bayes             import MultinomialNB
from   sklearn.svm                     import LinearSVC
from   sklearn.linear_model            import LogisticRegression

from   sklearn.model_selection         import learning_curve

from   matplotlib import pyplot as plt
from   matplotlib import colors

import numpy as np
import os
import sys


def remove_headers(t_data):
    for i in range(len(t_data.data)):
        lines = t_data.data[i].splitlines()
        for j, line in enumerate(lines):
            if ':' not in line:
                t_data.data[i] = '\n'.join(lines[j+1:])
                break

def load_files_correctly(loc):
    t_data = load_files(loc)
    remove_headers(t_data)
    return t_data


def main():
    data_loc, out_file = sys.argv[1:]

    # Get data
    train = load_files_correctly(os.path.join(data_loc, 'Training'))
    test  = load_files_correctly(os.path.join(data_loc, 'Test'))

    # Vectorize data
    # Unigram
    u_vec = CountVectorizer(ngram_range=(1,1), decode_error='ignore')
    u_train_data = u_vec.fit_transform(train.data)
    u_test_data  = u_vec.transform(test.data)

    # Bigram
    b_vec = CountVectorizer(ngram_range=(1,2), decode_error='ignore')
    b_train_data = b_vec.fit_transform(train.data)
    b_test_data  = b_vec.transform(test.data)

    # Results
    res = {}

    for i in range(2):
        if i == 0:
            train_data, test_data, tag = u_train_data, u_test_data, ' Unigram '
        else:
            train_data, test_data, tag = b_train_data, b_test_data, ' Bigram '

        # Naive Bayes
        name = 'Naive Bayes' + tag
        print 'Processing', name
        clf = MultinomialNB() # alpha = 0.01
        clf.fit(train_data, train.target)
        pred = clf.predict(test_data)
        res[name] = {'clf': clf, 'pred': pred}

        # Random Forest
        name = 'Random Forest' + tag
        print 'Processing', name
        clf = RandomForestClassifier() # n_estimators = 20
        clf.fit(train_data, train.target)
        pred = clf.predict(test_data)
        res[name] = {'clf': clf, 'pred': pred}

        # SVM
        name = 'SVM' + tag
        print 'Processing', name
        clf = LinearSVC()
        clf.fit(train_data, train.target)
        pred = clf.predict(test_data)
        res[name] = {'clf': clf, 'pred': pred}

        # Logistic Regression
        name = 'Logistic Regression' + tag
        print 'Processing', name
        clf = LogisticRegression()
        clf.fit(train_data, train.target)
        pred = clf.predict(test_data)
        res[name] = {'clf': clf, 'pred': pred}

    # Print
    headers = ['Name', 'Precision', 'Recall', 'Precision/Recall', 'F1 Score']
    rows = []
    for key in sorted(res):
        val = res[key]
        precision = metrics.precision_score(test.target, val['pred'], average='macro')
        recall    = metrics.recall_score(test.target, val['pred'], average='macro')
        f1_score  = metrics.f1_score(test.target, val['pred'], average='macro')
        rows.append([key, precision, recall, precision/recall, f1_score])

    print ('\nResults - ')
    from tabulate import tabulate
    print tabulate(rows, headers, tablefmt='orgtbl')

    # plotting
    print '\nPlotting learning curves...'
    step_size = 200.0
    train_size = u_train_data.shape[0]
    data = {}
    for i in range(int(np.ceil(train_size / step_size))):
        end = (i+1) * step_size
        size_to_use = int(end if end < train_size else train_size)
        for j in range(2):
            if j == 0:
                train_data, test_data, tag = u_train_data, u_test_data, ' Unigram'
            else:
                train_data, test_data, tag = b_train_data, b_test_data, ' Bigram'
            
            # Naive Bayes
            name = 'Naive Bayes' + tag
            # print 'Processing', name, 'train_data: ', size_to_use
            clf = MultinomialNB() # alpha = 0.01
            clf.fit(train_data[:size_to_use], train.target[:size_to_use])
            pred = clf.predict(test_data)
            f1_score = metrics.f1_score(test.target, pred, average='macro')
            data.setdefault(name, {'sizes': [], 'f1_scores': []})
            data[name]['sizes'].append(size_to_use)
            data[name]['f1_scores'].append(f1_score)

            # Random Forest
            name = 'Random Forest' + tag
            # print 'Processing', name, 'train_data: ', size_to_use
            clf = RandomForestClassifier() # n_estimators = 20
            clf.fit(train_data[:size_to_use], train.target[:size_to_use])
            pred = clf.predict(test_data)
            f1_score = metrics.f1_score(test.target, pred, average='macro')
            data.setdefault(name, {'sizes': [], 'f1_scores': []})
            data[name]['sizes'].append(size_to_use)
            data[name]['f1_scores'].append(f1_score)

            # SVM
            name = 'SVM' + tag
            # print 'Processing', name, 'train_data: ', size_to_use
            clf = LinearSVC()
            clf.fit(train_data[:size_to_use], train.target[:size_to_use])
            pred = clf.predict(test_data)
            f1_score = metrics.f1_score(test.target, pred, average='macro')
            data.setdefault(name, {'sizes': [], 'f1_scores': []})
            data[name]['sizes'].append(size_to_use)
            data[name]['f1_scores'].append(f1_score)

            # Logistic Regression
            name = 'Logistic Regression' + tag
            # print 'Processing', name, 'train_data: ', size_to_use
            clf = LogisticRegression()
            clf.fit(train_data[:size_to_use], train.target[:size_to_use])
            pred = clf.predict(test_data)
            f1_score = metrics.f1_score(test.target, pred, average='macro')
            data.setdefault(name, {'sizes': [], 'f1_scores': []})
            data[name]['sizes'].append(size_to_use)
            data[name]['f1_scores'].append(f1_score)

    fig = plt.figure()
    plt.title('Learning curves')
    plt.ylim(0.4, 0.95)
    plt.xlabel("Size of training data (step_size=%s)" % int(step_size))
    plt.ylabel("F1 Score")
    plt.grid()
    clrs = ['red', 'blue', 'green', 'violet', 'brown', 'orange', 'aqua', 'magenta']
    for name, clr in zip(sorted(data), clrs):
        sizes, f1_scores = data[name]['sizes'], data[name]['f1_scores']
        color = colors.cnames[clr]
        plt.plot(sizes, f1_scores, 'o-', color=color, label=name)
    lgd = plt.legend(loc=2, bbox_to_anchor=(0.5,-0.1))
    fig.savefig(out_file, bbox_extra_artists=(lgd,), bbox_inches='tight')

         
if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == '--help':
        print __doc__
        sys.exit()
    main()
