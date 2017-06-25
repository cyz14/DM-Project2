#!/usr/bin/env python

from const import *
from preprocess import load_data

import numpy as np

from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, GridSearchCV

# classifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

from optparse import OptionParser
parser = OptionParser()

parser.add_option("-c", "--class", default="lgr", help="choose a classifier among lgr, nb, sgd, dct and mlp",
                  action="store", type="string", dest="clf")

parser.add_option("-s", "--size", default=1000, help="config the data set size",
                    action="store", type="int", dest="size")

(options, args) = parser.parse_args()

if options.size != DATASET_SIZE:
    dataset_size = options.size
else:
    dataset_size = DATASET_SIZE

# Classifiers

clf = None

# Logistic regression
lgr_clf = Pipeline([('clf', LogisticRegression(penalty='l2', 
    dual=False, tol=1e-4, C=1.0, fit_intercept=True, intercept_scaling=1, 
    class_weight=None, random_state=None, max_iter=100, multi_class='ovr', 
    verbose=0, warm_start=True, n_jobs=1))])

# Naive bayes
nb_clf = MultinomialNB()

# SVM
sgd_clf = Pipeline([('clf', SGDClassifier(loss='hinge', penalty='l2',
                        alpha=1e-3, n_iter=5, random_state=42)),])

# Decision tree
dct_clf = DecisionTreeClassifier(criterion="gini", splitter="best", 
    max_depth=None, min_samples_split=2, min_samples_leaf=1, 
    min_weight_fraction_leaf=0., max_features=None, random_state=None, 
    max_leaf_nodes=None, min_impurity_split=1e-7, class_weight=None, 
    presort=False)

# MLP
mlp_clf = MLPClassifier(hidden_layer_sizes=(128,), 
    activation="relu", solver='adam', alpha=0.0001, 
    batch_size='auto', learning_rate="constant", 
    learning_rate_init=0.001, power_t=0.5, 
    max_iter=200, shuffle=True, random_state=None, 
    tol=1e-4, verbose=False, warm_start=False, 
    momentum=0.9, nesterovs_momentum=True, early_stopping=False, 
    validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-8)

if options.clf == 'lgr':
    clf = lgr_clf
elif options.clf == 'nb':
    clf = nb_clf
elif options.clf == 'sgd':
    clf = sgd_clf
elif options.clf == 'dct':
    clf = dct_clf
elif options.clf == 'mlp':
    clf = mlp_clf
else:
    clf = lgr_clf # default classifier

print 'Using', options.clf, 'classifier'

def evaluate(x_train_tfidf, target, target_names):
    print 'Data size:', len(target)
    kf = KFold(n_splits=10, shuffle=False)
    x = x_train_tfidf
    y = np.array(target)

    training_iter = 0
    acc_list = []
    for train_index, test_index in kf.split(x):
        print 'Training iter:', training_iter
        training_iter += 1
        x_train, x_test = x_train_tfidf[train_index], x_train_tfidf[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Logistic Regression
        # lgr_clf.fit(x_train, y_train)        
        # predicted = lgr_clf.predict(x_test)

        # Naive Bayes
        # nb_clf.fit(x_train, y_train)
        # predicted = nb_clf.predict(x_test)
        
        # SVM
        clf.fit(x_train, y_train)
        predicted = clf.predict(x_test)

        # Desicion Tree
        # dct_clf.fit(x_train, y_train)
        # predicted = dct_clf.predict(x_test)

        # MLP
        # mlp_clf.fit(x_train, y_train)
        # predicted = mlp_clf.predict(x_test)

        acc_list.append(np.mean(predicted == y_test))
        print(metrics.classification_report(y_test, predicted, 
            target_names=target_names))
        print acc_list[-1]
    
    print '# Total avg acc rate is:', np.mean(acc_list) * 100, '%'


def main():
    global dataset_size
    (x_train_tfidf, target, target_names) = load_data()
    dataset_size = min(len(target), dataset_size)
    print 'Data set size:', dataset_size
    evaluate(x_train_tfidf[0:dataset_size], target[0:dataset_size], target_names[0:dataset_size])
    

if __name__ == '__main__':
    main()