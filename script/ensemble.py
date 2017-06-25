#!/usr/bin/env python

from const import *
from preprocess import load_data

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier


dct_clf = DecisionTreeClassifier(max_depth=None, min_samples_split=2, 
    random_state=0)

rdf_clf = RandomForestClassifier(n_estimators=10, criterion="gini", 
    max_depth=None, min_samples_split=2, min_samples_leaf=1, 
    min_weight_fraction_leaf=0., max_features="auto", 
    max_leaf_nodes=None, min_impurity_split=1e-7, bootstrap=True, 
    oob_score=False, n_jobs=1,random_state=None,verbose=0, 
    warm_start=False, class_weight=None)


def ensemble(x_train_tfidf, target, target_names):
    print 'Data size:', len(target)
    # kf = KFold(n_splits=10, shuffle=False)
    x = x_train_tfidf
    y = np.array(target)

    scores = cross_val_score(rdf_clf, x, y)
    print scores.mean()
    

def main():
    (x_train_tfidf, target, target_names) = load_data()
    ensemble(x_train_tfidf[0:DATASET_SIZE], target[:DATASET_SIZE], target_names[:DATASET_SIZE])
    

if __name__ == '__main__':
    main()
