#!/usr/bin/env python

from const import *
from preprocess import load_data

import pickle

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier

from optparse import OptionParser
parser = OptionParser()

parser.add_option("-c", "--class", default="rdf", help="choose a classifier among bag, rdf, ada, grd",
                  action="store", type="string", dest="ensemble")

parser.add_option("-s", "--size", default=1000, help="config the data set size",
                    action="store", type="int", dest="size")

(options, args) = parser.parse_args()

if options.size != DATASET_SIZE:
    dataset_size = options.size
else:
    dataset_size = DATASET_SIZE

bag_clf = BaggingClassifier( KNeighborsClassifier(), 
    max_samples=0.5, max_features=0.5)

ada_clf = AdaBoostClassifier(n_estimators=5)

rdf_clf = RandomForestClassifier(n_estimators=5, criterion="gini", 
    max_depth=None, min_samples_split=2, min_samples_leaf=1, 
    min_weight_fraction_leaf=0., max_features="auto", 
    max_leaf_nodes=None, min_impurity_split=1e-7, bootstrap=True, 
    oob_score=False, n_jobs=1,random_state=None,verbose=0, 
    warm_start=False, class_weight=None)

# grd_clf = GradientBoostingClassifier(n_estimators=10, 
#     learning_rate=0.1, max_depth=1, random_state=0)
grd_clf = XGBClassifier()

clf = None
if options.ensemble == 'bag':
    clf = bag_clf
elif options.ensemble == 'ada':
    clf = ada_clf
elif options.ensemble == 'rdf':
    clf = rdf_clf
elif options.ensemble == 'grd':
    clf = grd_clf
else: # default choice
    clf = ada_clf

print 'Using', options.ensemble, ' method'

def ensemble(x_train_tfidf, target, target_names):
    print 'Data size:', len(target)
    # kf = KFold(n_splits=10, shuffle=False)
    x = x_train_tfidf.toarray()
    y = np.array(target)

    if options.ensemble == 'grd':
        clf.fit(x, y)
        x_test = x[:100]
        y_test = y[:100]
        print clf.score(x_test, y_test)
    elif options.ensemble == 'grd':
        # if os.path.exists():
            
        clf.fit(x, y)
        pickle.dump(clf, open('xgb.pkl', 'wb'))
    else:
        scores = cross_val_score(clf, x, y)
        print scores.mean()
    

def main():
    global dataset_size
    (x_train_tfidf, target, target_names) = load_data()
    dataset_size = min(len(target), dataset_size)
    print 'Data set size:', dataset_size
    ensemble(x_train_tfidf[:dataset_size], target[:dataset_size], target_names[:dataset_size])


if __name__ == '__main__':
    main()
