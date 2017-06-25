#!/usr/bin/env python
import time

from const import *
from preprocess import load_data

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from optparse import OptionParser
parser = OptionParser()

parser.add_option("-c", "--class", default="kms", help="choose a cluster between kms and dbs",
                  action="store", type="string", dest="cluster")

parser.add_option("-s", "--size", default=1000, help="config the data set size",
                    action="store", type="int", dest="size")

(options, args) = parser.parse_args()

if options.size != DATASET_SIZE:
    dataset_size = options.size
else:
    dataset_size = DATASET_SIZE

n_of_clusters = 10

kms_cluster = KMeans(n_clusters=n_of_clusters, init='k-means++', n_init=5,
    max_iter=300, tol=0.0001, precompute_distances='auto', verbose=1,
    random_state=None, copy_x=True, n_jobs=1, algorithm='auto')

dbs_cluster = DBSCAN(eps=0.1, min_samples=5, metric='euclidean',
    algorithm='auto', leaf_size=30, p=None, n_jobs=1)

colors = np.array([x for x in 'bgrcmykbgrcmyk'])
colors = colors
pca = PCA(n_components = 10)
tsne = TSNE(n_components = 2)

cluster = None
if options.cluster == 'dbs':
    cluster = dbs_cluster
else: # default choice
    cluster = kms_cluster

print 'Using', options.cluster, ' method'

def clustering(x_train_tfidf, target, target_names):
    global cluster
    print 'Data size:', len(target)
    # kf = KFold(n_splits=10, shuffle=False)
    x = x_train_tfidf.toarray()
    y = np.array(target)

    # Dim reduct
    print 'PCA begin...'
    pcaData = pca.fit_transform(x)
    print 'PCA ends!'

    t0 = time.time()
    result = cluster.fit(pcaData)
    t1 = time.time()
    print '# NMI is: ', normalized_mutual_info_score(y, result.labels_)
    print '# AMI is: ', adjusted_mutual_info_score(y, result.labels_)

    # Dim reduct
    print 'TSNE begin...'
    tsneData = tsne.fit_transform(pcaData)
    print 'TSNE ends!'

    # plot
    plt.scatter(tsneData[:, 0], tsneData[:, 1], color=colors[result.labels_].tolist(), s=10)

    plt.xticks(())
    plt.yticks(())
    plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
             transform=plt.gca().transAxes, size=15,
             horizontalalignment='right')
    plt.show()

def main():
    global dataset_size
    (x_train_tfidf, target, target_names) = load_data()
    dataset_size = min(len(target), dataset_size)
    print 'Data set size:', dataset_size
    clustering(x_train_tfidf[:dataset_size], target[:dataset_size], target_names[:dataset_size])


if __name__ == '__main__':
    main()
