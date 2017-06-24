#!/usr/bin/env python

from const import *

import os
import glob
import codecs
import shutil
from datetime import date

import pickle
import numpy as np
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

if not os.path.exists(trash_dir):
    os.mkdir(trash_dir)

file_names = sorted(glob.glob(corpus_dir + '/*.xml'))
print 'File number:', len(file_names)

extra_num = 0

def getTextEntry(soup, file_name):
    global extra_num
    # (1) Find text
    text_node = soup.find('block', 'full_text')
    if text_node == None:
        extra_num += 1
        fname = os.path.basename(file_name)
        print fname
        shutil.move(file_name, os.path.join(trash_dir, fname))
        return None, None, None
    
    full_text = text_node.p.string

    # (2) Find Date    
    year = int(soup.find('meta', {'name':'publication_year'})['content'])
    month = int(soup.find('meta', {'name':'publication_month'})['content'])
    day = int(soup.find('meta', {'name': 'publication_day_of_month'})['content'])
    # weekday = soup.find('meta', {'name':'publication_day_of_week'})['content']
    dttime = date(year, month, day)
    # print dttime

    # (3) Find Class
    classifier = soup.find('classifier', {'type':'taxonomic_classifier'})
    if classifier == None:
        extra_num += 1
        print file_name
        fname = os.path.basename(file_name)
        shutil.move(file_name, os.path.join(trash_dir, fname))
        return None, None, None

    classifier = classifier.string
    topics = classifier.split('/')
    if len(topics) == 0:
        print 'Warning! topic missing'
        return None, None, None
    
    topic = topics[-1]
    return full_text, dttime, topic


def preprocess():
    corpus = []
    target_names = []
    target = []

    for index, file_name in enumerate(file_names):
        if index > DATASET_SIZE:
            break
        file = codecs.open(file_name, encoding="utf-8").read()
        soup = BeautifulSoup(file, 'xml')
        full_text, dttime, topic = getTextEntry(soup, file_name)
        if full_text == None:
            continue
        corpus.append(full_text)
        if not topic in target_names:
            target_names.append(topic)
        target.append(target_names.index(topic))
        if index % 1000 == 0:
            print index, 'processed.'

    print extra_num
    return corpus, target, target_names


def get_tfidf(corpus):
    count_vect = CountVectorizer()
    tfidf_transformer = TfidfTransformer()
    x_train_counts = count_vect.fit_transform(corpus)
    x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)
    return x_train_tfidf


def load_data(filename=None):
    x_train_tfidf = None
    target        = None
    target_names  = None

    tfidf_filename = 'tfidf.pkl'
    if filename != None:
        tfidf_filename = filename
    
    if not os.path.exists(tfidf_filename):
        fw = open(tfidf_filename, 'wb')
        corpus, target, target_names = preprocess() # print len(corpus), len(target)
        x_train_tfidf = get_tfidf(corpus)
        print x_train_tfidf.shape
        data = (x_train_tfidf, target, target_names)
        pickle.dump(data, fw)
        fw.close()
    else:
        fr = open(tfidf_filename, 'rb')
        (x_train_tfidf, target, target_names) = pickle.load(fr)
        fr.close()

    return (x_train_tfidf, target, target_names)


def main():
    (x_train_tfidf, target, target_names) = load_data()
    print len(target)


if __name__ == '__main__':
    main()
