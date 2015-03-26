# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 18:54:08 2015

Training Classifiers and Determining Informative Features
===============================================================================

This file exhibits the use of functions from region_prediction_functions.py on
a Twitter data set collected between January and February 2015. It shows the
main results of the regional words project.

@author: gabrielmcneill
"""

import region_prediction_functions as rpg
import os
import pickle
from time import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.feature_selection import GenericUnivariateSelect



###############################################################################
# GLOBAL VARIABLES
###############################################################################

MY_REGIONS = ["West", "LWest", "UWest", "Midwest", "South", "East"]

SIX_REGION_NAMES = {1: "West", 2: "LWest", 3: "UWest", 4: "Midwest", 
                    5: "South", 6: "East"}

CLASSIFIER_LIST = ['MultinomialNB', 
                   'RandomizedLogisticRegression',
                   'Perceptron',
                   'PassiveAggressiveClassifier',
                   'SGDClassifier']



###############################################################################
# CREATE CLASSIFIERS
###############################################################################

# tweetdata2262015.p (data file) should be in the current directory.
raw_tweet_path = os.environ.get('PWD', None) + '/tweetdata2262015.p'
raw_tweet_data = pickle.load(open(raw_tweet_path, "rb"))


tfidf_vectz = TfidfVectorizer(stop_words='english', strip_accents='unicode', 
                              ngram_range=(1,3), max_df=1.0, min_df=2, 
                              max_features=None, binary=False, norm=u'l2',
                              use_idf=True, smooth_idf=True, 
                              sublinear_tf=True,
                              tokenizer=rpg.SnowballEnglishStemmer())


tweet_data = rpg.tweet_corpus_maker(raw_tweet_data)  # form usable by sklearn
binary_region = tweet_data[2]    # selecting the East/West binary splits

initial_time = time()
X_tfidf = tfidf_vectz.fit_transform(tweet_data[0])  # tfidf vectorization
tfidf_time = time() - initial_time
print ("TFIDF Vectorization Time: %0.3f" % tfidf_time)

feature_selector = GenericUnivariateSelect(chi2, mode='percentile', param=25)
X_selected = feature_selector.fit_transform(X_tfidf, binary_region)

# This splits data into training/test splits etc.
clf_tests = rpg.make_full_data_test_samples(targets=binary_region, 
                                            vectz_dict = {'tfidfV': X_selected})

# Fits classifiers and prints evaluation metrics.
clf_bench = rpg.test_classifiers(data=clf_tests, clfs=CLASSIFIER_LIST, 
                                 vectz='tfidfV', ssize=len(binary_region), 
                                 scoring=['roc_auc'])

# Prints summary of ROC AUC estimate from 10-fold cross validation, and 
# a number of best features of each classifier.
top_feats = rpg.print_classifier(clfs=clf_bench, vectz=tfidf_vectz, 
                                 feat_sel=feature_selector, num_feats=20)

