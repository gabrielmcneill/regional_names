# -*- coding: utf-8 -*-
"""
Created on Fri Feb 27 14:53:16 2015

Functions To Predict Regions From Tweet Text
===============================================================================

All functions used in the analysis phase of the regional words project are 
contained in this file.

@author: gabrielmcneill
"""
from __future__ import division

from collections import Counter
import random
import re
from time import time

import matplotlib.pyplot as plt

from nltk.tokenize import word_tokenize
from nltk.stem.snowball import EnglishStemmer

import numpy as np

from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectPercentile
from sklearn.lda import LDA
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import RandomizedLogisticRegression
from sklearn.linear_model import RidgeClassifierCV
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.qda import QDA
from sklearn.utils.extmath import density


###############################################################################
# Defined Classes
###############################################################################

class SnowballEnglishStemmer(object):
    """Stemmer class tokenizes and stems English strings.
    
    Used to create a class object whose __call__() method is used to take a 
    document containing a list of strings, split the strings into smaller
    substrings, and output them all as a list of tokenized and stemmed text.
    """
    def __init__(self):
        self.snes = EnglishStemmer()

    def __call__(self, document):
        ### any documentation on what that regex does?
        return filter(None, 
                      [self.snes.stem(token) 
                      for token in word_tokenize(document.decode('utf-8'))
                      if not re.search(("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])"
                                                  "|(\w+:\/\/\S+)"), token)])


###############################################################################
# Defined Functions
###############################################################################


def tweet_corpus_maker(rawdata):
    """Extracts tweet data from a list containing dicts returned from Twython.
    
    Outputs the strings corresponding to tweet text and the integer codes
    corresponding to US regions in a form that can be easily used with 
    scikit-learn functions.
    
    Input: (1) rawdata (list): elements are dicts with tweet data.
    
    Output: (1) (list): elements are strings (tweets).
            (2) (numpy array): elements are integers encoding 6 regions.
            (3) (numpy array): elements are integers, 0 East, and 1 West.
    
    Examples: tweet_corpus = tweet_corpus_maker(Regworddata)
    """
    tweetstrings = []
    tweetregions = []
    westregions = []
    
    for datum in rawdata:
        if datum['west'] is not None:   #This filters out unusable tweets
            tweetstrings.append(datum['text'])
            tweetregions.append(datum['region'])
            westregions.append(datum['west'])
    
    return (tweetstrings, np.array(tweetregions), np.array(westregions))

def prob_ratio(log_probs):
    """Takes 2-element numpy array of log probs and converts to ratio of 
       probabiltity for the positive class compared to probability for the 
       negative class. """       
    return np.exp(log_probs[1,:] - log_probs[0,:])


def best_features(vectorizer, feat_selector, clf, num_top_feat):
    """Prints top-ranked features of given classification method.
    
    In its current version this function requires that one use feature 
    selection for it to work.
    Input: (1) vectorizer (vectorizer object): instantiation of a 
           sklearn.feature_extraction.text vectorizer from scikit-learn, 
           (e.g. cVectorizer).
           (2) feat_selector (feature_selection object): instantiation of a 
           scikit-learn feature_selection object such as SelectPercentile()
           (3) clf (classifier object): instantiation of a scikit-learn 
           classifier (e.g. NB_clf).
           (4) num_top_feat (int): the desired number of displayed features.
           
    Output: (1) (dict) contains the best num_top_feat features of a given 
                classifier.
    
    Examples: ### add example or delete
    """
    assert hasattr(vectorizer, 'get_feature_names'), ("1st arg. must have "
                                                      "feature names")
    assert hasattr(feat_selector, 'get_support'), ("2nd arg. must have a"
                                                      " get_support() method.") 
    
    ### one-line documentation to explain this?
    has_features = np.asarray(
                   vectorizer.get_feature_names())[feat_selector.get_support()]
    
    print ("Density of clf: %s" % density(clf.coef_))
    print ("Classifier Length: %s" % clf.coef_.size)
    best_indices = {}        
    if clf.coef_.shape[0] == 1:    # in other words, if binary targets
                        
        if hasattr(clf, 'feature_log_prob_'):
            # probability for the positive class divided by 
            # probability for the negative class
            log_probs = clf.feature_log_prob_
            crit = np.exp(log_probs[1,:] - log_probs[0,:])            
            crit_label = 'P(West)/P(East)'

        else:
            crit = clf.coef_[0,:]
            crit_label = 'coef'

        indices = np.argsort(crit)
        
        lowest_indices = indices[:num_top_feat]
        best_indices['East'] = lowest_indices
        east_features = has_features[lowest_indices]
        east_values = crit[lowest_indices]

        highest_indices = indices[-num_top_feat:][::-1]
        best_indices['West'] = highest_indices
        west_features = has_features[highest_indices]
        west_values = crit[highest_indices]

        top_features = {
            'East': {'features': east_features, 'values': east_values},
            'West': {'features': west_features, 'values': west_values},
        }

        for region in top_features:
            print
            print("Best %s %s features (best to worst), "
                "%s:" % (num_top_feat, region, crit_label))
            
            features = top_features[region]['features']
            values = top_features[region]['values']
            for feature, value in zip(features, values):
                print "%s:  %0.5f" % (feature, value)

    else:     # more than two categories
        best_indices = {}
        print ("Top %s features are: " % num_top_feat)
        for i in clf.classes_.tolist():
            best_indices[i] = np.argsort(clf.coef_[i,:])[-num_top_feat:]
            best_features = has_features(best_indices[i])
            
            print "%s: %s" % (i, ", ".join(best_features))

    return best_indices

def print_classifier(clfs, vectz, feat_sel, num_feats):
    """Prints summary of ROC AUC estimated on test data with cross-validation.
    
    Used mainly to print out and save information about the efficacy of
    different classifiers, but also prints out a number of the best features 
    that each classifier uses.
    
    Input: (1) clfs (dict): output of test_classifiers() function.
           (2) vectz (sklearn vectorizer object): instantiation of vectorizer
               object used to convert the raw text data into the bag-of-words
               model.
           (3) feat_sel (sklearn feature_selection object): instantiation of 
               feature_selection object used to select features of the data
               that were used to fit the classifiers.
           (4) num_feats (int): number of best features of each classifier to 
               display.
    
    Output: (1) best_clf_features (dict): numpy arrays of best features for the
                various regions, for each classifier.
    
    Examples: top_feats = print_classifier(clf_bench, tfidf_vectz, 
                                                          feature_selector, 20)
    """
    best_clf_features = {}
    for clf_name in clfs:
        print("%s    %0.3f    %0.3f +/- %0.3f" % (clf_name, clfs[clf_name][0],
                                                  clfs[clf_name][1], 
                                                  clfs[clf_name][2]))
        best_clf_features[clf_name] = best_features(vectz, feat_sel, 
                                                    clfs[clf_name][3], 
                                                    num_feats)
        print
    
    return best_clf_features

def classification_eval(training_covariates, testing_covariates, 
                        testing_targets, predicted_values, scoring_metrics, 
                        cross_validation_scores, confusion_matrix, 
                        number_of_regions):
    """Prints classifier metrics and evaluation information.
    
    Input: (1) training_covariates: (scipy.sparse.csr.csr_matrix) bag-of-words
           representation of covariates in training split. 
           (2) testing_covariates: (scipy.sparse.csr.csr_matrix) bag-of-words
           representation of covariates in testing split.
           (3) testing_targets: (numpy.ndarray) true region labels.
           (4) predicted_values: (numpy.ndarray) region predictions of fitted 
           classifier.
           (5) scoring_metrics: (list) contains strings with scoring metric 
           names usable by sklearn.cross_validation.cross_val_score().
           (6) cross_validation_scores: (dict) keys are strings of scoring
           metric names, and values are lists of numpy ndarrays containing
           the scores of the metrics given in (5) scoring metrics.
           (7) confusion_matrix: (numpy.ndarray) matrix of classification
           counts by region label, produced by 
           sklearn.metrics.confusion_matrix.
           (8) number_of_regions: (int) the number of distinct region labels in
           targets.
           
    Output: None if roc_auc is not in scoring_metrics
            (tuple) (1) cross_validation_scores['roc_auc'].mean() is the mean 
                    ROC area under the curve of the cross-validation scores.
                    (2) 2*cross_validation_scores['roc_auc'].std() is 2 times 
                    the standard error of the ROC area under the curve scores
                    determined by cross-validation.
                    
    Examples: ### give one or delete
    """
    ### remove them
    # All of the try/except statements make debugging issues easier
    try:    
        print("Training Data Shape: {}".format(np.shape(training_covariates)))
    except Exception, e:
        print ("TrainDataShapeException: %s" % e)

    try:
        print("Test Data Shape: {}".format(np.shape(testing_covariates)))
    except Exception, e:
        print ("TestDataShapeException: %s" % e)
        print

    try:
        print("Mean Correct Predictions: %0.4f" % 
              np.mean(predicted_values == testing_targets))
    except Exception, e:
        print ("MeanCorrectPredictionsException: %s" % e)

    if number_of_regions == 2:
        try:
            print ("ROC Area Under Curve (test data): %0.4f" % 
                   roc_auc_score(testing_targets, predicted_values, 
                                 average='weighted'))
        except Exception, e:
            print ("ROCAUCScoreException: %s" % e)

        try:
            print ("Average Precision Score: %0.4f" % 
                   average_precision_score(testing_targets, predicted_values, 
                                           average='weighted'))
        except Exception, e:
            print ("AveragePrecisionScoreException: %s" % e)

    # Confusion Matrix with Matplotlib
    try:
        plt.matshow(confusion_matrix, cmap=plt.cm.Greys)
        plt.title('Confusion Matrix')
        plt.colorbar()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()
    except Exception, e:
        print ("MatplotlibConfusionMatrixException: %s" % e)

    print ("Confusion Matrix Counts: ")
    print(confusion_matrix)

    for metric in scoring_metrics:
        try:
            print("mean %s CV score: %0.3f (+/-2SE: %0.3f)" % 
                  (metric, cross_validation_scores[metric].mean(), 
                   2*cross_validation_scores[metric].std()))
        except Exception, e:
            print ("cv%sScoreException: %s" % (metric, e))

    if 'roc_auc' in scoring_metrics:
        try:
            return (cross_validation_scores['roc_auc'].mean(), 
                    2*cross_validation_scores['roc_auc'].std())
        except Exception, e:
            print ("ReturnROCAUCScoresException: %s" % e)
    else:
        return None

def make_prediction(prediction_strings, classifier, vectorizer, 
                    feature_transform=None):
    """Generates predictions of region from arbitrary text.
    
    Input: (1) prediction_strings (list): contains strings of text to make
           predicitons from.
           (2) classfier: instance of classifier object with predict method
           (3) vectorizer: instance of vectorizer used to process test data
           (4) feature_transform: an object with .transform method, used to 
           perform a second transformation if necessary (as when one performs
           feature selection).
           
    Output: (numpy.ndarray) target labels corresponding to classifier
            predictions.
    
    Examples:
    """
    assert type(prediction_strings) == list, ("prediction_strings must be a "
                                              "list.")
    assert hasattr(classifier, 'coef_'), ("2nd arg. must have a .predict "
                                          "method.")
    assert hasattr(vectorizer, 'transform'), ("3rd arg. must have a .transform"
                                              "method.")

    transformed_data = vectorizer.transform(prediction_strings)
    if feature_transform is not None:
        assert hasattr(feature_transform, 'transform'), ("4th arg. must have"
            " .transform method.")
        
        retransformed_data = feature_transform.transform(transformed_data)
        return(classifier.predict(retransformed_data))

    return(classifier.predict(transformed_data))

def find_classifier(covariates, targets, vectorizer, sel_pct, alphapar, 
                    class_names, num_feat, scoring_metrics, cv_fold):
    """Simultaneously fits classifiers and displays evaluation metrics.
    
    Input: (1) covariates: The output of 
           "vectorizer".fit_transform(tweetCorpus)
           (2) targets: (numpy.ndarray) all target labels (int).
           (3) vectorizer: instantiation of sklearn.feature_extraction.text
               vectorizer.
           (4) sel_pct: Percentage of features to keep after selection.
           (5) alphapar: (float) alpha smoothing parameter of Naive Bayes 
               classifier.
           (6) class_names: (dict) keys are integers corresponding to regional
               labels used in targets; values are strings indicating region 
               name.
           (7) num_feat: (int) number of features to retain (USED???)
           (8) scoring_metrics: (list) of strings usable by cross_val_score(),
               which determine the classification metrics.
           (9) cv_fold: (int) number that determines what fold of
               cross-validation to use.
           
    Ouput: (1) Xtrain: (numpy.ndarray) training split of covariates.
           (2) Xtest: (numpy.ndarray) test split of covariates.
           (3) target_test: (numpy.ndarray) training split of targets.
           (4) target_train: (numpy.ndarray) test split of targets.
           (5) nb_clf_fs: The Naive Bayes classifier fitted to training data.
           
    Examples: test = find_classifier(tweetTextXtfidf, westEast, tfidfVectorize,
                                     30, 0.1, WEST_EAST_NAMES, None, 
                                     scoreMetrics, 10)
    """
    num_obs = len(targets)               # the number of observations
    class_counts = Counter(targets)      # number of tweets by region
    class_percentages = {}               # percentages of tweets by region
    cv_scores = {}                       # scores from cross-validation
    for region in class_counts.keys():
        class_percentages[region] = class_counts[region] / num_obs
    
    # Printing Summary Statistics
    print "Percentages of tweets by region:"
    for region in class_counts.keys():
        print ("Percent %s: %0.4f, Count: %s" % (class_names[region], 
                                                 class_percentages[region],
                                                 class_counts[region]))
    print "Total Count of tweets: %s" % num_obs
    print "n-gram Range: {}".format(vectorizer.ngram_range)
    
    # FEATURE SELECTION SEEMS TO WORK BEST AROUND 25-30%
    fs_object = SelectPercentile(chi2, percentile=sel_pct)
    selected_feat = fs_object.fit_transform(covariates, targets)
    
    Xtrain, Xtest, target_train, target_test = train_test_split(selected_feat,
                                                                targets)
    
    nb_clf_fs = MultinomialNB(alpha=alphapar).fit(Xtrain, target_train)
    predictions = nb_clf_fs.predict(Xtest)
    
    confusion_mat = confusion_matrix(target_test, predictions)
    for metric in scoring_metrics:
        cv_scores[metric] = cross_val_score(nb_clf_fs, selected_feat, targets, 
                                            scoring=metric, cv=cv_fold)
    classification_eval(Xtrain, Xtest, target_test, predictions, 
                        scoring_metrics, cv_scores, confusion_mat, 
                        len(class_counts.keys()))
    print ("Classification Report: ")
    print(classification_report(target_test, predictions, 
                                target_names=class_names.values()))
    return (Xtrain, Xtest, target_train, target_test, nb_clf_fs, fs_object)

def make_test_samples(samp_sizes, vectz, datalist):
    """Creates dict containing data for testing classfiers.
    
    Input: (1) samp_sizes (list): values are various sample sizes of data to 
           create.
           ### should not be an argument, it's always the same
           (2) vectz (list): contains vectorizer objects. CountVectorizer
           should be before TFIDFVectorizer.
           (3) datalist (list): contains raw data to create samples from.
           
    Output: (1) multi-level dict containing dicts with required data samples.
    
    Examples: data_for_classifiers = make_test_samples([1000, 2000], 
                  [cVectorize, tfidfVectorize], usableTweetData)
    """

    samp_corpora = {}
    random.seed(0)    # ensures reproducibility of results
    for samsize in samp_sizes:
        samp_corpora[samsize] = tweet_corpus_maker(random.sample(datalist, 
                                                                 samsize))

    clf_test_data = {}
    ### i'm not sure i understand how these labels go with the vectz argument
    ### to the function; is the vectz argument always the same?
    vectorizer_labels = ['countV', 'tfidfV']
    for index, vectlabel in enumerate(vectorizer_labels):
        vectorizer_dict = {}
        for samsize in samp_sizes:
            ssdic = {}
            ssdic['X'] = vectz[index].fit_transform(samp_corpora[samsize][0])
            ssdic['westTargets'] = samp_corpora[samsize][2]

            Xtrain, Xtest, Ttrain, Ttest = train_test_split(
                ssdic['X'], 
                ssdic['westTargets'], 
                random_state=0
            )
            ssdic['Xtrain'] = Xtrain
            ssdic['Xtest'] = Xtest
            ssdic['Ttrain'] = Ttrain
            ssdic['Ttest'] = Ttest
            vectorizer_dict[samsize] = ssdic
        clf_test_data[vectlabel] = vectorizer_dict

    return clf_test_data

def make_full_data_test_samples(targets, vectz_dict):
    """Makes dict containing data for testing classfiers. 
    
    This function allows a redundant dict in the output data in order to 
    properly interface with the test_classifiers function that expects it.
    
    Input: (1) targets (np.array): all of the targets in the data for making 
           training and test sets.
           (2) vectz_dict (dict): keys are vectorizer names and values are 
           vectorized data.
           
    Output: (1) multi-level dict containing dicts with required data samples.
    
    Examples: dataForClassifiers = make_full_data_test_samples(westEast, 
                  [cVectorize, tfidfVectorize], usableTweetData)
    """

    samp_sizes = [len(targets)]
    clf_test_data = {}
    for vectlabel in vectz_dict:
        vectorizer_dict = {}

        for samsize in samp_sizes:
            ssdic = {}
            ssdic['X'] = vectz_dict[vectlabel]
            ssdic['westTargets'] = targets

            Xtrain, Xtest, Ttrain, Ttest = train_test_split(
                ssdic['X'], 
                ssdic['westTargets'], 
                random_state=0
            )
            ssdic['Xtrain'] = Xtrain
            ssdic['Xtest'] = Xtest
            ssdic['Ttrain'] = Ttrain
            ssdic['Ttest'] = Ttest
            vectorizer_dict[samsize] = ssdic

        clf_test_data[vectlabel] = vectorizer_dict

    return clf_test_data

def check_classifier(classifier, predictions, fittime, X, 
                     Xtrain, Xtest, region_test, all_targets, scoring_metrics, 
                     region_names):
    """Performs classifier comparisons.
    
    Utility function for the below classification tests.
    
    Input: (1) classifier: (sklearn classifier object) classfier with .predict 
               method.
           (2) predictions: (numpy.ndarray) target predictions of classifier.
           (3) fittime: (float) the time it took to make a model fit.
           (4) X (scipy.sparse.csr.csr_matrix): matrix of all selected 
               features.
           (5) Xtrain (scipy.sparse.csr.csr_matrix): matrix of selected text 
               features used to train classifier.
           (6) Xtest (scipy.sparse.csr.csr_matrix): matrix of selected text 
               features used to test the classifier.
           (7) region_test (numpy.ndarray): array of int, which code for the 
               target regions in the test set.
           (8) all_targets (numpy.ndarray): array of int, which correspond to 
               all target data.
           (9) scoring_metrics (list): contains strings corresponding to 
               scoring metrics used in cross validation or classifier 
               evaluation.
           (10) region_names (list):contains strings corresponding to the names
               of the regions to predict.
           
    Output: (1) cvroc_auc (tuple): mean roc area under curve score calculated
                using cross-validation (performed on all data), and 2*SE of the
                mean.
            
    Examples: check_classifier(NBclassifier, NBpredicted, NBTime, 
                               tweetTextXtfidf, 
                               clf_tests['countV'][1000]['Xtrain'],
                               clf_tests['countV'][1000]['Xtest'],
                               clf_tests['countV'][1000]['Ttest'],
                               clf_tests['countV'][1000]['westTargets'],
                               ['roc_auc'], WEST_EAST_NAMES.values())
    """
    assert hasattr(classifier, 'predict'), ("Classifier must have .predict "
                                                                     "method.")

    try:
        confusion_mat = confusion_matrix(region_test, predictions)
    except Exception, e:
        print ("confusionMatrixCalculationException: %s" % e)

    cv_score_vals = {}
    for metric in scoring_metrics:
        try:
            cv_score_vals[metric] = cross_val_score(classifier, X, 
                                        all_targets, scoring=metric, n_jobs=2, 
                                        cv=StratifiedKFold(all_targets, 10, 
                                                           random_state=0))
        except Exception, e:
            print ("cv%sMetricScoreException: %s" % (metric, e))
    try:
        cvroc_auc = classification_eval(Xtrain, Xtest, region_test, 
                                        predictions, scoring_metrics, 
                                        cv_score_vals, confusion_mat, 
                                        len(region_names))
    except Exception, e:
        print ("clasificationEvalException: %s" % e)

    try:
        print(classification_report(region_test, predictions, 
                                    target_names=region_names))
    except Exception, e:
        print ("classificationReportException: %s" % e)

    print ("Fit Time: %0.3f" % (fittime))
    return cvroc_auc

def test_classifiers(data, clfs, vectz, ssize, scoring):
    """Runs benchmark tests of classfiers.
    
    Currently, this function trains classifiers with static parameter values.
    If one wishes to change the parameter values of any given classifier, one
    must change the parameters in this function definition. This is acceptable
    only because these parameters have been generally found to lead to good
    classifier performance on the twitter data set considered in this project.
    If one desires to use this function more generally, then it must be 
    redesigned to allow one to pass parameters into the classifiers.

    Input: (1) data (dict): the output of make_test_samples()
           (2) clfs (list): has strings corresponding to names of various
               classifiers in scikit-learn.
           (3) vectz (str): either 'countV' or 'tfidfV'
           (4) ssize (int): chooses sample size to run classifier on.
           (5) scoring (list): list of strings determining which score metrics
               to use with check_classifier()
           
    Output: (1) benchmark_output[0] (float): time elapsed for fitting of
                classifier and calculation of predictions.
            (2) benchmark_output[1] (float): average of roc area under the 
                curve calculated using 10-fold cross-validation.
            (3) benchmark_output[2] (float): 2 times the standard error of the
                10 estimates of roc area under the curve calculated during 
                10-fold cross-validation.
            (4) classifier_objects (dict): keys are classifier names and values
                are the fitted scikit-learn classifier objects.
            
    Examples: cBench = test_classifiers(testClfs, classifierList, 
                           len(westTargets), ['roc_auc', 'f1'])
    """
    
    # some summary statistics of the data set
    target_sets = ['westTargets', 'Ttrain', 'Ttest']
    target_set_names = ['', ' in Training Set', ' in Test Set']
    west_east_names = {0: 'East', 1: 'West'}
    num_obs = {}
    tweet_percentages = {}
    class_counts = {}
    for tset in target_sets:
        num_obs[tset] = len(data[vectz][ssize][tset])
        class_counts[tset] = Counter(data[vectz][ssize][tset])
        class_pcts = {}
        for region in west_east_names:
            class_pcts[region] = class_counts[tset][region] / num_obs[tset]
        tweet_percentages[tset] = class_pcts

    print ("Percentages of Tweets by Region:")
    for tset, tsetname in zip(target_sets, target_set_names):
        for region in west_east_names:
            print ("Percent %s%s: %0.4f, Count: %s" % (west_east_names[region], 
                    tsetname, tweet_percentages[tset][region], 
                    class_counts[tset][region]))
        print ("Tweet Count Total%s: %s" % (tsetname, num_obs[tset]))

    
    # library of possible classifiers
    clf_lib = {
      'MultinomialNB': {
          'clf': MultinomialNB(alpha=0.005),
          'dense': False,
          'find_parameters': None
      },
      'SVC': {
          'clf': SVC(C=1.0, kernel='linear', probability =True, 
                     class_weight='auto', random_state=0),
          'dense': False,
          'find_parameters': None
      },
      'RandomForestClassifier': {
          'clf': RandomForestClassifier(min_samples_leaf=24, n_jobs=2,
                                        random_state=0),
          'dense': True,
          'find_parameters': None
      },
      'RidgeClassifierCV': {
          'clf': RidgeClassifierCV(
              cv=StratifiedKFold(data[vectz][ssize]['Ttrain'], 
                                 10, random_state=0), 
              scoring='roc_auc'),
          'dense': False,
          'find_parameters': None
      },
      'KNeighborsClassifier': {
          'clf': KNeighborsClassifier(weights='distance'),
          'dense': False,
          'find_parameters': None
      },
      'Perceptron': {
          'clf': Perceptron(penalty='l2', class_weight='auto', 
                            n_jobs=2, random_state=0),
          'dense': False,
          'find_parameters': None
      },
      'PassiveAggressiveClassifier': {
          'clf': PassiveAggressiveClassifier(C=0.5, n_jobs=2),
          'dense': False,
          'find_parameters': None
      },
      'LDA': {
          'clf': LDA(),
          'dense': True,
          'find_parameters': None
      },
      'QDA': {
          'clf': QDA(),
          'dense': True,
          'find_parameters': None
      },
      'SGDClassifier': {
          'clf': SGDClassifier(loss='squared_hinge', n_jobs=2, random_state=0,
                               class_weight='auto'),
          'dense': False,
          'find_parameters': None
      },
      'AdaBoostClassifier': {
          'clf': AdaBoostClassifier(random_state=0),
          'dense': True,
          'find_parameters': None
      },
      'BaggingClassifier': {
          'clf': BaggingClassifier(n_jobs=2, random_state=0),
          'dense': True,
          'find_parameters': None
      },
      'ExtraTreesClassifier': {
          'clf': ExtraTreesClassifier(min_samples_leaf=24, n_jobs=2, 
                                      random_state=0),
          'dense': True,
          'find_parameters': None
      },
      'GradientBoostingClassifier': {
          'clf': GradientBoostingClassifier(min_samples_leaf=24),
          'dense': True,
          'find_parameters': None
      },
      'RandomizedLogisticRegression': {
          'clf': LogisticRegression(C=16.0, random_state=0, 
                                                          class_weight='auto'),
          'dense': False,
          'find_parameters': RandomizedLogisticRegression(C=16.0, 
                                                          random_state=0, 
                                                          n_jobs=2)
      },
      'LogisticRegression': {
          'clf': LogisticRegression(C=16.0, random_state=0, 
                                                          class_weight='auto'),
          'dense': False,
          'find_parameters': None
      }

    }

    
    benchmark_output = {}    
    for clf_label in clfs:
        
        print("Testing: %s" % clf_label)
        clf_info = clf_lib[clf_label]

        if clf_info['dense']:
            X = data[vectz][ssize]['X'].toarray()
            Xtrain = data[vectz][ssize]['Xtrain'].toarray()
            Xtest = data[vectz][ssize]['Xtest'].toarray()

        else:
            X = data[vectz][ssize]['X']
            Xtrain = data[vectz][ssize]['Xtrain']
            Xtest = data[vectz][ssize]['Xtest']


        base_time = time()
        if clf_info['find_parameters']:
            try:
                fp_clf = clf_info['find_parameters']
                Xtrain = fp_clf.fit_transform(
                    Xtrain, 
                    data[vectz][ssize]['Ttrain']
                )
                Xtest = fp_clf.transform(Xtest)

            except Exception, e:
                print "%s Pre-fitting: %s" % (clf_label, e)

        try:
            clf = clf_info['clf']
            clf.fit(Xtrain, data[vectz][ssize]['Ttrain'])
        except Exception, e:
            print ("%s ClfFitException: %s" % (clf_label, e))

        try:
            predictions = clf.predict(Xtest)
        except Exception, e:
            print ("%s ClfPredictException: %s" % (clf_label, e))

        fit_time = time() - base_time
        try:
            clf_roc_auc = check_classifier(
                 clf, 
                 predictions, fit_time, X, 
                 Xtrain, Xtest, data[vectz][ssize]['Ttest'],
                 data[vectz][ssize]['westTargets'], scoring, 
                 west_east_names.values()
            )
        except Exception, e:
            print ("%s CheckClassifierException: %s" % (clf_label, e))

        try:
            benchmark_output[clf_label] = (
                fit_time, 
                clf_roc_auc[0], 
                clf_roc_auc[1], 
                clf
            )
        except Exception, e:
            print ("%s BenchmarkOutputException: %s" % (clf_label, e))

    return benchmark_output

### this function is not used elsewhere, delete?
def parm_set(parameter_string):
    """Formats string for use in a dict.
    
    This function takes string from documentation providing parameter list to
    classifier functions and returns a string with the parameters formatted
    to be included in a dict object.
    
    Input: (1) parameter_string (str): string from scikit-learn function 
               argument list.
    
    Output: (1) (str): string formatted so that function argument names and 
                their default values can be easily included in a dict object.
    
    Examples: parm_set('alpha=0.05, n_jobs=2') has output: 'alpha': 0.05, 
                  'n_jobs': 2
    """
    parm_list = []
    for sub_string in parameter_string.split(','):
        key, value = sub_string.split("=")
        parm_list.append("'%s': %s" % (key.strip(), value.strip()))

    return ",".join(parm_list)

### this function is only used in feat_rates, delete?
def feat_index(feature, vectorizer):
    """Returns index if feature in feature set, None otherwise."""
    feature_names = vectorizer.get_feature_names()
    for i, name in enumerate(feature_names):
        if re.match('\\b' + feature + '\\b', name) is not None:
            return i
    else:
      return None

### this function is not used elsewhere, delete?
def feat_rates(feature, vectorizer, cX):
    """Returns the number of times feature appears in all tweets, and rate in 
       tweets and rate in features.

    Input: (1) feature (str): a string one will find an exact match of among 
               selected features.
           (2) vectorizer (sklearn CountVectorizer object): contains 
               information about bag-of-words model of tweets.
           (3) cX (numpy sparse matrix): rows are for individual tweets and 
               columns hold integers corresponding to number of times the 
               feature for that column appears in the tweet.

    Output: (1) (int): number of times feature appears among all tweets.
            (2) (float): number of times the feature appears per tweet ((1) 
                divided by number of tweets).
            (3) (float): number of times the feature appears per feature ((1) 
                divided by the number of features).
    """
    index = feat_index(feature, vectorizer)
    num_feats = cX[:,index]
    return (num_feats.sum(), 
            num_feats.sum() / cX.shape[0], 
            num_feats.sum() / cX.shape[1])
