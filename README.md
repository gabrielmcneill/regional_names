# Regional Words Project

## Overview
This data science project analyzes Twitter data to classify individual tweets as originating in one of two pre-defined  US regions: East or West. I train several classifiers, scoring them based on their performance as measured by the area under their receiver operating characteristic (ROC) curves. The features each classifier picks as being most informative are noted and analyzed.

## Table of Contents

1. [Introduction](##Introduction)
2. [Region Definition](##Region Definition)
3. [Data Collection](##Data Collection - Interfacing with the Twitter API)
4. [Data Cleaning](##Data Cleaning)
  1. [Basic Issues](#Basic Issues)
  2. [Tokenization and Stemming](#Tokenization and Stemming)
5. [Data Exploration](##Data Exploration)
6. [Modeling and Choice of Classifier](##Modeling and Choice of Classifier)
7. [Training and Testing](##Training and Testing of Classifiers)
8. [Best Features](##Best Features)
9. [Conclusion](##Conclusion)


#Introduction
The initial motivation for this project was to evaluate the hypothesis that the geographical origin of a tweet can be predicted by using its text alone. Well known studies of regional linguistic variation have demonstrated that it is possible to cluster variations in vocabulary, pronunciation, and grammar in English by geographical regions in the United States ([New York Times interactive dialect map](http://www.nytimes.com/interactive/2013/12/20/sunday-review/dialect-quiz-map.html)). However, when attempting to classify tweet text there are a couple of challenges one must overcome to create a reasonable classifier: (1) Twitter enforces a 140 character limit on individual tweets, resulting in extremely sparse feature matrices (using the bag-of-words model) and (2) most defining features of regional linguistic variation are not common in normal English text, let alone the abbreviated form often used in tweets.

The character limit on tweets makes it crucial to collect as large a sample as possible, and to carefully select features used in training a classifier. Perhaps the most critical aspect of training any text classifier is in the choice of text features. My initial approach was to compile a list of words and phrases that could possibly serve as informative features. I did so by compiling a list of 381 words and phrases identified by linguists as being characteristic of certain regional dialects. However, none of the 381 words or phrases appeared among an initial test sample of 5,499 tweets, which made it clear that such words were most likely not going to be helpful in training a good classifier (only 78 of the 381 appear in my final sample of 145,559 tweets, and none are among the most informative features). It was clear to me that a pure machine learning approach would be necessary to train a strong classifier.


#Region Definition
I define two regions of the US in which to classify individual tweets as originating: West and East. The West region is defined as the area bounded by the latitude lines at 24.545905 and 48.997502 degrees, and the longitude lines at -124.78366 and -93.562895 degrees. The East region is likewise defined as the are bounded by the latitude lines at 24.545905 and 48.997502 degrees, and the longitude lines at -93.562895 and -67.097198 degrees. This roughly corresponds to the West region as the rectangle encompassing all states to the west of the border between Texas and Louisiana; all states to the east of that border belong to the East region.

A large proportion of the tweets that I collected originated in states on the East coast, in the South, or in the Midwest. In my final sample of 145,559 tweets, 39.3 percent of them originated in the West region and 60.7 percent of them originated in the East region despite having drawn the dividing longitude line so far to the East. Several factors may have contributed to having collected a larger proportion of tweets in the East region than the West region: (1) people may tweet at higher rates in the South, East, and Midwest than do in the states in the West, and (2) the time of day of data collection may have resulted in a sample that covers the peak times of tweet activity in eastern and southern states but not the peak times of the western states.

#Data Collection - Interfacing With The Twitter API

#Data Cleaning

##Basic Issues

##Tokenization and Stemming

Selecting a subset of features based on feature selection criteria is the most effective way to train classifiers on datasets containing brief text. The process of tokenization, removing stopwords, and stemming is an important first pass in 

#Data Exploration

#Modeling and Choice of Classifier

#Training and Testing Classifiers

#Best Features

#Conclusion
