# Regional Words Project

## Overview
This data science project analyzes Twitter data to classify individual tweets as originating in one of two pre-defined  US regions: East or West. I train several classifiers, scoring them based on their performance as measured by the area under their receiver operating characteristic (ROC) curves. Finally, I present and analyze the most informative features that each classifier finds.

## Table of Contents

1. [Introduction](#Introduction)
2. [Region Definition](#Region Definition)
3. [Data Collection](#Data Collection - Interfacing with the Twitter API)
4. [Data Cleaning](#Data Cleaning)
  1. [Basic Issues](##Basic Issues)
  2. [Tokenization and Stemming](##Tokenization and Stemming)
5. [Data Exploration](#Data Exploration)
6. [Data Modeling and Choice of Classifier](#Data Modeling and Choice of Classifier)
7. [Training and Testing](#Training and Testing of Classifiers)
8. [Best Features](#Best Features)
9. [Conclusion](#Conclusion)


#Introduction
The motivation for this project is to evaluate the hypothesis that the geographical origin of a tweet can be predicted using its text alone. Well known studies of regional linguistic variation have demonstrated that it is possible to cluster variations in vocabulary, pronunciation, and grammar in English by geographical regions of the United States ([New York Times interactive dialect map](http://www.nytimes.com/interactive/2013/12/20/sunday-review/dialect-quiz-map.html)). However, when attempting to classify tweet text by US region there are a couple of challenges to overcome in order to create a reasonable classifier: (1) Twitter enforces a 140 character limit on individual tweets, resulting in extremely sparse feature matrices (using the bag-of-words model) and (2) most defining features of regional linguistic variation are not common in normal English text, let alone the abbreviated form often used in tweets. 

The character limit on tweets makes it crucial to collect as large a sample as possible, and to carefully select features used in training a classifier. Perhaps the most critical aspect of training any text classifier is in the choice of text features. My initial approach was to compile a list of words and phrases that could possibly serve as informative features. I did so by compiling a list of 381 words and phrases identified by linguists as being characteristic of certain regional dialects. However, none of the 381 words or phrases appeared among an initial test sample of 5,499 tweets, which made it clear that such words were most likely not going to be helpful in training a good classifier (only 78 of the 381 appear in my final sample of 145,559 tweets, and none are among the most informative features). It was clear to me that a pure machine learning approach would be necessary to train a strong classifier.

I considered a wide range of potential classification algorithms including Support Vector Machines, Random Forests, and ensemble methods (AdaBoost, Bagging, Gradient Boosting), but in benchmark timing tests that I ran I determined them to be infeasible due to their poor scaling when applied to my tweet database (145,599 observations of 108,400 features), and when performed on the machines I have access to. I ruled out other classifiers (Ridge Classification, K-Nearest Neighbors, Linear Discriminant Analysis, Quadratic Discriminant Analysis), and in the end evaluated five different feasible classification methods: Multinomial Naive Bayes, Logistic Regression, Stochastic Gradient Descent, Perceptron, and the Passive Aggressive Classifier. 


#Region Definition
I define two regions of the US in which to classify individual tweets as originating: West and East. The West region is defined as the area bounded by the latitude lines at 24.545905 and 48.997502 degrees, and the longitude lines at -124.78366 and -93.562895 degrees. The East region is likewise defined as the are bounded by the latitude lines at 24.545905 and 48.997502 degrees, and the longitude lines at -93.562895 and -67.097198 degrees. This roughly corresponds to the West region as the rectangle encompassing all states to the west of the border between Texas and Louisiana; all states to the east of that border belong to the East region.

A large proportion of the tweets that I collected originated in states on the East coast, in the South, or in the Midwest. In my final sample of 145,559 tweets, 39.3 percent of them originated in the West region and 60.7 percent of them originated in the East region despite having drawn the dividing longitude line so far to the East. Several factors may have contributed to having collected a larger proportion of tweets in the East region than the West region: (1) people may tweet at higher rates in the South, East, and Midwest than do in the states in the West, and (2) the time of day of data collection may have resulted in a sample that covers the peak times of tweet activity in eastern and southern states but not the peak times of the western states.

At the beginning of the project I originally defined six different US regions (West, Lowerwest, Upperwest, Midwest, South, and East), but because of the comparatively few tweets in the Lowerwest and Upperwest regions training a classifier proved difficult. The large proportion of tweets from the South tended to heavily bias the classifiers to only predict South, leading to classification accuracy only slightly above what one would expect if one were to always classify each tweet as originating in the South. I switched to a binary regional split between East and West in order to have both a more balanced proportion of tweets per region, and to maintain interpretability, defining regional divisions that roughly correspond to intuitively coherent geographical areas.

#Data Collection - Interfacing With The Twitter API
The program (`twitter_interface.py`) runs from the command line, samples tweets by interacting with the [Twitter API](https://dev.twitter.com/streaming/overview), and deposits them into a pickle (`.p`) file into the current directory. It accepts two command-line arguments: `MAX_TWEETS`, which determines the number of tweets to sample, and an optional `file_name` that the user can set to name the name of the file containing the tweet data.

A modified `MyStreamer` class that inheirits the `TwythonStreamer` class from the [twython module](https://twython.readthedocs.org/en/latest/) performs the substantive interface with Twitter and appends to a list each dict containing an individual tweet and its corresponding geocoordinates of origin. The `process_tweets()` function takes the list of tweet data as its argument and adds to each tweet dict in the list a region label key and the regional classification for each observation using the above region definition. 

The 145,599 tweets in the data set I analyze here were collected different times, ranging from late February 2015 to early March 2015, but most were collected at different times (PST) throughout the day on February 29, 2015.

#Data Cleaning

##Basic Issues
Despite the use of filtering variables (`language='en, locations=US_BOUNDING_BOX'`) when interfacing with the Twitter streaming API, I nevertheless found in my data some tweets written in Spanish, and some tweets whose coordinates were outside of the `US_BOUNDING_BOX` that I used to define the region of interest. Additionally, although all dicts containing tweet data had keys corresponding to geocoordinates of origin (`'coordinates'`), there were many whose values were `None`. I excluded all tweets that were not in English, that did not originate within the West or East region, or that did not have coordinate data. All tweets with region key `'west'` and a corresponding value of `None` were excluded.

##Tokenization and Stemming

Selecting a subset of features based on feature selection criteria is the most effective way to train classifiers on datasets containing brief text. The process of tokenization, removing stopwords, and stemming are important first steps in modifying the data in preparation for analysis. 

After experimenting with different methods of splitting the individual strings of tweet text into tokens, including simply splitting on whitespace, I settled on the [NLTK](http://www.nltk.org/) `word_tokenize()` function from the [`nltk.tokenize` module](http://www.nltk.org/_modules/nltk/tokenize.html). The `word_tokenize()` function's performance was generally equvalent to the recommended `TreebankWordTokenizer` in the `nltk.tokenize.treebank` module, but to my eye it tended to produce more natural splits of words than did the `TreebankWordTokenizer`.

I experimented with different stopword lists, including the English list (`stopwords.words('english')`) in the `nltk.corpus` module, but settled on the larger default English stopword list used by scikit-learn vectorizers like the [TfidfVectorizer](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) because it tended to result in generally better performance (as measured by 10-fold cross-validated measures of the area under the ROC curve). In addition to the default scikit-learn stopword list, I remove all tokens that match the regular expression `"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"`, which is designed to match Twitter usernames, web URLs, and isolated digits or characters.

In addition to tokenization and stopword removal I defined my own class (`SnowballEnglishStemmer`) that implements the NLTK stemmer `EnglishStemmer` from the [`nltk.stem.snowball`](http://www.nltk.org/_modules/nltk/stem/snowball.html) module. I also tested the performance of the NLTK Porter and Lancaster stemmers, but they offered no improvements over the `EnglishStemmer`, so I settled on using it.

#Data Exploration


#Data Modeling and Choice of Classifier

#Training and Testing Classifiers

#Best Features
The **Multinomial Naive Bayes classifier** (with smoothing parameter `alpha = 0.005`) determined the following most infomative features:

| Best 20 Features for West | P(West)/P(East) | Best 20 Features for East | P(West)/P(East) | 
|:--------------------------|----------------:|:--------------------------|----------------:|
| tx http                   | 5240.41822      | fl |  0.00015 |
| tx https                  | 4536.77288      | ny https |  0.00020 |
| pdx911                    | 3240.31370      | fl https |  0.00025 |
| houston tx                | 3235.68498      | nj |0.00027 |
| san francisco ca          | 2362.94725      | va |  0.00032 |
| francisco ca              | 2362.94725      | york ny |  0.00033 |
| portland portland         | 2321.00500      | new york ny | 0.00033 |
| tx http job               | 2265.57633      | york ny https |  0.00037 |
| francisco ca https        | 2203.24216      | md |  0.00038 |
| searchfest                | 2043.20840      | il https |  0.00040 |
| san antonio               | 2037.46210      | orlpol |  0.00040 |
| street sidewalk           | 2008.45176      | lrt |  0.00044 |
| california https          | 1967.74754      | pa https |  0.00051 |
| clean request             | 1894.42929      | chicago il |  0.00054 |
| sidewalk clean request    | 1894.42929      | fl http |  0.00057 |
| sidewalk clean            | 1894.42929      | nj https |  0.00058 |
| street sidewalk clean     | 1894.42929      | oh https |  0.00059 |
| seattl wa                 | 1854.11638      | ny http |  0.00060 |
| seattlefre                | 1836.55585      | ciaa |  0.00062 |
| houston tx http           | 1817.73192      | mi https |  0.00063 |

**Logistic Regression** (with regularization parameter `C = 16.0`) determined the following most informative features:

| Best 20 Features for West | Coefficient | Best 20 Features for East | Coefficient | 
|:--------------------------|------------:|:--------------------------|------------:|
| retweet favorit | 11.47490 | dnvrlostfound | -14.16393 |
| bloglovin | 9.76067 | let gooo | -11.67939 |
| mage | 9.09616 | rosco chicken | -11.03252 |
| block se | 8.72449 | kuntz | -10.20301 |
| nb turn | 8.54387 | llamadrama http | -9.27854 |
| gabriel armandariz | 8.09304 | just dress | -8.80247 |
| school earli today | 7.65346 | read new | -8.67802 |
| sahuarita | 7.48578 | glorious goodby | -8.12342 |
| appl cheek | 7.41900 | life plan | -6.86772 |
| high court | 7.41687 | littlegingerr | -6.80091 |
| retweet jump | 7.37838 | hilton https | -6.44863 |
| nation park | 7.21802 | click forget | -6.37359 |
| patisseri amp | 7.15735 | golobo | -6.34657 |
| jazz https | 6.85594 | bout late | -6.33936 |
| https follow rhythmmjockey | 6.71567 | na right | -6.31088|
| pictur https | 6.69665| pleas https | -6.17703|
| music video right | 6.67440 | norwood | -5.97959 |
| round 4a bracket | 6.66083 | like care color | -5.97925 |
| long final | 6.65117 | life point | -5.93785 |

The **Passive Aggressive Classifier** determined the following most informative features:

| Best 20 Features for West | Coefficient | Best 20 Features for East | Coefficient | 
|:--------------------------|------------:|:--------------------------|------------:|
| tx |  8.02895 | fl |  -4.68202 |
| ca http |  6.67014 | toronto|  -4.50331 |
| ca https |  5.92106 | nc |  -3.79491 |
| seattlefre |  5.89452 | il |  -3.55993 |
| houston |  5.79093 | carolina |  -3.42067 |
| wa |  5.64550 | ny |  -3.36219 |
| portland |  5.62727 | tampa |  -3.35405 |
| az |  5.40789 | nj |  -3.31713 |
| suonus |  5.28571 | pa | -3.25956 |
| lwtsummit |  5.07407 | kyri |  -3.25862 |
| texa |  5.02961 | just offer |  -3.19816 |
| fucken |  4.97737 | michael kor |  -3.14741 |
| disneyland |  4.91535 | va |  -3.13922 |
| seattl |  4.90604 | becaus miss |  -2.96753 |
| katramsland |  4.83358 | think dalla |  -2.94594 |
| like noth |  4.70096 | ddlovato |  -2.90042 |
| colorado |  4.58926 | orlpol |  -2.86077 |
| http yes |  4.58678 | thank sm |  -2.81554 |
| bois |  4.54638 | brooklyn |  -2.79378 |
| ks |  4.53998 | warm weather |  -2.78641 |

#Conclusion
