# Regional Words Project

## Overview
This data science project analyzes Twitter data to classify individual tweets as originating in one of two pre-defined US regions: East or West. I train several classifiers, scoring them based on their performance as measured by the area under their receiver operating characteristic (ROC) curves. Finally, I present and analyze the most informative features that each classifier finds.

## Table of Contents

1. [Introduction](#Introduction)
2. [Region Definition](#Region Definition)
3. [Data Collection](#Data Collection - Interfacing with the Twitter API)
4. [Data Cleaning](#Data Cleaning)
  1. [Basic Issues](##Basic Issues)
  2. [Tokenization and Stemming](##Tokenization and Stemming)
5. [Data Exploration](#Data Exploration)
6. [Data Modeling and Choice of Classifier](#Data Modeling and Choice of Classifier)
7. [Training and Testing of Classifiers](#Training and Testing of Classifiers)
8. [Best Features](#Best Features)
9. [Conclusion](#Conclusion)


#Introduction
The motivation for this project is to evaluate the hypothesis that the geographical origin of a tweet can be predicted using its text alone. Well known studies of regional linguistic variation have demonstrated that it is possible to cluster variations in vocabulary, pronunciation, and grammar in English by geographical regions of the United States ([New York Times interactive dialect map](http://www.nytimes.com/interactive/2013/12/20/sunday-review/dialect-quiz-map.html "New York Times Dialect Quiz")). However, when attempting to classify tweet text by US region there are a couple of challenges to overcome in order to create a reasonable classifier: (1) Twitter enforces a 140 character limit on individual tweets, resulting in extremely sparse feature matrices (using the bag-of-words model) and (2) most defining features of regional linguistic variation are not common in normal English text, let alone in the abbreviated form often used in tweets. 

The character limit on tweets makes it crucial to collect as large a sample as possible, and to carefully select features used in training a classifier. Perhaps the most critical aspect of training any text classifier is in the choice of text features, for it is not only an important factor in training a strong classifier, but it also can affect how well a classification method scales under increasing numbers of observations. My initial approach in this project was to compile a list of words and phrases that could possibly serve as informative features. I did so by compiling a list of 381 words and phrases identified by linguists as being characteristic of certain regional dialects. However, none of the 381 words or phrases appeared among an initial test sample of 5,499 tweets, which made it clear that such words were most likely not going to be helpful in training a good classifier (only 78 of the 381 appear in my final sample of 145,559 tweets, and none is among the most informative features). It was clear to me that a pure machine learning approach would be necessary to train a strong classifier.

I considered a wide range of potential classification algorithms including Support Vector Machines, Random Forests, and ensemble methods (AdaBoost, Bagging, Gradient Boosting), but benchmark timing tests determined them to be infeasible due to their poor scaling when applied to my tweet database (145,599 observations of 108,400 features), and when performed on the machines I have access to. I ruled out other classifiers (Ridge Classification, K-Nearest Neighbors, Linear Discriminant Analysis, Quadratic Discriminant Analysis) for similar reasons, and in the end evaluated five different feasible classification methods: Multinomial Naive Bayes, Logistic Regression, Stochastic Gradient Descent, Perceptron, and the Passive Aggressive Classifier. 

#Region Definition
I define two regions of the US in which to classify individual tweets as originating: West and East. The West region is defined as the area bounded by the latitude lines at 24.545905 and 48.997502 degrees, and the longitude lines at -124.78366 and -93.562895 degrees. The East region is likewise defined as the are bounded by the latitude lines at 24.545905 and 48.997502 degrees, and the longitude lines at -93.562895 and -67.097198 degrees. This roughly corresponds to the West region as the rectangle encompassing all states to the west of the border between Texas and Louisiana; all states to the east of that border belong to the East region.

A large proportion of the tweets that I collected originated in states on the East coast, in the South, or in the Midwest. In my final sample of 145,559 tweets, 39.3 percent of them originated in the West region and 60.7 percent of them originated in the East region despite having drawn the dividing longitude line so far to the East. Several factors may have contributed to having collected a larger proportion of tweets in the East region than the West region: (1) people may tweet at higher rates in the South, East, and Midwest than do in the states in the West, and (2) the time of day of data collection may have resulted in a sample that covers the peak times of tweet activity in eastern and southern states but not the peak times of the western states.

At the beginning of the project I originally defined six different US regions (West, Lowerwest, Upperwest, Midwest, South, and East), but because of the comparatively few tweets in the Lowerwest and Upperwest regions training a classifier proved difficult. The large proportion of tweets from the South tended to heavily bias the classifiers to only predict South, leading to classification accuracy only slightly above what one would expect if one were to always classify each tweet as originating in the South. I switched to a binary regional split between East and West in order to have both a more balanced proportion of tweets per region, and to maintain interpretability, defining regional divisions that roughly correspond to intuitively coherent geographical areas.

#Data Collection - Interfacing With The Twitter API
The program `twitter_interface.py` runs from the command line, samples tweets by interacting with the [Twitter API](https://dev.twitter.com/streaming/overview), and saves them in a pickle file in the present working directory. It accepts two command-line arguments: `MAX_TWEETS`, which determines the number of tweets to sample, and an optional `file_name` that the user can set to name the pickle file containing the tweet data. The Twitter streaming API requires [authentication](https://dev.twitter.com/oauth/overview), so in order to use `twitter_interface.py` one must have a twitter account and apply for the proper authentication keys:
```python
APP_KEY='XXX'
APP_SECRET='XXX'
OAUTH_TOKEN='XXX'
OAUTH_TOKEN_SECRET='XXX'
```
`twitter_interface.py` expects that these authentication tokens be stored as environment variables with the names:
```python
TWITTER_APP_KEY
TWITTER_APP_SECRET
TWITTER_OAUTH_TOKEN
TWITTER_OAUTH_TOKEN_SECRET
```
Non-persistent environment variables can be created from the command line in the following way (`$` is a shell prompt):
```
$ export TWITTER_APP_KEY='XXX'
$ export TWITTER_APP_SECRET='XXX'
$ export TWITTER_OAUTH_TOKEN='XXX'
$ export TWITTER_OAUTH_TOKEN_SECRET='XXX'
```

A modified `MyStreamer` class that inheirits the `TwythonStreamer` class from the [twython module](https://twython.readthedocs.org/en/latest/) performs the substantive interface with Twitter and appends to a list each dict containing an individual tweet and its corresponding geocoordinates of origin. The `process_tweets()` function takes the list of tweet data as its argument and adds to each tweet dict in the list a region label key and the regional classification for each observation using the above region definition. 

The 145,599 tweets in the data set I analyze here were collected different times, ranging from late February 2015 to early March 2015, but most were collected at different times (PST) throughout the day on February 29, 2015.

#Data Cleaning

##Basic Issues

Despite the use of filtering variables (`language='en, locations=US_BOUNDING_BOX'`) when interfacing with the Twitter streaming API, I nevertheless found in my data some tweets written in Spanish, and some tweets whose coordinates were outside of the `US_BOUNDING_BOX` that I used to define the region of interest. Additionally, although all dicts containing tweet data had keys corresponding to geocoordinates of origin (`'coordinates'`), there were many whose values were `None`. I excluded all tweets that were not in English, that did not originate within the West or East region, or that did not have coordinate data. All tweet dicts with region key `'west'` and value `None` were excluded.

##Tokenization and Stemming

Selecting a subset of features based on feature selection criteria is the most effective way to train classifiers on datasets containing brief text. The process of tokenization, removing stopwords, and stemming are important first steps in modifying the data in preparation for analysis. 

After experimenting with different methods of splitting the individual strings of tweet text into tokens, including simply splitting on whitespace, I settled on the [NLTK](http://www.nltk.org/) `word_tokenize()` function from the [`nltk.tokenize` module](http://www.nltk.org/_modules/nltk/tokenize.html). The `word_tokenize()` function's performance was generally equvalent to the recommended `TreebankWordTokenizer` in the `nltk.tokenize.treebank` module, but to my eye it tended to produce more natural splits of words than did the `TreebankWordTokenizer`.

I experimented with different stopword lists, including the English list (`stopwords.words('english')`) in the `nltk.corpus` module, but settled on the larger default English stopword list used by scikit-learn vectorizers like the [TfidfVectorizer](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) because it tended to result in generally better performance (as measured by 10-fold cross-validated measures of the area under the ROC curve). In addition to the default scikit-learn stopword list, I remove all tokens that match the regular expression `"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"`, which is designed to match Twitter usernames, web URLs, and isolated digits or characters.

In addition to tokenization and stopword removal I defined my own class (`SnowballEnglishStemmer`) that implements the NLTK stemmer `EnglishStemmer` from the [`nltk.stem.snowball`](http://www.nltk.org/_modules/nltk/stem/snowball.html) module. I also tested the performance of the NLTK Porter and Lancaster stemmers, but they offered no improvements over the `EnglishStemmer`, so I settled on using it.

#Data Exploration
Working with text data can be challenging since there are so many non-ascii character encodings. Determining simple summary statistics for a large database of tweet text can be difficult due to its size, but also because of the different decisions one must make about what counts as a character, and what does not. For example, emoji are increasingly common, and while they may appear as one coherent character in tweets, when decoded into unicode they are composed of many characters. :smile: becomes `\xF0\x9F\x98\x84` in utf-8 encoding, and `\U0001f604` in unicode. Twitter has a 140 character limit, but where special characters are concerned all are counted as single characters when they appear as one in a tweet. Thus, when tweet text is decoded, as is necessary for text processing, one may find tweets with an apparent character length of 500 or more. In addition to emoji, there are other specially encoded characters that one must be aware of when processing text.

Ignoring all non-ascii characters, my data set of 145,559 tweets can be summarized as follows:

| Measure on Number of Characters | Value | Measure on Number of Words | Value |
|:--------------------------------|------:|:---------------------------|------:|
| Mean Tweet Length (chars)       | 63.77 | Mean # Words per Tweet        | 11.44 |
| Median Tweet Length (chars)     | 56    | Median # Words per Tweet      | 10    |
| Minimum Tweet Length (chars)    | 4     | Minimum # Words in Tweet     | 1     |
| Maximum Tweet Length (chars)    | 140   | Maximum # Words in Tweet     | 43    |

![char hist](https://cloud.githubusercontent.com/assets/10871563/6852990/6a404e20-d3a5-11e4-8a44-848c7a7ba65e.png "Histogram of Number of Characters per Tweet")

Many of the tweets in the 120 to 125 character length range have several URLs, which inflates the character length. The steady rise up to the character limit of 140 seems to be due to tweets that are composed primarily of standard ascii characters. I hypothsize that this is partially an artifact of having ignored any non-ascii characters in the calculation of the number of characters and words.

An example of a tweet with minimum length: `u''Pist`.

An example of a tweet with maximum length: `u'#SocialMedia Series Sale\nFOLLOW: http://t.co/Umt0LljucZ\nPROFILE: http://t.co/rnD0fWojTa\nHOME - http://t.co/on6F0R0u0i http://t.co/UQWrbzvSbU'`.

![words hist](https://cloud.githubusercontent.com/assets/10871563/6831923/92e6006a-d2e0-11e4-9fdb-94fd83077e94.png "Histogram of Number of Words per Tweet")

An example of a tweet with the minimum number of words: `'Anyyyywayssssss.'`.

An example of a tweet with the maximum number of words: `'@jasonthejman p4 plays:\n1,2,6,8,9,10/3/7/3,5,8,10\n8,9,10/3/7/3,5,8,10\n8,9,10/4/7/3,5,8,10\n8,9,10/3/2,8/3,5,8,10'`. This of course assumes that isolated numerals count as words.

#Data Modeling and Choice of Classifier
For text classification the bag-of-words model is standard, and involves creating a data matrix with as many rows as documents (in this case tweets), and as many columns as features. The `[i,j]`th element of the data matrix is an integer equal to the number of times feature j appears in document i ( `tf[i,j]` ). Alternatively, one may employ tf-idf reweighting to construct a data matrix such that the `[i,j]`th element is a floating point number equal to a function of the number of times feature j appears in document i ( `f( tf[i,j] )` ), divided by a function of the number of documents feature j appears in ( `g( idf[j]` ) ). In the tf-idf bag-of-words model the `[i,j]`th element is then `f( tf[i,j] )/g( idf[j] )`. 

I considered two methods of creating a data matrix: (1) the standard bag-of-words model, and (2) the tf-idf model, where `f( tf[i,j] ) = 1 + log( tf[i,j] )` and `g( idf[i,j] ) = log( 1 + idf[j] )`. These values for the functions `f()` and `g()` were settled upon after evaluating the performance of the classifiers (Multinomial Naive Bayes, Logistic Regression, Stochastic Gradient Descent, Perceptron, and the Passive Aggressive Classifier) for different functions.

In this project 2-grams and 3-grams of text tokens are considered in addition to the individual tokens themselves, with the restriction that each such feature appear a minimum of two times ( `min( idf[j] ) = 2` ) in the entire collection of documents (tweets). Again, this decision was made after observing that the classifiers (Multinomial Naive Bayes, Logistic Regression, Stochastic Gradient Descent, Perceptron, and the Passive Aggressive Classifier) performed better with these values. 

Using the scikit-learn vectorizers [`CountVectorizer`](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html#sklearn.feature_extraction.text.CountVectorizer) and [`TfidfVectorizer`](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) in the `sklearn.feature_extraction.text` module, I instantiated these objects in the following way:
```python
count_vectz = CountVectorizer(stop_words='english', strip_accents='unicode', 
                              ngram_range=(1,3), max_df=1.0, min_df=2,
                              max_features=None, 
                              tokenizer=SnowballEnglishStemmer())

tfidf_vectz = TfidfVectorizer(stop_words='english', strip_accents='unicode', 
                              ngram_range=(1,3), max_df=1.0, min_df=2, 
                              max_features=None, binary=False, norm=u'l2',
                              use_idf=True, smooth_idf=True, 
                              sublinear_tf=True,
                              tokenizer=rpg.SnowballEnglishStemmer())
```

The below table shows the area under the ROC curves for classifiers, where the only difference is the vectorization method (ROC AUC means and standard errors were calculated using 10-fold cross-validation). These estimated area under the ROC curve scores represent the values for the highest performing classifiers I was able to train.

| Classifier | Area Under ROC Curve (Count Vectorization) | Area Under ROC Curve (Tf-idf Vectorization) |
|:-----------|:------------------------------------------:|:-------------------------------------------:|
| Multinomial Naive Bayes | 0.769 (+/-2SE: 0.038) | 0.791 (+/-2SE: 0.037) | 
| Logistic Regression | 0.747 (+/-2SE: 0.041) | 0.772 (+/-2SE: 0.038) | 
| Perceptron | 0.552 (+/-2SE: 0.039) | 0.572 (+/-2SE: 0.033) | 
| Passive Aggressive Classifier | 0.673 (+/-2SE: 0.045) | 0.738 (+/-2SE: 0.042) |
| Stochastic Gradient Descent | 0.647 (+/-2SE: 0.038) | 0.6973 (+/- 2SE 0.0395) | 

 It generally seems to be the case that the tf-idf vectorization leads to higher performing classifiers in the data I collected. But because the size of the standard errors it is hard to say that there is a statistical difference between the classifiers trained on data produced by either method. Likewise, one cannot definitively say that there is a difference between Multinomial Naive Bayes, Logistic Regression, and the Passive Aggressive Classifier, but it is clear that Perceptron does not perform as well as the former three. There is also evidence that Multiomial Naive Bayes and Logistic Regression perform better than Stochastic Gradient Descent. In nearly all tests Multinomial Naive Bayes had the highest average area under the ROC curve of all other classifiers.

As I noted in the introduction, I attempted to use several well-known and generally high-performing classification methods, but ruled them out due to their performance on smaller test sets as follows (all times are in seconds, and N is the number of documents):

| Classifier | N = 1000 | N = 2000 | N = 4000 | N = 8000 | N = 10000 | N = 16000 |
|:-----------|:------------:|:------------:|:------------:|:------------:|:-------------:|:-------------:|
| Multinomial Naive Bayes                | **0.001** | **0.001** | **0.002** | 0.027 | **0.003** | **0.004** |
| Logistic Regression *                  | 0.428     | 0.584     | 0.915 | 2.011 | 2.177 | 3.687 |
| Support Vector Machines	(linear)       | 0.117     | 0.521     | 2.487 | 11.939 | 19.589 | 56.015 |
| Random Forests                         | 0.256     | 0.269     | 0.280 | 0.508 | 0.790 | 3.878 |
| Ridge Classifier with Cross-validation | 0.132     | 0.162     | 0.208 | 0.539 | 0.421 | 0.687 |
| K-Nearest Neighbors                    | 0.008     | 0.023     | 0.084 | 0.485 | 0.483 | 1.166 |
| Perceptron                             | **0.001** | 0.002     | **0.002** | 0.074 | 0.004 | **0.004** |
| Passive Aggressive Classifier          | **0.001** | **0.001** | **0.002** | **0.004** | **0.003** | 0.005 |
| Linear Discriminant Analysis	         | 0.205     | 0.989 | 5.586 | X | X | X |
| Stochastic Gradient Descent	           | **0.001** | **0.001** | **0.002** | **0.004** | **0.003** | 0.005 |
| AdaBoost                               | 0.303     | 1.253 | 4.911 | 21.391 | 36.622 | 97.917 |
| Bagging	                               | 1.310     | 7.261 | 46.271 | X | X | X |
| Extra Trees Classifier                 | 0.160     | 0.945 | 5.496 | 29.780 | 48.904 | 173.738 |
| Gradient Boosting                      | 1.255     | 8.627 | 35.323 | X | X | X |

From the above data and other tests, I ruled out a large number of possible classifiers. After performing several regressions to estimate the functions describing how several of the classifiers scale with time I empirically determined that Random Forests scaled approximately exponentially (empirical scaling function: `Time = 0.171*exp(N *1.68**-4)`). The Support Vector Machines classifier appeared to scale in polynomial time (emprirical scaling function: `Time = (2.03**-8)*N**(2.28)`). With these empirical scaling functions I calculated that Random Forests would require about 226 years to finish on my machine, and SVM would require around 3 and a third hours, when run on my entire dataset of 145,559 tweets. Needless to say, I did not consider either case within the limits of feasibility

At the end of the data modeling phase I settled on the five classifiers that were most feasible in terms of their scaling properties. Had this project not been conducting text classification other classification methods may have been feasible. However, the large number of tweets combined with the large number of features of each tweet results in very poor classifier scaling in many cases. Feature selection is one method to mitigate this problem and simultaneously improve accuracy, but even for stringent feature selection (10, 100, and 1000 features) the performance of these classifiers (SVC, Random Forests, and ensemble methods) was either not very good, or still required infeasibly large amounts of time to complete.

(* Note: The time test for Logistic Regression also included an extra feature selection step with randomized logistic regression that added to the time that it took to train the Logistic Regression classifier.)

#Training and Testing Classifiers
Once the five classifiers were selected (Multinomial Naive Bayes, Logistic Regression, Stochastic Gradient Descent, Perceptron, and the Passive Aggressive Classifier) the next step was to determine parameters that result in higher classifier performance. 

![NB parm](https://cloud.githubusercontent.com/assets/10871563/6831993/e9adfe84-d2e0-11e4-9c4f-d36bce150dd2.png "ROC AUC for Different Values of Naive Bayes Smoothing Parameters")

Using 10-fold cross-validation to estimate the area under the ROC curve for Multinomial Naive Bayes for different values of the Lidstone smoothing parameter (`alpha` in the scikit-learn implementation), it was apparent that lower values resulted in better performance, as one can see in the above plot. I determined an approximate value for the smoothing parameter as `alpha = 0.005` through an iterative selection process.

![LR parm](https://cloud.githubusercontent.com/assets/10871563/6832002/f39a86a6-d2e0-11e4-8158-878b7b8055cf.png "ROC AUC for Different Values of Logistic Regression Regularization Parameters")

As the above plot indicates for Logistic Regression, regularization parameters above 16.0 do not appear to improve the performance. I selected 16.0 after engaging in an iterative search process.

![PA parm](https://cloud.githubusercontent.com/assets/10871563/6832019/01a04b46-d2e1-11e4-9e31-887ac3665a59.png "ROC AUC for Different Values of Passive Aggressive Classifier Regularization Parameters")

There is a slight performance maximum when the regularization parameter of the Passive Aggressive Classifier is near 0.5, as can be seen in the above plot. `C = 0.5` is the parameter that I settled on as leading to the best ROC area under the curve scores for the Passive Aggressive Classifier.

I used the L2 penalty for Perceptron since the `'elasticnet'` option in the scikit-learn implementation did not appear to alter the performance, and the L1 penalty resulted in worse performance. The best loss function for Stochastic Gradient Descent on my data appears to be the `'squared_huber'` option with L2 penalty. 

![NB cvs](https://cloud.githubusercontent.com/assets/10871563/6833185/667e326a-d2e8-11e4-823a-01a6f2d546cf.png "ROC AUC for Different Cross-validation Folds on Naive Bayes (alpha=0.005) Classifier")

To answer the question of how many folds of cross-validation would be sufficient for comparing the performance of the various classifiers I investigated how the estimates of the area under the ROC curves changed with different folds. In general, 10-fold cross-validation results in estimates that are very similar to higher fold cross-validation estimates. Because the computation time increases somewhat with the fold of cross-validation, I settled on 10-fold cross-validation as the standard method for computing the ROC scoring metrics.

#Best Features
The **Multinomial Naive Bayes classifier** (with smoothing parameter `alpha = 0.005`) determined the following most informative features:

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

P(West) and P(East) are actually the conditional probabilities P(West|feature) and P(East|feature), calculated from the log probabilities returned by the fitted scikit-learn `MultinomialNB` object. When P(West)/P(East) > 1, this indicates that P(West|feature) > P(East|feature). P(West)/P(East) = 1 indicates that P(West|feature) = P(East|feature), and P(West)/P(East) < 1 indicates that P(West|feature) < P(East|feature).

**Logistic Regression** (with regularization parameter `C = 16.0`) determined the following most informative features:

| Best 20 Features for West | Coefficient | Best 20 Features for East | Coefficient | 
|:--------------------------|------------:|:--------------------------|------------:|
| tx |  12.99549 | fl |   -11.96057 | 
| ca https |  8.23921 | toronto |   -9.42345 | 
| seattlefre |   7.99536 | nc |   -9.25324 | 
| san |   7.22084 | nj |   -9.25277 | 
| sacramento |   7.11751 | pa |   -8.47564 | 
| ks |   7.09738 | va |   -8.26559 | 
| portland |   7.06427 | il |   -7.69512 | 
| az |   6.82345 | sc http |   -7.01107 | 
| suonus |   6.57616 | byebyedwi |   -6.34649 | 
| searchfest |   6.48070 | carolina |   -6.28081 | 
| scottsdal |   6.45495 | tn http |   -6.08359 | 
| omaha |   6.34766 | mississippi |   -5.96338 | 
| wa |   6.25125 | tampa |   -5.83711 | 
| katramsland |   6.24692 | orlpol |   -5.83264 | 
| nv |   6.06459 | mi http |   -5.79369 | 
| like noth |   5.93297 | nyc |   -5.75980 | 
| weallgrow |   5.76669 | oh https |   -5.75501 | 
| mtscore |   5.75638 | charlott |   -5.73775 | 
| lwtsummit |   5.75490 | ny |   -5.72951 | 
| texa |   5.74712 | like said |   -5.72190 | 

The coefficient reported here is the coefficient in the Logistic Regresion model where West is coded with 1 and East with 0. Thus high positive coefficient values that correspond to a feature indicate that the feature is associated with the West region, and low negative coefficient values for a feature indicate that the feature is associated with the East region.

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

The interpretation for the coefficient value produced by the Passive Aggressive Classifier is similar to that of Logistic Regression: high positive values for a feature indicate that the feature is associated with the West, and low negative values indicate that the feature is associated with the East.

#Conclusion
Multinomial Naive Bayes with Lidstone smoothing parameter `alpha = 0.005` and Logistic Regression with regularization parameter `C = 16.0` are the top two classification methods for predicting the binary region in which a tweet originates in the tweet data set of 145,559 tweets considered in this project. The Passive Aggressive Classifier with regularization parameter `C = 0.5` comes in third place. 

The most informative features produced by these three classifiers make it clear that names of states or cities are generally of greatest use in predicting the origin of a tweet. The most interesting features among the top 20 are those like: [pdx911](https://twitter.com/pdx911police), [ciaa](http://en.wikipedia.org/wiki/Central_Intercollegiate_Athletic_Association), [searchfest](https://www.sempdx.org/searchfest/), sidewalk clean, [suonus](https://twitter.com/hashtag/suonus), [weallgrow](http://www.weallgrowsummit.com/), [mtscore](https://twitter.com/hashtag/mtscore), [lwtsummit](http://lesbianswhotech.org/summit2015/), like said, fucken, like noth, just offer, [michael kor](http://en.wikipedia.org/wiki/Michael_Kors), and warm weather. These 14 features roughly fall into three main categories: (1) organizations or events that are regionally based, (2) topics of interest and (3) regional expressions.

Those among the first group: pdx911 - a Portland-based police watch-group (indicates West); searchfest - a Portland area digital marketing conference (indicates West); weallgrow - a Los Angeles conference for Latina bloggers (indicates West); lwtsummit - a San Francisco conference for lesbians in tech (indicates West); ciaa - an East Coast collegiate athletic conference (indicates East).

In the second group: the constellation of sidewalk clean, clean request, sidewalk clean request, and street sidewalk clean (all indicate West); mtscore - a twitter hashtag that collects information on Montana sports (indicates West); suonus - a twitter hashtag that collects tweets of concern to Southwestern University students (indicates West); michael kor - (Michael Kors) a New York-based fashion designer and his brand name (indicates East); warm weather - an understandable topic considering the colder and snowier than usual Winter 2015 on the East Coast (one of the few places in the world where temperatures were actually cooler than the average).

The third group: like said - probably "like I said" ("I" would have been removed as a stopword. Indicates East); fucken - this spelling is apparently indicative of the West; like noth - probably "like nothing", a colloquial expression (indicates West); just offer - a somewhat unexpected phrase that sounds vaguely archaic (indicates East). The original motivation for this project was to attempt to classify tweets based on regional English differences; the presence of these phrases in the top 20 features of different classifiers seems to show that this may indeed be possible. With larger data sets collected over longer periods of time it may be easier to build classifiers that make substantial use of regional English variation.

I interpret the final results of this project as providing support for the original hypothesis that Tweets can be effectively classified by region of origin simply from the words in their text. The most promising classifiers in this context appear to be Naive Bayes and Logistic Regression. While one should expect that regional places names (state and city names) will be picked out as being informative of a region, the names of regional events and organizations, user names of popular twitter users, regional topics of interest, and regional differences in English language use all stand out among the most informative features of such classifiers. 
