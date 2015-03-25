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
The motivation for this project is to evaluate the hypothesis that the geographical origin of a tweet can be predicted using its text alone. Well known studies of regional linguistic variation have demonstrated that it is possible to cluster variations in vocabulary, pronunciation, and grammar in English by geographical regions of the United States ([New York Times interactive dialect map](http://www.nytimes.com/interactive/2013/12/20/sunday-review/dialect-quiz-map.html)). However, when attempting to classify tweet text by US region there are a couple of challenges to overcome in order to create a reasonable classifier: (1) Twitter enforces a 140 character limit on individual tweets, resulting in extremely sparse feature matrices (using the bag-of-words model) and (2) most defining features of regional linguistic variation are not common in normal English text, let alone in the abbreviated form often used in tweets. 

The character limit on tweets makes it crucial to collect as large a sample as possible, and to carefully select features used in training a classifier. Perhaps the most critical aspect of training any text classifier is in the choice of text features, for it is not only an important factor in training a strong classifier, but it also can affect how well a classification method scales under increasing numbers of observations. My initial approach in this project was to compile a list of words and phrases that could possibly serve as informative features. I did so by compiling a list of 381 words and phrases identified by linguists as being characteristic of certain regional dialects. However, none of the 381 words or phrases appeared among an initial test sample of 5,499 tweets, which made it clear that such words were most likely not going to be helpful in training a good classifier (only 78 of the 381 appear in my final sample of 145,559 tweets, and none is among the most informative features). It was clear to me that a pure machine learning approach would be necessary to train a strong classifier.

I considered a wide range of potential classification algorithms including Support Vector Machines, Random Forests, and ensemble methods (AdaBoost, Bagging, Gradient Boosting), but in benchmark timing tests that I ran I determined them to be infeasible due to their poor scaling when applied to my tweet database (145,599 observations of 108,400 features), and when performed on the machines I have access to. I ruled out other classifiers (Ridge Classification, K-Nearest Neighbors, Linear Discriminant Analysis, Quadratic Discriminant Analysis) for similar reasons, and in the end evaluated five different feasible classification methods: Multinomial Naive Bayes, Randomized Logistic Regression, Stochastic Gradient Descent, Perceptron, and the Passive Aggressive Classifier. 


#Region Definition
I define two regions of the US in which to classify individual tweets as originating: West and East. The West region is defined as the area bounded by the latitude lines at 24.545905 and 48.997502 degrees, and the longitude lines at -124.78366 and -93.562895 degrees. The East region is likewise defined as the are bounded by the latitude lines at 24.545905 and 48.997502 degrees, and the longitude lines at -93.562895 and -67.097198 degrees. This roughly corresponds to the West region as the rectangle encompassing all states to the west of the border between Texas and Louisiana; all states to the east of that border belong to the East region.

A large proportion of the tweets that I collected originated in states on the East coast, in the South, or in the Midwest. In my final sample of 145,559 tweets, 39.3 percent of them originated in the West region and 60.7 percent of them originated in the East region despite having drawn the dividing longitude line so far to the East. Several factors may have contributed to having collected a larger proportion of tweets in the East region than the West region: (1) people may tweet at higher rates in the South, East, and Midwest than do in the states in the West, and (2) the time of day of data collection may have resulted in a sample that covers the peak times of tweet activity in eastern and southern states but not the peak times of the western states.

At the beginning of the project I originally defined six different US regions (West, Lowerwest, Upperwest, Midwest, South, and East), but because of the comparatively few tweets in the Lowerwest and Upperwest regions training a classifier proved difficult. The large proportion of tweets from the South tended to heavily bias the classifiers to only predict South, leading to classification accuracy only slightly above what one would expect if one were to always classify each tweet as originating in the South. I switched to a binary regional split between East and West in order to have both a more balanced proportion of tweets per region, and to maintain interpretability, defining regional divisions that roughly correspond to intuitively coherent geographical areas.

#Data Collection - Interfacing With The Twitter API
The program (`twitter_interface.py`) runs from the command line, samples tweets by interacting with the [Twitter API](https://dev.twitter.com/streaming/overview), and deposits them into a pickle (`.p`) file into the current directory. It accepts two command-line arguments: `MAX_TWEETS`, which determines the number of tweets to sample, and an optional `file_name` that the user can set to name the name of the file containing the tweet data. The Twitter streaming API requires [authentication](https://dev.twitter.com/oauth/overview), so in order to use `twitter_interface.py` one must have a twitter account and apply for the proper authentication keys:
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

When ignoring all non-ascii characters, my data set of 145,559 tweets can be summarized as follows:

| Measure on Number of Characters | Value | Measure on Number of Words | Value |
|:--------------------------------|------:|:---------------------------|------:|
| Mean Tweet Length (chars)       | 63.77 | Mean # Words per Tweet        | 11.44 |
| Median Tweet Length (chars)     | 56    | Median # Words per Tweet      | 10    |
| Minimum Tweet Length (chars)    | 4     | Minimum # Words in Tweet     | 1     |
| Maximum Tweet Length (chars)    | 140   | Maximum # Words in Tweet     | 43    |

![char hist](https://cloud.githubusercontent.com/assets/10871563/6831921/908ef998-d2e0-11e4-9e47-20265625a1b3.png "Histogram of Number of Characters per Tweet")

Many of the tweets in the 120 to 125 character length range have several URLs, which inflates the character length. The steady rise up to the character limit of 140 seems to be due to tweets that are composed primarily of standard ascii characters.

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

The below table shows the area under the ROC curves for classifiers, where the only difference is the vectorization method (ROC AUC means and standard errors were calculated using 10-fold cross-validation).

| Classifier | Area Under ROC Curve (Count Vectorization) | Area Under ROC Curve (Tf-idf Vectorization) |
|:-----------|:------------------------------------------:|:-------------------------------------------:|
| Multinomial Naive Bayes | 0.769 (+/-2SE: 0.038) | 0.791 (+/-2SE: 0.037) | 
| Randomized Logistic Regression | 0.747 (+/-2SE: 0.041) | 0.772 (+/-2SE: 0.038) | 
| Perceptron | 0.552 (+/-2SE: 0.039) | 0.572 (+/-2SE: 0.033) | 
| Passive Aggressive Classifier | 0.673 (+/-2SE: 0.045) | 0.738 (+/-2SE: 0.042) |
| Stochastic Gradient Descent | 0.647 (+/-2SE: 0.038) | 0.632 (+/-2SE: 0.040) | 

As the above table suggests, it is generally seems to be the case that the tf-idf vectorization leads to higher performing classifiers in the data I collected.

As I noted in the introduction, I attempted to use several well-known and high-performing classification methods, but ruled them out due to their performance on smaller test sets as follows (all times are in seconds, and N is the number of documents):

| Classifier | N = 1000 | N = 2000 | N = 4000 | N = 8000 | N = 10000 | N = 16000 |
|:-----------|:------------:|:------------:|:------------:|:------------:|:-------------:|:-------------:|
| Multinomial Naive Bayes                | **0.001** | **0.001** | **0.002** | 0.027 | **0.003** | **0.004** |
| Randomized Logistic Regression         | 0.428     | 0.584     | 0.915 | 2.011 | 2.177 | 3.687 |
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

From the above data and other tests, I ruled out a large number of possible classifiers. After performing several regressions to estimate the functions describing how several of the classifiers scale with time I empirically determined that although Randomized Logistic Regression appeared to have similar times as Random Forests, Randomized Logistic Regression scaled approximately linearly, and Random Forests scaled approximately exponentially (empirical scaling function: `Time = 0.171*exp(N *1.68**-4)`). The Support Vector Machines method appeared to scale in polynomial time (emprirical scaling function: `Time = (2.03**-8)*N**(2.28)`). With these empirical scaling functions I calculated that Random Forests would require about 226 years to finish on my machine, and SVM would require around 3 and a third hours, when run on my entire dataset of 145,559 tweets.

At the end of the data modeling phase I settled on the five classifiers that were most feasible in terms of their scaling properties. Had this project not been conducting text classification other classification methods may have been feasible. However, the large number of tweets combined with the large number of features of each tweet results in very poor classifier scaling in many cases. Feature selection is one method to mitigate this problem and simultaneously improve accuracy, but even for stringent feature selection (10, 100, and 1000 features) the performance of these classifiers was either not very good, or still required infeasibly large amounts of time to complete.

#Training and Testing Classifiers
Once the five classifiers were selected (Multinomial Naive Bayes, Logistic Regression, Stochastic Gradient Descent, Perceptron, and the Passive Aggressive Classifier) the next step was to determine parameters that result in higher classifier performance. 

![NB parm](https://cloud.githubusercontent.com/assets/10871563/6831993/e9adfe84-d2e0-11e4-9c4f-d36bce150dd2.png "ROC AUC for Different Values of Naive Bayes Smoothing Parameters")

Using 10-fold cross-validation to estimate the area under the ROC curve for Multinomial Naive Bayes for different values of the Lidstone smoothing parameter (`alpha` in the scikit-learn implementation), it was apparent that lower values resulted in better performance, as one can see in the above plot. I determined an approximate value for the smoothing parameter as `alpha = 0.005`.

![LR parm](https://cloud.githubusercontent.com/assets/10871563/6832002/f39a86a6-d2e0-11e4-8158-878b7b8055cf.png "ROC AUC for Different Values of Logistic Regression Regularization Parameters")

As the above plot indicates for Logistic Regression, regularization parameters above 16.0 do not appear to improve the performance. I selected 16.0 after engaging in an iterative search process.

![PA parm](https://cloud.githubusercontent.com/assets/10871563/6832019/01a04b46-d2e1-11e4-9e31-887ac3665a59.png "ROC AUC for Different Values of Passive Aggressive Classifier Regularization Parameters")

There is a slight performance maximum when the regularization parameter of the Passive Aggressive Classifier is near 0.5, as can be seen in the above plot. `C = 0.5` is the parameter that I settled on as leading to the best ROC area under the curve scores for the Passive Aggressive Classifier.

![NB cvs](https://cloud.githubusercontent.com/assets/10871563/6833185/667e326a-d2e8-11e4-823a-01a6f2d546cf.png "ROC AUC for Different Cross-validation Folds on Naive Bayes (alpha=0.005) Classifier")

To answer the question of how many folds of cross-validation would be sufficient for comparing the performance of the various classifiers I investigated how the estimates of the area under the ROC curves changed with different folds. In general, 10-fold cross-validation results in estimates that are very similar to higher fold cross-validation estimates. Because the computation time increases with the fold of cross-validation, I settled on 10-fold cross-validation as the standard method for computing the ROC scoring metrics.

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
