#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 12:42:11 2015

Interface with Twitter Api
===============================================================================

Each tweet is assigned a region category after downloading tweets from Twitter.
GPS coordinates define the appropriate bounding boxes corresponding to
 each of six regions (West, Lower West, Upper West, Midwest, South, and East).

@author: gabrielmcneill
"""

import argparse
import os
import pickle
import sys
import time

from twython import TwythonStreamer


###############################################################################
# PARSING COMMAND LINE ARGUMENTS
###############################################################################


parser = argparse.ArgumentParser(description='Save tweets in current dir.')

parser.add_argument('MAX_TWEETS', action='store', default=10, type=int, 
                    help='Maximum number of tweets to get from Twitter.')
parser.add_argument('-f', action='store', type=str, metavar='file_name', 
                    dest='command_file', help='Optional file tweets stored in.'
                    )

args = parser.parse_args()


###############################################################################
# GLOBAL VARIABLES
###############################################################################

# MUST HAVE TWITTER ACCESS TOKENS STORED AS ENVIRONMENT VARIABLES
try:
    APP_KEY = os.environ.get('TWITTER_APP_KEY', None)
    APP_SECRET = os.environ.get('TWITTER_APP_SECRET', None)
    OAUTH_TOKEN = os.environ.get('TWITTER_OAUTH_TOKEN', None)
    OAUTH_TOKEN_SECRET = os.environ.get('TWITTER_OAUTH_TOKEN_SECRET', None)
except:
    e = sys.exc_info()[0]
    print ("Exception: " % e)
    print time.strftime('%X %x %Z')

# bounds define limits as follows: [left, bottom, right, top]
US_BOUNDING_BOX = [-124.78366, 24.545905,-67.097198, 48.997502]

US_REGIONS = {"West": {'box': [-124.78366, 24.545905,-110.897437, 48.997502], 
                       'value': 1},
              "LWest": {'box': [-110.897437, 24.545905, -97.274391, 39.881000], 
                        'value': 2},
              "UWest": {'box': [-110.897437, 39.881000, -97.274391, 48.997502], 
                        'value': 3},
              "Midwest": {'box': [-97.274391, 39.475135, -80.926735, 48.997502], 
                          'value': 4},
              "South": {'box': [-97.274391, 24.545905, -67.097198, 39.475135], 
                        'value': 5},
              "East": {'box': [-80.926735, 39.475135, -67.097198, 48.997502],
                       'value': 6}}

WEST_REGION_BOX = [-124.78366, 24.545905, -93.562895, 48.997502]

MAX_TWEETS = args.MAX_TWEETS

COMMAND_FILE = args.command_file

all_tweets = []

###############################################################################
# CLASS DEFINITIONS
###############################################################################


class MyStreamer(TwythonStreamer):

    def on_success(self, data):
        """Collects tweet data.
        
        Input: (1) data (list): elements are dicts with data on individual 
                   tweets.
        
        Output: None
        """
        self.num_420_errors = 0
        if 'text' in data and 'coordinates' in data:
            all_tweets.append({'text': data['text'].encode('utf-8'),
                              'coordinates': data['coordinates']
                              })
        if len(all_tweets) % 100 == 0:
            print ("Tweets sampled: %s" % len(all_tweets))
            print time.strftime('%X %x %Z')
        
        if len(all_tweets) % 500 == 0:
            pickle_path = "./pickledTweets.p"
            with open(pickle_path, 'wb') as pickle_file:
                pickle.dump(all_tweets, pickle_file)
            print ("Tweets sampled: %s" % len(all_tweets))
            print time.strftime('%X %x %Z')
            
        if len(all_tweets) >= MAX_TWEETS:
            self.disconnect()

    def on_error(self, status_code, data):
        """Detects error codes, displays error info., and implements back-off.
        
        Input: (1) status_code (int): error code.
               (2) data (list): data transmitted from twitter, elements are 
                   dicts with information on individual tweets.
        
        Output: None
        """
        print status_code
        if status_code == 420:
            print "Being rate-limited: too many calls in too short a time."
            time.sleep(60*(2**(self.num_420_errors)))
            self.num420errors += 1
            print ("Num 420 errors: %s" % self.num_420_errors)
            if self.num_420_errors > 4:
                self.disconnect()
        elif status_code == 304:
            print "No new data returned."
        elif status_code == 400:
            print "Invalid Request: check authentication."
        elif status_code == 401:
            print "Unauthorized Credentials."
        elif status_code == 403:
            print "Denied: Update Limit Reached."
        elif status_code == 404:
            print "Invalid URI used."
        elif status_code == 406:
            print "Invalid format in Search request."
        elif status_code == 410:
            print "REST API changed: use v1.1 instead of v.1."
        elif status_code == 429:
            print "Too Many Requests."
        elif status_code == 500:
            print "Twitter Internal Server Error - contact Twitter."
        elif status_code == 502:
            print "Bad Gateway: Twitter down or being upgraded."
        elif status_code == 503:
            print "Twitter servers overloaded with requests - try again later."
        elif status_code == 504:
            print "Twitter servers up, but error in stack. Gateway Timeout."
        else:
            print "Non-standard error: investigate further."


###############################################################################
# FUNCTION DEFINITIONS
###############################################################################


def in_region(region, coord):
    """Tests if coordinate points fall within one of six regions.
    
    Input: (1) region (list): the four elements are coordinates of bounding box
               that defines a region: [left, bottom, right, top]
           (2) coord (list): the two elements are point coordinates of tweet 
               origin: [longitude, latitude]
    
    Output: (bool) returns true if coord coordinates are within the box defined
            by region.
    """
    if coord[0] < region[0]:
        return False
    elif coord[0] > region[2]:
        return False
    elif coord[1] < region[1]:
        return False
    elif coord[1] > region[3]:
        return False
    else:
        return True


def process_tweets(tweet_data):
    """"Adds region and west variables to data using coordinate information.
    
    Input: (1) tweet_data (list): elements are dicts containing information on
               individual tweets.
    
    Output: ():
    
    """

    for tweet in tweet_data:
        if tweet['coordinates'] is None:
            tweet['region'] = None
            tweet['west'] = None
        else:
            coordinates = tweet['coordinates']['coordinates']
            if in_region(US_BOUNDING_BOX, coordinates):
                for key in US_REGIONS:
                    if in_region(US_REGIONS[key]['box'], coordinates):                                 
                        tweet['region'] = US_REGIONS[key]['value']
                        break
                if in_region(WEST_REGION_BOX, coordinates):
                    tweet['west'] = 1
                else:
                    tweet['west'] = 0  # if not in West, then in East
            else:
                tweet['region'] = None  # if not in US then disregard tweet
                tweet['west'] = None
    return None


###############################################################################
# GETTING TWEETS
###############################################################################


stream = MyStreamer(APP_KEY, APP_SECRET, OAUTH_TOKEN, OAUTH_TOKEN_SECRET, 
                    timeout = 300, retry_count = 3, retry_in = 15, 
                    client_args = None, handlers = None, chunk_size = 100)
    
while len(all_tweets) < MAX_TWEETS:
    try:
        stream.statuses.filter(language='en', locations=US_BOUNDING_BOX, 
                               stall_warnings=True)
    except:
        e = sys.exc_info()[0]
        print("Exception raised: %s" % e)
        print time.strftime('%X %x %Z')
        continue


process_tweets(all_tweets)


if COMMAND_FILE is None:
    tweet_file = 'tweets' + time.strftime("%Y%m%d") + '.p'
else:
    tweet_file = COMMAND_FILE

tweet_path = os.path.join(os.environ.get('PWD', None), tweet_file)
with open(tweet_path, 'wb') as tweet_file:
    pickle.dump(all_tweets, tweet_file)
