import tweepy
import os
import time
import pandas as pd     
import numpy as np 
from collections import Counter 
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.feature_extraction import stop_words

sorted(list(stop_words.ENGLISH_STOP_WORDS))[:20]

consumer_key = ""
access_token = ""
access_secret = ""
consumer_secret = ""

def get_api():
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)    
    auth.set_access_token(access_token, access_secret)
    # api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
    api = tweepy.API(auth)
    return api 

def get_tweets():
    api = get_api() 
    # tweets = api.get_tweets(query = 'Donald Trump', count = 200) 
    tweets = api.user_timeline(screen_name="realDonaldTrump", count=200, tweet_mode="extended")
    print("Number of tweets extracted: {}.\n".format(len(tweets)))

    print("5 recent tweets:\n")
    for tweet in tweets[:5]:
        print(tweet.full_text)

    split_it = tweets[0].full_text.split()
    # print(split_it)

    for tweet in tweets[:200]:
        tweet.full_text = tweet.full_text.replace("RT", "")
        split_it = split_it+ tweet.full_text.split()
split_it = list()
get_tweets()

begin = list()
mid = list()
mid2 = list()
mid3 = list()
end = list()

# f= open("splits.txt","w+")
# with open("splits.txt", "r") as tweetskd:

for tweet in tweets:
    # tweet.full_text = tweet.full_text.replace("RT", "")
    # # f.write(tweet.full_text+ "\n")
    this = tweet.full_text.split()

    this = tweet.split()
    for i in range(len(this)):
        if ("http" in this[i]) or  ("@" in this[i]) or ("&amp" in this[i]) or ("RT" in this[i]):
            this[i] = ''
        if(this[i] in stop_words.ENGLISH_STOP_WORDS):
            # print("STOP WORD FOUND")
            this[i] = '' 
    while("" in this) : 
        this.remove("")  
    split_it = split_it+ this 
    n = int(len(this)/5)-1 
    begin.append(" ".join(this[0]))
    mid.append(" ".join(this[n:2*n]))
    mid2.append(" ".join(this[2*n:3*n]))
    mid3.append(" ".join(this[3*n:4*n]))
    end.append(" ".join(this[4*n:len(this)])) 
# f.close()  
# print(len(mid))
# print(len(end))
mid =  mid2 + mid3 + mid
contweet = list()
print(len(mid))
for i in range(60):
    # break
    contweet.append(begin[random.randint(0,199)]+" "+mid[random.randint(0,599)]+" "+mid[random.randint(0,599)]+" "+mid[random.randint(0,599)]+" "+end[random.randint(0,199)])
# print(begin[random.randint(1,198)]+" "+mid[random.randint(1,198)]+" "+end[random.randint(1,198)]) 
# print(split_it) 
f= open("trtweets.txt","w+")
for line in contweet: 
        line = line.replace("  ", " ")
        # api.update_status(line)
        # time.sleep(20)
        f.write(line + "\n\n")
f.close() 

# Pass the split_it list to instance of Counter class. 
Counter = Counter(split_it) 
  
# most_common() produces k frequently encountered 
# input values and their respective counts. 
most_occur = Counter.most_common(50) 
  
print(most_occur) 
 
