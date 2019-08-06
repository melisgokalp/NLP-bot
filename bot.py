import tweepy
import os
import time

consumer_key = ""
# consumer_token = ''
access_token = ""
access_secret = ""
consumer_secret = ""

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)    
auth.set_access_token(access_token, access_secret)
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

# os.chdir('images')

# for image in os.listdir('.'):
#     api.update_with_media(image)
#     time.sleep(20)
with open("dwight.txt", "r") as ins:
    for line in ins: 
        api.update_status(line)
        time.sleep(20)
