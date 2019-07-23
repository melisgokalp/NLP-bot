import tweepy
import os
import time

consumer_key = 'BivcUFjhHLUqJh2wHx7vTbhqM'
# consumer_token = ''
access_token = '159033588-ooqeO1LuZ1SL4qXUqGnTVs7mkClvqZJz66T4cWii'
access_secret = 'gP84nBtHPo3HiDjPsN7BR6o0pzHVpSyjjs57gjbQCNUie'
consumer_secret = "2l4FlvW7ftgTbq6a4ldgeWgR8SqJ82UyiLINyXGZXFsUPHVCz1"

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