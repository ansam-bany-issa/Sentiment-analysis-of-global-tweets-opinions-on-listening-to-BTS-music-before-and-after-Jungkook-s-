#Collecting Data from Twitter
#before you start install snscrape and tweepy library
import snscrape.modules.twitter as sntwitter
import pandas as pd 

query = "(#Jungkook OR #KPOP) until:2022-11-20 since:2022-10-20"
tweets = [] 
limits = 300000 

for tweet in sntwitter.TwitterSearchScraper(query).get_items(): 
    if len(tweets)==limits: 
        break 
    else: tweets.append([tweet.date,tweet.user.location,tweet.content,tweet.lang]) 

pf = pd.DataFrame(tweets, columns=['Date', 'Location','Tweet', 'Language']) 

pf.head(20)
pf.to_csv('dataset_before.csv')