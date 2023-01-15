#cleaning the data
import neattext.functions as nfx
import pandas as pd 

# Read The Dataset
data = pd.read_csv('removeNull_after.csv')

# Select the desired column from the dataset
tweet_text = data['Tweet']

# Preprocessing -- Data Cleaning 
# Remove Hashtags
clean_tweet = tweet_text.apply(nfx.remove_hashtags)
#Remove Users
clean_tweet = clean_tweet.apply(lambda x: nfx.remove_userhandles(x))
#Remove Multiple Spaces
clean_tweet = clean_tweet.apply(nfx.remove_multiple_spaces)
#Remove URLs
clean_tweet = clean_tweet.apply(nfx.remove_urls)
#Remove Special Characters
clean_tweet = clean_tweet.apply(nfx.remove_puncts)
clean_tweet = clean_tweet.apply(nfx.remove_emojis)
clean_tweet = clean_tweet.apply(nfx.remove_special_characters)
clean_tweet = clean_tweet.apply(nfx.remove_punctuations)
#Remove Dates
clean_tweet = clean_tweet.apply(nfx.remove_dates)
#Remove Emails and Phone numbers
clean_tweet = clean_tweet.apply(nfx.remove_emails)
clean_tweet = clean_tweet.apply(nfx.remove_phone_numbers)


#clean_tweet.to_csv('removeNull_after_pre00.csv')
print('done')






#to classifiy the data
from textblob import TextBlob
# Model 
def get_sentiment(clean_tweet):
    blob = TextBlob(clean_tweet)
    sentiment_polarity = blob.sentiment.polarity
    sentiment_subjectivity = blob.sentiment.subjectivity
    if sentiment_polarity > 0:
        sentiment_label = 'Positive'
    elif sentiment_polarity < 0:
        sentiment_label = 'Negative'
    else:
        sentiment_label = 'Neutral'
    result = {'polarity':sentiment_polarity,
              'subjectivity':sentiment_subjectivity,
              'sentiment':sentiment_label}
    return result

# Save the results 
sentiment_results= clean_tweet.apply(get_sentiment)

# Add the results to the data file
data = data.join(pd.json_normalize(sentiment_results)) 
#sentiment_results.to_csv('sentement_before.csv')
print('done')


#Roberta model
# To install the package "pytorch-transformers"
#pip install fastai
import fastai, torch, pytorch_transformers
from fastai.text import *
from fastai.metrics import *
import torch
import torch.nn as nn
from pytorch_transformers import RobertaTokenizer
from pytorch_transformers import RobertaModel
# Garbage Collector
import gc 

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax 

#Loading Model and Tokenizer
roberta = "cardiffnlp/twitter-roberta-base-sentiment" 
model = AutoModelForSequenceClassification.from_pretrained(roberta) 
tokenizer = AutoTokenizer.from_pretrained(roberta)

#The Labels
labels = ['Negative', 'Neutral', 'Positive']
labels_df = []
for index in data.index:
    #Sentiment Analysis
    encoded_tweet = tokenizer(clean_tweet[index], return_tensors='pt')

    output = model(**encoded_tweet)

    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    highest_score = scores[0]
    label = labels[0]
    i = 0
    for score in scores:
        if score > highest_score:
            highest_score = score
            label = labels[i]
        i+=1
    print("The result: ",highest_score, label)

    labels_df.insert(index,label)
    print('************************************************\n')

#Add the preprossed text and the labels into the data frame
data['Tweet'] = clean_tweet
data['Label'] = labels_df

#visualization
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

label_theme = ['#1c78ac', '#dbb152','#89acff'] # color palette
pie = data['Label'].value_counts().plot.pie(figsize=(8,7),autopct= "%.2f%%", colors = label_theme, 
                                                  textprops={'color':"black", 'size' : 'large', 'fontweight' : 'bold'})
pie.set_title('Roberta result before the Show', fontdict = {'color':"black", 'size' : 'xx-large', 'fontweight' : 'bold'})
