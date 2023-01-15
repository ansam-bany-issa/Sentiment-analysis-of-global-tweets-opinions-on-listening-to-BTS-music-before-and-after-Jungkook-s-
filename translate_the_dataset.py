#translate the dataset
import neattext.functions as nfx
import pandas as pd
from textblob import TextBlob
import textblob.exceptions

# Read The Dataset
data = pd.read_csv('removeNull_after.csv')

translated_tweet = []
for index in data.index:
    tweet = TextBlob(data['Tweet'][index])
    language = data['Language'][index]
    if(language == 'bg' or language=='hr' or language=='cs' or language=='da' or language=='nl' or language=='et'
      or language=='ga' or language=='hu' or language=='el' or language=='de' or language=='fr' or language=='fi'
      or language=='it' or language=='lv' or language=='lt' or language=='mt' or language=='pl' or language=='pt'
      or language=='sv' or language=='es' or language=='sl' or language=='sk' or language=='ro' or language=='ar'
      or language=='ml' or language=='mr' or language=='te' or language=='my' or language=='bn' or language=='ta'
      or language=='ur' or language=='is' or language=='cy' or language=='uk' or language=='ckb' or language=='fa'
      or language=='eu' or language=='hi' or language=='no' or language=='ht' or language=='zh' or language=='ca'
      or language=='vi' or language=='ru' or language=='tl' or language=='tr' or language=='qht' or language=='in'
      or language =='und' or language=='th' or language=='qme' or language=='ko' or language=='ja'):
        #print(language)
        try:
            tr_tweet = tweet.translate(from_lang=language, to='en')
            translated_tweet.insert(index,tr_tweet)
        except:    
            print('skip!')
            translated_tweet.insert(index,data['Tweet'][index])
            continue
        else:
            translated_tweet.insert(index,data['Tweet'][index])
            
data['Translated_Tweet'] = translated_tweet
data.to_csv('before_tran.csv')
