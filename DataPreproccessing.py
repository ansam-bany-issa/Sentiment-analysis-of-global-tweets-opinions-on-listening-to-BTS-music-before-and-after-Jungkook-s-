import numpy as np
import pandas as pd 
import re
import string
import os
for dirname, _, filenames in os.walk('dataset_before.csv'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
#read csv file
df = pd.read_csv("dataset_before.csv", encoding="latin-1")
print('Dataset size:',df.shape)

#cleaning Data
def clean_text(text):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text
    
#apply cleaning on Tweet column
df['Tweet'] = df['Tweet'].apply(clean_text)
df.head(20)

#save the table after cleaning in new file
df.to_csv("before_preproccessing.csv")
print('Done')

#Repeat this process for the second dataset
