# wordclod for all the words in train
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import os
for dirname, _, filenames in os.walk('after_pre.csv'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
test = pd.read_csv('after_pre.csv')
print('read')

txt = " ".join(text for text in test['Tweet'])

wordcloud = WordCloud(max_font_size = 100, max_words = 150, background_color = 'black').generate(txt)
fig = plt.figure(1, figsize=(10,10))
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis("off")
plt.show()
