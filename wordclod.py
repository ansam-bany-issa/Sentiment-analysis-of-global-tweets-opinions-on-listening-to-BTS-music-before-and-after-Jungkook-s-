#pip install wordcloud
#pip install matplotlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

#READ THE FILE
test = pd.read_csv('dataset_after.csv')


# wordclod for all the words in train
text = " ".join(text for text in test['Tweet'])

# Create and Generate a Word Cloud Image
wordcloud = WordCloud().generate(text)


# Generate a word cloud image
stopwords = set(STOPWORDS)
mask = np.array(Image.open("kkk.jpg"))
wordcloud = WordCloud(stopwords=stopwords,background_color='black', max_words=800, 
                      mask=mask,contour_color='#023075',max_font_size = 100,contour_width=3,
                      colormap='rainbow').generate(' '.join(test['Tweet']))
# create image as cloud
fig = plt.figure(1, figsize=(7,7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
# store to file
plt.savefig("KPOP.png", format="png")
plt.show()
