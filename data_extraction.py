import pandas as pd
import json
from nltk import *
import matplotlib.pyplot as plt
from collections import Counter
import re
from wordcloud import WordCloud, STOPWORDS
import numpy as np
import text_process as tc

def listAsString(list_string):
    one_string = ''
    for idx, word in enumerate(list_string):
        if idx == len(list_string) - 1:
            one_string = one_string + word
        else:
            temp_string = word + ' '
            one_string = one_string + temp_string 
    return one_string

def tweet_clean(text):
    text = tc.compound_split(text)
    text = tc.remove_punct(text)
    text = tc.remove_nonalpha(text)
    text = tc.tokenizer(text)
    text = tc.remove_stopwords(text)
    text = tc.alpha_filter(text)
    text = [word for word in text if word not in get_stop_words('english')]
    text = tc.stemmer(text)
    text = listAsString(text)
    return text
'''
get_stop_words('english'):
['a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 
"aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 
'but', 'by', "can't", 'cannot', 'could', "couldn't", 'did', "didn't", 'do', 'does', "doesn't", 
'doing', "don't", 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', "hadn't", 
'has', "hasn't", 'have', "haven't", 'having', 'he', "he'd", "he'll", "he's", 'her', 'here', 
"here's", 'hers', 'herself', 'him', 'himself', 'his', 'how', "how's", 'i', "i'd", "i'll", "i'm", 
"i've", 'if', 'in', 'into', 'is', "isn't", 'it', "it's", 'its', 'itself', "let's", 'me', 'more', 
'most', "mustn't", 'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 
'other', 'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same', "shan't", 'she', "she'd", 
"she'll", "she's", 'should', "shouldn't", 'so', 'some', 'such', 'than', 'that', "that's", 'the', 'their', 
'theirs', 'them', 'themselves', 'then', 'there', "there's", 'these', 'they', "they'd", "they'll", "they're", 
"they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', "wasn't", 'we', 
"we'd", "we'll", "we're", "we've", 'were', "weren't", 'what', "what's", 'when', "when's", 'where', "where's", 
'which', 'while', 'who', "who's", 'whom', 'why', "why's", 'with', "won't", 'would', "wouldn't", 'you', 
"you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves']
'''

data = pd.read_csv("tweets.csv")

df = data[['id', 'handle', 'time','text']]

#only get unique 'text'
df = df.drop_duplicates(subset = ["text"])

df = df[df['text'].notnull()].copy() 

df['text'] = df['text'].map(lambda x: " ".join([word for word in str(x).split() if 'http' not in word and '@' not in word and '<' not in word and '#' not in word and 'pic.' not in word]))

# Without combining the text
df_clinton_0 = df[df['handle']=='HillaryClinton']
#df_clinton_0 = df[df['handle']=='realDonaldTrump']

#save the text for each candidate
df_clinton_0.to_csv('clinton.csv', index = False)

df_clinton_0 = df_clinton_0[df_clinton_0['text'].notnull()].copy() 
df_clinton_0['text'] = df_clinton_0['text'].map(lambda x: tweet_clean(x))
df_c = df_clinton_0[df_clinton_0['text'].str.contains("peopl")]
df_c = df_c.reset_index()
df_c.to_csv('trump_stemmed.csv', index = False)


#combine the text
df_filter = df.groupby(['handle'])['text'].apply(lambda x: '.'.join(x.astype(str))).reset_index()
df_trump = df_filter[df_filter['handle']=='realDonaldTrump']

#Clinton: 3224 x 3
df_clinton = df_filter[df_filter['handle']=='HillaryClinton']

text = df_clinton['text'][0]
text = df_trump['text'][1]
text = df_filter['text'][0]

#save the combined text for each candidate
f=open("t_health.txt","w+",encoding="utf-8") 
f.write(text)

#histogram of word frequency in tweets
text = tweet_clean(text)
fdist = FreqDist(text).most_common(20)

word, frequency = zip(*fdist)

indices = np.arange(len(fdist))
plt.bar(indices, frequency, color='purple')
plt.xticks(indices, word, rotation='vertical', fontsize=8)
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.title('Histogram of top 20 words - query: Hillary Clinton')
#plt.title('Histogram of top 20 words - @realDonaldTrump')
plt.tight_layout()
plt.show()

# Word Cloud
def wordcloud(text):
    stopwords = set(STOPWORDS)
    wordcloud = WordCloud(background_color="white",max_font_size=50, max_words=200,stopwords=stopwords,random_state = 2016).generate(text)
    plt.figure(figsize=(10,8), facecolor='k')
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.title("WordCloud - @realDonaldTrump")
    plt.show()

data = pd.read_csv('trump.csv')
data = data.dropna()
data['text'] = data['text'].map(lambda x: x.lower())
df_c = data[data['text'].str.contains('obama|barack obama')]
df = df_c[(df_c['time'] >= "2016-04-17") & (df_c['time'] < "2016-09-18")]
df_filter = df.groupby(['handle'])['text'].apply(lambda x: '.'.join(x.astype(str))).reset_index()

text = df_filter['text'][0]
wordcloud = wordcloud(text)


