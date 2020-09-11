#-----------------------------------named entity recognition---------------------------------------#
import spacy
from spacy import displacy
from collections import Counter
import requests
import en_core_web_sm
import re
from nltk import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#load the model
nlp = en_core_web_sm.load()

#input the text
#data = data[:999999] #data <1000000
text = open("trump.txt","r+",encoding="utf8")
data = text.read()

article = nlp(data)
len(article.ents)

#entities
ents = article.ents

#labels - Person, ORG...
labels = [x.label_ for x in article.ents]
Counter(labels)

person = []
for i in range(len(ents)):
    if ents[i].label_ == "PERSON":
        person.append(ents[i].text)

org = []
for i in range(len(ents)):
    if ents[i].label_ == "ORG":
        org.append(ents[i].text)

#plot the top 20 person names
fdist = FreqDist(person).most_common(20)
word, frequency = zip(*fdist)

indices = np.arange(len(fdist))
plt.bar(indices, frequency, color='purple')

plt.xticks(indices, word, rotation='vertical', fontsize=12)
plt.xlabel('Person Names')
plt.ylabel('Frequency')
#plt.title('Histogram of top 20 Person names - @HillaryClinton')
plt.title('Histogram of top 20 Person names - @realDonaldTrump')
plt.tight_layout()
plt.show()

#plot the top 20 organization names
fdist = FreqDist(org).most_common(20)
word, frequency = zip(*fdist)

indices = np.arange(len(fdist))
plt.bar(indices, frequency, color='blue')
plt.xticks(indices, word, rotation='vertical', fontsize=12)
plt.xlabel('Organization Names')
plt.ylabel('Frequency')
#plt.title('Histogram of top 20 Organization names - @HillaryClinton')
plt.title('Histogram of top 20 Organization names - @realDonaldTrump')
plt.tight_layout()
plt.show()
