import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk import *
from stop_words import get_stop_words

# Words overlapping plot
# remove punctuations [!”#$%&’()*+,-./:;<=>?@[\]^_`{|}~]
def remove_punct(text):#output-sentence
    exclude = set(string.punctuation)
    s_no_punct = ''.join(ch for ch in text if ch not in exclude)
    return s_no_punct

def compound_split(text):#output-sentence
    # change compound words to separate words ie. 'conditional-statements' -> 'conditional', 'statements'
    regex = re.compile("[-_'.]")
    trimmed = regex.sub(' ', text)
    return trimmed

def remove_stopwords(word_list):#output-words
    english_stop_words = set(stopwords.words('english'))
    processed_word_list = []
    for word in word_list:
        if word not in english_stop_words:
            processed_word_list.append(word)
    return processed_word_list
'''
set(stopwords.words('english')):
{'for', 'by', 'until', 'mustn', 'other', 'same', "mustn't", 'or', 'in', 'off', 'aren', 'now', 
'don', 'what', 'haven', 'as', 'have', "hasn't", "needn't", 'they', 'nor', 'to', 'not', "isn't", 
't', 'few', 'myself', 'it', "you're", 'that', 'being', 'out', 'if', 'yourself', "couldn't", 
"don't", 'our', 'down', 'so', 'over', 'more', 'o', 'their', 'y', "you'd", 'again', 'no', 'into', 
'each', 'herself', 'on', 'do', 'against', "she's", 'hers', "it's", 'wasn', "shouldn't", 'very', 
'were', 'through', 'at', 'himself', 'has', 'does', 'doing', 'under', "wouldn't", 'from', 're', 
'then', 'how', 'above', 'didn', 'an', 'during', 've', 'this', 'me', 'which', 'mightn', 'shan', 
'too', 'why', 'after', 'them', 'yours', 'him', 'doesn', 'itself', 'been', 'did', 'themselves', 
'between', 'before', 'below', 'needn', 'and', 'while', 'these', 'than', 'there', 'a', 's', 'yourselves', 
"won't", 'she', 'the', 'any', 'some', 'should', 'wouldn', "that'll", 'further', 'who', 'once', 
'are', 'weren', 'because', "shan't", 'm', 'you', 'be', 'was', 'of', "mightn't", 'theirs', 'your', 
'here', 'ours', "weren't", 'couldn', "wasn't", 'isn', 'am', 'just', 'i', 'ain', 'shouldn', 'those', 
'is', "you'll", 'where', 'hadn', 'won', 'we', "didn't", 'when', 'most', "you've", 'hasn', 'd', "hadn't", 
'up', 'his', 'both', 'only', 'will', 'ma', 'such', 'having', "aren't", "haven't", 'about', 'with', "should've", 
'its', 'my', 'own', 'can', 'whom', 'ourselves', 'but', "doesn't", 'her', 'all', 'll', 'had', 'he'}
'''
def remove_nonalpha(text):#output-sentence
    regex = re.compile('[^a-zA-Z]')
    nonAlphaRemoved = regex.sub(' ', text)
    return nonAlphaRemoved

def tokenizer(sentence):#output-words
    return word_tokenize(sentence.lower())

def alpha_filter(wordslist):
    words_list = [word for word in wordslist if len(set(word)) > 1]
    return words_list

def stemmer(word_list):#output-words
    #PorterStemmer
    porter = PorterStemmer()
    stemmed = [porter.stem(word) for word in word_list]
    return stemmed

