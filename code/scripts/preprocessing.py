# -*- coding: utf-8 -*-
"""
Created on Sat May 28 14:32:04 2022

@author: Emmanuelle R Nunes
"""

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from re import sub, search
from string import punctuation
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

punctuation_list = list(punctuation) + ['`', '’', '…', '\n']
    
def remove_markdown(sentence):
    # remove markdown tags that can be encountered on Steam reviews and links
    # link format: [text to keep](www.urltoremove.com)
    link = search(r'\[(.*?)\]\(.*?\)', sentence)
    if(link):
        link_to_keep = link.group(1)
        span = link.span()
        sentence = sentence[0:span[0]] + link_to_keep
    return sub(r'\[.*?\]', '', sentence)
    
def remove_punctuation(sentence):
    # remove all punctuation
    return sentence.translate(str.maketrans('', '', ''.join(punctuation_list)))
    
def tokenize(sentence):
    # tokenize words with only numbers and latin characters
    # also turns everything to lowercase
    # input is a single string, output is a list of strings
    tokenizer = RegexpTokenizer(r'[a-zA-Z0-9]+')
    return tokenizer.tokenize(sentence.lower())
    
def lemmatize(sentence):
    # expects list of strings as input
    lemmatizer = WordNetLemmatizer()
    return list(map(lemmatizer.lemmatize, sentence))
    
def make_bigrams(sentence, n):
    # expects list of strings as input
    # adds bigrams onto existing tokens
    grams = []
    
    for i in range(len(sentence)-(n-1)):
        gram = []
        for j in range(i, i+n):
            gram.append(sentence[j])
        grams.append(' '.join(gram))
    return sentence + grams
    
def remove_stopwords(sentence):
    # expects list of strings as input
    stopwords_list = stopwords.words('english') + punctuation_list
    return [word for word in sentence if word not in stopwords_list]
    
def unsplit(sentence):
    # recombines list of strings into single string
    # needed for TF-IDF vectorizer
    # not needed with doc2vec or make_bigrams
    return ' '.join(sentence)