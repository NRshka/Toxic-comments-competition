import os
import re
import csv
import codecs
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from string import punctuation
from collections import defaultdict
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import sys
from configs.config import *


# Regex to remove all Non-Alpha Numeric and space
special_character_removal=re.compile(r'[^?!.,:a-z\d ]',re.IGNORECASE)

# regex to replace all numerics
replace_numbers=re.compile(r'\d+',re.IGNORECASE)
word_count_dict = defaultdict(int)
toxic_dict = {}

def clean_text(text, remove_stopwords=False, stem_words=False, count_null_words=True, clean_wiki_tokens=True):
    # dirty words
    text = text.lower()
    text = re.sub(r"https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)", "", text)
    text = re.sub(r"(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)){3}", "", text)
    
    if clean_wiki_tokens:
        # Drop the image
        text = re.sub(r"image:[a-zA-Z0-9]*\.jpg", " ", text)
        text = re.sub(r"image:[a-zA-Z0-9]*\.png", " ", text)
        text = re.sub(r"image:[a-zA-Z0-9]*\.gif", " ", text)
        text = re.sub(r"image:[a-zA-Z0-9]*\.bmp", " ", text)

        # Drop css
        text = re.sub(r"#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})", " ",text)
        text = re.sub(r"\{\|[^\}]*\|\}", " ", text)
        
        # Clean templates
        text = re.sub(r"\[?\[user:.*\]", " ", text)
        text = re.sub(r"\[?\[user:.*\|", " ", text)        
        text = re.sub(r"\[?\[wikipedia:.*\]", " ", text)
        text = re.sub(r"\[?\[wikipedia:.*\|", " ", text)
        text = re.sub(r"\[?\[special:.*\]", " ", text)
        text = re.sub(r"\[?\[special:.*\|", " ", text)
        text = re.sub(r"\[?\[category:.*\]", " ", text)
        text = re.sub(r"\[?\[category:.*\|", " ", text)
    
    for typo, correct in clean_word_dict.items():
        text = re.sub(typo, " " + correct + " ", text)

    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\?", " ? ", text)
    text = re.sub(r"\!", " ! ", text)
    text = re.sub(r"\"", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = replace_numbers.sub(' ', text)
    #text = special_character_removal.sub('',text)

    if count_null_words:
        text = text.split()
        for t in text:
            word_count_dict[t] += 1
        text = " ".join(text)
    
    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)

    return (text)

def clean_data(train_df, test_df = None): 
    list_sentences_train = train_df["comment_text"].fillna("no comment").values

    comments = [clean_text(text) for text in list_sentences_train]

    if not test_df is None:
        list_sentences_test = test_df["comment_text"].fillna("no comment").values
        test_comments=[clean_text(text) for text in list_sentences_test]
        test_df['comment_text'] = test_comments
    
    train_df['comment_text'] = comments
    
    if test_df is None:
        return train_df
    else:
        return train_df, test_df