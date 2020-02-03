#-*-coding:utf-8 -*-

import csv
import json
import os
import re
import pickle
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from gensim.test.utils import common_texts, get_tmpfile


DATA_PATH_SPORT     = "/Users/denhiroshi/Downloads/datas/AWS/reviews_Sports_and_Outdoors_5.json"
DATA_PATH_MUSIC     = "/Users/denhiroshi/Downloads/datas/AWS/reviews_Digital_Music_5.json"
DATA_PATH_OFFICE    = "/Users/denhiroshi/Downloads/datas/AWS/reviews_Office_Products_5.json"
DATA_PATH_MUSIC2    = "/Users/denhiroshi/Downloads/datas/AWS/reviews_Musical_Instruments_5.json"

def main(path):
    sentences    = []
    user_reviews = pickle.load(open(path.replace('.json', 'user_review'), 'rb'))
    item_reviews = pickle.load(open(path.replace('.json', 'item_review'), 'rb'))
    u_text       = []
    i_text       = []
    for urid, review in user_reviews.items():
        for sen in review:
            line_cleaned = clean_str(sen).split(' ')
            u_text.append(line_cleaned)

    for irid, review in item_reviews.items():
        for sen in review:
            line_cleaned = clean_str(sen).split(' ')
            i_text.append(line_cleaned)

    with open(path) as f:
        for line in f:
            line = line.strip()
            info = json.loads(line)
            sentences.append(info['reviewText'].split(' '))
    sentences += u_text
    sentences += i_text
    model = Word2Vec(sentences, size=100, window=3, min_count=1, workers=3)
    model.save(path.replace('.json', '.model'))
    del sentences
    del model
    del u_text
    del i_text
    print(path + " OVER")

def clean_str(string):
    string = re.sub(r"[^A-Za-z]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

if __name__ == "__main__":
    # main(DATA_PATH_MUSIC)
    # main(DATA_PATH_SPORT)
    # main(DATA_PATH_OFFICE)
    main(DATA_PATH_MUSIC2)