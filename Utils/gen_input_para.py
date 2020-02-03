#-*-coding:utf-8 -*-

import csv
import json
import os
import pickle
import re
import numpy as np
import pandas as pd
from gensim.models import Word2Vec

DATA_PATH_SPORT     = "/Users/denhiroshi/Downloads/datas/AWS/reviews_Sports_and_Outdoors_5.json"
DATA_PATH_MUSIC     = "/Users/denhiroshi/Downloads/datas/AWS/reviews_Digital_Music_5.json"
DATA_PATH_OFFICE    = "/Users/denhiroshi/Downloads/datas/AWS/reviews_Office_Products_5.json"
PADDING_WORD        = "<PAD/>"
REVIEW_SIZE         = 15

def main(path):
    user_reviews = pickle.load(open(path.replace('.json', 'user_review'), 'rb'))
    item_reviews = pickle.load(open(path.replace('.json', 'item_review'), 'rb'))
    max_len_u    = 0
    max_len_i    = 0
    u_text       = {}
    i_text       = {}
    for urid, review in user_reviews.items():
        line_cleaned = []
        for sen in review[:REVIEW_SIZE]:
            now = clean_str(str(sen)).split(' ')
            max_len_u = max(max_len_u, len(now))
            line_cleaned.append(now)
        u_text[int(urid)] = line_cleaned

    for irid, review in item_reviews.items():
        line_cleaned = []
        for sen in review[:REVIEW_SIZE]:
            now = clean_str(str(sen)).split(' ')
            max_len_i = max(max_len_i, len(now))
            line_cleaned.append(now)
        i_text[int(irid)] = line_cleaned

    para={}
    para['user_num'] = len(u_text)
    para['item_num'] = len(i_text)
    para['user_length'] = max_len_u
    para['item_length'] = max_len_i
    para['u_text'] = u_text
    para['i_text'] = i_text
    para['review_size'] = REVIEW_SIZE
    pickle.dump(para, open(path.replace('.json', '.para'), 'wb'))
    print("OVER")

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
    main(DATA_PATH_MUSIC)