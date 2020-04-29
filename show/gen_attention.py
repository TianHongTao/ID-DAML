#-*-coding:utf-8 -*-

import csv
import copy
import json
import os
import pickle
import re
import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
import pandas as pd
import logging
import random
# from FM import FactorizationMachine
from tqdm import tqdm
from gensim.models import Word2Vec
from torch.utils.data.dataset import Dataset
import sys
sys.path.append('..')
from search.DistanceImprovedDAML import IDDAML
from search.ImprovedDAML import IDAML
import webbrowser


DATA_PATH_MUSIC     = "/Users/denhiroshi/Downloads/datas/AWS/reviews_Musical_Instruments_5.json"
DATA_PATH_MUSIC2    = "/Users/denhiroshi/Downloads/datas/AWS/reviews_Sports_and_Outdoors_5.json"
DATA_PATH_MUSIC3     = "/Users/denhiroshi/Downloads/datas/AWS/reviews_Digital_Music_5.json"
MODEL_DICT          = '../test/ImprovedDAML.tar' #reviews_Musical_Instruments_5
# MODEL_DICT          = '../test/ImprovedDAML1.tar' #reviews_Sports_and_Outdoors_5
# MODEL_DICT          = '../test/ImprovedDAML2.tar' #reviews_Digital_Music_5

BATCH_SIZE          = 1
EPOCHS              = 50
LEARNING_RATE       = 0.001
CONV_LENGTH         = 3
CONV_KERNEL_NUM     = 16
FM_K                = 1 #Factorization Machine 交叉向量维度
LATENT_FACTOR_NUM   = 32
GPU_DEVICES         = 0
ID_EMBEDDING_DIM    = 32
ATTEN_VEC_DIM       = 16
REVIEW_SIZE         = 15
ATT_CONV_SIZE       = 3

def gen_texts(texts, word_dict, max_len, review_size):
    text_dict  = {}
    for t_id, text in texts.items():
        sen_indices = []
        for sen in text:
            if len(sen) < max_len:
                num_padding = max_len - len(sen)
                sen += [ "<PAD/>"] * num_padding
            word_indices = [word_dict[w] if w in word_dict else word_dict["<UNK/>"] for w in sen]
            sen_indices.append(word_indices)
        if(review_size > len(sen_indices)):
            num_padding = review_size - len(sen_indices)
            sen_indices += [[ word_dict["<PAD/>"]] * max_len] * num_padding
        text_dict[t_id] = sen_indices
    return text_dict

def main(path):
    SAVE_DIR    = os.path.sep.join(path.split(os.path.sep)[:-1])
    print("SAVE_DIR: " + SAVE_DIR)

    para        = pickle.load(open(path.replace('.json', '.para'), 'rb'))
    review_size = para['review_size']
    word_model  = Word2Vec.load(path.replace('.json', '.model'))
    word_model.wv.add("<UNK/>", np.zeros(word_model.vector_size))
    word_model.wv.add("<PAD/>", np.zeros(word_model.vector_size))
    word_dict       = {w: i for i, w in enumerate(word_model.wv.index2entity)}
    word_weights    = torch.FloatTensor(word_model.wv.vectors)
    u_texts         = para['u_text']
    i_texts         = para['i_text']
    u_keys  =  list(para['u_text'].keys())
    i_keys  =  list(para['i_text'].keys())
    u_text_dict     = gen_texts(para['u_text'], word_dict, para['user_length'], review_size)
    i_text_dict     = gen_texts(para['i_text'], word_dict, para['item_length'], review_size)
    review_length   = para['user_length']
    word_vec_dim    = word_weights.shape[1]
    user_num        = para['user_num']
    item_num        = para['item_num']
    del para
    del word_model
    del word_dict

    model = IDAML(
        filter_size=CONV_LENGTH, 
        latent_factor_num=LATENT_FACTOR_NUM, 
        conv_kernel_num=CONV_KERNEL_NUM, 
        word_vec_dim=word_vec_dim, 
        att_conv_size=ATT_CONV_SIZE,
        u_id_len=user_num,
        i_id_len=item_num,
        fm_k=FM_K,
        word_weights=word_weights,
        review_size=review_size,
        is_gen=True
    )
    model.load_state_dict(torch.load(MODEL_DICT, map_location=lambda storage, loc: storage))
    u_train = []
    i_train = []
    r_train = []
    with open(path.replace('.json', '_rating_train.csv')) as f:
        for line in f.readlines():
            line = line.strip()
            line=line.split(',')
            u_train.append(int(line[0]))
            i_train.append(int(line[1]))
            r_train.append(float(line[2]))
    test_index  = random.randint(0, len(u_train)-1)
    uid = u_train[test_index]
    iid = i_train[test_index]
    rate = r_train[test_index]
    # u_valid = []
    # i_valid = []
    # r_valid = []
    # with open(path.replace('.json', '_rating_valid.csv')) as f:
    #     for line in f.readlines():
    #         line = line.strip()
    #         line=line.split(',')
    #         u_valid.append(int(line[0]))
    #         i_valid.append(int(line[1]))
    #         r_valid.append(float(line[2]))
    # test_index  = random.randint(0, len(u_valid)-1)
    # uid = u_valid[test_index]
    # iid = i_valid[test_index]
    # rate = r_valid[test_index]
    text_u = u_texts[uid]
    text_i = i_texts[iid]
    u_text = u_text_dict[uid]
    i_text = i_text_dict[iid]
    u_text = torch.LongTensor(u_text).unsqueeze(0)
    i_text = torch.LongTensor(i_text).unsqueeze(0)
    uid = torch.LongTensor([uid])
    iid = torch.LongTensor([iid])
    pred = model(u_text, i_text, uid, iid)
    Attention_rates = model.Attention_rates
    attention_sen = []
    with torch.no_grad():
        # print(Attention_rates)
        for i, text in enumerate(text_u):
            now = " ".join(text).replace(" <PAD/>", "").split(" ")
            # print(" ".join(text).replace(" <PAD/>", ""))
            # print(len(now))
            for key, value in Attention_rates.items():
                if "_i_" in key:
                    continue
                value = value.reshape((value.shape[0],-1))
                attention_sen.append("<strong>" +key + ":</strong> " + mk_html(now, F.softmax(value[i, 0:len(now)]).tolist()))
        attention_sen.append("<br><br>\n")
        for i, text in enumerate(text_i):
            now = " ".join(text).replace(" <PAD/>", "").split(" ")
            # print(" ".join(text).replace(" <PAD/>", ""))
            # print(len(now))
            for key, value in Attention_rates.items():
                if "_u_" in key:
                    continue
                value = value.reshape((value.shape[0],-1))
                attention_sen.append("<strong>" +key + ":</strong> " + mk_html(now, F.softmax(value[i, 0:len(now)]).tolist()))
        #命名生成的html
        GEN_HTML = "Attention.html" 
        #打开文件，准备写入
        f = open(GEN_HTML,'w')
        #准备相关变量
        message = """
        <html>
        <head></head>
        <body>
        <h2>index: %s</h2>
        <h2>pred: %s</h2>
        <h2>rate: %s</h2>
        <p>%s</p>
        </body>
        </html>"""%(test_index, pred.tolist()[0], rate, "".join(attention_sen))  
        #写入文件
        f.write(message) 
        #关闭文件
        f.close()
        #运行完自动在网页中显示
        webbrowser.open(GEN_HTML,new = 1)


def highlight(word, attn):
    html_color = '#%02X%02X%02X' % (255, int(255*(1 - attn)), int(255*(1 - attn)))
    return '<span style="background-color: {}">{}</span>'.format(html_color, word)

def mk_html(seq, attns):
    html = ""
    for i, word in enumerate(seq): 
        html += ' ' + highlight(
            word,
            attns[i]
        )
    return html + "<br><br>\n"


if __name__ == "__main__":
    path = DATA_PATH_MUSIC3
    main(path)