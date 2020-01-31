#-*-coding:utf-8 -*-

import csv2json(path,)
import copy
import json
import os
import pickle
import math
import random
import re
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import logging
from FM import FactorizationMachine
from tqdm import tqdm
from gensim.models import Word2Vec
from torch.utils.data.dataset import Dataset

DATA_PATH_MUSIC     = "/Users/denhiroshi/Downloads/datas/AWS/reviews_Digital_Music_5.json"
BATCH_SIZE          = 128
EPOCHS              = 100
LEARNING_RATE       = 0.02
CONV_LENGTH         = 3
CONV_KERNEL_NUM     = 100
FM_K                = 32 #Factorization Machine 交叉向量维度
LATENT_FACTOR_NUM   = 100
GPU_DEVICES         = 0


class DeepCoNN(nn.Module):
    def __init__(self, review_length, word_vec_dim, fm_k, conv_length,
                 conv_kernel_num, latent_factor_num, word_weights):

        # :param review_length: 评论单词数
        # :param word_vec_dim: 词向量维度
        # :param conv_length: 卷积核的长度
        # :param conv_kernel_num: 卷积核数量
        # :param latent_factor_num: 全连接输出的特征维度

        super(DeepCoNN, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(word_weights)
        self.conv_u = nn.Sequential(  # input shape (batch_size, 1, review_length, word_vec_dim)
            nn.Conv2d(
                in_channels=1,
                out_channels=conv_kernel_num,
                kernel_size=(conv_length, word_vec_dim),
            ), 
            # (batch_size, conv_kernel_num, review_length-conv_word_size + 1, 1)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(review_length-conv_length+1, 1)),
            Flatten(),
            nn.Linear(conv_kernel_num, latent_factor_num),
            nn.Dropout(p=0.5),
            nn.ReLU(),
        )
        self.conv_i = nn.Sequential(  # input shape (batch_size, 1, review_length, word_vec_dim)
            nn.Conv2d(
                in_channels=1,
                out_channels=conv_kernel_num,
                kernel_size=(conv_length, word_vec_dim),
            ), 
            # (batch_size, conv_kernel_num, review_length-conv_word_size + 1, 1)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(review_length-conv_length+1, 1)),
            Flatten(),
            nn.Linear(conv_kernel_num, latent_factor_num),
            nn.Dropout(p=0.5),
            nn.ReLU(),
        )
        self.out = FactorizationMachine(latent_factor_num * 2, fm_k)

    def forward(self, user_review, item_review):
        u_vec = self.embedding(user_review).unsqueeze(1)
        i_vec = self.embedding(item_review).unsqueeze(1)

        user_latent = self.conv_u(u_vec)
        item_latent = self.conv_i(i_vec)
        
        concat_latent = torch.cat((user_latent, item_latent), dim=1)
        prediction = self.out(concat_latent)
        return prediction


class Co_Dataset(Dataset):
    def __init__(self, urids, irids, ratings):
        self.urids      = urids
        self.irids      = irids
        self.ratings    =ratings
 
    def __getitem__(self, index):
        return self.urids[index], self.irids[index], self.ratings[index]
 
    def __len__(self):
        return len(self.ratings)

class Flatten(nn.Module):

    def forward(self, x):
        return x.squeeze()


def gen_texts(texts, word_dict, max_len):
    for t_id, text in texts.items():
        if len(text) < max_len:
            num_padding = max_len - len(text)
            text = text + [ "<PAD/>"] * num_padding
        word_indices = [word_dict[w] if w in word_dict else word_dict["<UNK>"] for w in text]
        texts[t_id] = word_indices
    return texts


def main(path):
    SAVE_DIR    = os.path.sep.join(path.split(os.path.sep)[:-1])
    print("SAVE_DIR: " + SAVE_DIR)
    para        = pickle.load(open(path.replace('.json', '.para'), 'rb'))
    word_model  = Word2Vec.load(path.replace('.json', '.model'))
    word_model.wv.add("<UNK/>", np.zeros(word_model.vector_size))
    word_model.wv.add("<PAD/>", np.zeros(word_model.vector_size))
    word_dict   = {w: i for i, w in enumerate(word_model.wv.index2entity)}
    word_weights = torch.FloatTensor(word_model.wv.vectors)
    u_text      = gen_texts(para['u_text'], word_dict, para['user_length'])
    i_text      = gen_texts(para['i_text'], word_dict, para['item_length'])
    review_length = len(u_text[0])
    word_vec_dim = word_weights.shape[1]
    del para
    del word_model
    del word_dict
    u_train = []
    i_train = []
    r_train = []
    with open(path.replace('.json', '_rating_train.csv')) as f:
        for line in f.readlines():
            line = line.strip()
            line=line.split(',')
            u_train.append(u_text[int(line[0])])
            i_train.append(i_text[int(line[1])])
            r_train.append(float(line[2]))
    u_train = torch.LongTensor(u_train)
    i_train = torch.LongTensor(i_train)
    r_train = torch.FloatTensor(r_train)

    u_valid = []
    i_valid = []
    r_valid = []
    with open(path.replace('.json', '_rating_valid.csv')) as f:
        for line in f.readlines():
            line = line.strip()
            line=line.split(',')
            u_valid.append(u_text[int(line[0])])
            i_valid.append(i_text[int(line[1])])
            r_valid.append(float(line[2]))
    u_valid = torch.LongTensor(u_valid)
    i_valid = torch.LongTensor(i_valid)
    r_valid = torch.FloatTensor(r_valid)
    del u_text
    del i_text
    
    model = DeepCoNN(
        review_length=review_length,
        word_vec_dim=word_vec_dim,
        fm_k=FM_K,
        conv_length=CONV_LENGTH,
        conv_kernel_num=CONV_KERNEL_NUM,
        latent_factor_num=LATENT_FACTOR_NUM,
        word_weights=word_weights
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LATENT_FACTOR_NUM
    )
    loss_func = torch.nn.MSELoss()
    print("DeepCoNN epochs {epochs} batch_size {batch_size}".format(epochs=EPOCHS, batch_size=BATCH_SIZE))
    if torch.cuda.is_available():
        print("GPU mode")
        model = model.cuda()
        loss_func = loss_func.cuda()
    else:
        print("CPU mode")
    
    print('Start training.')
    best_valid_loss     = float('inf')
    best_valid_epoch    = 0
    train_data_loader   = torch.utils.data.DataLoader(
        Co_Dataset(u_train, i_train, r_train), 
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=3
    )
    valid_data_loader   = torch.utils.data.DataLoader(
        Co_Dataset(u_valid, i_valid, r_valid), 
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=3
    )

    for epoch in range(EPOCHS):
        train_loss = None
        for u_text, i_text, rating in tqdm(train_data_loader):
            pred = model(u_text, i_text)
            train_loss = loss_func(pred, rating.flatten())
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

        error = []
        for u_text, i_text, rating in tqdm(valid_data_loader):
            with torch.no_grad():
                batch_pred = model(u_text, i_text)
                batch_error = batch_pred - rating
                error.append(batch_error.cpu().numpy())
        error = np.concatenate(error, axis=None)**2
        error = error.mean().item()
        if best_valid_loss > error:
            best_model_state_dict = copy.deepcopy(model.state_dict())
            best_valid_loss = error
            best_valid_epoch = epoch
            torch.save(
                best_model_state_dict,
                os.path.join(SAVE_DIR, 'DeepCoNN.tar')
            )
        print(
            'epoch: {}, train mse_loss: {:.5f}, valid mse_loss: {:.5f}'
            .format(e, train_loss, error)
        )
    
    with open(os.path.join(SAVE_DIR,'training.json'), 'w') as f:
        json.dump(
            {'epoch': best_valid_epoch,'valid_loss': best_valid_loss},
            f
        )


if __name__ == "__main__":
    path = DATA_PATH_MUSIC
    main(path)