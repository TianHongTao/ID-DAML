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
from FM import FactorizationMachine
from tqdm import tqdm
from gensim.models import Word2Vec
from torch.utils.data.dataset import Dataset

DATA_PATH_MUSIC     = "/Users/denhiroshi/Downloads/datas/AWS/reviews_Digital_Music_5.json"
DATA_PATH_MUSIC2    = "/Users/denhiroshi/Downloads/datas/AWS/reviews_Musical_Instruments_5.json"
BATCH_SIZE          = 32
EPOCHS              = 40
LEARNING_RATE       = 0.02
CONV_LENGTH         = 3
CONV_KERNEL_NUM     = 32
FM_K                = 1 #Factorization Machine 交叉向量维度
LATENT_FACTOR_NUM   = 64
GPU_DEVICES         = 0
ID_EMBEDDING_DIM    = 32
ATTEN_VEC_DIM       = 80
REVIEW_SIZE         = 15
ATT_CONV_SIZE       = 5

class LocalAttention(nn.Module):
    # As the paper in DAML the attention feature is the conv feature
    def __init__(self, word_vec_dim, att_conv_size):
        super(LocalAttention, self).__init__()
        self.att_feature_i = nn.Conv1d(
            in_channels=word_vec_dim,
            out_channels=1, 
            kernel_size=att_conv_size,
            stride=1,
            padding=att_conv_size//2
        )
        self.att_feature_u = nn.Conv1d(
            in_channels=word_vec_dim,
            out_channels=1, 
            kernel_size=att_conv_size,
            stride=1,
            padding=att_conv_size//2
        )
        self.bias = 0.1

    def forward(self, u_emm, i_emm):
        u_fea = self.att_feature_u(u_emm.permute(0,2,1))
        i_fea = self.att_feature_u(i_emm.permute(0,2,1))
        # (batch_size, 1, review_length)

        u_fea = torch.sigmoid(u_fea + self.bias)
        i_fea = torch.sigmoid(i_fea + self.bias)

        att_u = u_emm * u_fea.permute(0,2,1)
        att_i = i_emm * i_fea.permute(0,2,1)
        # (batch_size, review_length, word_vec_dim)
        return att_u, att_i


class MutualAttention(nn.Module):
    def __init__(self, filter_size, word_vec_dim, conv_kernel_num):
        super(MutualAttention, self).__init__()
        self.conv_u = nn.Conv1d(
            in_channels=word_vec_dim,
            out_channels=conv_kernel_num, 
            kernel_size=filter_size,
            stride=1,
            padding=filter_size//2
        )
        self.conv_i = nn.Conv1d(
            in_channels=word_vec_dim,
            out_channels=conv_kernel_num, 
            kernel_size=filter_size,
            stride=1,
            padding=filter_size//2
        )
        self.bias = 0.1

    def forward(self, local_att_u, local_att_i):

        # Originl Attentiion func, the mem has been alloced so large
        # conv_fea_u = self.conv_u(local_att_u.permute(0,2,1)).unsqueeze(2)
        # conv_fea_i = self.conv_i(local_att_i.permute(0,2,1)).unsqueeze(3)
        # # (batch_size,  conv_kernel_num, 1, review_length)
        # # (batch_size,  conv_kernel_num, review_length, 1)
        # distance    = self.get_distance(conv_fea_u, conv_fea_i)
        # A           = torch.reciprocal(distance+1)
        # i_att       = F.softmax(torch.sum(A,dim=2), dim=1)
        # u_att       = F.softmax(torch.sum(A,dim=1), dim=1)

        # My Attention Function  Accord to the paper <ATTENTION IS ALL YOUR NEED>, it will save the mem.

        conv_fea_u = self.conv_u(local_att_u.permute(0,2,1))
        conv_fea_i = self.conv_i(local_att_i.permute(0,2,1)).permute(0,2,1)
        # (batch_size,  conv_kernel_num, review_length)
        # (batch_size,  review_length, conv_kernel_num)

        A = torch.bmm(conv_fea_i, conv_fea_u)
        # (batch_size,  i_review_length, u_review_length) 
        # i_review_length == u_review_length in this Module
        i_att      = F.softmax(torch.sum(A, dim=2), dim=1)
        u_att      = F.softmax(torch.sum(A, dim=1), dim=1)
        
        # (batch_size, review_length)
        return u_att, i_att

    def get_distance(self, conv_fea_u, conv_fea_i):
        conv_sub = torch.sub(conv_fea_u, conv_fea_i)
        conv_pow = torch.pow(conv_sub, 2)
        del conv_sub
        conv_sum = torch.sum(conv_pow, dim=1)
        del conv_pow
        return torch.sqrt(conv_sum)


class Flatten(nn.Module):

    def forward(self, x):
        # print(x.shape)
        return x.squeeze()


class DAML(nn.Module):
    def __init__(self, filter_size, latent_factor_num, conv_kernel_num, 
                 word_vec_dim, att_conv_size, u_id_len, i_id_len, 
                 fm_k, word_weights, review_size):
        super(DAML, self).__init__()
        self.review_size = review_size
        self.local_att = LocalAttention(word_vec_dim, att_conv_size)
        self.mutual_att = MutualAttention(filter_size, word_vec_dim, conv_kernel_num)
        self.id_embedding_u = nn.Embedding(u_id_len, latent_factor_num)
        self.id_embedding_i = nn.Embedding(i_id_len, latent_factor_num)
        self.text_embedding = nn.Embedding.from_pretrained(word_weights)
        self.conv_u = nn.Sequential(
            nn.Conv1d( # input shape (batch_size, review_size, word_vec_dim)
                in_channels = word_vec_dim,
                out_channels = conv_kernel_num, 
                kernel_size = filter_size,
                padding = (filter_size -1) //2
            ),# output shape (batch_size, conv_kernel_num, review_size)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, review_size)),
            Flatten(),
            nn.Linear(conv_kernel_num, latent_factor_num),
            nn.ReLU(),
        )
        self.conv_i  = nn.Sequential(
            nn.Conv1d( # input shape (batch_size, review_size, word_vec_dim)
                in_channels = word_vec_dim,
                out_channels = conv_kernel_num, 
                kernel_size = filter_size,
                padding = (filter_size -1) //2
            ),# output shape (batch_size, conv_kernel_num, review_size)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, review_size)),
            Flatten(),
            nn.Linear(conv_kernel_num, latent_factor_num),
            nn.ReLU(),
        )
        self.out = FactorizationMachine(latent_factor_num * 2, fm_k)
        self.drop_u = nn.Dropout(p=1.0)
        self.drop_i = nn.Dropout(p=1.0)

    def forward(self, u_text, i_text, u_ids, i_ids):
        # (batch_size , review_size, review_length)
        batch_size                  = len(u_text)
        new_batch                   = batch_size * self.review_size
        u_text                      = u_text.reshape(new_batch, -1)
        i_text                      = i_text.reshape(new_batch, -1)
        # (batch_size * review_size, review_length)
        u_text                      = self.text_embedding(u_text)
        i_text                      = self.text_embedding(i_text)
        # print(u_text.shape, i_text.shape)
        # (batch_size * review_size, review_length, word_vec_dim)
        local_att_u, local_att_i    = self.local_att(u_text, i_text)
        del u_text
        del i_text
        # print(local_att_u.shape, local_att_i.shape)
        mutual_att_u, mutual_att_i  = self.mutual_att(local_att_u, local_att_i)
        # print(mutual_att_u.shape, mutual_att_i.shape)
        # (batch_size * review_size, word_vec_dim, review_length)
        pools_u, pools_i            = self.pool_mean(local_att_u*mutual_att_u.unsqueeze(2), local_att_i*mutual_att_i.unsqueeze(2))
        del local_att_u
        del local_att_i
        del mutual_att_u
        del mutual_att_i
        # (batch_size * word_vec_dim, review_size)
        # print(pools_u.shape, pools_i.shape)
        pools_u                     = pools_u.reshape(batch_size, -1, self.review_size)
        pools_i                     = pools_i.reshape(batch_size, -1, self.review_size)
        # [batch, word_vec_dim, review_size]
        # print(pools_u.shape, pools_i.shape)
        user_latent                  = self.conv_u(pools_u)
        item_latent                  = self.conv_i(pools_i)
        del pools_u
        del pools_i
        # [batch, latent_factor_num]
        # print(user_latent.shape, item_latent.shape)
        u_ids                       = self.id_embedding_u(u_ids)
        i_ids                       = self.id_embedding_i(i_ids)
        user_latent += u_ids
        item_latent += i_ids

        concat_latent = torch.cat((self.drop_u(user_latent), self.drop_u(item_latent)), dim=1)
        prediction = self.out(concat_latent)
        return prediction
    
    def pool_mean(self, pool_u, pool_i):
        return torch.mean(pool_u, dim=1) , torch.mean(pool_i, dim=1)
        # return torch.max(pool_u, dim=1)[0], torch.max(pool_u,dim=1)[0]


class Co_Dataset(Dataset):
    def __init__(self, urids, irids, ratings):
        self.urids      = urids
        self.irids      = irids
        self.ratings    =ratings
 
    def __getitem__(self, index):
        return self.urids[index], self.irids[index], self.ratings[index]
 
    def __len__(self):
        return len(self.ratings)


def gen_texts(texts, word_dict, max_len, review_size):
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
        texts[t_id] = sen_indices
    return texts


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
    u_text_dict     = gen_texts(para['u_text'], word_dict, para['user_length'], review_size)
    i_text_dict     = gen_texts(para['i_text'], word_dict, para['item_length'], review_size)
    review_length   = para['user_length']
    word_vec_dim    = word_weights.shape[1]
    user_num        = para['user_num']
    item_num        = para['item_num']
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
            u_train.append(int(line[0]))
            i_train.append(int(line[1]))
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
            u_valid.append(int(line[0]))
            i_valid.append(int(line[1]))
            r_valid.append(float(line[2]))
    u_valid = torch.LongTensor(u_valid)
    i_valid = torch.LongTensor(i_valid)
    r_valid = torch.FloatTensor(r_valid)
    

    model = DAML(
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
        for u_ids, i_ids, rating in tqdm(train_data_loader):
            u_text = u_ids.tolist()
            i_text = i_ids.tolist()
            for i, u_id in enumerate(u_text):
                u_text[i] = u_text_dict[u_id]
            for i, i_id in enumerate(i_text):
                i_text[i] = i_text_dict[i_id]
            u_text = torch.LongTensor(u_text)
            i_text = torch.LongTensor(i_text)
            if torch.cuda.is_available():
                u_text=u_text.cuda()
                i_text=i_text.cuda()
                u_ids=u_ids.cuda()
                i_ids=i_ids.cuda()
                rating=rating.cuda()
            pred = model(u_text, i_text, u_ids, i_ids)
            train_loss = loss_func(pred, rating.flatten())
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

        error = []
        for u_ids, i_ids, rating in tqdm(valid_data_loader):
            with torch.no_grad():
                u_text = u_ids.tolist()
                i_text = i_ids.tolist()
                for i, u_id in enumerate(u_text):
                    u_text[i] = u_text_dict[u_id]
                for i, i_id in enumerate(i_text):
                    i_text[i] = i_text_dict[i_id]
                u_text = torch.LongTensor(u_text)
                i_text = torch.LongTensor(i_text)
                if torch.cuda.is_available():
                    u_text=u_text.cuda()
                    i_text=i_text.cuda()
                    u_ids=u_ids.cuda()
                    i_ids=i_ids.cuda()
                    rating=rating.cuda()
                batch_pred = model(u_text, i_text, u_ids, i_ids)
                batch_error = batch_pred - rating
                error.append(batch_error.cpu().numpy())
        error = np.concatenate(error, axis=None)**2
        error = error.mean().item()
        if best_valid_loss > error and epoch > 1:
            best_model_state_dict = copy.deepcopy(model.state_dict())
            best_valid_loss = error
            best_valid_epoch = epoch
            torch.save(
                best_model_state_dict,
                os.path.join(SAVE_DIR, 'DAML.tar')
            )
        print(
            'epoch: {}, train mse_loss: {:.5f}, valid mse_loss: {:.5f}'
            .format(epoch, train_loss, error)
        )
    
    with open(os.path.join(SAVE_DIR,'training_DAML.json'), 'w') as f:
        json.dump(
            {'epoch': best_valid_epoch,'valid_loss': best_valid_loss},
            f
        )


if __name__ == "__main__":
    # path = DATA_PATH_MUSIC
    path = DATA_PATH_MUSIC2
    main(path)
        