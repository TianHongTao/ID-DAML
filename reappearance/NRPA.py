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
BATCH_SIZE          = 12
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

class ReviewEncoder(nn.Module):
    def __init__(self, word_vec_dim, conv_length, conv_kernel_num, word_weights,
                 id_matrix_len, id_embedding_dim, atten_vec_dim):
        # :param word_vec_dim: 词向量维度
        # :param conv_length: 卷积核的长度
        # :param conv_kernel_num: 卷积核数量
        # :param word_weights: 词向量矩阵权重
        # :param id_matrix_len: id总个数
        # :param id_embedding_dim: id向量维度
        # :param atten_vec_dim: attention向量的维度
        super(ReviewEncoder, self).__init__()
        self.embedding_review = nn.Embedding.from_pretrained(word_weights)
        self.embedding_id     = nn.Embedding(id_matrix_len, id_embedding_dim)
        self.conv = nn.Conv1d( # input shape (batch_size, review_length, word_vec_dim)
            in_channels = word_vec_dim,
            out_channels = conv_kernel_num, 
            kernel_size = conv_length,
            padding = (conv_length -1) //2
        )# output shape (batch_size, conv_kernel_num, review_length)
        self.drop = nn.Dropout(p=1.0)
        self.l1 = nn.Linear(id_embedding_dim, atten_vec_dim)
        self.A1 = nn.Parameter(torch.randn(atten_vec_dim, conv_kernel_num), requires_grad=True)

    def forward(self, review, ids):
        # now the batch_size = user_batch * review_size
        review_vec      = self.embedding_review(review) #(batch_size, review_length, word_vec_dim)
        review_vec      = review_vec.permute(0, 2, 1)
        c               = F.relu(self.conv(review_vec)) #(batch_size, conv_kernel_num, review_length)
        c               = self.drop(c)
        id_vec          = self.embedding_id(ids) #(batch_size, id_embedding_dim)
        qw              = F.relu(self.l1(id_vec)) #(batch_size, atten_vec_dim)
        g               = torch.mm(qw, self.A1).unsqueeze(1) #(batch_size, 1, conv_kernel_num)
        g               = torch.bmm(g, c) #(batch_size, 1, review_length)
        alph            = F.softmax(g, dim=2) #(batch_size, 1, review_length)
        d               = torch.bmm(c, alph.permute(0, 2, 1)) #(batch_size, conv_kernel_num, 1)
        return d


class UIEncoder(nn.Module):
    def __init__(self, conv_kernel_num, id_matrix_len, id_embedding_dim, atten_vec_dim):

        # :param conv_kernel_num: 卷积核数量
        # :param id_matrix_len: id总个数
        # :param id_embedding_dim: id向量维度
        # :param atten_vec_dim: attention向量的维度
        super(UIEncoder, self).__init__()
        self.embedding_id = nn.Embedding(id_matrix_len, id_embedding_dim)
        self.review_f = conv_kernel_num
        self.l1 = nn.Linear(id_embedding_dim, atten_vec_dim)
        self.A1 = nn.Parameter(torch.randn(atten_vec_dim, conv_kernel_num), requires_grad=True)

    def forward(self, word_Att, ids):
        # now the batch_size = user_batch
        # word_Att => #(batch_size, conv_kernel_num, review_size)
        id_vec          = self.embedding_id(ids) #(batch_size, id_embedding_dim)
        qr              = F.relu(self.l1(id_vec)) #(batch_size, atten_vec_dim)
        e               = torch.mm(qr, self.A1).unsqueeze(1) #(batch_size, 1, conv_kernel_num)
        e               = torch.bmm(e, word_Att) #(batch_size, 1, review_size)
        beta            = F.softmax(e, dim=2) #(batch_size, 1, review_size)
        q               = torch.bmm(word_Att, beta.permute(0, 2, 1)) #(batch_size, conv_kernel_num, 1)
        return q


class NRPA(nn.Module):
    def __init__(self, review_size, word_vec_dim, fm_k, conv_length,
                 conv_kernel_num, latent_factor_num, word_weights,
                 u_id_matrix_len, i_id_matrix_len, id_embedding_dim, atten_vec_dim):
        # :param review_size: review句子的数量
        # :param word_vec_dim: 词向量维度
        # :param conv_length: 卷积核的长度
        # :param conv_kernel_num: 卷积核数量
        # :param word_weights: 词向量矩阵权重
        # :param u_id_matrix_len: user_id总个数
        # :param i_id_matrix_len: item_id总个数
        # :param id_embedding_dim: id向量维度
        # :param atten_vec_dim: attention向量的维度
        super(NRPA, self).__init__()
        self.review_size = review_size
        self.word_weights = word_weights
        self.user_reveiw_net = ReviewEncoder(
            word_vec_dim=word_vec_dim, 
            conv_length=conv_length, 
            conv_kernel_num=conv_kernel_num, 
            word_weights=self.word_weights,
            id_matrix_len=u_id_matrix_len, 
            id_embedding_dim=id_embedding_dim, 
            atten_vec_dim=atten_vec_dim
        )
        self.item_review_net = ReviewEncoder(
            word_vec_dim=word_vec_dim, 
            conv_length=conv_length, 
            conv_kernel_num=conv_kernel_num, 
            word_weights=self.word_weights,
            id_matrix_len=i_id_matrix_len, 
            id_embedding_dim=id_embedding_dim, 
            atten_vec_dim=atten_vec_dim
        )
        self.user_net = UIEncoder(
            conv_kernel_num=conv_kernel_num, 
            id_matrix_len=u_id_matrix_len, 
            id_embedding_dim=id_embedding_dim, 
            atten_vec_dim=atten_vec_dim
        )
        self.item_net = UIEncoder(
            conv_kernel_num=conv_kernel_num, 
            id_matrix_len=i_id_matrix_len, 
            id_embedding_dim=id_embedding_dim, 
            atten_vec_dim=atten_vec_dim
        )
        self.fm = FactorizationMachine(conv_kernel_num * 2, fm_k)

    def forward(self, u_text, i_text, u_ids, i_ids):
        # u_text (u_batch, review_size, review_length)
        # i_text (i_batch, review_size, review_length)
        batch_size      = len(u_ids)
        new_batch       = batch_size * self.review_size
        u_text          = u_text.reshape(new_batch, -1)
        mul_u_ids       = u_ids.unsqueeze(1)
        mul_u_ids       = torch.cat((mul_u_ids,) * self.review_size,dim=1).reshape(-1)
        d_matrix_user   = self.user_reveiw_net(u_text, mul_u_ids)
        d_matrix_user   = d_matrix_user.reshape(batch_size, self.review_size, -1).permute(0,2,1)

        i_text          = i_text.reshape(new_batch, -1)
        mul_i_ids       = i_ids.unsqueeze(1)
        mul_i_ids       = torch.cat((mul_i_ids,) * self.review_size,dim=1).reshape(-1)
        d_matrix_item   = self.item_review_net(i_text, mul_i_ids)
        d_matrix_item   = d_matrix_item.reshape(batch_size, self.review_size, -1).permute(0,2,1)

        pu = self.user_net(d_matrix_user, u_ids).squeeze(2)
        qi = self.item_net(d_matrix_item, i_ids).squeeze(2)

        x = torch.cat((pu, qi), dim=1)
        rate = self.fm(x)
        return rate


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
    
    model = NRPA(
        review_size=review_size, 
        word_vec_dim=word_vec_dim, 
        fm_k=FM_K, 
        conv_length=CONV_LENGTH,
        conv_kernel_num=CONV_KERNEL_NUM, 
        latent_factor_num=LATENT_FACTOR_NUM, 
        word_weights     = word_weights,
        u_id_matrix_len  = user_num, 
        i_id_matrix_len  = item_num, 
        id_embedding_dim = ID_EMBEDDING_DIM, 
        atten_vec_dim    = ATTEN_VEC_DIM
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
        if best_valid_loss > error:
            best_model_state_dict = copy.deepcopy(model.state_dict())
            best_valid_loss = error
            best_valid_epoch = epoch
            torch.save(
                best_model_state_dict,
                os.path.join(SAVE_DIR, 'NRPA.tar')
            )
        print(
            'epoch: {}, train mse_loss: {:.5f}, valid mse_loss: {:.5f}'
            .format(epoch, train_loss, error)
        )
    
    with open(os.path.join(SAVE_DIR,'training_NRPA.json'), 'w') as f:
        json.dump(
            {'epoch': best_valid_epoch,'valid_loss': best_valid_loss},
            f
        )


if __name__ == "__main__":
    # path = DATA_PATH_MUSIC
    path = DATA_PATH_MUSIC2
    main(path)