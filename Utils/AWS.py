#-*-coding:utf-8 -*-

import csv
import json
import os
import pickle
import numpy as np
import pandas as pd
DATA_PATH_SPORT     = "/Users/denhiroshi/Downloads/datas/AWS/reviews_Sports_and_Outdoors_5.json"
DATA_PATH_MUSIC     = "/Users/denhiroshi/Downloads/datas/AWS/reviews_Digital_Music_5.json"
DATA_PATH_OFFICE    = "/Users/denhiroshi/Downloads/datas/AWS/reviews_Office_Products_5.json"

def main(path):
    jsonfile = path.replace('.csv','.json')
    if not os.path.exists(jsonfile):
        csv2json(path, jsonfile)
    
    users_id=[]
    items_id=[]
    ratings=[]
    reviews=[]
    np.random.seed(2019)
    with open(jsonfile) as f:
        for line in f:
            line = line.strip()
            info = json.loads(line)
            users_id.append(info['reviewerID'])
            items_id.append(info['asin'])
            ratings.append(info['overall'])
            reviews.append(info['reviewText'])
    data = pd.DataFrame({'user_id':pd.Series(users_id),
                   'item_id':pd.Series(items_id),
                   'ratings':pd.Series(ratings),
                   'reviews':pd.Series(reviews)})[['user_id','item_id','ratings','reviews']]
    # print(data)
    data_test, data_train = save_rating(data, path)
    save_data(data_test, data_train, path)
    print("DATASETS " + path + " OVER")

def save_rating(data, path):
    data = numerize(data)
    print(data)

    data_rating = data[['user_id','item_id','ratings']]
    n_ratings   = data_rating.shape[0]

    test            = np.random.choice(n_ratings, size=int(0.20 * n_ratings), replace=False)
    test_idx        = np.zeros(n_ratings, dtype=bool)
    test_idx[test]  = True
    rating_testing  = data_rating[test_idx]
    rating_train    = data_rating[~test_idx]
    data_test       = data[test_idx]
    data_train      = data[~test_idx]
    n_ratings       = rating_testing.shape[0]
    test            = np.random.choice(n_ratings, size=int(0.50 * n_ratings), replace=False)
    test_idx        = np.zeros(n_ratings, dtype=bool)
    test_idx[test]  = True
    rating_test     = rating_testing[test_idx]
    rating_valid    = rating_testing[~test_idx]
    rating_train.to_csv(path.replace('.json', '_rating_train.csv'), index=False,header=None)
    rating_valid.to_csv(path.replace('.json', '_rating_valid.csv'), index=False,header=None)
    rating_test.to_csv(path.replace('.json', '_rating_test.csv'), index=False,header=None)
    print("rating_test:")
    print(rating_test)
    return data_test, data_train

def save_data(data_test, data_train, path):
    user_reviews={}
    item_reviews={}
    user_rid={}
    item_rid={}
    for i in data_train.values:
        if user_reviews.get(i[0]):
            user_reviews[i[0]].append(i[3])
            user_rid[i[0]].append(i[1])
        else:
            user_rid[i[0]]=[i[1]]
            user_reviews[i[0]]=[i[3]]
        if item_reviews.get(i[1]):
            item_reviews[i[1]].append(i[3])
            item_rid[i[1]].append(i[0])
        else:
            item_reviews[i[1]] = [i[3]]
            item_rid[i[1]]=[i[0]]

    for i in data_test.values:
        if user_reviews.get(i[0]):
            user_reviews[i[0]].append(i[3])
            user_rid[i[0]].append(i[1])
        else:
            user_rid[i[0]]=[i[1]]
            user_reviews[i[0]]=[i[3]]
        if item_reviews.get(i[1]):
            item_reviews[i[1]].append(i[3])
            item_rid[i[1]].append(i[0])
        else:
            item_reviews[i[1]] = [i[3]]
            item_rid[i[1]]=[i[0]]

    pickle.dump(user_reviews, open(path.replace(".json", 'user_review'), 'wb'))
    pickle.dump(item_reviews, open(path.replace(".json", 'item_review'), 'wb'))
    pickle.dump(user_rid, open(path.replace(".json", 'user_rid'), 'wb'))
    pickle.dump(item_rid, open(path.replace(".json", 'item_rid'), 'wb'))
    print("user_count, item_count:")
    print(len(user_rid), len(item_rid))

def get_count(data, id_group_by):
    count = data[[id_group_by, 'ratings']].groupby(id_group_by, as_index=False)
    count = count.size()
    return count

def csv2json(path, out):
    csv_file = open(path,'r')
    json_file = open(out,'w')
    pd_file = pd.read_csv(path)
    title = pd_file.columns
    title = tuple(title)
    reader = csv.DictReader(csv_file,title)
    for row in reader:
        json.dump(row, json_file)
        json_file.write('\n')
    json_file.close()
    csv_file.close()
    print("succees!")

def numerize(data):
    usercount, itemcount = get_count(data, 'user_id'), get_count(data, 'item_id')
    unique_uid  = usercount.index
    unique_sid  = itemcount.index
    item2id     = dict((sid, i) for (i, sid) in enumerate(unique_sid))
    user2id     = dict((uid, i) for (i, uid) in enumerate(unique_uid))

    uid = map(lambda x: user2id[x], data['user_id'])
    sid = map(lambda x: item2id[x], data['item_id'])

    data['user_id'] = list(uid)
    data['item_id'] = list(sid)
    return data

if __name__ == "__main__":
    main(DATA_PATH_SPORT)
    main(DATA_PATH_MUSIC)
    main(DATA_PATH_OFFICE)