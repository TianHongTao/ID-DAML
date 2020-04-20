#-*-coding:utf-8 -*-
from py2neo import Graph, Node, Relationship
from collections import defaultdict
import csv
import json
import os
import re
import pickle
import numpy as np
import pandas as pd
import py2neo


DATA_PATH_SPORT     = "/Users/denhiroshi/Downloads/datas/AWS/reviews_Sports_and_Outdoors_5.json"
DATA_PATH_MUSIC     = "/Users/denhiroshi/Downloads/datas/AWS/reviews_Digital_Music_5.json"
DATA_PATH_OFFICE    = "/Users/denhiroshi/Downloads/datas/AWS/reviews_Office_Products_5.json"
DATA_PATH_MUSIC2    = "/Users/denhiroshi/Downloads/datas/AWS/reviews_Musical_Instruments_5.json"
NEO4J_IP            = "http://localhost:7474/browser/"
NEO4J_USERNAME      = "neo4j"
NEO4J_PASSWORDS     = "neo4j_test"
graph = Graph(NEO4J_IP, user=NEO4J_USERNAME, password=NEO4J_PASSWORDS)

def main(path):
    jsonfile = path.replace('.csv','.json')
    database_name = path.split('/')[-1].replace('.json','')
    if not os.path.exists(jsonfile):
        csv2json(path, jsonfile)
    id_2_Uinfos = {}
    id_2_Iinfos = {}

    with open(jsonfile) as f:
        for line in f:
            line = line.strip()
            info = json.loads(line)
            if len(info['reviewText']) > 800:
                continue
            for key, value in info.items():
                info[key] = str(value)
            # User
            if id_2_Uinfos.get(info['reviewerID']):
                id_2_Uinfos[info['reviewerID']]["reviewsInfos"].append([info['reviewText'], info['overall'], info['asin'], info["summary"]])
            else:
                id_2_Uinfos[info['reviewerID']] = {
                    "reviewsInfos":[[info['reviewText'], info['overall'], info['asin'], info["summary"]]],
                    "reviewerID":info['reviewerID'],
                    "reviewerName":info.get('reviewerName', "None"),
                    "rates": defaultdict(lambda: 0)
                }
            # Item
            if id_2_Iinfos.get(info["asin"]):
                id_2_Iinfos[info['asin']]["reviewerInfos"].append([info["reviewText"], info['overall'], info['reviewerID'], info["summary"]])
            else:
                id_2_Iinfos[info['asin']] = {
                    "itemId":info['asin'],
                    "reviewerInfos":[[info['reviewText'], info['overall'], info['reviewerID'], info["summary"]]],
                    "rates": defaultdict(lambda: 0)
                }
            # 记录打分状况
            id_2_Uinfos[info["reviewerID"]]["rates"][info['overall']] += 1
            id_2_Iinfos[info["asin"]]["rates"][info['overall']] += 1

    tx  = graph.begin()
    ItemNode = {}
    for uid, Uinfo in id_2_Uinfos.items():
        reviewer = Node(database_name+"_Reviewer", reviewerName=Uinfo["reviewerName"], reviewerID=uid, reviewsInfos=str(Uinfo["reviewsInfos"]), rates=str(Uinfo["rates"]))
        tx.create(reviewer)
        for info in Uinfo["reviewsInfos"]:
            # if id_2_Iinfos.get(info[2]):
            itemInfo = id_2_Iinfos[info[2]]
            if ItemNode.get(info[2]) is None:
                item = Node(database_name+"_Item", itemId=info[2], reviewerInfos=str(itemInfo["reviewerInfos"]), rates=str(itemInfo["rates"]))
                ItemNode[info[2]] = item
                tx.create(item)
            else:
                item = ItemNode[info[2]]
            relation = Relationship(reviewer, info[1], item, text=info[0], summary=info[3])
            tx.create(relation)
        # tx.push(reviewer)
        # tx.push(item)
        # tx.push(relation)
    tx.commit()
    print(path + " to neo4j database over")
    pass


if __name__ == "__main__":
    # 清库
    graph.delete_all()
    main(DATA_PATH_MUSIC)
    main(DATA_PATH_OFFICE)
    main(DATA_PATH_MUSIC2)
    main(DATA_PATH_SPORT)